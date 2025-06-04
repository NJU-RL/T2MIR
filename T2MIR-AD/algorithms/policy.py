import torch
from torch import nn, Tensor
from transformers.activations import ACT2FN
from typing import Dict, Any, Optional

from algorithms.tools import weight_init_
from algorithms.transformers_moe import Transformer as MoETransformer


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1, activation: str = 'leaky_relu'):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = ACT2FN[activation]

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(self.dropout(self.activation(self.layer1(x))))


class DecisionTransformerMoE(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        action_tanh: bool = True,
        n_layer: int = 4,
        n_head: int = 8,
        activation_function: str = 'leaky_relu',
        emb_pdrop: float = 0.1,
        ff_pdrop: float = 0.1,
        ff_moe_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        pos_emb = 'learnable',
        attention: str = 'flash',
        sigma_reparam: bool = True,
    ):
        super(DecisionTransformerMoE, self).__init__()
        self.pos_emb = pos_emb
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tstep_dim = config['tstep_dim']
        self.hidden_dim = config['hidden_dim']
        self.train_episodes = config['train_episode_horizon']
        self.max_episode_steps = config['max_episode_steps']
        self.ff_dim = config['ff_dim'] if config['ff_dim'] is not None else self.hidden_dim * 4

        # networks
        self.state_emb = MLP(state_dim, self.tstep_dim * 2, self.tstep_dim, activation=activation_function)
        self.action_emb = MLP(action_dim, self.tstep_dim * 2, self.tstep_dim, activation=activation_function)
        self.reward_emb = MLP(1, self.tstep_dim * 2, self.tstep_dim, activation=activation_function)
        self.transformer = MoETransformer(self.tstep_dim, self.max_episode_steps, config['moe_config'], self.hidden_dim, self.ff_dim, n_heads=n_head, layers=n_layer, activation=activation_function, causal=True, dropout_emb=emb_pdrop, dropout_ff=ff_pdrop, dropout_ff_moe=ff_moe_pdrop, dropout_attn=attn_pdrop, dropout_qkv=attn_pdrop, pos_emb=pos_emb, attention=attention, sigma_reparam=sigma_reparam)
        self.predict_action = nn.Sequential(*([nn.Linear(self.hidden_dim, action_dim)] + ([nn.Tanh()] if action_tanh else [])))

        self.init_weights()

    def init_weights(self):
        self.state_emb.apply(weight_init_)
        self.action_emb.apply(weight_init_)
        self.predict_action.apply(weight_init_)
        self.reward_emb.apply(weight_init_)

    def forward(self, states: Tensor, actions: Tensor, rewards: Tensor, timesteps: Optional[Tensor] = None, attention_masks: Optional[Tensor] = None, eval: bool = False):
        batch_size, seq_len = states.shape[:2]
        states = self.state_emb(states)
        actions = self.action_emb(actions)
        rewards = self.reward_emb(rewards)
        if self.pos_emb == 'learnable':
            if timesteps is None:
                timesteps = torch.arange(self.max_episode_steps, device=states.device).view(1, self.max_episode_steps, 1).repeat(self.train_episodes, 1, 3).reshape(1, -1).repeat(batch_size, 1)
            else:
                timesteps = timesteps.unsqueeze(2).repeat(1, 1, 3).reshape(batch_size, -1)
        else:  # if fixed position embedding, ignore input timesteps
            timesteps = torch.arange(seq_len * 3, device=states.device).reshape(1, -1).repeat(batch_size, 1)
        input_seq = torch.stack([states, actions, rewards], dim=2).reshape(batch_size, seq_len * 3, -1)
        output_seq, balance_loss, contrastive_loss, _ = self.transformer(input_seq, timesteps, None)
        output_seq = output_seq.reshape(batch_size, seq_len, 3, self.hidden_dim).permute(0, 2, 1, 3)
        if eval:
            return self.predict_action(output_seq[:, 0, -1:]), balance_loss, contrastive_loss
        else:
            return self.predict_action(output_seq[:, 0]), balance_loss, contrastive_loss

    def get_action(self, states: Tensor, actions: Tensor, rewards: Tensor, timesteps: Tensor) -> Tensor:
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
        action_preds, balance_loss, contrastive_loss = self.forward(states, actions, rewards, timesteps, eval=True)

        return action_preds[:, -1]

    def reset_experts(self):
        for layer in self.transformer.layers:
            layer.reset_experts()

    def update_target_network(self):
        self.transformer.update_target_network()