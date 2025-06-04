import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
from torch import Tensor
from transformers import GPT2Config

from algorithms.gpt2moe_model import GPT2MOEModel


class DPTTransformerMOE(nn.Module):
    """Transformer class."""

    def __init__(self, state_dim, action_dim, config, action_tanh=True):
        super(DPTTransformerMOE, self).__init__()

        self.config = config
        self.n_embd = self.config['hidden_dim']
        self.hidden_dim = self.config['hidden_dim']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ff_pdrop = self.config['ff_pdrop']
        self.attn_pdrop = self.config['attn_pdrop']
        self.emb_pdrop = self.config['emb_pdrop']
        self.prompt_horizon = self.config['prompt_horizon']
        self.dropout_ff_moe = self.config['ff_moe_pdrop']
        self.ff_dim = self.config['ff_dim']
        self.env_discrete = not action_tanh

        args = GPT2Config(
            n_positions=3 * (1 + self.prompt_horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.ff_pdrop,
            embd_pdrop=self.emb_pdrop,
            attn_pdrop=self.attn_pdrop,
            activation_function=config['activation_function'],
            use_cache=False,
        )
        self.transformer = GPT2MOEModel(args, self.config['moe_config'], dropout_ff_moe=self.dropout_ff_moe, ff_dim=self.ff_dim)

        self.embed_timestep = nn.Embedding(self.prompt_horizon + 1, self.hidden_dim)
        if self.env_discrete:
            self.embed_state = nn.Embedding(self.state_dim, self.hidden_dim)
            self.embed_action = nn.Embedding(self.action_dim, self.hidden_dim)
        else:
            self.embed_state = nn.Linear(self.state_dim, self.hidden_dim)
            self.embed_action = nn.Linear(self.action_dim, self.hidden_dim)
        self.embed_return = nn.Linear(1, self.hidden_dim)
        self.pred_actions = nn.Sequential(
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh() if action_tanh else nn.Identity(),
        )

    def forward(self, state_seq: Tensor, action_seq: Tensor, reward_seq: Tensor, query_states: Tensor, timesteps: Tensor = None, attention_masks: Tensor = None, eval = False):
        batch_size, seq_len = state_seq.size(0), state_seq.size(1)

        if timesteps is None:
            timesteps = torch.arange(self.prompt_horizon + 1, device=state_seq.device).reshape(1, -1).repeat(batch_size, 1)
        # else:
        #     timesteps = timesteps.unsqueeze(2).repeat(1, 1, 1).reshape(batch_size, -1)

        if attention_masks is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_masks = torch.ones([batch_size, seq_len], dtype=torch.long, device=state_seq.device)
        timesteps = self.embed_timestep(timesteps)
        query_states = self.embed_state(query_states) + timesteps[:, :1]
        states = self.embed_state(state_seq) + timesteps[:, 1:]
        actions = self.embed_action(action_seq) + timesteps[:, 1:]
        rewards = self.embed_return(reward_seq) + timesteps[:, 1:]
        
        stacked_inputs = torch.stack([states, actions, rewards], dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.hidden_dim)
        stacked_attention_masks = torch.stack([attention_masks, attention_masks, attention_masks], dim=1).permute(0, 2, 1).reshape(batch_size, 3 * seq_len)
        stacked_inputs = torch.cat([query_states, stacked_inputs], dim=1)
        stacked_attention_masks = torch.cat([torch.ones([batch_size, 1], dtype=torch.long, device=state_seq.device), stacked_attention_masks], dim=1)

        transformer_outputs, balance_loss, contrastive_loss = self.transformer(inputs_embeds=stacked_inputs, attention_mask=stacked_attention_masks)
        transformer_outputs = transformer_outputs['last_hidden_state']
        state_ids = [0] + list(range(1, 3 * seq_len + 1, 3))
        x = transformer_outputs[:, state_ids]

        if eval:
            return self.pred_actions(x[:, -1])
        return self.pred_actions(x), balance_loss, contrastive_loss

    def update_target_network(self):
        self.transformer.update_target_network()