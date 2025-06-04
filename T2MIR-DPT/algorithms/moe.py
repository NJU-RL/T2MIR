import math
import torch
import warnings

from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from typing import Optional

from algorithms.tools import hard_update, soft_update, freeze


@dataclass
class CalculatorOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None


@dataclass
class MoEMlpOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    balance_loss: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None
    gate_load: Optional[torch.LongTensor] = None
    gate_importance: Optional[torch.FloatTensor] = None
    expert2tokens: Optional[dict] = None
    gate_logits: Optional[torch.FloatTensor] = None
    gate_logits_target: Optional[torch.FloatTensor] = None


class TopKBalancedNoisyGate(nn.Module):
    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        use_balance=True,
        balance_loss_weight=1e-2,
        add_noise=True,
        noise_epsilon=1e-2,
        add_softmax: bool = False,
        use_top_k_indices: bool = False,
    ):
        super(TopKBalancedNoisyGate, self).__init__()
        assert num_selects <= num_experts
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.gate_network = self.get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight

        # add_noise
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.warned = False
        if self.add_noise:
            self.weight_noise = nn.Linear(input_size, num_experts, bias=False)
            self.weight_noise.weight.data = torch.zeros(
                (num_experts, input_size),
                requires_grad=True,
                device=self.weight_noise.weight.data.device,
                dtype=self.weight_noise.weight.data.dtype,
            )
            self.mean = 0.0
            self.std = 1.0
            self.normal = Normal(self.mean, self.std)
            self.softplus = nn.Softplus()

        # gate logits used for contrastive learning
        self.add_softmax = add_softmax
        self.use_top_k_indices = use_top_k_indices

        self.reset_parameters()

    def get_gate_network(self, gate_network_type, input_size, num_experts):
        gate_network_type = gate_network_type.lower()

        if gate_network_type == "linear":
            gate_network = nn.Linear(input_size, num_experts, bias=False)
            nn.init.zeros_(gate_network.weight)
        elif gate_network_type == "mlp":
            gate_network = torch.nn.Sequential(
                torch.nn.Linear(input_size, num_experts, bias=False),
                torch.nn.Tanh(),
                torch.nn.Linear(num_experts, num_experts, bias=False),
            )
        else:
            raise ValueError(f"Unexpected gate network type: {gate_network_type}.")

        return gate_network

    def reset_gate_network(self):
        self.gate_network = self.get_gate_network(
            self.gate_network_type, self.input_size, self.num_experts
        )

    def reset_parameters(self):
        if self.add_noise:
            nn.init.zeros_(self.weight_noise.weight)

    def cv_squared(self, x, eps=1e-10):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.s
        """
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x):
        logits_gate = self.gate_network(x)
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)
            noise_control = self.softplus(noise_mm) + self.noise_epsilon
            logits_noise = torch.randn_like(logits_gate) * noise_control
            logits = logits_gate + logits_noise
        else:
            logits = logits_gate

        top_logits, top_indices = logits.topk(
            min(self.num_selects + 1, self.num_experts), dim=1
        )  # select the top (k+1) experts
        top_k_logits = top_logits[:, : self.num_selects]
        top_k_indices = top_indices[:, : self.num_selects]
        top_k_scores = (
            self.softmax(top_k_logits.to(torch.float32))
            if self.use_softmax
            else top_k_logits
        )
        top_k_scores = top_k_scores.to(logits.dtype)

        zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
        scores_filtered = zeros.scatter(
            dim=1, index=top_k_indices, src=top_k_scores
        )
        importance = scores_filtered.sum(0)

        if self.training:
            if self.add_noise and self.num_selects != self.num_experts:
                batch_size = top_logits.size(0)
                m = top_logits.size(1)
                top_values_flat = top_logits.flatten()
                threshold_positions_if_in = (
                    torch.arange(batch_size, device=x.device) * m + self.num_selects
                )
                threshold_if_in = torch.unsqueeze(
                    torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
                )
                is_in = torch.gt(logits_noise, threshold_if_in)
                threshold_positions_if_out = threshold_positions_if_in - 1
                threshold_if_out = torch.unsqueeze(
                    torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
                )
                # is each value currently in the top k.
                prob_if_in = self.normal.cdf(
                    (logits_gate - threshold_if_in) / noise_control
                )
                prob_if_out = self.normal.cdf(
                    (logits_gate - threshold_if_out) / noise_control
                )
                prob = torch.where(is_in, prob_if_in, prob_if_out)
                load = prob.sum(0)
            else:
                load = (scores_filtered > 0).sum(0)
                if not self.add_noise and not self.warned:
                    warnings.warn(
                        'Gradient-trackable implementation for load calculation is only available when "add_noise=True". '
                        'Training without noise will block the gradient from "load" path and lead to inconsistency in optimization objectives.'
                    )
                    self.warned = True
        else:
            load = (scores_filtered > 0).sum(0)

        if self.use_balance:
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = torch.tensor(-100.0, device=x.device)

        if self.use_top_k_indices:
            return_gate_logits = scores_filtered
        elif self.add_softmax:
            return_gate_logits = F.softmax(logits_gate, dim=-1)
        else:
            return_gate_logits = logits_gate

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
            "gate_logits": return_gate_logits,
        }


class LinearGLUExperts(nn.Module):
    __constants__ = [
        "bias",
        "in_features",
        "hidden_features",
        "out_features",
        "hidden_act",
        "num_experts",
        "size_experts",
    ]

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_act,
        num_experts,
        dropout: float = 0.4,
        size_experts=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        # hidden_act = 'swish'
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearGLUExperts, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_act = hidden_act
        self.num_experts = num_experts

        if size_experts is None:
            assert hidden_features % num_experts == 0
            size_per_expert = hidden_features // num_experts
            size_experts = [size_per_expert for _ in range(num_experts)]
        else:
            assert (
                len(size_experts) == num_experts
                and sum(size_experts) == hidden_features
            )
        self.size_experts = size_experts

        self.act_fn = ACT2FN[hidden_act]

        self.weight_up = nn.ParameterList()
        self.weight_down = nn.ParameterList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_experts):
            this_expert_weight_up = nn.Parameter(
                torch.empty((size_experts[i], in_features), **factory_kwargs)
            )
            this_expert_weight_down = nn.Parameter(
                torch.empty((out_features, size_experts[i]), **factory_kwargs)
            )
            self.weight_up.append(this_expert_weight_up)
            self.weight_down.append(this_expert_weight_down)

        if bias:
            self.bias_up = nn.ParameterList()
            self.bias_down = nn.ParameterList()

            for i in range(num_experts):
                this_expert_bias_up = nn.Parameter(
                    torch.empty((size_experts[i],), **factory_kwargs)
                )
                this_expert_bias_down = nn.Parameter(
                    torch.empty((out_features,), **factory_kwargs)
                )
                self.bias_up.append(this_expert_bias_up)
                self.bias_down.append(this_expert_bias_down)
        else:
            self.register_parameter("bias_up", None)
            self.register_parameter("bias_down", None)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight_up[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_down[i], a=math.sqrt(5))
            if self.bias_up is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_up[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_up[i], -bound, bound)
            if self.bias_down is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_down[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_down[i], -bound, bound)

    def forward(self, input, i):
        up = F.linear(
            input,
            self.weight_up[i],
            self.bias_up[i] if self.bias_up is not None else None,
        )
        down = F.linear(
            self.dropout(self.act_fn(up)),
            self.weight_down[i],
            self.bias_down[i] if self.bias_down is not None else None,
        )
        return down

    def extra_repr(self):
        return (
            "in_features={}, hidden_features={}, out_features={}, hidden_act={},"
            " num_experts={}, size_experts={}, bias={}".format(
                self.in_features,
                self.hidden_features,
                self.out_features,
                self.hidden_act,
                self.num_experts,
                self.size_experts,
                self.bias_up is not None,
            )
        )


class UniversalCalculator(nn.Module):
    def __init__(
        self,
        experts: LinearGLUExperts,
        multiply_gate_scores=True,
        score_scale_factor=1.0,
    ):
        super(UniversalCalculator, self).__init__()
        self.experts = experts
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = experts.num_experts
        self.mlp_norm = None

    def reset_experts(self):
        self.experts.reset_parameters()

    def forward(
        self, x, topK_indices, topK_scores, expert_batch_size=None, **kwargs
    ) -> CalculatorOutput:
        batch_size = topK_indices.size(0)
        num_selects = topK_indices.size(1)
        topK_indices = topK_indices.flatten()
        topK_scores = topK_scores.flatten()
        batch_indices = torch.arange(
            batch_size, device=topK_scores.device
        ).repeat_interleave(num_selects)

        _, index_sorted_topK_indices = topK_indices.sort(0)

        sorted_topK_scores = topK_scores.index_select(0, index_sorted_topK_indices)
        sorted_batch_indices = batch_indices.index_select(0, index_sorted_topK_indices)

        if expert_batch_size is None:
            expert_batch_size = topK_indices.bincount(
                minlength=self.num_experts
            ).tolist()

        sorted_x = x.index_select(0, sorted_batch_indices)
        split_x = torch.split(sorted_x, expert_batch_size, dim=0)

        expert_outputs = [
            self.experts(split_x[i], i)
            for i in range(self.num_experts)
            if split_x[i].shape[0] > 0
        ]

        cat_expert_outputs = torch.cat(expert_outputs, 0)
        output_dim = cat_expert_outputs.size(1)
        if self.multiply_gate_scores:
            if self.mlp_norm is None:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs,
                    sorted_topK_scores.reshape(-1, 1) * self.score_scale_factor,
                )
            else:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs, sorted_topK_scores.reshape(-1, 1)
                )
                cat_expert_outputs = self.mlp_norm(cat_expert_outputs)

        zeros = torch.zeros(
            (batch_size, output_dim),
            device=cat_expert_outputs.device,
            dtype=cat_expert_outputs.dtype,
        )
        y = zeros.index_add(0, sorted_batch_indices, cat_expert_outputs)

        return CalculatorOutput(hidden_states=y, num_dropped_tokens=torch.tensor(-1.0))


class LinearGLUMoELayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        hidden_act,
        num_experts,
        num_selects,
        dropout: float = 0.4,
        size_experts=None,
        bias=True,
        **kwargs,
    ):
        super(LinearGLUMoELayer, self).__init__()
        assert num_selects <= num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.size_experts = size_experts
        self.bias = bias

        # expert networks
        experts = LinearGLUExperts(
            input_size,
            hidden_size,
            output_size,
            hidden_act,
            num_experts,
            dropout=dropout,
            size_experts=size_experts,
            bias=bias,
        )

        # create gate
        self.gate = TopKBalancedNoisyGate(
            self.input_size,
            self.num_experts,
            self.num_selects,
            gate_network=kwargs.get("gate_network", "mlp"),
            use_softmax=kwargs.get("gate_use_softmax", True),
            use_balance=kwargs.get("gate_use_balance", True),
            balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
            add_noise=kwargs.get("gate_add_noise", True),
            noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
        )

        # create calculator
        self.calculator = UniversalCalculator(
            experts,
            multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
            score_scale_factor=kwargs.get("score_scale_factor", 1.0),
        )

    def forward(self, x) -> MoEMlpOutput:
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_size)
        gate_outputs: dict = self.gate(x)
        calc_outs: CalculatorOutput = self.calculator(x, **gate_outputs)
        y = calc_outs.hidden_states
        y = y.reshape(original_shape + (self.output_size,))

        return MoEMlpOutput(
            hidden_states=y,
            balance_loss=gate_outputs.get("balance_loss"),
            num_dropped_tokens=calc_outs.num_dropped_tokens,
            gate_load=gate_outputs.get("load", torch.tensor(-1)),
            gate_importance=gate_outputs.get("importance", torch.tensor(-1)),
            gate_logits=gate_outputs.get("gate_logits", torch.tensor(-1)),
        )

    def set_num_selects(self, num_selects):
        if num_selects > self.gate.num_experts:
            raise ValueError(
                'The value of "num_selects" must satisfy "num_selects <= num_experts"!'
            )
        else:
            self.num_selects = num_selects
            self.gate.num_selects = num_selects

    def set_gate_use_softmax(self, use_softmax):
        self.gate.use_softmax = use_softmax

    def set_gate_use_balance(self, use_balance):
        self.gate.use_balance = use_balance

    def set_gate_balance_loss_weight(self, balance_loss_weight):
        self.gate.balance_loss_weight = balance_loss_weight

    def set_gate_add_noise(self, add_noise):
        self.gate.add_noise = add_noise

    def set_gate_noise_epsilon(self, noise_epsilon):
        self.gate.noise_epsilon = noise_epsilon

    def set_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.calculator.multiply_gate_scores = multiply_gate_scores

    def set_calculator_score_scale_factor(self, score_scale_factor):
        self.calculator.score_scale_factor = score_scale_factor

    def reset_gate_network(self):
        self.gate.reset_gate_network()

    def reset_experts(self):
        self.calculator.reset_experts()


class LinearGLUMoELayerContrastive(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        hidden_act,
        num_experts_contrastive,
        num_selects_contrastive,
        dropout: float = 0.4,
        size_experts=None,
        bias=True,
        **kwargs,
    ):
        super(LinearGLUMoELayerContrastive, self).__init__()
        assert num_selects_contrastive <= num_experts_contrastive
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts_contrastive
        self.num_selects = num_selects_contrastive
        self.size_experts = size_experts
        self.hard_router = kwargs.get('task_hard_router', False)
        self.bias = bias
        self.tau = kwargs.get("tau", 0.05)
        self.detach_gate_input = kwargs.get("detach_gate_input", False)
        if self.detach_gate_input:
            assert kwargs.get("moe_layers_contrastive") == []

        # expert networks
        experts = LinearGLUExperts(
            input_size,
            hidden_size,
            output_size,
            hidden_act,
            num_experts_contrastive,
            dropout=dropout,
            size_experts=size_experts,
            bias=bias,
        )

        # create gate
        self.gate = TopKBalancedNoisyGate(
            self.input_size,
            self.num_experts,
            self.num_selects,
            gate_network=kwargs.get("gate_network", "mlp"),
            use_softmax=kwargs.get("gate_use_softmax", True),
            use_balance=kwargs.get("gate_use_balance_contrastive", False),
            balance_loss_weight=kwargs.get("gate_balance_loss_weight_contrastive", 1e-2),
            add_noise=kwargs.get("gate_add_noise_contrastive", False),
            noise_epsilon=kwargs.get("gate_noise_epsilon_contrastive", 1e-2),
            add_softmax=kwargs.get("add_softmax", False),
            use_top_k_indices=kwargs.get("use_top_k_indices", False),
        )
        self.gate_target = TopKBalancedNoisyGate(
            self.input_size,
            self.num_experts,
            self.num_selects,
            gate_network=kwargs.get("gate_network", "mlp"),
            use_softmax=kwargs.get("gate_use_softmax", True),
            use_balance=kwargs.get("gate_use_balance_contrastive", False),
            balance_loss_weight=kwargs.get("gate_balance_loss_weight_contrastive", 1e-2),
            add_noise=kwargs.get("gate_add_noise_contrastive", False),
            noise_epsilon=kwargs.get("gate_noise_epsilon_contrastive", 1e-2),
            add_softmax=kwargs.get("add_softmax", False),
            use_top_k_indices=kwargs.get("use_top_k_indices", False),
        )
        
        self.init_target_network()

        # create calculator
        self.calculator = UniversalCalculator(
            experts,
            multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
            score_scale_factor=kwargs.get("score_scale_factor", 1.0),
        )

    def forward(self, x) -> MoEMlpOutput:
        original_shape = x.shape[:-1]
        if self.hard_router:
            gate_outputs: dict = self.gate(x.mean(dim=1))
            gate_outputs_target: dict = self.gate_target(x.mean(dim=1).detach())
            gate_outputs['topK_indices'] = gate_outputs['topK_indices'].unsqueeze(1).expand(-1, x.shape[1], -1).reshape(-1, self.num_selects)
            gate_outputs['topK_scores'] = gate_outputs['topK_scores'].unsqueeze(1).expand(-1, x.shape[1], -1).reshape(-1, self.num_selects)
            x = x.reshape(-1, self.input_size)
        else:
            x = x.reshape(-1, self.input_size)
            gate_outputs: dict = self.gate(x)
            gate_outputs_target: dict = self.gate_target(x.detach())
        calc_outs: CalculatorOutput = self.calculator(x, **gate_outputs)
        y = calc_outs.hidden_states
        y = y.reshape(original_shape + (self.output_size,))

        return MoEMlpOutput(
            hidden_states=y,
            balance_loss=gate_outputs.get("balance_loss"),
            num_dropped_tokens=calc_outs.num_dropped_tokens,
            gate_load=gate_outputs.get("load", torch.tensor(-1)),
            gate_importance=gate_outputs.get("importance", torch.tensor(-1)),
            gate_logits=gate_outputs.get("gate_logits", torch.tensor(-1)),
            gate_logits_target=gate_outputs_target.get("gate_logits", torch.tensor(-1)),
        )

    def update_target_network(self):
        soft_update(self.gate_target.gate_network, self.gate.gate_network, tau=self.tau)
        if self.gate.add_noise:
            soft_update(self.gate_target.weight_noise, self.gate.weight_noise, tau=self.tau)

    def init_target_network(self):
        self.gate.reset_parameters()
        freeze(self.gate_target.gate_network)
        hard_update(self.gate_target.gate_network, self.gate.gate_network)
        if self.gate.add_noise:
            freeze(self.gate_target.weight_noise)
            hard_update(self.gate_target.weight_noise, self.gate.weight_noise)

    def set_num_selects(self, num_selects):
        if num_selects > self.gate.num_experts:
            raise ValueError(
                'The value of "num_selects" must satisfy "num_selects <= num_experts"!'
            )
        else:
            self.num_selects = num_selects
            self.gate.num_selects = num_selects

    def set_gate_use_softmax(self, use_softmax):
        self.gate.use_softmax = use_softmax

    def set_gate_use_balance(self, use_balance):
        self.gate.use_balance = use_balance

    def set_gate_balance_loss_weight(self, balance_loss_weight):
        self.gate.balance_loss_weight = balance_loss_weight

    def set_gate_add_noise(self, add_noise):
        self.gate.add_noise = add_noise

    def set_gate_noise_epsilon(self, noise_epsilon):
        self.gate.noise_epsilon = noise_epsilon

    def set_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.calculator.multiply_gate_scores = multiply_gate_scores

    def set_calculator_score_scale_factor(self, score_scale_factor):
        self.calculator.score_scale_factor = score_scale_factor

    def reset_gate_network(self):
        self.gate.reset_gate_network()

    def reset_experts(self):
        self.calculator.reset_experts()