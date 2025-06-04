import flash_attn
import gin
import math
import torch
import torch.nn.functional as F

from einops import rearrange, einsum
from torch import nn
from transformers.activations import ACT2FN
from typing import Dict, Any, List

from algorithms.moe import LinearGLUMoELayer, LinearGLUMoELayerContrastive, MoEMlpOutput
from algorithms.tools import weight_init_


def contrastive_loss(logits_q: torch.Tensor, logits_k: torch.Tensor, batch_size: int, W: torch.Tensor, training: bool = True):
    if not training:
        return torch.FloatTensor([0.0])

    if logits_q.size(0) != batch_size:
        logits_q = logits_q.reshape(batch_size, -1, logits_q.size(-1)).mean(dim=1)  # [batch_size, num_experts]
        logits_k = logits_k.reshape(batch_size, -1, logits_k.size(-1)).mean(dim=1)  # [batch_size, num_experts]
    labels = torch.arange(batch_size // 2, device=logits_q.device)

    w_k1 = torch.mm(W, logits_k[batch_size//2:].transpose(0, 1))
    logits1 = torch.mm(logits_q[:batch_size//2], w_k1)
    logits1 = logits1 - torch.max(logits1, dim=1, keepdim=True)[0]

    w_k2 = torch.mm(W, logits_k[:batch_size//2].transpose(0, 1))
    logits2 = torch.mm(logits_q[batch_size//2:], w_k2)
    logits2 = logits2 - torch.max(logits2, dim=1, keepdim=True)[0]

    return 0.5 * F.cross_entropy(logits1, labels) + 0.5 * F.cross_entropy(logits2, labels)


class Normalization(nn.Module):
    def __init__(self, method: str, d_model: int):
        super().__init__()
        assert method in ["layer", "none"]
        if method == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif method == "none":
            self.norm = lambda x: x
        self.method = method

    def forward(self, x):
        return self.norm(x)


@gin.configurable(allowlist=["window_size"])
class FlashAttention(nn.Module):
    def __init__(
        self,
        causal: bool = True,
        attention_dropout: float = 0.0,
        window_size: tuple[int, int] = (-1, -1),
    ):
        super().__init__()
        self.dropout = attention_dropout
        self.causal = causal
        self.window_size = window_size

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        qkv = qkv.to(torch.bfloat16)
        if key_cache is None or val_cache is None or cache_seqlens is None:
            out = flash_attn.flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.window_size,
            )
        else:
            assert not self.training
            q, k, v = qkv.unbind(2)
            out = flash_attn.flash_attn_with_kvcache(
                q=q,
                k_cache=key_cache,
                v_cache=val_cache,
                cache_seqlens=cache_seqlens,
                k=k,
                v=v,
                causal=self.causal,
                window_size=self.window_size,
            )
        return out


class SigmaReparam(nn.Module):
    def __init__(self, d_in, d_out, bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(d_out), requires_grad=True) if bias else None
        u = torch.randn(d_out)
        self.register_buffer("u", u / u.norm(dim=0))
        v = torch.randn(d_in)
        self.register_buffer("v", v / v.norm(dim=0))
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # same as nn.Linear
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                u = (self.W @ self.v).float()
                self.u.data = u / u.norm(dim=0)
                v = (self.W.T @ self.u).float()
                self.v.data = v / v.norm(dim=0)
        sigma = einsum(self.u, self.W, self.v, "d, d c , c->")
        W_hat = self.gamma / sigma * self.W
        out = F.linear(x, W_hat, self.b)
        return out


class VanillaAttention(nn.Module):
    def __init__(self, causal: bool = True, attention_dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.causal = causal
        self._mask = None

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        assert (
            key_cache is None and val_cache is None
        ), "VanillaAttention does not support `fast_inference` mode"
        queries, keys, values = torch.unbind(qkv, dim=2)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self._mask is None or self._mask.shape != (B, 1, L, L):
            self._mask = torch.triu(
                torch.ones((B, 1, L, L), dtype=torch.bool, device=qkv.device),
                diagonal=1,
            )
        if self.causal:
            scores.masked_fill_(self._mask, -torch.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V


@gin.configurable(allowlist=["head_scaling", "sigma_reparam"])
class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_qkv,
        n_heads,
        dropout_qkv=0.0,
        head_scaling: bool = True,
        sigma_reparam: bool = True,
    ):
        super().__init__()
        self.attention = attention
        FF = SigmaReparam if sigma_reparam else nn.Linear
        self.qkv_projection = FF(d_model, 3 * d_qkv * n_heads, bias=False)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.out_projection = FF(d_qkv * n_heads, d_model)
        self.head_scaler = nn.Parameter(
            torch.ones(1, 1, n_heads, 1), requires_grad=head_scaling
        )
        self.n_heads = n_heads

    def forward(self, sequence, key_cache=None, val_cache=None, cache_seqlens=None):
        qkv = self.dropout_qkv(self.qkv_projection(sequence))
        qkv = rearrange(
            qkv,
            "batch len (three d_qkv heads) -> batch len three heads d_qkv",
            heads=self.n_heads,
            three=3,
        )
        out = self.head_scaler * self.attention(
            qkv=qkv,
            key_cache=key_cache,
            val_cache=val_cache,
            cache_seqlens=cache_seqlens,
        )
        out = rearrange(out, "batch len heads dim -> batch len (heads dim)")
        out = self.out_projection(out)
        return out


@gin.configurable(denylist=["activation", "norm", "dropout_ff"])
class TransformerLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        d_model: int,
        d_ff: int,
        dropout_ff: float = 0.1,
        activation: str = "leaky_relu",
        norm: str = "layer",
        sigma_reparam: bool = True,
        normformer_norms: bool = True,
    ):
        """
        Pre-Norm Self-Attention. The plain transformer layer. 
        """
        super(TransformerLayer, self).__init__()
        self.self_attention = self_attention
        FF = SigmaReparam if sigma_reparam else nn.Linear
        self.ff1 = FF(d_model, d_ff)
        self.ff2 = FF(d_ff, d_model)
        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = (
            Normalization(method=norm, d_model=d_model)
            if normformer_norms
            else lambda x: x
        )
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.norm4 = (
            Normalization(method=norm, d_model=d_ff)
            if normformer_norms
            else lambda x: x
        )
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.activation = ACT2FN[activation]

    def forward(self, self_seq, key_cache=None, val_cache=None, cache_seqlens=None):
        q1 = self.norm1(self_seq)  # pre-norm
        q1 = self.self_attention(
            q1, key_cache=key_cache, val_cache=val_cache, cache_seqlens=cache_seqlens
        )
        q1 = self.norm2(q1)  # normformer extra norm 1
        self_seq = self_seq + q1

        q1 = self.norm3(self_seq)  # regular norm
        # normformer extra norm 2
        q1 = self.norm4(self.activation(self.ff1(q1)))
        q1 = self.dropout_ff(self.ff2(q1))
        self_seq = self_seq + q1
        return self_seq, None, None


@gin.configurable(denylist=["activation", "norm", "dropout_ff_moe"])
class TransformerLayerMoEandContrastiveMoE(nn.Module):
    def __init__(
        self,
        self_attention,
        d_model: int,
        moe_config: Dict[str, Any],
        dropout_ff_moe: float = 0.,
        activation: str = "leaky_relu",
        norm: str = "layer",
        normformer_norms: bool = True,
    ):
        """
        Pre-Norm Self-Attention. The transformer layer with two MoE, balance loss for one and contrastive loss for the other. The outputs of the two MoE are concatenated.
        """
        super(TransformerLayerMoEandContrastiveMoE, self).__init__()
        self.self_attention = self_attention
        self.W = nn.Parameter(torch.randn(moe_config['num_experts_contrastive'], moe_config['num_experts_contrastive']))
        self.ff_moe1 = LinearGLUMoELayer(d_model, moe_config['expert_dim'], d_model // 2, activation, dropout=dropout_ff_moe, **moe_config)
        self.ff_moe2 = LinearGLUMoELayerContrastive(d_model, moe_config['expert_dim_contrastive'], d_model // 2, activation, dropout=dropout_ff_moe, **moe_config)
        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = (
            Normalization(method=norm, d_model=d_model)
            if normformer_norms
            else lambda x: x
        )
        self.norm3 = Normalization(method=norm, d_model=d_model)

        self.init_weights()

    def init_weights(self):
        self.norm1.apply(weight_init_)
        self.norm2.apply(weight_init_)
        self.norm3.apply(weight_init_)

    def forward(self, self_seq, key_cache=None, val_cache=None, cache_seqlens=None):
        '''
        :return: (self_seq, balance_loss, num_dropped_tokens, gate_load, gate_importance, expert2tokens)
        '''
        q1 = self.norm1(self_seq)  # pre-norm
        q1 = self.self_attention(
            q1, key_cache=key_cache, val_cache=val_cache, cache_seqlens=cache_seqlens
        )
        q1 = self.norm2(q1)  # normformer extra norm 1
        self_seq = self_seq + q1
        q1 = self.norm3(self_seq)  # regular norm
        mlp_outs1: MoEMlpOutput = self.ff_moe1(q1)
        mlp_outs2: MoEMlpOutput = self.ff_moe2(q1)
        self_seq = self_seq + torch.cat([mlp_outs1.hidden_states, mlp_outs2.hidden_states], dim=-1)

        return self_seq, mlp_outs1.balance_loss, contrastive_loss(mlp_outs2.gate_logits, mlp_outs2.gate_logits_target.detach(), self_seq.size(0), self.W, self.training)

    def update_target_network(self):
        self.ff_moe2.update_target_network()


class Cache:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
    ):
        self.data = torch.zeros(
            (batch_size, max_seq_len, n_heads, head_dim), dtype=dtype, device=device
        )
        # make silent bugs in k/v cache... much louder
        self.data[:] = torch.nan

    def __len__(self):
        return self.data.shape[1]

    def roll_back(self, idx):
        roll = self.data[idx, 1:].clone()
        self.data[idx, :-1] = roll
        self.data[idx, -1] = torch.nan  # no silent bugs


class TformerHiddenState:
    def __init__(
        self, key_cache: list[Cache], val_cache: list[Cache], timesteps: torch.Tensor
    ):
        assert isinstance(key_cache, list) and len(key_cache) == len(val_cache)
        assert timesteps.dtype == torch.int32
        self.n_layers = len(key_cache)
        self.key_cache = key_cache
        self.val_cache = val_cache
        self.timesteps = timesteps

    def reset(self, idxs):
        self.timesteps[idxs] = 0

    def update(self):
        self.timesteps += 1
        for i, timestep in enumerate(self.timesteps):
            if timestep == len(self.key_cache[0]):
                for k, v in zip(self.key_cache, self.val_cache):
                    k.roll_back(i)
                    v.roll_back(i)
                self.timesteps[i] -= 1

    def __getitem__(self, layer_idx):
        assert layer_idx < self.n_layers
        return (
            self.key_cache[layer_idx].data,
            self.val_cache[layer_idx].data,
            self.timesteps,
        )


class FixedPosEmb(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, pos_idxs: torch.LongTensor):
        B, L = pos_idxs.shape
        emb = torch.zeros(
            (B, L, self.d_model), device=pos_idxs.device, dtype=torch.float32
        )
        coeff = torch.exp(
            (
                torch.arange(0, self.d_model, 2, device=emb.device, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )
        )
        emb[..., 0::2] = torch.sin(pos_idxs.float().unsqueeze(-1) * coeff)
        emb[..., 1::2] = torch.cos(pos_idxs.float().unsqueeze(-1) * coeff)
        return emb


class Transformer(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        max_pos_idx: int,
        moe_config: Dict[str, Any],
        d_model: int = 128,
        d_ff: int | List[int] = 512,
        d_emb_ff: int = None,
        n_heads: int = 4,
        layers: int = 3,
        dropout_emb: float = 0.05,
        dropout_ff: float = 0.05,
        dropout_ff_moe: float = 0.05,
        dropout_attn: float = 0.00,
        dropout_qkv: float = 0.00,
        attention: str = "flash",
        activation: str = "leaky_relu",
        norm: str = "layer",
        causal: bool = True,
        pos_emb: str = "learnable",
        sigma_reparam: bool = True,
    ):
        super().__init__()
        assert attention in ["flash", "vanilla"]
        assert pos_emb in ["learnable", "fixed"]

        # embedding
        if pos_emb == "learnable":
            self.position_embedding = nn.Embedding(
                max_pos_idx + 1, embedding_dim=d_model
            )
        elif pos_emb == "fixed":
            self.position_embedding = FixedPosEmb(d_model)
        d_emb_ff = d_emb_ff or d_model
        self.inp = nn.Linear(inp_dim, d_model)
        self.dropout = nn.Dropout(dropout_emb)

        self.head_dim = d_model // n_heads
        assert self.head_dim in range(8, 129, 8)
        self.n_heads = n_heads
        self.n_layers = layers
        Attn = FlashAttention if attention == "flash" else VanillaAttention

        def make_layer(d_ff: int):
            return TransformerLayer(
                self_attention=AttentionLayer(
                    attention=Attn(causal=causal, attention_dropout=dropout_attn),
                    d_model=d_model,
                    d_qkv=self.head_dim,
                    n_heads=self.n_heads,
                    dropout_qkv=dropout_qkv,
                    sigma_reparam=sigma_reparam,
                ),
                d_model=d_model,
                d_ff=d_ff,
                dropout_ff=dropout_ff,
                activation=activation,
                norm=norm,
            )

        def make_layer_moe_contrastive_cat():
            return TransformerLayerMoEandContrastiveMoE(
                self_attention=AttentionLayer(
                    attention=Attn(causal=causal, attention_dropout=dropout_attn),
                    d_model=d_model,
                    d_qkv=self.head_dim,
                    n_heads=self.n_heads,
                    dropout_qkv=dropout_qkv,
                    sigma_reparam=sigma_reparam,
                ),
                moe_config=moe_config,
                d_model=d_model,
                dropout_ff_moe=dropout_ff_moe,
                activation=activation,
                norm=norm,
            )

        assert moe_config['moe_layers_contrastive_and_balance'] == [] or max(moe_config['moe_layers_contrastive_and_balance']) < layers

        self.layers = nn.ModuleList()
        for i in range(layers):
            if i in moe_config['moe_layers_contrastive_and_balance']:
                self.layers.append(make_layer_moe_contrastive_cat())
            else:
                self.layers.append(make_layer(d_ff if isinstance(d_ff, int) else d_ff[i]))
        self.norm = Normalization(method=norm, d_model=d_model)
        self.d_model = d_model
        self.contrastive_loss_weight = moe_config['contrastive_loss_weight']

        self.init_weights()

    @property
    def emb_dim(self):
        return self.d_model

    def init_weights(self):
        self.position_embedding.apply(weight_init_)
        self.inp.apply(weight_init_)
        self.norm.apply(weight_init_)

    def forward(self, seq, pos_idxs, hidden_state: None | TformerHiddenState):
        if self.training:
            assert hidden_state is None
        h = hidden_state or [[None, None, None] for _ in range(self.n_layers)]

        # emedding
        pos_emb = self.position_embedding(pos_idxs)
        traj_emb = self.inp(seq)
        traj_emb = self.dropout(traj_emb + pos_emb)

        balance_losses = 0.0
        contrastive_losses = 0.0
        for i in range(self.n_layers):
            traj_emb, balance_loss, contrastive_loss = self.layers[i](traj_emb, *h[i])
            if balance_loss is not None:
                balance_losses = balance_losses + balance_loss
            if contrastive_loss is not None:
                contrastive_losses = contrastive_losses + contrastive_loss
        traj_emb = self.norm(traj_emb)

        if hidden_state is not None:
            # controls the sequence length of the k/v cache
            hidden_state.update()

        return traj_emb, balance_losses, contrastive_losses * self.contrastive_loss_weight, hidden_state

    def update_target_network(self):
        for layer in self.layers:
            if hasattr(layer, 'update_target_network'):
                layer.update_target_network()