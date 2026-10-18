# Adapted from david3684/flm (Apache-2.0).
import math
import typing
from dataclasses import dataclass
import einops
import torch
from torch import nn
from torch.nn import functional as F

# Match the source launchers: uncompiled PyTorch SDPA.
HAS_FLASH_ATTN = False
jit_deco = lambda fn: fn


@dataclass
class DitConfig:
    hidden_size: int = 256
    cond_dim: int = 128
    length: int = 81
    n_blocks: int = 7
    n_heads: int = 8
    dropout: float = 0.1
    self_cond: bool = True


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    out = scale * F.dropout(
        x + bias if bias is not None else x, p=prob, training=training
    )
    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _fn(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _fn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@jit_deco
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@jit_deco
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@jit_deco
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1, positions=None):
        # positions: optional per-token position indices. Two forms:
        #   (seq_len,)     1D indices  -> standard RoPE angle t*inv_freq (used by the two-grid RESET).
        #   (seq_len, 2)   2D (row,col) -> factorized RoPE: the first half of the frequency bank rotates
        #                  by ROW, the second half by COL, so a grid cell (r,c) gets a genuine 2D
        #                  position (within-grid adjacency along BOTH axes + cross-grid (r,c)<->(r,c)
        #                  alignment when both halves share (r,c)). head_dim/2 must be even.
        # Default None => 0..seq_len-1 (stock). Non-arange positions skip the seq-len cache (recompute).
        seq_len = x.shape[seq_dim]
        if positions is not None:
            positions = positions.to(device=x.device)
            if positions.ndim == 2 and positions.shape[-1] == 2:
                inv = self.inv_freq.clone()  # (dim/2,)
                half = inv.shape[0] // 2  # split freq bank: rows | cols
                row = positions[:, 0].type_as(inv)
                col = positions[:, 1].type_as(inv)
                fr = torch.einsum("i,j->ij", row, inv[:half])  # (seq, half) row angles
                fc = torch.einsum("i,j->ij", col, inv[half:])  # (seq, half) col angles
                freqs = torch.cat(
                    (fr, fc), dim=-1
                )  # (seq, dim/2): [row-freqs | col-freqs]
            else:
                t = positions.type_as(self.inv_freq)
                freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            cos[:, :, 2, :, :].fill_(1.0)
            sin[:, :, 2, :, :].fill_(0.0)
            return cos, sin
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # Identity transform on v
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_manual(x, cos, sin):
    """Apply rotary embeddings without flash-attn (works on any device)."""
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return x * cos + rotate_half(x) * sin


def apply_rotary_pos_emb(qkv, cos, sin, use_flash: bool = True):
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]

    # Fallback: manual rotary application
    q, k, v = qkv.unbind(dim=2)
    q_rot = _apply_rotary_manual(q, cos, sin)
    k_rot = _apply_rotary_manual(k, cos, sin)
    return torch.stack([q_rot, k_rot, v], dim=2)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DDiTBlock(nn.Module):
    """Bidirectional adaptive-LayerNorm transformer block."""

    def __init__(self, dim, n_heads, adaLN, cond_dim=None, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.adaLN = adaLN
        self.softcap = 50
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout = dropout
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def custom_sdpa(self, q, k, v, softcap=-1.0):
        B, H, S, D = q.shape
        q = q / (D**0.5)
        attn_weights = torch.einsum("bhid,bhjd->bhij", q, k)
        if softcap > 0.0:
            attn_weights = softcap * torch.tanh(attn_weights / softcap)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        return torch.einsum("bhij,bhjd->bhid", attn_probs, v)

    def forward(
        self,
        x,
        rotary_cos_sin=None,
        c=None,
        seqlens=None,
        exclude_last_token=False,
        use_jvp_attn=False,
    ):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)
        if self.adaLN:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
                self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            )
            x = modulate_fused(x, shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.amp.autocast(device_type=qkv.device.type, enabled=False):
            cos, sin = rotary_cos_sin
            # use_flash=False when use_jvp_attn (needs grads through attn)
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype), use_flash=(not use_jvp_attn)
            )

        if use_jvp_attn:
            # Custom attention needed for JVP (torch.func.jvp) compatibility
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = self.custom_sdpa(q, k, v, softcap=self.softcap)
            x = x.transpose(1, 2)
        else:
            # F.scaled_dot_product_attention uses memory-efficient fused kernels
            # on PyTorch 2.x — avoids storing the O(B*H*S²) attention matrix
            q, k, v = qkv.unbind(dim=2)
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        x = einops.rearrange(x, "b s h d -> b s (h d)")

        if self.adaLN:
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, gate_msa, x_skip, self.dropout
            )
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                None,
                gate_mlp,
                x,
                self.dropout,
            )
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, scale, x_skip, self.dropout
            )
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, scale, x, self.dropout
            )
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        if x.ndim == 2:
            return self.embedding[x]
        assert x.ndim == 3
        return torch.einsum("blv,ve->ble", x.float(), self.embedding.float()).to(
            x.dtype
        )


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, adaLN, bias: bool = True):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=bias)
        self.linear.weight.data.zero_()
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()
        self.adaLN = adaLN
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c, tied_weight=None, return_hidden=False):
        x = self.norm_final(x)
        if self.adaLN:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate_fused(x, shift, scale)
        if return_hidden:
            # ELF-style MSE flow: the post-norm E-dim state IS the predicted clean embedding.
            return x
        if tied_weight is not None:
            # Weight-tied unembed for learned-embedding-space flow: logits = norm_final(h) @ Wᵀ.
            # (Bypasses self.linear; W is the (V,E) vocab_embed table, scale-normalized upstream.)
            return torch.matmul(x, tied_weight.transpose(0, 1).to(x.dtype))
        return self.linear(x)


class DIT(nn.Module):
    """Bidirectional DiT with raw-logit self-conditioning.

    Parameter names and registration order match the public FLM checkpoints.
    """

    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.vocab_embed = EmbeddingLayer(config.hidden_size, vocab_size)
        self.self_embed = (
            EmbeddingLayer(config.hidden_size, vocab_size) if config.self_cond else None
        )
        if self.self_embed is not None:
            with torch.no_grad():
                self.self_embed.embedding.zero_()
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = Rotary(config.hidden_size // config.n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    config.hidden_size,
                    config.n_heads,
                    True,
                    config.cond_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.output_layer = DDiTFinalLayer(
            config.hidden_size, vocab_size, config.cond_dim, True
        )

    def forward(self, x, sigma, self_cond=None):
        x = self.vocab_embed(x)
        if self_cond is not None and self.self_embed is not None:
            x = x + self.self_embed(self_cond)
        t_cond = F.silu(self.sigma_map(sigma))
        rotary = self.rotary_emb(x)
        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary, c=t_cond)
            return self.output_layer(x, c=t_cond)
