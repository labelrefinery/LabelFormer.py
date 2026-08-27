"""Pre-norm transformer encoder with symmetric ALiBi positional bias.

LabelFormer processes the whole trajectory at once with a plain (non-causal)
self-attention stack. Instead of sinusoidal or learned position embeddings it
uses ALiBi: a per-head linear penalty on the temporal distance between frames,
which keeps the model agnostic to trajectory length.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def alibi_slopes(num_heads: int, *, device=None, dtype=None) -> Tensor:
    """Per-head ALiBi slopes ``m_h = 2^(-8 (h + 1) / H)`` for ``h = 0..H-1``."""
    h = torch.arange(1, num_heads + 1, device=device, dtype=dtype or torch.float32)
    return torch.pow(2.0, -8.0 * h / num_heads)


def build_alibi_bias(
    seq_len: int, num_heads: int, *, device=None, dtype=None
) -> Tensor:
    """Symmetric ALiBi attention bias ``(H, T, T)``: ``bias[h, i, j] = -m_h |i - j|``."""
    slopes = alibi_slopes(num_heads, device=device, dtype=dtype)
    idx = torch.arange(seq_len, device=device, dtype=slopes.dtype)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return -slopes.view(num_heads, 1, 1) * dist.unsqueeze(0)


class AlibiSelfAttention(nn.Module):
    """Multi-head self-attention with an additive float attention mask."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """Attend over ``(B, T, D)`` with a broadcastable float mask ``(B, H, T, T)``."""
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, H, T, head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out_proj(out.transpose(1, 2).reshape(b, t, d))


class EncoderLayer(nn.Module):
    """Pre-norm block: ``x + MHA(LN(x))`` then ``x + FFN(LN(x))``."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = AlibiSelfAttention(d_model, nhead, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """Run one encoder block over ``(B, T, D)``."""
        x = x + self.dropout1(self.attn(self.norm1(x), attn_mask))
        return x + self.dropout2(self.ffn(self.norm2(x)))


class AlibiTransformerEncoder(nn.Module):
    """Stack of pre-norm encoder blocks with symmetric ALiBi bias."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def attention_mask(self, seq_len: int, frame_mask: Tensor | None, x: Tensor) -> Tensor:
        """ALiBi bias plus key padding, as a float mask ``(B, H, T, T)``.

        Padded keys get a large finite negative value (rather than ``-inf``) so
        that a fully-masked row degrades to uniform attention instead of NaN.
        """
        bias = build_alibi_bias(
            seq_len, self.nhead, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        if frame_mask is None:
            return bias
        neg = torch.finfo(x.dtype).min / 2.0
        key_bias = torch.where(
            frame_mask.view(frame_mask.shape[0], 1, 1, seq_len),
            torch.zeros((), device=x.device, dtype=x.dtype),
            torch.full((), neg, device=x.device, dtype=x.dtype),
        )
        return bias + key_bias

    def forward(self, x: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        """Encode ``(B, T, D)``; padded frames are ignored and returned as zeros."""
        attn_mask = self.attention_mask(x.shape[1], frame_mask, x)
        if frame_mask is not None:
            x = x * frame_mask.unsqueeze(-1).to(x.dtype)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.norm(x)
        if frame_mask is not None:
            x = x * frame_mask.unsqueeze(-1).to(x.dtype)
        return x
