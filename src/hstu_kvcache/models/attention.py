from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PointwiseAttentionConfig:
    hidden_size: int
    num_heads: int
    head_dim: int | None = None
    bias: bool = False
    # Scale applied to Q.K^T before the activation.
    qk_scale: float = 1.0
    # Dropout on the (post-activation) attention matrix. 0 = off.
    attn_dropout: float = 0.0
    # Activation type: "elu_plus1" (HSTU original), "relu" (no +1 baseline, forces peaked attention)
    activation: str = "elu_plus1"


class PointwiseAttention(nn.Module):
    """HSTU pointwise aggregated attention (PMA).

    Defining feature (vs standard transformer attention):
      * No softmax normalisation. The attention activation is applied
        pointwise: ``a_ij = elu(q_i . k_j) + 1`` which is non-negative.
      * Causal masking is therefore *multiplicative* (zero-out future),
        not additive -inf (which would corrupt the unnormalised sum).

    This module owns the Q/K/V linear projections and returns, alongside the
    block output, the per-layer K and V tensors. Those K/V tensors are exactly
    the cached representation ``F(theta, x_u)`` whose drift we study, so they
    are a first-class output rather than an internal detail.
    """

    def __init__(self, cfg: PointwiseAttentionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim or (cfg.hidden_size // cfg.num_heads)
        inner = self.num_heads * self.head_dim
        self.inner = inner
        self.q_proj = nn.Linear(cfg.hidden_size, inner, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.hidden_size, inner, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.hidden_size, inner, bias=cfg.bias)
        self.out_proj = nn.Linear(inner, cfg.hidden_size, bias=cfg.bias)
        self.scale = cfg.qk_scale * (self.head_dim ** -0.5) if cfg.qk_scale != 1.0 else 1.0
        self.attn_dropout = nn.Dropout(cfg.attn_dropout) if cfg.attn_dropout > 0 else nn.Identity()
        self.activation = cfg.activation

    def _activate(self, attn: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(attn)
        return F.elu(attn) + 1.0

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_kv: bool = False,
    ):
        """Args:
            x: [B, L, H] hidden states (already normalised by the block caller).
            attn_mask: additive float mask [B,1,L,L] or [L,L] with 0 for keep and
                -inf for mask. Converted to a multiplicative 0/1 mask internally
                because elu+1 is non-negative. None => full causal.
            return_kv: if True, also return (k, v) [B, L, num_heads*head_dim] which
                are the cacheable per-user representation F(theta, x_u).

        Returns: out [B, L, H], and optionally (k, v).
        """
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # q,k,v: [B, num_heads, L, head_dim]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, L, L]
        attn = self._activate(attn)

        keep = self._build_keep_mask(L, attn_mask, x.device, x.dtype)
        attn = attn * keep  # multiplicative causal mask (0 = drop future)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # [B, h, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, self.inner)
        out = self.out_proj(out)

        if return_kv:
            k_ret = k.transpose(1, 2).reshape(B, L, self.inner)
            v_ret = v.transpose(1, 2).reshape(B, L, self.inner)
            return out, (k_ret, v_ret)
        return out

    def forward_with_cache(
        self,
        x_new: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
    ):
        """Incremental attention with a prefix KV cache (possibly from a different theta).

        Args:
            x_new: [B, m, H] hidden states for NEW positions only (already normed).
            cached_k/v: [B, n, inner] prefix K,V from a previous forward (may be
                produced by theta_old). These are NOT recomputed - that is the
                entire point of KV cache reuse.

        New positions attend to ALL prefix positions + causal within new positions.
        Returns: out [B, m, H], (k_all, v_all) [B, n+m, inner].
        """
        B, m, _ = x_new.shape
        n = cached_k.shape[1]
        q = self.q_proj(x_new).view(B, m, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x_new).view(B, m, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x_new).view(B, m, self.num_heads, self.head_dim).transpose(1, 2)
        k_cached = cached_k.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_cached = cached_v.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        k_all = torch.cat([k_cached, k_new], dim=2)  # [B, h, n+m, d]
        v_all = torch.cat([v_cached, v_new], dim=2)

        attn = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale  # [B, h, m, n+m]
        attn = self._activate(attn)
        # mask: [m, n+m] - prefix all visible, new positions causal
        mask = torch.ones(m, n + m, device=x_new.device, dtype=attn.dtype)
        mask[:, n:] = torch.ones(m, m, device=x_new.device, dtype=attn.dtype).tril()
        attn = attn * mask[None, None, :, :]
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v_all)  # [B, h, m, d]
        out = out.transpose(1, 2).reshape(B, m, self.inner)
        out = self.out_proj(out)
        k_all_flat = k_all.transpose(1, 2).reshape(B, n + m, self.inner)
        v_all_flat = v_all.transpose(1, 2).reshape(B, n + m, self.inner)
        return out, (k_all_flat, v_all_flat)

    def forward_stale_kv(
        self,
        x: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
    ) -> torch.Tensor:
        """Full-sequence forward with ENTIRELY stale K,V (from theta_old).

        Only Q is computed from the current model; K,V are the cached values
        from a previous forward under theta_old. This is the most extreme
        staleness: every position's attention uses mismatched Q (new) and K,V (old).

        Args:
            x: [B, L, H] hidden states (normed by the block caller).
            cached_k/v: [B, L, inner] stale K,V for ALL positions.

        Returns: out [B, L, H].
        """
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = cached_k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = cached_v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, L, L]
        attn = self._activate(attn)
        causal = torch.ones(L, L, device=x.device, dtype=attn.dtype).tril()
        attn = attn * causal[None, None, :, :]
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, self.inner)
        return self.out_proj(out)

    def _build_keep_mask(
        self,
        L: int,
        attn_mask: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if attn_mask is not None:
            keep = (attn_mask > -float("inf")).to(dtype)
            if keep.dim() == 2:
                keep = keep[None, None, :, :]
            return keep
        # default: strict causal lower-triangular (position i attends to j <= i)
        causal = torch.ones(L, L, device=device, dtype=dtype).tril()
        return causal[None, None, :, :]
