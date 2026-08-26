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
    max_seq_len: int = 2048
    block_variant: str = "legacy"
    relative_position_bias: bool = False
    causal_diagonal: str = "inclusive"


class PointwiseAttention(nn.Module):
    """HSTU pointwise aggregated attention (PMA).

    Defining feature (vs standard transformer attention):
      * No softmax normalisation. The attention activation is applied
        pointwise: ``a_ij = elu(q_i . k_j) + 1`` which is non-negative.
      * Causal masking is therefore *multiplicative* (zero-out future),
        not additive -inf (which would corrupt the unnormalised sum).

    This module owns the Q/K/V linear projections and returns, alongside the
    block output, the per-layer K and V tensors. Those K/V tensors are the
    versioned prefix representation being migrated, so they are a first-class
    output rather than an internal detail.
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
        self.block_variant = cfg.block_variant
        self.causal_diagonal = cfg.causal_diagonal
        if cfg.relative_position_bias:
            self.position_bias = nn.Embedding(2 * cfg.max_seq_len - 1, self.num_heads)
            nn.init.normal_(self.position_bias.weight, mean=0.0, std=0.02)
        else:
            self.position_bias = None

    def _activate(self, attn: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(attn)
        if self.activation == "silu":
            return F.silu(attn)
        return F.elu(attn) + 1.0

    def _relative_position_bias(
        self,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.position_bias is None:
            return None
        offset = self.cfg.max_seq_len - 1
        indices = key_positions.unsqueeze(0) - query_positions.unsqueeze(1) + offset
        values = self.position_bias(indices).permute(2, 0, 1).unsqueeze(0)
        return values.to(dtype=dtype)

    def _project(self, x: torch.Tensor):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        if self.block_variant == "hstu_reference":
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)
        return q, k, v

    def project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project normalized token states to cache-layout K/V without attention.

        Layer-0 K/V for a token depends only on that token's normalized input.
        Exposing this projection allows a dependency-closed state transition
        to refresh selected layer-0 positions without pretending that upper
        layer K/V have the same locality.
        """
        if x.ndim != 3 or x.shape[-1] != self.cfg.hidden_size:
            raise ValueError("project_kv input must have shape [B, L, hidden]")
        batch, length, _ = x.shape
        _, k, v = self._project(x)
        return (
            k.transpose(1, 2).reshape(batch, length, self.inner),
            v.transpose(1, 2).reshape(batch, length, self.inner),
        )

    def _aggregate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        keep: torch.Tensor,
    ) -> torch.Tensor:
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        bias = self._relative_position_bias(query_positions, key_positions, attn.dtype)
        if bias is not None:
            attn = attn + bias
        attn = self._activate(attn)
        if self.block_variant == "hstu_reference":
            attn = attn / self.cfg.max_seq_len
        attn = self.attn_dropout(attn * keep)
        return torch.matmul(attn, v)

    def _finish(self, out: torch.Tensor) -> torch.Tensor:
        B, _, L, _ = out.shape
        out = out.transpose(1, 2).reshape(B, L, self.inner)
        if self.block_variant == "hstu_reference":
            return out
        return self.out_proj(out)

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
                are the cacheable batched prefix representation F(theta, x).

        Returns: out [B, L, H], and optionally (k, v).
        """
        B, L, _ = x.shape
        q, k, v = self._project(x)
        keep = self._build_keep_mask(L, attn_mask, x.device, x.dtype)
        positions = torch.arange(L, device=x.device)
        out = self._finish(self._aggregate(q, k, v, positions, positions, keep))

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
        q, k_new, v_new = self._project(x_new)
        k_cached = cached_k.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_cached = cached_v.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        k_all = torch.cat([k_cached, k_new], dim=2)  # [B, h, n+m, d]
        v_all = torch.cat([v_cached, v_new], dim=2)

        mask = torch.ones(m, n + m, device=x_new.device, dtype=x_new.dtype)
        diagonal = -1 if self.causal_diagonal == "exclusive" else 0
        mask[:, n:] = torch.ones(
            m, m, device=x_new.device, dtype=x_new.dtype
        ).tril(diagonal=diagonal)
        query_positions = torch.arange(n, n + m, device=x_new.device)
        key_positions = torch.arange(n + m, device=x_new.device)
        out = self._finish(
            self._aggregate(
                q,
                k_all,
                v_all,
                query_positions,
                key_positions,
                mask[None, None, :, :],
            )
        )
        k_all_flat = k_all.transpose(1, 2).reshape(B, n + m, self.inner)
        v_all_flat = v_all.transpose(1, 2).reshape(B, n + m, self.inner)
        return out, (k_all_flat, v_all_flat)

    def forward_with_cache_new_kv(
        self,
        x_new: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
    ):
        B, m, _ = x_new.shape
        if m == 1:
            return self.forward_one_with_cache_new_kv(x_new, cached_k, cached_v)
        n = cached_k.shape[1]
        q, k_new, v_new = self._project(x_new)
        k_cached = cached_k.view(
            B, n, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v_cached = cached_v.view(
            B, n, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k_all = torch.cat([k_cached, k_new], dim=2)
        v_all = torch.cat([v_cached, v_new], dim=2)
        mask = torch.ones(m, n + m, device=x_new.device, dtype=x_new.dtype)
        diagonal = -1 if self.causal_diagonal == "exclusive" else 0
        mask[:, n:] = torch.ones(
            m, m, device=x_new.device, dtype=x_new.dtype
        ).tril(diagonal=diagonal)
        query_positions = torch.arange(n, n + m, device=x_new.device)
        key_positions = torch.arange(n + m, device=x_new.device)
        out = self._finish(
            self._aggregate(
                q,
                k_all,
                v_all,
                query_positions,
                key_positions,
                mask[None, None, :, :],
            )
        )
        k_new_flat = k_new.transpose(1, 2).reshape(B, m, self.inner)
        v_new_flat = v_new.transpose(1, 2).reshape(B, m, self.inner)
        return out, (k_new_flat, v_new_flat)

    def forward_one_with_cache_new_kv(
        self,
        x_new: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
    ):
        """Append one token without materialising a copied full K/V cache.

        The ordinary incremental path concatenates old and new K/V in order to
        return the whole updated cache. Serving storage normally keeps the
        prefix in place and writes only the new K/V row. This inference path
        computes the mathematically identical one-token attention result while
        returning only that row.
        """
        B, m, _ = x_new.shape
        if m != 1:
            raise ValueError("append-only attention requires exactly one new token")
        n = cached_k.shape[1]
        q, k_new, v_new = self._project(x_new)
        k_cached = cached_k.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_cached = cached_v.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        query_positions = torch.tensor([n], device=x_new.device)
        prefix_positions = torch.arange(n, device=x_new.device)

        prefix_weights = torch.matmul(q, k_cached.transpose(-2, -1)) * self.scale
        prefix_bias = self._relative_position_bias(
            query_positions, prefix_positions, prefix_weights.dtype
        )
        if prefix_bias is not None:
            prefix_weights = prefix_weights + prefix_bias
        prefix_weights = self.attn_dropout(self._activate(prefix_weights))
        out = torch.matmul(prefix_weights, v_cached)

        self_weights = (q * k_new).sum(dim=-1, keepdim=True) * self.scale
        self_bias = self._relative_position_bias(
            query_positions, query_positions, self_weights.dtype
        )
        if self_bias is not None:
            self_weights = self_weights + self_bias
        self_weights = self.attn_dropout(self._activate(self_weights))
        out = out + self_weights * v_new
        return (
            self._finish(out),
            (
                k_new.transpose(1, 2).reshape(B, 1, self.inner),
                v_new.transpose(1, 2).reshape(B, 1, self.inner),
            ),
        )

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
        q, _, _ = self._project(x)
        k = cached_k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = cached_v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        diagonal = -1 if self.causal_diagonal == "exclusive" else 0
        causal = torch.ones(L, L, device=x.device, dtype=x.dtype).tril(
            diagonal=diagonal
        )
        positions = torch.arange(L, device=x.device)
        return self._finish(
            self._aggregate(
                q,
                k,
                v,
                positions,
                positions,
                causal[None, None, :, :],
            )
        )

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
        diagonal = -1 if self.causal_diagonal == "exclusive" else 0
        causal = torch.ones(L, L, device=device, dtype=dtype).tril(
            diagonal=diagonal
        )
        return causal[None, None, :, :]
