"""Signed, head-wise diagnostic interventions for candidate-shared HSTU evidence.

The functions in this module are observation-only.  They read both Current
Exact and Parent Reuse caches to decompose the Current-minus-Reuse prefix read
at the same query state.  That oracle decomposition is not an executable cache
transition and must never be added to the scale action set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTUKVCache


InterventionMode = Literal[
    "exact",
    "reuse",
    "shared_only",
    "residual_only",
    "full_delta",
]


@dataclass(frozen=True)
class SignedInterventionResult:
    scores: torch.Tensor
    readout: torch.Tensor
    shared_components: tuple[torch.Tensor, ...]
    head_metrics: tuple[dict[str, torch.Tensor], ...]


def nested_width_indices(full_width: int, width: int) -> np.ndarray:
    """Return deterministic, nested, candidate-order-independent stride slots."""
    if full_width < 1 or width < 1 or width > full_width:
        raise ValueError("candidate widths must satisfy 1 <= width <= full_width")
    if full_width % width:
        raise ValueError("formal nested widths must divide the full candidate width")
    return np.arange(0, full_width, full_width // width, dtype=np.int64)


def signed_candidate_decomposition(
    exact_heads: torch.Tensor,
    reuse_heads: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Orthogonally split signed per-head delta into candidate mean and residual.

    Inputs have shape ``[batch, candidates, heads, query_positions, head_dim]``.
    No norm, absolute value, or candidate-wise history normalization is applied
    before the split.
    """
    if exact_heads.shape != reuse_heads.shape or exact_heads.ndim != 5:
        raise ValueError("signed head contributions must share [B,C,H,Q,D] shape")
    if exact_heads.shape[1] < 1:
        raise ValueError("candidate bank must be nonempty")
    delta = exact_heads.float() - reuse_heads.float()
    shared = delta.mean(dim=1, keepdim=True)
    residual = delta - shared
    broadcast = shared.expand_as(delta)
    reduce = (1, 3, 4)
    total_energy = delta.square().sum(dim=reduce)
    shared_energy = broadcast.square().sum(dim=reduce)
    residual_energy = residual.square().sum(dim=reduce)
    cross = (broadcast * residual).sum(dim=reduce)
    metrics = {
        "total_energy": total_energy,
        "shared_energy": shared_energy,
        "residual_energy": residual_energy,
        "shared_energy_fraction": shared_energy / total_energy.clamp_min(1e-20),
        "orthogonality_error": cross.abs() / total_energy.clamp_min(1e-20),
    }
    return shared.to(exact_heads.dtype), residual.to(exact_heads.dtype), metrics


def _cached_prefix_heads(
    attention,
    q: torch.Tensor,
    cache: HSTUKVCache,
    layer: int,
    candidate_count: int,
) -> torch.Tensor:
    """Return signed, pre-output-projection prefix reads per attention head."""
    batch_candidates = q.shape[0]
    if batch_candidates % candidate_count:
        raise ValueError("flattened query batch is not divisible by candidate count")
    batch = batch_candidates // candidate_count
    length = cache.seq_len
    cached_k = cache.k[layer].repeat_interleave(candidate_count, dim=0)
    cached_v = cache.v[layer].repeat_interleave(candidate_count, dim=0)
    cached_k = cached_k.view(
        batch_candidates, length, attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    cached_v = cached_v.view(
        batch_candidates, length, attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    weights = attention._activate(
        torch.matmul(q, cached_k.transpose(-2, -1)) * attention.scale
    )
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    weights = attention.attn_dropout(weights)
    heads = torch.matmul(weights, cached_v)
    return heads.reshape(
        batch,
        candidate_count,
        attention.num_heads,
        q.shape[2],
        attention.head_dim,
    )


def _block_update(block, x_norm: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    attention_out = block.attn._finish(heads)
    if block.block_variant == "hstu_reference":
        assert block.attn_output_norm is not None
        return block.attn.out_proj(
            block.attn_output_norm(attention_out) * F.silu(block.gate_proj(x_norm))
        )
    if block.gating == "silu_gate":
        return attention_out * F.silu(block.gate_proj(x_norm))
    if block.gating == "glu":
        return attention_out * torch.sigmoid(block.gate_proj(x_norm))
    if block.gating == "ffn":
        return block.fc2(F.silu(block.fc1(x_norm)) * block.fc3(x_norm))
    return attention_out


@torch.inference_mode()
def signed_head_intervention(
    model,
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    mode: InterventionMode,
) -> SignedInterventionResult:
    """Run a coherent Current reader with a dynamic signed cache-read intervention.

    At every layer, Exact and Reuse prefix reads are evaluated at the path's
    current query hidden state.  Their signed per-head delta is split across
    the candidate dimension.  ``full_delta`` must therefore equal the Exact
    cache read at every layer while ``shared_only`` and ``residual_only`` are
    diagnostic ablations.
    """
    if mode not in {"exact", "reuse", "shared_only", "residual_only", "full_delta"}:
        raise ValueError(f"unknown signed intervention mode: {mode}")
    if exact_cache.seq_len != reuse_cache.seq_len:
        raise ValueError("Exact and Reuse caches must have the same sequence length")
    if exact_cache.k.shape != reuse_cache.k.shape or exact_cache.v.shape != reuse_cache.v.shape:
        raise ValueError("Exact and Reuse cache tensor shapes differ")
    if candidate_ids.ndim != 2 or candidate_ids.shape[1] < 1:
        raise ValueError("candidate_ids must have shape [B,C] with C >= 1")
    if candidate_ids.shape[0] != exact_cache.k.shape[1]:
        raise ValueError("candidate bank and cache batches differ")
    if model.cfg.relative_position_bias:
        raise ValueError("the signed intervention requires no relative-position bias")

    batch, candidates = candidate_ids.shape
    x = model.embed_query_tokens(candidate_ids, query_time_deltas).reshape(
        batch * candidates, 1, model.cfg.hidden_size
    )
    shared_components: list[torch.Tensor] = []
    head_metrics: list[dict[str, torch.Tensor]] = []

    for layer, block in enumerate(model.blocks):
        residual_x = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        exact_heads = _cached_prefix_heads(
            block.attn, q, exact_cache, layer, candidates
        )
        reuse_heads = _cached_prefix_heads(
            block.attn, q, reuse_cache, layer, candidates
        )
        shared, candidate_residual, metrics = signed_candidate_decomposition(
            exact_heads, reuse_heads
        )
        shared_components.append(shared[:, 0, :, 0, :].float().detach().cpu())
        head_metrics.append({key: value.detach().cpu() for key, value in metrics.items()})

        if mode == "exact":
            selected = exact_heads
        elif mode == "reuse":
            selected = reuse_heads
        elif mode == "shared_only":
            selected = reuse_heads + shared
        elif mode == "residual_only":
            selected = reuse_heads + candidate_residual
        else:
            selected = reuse_heads + shared + candidate_residual

        selected = selected.reshape(
            batch * candidates,
            block.attn.num_heads,
            1,
            block.attn.head_dim,
        )
        if block.attn.causal_diagonal == "inclusive":
            self_weight = block.attn._activate(
                (q * k_new).sum(dim=-1, keepdim=True) * block.attn.scale
            )
            if block.attn.block_variant == "hstu_reference":
                self_weight = self_weight / block.attn.cfg.max_seq_len
            selected = selected + self_weight * v_new
        update = _block_update(block, x_norm, selected)
        x = residual_x + update

    readout = model.final_norm(x).reshape(batch, candidates, model.cfg.hidden_size)
    scores = model.cc_score_head(readout).squeeze(-1)
    return SignedInterventionResult(
        scores=scores,
        readout=readout,
        shared_components=tuple(shared_components),
        head_metrics=tuple(head_metrics),
    )
