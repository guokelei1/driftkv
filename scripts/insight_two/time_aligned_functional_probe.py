"""Time-aligned fixed probes for an executable S4 correction.

The legacy lightweight probe hard-codes a zero query-time delta.  This module
keeps its Parent-state/current-parameter computation unchanged but evaluates
the probe at the known release cutover delta used by the target requests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from candidate_shared_causal import _block_update, _cached_prefix_heads
from hstu_kvcache.models import HSTUKVCache
from pro_lazy_reader import _cache_slice, fused_joint_map_prefix_heads
from reader_compatibility_correction import _self_heads, intervene_reader_correction


@dataclass(frozen=True)
class TimeAlignedProbeResult:
    corrections: tuple[torch.Tensor, ...]
    source_scores: torch.Tensor
    source_readout: torch.Tensor
    replay_max_abs_error: float


@torch.inference_mode()
def generate_time_aligned_probe(
    model,
    parent_reuse_cache: HSTUKVCache,
    carrier_cache: HSTUKVCache,
    joint_maps: tuple[torch.Tensor, ...],
    probe_item_ids: torch.Tensor,
    probe_time_deltas: torch.Tensor,
    *,
    old_positions: int,
) -> TimeAlignedProbeResult:
    if probe_item_ids.ndim != 1 or probe_time_deltas.ndim != 1:
        raise ValueError("probe items and time deltas must have shape [B]")
    if probe_item_ids.shape != probe_time_deltas.shape:
        raise ValueError("probe item and time-delta batches differ")
    if probe_item_ids.shape[0] != parent_reuse_cache.k.shape[1]:
        raise ValueError("probe and Parent cache batches differ")
    if carrier_cache.k.shape[1] != probe_item_ids.shape[0]:
        raise ValueError("probe and carrier batches differ")
    if len(joint_maps) != len(model.blocks):
        raise ValueError("joint-map layer count differs")
    if model.cfg.relative_position_bias:
        raise ValueError("time-aligned fused probes require no relative-position bias")

    candidates = probe_item_ids[:, None]
    batch = probe_item_ids.shape[0]
    x = model.embed_query_tokens(candidates, probe_time_deltas).reshape(
        batch, 1, model.cfg.hidden_size
    )
    old_reuse = _cache_slice(parent_reuse_cache, 0, old_positions)
    recent_reuse = _cache_slice(
        parent_reuse_cache, old_positions, parent_reuse_cache.seq_len
    )
    corrections = []
    for layer, block in enumerate(model.blocks):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        mapped_old = fused_joint_map_prefix_heads(
            block.attn,
            q,
            parent_reuse_cache,
            joint_maps[layer],
            layer=layer,
            length=old_positions,
        )
        carrier_heads = _cached_prefix_heads(
            block.attn, q, carrier_cache, layer, 1
        ).reshape(batch, block.attn.num_heads, 1, block.attn.head_dim)
        reuse_old = _cached_prefix_heads(block.attn, q, old_reuse, layer, 1).reshape(
            batch, block.attn.num_heads, 1, block.attn.head_dim
        )
        reuse_recent = _cached_prefix_heads(
            block.attn, q, recent_reuse, layer, 1
        ).reshape(batch, block.attn.num_heads, 1, block.attn.head_dim)
        correction = (mapped_old + carrier_heads - reuse_old - reuse_recent).squeeze(2)
        corrections.append(correction.detach())
        source_heads = mapped_old + carrier_heads + _self_heads(
            block, q, k_new, v_new
        )
        x = residual + _block_update(block, x_norm, source_heads)
    readout = model.final_norm(x).reshape(batch, 1, model.cfg.hidden_size)
    source_scores = model.cc_score_head(readout).squeeze(-1)
    replay_scores, _ = intervene_reader_correction(
        model,
        parent_reuse_cache,
        candidates,
        probe_time_deltas,
        stage="av_aggregation",
        corrections=tuple(corrections),
    )
    return TimeAlignedProbeResult(
        corrections=tuple(corrections),
        source_scores=source_scores.detach(),
        source_readout=readout[:, 0].detach(),
        replay_max_abs_error=float(torch.max(torch.abs(replay_scores - source_scores))),
    )
