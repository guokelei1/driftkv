"""Executable compact-probe AV broadcast residual canary primitive."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache
from reader_compatibility_correction import _stage_path, intervene_reader_correction


@dataclass(frozen=True)
class BroadcastResidual:
    corrections: tuple[torch.Tensor, ...]
    replay_max_abs_error: float


@torch.inference_mode()
def generate_av_broadcast_residual(
    model,
    probe_source_cache: HSTUKVCache,
    parent_reuse_cache: HSTUKVCache,
    probe_item_ids: torch.Tensor,
) -> BroadcastResidual:
    """Generate one per-user AV sidecar from a fixed one-item Current probe."""
    if probe_item_ids.ndim != 1:
        raise ValueError("probe item IDs must have shape [B]")
    if probe_source_cache.seq_len != parent_reuse_cache.seq_len:
        raise ValueError("probe source and Parent Reuse cache lengths differ")
    probe_candidates = probe_item_ids[:, None]
    probe_deltas = torch.zeros(
        probe_item_ids.shape[0], dtype=torch.float32, device=probe_item_ids.device
    )
    readout, corrections, _ = _stage_path(
        model,
        probe_source_cache,
        parent_reuse_cache,
        probe_candidates,
        probe_deltas,
        stage="av_aggregation",
        mode="shared",
    )
    expected_scores = model.cc_score_head(readout).squeeze(-1)
    replay_scores, _ = intervene_reader_correction(
        model,
        parent_reuse_cache,
        probe_candidates,
        probe_deltas,
        stage="av_aggregation",
        corrections=corrections,
    )
    return BroadcastResidual(
        corrections=tuple(value.detach() for value in corrections),
        replay_max_abs_error=float(torch.max(torch.abs(replay_scores - expected_scores))),
    )
