"""Dependency-closed Tail replay estimator for an S4 functional sidecar.

The estimator replaces no persistent K/V.  It transiently replays the frozen
recent suffix with Current weights against the untouched Parent prefix, reads
the mixed-versus-Parent response on fixed history queries, and persists only a
layerwise S4 correction.  Current-Exact state is absent from this API.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache, hybrid_tail_refresh
from insight_two.functional_probe_estimator import PROBE_INDEX_SETS
from reader_compatibility_correction import _stage_path


TAIL_WIDTH = 128
PROBE_COUNTS = (1, 2, 4)


@dataclass(frozen=True)
class TailFunctionalEstimate:
    corrections_by_probe_count: dict[int, tuple[torch.Tensor, ...]]
    single_probe_replay_max_abs_error: float
    parent_prefix_max_abs_change: float


@torch.inference_mode()
def estimate_tail_functional_sidecars(
    model,
    parent_cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    probe_items: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    tail_width: int = TAIL_WIDTH,
) -> TailFunctionalEstimate:
    """Generate nested P1/P2/P4 S4 corrections without Current-Exact K/V."""
    if item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape:
        raise ValueError("raw prefix tensors differ")
    if item_ids.ndim != 2 or item_ids.shape[1] != parent_cache.seq_len:
        raise ValueError("raw history width differs from Parent cache")
    if probe_items.ndim != 2 or probe_items.shape[1] != 8:
        raise ValueError("tail functional estimator requires eight frozen probe items")
    if query_time_deltas.shape != (item_ids.shape[0],):
        raise ValueError("query time deltas differ from the user batch")
    if not 1 <= tail_width <= parent_cache.seq_len:
        raise ValueError("tail width must be within the Parent cache")

    parent_k = parent_cache.k.clone()
    parent_v = parent_cache.v.clone()
    mixed_cache = hybrid_tail_refresh(
        model,
        parent_cache,
        item_ids,
        behaviors,
        time_deltas,
        width=tail_width,
    )
    prefix = parent_cache.seq_len - tail_width
    prefix_error = max(
        float(torch.max(torch.abs(mixed_cache.k[:, :, :prefix] - parent_cache.k[:, :, :prefix])))
        if prefix
        else 0.0,
        float(torch.max(torch.abs(mixed_cache.v[:, :, :prefix] - parent_cache.v[:, :, :prefix])))
        if prefix
        else 0.0,
        float(torch.max(torch.abs(parent_cache.k - parent_k))),
        float(torch.max(torch.abs(parent_cache.v - parent_v))),
    )

    corrections: dict[int, tuple[torch.Tensor, ...]] = {}
    single_probe_error = float("inf")
    for count in PROBE_COUNTS:
        indices = torch.tensor(
            PROBE_INDEX_SETS[count], dtype=torch.long, device=probe_items.device
        )
        selected = probe_items.index_select(1, indices)
        shared_readout, correction, _ = _stage_path(
            model,
            mixed_cache,
            parent_cache,
            selected,
            query_time_deltas,
            stage="av_aggregation",
            mode="shared",
        )
        corrections[count] = tuple(value.detach().clone() for value in correction)
        if count == 1:
            _, mixed_readout = model.observe_cc_reuse(
                mixed_cache, selected, query_time_deltas
            )
            single_probe_error = float(
                torch.max(torch.abs(shared_readout - mixed_readout))
            )
    return TailFunctionalEstimate(
        corrections_by_probe_count=corrections,
        single_probe_replay_max_abs_error=single_probe_error,
        parent_prefix_max_abs_change=prefix_error,
    )


def tail_functional_cost(
    *,
    layers: int,
    hidden: int,
    heads: int,
    context: int,
    tail_width: int,
    probes: int,
    temporal_freqs: int = 16,
) -> dict[str, int | float | str]:
    """Conservative causal FLOPs for dense mixed-tail replay plus two probe reads."""
    if min(layers, hidden, heads, context, tail_width, probes, temporal_freqs) < 1:
        raise ValueError("architecture and estimator dimensions must be positive")
    if hidden % heads or tail_width > context or probes not in PROBE_COUNTS:
        raise ValueError("invalid tail-functional configuration")

    def input_projection(tokens: int) -> int:
        return 2 * tokens * (2 * temporal_freqs) * hidden + 2 * tokens * hidden * hidden

    def block_linear(tokens: int) -> int:
        return 2 * tokens * (5 * hidden * hidden)

    def attention(pairs: int) -> int:
        return 4 * pairs * hidden

    full_pairs = context * (context + 1) // 2
    full = input_projection(context) + layers * (
        block_linear(context) + attention(full_pairs)
    )
    prefix = context - tail_width
    tail_pairs = prefix * tail_width + tail_width * (tail_width + 1) // 2
    tail = input_projection(tail_width) + layers * (
        block_linear(tail_width) + attention(tail_pairs)
    )
    # Each probe runs one Current query path and conservatively reads both the
    # full mixed cache and the full Parent cache at every layer.
    one_probe = input_projection(1) + layers * (
        block_linear(1) + 2 * attention(context)
    )
    # Conservative bookkeeping for two response tensors, their signed
    # difference, accumulation and mean.  This matches the repository's prior
    # probe-estimator convention rather than claiming these pointwise ops free.
    response_difference_and_mean = 5 * probes * layers * hidden
    injection_adds_one_request = layers * hidden
    total = tail + probes * one_probe + response_difference_and_mean
    return {
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "context": context,
        "tail_width": tail_width,
        "probes": probes,
        "full_recompute_flops_per_user": full,
        "tail_replay_flops_per_user": tail,
        "probe_reads_flops_per_user": probes * one_probe,
        "response_difference_and_mean_flops_per_user": response_difference_and_mean,
        "total_generation_flops_per_user": total,
        "over_full_fraction": total / full,
        "sidecar_scalars": layers * hidden,
        "sidecar_bytes_fp32": 4 * layers * hidden,
        "request_injection_adds_per_candidate": injection_adds_one_request,
        "transient_Current_positions": tail_width,
        "persistent_Current_KV_positions": 0,
        "cost_semantics": "generation_primary_request_injection_reported_separately",
    }


def medium_tail_functional_costs() -> tuple[dict[str, int | float | str], ...]:
    return tuple(
        tail_functional_cost(
            layers=6,
            hidden=192,
            heads=6,
            context=1024,
            tail_width=TAIL_WIDTH,
            probes=probes,
        )
        for probes in PROBE_COUNTS
    )
