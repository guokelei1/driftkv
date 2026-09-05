"""Executable, label-free probe estimator for an S4 functional correction.

The estimator reads Parent K/V, fixed pre-cutover history probes and Current
parameters.  It reuses the no-materialized-prefix fused reader and compact
Current carrier primitives, then averages responses from a preregistered set
of history-derived probes.  Current-Exact K/V is deliberately absent from the
API; it is used only by the surrounding evaluation as a reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache
from pro_lazy_reader import generate_lazy_pro_probe_components


MAX_PROBES = 8
PROBE_COUNTS = (1, 2, 4, 8)
# Eight equally spaced points in the latest 128 events. Smaller sets are
# deterministic subsets and never depend on candidates or outcomes.
PROBE_HISTORY_OFFSETS = (-128, -110, -92, -74, -55, -37, -19, -1)
PROBE_INDEX_SETS = {
    1: (7,),
    2: (0, 7),
    4: (0, 2, 5, 7),
    8: tuple(range(8)),
}


@dataclass(frozen=True)
class FunctionalProbeEstimate:
    corrections_by_probe_count: dict[int, tuple[torch.Tensor, ...]]
    individual_probe_corrections: tuple[torch.Tensor, ...]
    replay_max_abs_error: float


def fixed_history_probe_items(item_ids: torch.Tensor) -> torch.Tensor:
    """Select the frozen eight history-derived probe items."""
    if item_ids.ndim != 2 or item_ids.shape[1] < 128:
        raise ValueError("functional probes require at least 128 history items")
    positions = torch.tensor(
        [item_ids.shape[1] + offset for offset in PROBE_HISTORY_OFFSETS],
        dtype=torch.long,
        device=item_ids.device,
    )
    return item_ids.index_select(1, positions)


def repeat_cache_for_probes(cache: HSTUKVCache, probes: int) -> HSTUKVCache:
    if probes < 1:
        raise ValueError("probe count must be positive")
    return HSTUKVCache(
        k=cache.k.repeat_interleave(probes, dim=1),
        v=cache.v.repeat_interleave(probes, dim=1),
        seq_len=cache.seq_len,
    )


@torch.inference_mode()
def estimate_functional_probe_means(
    model,
    parent_reuse_cache: HSTUKVCache,
    carrier_cache: HSTUKVCache,
    joint_maps: tuple[torch.Tensor, ...],
    probe_items: torch.Tensor,
    *,
    old_positions: int,
) -> FunctionalProbeEstimate:
    """Estimate nested 1/2/4/8-probe candidate-shared S4 sidecars."""
    if probe_items.ndim != 2 or probe_items.shape[1] != MAX_PROBES:
        raise ValueError(f"probe items must have shape [B,{MAX_PROBES}]")
    batch = probe_items.shape[0]
    if parent_reuse_cache.k.shape[1] != batch or carrier_cache.k.shape[1] != batch:
        raise ValueError("probe and cache batch dimensions differ")
    repeated_parent = repeat_cache_for_probes(parent_reuse_cache, MAX_PROBES)
    repeated_carriers = repeat_cache_for_probes(carrier_cache, MAX_PROBES)
    components = generate_lazy_pro_probe_components(
        model,
        repeated_parent,
        repeated_carriers,
        joint_maps,
        probe_items.reshape(-1),
        old_positions=old_positions,
    )
    corrections_by_count: dict[int, tuple[torch.Tensor, ...]] = {}
    reshaped = tuple(
        value.reshape(batch, MAX_PROBES, *value.shape[1:])
        for value in components.corrections
    )
    for count in PROBE_COUNTS:
        indices = torch.tensor(
            PROBE_INDEX_SETS[count], dtype=torch.long, device=probe_items.device
        )
        corrections_by_count[count] = tuple(
            value.index_select(1, indices).mean(dim=1) for value in reshaped
        )
    return FunctionalProbeEstimate(
        corrections_by_probe_count=corrections_by_count,
        individual_probe_corrections=reshaped,
        replay_max_abs_error=components.replay_max_abs_error,
    )


def functional_probe_cost(
    *,
    layers: int,
    hidden: int,
    heads: int,
    context: int,
    repair_evidence: int,
    carriers: int,
    probes: int,
    temporal_freqs: int = 16,
) -> dict[str, int | float | list[int] | str]:
    """Conservative release-time FLOPs for carrier construction plus probes."""
    values = (
        layers,
        hidden,
        heads,
        context,
        repair_evidence,
        carriers,
        probes,
        temporal_freqs,
    )
    if min(values) < 1:
        raise ValueError("architecture and estimator values must be positive")
    if hidden % heads:
        raise ValueError("hidden size must be divisible by heads")
    if repair_evidence > context or repair_evidence % carriers:
        raise ValueError("carrier layout must evenly partition repair evidence")
    if probes not in PROBE_COUNTS:
        raise ValueError(f"probe count must be one of {PROBE_COUNTS}")

    old_positions = context - repair_evidence

    def input_projection_flops(tokens: int) -> int:
        return 2 * tokens * (2 * temporal_freqs) * hidden + 2 * tokens * hidden * hidden

    def block_linear_flops(tokens: int) -> int:
        return 2 * tokens * (5 * hidden * hidden)

    def attention_flops(pairs: int) -> int:
        return 4 * pairs * hidden

    full_pairs = context * (context + 1) // 2
    full = input_projection_flops(context) + layers * (
        block_linear_flops(context) + attention_flops(full_pairs)
    )
    fused_reads = probes * layers * (
        8 * hidden * hidden + 8 * heads * old_positions * hidden
    )
    carrier_pairs = old_positions * carriers + carriers * (carriers + 1) // 2
    probe_pairs = probes * (context + carriers + 1)
    tokens = carriers + probes
    carrier_and_probes = input_projection_flops(tokens) + layers * (
        block_linear_flops(tokens) + attention_flops(carrier_pairs + probe_pairs)
    )
    carrier_mass_scale = layers * carriers * hidden
    response_difference_and_mean = probes * layers * hidden
    total = (
        fused_reads
        + carrier_and_probes
        + carrier_mass_scale
        + response_difference_and_mean
    )
    sidecar_scalars = layers * hidden
    unique_parent_scalars = layers * context * 2 * hidden
    parent_stream_scalars = layers * (
        old_positions + probes * (old_positions + context)
    ) * 2 * hidden
    return {
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "context": context,
        "old_positions": old_positions,
        "repair_evidence": repair_evidence,
        "carriers": carriers,
        "probes": probes,
        "full_recompute_flops_per_user": full,
        "fused_parent_reads_flops_per_user": fused_reads,
        "carrier_and_probes_flops_per_user": carrier_and_probes,
        "carrier_mass_scale_flops_per_user": carrier_mass_scale,
        "response_difference_and_mean_flops_per_user": response_difference_and_mean,
        "total_flops_per_user": total,
        "over_full_fraction": total / full,
        "sidecar_write_scalars": sidecar_scalars,
        "sidecar_write_bytes_fp32": 4 * sidecar_scalars,
        "unique_parent_state_read_scalars": unique_parent_scalars,
        "conservative_parent_state_stream_scalars": parent_stream_scalars,
        "materialized_version_translated_prefix_scalars": 0,
        "version_map_shape_per_layer": [2 * hidden, 2 * hidden],
        "version_map_construction_frequency": "once_per_release_edge_not_per_user",
    }


def medium_cost_grid() -> tuple[dict[str, int | float | list[int] | str], ...]:
    return tuple(
        functional_probe_cost(
            layers=6,
            hidden=192,
            heads=6,
            context=1024,
            repair_evidence=128,
            carriers=carriers,
            probes=probes,
        )
        for carriers in (8, 16, 32, 64)
        for probes in PROBE_COUNTS
    )
