"""Fair release-time FLOP and state-I/O model for lightweight PRO."""

from __future__ import annotations


LAYERS = 4
HIDDEN = 128
INNER = 128
HEADS = 4
TEMPORAL_FREQS = 16
CONTEXT = 512
REPAIR_EVIDENCE = 128
OLD_POSITIONS = CONTEXT - REPAIR_EVIDENCE


def input_projection_flops(tokens: int) -> int:
    return 2 * tokens * (2 * TEMPORAL_FREQS) * HIDDEN + 2 * tokens * HIDDEN * HIDDEN


def block_linear_flops(tokens: int) -> int:
    return 2 * tokens * (
        3 * HIDDEN * INNER + INNER * HIDDEN + HIDDEN * HIDDEN
    )


def attention_flops(pairs: int) -> int:
    return 4 * pairs * INNER


def architecture_pro_cost(
    *,
    layers: int,
    hidden: int,
    heads: int,
    context: int,
    repair_evidence: int,
    carriers: int,
    temporal_freqs: int = TEMPORAL_FREQS,
) -> dict[str, int | float | list[int] | str]:
    """Scale the frozen one-probe PRO structure to another HSTU shape.

    The calculation deliberately keeps the original arithmetic convention and
    does not infer a carrier count from quality.  Callers freeze the structural
    ratios first and use this function only to verify the Exact-relative cost.
    """
    values = (layers, hidden, heads, context, repair_evidence, carriers, temporal_freqs)
    if min(values) < 1:
        raise ValueError("PRO architecture and layout values must be positive")
    if hidden % heads:
        raise ValueError("hidden size must be divisible by heads")
    if repair_evidence > context:
        raise ValueError("repair evidence cannot exceed context")
    if repair_evidence % carriers:
        raise ValueError("carriers must divide repair evidence")

    inner = hidden
    old_positions = context - repair_evidence

    def scaled_input_projection_flops(tokens: int) -> int:
        return (
            2 * tokens * (2 * temporal_freqs) * hidden
            + 2 * tokens * hidden * hidden
        )

    def scaled_block_linear_flops(tokens: int) -> int:
        return 2 * tokens * (
            3 * hidden * inner + inner * hidden + hidden * hidden
        )

    def scaled_attention_flops(pairs: int) -> int:
        return 4 * pairs * inner

    full_pairs = context * (context + 1) // 2
    full = scaled_input_projection_flops(context) + layers * (
        scaled_block_linear_flops(context) + scaled_attention_flops(full_pairs)
    )
    lazy_per_layer = (
        8 * inner * inner + 8 * heads * old_positions * inner
    )
    lazy = layers * lazy_per_layer
    carrier_pairs = old_positions * carriers + carriers * (carriers + 1) // 2
    probe_standard_pairs = context + carriers + 1
    tokens = carriers + 1
    carrier_and_probe = scaled_input_projection_flops(tokens) + layers * (
        scaled_block_linear_flops(tokens)
        + scaled_attention_flops(carrier_pairs + probe_standard_pairs)
    )
    carrier_mass_scale = layers * carriers * inner
    sidecar_difference = layers * inner
    total = lazy + carrier_and_probe + carrier_mass_scale + sidecar_difference
    scalar_bytes = 4
    unique_parent_scalars = layers * context * 2 * inner
    parent_stream_scalars = layers * (
        old_positions + old_positions + context
    ) * 2 * inner
    return {
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "head_dimension": hidden // heads,
        "context": context,
        "old_positions": old_positions,
        "repair_evidence": repair_evidence,
        "carriers": carriers,
        "represented_mass": repair_evidence // carriers,
        "full_recompute_flops_per_user": full,
        "lazy_joint_map_flops_per_user": lazy,
        "carrier_and_probe_flops_per_user": carrier_and_probe,
        "carrier_mass_scale_flops_per_user": carrier_mass_scale,
        "sidecar_difference_flops_per_user": sidecar_difference,
        "total_flops_per_user": total,
        "over_full_fraction": total / full,
        "reduction_vs_full_fraction": 1.0 - total / full,
        "unique_parent_state_read_scalars": unique_parent_scalars,
        "unique_parent_state_read_bytes_fp32": unique_parent_scalars * scalar_bytes,
        "conservative_parent_state_stream_scalars": parent_stream_scalars,
        "conservative_parent_state_stream_bytes_fp32": parent_stream_scalars * scalar_bytes,
        "sidecar_write_scalars": layers * inner,
        "sidecar_write_bytes_fp32": layers * inner * scalar_bytes,
        "post_release_coverage_scale_flops_per_user_request": layers * inner,
        "post_release_injection_adds_per_candidate": layers * inner,
        "materialized_version_translated_prefix_scalars": 0,
        "version_map_shape_per_layer": [2 * inner, 2 * inner],
        "version_map_construction_frequency": "once_per_release_edge_not_per_user",
    }


def full_recompute_flops() -> int:
    pairs = CONTEXT * (CONTEXT + 1) // 2
    return input_projection_flops(CONTEXT) + LAYERS * (
        block_linear_flops(CONTEXT) + attention_flops(pairs)
    )


def lazy_joint_map_flops_per_layer(probe_queries: int = 1) -> int:
    # Per head: q @ A_K^T, two length-(2I) history reductions, then @ A_V.
    query_and_value_maps = 8 * probe_queries * INNER * INNER
    streamed_parent_reductions = (
        8 * probe_queries * HEADS * OLD_POSITIONS * INNER
    )
    return query_and_value_maps + streamed_parent_reductions


def pro_cost(carriers: int) -> dict[str, int | float]:
    if carriers not in (16, 32):
        raise ValueError("the frozen lightweight PRO cost axis is 16/32 carriers")
    return architecture_pro_cost(
        layers=LAYERS,
        hidden=HIDDEN,
        heads=HEADS,
        context=CONTEXT,
        repair_evidence=REPAIR_EVIDENCE,
        carriers=carriers,
    )


def progressive_pro_cost(
    carriers: int, *, probes: int = 2
) -> dict[str, int | float | list[int] | str]:
    """Conservative FLOP/I/O model for the self-calibrating PRO frontier.

    Carriers are constructed once.  Every fixed probe independently executes
    the fused old-prefix read, compact-carrier read, Reuse counterfactual and
    Current block path.  The small calibration term covers two vector norms,
    direction combination and old/recent projections; it is deliberately not
    hidden inside the common serving cost.
    """
    if carriers not in (32, 48, 64):
        raise ValueError("the progressive PRO cost axis is 32/48/64 carriers")
    if probes not in (2, 3):
        raise ValueError("progressive PRO supports two or three fixed probes")
    carrier_pairs = OLD_POSITIONS * carriers + carriers * (carriers + 1) // 2
    probe_standard_pairs = probes * (CONTEXT + carriers + 1)
    tokens = carriers + probes
    lazy = LAYERS * lazy_joint_map_flops_per_layer(probe_queries=probes)
    carrier_and_probes = input_projection_flops(tokens) + LAYERS * (
        block_linear_flops(tokens)
        + attention_flops(carrier_pairs + probe_standard_pairs)
    )
    carrier_mass_scale = LAYERS * carriers * INNER
    # Per layer: two probe norms/directions, a combined direction, two segment
    # projections per probe, and scalar averaging/bookkeeping.
    amplitude_calibration = LAYERS * (
        probes * 6 * INNER + 4 * INNER + probes * 4 * INNER + 16
    )
    sidecar_scalars = LAYERS * INNER + 2 * LAYERS
    total = lazy + carrier_and_probes + carrier_mass_scale + amplitude_calibration
    full = full_recompute_flops()
    scalar_bytes = 4
    unique_parent_scalars = LAYERS * CONTEXT * 2 * INNER
    # Carrier replay streams old once. Each probe streams old through the fused
    # source path and full context through its Reuse counterfactual.
    parent_stream_scalars = LAYERS * (
        OLD_POSITIONS + probes * (OLD_POSITIONS + CONTEXT)
    ) * 2 * INNER
    return {
        "carriers": carriers,
        "probes": probes,
        "full_recompute_flops_per_user": full,
        "lazy_joint_map_flops_per_user": lazy,
        "carrier_and_probe_flops_per_user": carrier_and_probes,
        "carrier_mass_scale_flops_per_user": carrier_mass_scale,
        "amplitude_calibration_flops_per_user": amplitude_calibration,
        "total_flops_per_user": total,
        "over_full_fraction": total / full,
        "reduction_vs_full_fraction": 1.0 - total / full,
        "unique_parent_state_read_scalars": unique_parent_scalars,
        "unique_parent_state_read_bytes_fp32": unique_parent_scalars * scalar_bytes,
        "conservative_parent_state_stream_scalars": parent_stream_scalars,
        "conservative_parent_state_stream_bytes_fp32": parent_stream_scalars * scalar_bytes,
        "sidecar_write_scalars": sidecar_scalars,
        "sidecar_write_bytes_fp32": sidecar_scalars * scalar_bytes,
        "post_release_segment_scale_flops_per_user_request": 2 * LAYERS + LAYERS * INNER,
        "post_release_injection_adds_per_candidate": LAYERS * INNER,
        "materialized_version_translated_prefix_scalars": 0,
        "version_map_shape_per_layer": [2 * INNER, 2 * INNER],
        "version_map_construction_frequency": "once_per_release_edge_not_per_user",
    }


def exact_lazy_carrier_cost(carriers: int) -> dict[str, int | float]:
    """Cost rejected before execution: map every carrier dependency exactly."""
    if carriers not in (16, 32):
        raise ValueError("the frozen lightweight PRO cost axis is 16/32 carriers")
    tokens = carriers + 1
    # Carrier-to-carrier causal attention remains native. The old-prefix part
    # of every carrier query, plus the final probe query, uses the wide fused
    # joint-state reduction.
    standard_pairs = carriers * (carriers + 1) // 2 + CONTEXT + carriers + 1
    lazy = LAYERS * lazy_joint_map_flops_per_layer(probe_queries=carriers + 1)
    total = input_projection_flops(tokens) + LAYERS * (
        block_linear_flops(tokens) + attention_flops(standard_pairs)
    ) + lazy + LAYERS * carriers * INNER + LAYERS * INNER
    full = full_recompute_flops()
    return {
        "carriers": carriers,
        "total_flops_per_user": total,
        "over_full_fraction": total / full,
        "rejection_reason": "repeated_multihead_joint_state_scan_exceeds_20_percent_budget",
    }


def report() -> dict:
    return {
        "format": "evokv_pro_lazy_reader_theoretical_compute_v1",
        "scope": "release-time per-user PRO sidecar generation",
        "flop_convention": "one multiply plus one add equals two FLOPs",
        "architecture": {
            "layers": LAYERS,
            "hidden": HIDDEN,
            "inner": INNER,
            "heads": HEADS,
            "context": CONTEXT,
            "old_positions": OLD_POSITIONS,
            "repair_evidence": REPAIR_EVIDENCE,
        },
        "excluded": [
            "embedding lookup and pointwise activation/norm",
            "raw-history and state memory/network transport beyond reported logical bytes",
            "one-time per-release parameter-map pseudoinverse",
            "common post-release request serving",
        ],
        "budgets": [pro_cost(16), pro_cost(32)],
        "rejected_exact_lazy_carrier_replay": [
            exact_lazy_carrier_cost(16),
            exact_lazy_carrier_cost(32),
        ],
    }
