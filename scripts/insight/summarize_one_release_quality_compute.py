#!/usr/bin/env python3
"""Combine sealed one-release AUC with a reproducible theoretical FLOP count.

The count covers release-time state construction only.  One multiply and one
add count as two FLOPs.  It includes the dominant matrix multiplications and
SCALE, but excludes embedding lookup, pointwise activation/norm, memory I/O,
version-pair CAST-map construction, future append, and request serving.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    ROOT
    / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
    / "d14_one_release_refinement_auc_v1"
)

# Frozen Small architecture and fixed one-release plan.
LAYERS = 4
HIDDEN = 128
INNER = 128
TEMPORAL_FREQS = 16
CONTEXT = 512
REPAIR_EVIDENCE = 128
CARRIERS = 64
CAST_POSITIONS = CONTEXT - REPAIR_EVIDENCE


def input_projection_flops(tokens: int) -> int:
    temporal_projection = 2 * tokens * (2 * TEMPORAL_FREQS) * HIDDEN
    input_projection = 2 * tokens * HIDDEN * HIDDEN
    return temporal_projection + input_projection


def block_linear_flops(tokens: int) -> int:
    # Q/K/V, attention output projection, and HSTU gate projection.
    return 2 * tokens * (
        3 * HIDDEN * INNER + INNER * HIDDEN + HIDDEN * HIDDEN
    )


def attention_flops(pairs: int) -> int:
    # QK and attention-value products across all heads.
    return 4 * pairs * INNER


def cost(*, dense_attention: bool) -> dict[str, int | float]:
    exact_pairs = CONTEXT * CONTEXT if dense_attention else CONTEXT * (CONTEXT + 1) // 2
    patch_pairs = (
        CARRIERS * (CAST_POSITIONS + CARRIERS)
        if dense_attention
        else CAST_POSITIONS * CARRIERS + CARRIERS * (CARRIERS + 1) // 2
    )

    exact = input_projection_flops(CONTEXT) + LAYERS * (
        block_linear_flops(CONTEXT) + attention_flops(exact_pairs)
    )
    cast = LAYERS * 2 * CAST_POSITIONS * (2 * INNER) * (2 * INNER)
    compact_patch = input_projection_flops(CARRIERS) + LAYERS * (
        block_linear_flops(CARRIERS) + attention_flops(patch_pairs)
    )
    scale = LAYERS * CARRIERS * INNER
    ours = cast + compact_patch + scale
    return {
        "recompute_flops_per_full_user": exact,
        "reuse_state_conversion_flops_per_full_user": 0,
        "our_cast_flops_per_full_user": cast,
        "our_group_patch_flops_per_full_user": compact_patch,
        "our_scale_flops_per_full_user": scale,
        "our_total_flops_per_full_user": ours,
        "our_over_recompute_fraction": ours / exact,
        "our_reduction_vs_recompute_fraction": 1.0 - ours / exact,
    }


def render(auc_rows: list[dict], report: dict) -> str:
    causal = report["conservative_causal_flops"]
    compute = 100.0 * causal["our_over_recompute_fraction"]
    lines = [
        "# One-release quality and theoretical compute",
        "",
        "The headline compute uses a conservative ideal causal-attention FLOP count. "
        "It assumes a 512-position full state and includes per-user CAST plus compact "
        "Current PATCH and SCALE. Reuse state conversion is 0 FLOPs; state I/O and "
        "common serving work are outside this boundary.",
        "",
        "| Edge | Recompute AUC | Reuse AUC | Our AUC | Our - Reuse (pp) | Our gain retained | Reuse harm recovered | Recompute compute | Reuse compute | Our compute |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in auc_rows:
        lines.append(
            f"| {row['edge'].replace('_to_', ' -> ')} | "
            f"{row['recompute_rolling_AUC']:.6f} | {row['reuse_rolling_AUC']:.6f} | "
            f"{row['our_rolling_AUC']:.6f} | {row['our_minus_reuse_ROC_AUC_pp']:+.6f} | "
            f"{row['our_gain_retained_percent']:+.1f}% | "
            f"{100.0 * row['reuse_harm_recovered_fraction']:+.1f}% | "
            f"100.0% | 0.0%* | {compute:.1f}% |"
        )
    dense = report["current_dense_graph_flops"]
    lines.extend([
        "",
        f"Conservative causal FLOPs per full user: Recompute "
        f"{causal['recompute_flops_per_full_user'] / 1e9:.3f} GFLOPs; Our "
        f"{causal['our_total_flops_per_full_user'] / 1e9:.3f} GFLOPs "
        f"({compute:.1f}% of Recompute, "
        f"{100.0 * causal['our_reduction_vs_recompute_fraction']:.1f}% lower).",
        "",
        f"The current dense PyTorch attention graph gives Recompute "
        f"{dense['recompute_flops_per_full_user'] / 1e9:.3f} GFLOPs and Our "
        f"{dense['our_total_flops_per_full_user'] / 1e9:.3f} GFLOPs "
        f"({100.0 * dense['our_over_recompute_fraction']:.1f}% of Recompute). "
        "This is a secondary implementation-graph count, not a runtime claim.",
        "",
        "`*` Reuse has zero release-time neural recomputation in this table, not zero "
        "state read, network, storage, or serving cost. The v3 -> v4 retained ratio has "
        "a near-zero Full-only gain denominator; v4 -> v5 is a genuine negative edge.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    auc_rows = json.loads((RESULT / "auc_summary.json").read_text())
    if len(auc_rows) != 5:
        raise RuntimeError("the sealed one-release summary must contain all five edges")
    report = {
        "format": "evokv_one_release_theoretical_compute_v1",
        "scope": "release-time full-512 state construction; one-hop only",
        "architecture": {
            "layers": LAYERS,
            "hidden": HIDDEN,
            "inner": INNER,
            "context": CONTEXT,
        },
        "plan": {
            "cast_positions": CAST_POSITIONS,
            "repair_evidence": REPAIR_EVIDENCE,
            "carriers": CARRIERS,
            "represented_mass_per_carrier": 2,
        },
        "flop_convention": "one multiply plus one add equals two FLOPs; dominant matmuls plus SCALE",
        "excluded": [
            "embedding lookup and pointwise activation/norm",
            "GROUP gather and all memory/storage/network I/O",
            "one-time version-pair CAST-map construction",
            "future append and request serving shared by all paths",
        ],
        "conservative_causal_flops": cost(dense_attention=False),
        "current_dense_graph_flops": cost(dense_attention=True),
    }
    (RESULT / "theoretical_compute.json").write_text(json.dumps(report, indent=2) + "\n")
    (RESULT / "quality_compute_summary.md").write_text(render(auc_rows, report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
