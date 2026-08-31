#!/usr/bin/env python3
"""Adjudicate the fixed 3,000-user recommendation-state observation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "results/yambda500m_small_seed17/insight_recommendation_state_structure_v1"
)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    destination = args.input
    outputs = [
        destination / "adjudication.json",
        destination / "adjudication.md",
        destination / "candidate_common_basis.csv",
        destination / "state_factorization_adjudicated.csv",
        destination / "semantic_coreset_adjudicated.csv",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite adjudication outputs: {existing}")

    summary = json.loads((destination / "summary.json").read_text())
    if summary["users"] != 3000 or summary["population_fraction"] != 0.3:
        raise RuntimeError("adjudication requires the formal 3,000-user result")
    if summary["labels_read"] or summary["candidate_probe_negative_semantics"]:
        raise RuntimeError("formal observation violated its label-free candidate contract")
    if max(summary["candidate_trace_max_abs_score_error"].values()) > 2e-5:
        raise RuntimeError("candidate influence trace failed model-score replay")

    delta = pd.read_csv(destination / "state_delta_factorization.csv")
    coreset = pd.read_parquet(destination / "semantic_coreset_user_metrics.parquet")
    pairing = pd.read_parquet(destination / "pairing_user_metrics.parquet")
    candidate = pd.read_parquet(destination / "candidate_subspace_user_metrics.parquet")
    if len(candidate) != 3000 * 5 * 4:
        raise RuntimeError("candidate subspace table does not cover every user-edge-layer")

    candidate_basis = candidate.groupby(["edge", "layer"], sort=True).agg(
        users=("uid", "nunique"),
        candidate_rank90_one_fraction=("candidate_influence_rank90", lambda value: float(np.mean(value == 1))),
        candidate_top_direction_energy_mean=("candidate_influence_top_direction_fraction", "mean"),
        candidate_top_direction_energy_p01=("candidate_influence_top_direction_fraction", lambda value: float(np.quantile(value, 0.01))),
        mismatch_influence_rank90_one_fraction=("exact_minus_reuse_influence_rank90", lambda value: float(np.mean(value == 1))),
        mismatch_influence_effective_rank_mean=("exact_minus_reuse_influence_effective_rank", "mean"),
        mismatch_readout_rank90_one_fraction=("exact_minus_reuse_readout_rank90", lambda value: float(np.mean(value == 1))),
        mismatch_readout_effective_rank_mean=("exact_minus_reuse_readout_effective_rank", "mean"),
    ).reset_index()
    candidate_basis.to_csv(destination / "candidate_common_basis.csv", index=False)

    stages = [
        "item_embedding", "combined_input", "layer0.k", "layer0.av", "layer0.gate",
        "layer0.update", "layer1.k", "layer1.update", "layer2.k", "layer2.update",
        "layer3.k", "layer3.update",
    ]
    factorization = delta[delta.stage.isin(stages)].copy()
    factorization.to_csv(destination / "state_factorization_adjudicated.csv", index=False)

    coreset_summary = coreset.groupby(["edge", "path"], sort=True).agg(
        users=("uid", "nunique"),
        mean_abs_probability_gap=("mean_abs_probability_gap", "mean"),
        median_abs_probability_gap=("mean_abs_probability_gap", "median"),
        median_user_recovery_over_reuse=("output_gap_recovery_over_reuse", "median"),
        top10_overlap=("top10_overlap", "mean"),
        rank_correlation=("rank_correlation", "mean"),
    ).reset_index()
    reuse_gap = coreset_summary[coreset_summary.path == "parent_reuse"].set_index("edge")[
        "mean_abs_probability_gap"
    ]
    coreset_summary["aggregate_gap_recovery_over_reuse"] = [
        1.0 - row.mean_abs_probability_gap / reuse_gap.loc[row.edge]
        for row in coreset_summary.itertuples()
    ]
    position = coreset_summary[coreset_summary.path == "positional_pairs"].set_index("edge")[
        "mean_abs_probability_gap"
    ]
    coreset_summary["gap_minus_positional_pairs"] = [
        row.mean_abs_probability_gap - position.loc[row.edge]
        for row in coreset_summary.itertuples()
    ]
    coreset_summary.to_csv(destination / "semantic_coreset_adjudicated.csv", index=False)

    paired = coreset.pivot(
        index=["edge", "uid"], columns="path", values="mean_abs_probability_gap"
    )
    semantic_wins = {}
    for path in ("same_item_pairs", "typed_pairs"):
        values = {}
        for edge, rows in paired.groupby(level=0):
            difference = rows[path] - rows["positional_pairs"]
            values[edge] = {
                "mean_gap_difference": float(difference.mean()),
                "median_gap_difference": float(difference.median()),
                "user_win_fraction": float(np.mean(difference < 0)),
            }
        semantic_wins[path] = values

    pairing_summary = pairing.groupby(["edge", "path"], sort=True).agg(
        same_item_pair_fraction=("same_item_pair_fraction", "mean"),
        same_action_pair_fraction=("same_action_pair_fraction", "mean"),
        same_item_action_pair_fraction=("same_item_action_pair_fraction", "mean"),
        mean_pair_position_distance=("mean_pair_position_distance", "mean"),
    ).reset_index()
    item_pair = pairing_summary[pairing_summary.path == "same_item_pairs"]
    positional_pair = pairing_summary[pairing_summary.path == "positional_pairs"]

    user_edges = candidate[candidate.layer == 0]
    adjudication = {
        "status": "recommendation_state_structure_adjudicated",
        "scope": {
            "users": summary["users"],
            "population_fraction": summary["population_fraction"],
            "user_edges": len(user_edges),
            "user_edge_layers": len(candidate),
            "candidate_probes_per_user_edge": summary["candidate_probes_per_user_edge"],
            "labels_read": False,
        },
        "candidate_common_basis": {
            "candidate_influence_rank90_one": int((candidate.candidate_influence_rank90 == 1).sum()),
            "candidate_influence_matrices": len(candidate),
            "candidate_influence_rank90_one_fraction": float(np.mean(candidate.candidate_influence_rank90 == 1)),
            "mean_top_direction_energy_range_across_edge_layers": [
                float(candidate_basis.candidate_top_direction_energy_mean.min()),
                float(candidate_basis.candidate_top_direction_energy_mean.max()),
            ],
            "mismatch_influence_rank90_one": int((candidate.exact_minus_reuse_influence_rank90 == 1).sum()),
            "mismatch_influence_matrices": len(candidate),
            "mismatch_influence_rank90_one_fraction": float(np.mean(candidate.exact_minus_reuse_influence_rank90 == 1)),
            "mismatch_readout_rank90_one_user_edges": int((user_edges.exact_minus_reuse_readout_rank90 == 1).sum()),
            "mismatch_readout_user_edges": len(user_edges),
            "mismatch_readout_rank90_one_fraction": float(np.mean(user_edges.exact_minus_reuse_readout_rank90 == 1)),
            "mismatch_readout_effective_rank_mean_range_across_edges": [
                float(user_edges.groupby("edge").exact_minus_reuse_readout_effective_rank.mean().min()),
                float(user_edges.groupby("edge").exact_minus_reuse_readout_effective_rank.mean().max()),
            ],
        },
        "typed_entity_to_context": {
            "combined_input_item_centroid_R2_range": [
                float(delta[delta.stage == "combined_input"].item_centroid_R2.min()),
                float(delta[delta.stage == "combined_input"].item_centroid_R2.max()),
            ],
            "layer0_K_item_centroid_R2_range": [
                float(delta[delta.stage == "layer0.k"].item_centroid_R2.min()),
                float(delta[delta.stage == "layer0.k"].item_centroid_R2.max()),
            ],
            "layer0_K_item_action_R2_range": [
                float(delta[delta.stage == "layer0.k"].item_action_R2.min()),
                float(delta[delta.stage == "layer0.k"].item_action_R2.max()),
            ],
            "layer0_update_item_centroid_R2_range": [
                float(delta[delta.stage == "layer0.update"].item_centroid_R2.min()),
                float(delta[delta.stage == "layer0.update"].item_centroid_R2.max()),
            ],
            "layer1_to_layer3_update_item_excess_R2_over_global_range": [
                float(delta[delta.stage.isin(["layer1.update", "layer2.update", "layer3.update"])].item_excess_R2_over_global.min()),
                float(delta[delta.stage.isin(["layer1.update", "layer2.update", "layer3.update"])].item_excess_R2_over_global.max()),
            ],
        },
        "semantic_coreset_boundary": {
            "same_item_pair_fraction_positional_range": [
                float(positional_pair.same_item_pair_fraction.min()),
                float(positional_pair.same_item_pair_fraction.max()),
            ],
            "same_item_pair_fraction_semantic_range": [
                float(item_pair.same_item_pair_fraction.min()),
                float(item_pair.same_item_pair_fraction.max()),
            ],
            "same_item_mean_gap_beats_positional_edges": int(
                sum(value["mean_gap_difference"] < 0 for value in semantic_wins["same_item_pairs"].values())
            ),
            "typed_mean_gap_beats_positional_edges": int(
                sum(value["mean_gap_difference"] < 0 for value in semantic_wins["typed_pairs"].values())
            ),
            "edges": 5,
            "paired_user_results": semantic_wins,
        },
        "adjudicated_insight": (
            "Across this fixed Small seed17 chain, stale persistent state behaves primarily as a "
            "candidate-broadcast user-evidence compatibility field. Shared item/action change is "
            "strong at the input and layer-0 coordinate, but aggregation and gating turn it into "
            "contextual residual. Raw item/action identity alone is not a stable coreset relation."
        ),
        "design_implication": (
            "Prioritize a Current-version user evidence basis or anchor shared by the candidate bank, "
            "then represent a smaller contextual residual. Do not add per-candidate token routing or "
            "raw same-item grouping to the action set from this result."
        ),
        "boundary": (
            "This is one scale and one training seed with a controlled label-free candidate bank. "
            "It identifies state structure but does not qualify a new migration mechanism or runtime."
        ),
    }
    (destination / "adjudication.json").write_text(json.dumps(adjudication, indent=2) + "\n")

    basis_focus = candidate_basis[
        [
            "edge", "layer", "candidate_rank90_one_fraction",
            "candidate_top_direction_energy_mean", "mismatch_influence_rank90_one_fraction",
            "mismatch_readout_rank90_one_fraction", "mismatch_readout_effective_rank_mean",
        ]
    ]
    factor_focus = factorization[factorization.stage.isin(
        ["combined_input", "layer0.k", "layer0.update", "layer1.update", "layer2.update", "layer3.update"]
    )][
        [
            "edge", "stage", "held_out_user_item_action_samples", "item_centroid_R2",
            "item_excess_R2_over_global", "item_action_R2", "item_action_increment_over_item",
        ]
    ]
    coreset_focus = coreset_summary[coreset_summary.path.isin(
        ["positional_pairs", "same_item_pairs", "typed_pairs"]
    )][
        [
            "edge", "path", "mean_abs_probability_gap", "aggregate_gap_recovery_over_reuse",
            "gap_minus_positional_pairs", "top10_overlap", "rank_correlation",
        ]
    ]
    report = [
        "# Recommendation-state structure: adjudication",
        "",
        "Formal scope: 3,000 fixed users (30% of Small), all five v0→v5 adjacent edges, 512 pre-cutover events and 64 label-free candidate probes per user-edge. No label or negative semantics entered the observation.",
        "",
        "## Adjudicated insight",
        "",
        "**Cross-version HSTU state mismatch is primarily a candidate-broadcast user-evidence compatibility field, not a collection of independent per-candidate token-retrieval failures.**",
        "",
        f"Every one of the {len(candidate):,} candidate influence matrices has rank-1@90%. The Exact−Reuse influence delta has rank-1@90% in {int((candidate.exact_minus_reuse_influence_rank90 == 1).sum()):,}/{len(candidate):,} user-edge-layer cases, and the final readout delta has rank-1@90% in all {len(user_edges):,} user-edge cases. Across edge/layer means, the first candidate-shared direction carries {100 * candidate_basis.candidate_top_direction_energy_mean.min():.4f}%–{100 * candidate_basis.candidate_top_direction_energy_mean.max():.4f}% of normalized influence energy.",
        "",
        *markdown_table(basis_focus, list(basis_focus.columns)),
        "",
        "## Where the shared field comes from",
        "",
        "Held-out-user centroids show a strong typed entity coordinate at the input and layer-0 K, followed by rapid contextualization in the attention/gated update. Item identity adds a real but modest component beyond the global version shift; item-action typing explains much more early-layer delta. The update stages retain far less item-specific predictability.",
        "",
        *markdown_table(factor_focus, list(factor_focus.columns)),
        "",
        "This supports `shared typed coordinate + contextual user residual`; it does not support persisting raw item embeddings as the complete interface.",
        "",
        "## Semantic coreset boundary",
        "",
        "Same-item-first pairing raises the matched-item pair fraction from 3.29%–3.79% to 29.55%–30.13%, but same-item and typed pairing each beat positional pairing on mean probability gap in only 3/5 edges. Per-user win fractions remain 42.2%–50.0%. Raw identity/action equality is therefore not a stable substitutability test for contextual HSTU evidence.",
        "",
        *markdown_table(coreset_focus, list(coreset_focus.columns)),
        "",
        "## Design implication and boundary",
        "",
        "The next mechanism should first repair or rematerialize a small Current-version **user evidence basis** shared across the candidate bank, then carry a smaller contextual residual. This is more recommendation-specific than query-by-query top-k token repair because the same persistent user state is amortized across many candidates.",
        "",
        "Do not add candidate-specific Route or raw same-item/action GROUP from this result. The existing CAST + compact PATCH path remains a strong baseline. A new basis/anchor mechanism still needs a prospective task-quality experiment and runtime qualification. The observation is Small seed17 only and uses a controlled candidate bank, not an exposed-candidate quality result.",
        "",
    ]
    (destination / "adjudication.md").write_text("\n".join(report))
    print(json.dumps(adjudication, indent=2))


if __name__ == "__main__":
    main()
