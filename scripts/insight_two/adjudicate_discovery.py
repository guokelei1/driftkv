#!/usr/bin/env python3
"""Adjudicate the 512-user functional-boundary representation discovery.

The discovery uses Current-Exact anchor traces to construct target corrections.
Consequently this script can pass only the representation gate; it cannot
admit an executable estimator or freeze Design 1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insight_two.common import EDGES, RESULT_ROOT, sha256_file


BOOTSTRAP_SEED = 20_260_902
BOOTSTRAP_REPLICATES = 2_000
STAGE_ORDER = {
    "av_aggregation": 4,
    "u_gated_update": 5,
    "layer_hidden": 6,
    "final_readout": 7,
}


def aggregate_recovery(frame: pd.DataFrame, prefix: str) -> float:
    denominator = float(frame[f"reuse_{prefix}_gap"].sum())
    if denominator <= 0.0:
        raise RuntimeError(f"non-positive aggregate {prefix} reuse gap")
    return 1.0 - float(frame[f"observed_{prefix}_gap"].sum()) / denominator


def clustered_interval(frame: pd.DataFrame) -> tuple[float, float]:
    """Bootstrap users while preserving all five per-user recovery rows."""
    table = frame.pivot(
        index="uid",
        columns="edge",
        values="probability_gap_recovery",
    )
    table = table.reindex(columns=EDGES)
    if table.isna().any().any() or len(table) != 512:
        raise RuntimeError("bootstrap input is not a complete 512-user/five-edge panel")
    recovery = table.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 100):
        stop = min(start + 100, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(table), size=(stop - start, len(table)))
        estimates[start:stop] = recovery[indices].mean(axis=(1, 2))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    root = RESULT_ROOT / "discovery_functional_boundary"
    run_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not run_summary.get("passed") or run_summary.get("users") != 512:
        raise RuntimeError("the 512-user discovery did not pass")
    output = root / "analysis_v2"
    partial = root / "analysis_v2.partial"
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir()

    score_paths = [root / f"rank{rank}/score_records.parquet" for rank in range(4)]
    fit_paths = [root / f"rank{rank}/fit_records.parquet" for rank in range(4)]
    scores = pd.concat([pd.read_parquet(path) for path in score_paths], ignore_index=True)
    fits = pd.concat([pd.read_parquet(path) for path in fit_paths], ignore_index=True)
    expected_rows = 512 * len(EDGES) * len(STAGE_ORDER) * 5
    if len(scores) != expected_rows or scores.uid.nunique() != 512:
        raise RuntimeError("discovery score panel is incomplete")

    per_edge_rows: list[dict[str, Any]] = []
    for keys, frame in scores.groupby(["stage", "presentation", "rank", "edge"]):
        stage, presentation, rank, edge = keys
        per_edge_rows.append(
            {
                "stage": stage,
                "presentation": presentation,
                "rank": int(rank),
                "edge": edge,
                "users": int(frame.uid.nunique()),
                "probability_recovery": float(frame.probability_gap_recovery.mean()),
                "logit_recovery": float(frame.logit_gap_recovery.mean()),
                "harm_weighted_probability_recovery": aggregate_recovery(
                    frame, "probability"
                ),
                "harm_weighted_logit_recovery": aggregate_recovery(frame, "logit"),
                "median_user_probability_recovery": float(
                    frame.probability_gap_recovery.median()
                ),
                "p10_user_probability_recovery": float(
                    frame.probability_gap_recovery.quantile(0.10)
                ),
                "users_improved_fraction": float(
                    (frame.observed_probability_gap < frame.reuse_probability_gap).mean()
                ),
                "users_at_80_fraction": float(
                    (frame.probability_gap_recovery >= 0.80).mean()
                ),
                "top1_agreement": float(frame.top1_agreement.mean()),
                "top10_overlap": float(frame.top10_overlap.mean()),
                "rank_correlation": float(frame.rank_correlation.mean()),
                "storage_values_fp32_per_user": int(
                    frame.storage_values_fp32_per_user.max()
                ),
            }
        )
    per_edge = pd.DataFrame(per_edge_rows).sort_values(
        ["stage", "rank", "edge"]
    )

    frontier_rows: list[dict[str, Any]] = []
    for keys, frame in scores.groupby(["stage", "presentation", "rank"]):
        stage, presentation, rank = keys
        edge_rows = per_edge[
            (per_edge.stage == stage) & (per_edge["rank"] == rank)
        ]
        lower, upper = clustered_interval(frame)
        probability = edge_rows.probability_recovery
        frontier_rows.append(
            {
                "stage": stage,
                "stage_order": STAGE_ORDER[stage],
                "presentation": presentation,
                "rank": int(rank),
                "edge_equal_probability_recovery": float(probability.mean()),
                "bootstrap_95_lower": lower,
                "bootstrap_95_upper": upper,
                "minimum_edge_probability_recovery": float(probability.min()),
                "edge_equal_logit_recovery": float(edge_rows.logit_recovery.mean()),
                "edge_equal_harm_weighted_probability_recovery": float(
                    edge_rows.harm_weighted_probability_recovery.mean()
                ),
                "edges_at_80": int((probability >= 0.80).sum()),
                "edges_at_90": int((probability >= 0.90).sum()),
                "median_user_probability_recovery": float(
                    frame.probability_gap_recovery.median()
                ),
                "p10_user_probability_recovery": float(
                    frame.probability_gap_recovery.quantile(0.10)
                ),
                "users_improved_fraction": float(
                    (frame.observed_probability_gap < frame.reuse_probability_gap).mean()
                ),
                "users_at_80_fraction": float(
                    (frame.probability_gap_recovery >= 0.80).mean()
                ),
                "storage_values_fp32_per_user": int(
                    frame.storage_values_fp32_per_user.max()
                ),
            }
        )
    frontier = pd.DataFrame(frontier_rows)
    frontier["representation_gate_80"] = (
        (frontier.edge_equal_probability_recovery >= 0.80)
        & (
            (frontier.edges_at_80 >= 4)
            | (
                (frontier.edges_at_90 >= 3)
                & (frontier.edge_equal_probability_recovery >= 0.80)
            )
        )
    )
    frontier["strong_representation_gate_90"] = (
        (frontier.edge_equal_probability_recovery >= 0.90)
        & (frontier.edges_at_90 == len(EDGES))
    )
    frontier = frontier.sort_values(
        ["stage_order", "rank", "storage_values_fp32_per_user"]
    )

    fit_summary = (
        fits.groupby(["stage", "presentation", "rank", "layer"], as_index=False)
        .agg(
            users=("uid", "nunique"),
            mean_target_rank90=("target_rank90", "mean"),
            p90_target_rank90=("target_rank90", lambda values: values.quantile(0.90)),
            mean_target_rank95=("target_rank95", "mean"),
            retained_centered_energy=("rank_retained_centered_energy", "mean"),
            anchor_fit_relative_l2=("anchor_fit_relative_l2", "mean"),
        )
        .sort_values(["stage", "rank", "layer"])
    )

    strong = frontier[frontier.strong_representation_gate_90].copy()
    earliest = strong.sort_values(
        ["stage_order", "rank", "storage_values_fp32_per_user"]
    ).iloc[0]
    compact = strong.sort_values(
        ["storage_values_fp32_per_user", "stage_order", "rank"]
    ).iloc[0]
    rank_one = frontier[frontier["rank"] == 1].copy()
    rank_one_best = rank_one.sort_values(
        ["edge_equal_probability_recovery", "storage_values_fp32_per_user"],
        ascending=[False, True],
    ).iloc[0]

    per_edge.to_csv(partial / "per_edge.csv", index=False)
    frontier.to_csv(partial / "frontier.csv", index=False)
    fit_summary.to_csv(partial / "fit_structure.csv", index=False)
    lines = [
        "# Medium Insight 2 functional-boundary discovery",
        "",
        "This is a 512-user, five-edge, label-free representation test. Target corrections come from Current-Exact anchor traces; no row below is an executable estimator result.",
        "",
        "The preregistered primary recovery is the mean of each user's unclipped `1 - observed gap / Reuse gap` within an edge, followed by an equal-weight mean across edges. Gap-weighted recovery is retained in CSV as a sensitivity analysis. The interval is a 2,000-replicate user-cluster bootstrap.",
        "",
        "| stage | rank | recovery | 95% CI | min edge | edges >=90% | median user | users >=80% | FP32 values/user |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frontier.itertuples(index=False):
        lines.append(
            f"| {row.presentation} | {row.rank} | "
            f"{row.edge_equal_probability_recovery:.4f} | "
            f"[{row.bootstrap_95_lower:.4f}, {row.bootstrap_95_upper:.4f}] | "
            f"{row.minimum_edge_probability_recovery:.4f} | "
            f"{row.edges_at_90} | {row.median_user_probability_recovery:.4f} | "
            f"{row.users_at_80_fraction:.4f} | "
            f"{row.storage_values_fp32_per_user} |"
        )
    lines.extend(
        [
            "",
            "## Adjudication",
            "",
            f"- Earliest strong representation boundary: {earliest.presentation}, rank {int(earliest['rank'])}, recovery {earliest.edge_equal_probability_recovery:.4f}, minimum edge {earliest.minimum_edge_probability_recovery:.4f}.",
            f"- Most compact strong representation boundary: {compact.presentation}, rank {int(compact['rank'])}, recovery {compact.edge_equal_probability_recovery:.4f}, {int(compact.storage_values_fp32_per_user)} FP32 values/user.",
            f"- Best rank-1 response model: {rank_one_best.presentation}, recovery {rank_one_best.edge_equal_probability_recovery:.4f}, minimum edge {rank_one_best.minimum_edge_probability_recovery:.4f}.",
            "- S5 transformed update and S6 post-block residual are algebraically equivalent under this same-hidden additive intervention; they are one observation, not independent confirmation.",
            "- The representation gate passes. Estimator, 0--20% cost, persistence, and task-quality gates remain open; therefore neither Insight 2 nor Design 1 is frozen.",
            "",
        ]
    )
    (partial / "report.md").write_text("\n".join(lines), encoding="utf-8")
    result = {
        "status": "discovery_representation_adjudicated",
        "labels_read": False,
        "users": 512,
        "edges": list(EDGES),
        "bootstrap": {
            "unit": "user",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "raw_artifacts": {
            str(path.relative_to(root)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (*score_paths, *fit_paths)
        },
        "earliest_strong_boundary": {
            "stage": str(earliest.presentation),
            "rank": int(earliest["rank"]),
            "edge_equal_probability_recovery": float(
                earliest.edge_equal_probability_recovery
            ),
            "minimum_edge_probability_recovery": float(
                earliest.minimum_edge_probability_recovery
            ),
        },
        "most_compact_strong_boundary": {
            "stage": str(compact.presentation),
            "rank": int(compact["rank"]),
            "edge_equal_probability_recovery": float(
                compact.edge_equal_probability_recovery
            ),
            "storage_values_fp32_per_user": int(compact.storage_values_fp32_per_user),
        },
        "representation_gate": "pass",
        "estimator_gate": "not_run",
        "persistence_gate": "not_run",
        "design_status": "open",
    }
    atomic_json(partial / "summary.json", result)
    os.replace(partial, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
