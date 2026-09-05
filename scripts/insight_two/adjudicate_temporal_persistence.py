#!/usr/bin/env python3
"""Adjudicate frozen-cutover S4 persistence with user-equal aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_temporal_persistence_v1.yaml"
)
RESULT_ROOT = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_temporal_persistence_v1"
)
PRIMARY = "coverage_scaled_frozen_cutover_S4"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def user_equal_edges(
    frame: pd.DataFrame, value: str = "probability_gap_recovery"
) -> pd.DataFrame:
    users = frame.groupby(["edge", "method", "uid"], as_index=False)[value].mean()
    return users.groupby(["edge", "method"], as_index=False)[value].mean()


def grouped_user_equal(
    frame: pd.DataFrame, group: str, value: str = "probability_gap_recovery"
) -> pd.DataFrame:
    users = frame.groupby(["edge", "method", group, "uid"], as_index=False)[value].mean()
    edges = users.groupby(["edge", "method", group], as_index=False)[value].mean()
    return edges.groupby(["method", group], as_index=False)[value].mean()


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in frame[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            values.append(f"{value:.{digits}f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return [header, divider, *rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "discovery"), default="canary")
    args = parser.parse_args()
    source = RESULT_ROOT / args.scope
    analysis = source / "analysis"
    if analysis.exists():
        raise FileExistsError(f"refusing to overwrite {analysis}")
    analysis.mkdir()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_parquet(source / "metrics.parquet")
    drift = pd.read_parquet(source / "drift.parquet")
    coverage = pd.read_parquet(source / "coverage.parquet")
    if metrics.empty or drift.empty or bool(metrics.isna().any().any()):
        raise RuntimeError("persistence evidence is empty or non-finite")

    rolling_fixed = metrics[
        (metrics.phase == "rolling")
        & (metrics.candidate_source == "fixed_heldout_panel")
    ].copy()
    rolling_real = metrics[
        (metrics.phase == "rolling")
        & (metrics.candidate_source == "real_exposed_items")
    ].copy()
    cutover = metrics[
        (metrics.phase == "cutover")
        & (metrics.candidate_source == "fixed_heldout_panel")
    ].copy()
    edge_values = user_equal_edges(rolling_fixed)
    edge_pivot = edge_values.pivot(
        index="edge", columns="method", values="probability_gap_recovery"
    )
    primary_edges = edge_pivot[PRIMARY]
    primary_mean = float(primary_edges.mean())
    positive_edges = int((primary_edges > 0).sum())
    persistence = contract["gates"]["persistence"]
    gate_passed = bool(
        primary_mean
        >= float(persistence["edge_equal_user_equal_probability_recovery_at_least"])
        and positive_edges
        >= int(persistence["positive_probability_recovery_edges_minimum"])
    )

    real_edges = user_equal_edges(rolling_real)
    cutover_edges = user_equal_edges(cutover)
    time_values = grouped_user_equal(rolling_fixed, "time_bucket")
    append_values = grouped_user_equal(rolling_fixed, "append_bucket")
    primary_time = time_values[time_values.method == PRIMARY].copy()
    primary_append = append_values[append_values.method == PRIMARY].copy()

    drift_users = drift.groupby(["edge", "uid"], as_index=False).agg(
        direction_cosine=("direction_cosine", "mean"),
        norm_ratio=("current_to_frozen_norm_ratio", "mean"),
        relative_l2=("relative_l2", "mean"),
    )
    drift_edges = drift_users.groupby("edge", as_index=False).mean(numeric_only=True)
    drift_edge_equal = {
        name: float(drift_edges[name].mean())
        for name in ("direction_cosine", "norm_ratio", "relative_l2")
    }

    denominators = rolling_fixed.reuse_probability_gap.to_numpy(dtype=np.float64)
    denominator_report = {
        "minimum": float(np.min(denominators)),
        "q01": float(np.quantile(denominators, 0.01)),
        "q05": float(np.quantile(denominators, 0.05)),
        "median": float(np.median(denominators)),
        "fraction_below_1e_8": float(np.mean(denominators < 1e-8)),
        "fraction_below_1e_6": float(np.mean(denominators < 1e-6)),
        "fraction_below_1e_4": float(np.mean(denominators < 1e-4)),
    }
    floor_sensitivity = {}
    primary_rows = rolling_fixed[rolling_fixed.method == PRIMARY].copy()
    for floor in (1e-8, 1e-6, 1e-4):
        column = f"floor_{floor:g}"
        primary_rows[column] = 1.0 - primary_rows.observed_probability_gap / np.maximum(
            primary_rows.reuse_probability_gap, floor
        )
        users = primary_rows.groupby(["edge", "uid"], as_index=False)[column].mean()
        floor_sensitivity[f"{floor:g}"] = float(
            users.groupby("edge")[column].mean().mean()
        )

    edge_table = edge_pivot.reset_index()[
        [
            "edge",
            "same_request_S4_oracle",
            "frozen_cutover_S4",
            PRIMARY,
        ]
    ].rename(columns={PRIMARY: "coverage_scaled_frozen"})
    edge_table["gate_primary_positive"] = edge_table.coverage_scaled_frozen > 0
    real_table = real_edges.pivot(
        index="edge", columns="method", values="probability_gap_recovery"
    ).reset_index()[["edge", "same_request_S4_oracle", PRIMARY]]
    real_table = real_table.rename(columns={PRIMARY: "coverage_scaled_frozen"})
    cutover_table = cutover_edges.pivot(
        index="edge", columns="method", values="probability_gap_recovery"
    ).reset_index()[["edge", "same_request_S4_oracle"]]

    result = {
        "status": "temporal_persistence_adjudicated",
        "scope": args.scope,
        "contract_sha256": summary["contract_sha256"],
        "labels_read": False,
        "oracle_boundary_persistence_only": True,
        "primary_method": PRIMARY,
        "primary_edge_equal_user_equal_probability_recovery": primary_mean,
        "primary_positive_edges": positive_edges,
        "persistence_gate_passed": gate_passed,
        "same_request_oracle_edge_equal_probability_recovery": float(
            edge_pivot.same_request_S4_oracle.mean()
        ),
        "unscaled_frozen_edge_equal_probability_recovery": float(
            edge_pivot.frozen_cutover_S4.mean()
        ),
        "drift_edge_equal_user_equal": drift_edge_equal,
        "active_users_by_edge": summary["active_users_by_edge"],
        "request_groups_by_edge": summary["request_groups_by_edge"],
        "denominator_report": denominator_report,
        "floor_sensitivity": floor_sensitivity,
        "interpretation": (
            "direction_persists_but_a_once_generated_amplitude_does_not_pass_the_E14_gate"
            if not gate_passed
            else "frozen_cutover_functional_state_passes_the_E14_persistence_gate"
        ),
        "executable_estimator_gate": "not_tested_by_this_oracle_diagnostic",
    }
    atomic_json(analysis / "summary.json", result)
    edge_table.to_csv(analysis / "edge_table.csv", index=False)
    primary_time.to_csv(analysis / "time_bucket_table.csv", index=False)
    primary_append.to_csv(analysis / "append_bucket_table.csv", index=False)
    drift_edges.to_csv(analysis / "drift_edge_table.csv", index=False)

    report = [
        f"# Medium S4 temporal persistence: {args.scope}",
        "",
        "The correction is a Current-Exact cutover oracle. It is frozen once and never refreshed; therefore this report tests the representation boundary, not an executable estimator or a 0--20% action.",
        "",
        "## Fixed held-out panel, user-equal within edge",
        "",
        *markdown_table(
            edge_table,
            [
                "edge",
                "same_request_S4_oracle",
                "frozen_cutover_S4",
                "coverage_scaled_frozen",
                "gate_primary_positive",
            ],
        ),
        "",
        f"- Primary edge-equal recovery: {primary_mean:.4f}; positive edges: {positive_edges}/5; Gate C: {'PASS' if gate_passed else 'FAIL'}.",
        f"- Same-request S4 oracle remains at {result['same_request_oracle_edge_equal_probability_recovery']:.4f} recovery.",
        f"- Unscaled frozen correction is {result['unscaled_frozen_edge_equal_probability_recovery']:.4f}; it is retained as a negative companion rather than clipped.",
        "",
        "## Correction drift",
        "",
        *markdown_table(
            drift_edges,
            ["edge", "direction_cosine", "norm_ratio", "relative_l2"],
        ),
        "",
        f"The edge-equal user-equal direction cosine is {drift_edge_equal['direction_cosine']:.4f}, while the Current/cutover norm ratio is {drift_edge_equal['norm_ratio']:.4f}. A stable direction alone is not a persistent offset: its amplitude evolves as Current events enter and Parent positions leave the cache.",
        "",
        "## Time buckets for the preregistered coverage-scaled method",
        "",
        *markdown_table(
            primary_time.rename(columns={"probability_gap_recovery": "recovery"}),
            ["time_bucket", "recovery"],
        ),
        "",
        "## Append buckets for the preregistered coverage-scaled method",
        "",
        *markdown_table(
            primary_append.rename(columns={"probability_gap_recovery": "recovery"}),
            ["append_bucket", "recovery"],
        ),
        "",
        "## Real exposed-item companion",
        "",
        *markdown_table(
            real_table,
            ["edge", "same_request_S4_oracle", "coverage_scaled_frozen"],
        ),
        "",
        "## Adjudication",
        "",
        "- The S4 boundary continues to be causally sufficient when its correction is re-observed at the current request.",
        "- The cutover direction remains highly aligned, but neither an unscaled offset nor the preregistered linear remaining-coverage decay is sufficient over the complete E14 timeline in this scope." if not gate_passed else "- The frozen cutover correction passes the preregistered persistence gate.",
        "- No request was filtered for a small Reuse gap. Denominator quantiles and fixed-floor sensitivity are sealed in `summary.json`.",
        "- This oracle result cannot authorize a migration action, estimator, refresh policy or confirmation read.",
        "",
    ]
    (analysis / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
