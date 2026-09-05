#!/usr/bin/env python3
"""Adjudicate oracle temporal-coordinate ceilings on the cutover S4 basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_temporal_coefficient_v1.yaml"
)
RESULT_ROOT = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_temporal_coefficient_v1"
)
GLOBAL = "oracle_global_coefficient_times_cutover_direction"
LAYERWISE = "oracle_layerwise_coefficients_times_cutover_directions"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def user_equal_edges(frame: pd.DataFrame) -> pd.DataFrame:
    users = frame.groupby(["edge", "method", "uid"], as_index=False).agg(
        probability_recovery=("probability_gap_recovery", "mean"),
        logit_recovery=("logit_gap_recovery", "mean"),
    )
    return users.groupby(["edge", "method"], as_index=False).mean(numeric_only=True)


def coordinate_gate(
    edge_table: pd.DataFrame, method: str, rule: dict
) -> dict[str, float | int | bool]:
    values = edge_table[edge_table.method == method].probability_recovery
    mean = float(values.mean())
    positive = int((values > 0).sum())
    return {
        "edge_equal_probability_recovery": mean,
        "minimum_edge_probability_recovery": float(values.min()),
        "positive_edges": positive,
        "passed": bool(
            mean >= float(rule["edge_equal_probability_recovery_at_least"])
            and positive >= int(rule["positive_edges_minimum"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "discovery"), default="canary")
    args = parser.parse_args()
    source = RESULT_ROOT / args.scope
    output = source / "analysis"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    run = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_parquet(source / "metrics.parquet")
    coefficients = pd.read_parquet(source / "coefficients.parquet")
    fixed = metrics[metrics.candidate_source == "fixed_heldout_panel"].copy()
    real = metrics[metrics.candidate_source == "real_exposed_items"].copy()
    edge_table = user_equal_edges(fixed)
    real_edges = user_equal_edges(real)
    gates = contract["gates"]
    global_gate = coordinate_gate(edge_table, GLOBAL, gates["global_coordinate"])
    layer_gate = coordinate_gate(edge_table, LAYERWISE, gates["layerwise_coordinate"])
    if global_gate["passed"]:
        interpretation = "one_dimensional_temporal_coordinate_supported"
    elif layer_gate["passed"]:
        interpretation = "layerwise_low_dimensional_temporal_coordinates_supported"
    else:
        interpretation = "fixed_cutover_response_basis_is_not_causally_sufficient_over_time"

    projection_users = coefficients.groupby(["edge", "uid"], as_index=False).agg(
        global_projection_relative_l2=("global_projection_relative_l2", "mean"),
        layerwise_projection_relative_l2=("layerwise_projection_relative_l2", "mean"),
        global_coefficient=("global_coefficient", "mean"),
    )
    projection_edges = projection_users.groupby("edge", as_index=False).mean(numeric_only=True)
    time_users = fixed.groupby(
        ["method", "time_bucket", "edge", "uid"], as_index=False
    ).probability_gap_recovery.mean()
    time_table = (
        time_users.groupby(["method", "time_bucket", "edge"], as_index=False)
        .probability_gap_recovery.mean()
        .groupby(["method", "time_bucket"], as_index=False)
        .probability_gap_recovery.mean()
    )
    coefficient_summary = coefficients.groupby("edge", as_index=False).agg(
        global_coefficient_mean=("global_coefficient", "mean"),
        global_coefficient_median=("global_coefficient", "median"),
        global_coefficient_std=("global_coefficient", "std"),
    )
    result = {
        "status": "temporal_coordinate_adjudicated",
        "scope": args.scope,
        "contract_sha256": run["contract_sha256"],
        "labels_read": False,
        "oracle_coefficients_only": True,
        "global_coordinate": global_gate,
        "layerwise_coordinate": layer_gate,
        "same_request_oracle_edge_equal_probability_recovery": float(
            edge_table[
                edge_table.method == "same_request_S4_oracle"
            ].probability_recovery.mean()
        ),
        "frozen_cutover_edge_equal_probability_recovery": float(
            edge_table[
                edge_table.method == "frozen_cutover_S4"
            ].probability_recovery.mean()
        ),
        "edge_equal_projection_relative_l2": {
            "global": float(projection_edges.global_projection_relative_l2.mean()),
            "layerwise": float(projection_edges.layerwise_projection_relative_l2.mean()),
        },
        "interpretation": interpretation,
        "design1_gate": "not_tested_coefficients_are_oracle",
    }
    atomic_json(output / "summary.json", result)
    edge_table.to_csv(output / "edge_table.csv", index=False)
    real_edges.to_csv(output / "real_edge_table.csv", index=False)
    projection_edges.to_csv(output / "projection_edge_table.csv", index=False)
    coefficient_summary.to_csv(output / "coefficient_edge_table.csv", index=False)
    time_table.to_csv(output / "time_bucket_table.csv", index=False)

    pivot = edge_table.pivot(index="edge", columns="method", values="probability_recovery")
    report = [
        f"# Medium S4 temporal-coordinate diagnostic: {args.scope}",
        "",
        "All coefficients below are least-squares projections of the current request's Current-Exact S4 correction onto the direction frozen at cutover. They diagnose representation geometry and are not executable estimators.",
        "",
        "| edge | same-request oracle | frozen offset | global 1-scalar | layerwise 6-scalar |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for edge, row in pivot.iterrows():
        report.append(
            f"| {edge} | {row['same_request_S4_oracle']:.4f} | {row['frozen_cutover_S4']:.4f} | {row[GLOBAL]:.4f} | {row[LAYERWISE]:.4f} |"
        )
    report.extend(
        [
            "",
            "## Gates",
            "",
            f"- One global temporal coefficient: recovery {global_gate['edge_equal_probability_recovery']:.4f}, minimum edge {global_gate['minimum_edge_probability_recovery']:.4f}, positive {global_gate['positive_edges']}/5 — {'PASS' if global_gate['passed'] else 'FAIL'}.",
            f"- Six layerwise temporal coefficients: recovery {layer_gate['edge_equal_probability_recovery']:.4f}, minimum edge {layer_gate['minimum_edge_probability_recovery']:.4f}, positive {layer_gate['positive_edges']}/5 — {'PASS' if layer_gate['passed'] else 'FAIL'}.",
            f"- Same-request full S4 shared correction ceiling: {result['same_request_oracle_edge_equal_probability_recovery']:.4f}.",
            "",
            "## Representation conclusion",
            "",
            f"- Global/layerwise projection relative L2: {result['edge_equal_projection_relative_l2']['global']:.4f}/{result['edge_equal_projection_relative_l2']['layerwise']:.4f}.",
            f"- Adjudication: `{interpretation}`.",
            "- A passing oracle coordinate gate supports a response-basis representation only. Design 1 still needs a legal <=20% coefficient estimator/update path before any action can be frozen.",
            "- No label, confirmation user, serving action or target-KV fit is used.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
