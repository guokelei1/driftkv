#!/usr/bin/env python3
"""Adjudicate the frozen Tail-128 to S4 focused estimator family."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_tail_functional_v1.yaml"
)
SOURCE = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/estimator_tail_functional_v1/canary"
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    output = SOURCE / "analysis"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    run = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(SOURCE / "score_records.parquet")
    edges = frame.groupby(["edge", "probes"], as_index=False).agg(
        probability_recovery=("probability_gap_recovery", "mean"),
        logit_recovery=("logit_gap_recovery", "mean"),
        correction_cosine=("correction_cosine_to_oracle", "mean"),
        correction_norm_ratio=("correction_norm_ratio_to_oracle", "mean"),
        correction_relative_l2=("correction_relative_l2_to_oracle", "mean"),
        theoretical_compute=("theoretical_compute_fraction", "first"),
    )
    aggregate = edges.groupby("probes", as_index=False).agg(
        edge_equal_probability_recovery=("probability_recovery", "mean"),
        minimum_edge_probability_recovery=("probability_recovery", "min"),
        edge_equal_logit_recovery=("logit_recovery", "mean"),
        correction_cosine=("correction_cosine", "mean"),
        correction_norm_ratio=("correction_norm_ratio", "mean"),
        correction_relative_l2=("correction_relative_l2", "mean"),
        theoretical_compute=("theoretical_compute", "first"),
    )
    positive = (
        edges.assign(positive=edges.probability_recovery > 0)
        .groupby("probes", as_index=False)
        .positive.sum()
        .rename(columns={"positive": "positive_edges"})
    )
    aggregate = aggregate.merge(positive, on="probes")
    primary = aggregate[aggregate.probes == 4].iloc[0]
    gate = contract["gates"]["focused_family_gate"]
    passed = bool(
        primary.edge_equal_probability_recovery
        >= float(gate["edge_equal_probability_recovery_at_least"])
        and int(primary.positive_edges) >= int(gate["positive_edges_minimum"])
    )
    result = {
        "status": "tail_functional_canary_adjudicated",
        "contract_sha256": run["contract_sha256"],
        "labels_read": False,
        "Current_Exact_in_estimator": False,
        "instrumentation_passed": bool(run["passed"]),
        "primary": "P4",
        "primary_theoretical_compute_fraction": float(primary.theoretical_compute),
        "primary_edge_equal_probability_recovery": float(
            primary.edge_equal_probability_recovery
        ),
        "primary_minimum_edge_probability_recovery": float(
            primary.minimum_edge_probability_recovery
        ),
        "primary_positive_edges": int(primary.positive_edges),
        "focused_family_gate_passed": passed,
        "family_status": (
            "eligible_for_512_discovery"
            if passed
            else "retired_by_preregistered_stop_rule_no_512_discovery"
        ),
    }
    atomic_json(output / "summary.json", result)
    edges.to_csv(output / "edge_table.csv", index=False)
    aggregate.to_csv(output / "aggregate_table.csv", index=False)
    pivot = edges.pivot(index="edge", columns="probes", values="probability_recovery")
    report = [
        "# Medium Tail-128 functional estimator canary",
        "",
        "This is an executable, label-free estimator canary. It transiently performs dependency-closed Current Tail-128 replay against the Parent prefix, discards that mixed cache, and persists only a 1,152-scalar S4 sidecar. Current Exact is evaluation-only.",
        "",
        "| edge | P1 | P2 | P4 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for edge, values in pivot.iterrows():
        report.append(
            f"| {edge} | {values[1]:.4f} | {values[2]:.4f} | {values[4]:.4f} |"
        )
    report.extend(
        [
            "",
            "| probes | Exact-All compute | edge-equal recovery | min edge | positive edges | cosine | norm ratio |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate.itertuples(index=False):
        report.append(
            f"| {row.probes} | {row.theoretical_compute:.4%} | {row.edge_equal_probability_recovery:.4f} | {row.minimum_edge_probability_recovery:.4f} | {int(row.positive_edges)}/5 | {row.correction_cosine:.4f} | {row.correction_norm_ratio:.4f} |"
        )
    report.extend(
        [
            "",
            "## Adjudication",
            "",
            f"- P4 uses {primary.theoretical_compute:.4%} of Exact-All and recovers {primary.edge_equal_probability_recovery:.4f} edge-equal probability gap; {int(primary.positive_edges)}/5 edges are positive.",
            f"- Focused family gate: {'PASS' if passed else 'FAIL'}.",
            "- The family may proceed to 512 users." if passed else "- The preregistered stop rule retires this family. Width, positions, probes and scale are not changed, and no 512-user run is launched.",
            "- The result does not authorize serving promotion, qualification labels or a new predictor.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
