#!/usr/bin/env python3
"""Adjudicate fixed-history-query and release-basis oracle diagnostics."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from insight_two.common import EDGES, RESULT_ROOT, sha256_file


ROOT = RESULT_ROOT / "diagnostic_release_basis_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "discovery"), default="canary")
    args = parser.parse_args()
    run_root = ROOT / ("canary_v2" if args.scope == "canary" else "discovery")
    run = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    if not run.get("passed"):
        raise RuntimeError("release-basis raw diagnostic did not pass")
    output = run_root / "analysis"
    partial = run_root / "analysis.partial"
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir()
    score_paths = [run_root / f"rank{rank}/score_records.parquet" for rank in range(4)]
    structure_paths = [
        run_root / f"rank{rank}/structure_records.parquet" for rank in range(4)
    ]
    scores = pd.concat([pd.read_parquet(path) for path in score_paths], ignore_index=True)
    structures = pd.concat(
        [pd.read_parquet(path) for path in structure_paths], ignore_index=True
    ).drop_duplicates()

    edge = (
        scores.groupby(["method", "rank", "edge"], as_index=False)
        .agg(
            users=("uid", "nunique"),
            probability_recovery=("probability_gap_recovery", "mean"),
            logit_recovery=("logit_gap_recovery", "mean"),
            reuse_gap=("reuse_probability_gap", "sum"),
            observed_gap=("observed_probability_gap", "sum"),
            cosine_to_anchor=("correction_cosine_to_anchor_target", "mean"),
            norm_ratio_to_anchor=("correction_norm_ratio_to_anchor_target", "median"),
            relative_l2_to_anchor=("correction_relative_l2_to_anchor_target", "mean"),
            users_at_80_fraction=(
                "probability_gap_recovery",
                lambda values: float((values >= 0.80).mean()),
            ),
        )
        .sort_values(["method", "rank", "edge"])
    )
    edge["harm_weighted_probability_recovery"] = 1.0 - edge.observed_gap / edge.reuse_gap
    frontier = (
        edge.groupby(["method", "rank"], as_index=False)
        .agg(
            edge_equal_probability_recovery=("probability_recovery", "mean"),
            minimum_edge_probability_recovery=("probability_recovery", "min"),
            edge_equal_logit_recovery=("logit_recovery", "mean"),
            edge_equal_harm_weighted_probability_recovery=(
                "harm_weighted_probability_recovery",
                "mean",
            ),
            edges_at_80=("probability_recovery", lambda values: int((values >= 0.80).sum())),
            edges_at_90=("probability_recovery", lambda values: int((values >= 0.90).sum())),
            cosine_to_anchor=("cosine_to_anchor", "mean"),
            norm_ratio_to_anchor=("norm_ratio_to_anchor", "mean"),
            relative_l2_to_anchor=("relative_l2_to_anchor", "mean"),
            users_at_80_fraction=("users_at_80_fraction", "mean"),
        )
        .sort_values(["method", "rank"])
    )
    frontier["structure_gate_80"] = (
        (frontier.edge_equal_probability_recovery >= 0.80)
        & (frontier.edges_at_80 >= 4)
    )
    structure_summary = (
        structures.groupby("layer", as_index=False)
        .agg(
            edges=("edge", "nunique"),
            mean_rank90=("rank90", "mean"),
            maximum_rank90=("rank90", "max"),
            mean_rank95=("rank95", "mean"),
            mean_rank1_energy=("rank1_energy", "mean"),
            mean_rank2_energy=("rank2_energy", "mean"),
            mean_rank4_energy=("rank4_energy", "mean"),
        )
    )
    edge.to_csv(partial / "per_edge.csv", index=False)
    frontier.to_csv(partial / "frontier.csv", index=False)
    structure_summary.to_csv(partial / "structure.csv", index=False)
    lines = [
        f"# Medium release-functional-basis diagnostic: {args.scope}",
        "",
        "All rows are label-free oracle diagnostics. `exact_fixed_history_probe_mean` requires each target user's Current-Exact cache. Positive release-basis ranks also use each evaluation user's Exact target coefficients. Neither is an executable migration action.",
        "",
        "| method | rank | recovery | min edge | edges >=80% | edges >=90% | cosine to candidate-anchor target | user fraction >=80% | gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in frontier.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.rank} | {row.edge_equal_probability_recovery:.4f} | "
            f"{row.minimum_edge_probability_recovery:.4f} | {row.edges_at_80} | "
            f"{row.edges_at_90} | {row.cosine_to_anchor:.4f} | "
            f"{row.users_at_80_fraction:.4f} | "
            f"{'PASS' if row.structure_gate_80 else 'FAIL'} |"
        )
    history = frontier[frontier.method == "exact_fixed_history_probe_mean"].iloc[0]
    basis_rows = frontier[frontier.method == "oracle_release_basis"]
    best_basis = basis_rows.sort_values(
        ["edge_equal_probability_recovery", "rank"], ascending=[False, True]
    ).iloc[0]
    prerequisites = bool(history.structure_gate_80 and best_basis.structure_gate_80)
    lines.extend(
        [
            "",
            "## Adjudication",
            "",
            f"- Fixed history-query Exact ceiling: {history.edge_equal_probability_recovery:.4f} recovery, {history.edges_at_80}/5 edges >=80%.",
            f"- Best preregistered release-basis oracle ceiling: rank {int(best_basis['rank'])}, {best_basis.edge_equal_probability_recovery:.4f} recovery, {best_basis.edges_at_80}/5 edges >=80%.",
            f"- Mean per-layer calibration-cohort rank@90: {structures.rank90.mean():.2f}; mean rank-1/rank-2/rank-4 energy: {structures.rank1_energy.mean():.4f}/{structures.rank2_energy.mean():.4f}/{structures.rank4_energy.mean():.4f}.",
            f"- Structural prerequisites for considering a separately authorized release-calibration design: {'PASS' if prerequisites else 'FAIL'}.",
            "- Passing prerequisites does not relax the repository ban on adding predictor complexity to the current scale frontier and does not pass the 0--20% executable-estimator gate.",
            "",
        ]
    )
    (partial / "report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "status": f"release_basis_{args.scope}_adjudicated",
        "labels_read": False,
        "oracle_diagnostic_only": True,
        "history_query_ceiling": {
            "edge_equal_probability_recovery": float(
                history.edge_equal_probability_recovery
            ),
            "minimum_edge_probability_recovery": float(
                history.minimum_edge_probability_recovery
            ),
            "edges_at_80": int(history.edges_at_80),
        },
        "best_release_basis_ceiling": {
            "rank": int(best_basis["rank"]),
            "edge_equal_probability_recovery": float(
                best_basis.edge_equal_probability_recovery
            ),
            "minimum_edge_probability_recovery": float(
                best_basis.minimum_edge_probability_recovery
            ),
            "edges_at_80": int(best_basis.edges_at_80),
        },
        "structural_prerequisites": "pass" if prerequisites else "fail",
        "executable_estimator_gate": "not_tested_and_not_authorized",
        "raw_artifacts": {
            str(path.relative_to(run_root)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (*score_paths, *structure_paths)
        },
    }
    (partial / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
