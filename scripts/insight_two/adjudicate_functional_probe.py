#!/usr/bin/env python3
"""Adjudicate executable functional-probe estimator results without labels."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from insight_two.common import EDGES, RESULT_ROOT, sha256_file


OUTPUT_ROOT = RESULT_ROOT / "estimator_functional_probe_v1"


def harm_weighted_recovery(frame: pd.DataFrame, prefix: str) -> float:
    denominator = float(frame[f"reuse_{prefix}_gap"].sum())
    if denominator <= 0:
        raise RuntimeError(f"non-positive {prefix} reuse gap")
    return 1.0 - float(frame[f"observed_{prefix}_gap"].sum()) / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "discovery"), default="canary")
    args = parser.parse_args()
    root = OUTPUT_ROOT / args.scope
    run = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    expected_users = 32 if args.scope == "canary" else 512
    if not run.get("passed") or run.get("users") != expected_users:
        raise RuntimeError(f"functional-probe {args.scope} did not pass")
    output = root / "analysis"
    partial = root / "analysis.partial"
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir()
    paths = [root / f"rank{rank}/score_records.parquet" for rank in range(4)]
    scores = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)

    edge_rows = []
    for keys, frame in scores.groupby(["carriers", "probes", "edge"]):
        carriers, probes, edge = keys
        edge_rows.append(
            {
                "carriers": int(carriers),
                "probes": int(probes),
                "edge": edge,
                "users": int(frame.uid.nunique()),
                "theoretical_compute_fraction": float(
                    frame.theoretical_compute_fraction.max()
                ),
                "probability_recovery": float(frame.probability_gap_recovery.mean()),
                "logit_recovery": float(frame.logit_gap_recovery.mean()),
                "harm_weighted_probability_recovery": harm_weighted_recovery(
                    frame, "probability"
                ),
                "median_user_probability_recovery": float(
                    frame.probability_gap_recovery.median()
                ),
                "users_improved_fraction": float(
                    (frame.observed_probability_gap < frame.reuse_probability_gap).mean()
                ),
                "users_at_80_fraction": float(
                    (frame.probability_gap_recovery >= 0.80).mean()
                ),
                "correction_cosine_to_oracle": float(
                    frame.correction_cosine_to_oracle.mean()
                ),
                "correction_norm_ratio_to_oracle": float(
                    frame.correction_norm_ratio_to_oracle.median()
                ),
                "correction_relative_l2_to_oracle": float(
                    frame.correction_relative_l2_to_oracle.mean()
                ),
                "rank_correlation": float(frame.rank_correlation.mean()),
                "top1_agreement": float(frame.top1_agreement.mean()),
            }
        )
    per_edge = pd.DataFrame(edge_rows).sort_values(["carriers", "probes", "edge"])
    frontier = (
        per_edge.groupby(["carriers", "probes"], as_index=False)
        .agg(
            theoretical_compute_fraction=("theoretical_compute_fraction", "max"),
            edge_equal_probability_recovery=("probability_recovery", "mean"),
            minimum_edge_probability_recovery=("probability_recovery", "min"),
            edge_equal_logit_recovery=("logit_recovery", "mean"),
            edge_equal_harm_weighted_probability_recovery=(
                "harm_weighted_probability_recovery",
                "mean",
            ),
            edges_positive=("probability_recovery", lambda values: int((values > 0).sum())),
            edges_at_80=("probability_recovery", lambda values: int((values >= 0.80).sum())),
            edges_at_90=("probability_recovery", lambda values: int((values >= 0.90).sum())),
            median_user_probability_recovery=(
                "median_user_probability_recovery",
                "mean",
            ),
            users_improved_fraction=("users_improved_fraction", "mean"),
            users_at_80_fraction=("users_at_80_fraction", "mean"),
            correction_cosine_to_oracle=("correction_cosine_to_oracle", "mean"),
            correction_norm_ratio_to_oracle=("correction_norm_ratio_to_oracle", "mean"),
            correction_relative_l2_to_oracle=("correction_relative_l2_to_oracle", "mean"),
        )
        .sort_values(["theoretical_compute_fraction", "carriers", "probes"])
    )
    frontier["gate_80"] = (
        (frontier.theoretical_compute_fraction <= 0.20)
        & (frontier.edge_equal_probability_recovery >= 0.80)
        & (frontier.edges_positive >= 4)
        & (
            (frontier.edges_at_80 >= 4)
            | (
                (frontier.edges_at_90 >= 3)
                & (frontier.edge_equal_probability_recovery >= 0.80)
            )
        )
    )
    frontier["stretch_90"] = (
        (frontier.theoretical_compute_fraction <= 0.20)
        & (frontier.edge_equal_probability_recovery >= 0.90)
        & (frontier.edges_at_90 >= 4)
    )
    running_best = float("-inf")
    pareto = []
    for row in frontier.itertuples(index=False):
        retain = row.edge_equal_probability_recovery > running_best
        pareto.append(retain)
        if retain:
            running_best = row.edge_equal_probability_recovery
    frontier["cost_recovery_pareto"] = pareto

    per_edge.to_csv(partial / "per_edge.csv", index=False)
    frontier.to_csv(partial / "frontier.csv", index=False)
    lines = [
        f"# Medium executable functional-probe estimator: {args.scope}",
        "",
        "No label, request candidate, future event or Current-Exact K/V enters the estimator. Current Exact is used only after construction as the evaluation reference and for correction-shape diagnostics.",
        "",
        "| carriers | probes | Exact cost | recovery | min edge | edges >=80% | edges >=90% | cosine to oracle | norm ratio | gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in frontier.itertuples(index=False):
        lines.append(
            f"| {row.carriers} | {row.probes} | "
            f"{100 * row.theoretical_compute_fraction:.2f}% | "
            f"{row.edge_equal_probability_recovery:.4f} | "
            f"{row.minimum_edge_probability_recovery:.4f} | "
            f"{row.edges_at_80} | {row.edges_at_90} | "
            f"{row.correction_cosine_to_oracle:.4f} | "
            f"{row.correction_norm_ratio_to_oracle:.4f} | "
            f"{'PASS' if row.gate_80 else 'FAIL'} |"
        )
    passing = frontier[frontier.gate_80]
    if len(passing):
        cheapest = passing.sort_values(
            ["theoretical_compute_fraction", "edge_equal_probability_recovery"],
            ascending=[True, False],
        ).iloc[0]
        best = passing.sort_values(
            ["edge_equal_probability_recovery", "theoretical_compute_fraction"],
            ascending=[False, True],
        ).iloc[0]
        decision = "pass"
        decision_lines = [
            f"- Cheapest passing configuration: C{int(cheapest.carriers)}/P{int(cheapest.probes)}, {100 * cheapest.theoretical_compute_fraction:.2f}% Exact, recovery {cheapest.edge_equal_probability_recovery:.4f}.",
            f"- Highest-recovery passing configuration: C{int(best.carriers)}/P{int(best.probes)}, {100 * best.theoretical_compute_fraction:.2f}% Exact, recovery {best.edge_equal_probability_recovery:.4f}.",
        ]
    else:
        best = frontier.sort_values(
            ["edge_equal_probability_recovery", "theoretical_compute_fraction"],
            ascending=[False, True],
        ).iloc[0]
        decision = "fail"
        decision_lines = [
            f"- No configuration passes Gate D on this {args.scope}; best is C{int(best.carriers)}/P{int(best.probes)} at {best.edge_equal_probability_recovery:.4f} recovery.",
        ]
    lines.extend(
        [
            "",
            "## Adjudication",
            "",
            *decision_lines,
            "- This tests cutover construction only. Temporal persistence and task-label quality remain open even if the estimator gate passes.",
            "",
        ]
    )
    (partial / "report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "status": f"functional_probe_{args.scope}_adjudicated",
        "estimator_gate": decision,
        "labels_read": False,
        "users": expected_users,
        "edges": list(EDGES),
        "best_configuration": {
            "carriers": int(best.carriers),
            "probes": int(best.probes),
            "theoretical_compute_fraction": float(best.theoretical_compute_fraction),
            "edge_equal_probability_recovery": float(
                best.edge_equal_probability_recovery
            ),
            "minimum_edge_probability_recovery": float(
                best.minimum_edge_probability_recovery
            ),
            "edges_at_80": int(best.edges_at_80),
            "edges_at_90": int(best.edges_at_90),
        },
        "raw_artifacts": {
            str(path.relative_to(root)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        },
        "persistence_gate": "not_run",
        "quality_gate": "not_run",
    }
    (partial / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
