#!/usr/bin/env python3
"""Compare rank-0 and query-conditioned low-rank canaries."""

from __future__ import annotations

import json
import os

import pandas as pd

from insight_two.common import EDGES, RESULT_ROOT


def main() -> None:
    rank0_root = RESULT_ROOT / "canary_rank0"
    low_root = RESULT_ROOT / "canary_low_rank"
    for root in (rank0_root, low_root):
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if not summary["passed"]:
            raise RuntimeError(f"invalid canary: {root}")
    output = low_root / "analysis"
    partial = low_root / "analysis.partial"
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir()
    rank0 = pd.concat(
        [pd.read_parquet(rank0_root / f"rank{rank}/score_records.parquet") for rank in range(4)],
        ignore_index=True,
    )
    rank0 = rank0[(rank0.source == "heldout") & (rank0.stage != "reuse")].copy()
    rank0["rank"] = 0
    rank0 = rank0.rename(
        columns={"correction_numel_fp32_per_user": "storage_values_fp32_per_user"}
    )
    low = pd.concat(
        [pd.read_parquet(low_root / f"rank{rank}/score_records.parquet") for rank in range(4)],
        ignore_index=True,
    )
    combined = pd.concat(
        [
            rank0[
                [
                    "edge",
                    "uid",
                    "stage",
                    "presentation",
                    "rank",
                    "storage_values_fp32_per_user",
                    "probability_gap_recovery",
                    "logit_gap_recovery",
                    "rank_correlation",
                    "top1_agreement",
                    "top10_overlap",
                ]
            ],
            low,
        ],
        ignore_index=True,
    )
    # S3 has no positive-rank implementation and is not an eligible compact
    # boundary; retain it in raw rank-0 evidence but exclude it here.
    combined = combined[combined.stage != "kv_prefix_contribution"]
    per_edge = (
        combined.groupby(["stage", "presentation", "rank", "edge"], as_index=False)
        .agg(
            users=("uid", "nunique"),
            probability_recovery=("probability_gap_recovery", "mean"),
            logit_recovery=("logit_gap_recovery", "mean"),
            rank_correlation=("rank_correlation", "mean"),
            storage_values=("storage_values_fp32_per_user", "max"),
        )
        .sort_values(["stage", "rank", "edge"])
    )
    frontier = (
        per_edge.groupby(["stage", "presentation", "rank"], as_index=False)
        .agg(
            edge_equal_probability_recovery=("probability_recovery", "mean"),
            edge_equal_logit_recovery=("logit_recovery", "mean"),
            edges_at_80=("probability_recovery", lambda values: int((values >= 0.80).sum())),
            edges_at_90=("probability_recovery", lambda values: int((values >= 0.90).sum())),
            minimum_edge_recovery=("probability_recovery", "min"),
            maximum_edge_recovery=("probability_recovery", "max"),
            storage_values=("storage_values", "max"),
        )
        .sort_values("edge_equal_probability_recovery", ascending=False)
    )
    frontier["shape_gate"] = (
        (frontier.edge_equal_probability_recovery >= 0.80)
        & (
            (frontier.edges_at_80 >= 4)
            | ((frontier.edges_at_90 >= 3) & (frontier.edge_equal_probability_recovery >= 0.80))
        )
    )
    fit = pd.concat(
        [pd.read_parquet(low_root / f"rank{rank}/fit_records.parquet") for rank in range(4)],
        ignore_index=True,
    )
    fit_summary = (
        fit.groupby(["stage", "presentation", "rank", "layer"], as_index=False)
        .agg(
            users=("uid", "nunique"),
            target_rank90=("target_rank90", "mean"),
            target_rank95=("target_rank95", "mean"),
            retained_centered_energy=("rank_retained_centered_energy", "mean"),
            anchor_fit_relative_l2=("anchor_fit_relative_l2", "mean"),
        )
    )
    per_edge.to_csv(partial / "per_edge.csv", index=False)
    frontier.to_csv(partial / "frontier.csv", index=False)
    fit_summary.to_csv(partial / "fit_structure.csv", index=False)
    lines = [
        "# Medium Insight 2 query-conditioned low-rank canary",
        "",
        "All target corrections are anchor-side Exact oracles. This tests representation capacity, not an executable estimator.",
        "",
        "| stage | rank | probability recovery | edges >=80% | edges >=90% | min edge | FP32 values/user | shape gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in frontier.itertuples(index=False):
        lines.append(
            f"| {row.presentation} | {row.rank} | {row.edge_equal_probability_recovery:.4f} | "
            f"{row.edges_at_80} | {row.edges_at_90} | {row.minimum_edge_recovery:.4f} | "
            f"{int(row.storage_values)} | {'PASS' if row.shape_gate else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "A positive rank is retained only if its held-out causal recovery improves materially over rank 0; in-sample anchor fit alone is not a selection criterion.",
            "",
        ]
    )
    (partial / "report.md").write_text("\n".join(lines), encoding="utf-8")
    best = frontier.sort_values(
        ["shape_gate", "edge_equal_probability_recovery", "storage_values"],
        ascending=[False, False, True],
    ).iloc[0]
    summary = {
        "status": "low_rank_canary_adjudicated",
        "labels_read": False,
        "best_representation_canary": {
            "stage": str(best.presentation),
            "rank": int(best["rank"]),
            "edge_equal_probability_recovery": float(best.edge_equal_probability_recovery),
            "storage_values_fp32_per_user": int(best.storage_values),
        },
        "caveat": "oracle representation test only; no executable estimator or persistence",
    }
    (partial / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

