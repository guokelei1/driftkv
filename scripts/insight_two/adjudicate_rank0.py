#!/usr/bin/env python3
"""Aggregate the sealed rank-0 canary without reading labels."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from insight_two.common import EDGES, RESULT_ROOT, STAGE_PRESENTATION


def main() -> None:
    root = RESULT_ROOT / "canary_rank0"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not summary["passed"]:
        raise RuntimeError("cannot adjudicate an invalid instrumentation canary")
    output = root / "analysis_v2"
    partial = root / "analysis_v2.partial"
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir()
    scores = pd.concat(
        [pd.read_parquet(root / f"rank{rank}/score_records.parquet") for rank in range(4)],
        ignore_index=True,
    )
    energy = pd.concat(
        [pd.read_parquet(root / f"rank{rank}/energy_records.parquet") for rank in range(4)],
        ignore_index=True,
    )
    if set(scores.edge) != set(EDGES) or scores.uid.nunique() != 32:
        raise RuntimeError("canary score population is incomplete")
    grouped = (
        scores.groupby(["source", "edge", "stage", "presentation"], as_index=False)
        .agg(
            users=("uid", "nunique"),
            probability_gap_recovery=("probability_gap_recovery", "mean"),
            logit_gap_recovery=("logit_gap_recovery", "mean"),
            rank_correlation=("rank_correlation", "mean"),
            top1_agreement=("top1_agreement", "mean"),
            top10_overlap=("top10_overlap", "mean"),
            correction_numel=("correction_numel_fp32_per_user", "max"),
        )
        .sort_values(["source", "stage", "edge"])
    )
    structure = (
        energy.groupby(["edge", "stage", "presentation", "layer"], as_index=False)
        .agg(
            users=("uid", "nunique"),
            candidate_mean_energy_fraction=("candidate_mean_energy_fraction", "mean"),
            candidate_centered_energy_fraction=("candidate_centered_energy_fraction", "mean"),
        )
        .sort_values(["stage", "edge", "layer"])
    )
    heldout = grouped[(grouped.source == "heldout") & (grouped.stage != "reuse")]
    edge_equal = (
        heldout.groupby(["stage", "presentation"], as_index=False)
        .agg(
            edge_equal_probability_recovery=("probability_gap_recovery", "mean"),
            edge_equal_logit_recovery=("logit_gap_recovery", "mean"),
            edges_at_80=("probability_gap_recovery", lambda values: int((values >= 0.80).sum())),
            edges_at_90=("probability_gap_recovery", lambda values: int((values >= 0.90).sum())),
            minimum_edge_recovery=("probability_gap_recovery", "min"),
            maximum_edge_recovery=("probability_gap_recovery", "max"),
            correction_numel=("correction_numel", "max"),
        )
        .sort_values("edge_equal_probability_recovery", ascending=False)
    )
    edge_equal["canary_gate_b_shape_only"] = (
        (edge_equal.edge_equal_probability_recovery >= 0.80)
        & (
            (edge_equal.edges_at_80 >= 4)
            | ((edge_equal.edges_at_90 >= 3) & (edge_equal.edge_equal_probability_recovery >= 0.80))
        )
    )
    # The S3 correction is already summed over positions before injection.  The
    # legacy helper name ``final_readout`` denotes the H-dimensional final
    # representation before the scalar head, so it remains an eligible S7
    # boundary; only an actual scalar-logit correction would be disallowed.
    edge_equal.loc[
        edge_equal.stage == "kv_prefix_contribution",
        "canary_gate_b_shape_only",
    ] = False

    grouped.to_csv(partial / "edge_stage_metrics.csv", index=False)
    structure.to_csv(partial / "stage_energy.csv", index=False)
    edge_equal.to_csv(partial / "rank0_frontier.csv", index=False)
    lines = [
        "# Medium Insight 2 rank-0 focused canary",
        "",
        "This is a 32-user instrumentation and representation canary, not Design 1 qualification.",
        "The correction is estimated on 32 anchor candidates and intervened only on 32 held-out candidates.",
        "No label was read. `kv_prefix_contribution` is position-summed before injection and cannot qualify as a token-local action.",
        "",
        "## Anchor-to-heldout rank-0 frontier",
        "",
        "| stage | probability recovery | logit recovery | edges >=80% | edges >=90% | min edge | max edge | FP32 values/user | canary shape gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in edge_equal.itertuples(index=False):
        lines.append(
            f"| {row.presentation} | {row.edge_equal_probability_recovery:.4f} | "
            f"{row.edge_equal_logit_recovery:.4f} | {row.edges_at_80} | {row.edges_at_90} | "
            f"{row.minimum_edge_recovery:.4f} | {row.maximum_edge_recovery:.4f} | "
            f"{int(row.correction_numel)} | {'PASS' if row.canary_gate_b_shape_only else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Per-edge held-out probability recovery",
            "",
            "| stage | " + " | ".join(EDGES) + " |",
            "| --- | " + " | ".join(["---:"] * len(EDGES)) + " |",
        ]
    )
    for stage, presentation in STAGE_PRESENTATION.items():
        values = heldout[heldout.stage == stage].set_index("edge")["probability_gap_recovery"]
        lines.append(
            f"| {presentation} | "
            + " | ".join(f"{values[edge]:.4f}" for edge in EDGES)
            + " |"
        )
    lines.extend(
        [
            "",
            "Passing this canary only unlocks low-rank query-conditioned instrumentation and a resource estimate.",
            "It does not establish temporal persistence, an executable estimator, the 0%–20% cost gate, or task-label quality.",
            "",
        ]
    )
    (partial / "report.md").write_text("\n".join(lines), encoding="utf-8")
    adjudication = {
        "status": "rank0_canary_adjudicated",
        "instrumentation_passed": True,
        "users": 32,
        "edges": list(EDGES),
        "labels_read": False,
        "rank0_candidate_stages_passing_canary_shape_gate": edge_equal.loc[
            edge_equal.canary_gate_b_shape_only, "presentation"
        ].tolist(),
        "caveat": "representation canary only; not executable Design 1 evidence",
    }
    (partial / "summary.json").write_text(
        json.dumps(adjudication, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    print(json.dumps(adjudication, indent=2))


if __name__ == "__main__":
    main()
