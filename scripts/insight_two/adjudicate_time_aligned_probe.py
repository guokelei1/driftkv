#!/usr/bin/env python3
"""Adjudicate and compare cutover-time versus zero-time S4 probes."""

from __future__ import annotations

import json
import os

import pandas as pd

from insight_two.common import EDGES, RESULT_ROOT, sha256_file


def load(root):
    return pd.concat(
        [pd.read_parquet(root / f"rank{rank}/score_records.parquet") for rank in range(4)],
        ignore_index=True,
    )


def summarize(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    edge = (
        frame.groupby(["carriers", "edge"], as_index=False)
        .agg(
            probability_recovery=("probability_gap_recovery", "mean"),
            reuse_gap=("reuse_probability_gap", "sum"),
            observed_gap=("observed_probability_gap", "sum"),
            cosine=("correction_cosine_to_oracle", "mean"),
            norm_ratio=("correction_norm_ratio_to_oracle", "median"),
            relative_l2=("correction_relative_l2_to_oracle", "mean"),
            theoretical_compute_fraction=("theoretical_compute_fraction", "max"),
        )
    )
    edge["harm_weighted_recovery"] = 1.0 - edge.observed_gap / edge.reuse_gap
    result = (
        edge.groupby("carriers", as_index=False)
        .agg(
            theoretical_compute_fraction=("theoretical_compute_fraction", "max"),
            edge_equal_probability_recovery=("probability_recovery", "mean"),
            minimum_edge_probability_recovery=("probability_recovery", "min"),
            edges_at_80=("probability_recovery", lambda values: int((values >= 0.80).sum())),
            edge_equal_harm_weighted_recovery=("harm_weighted_recovery", "mean"),
            correction_cosine=("cosine", "mean"),
            correction_norm_ratio=("norm_ratio", "mean"),
            correction_relative_l2=("relative_l2", "mean"),
        )
    )
    result["time_semantics"] = label
    return result


def main() -> None:
    aligned_root = RESULT_ROOT / "estimator_time_aligned_probe_v1/canary"
    zero_root = RESULT_ROOT / "estimator_functional_probe_v1/canary"
    run = json.loads((aligned_root / "summary.json").read_text(encoding="utf-8"))
    if not run.get("passed"):
        raise RuntimeError("time-aligned raw canary did not pass")
    output = aligned_root / "analysis"
    partial = aligned_root / "analysis.partial"
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir()
    aligned_raw = load(aligned_root)
    zero_raw = load(zero_root)
    zero_raw = zero_raw[
        (zero_raw.probes == 1) & (zero_raw.carriers.isin([32, 64]))
    ]
    frontier = pd.concat(
        [summarize(zero_raw, "zero"), summarize(aligned_raw, "cutover")],
        ignore_index=True,
    ).sort_values(["carriers", "time_semantics"])
    frontier["gate_80"] = (
        (frontier.edge_equal_probability_recovery >= 0.80)
        & (frontier.edges_at_80 >= 4)
        & (frontier.theoretical_compute_fraction <= 0.20)
    )
    frontier.to_csv(partial / "frontier.csv", index=False)
    lines = [
        "# Medium time-aligned functional-probe canary",
        "",
        "The only changed variable is the fixed probe's query-time delta: zero versus the known release cutover delta. Neither estimator reads labels, request candidates or Current-Exact K/V.",
        "",
        "| carriers | probe time | Exact cost | recovery | min edge | edges >=80% | cosine | norm ratio | weighted sensitivity | gate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in frontier.itertuples(index=False):
        lines.append(
            f"| {row.carriers} | {row.time_semantics} | "
            f"{100 * row.theoretical_compute_fraction:.2f}% | "
            f"{row.edge_equal_probability_recovery:.4f} | "
            f"{row.minimum_edge_probability_recovery:.4f} | {row.edges_at_80} | "
            f"{row.correction_cosine:.4f} | {row.correction_norm_ratio:.4f} | "
            f"{row.edge_equal_harm_weighted_recovery:.4f} | "
            f"{'PASS' if row.gate_80 else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Adjudication",
            "",
            "- Correcting probe time modestly improves direction and average recovery but no edge reaches the 80% gate.",
            "- More probes were already flat and denser carriers improve cosine without improving score recovery. The parameter-only map plus Parent-conditioned-carrier response estimator family is retired.",
            "- This negative result does not invalidate the Exact-oracle S4 functional boundary; it isolates estimator bias as the open problem.",
            "- No 512-user run is authorized from this failed canary.",
            "",
        ]
    )
    (partial / "report.md").write_text("\n".join(lines), encoding="utf-8")
    aligned = frontier[frontier.time_semantics == "cutover"].sort_values(
        "edge_equal_probability_recovery", ascending=False
    ).iloc[0]
    summary = {
        "status": "time_aligned_probe_valid_negative",
        "estimator_gate": "fail",
        "labels_read": False,
        "best_time_aligned_configuration": {
            "carriers": int(aligned.carriers),
            "theoretical_compute_fraction": float(aligned.theoretical_compute_fraction),
            "edge_equal_probability_recovery": float(
                aligned.edge_equal_probability_recovery
            ),
            "minimum_edge_probability_recovery": float(
                aligned.minimum_edge_probability_recovery
            ),
            "edges_at_80": int(aligned.edges_at_80),
        },
        "family_decision": "retire_parameter_map_plus_Parent_conditioned_carrier_response_estimator",
        "discovery_512_authorized": False,
        "raw_artifacts": {
            f"rank{rank}/score_records.parquet": {
                "sha256": sha256_file(aligned_root / f"rank{rank}/score_records.parquet")
            }
            for rank in range(4)
        },
    }
    (partial / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
