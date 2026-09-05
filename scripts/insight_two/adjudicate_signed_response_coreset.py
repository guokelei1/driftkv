#!/usr/bin/env python3
"""Adjudicate the fit-free oracle signed-response coreset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_signed_response_coreset_v1.yaml"
)
RESULT_ROOT = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_signed_response_coreset_v1"
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "discovery"), required=True)
    args = parser.parse_args()
    source = RESULT_ROOT / args.scope
    output = source / "analysis"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    run = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if not run.get("passed"):
        raise RuntimeError("cannot adjudicate failed signed-response instrumentation")
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    metrics = pd.read_parquet(source / "metrics.parquet")
    signed = metrics[metrics.sample_count > 0].copy()
    edge_table = (
        signed.groupby(["edge", "method", "sample_count"], as_index=False)
        .agg(
            probability_recovery=("probability_gap_recovery", "mean"),
            logit_recovery=("logit_gap_recovery", "mean"),
            bernoulli_js=("bernoulli_js_to_exact", "mean"),
            top1_agreement=("top1_agreement", "mean"),
            top10_overlap=("top10_overlap", "mean"),
            rank_correlation=("rank_correlation", "mean"),
            stored_scalars=("stored_scalars", "max"),
            stored_to_full_KV_ratio=("stored_to_full_KV_ratio", "max"),
        )
    )
    grid_records = []
    for sample_count, group in edge_table.groupby("sample_count", sort=True):
        recovery = float(group.probability_recovery.mean())
        grid_records.append(
            {
                "sample_count": int(sample_count),
                "edge_equal_probability_recovery": recovery,
                "minimum_edge_probability_recovery": float(
                    group.probability_recovery.min()
                ),
                "positive_edges": int((group.probability_recovery > 0).sum()),
                "edges_at_or_above_0_80": int(
                    (group.probability_recovery >= 0.80).sum()
                ),
                "edge_equal_logit_recovery": float(group.logit_recovery.mean()),
                "stored_scalars": int(group.stored_scalars.max()),
                "stored_to_full_KV_ratio": float(
                    group.stored_to_full_KV_ratio.max()
                ),
            }
        )
    grid = pd.DataFrame(grid_records).sort_values("sample_count")
    r128 = grid[grid.sample_count == 128].iloc[0]
    launch_rule = contract["gates"]["canary_to_discovery"]
    launch = bool(
        r128.edge_equal_probability_recovery
        >= float(launch_rule["R128_edge_equal_probability_recovery_at_least"])
        and r128.positive_edges >= int(launch_rule["R128_positive_edges_minimum"])
    )
    passing = grid[
        grid.sample_count.isin([64, 128])
        & (grid.edge_equal_probability_recovery >= 0.80)
        & (grid.positive_edges >= 4)
    ]
    selected = None if passing.empty else int(passing.iloc[0].sample_count)
    continuation = bool(
        r128.edge_equal_probability_recovery >= 0.70 and r128.positive_edges >= 5
    )
    result = {
        "status": "signed_response_coreset_adjudicated",
        "scope": args.scope,
        "contract_sha256": run["contract_sha256"],
        "labels_read": False,
        "oracle_exact_cache_used": True,
        "discovery_launch_gate_passed": launch,
        "operator_support_gate_passed": selected is not None,
        "smallest_passing_operator_sample_count": selected,
        "exploratory_continuation_gate_passed": continuation,
        "R128_edge_equal_probability_recovery": float(
            r128.edge_equal_probability_recovery
        ),
        "R128_minimum_edge_probability_recovery": float(
            r128.minimum_edge_probability_recovery
        ),
        "R128_positive_edges": int(r128.positive_edges),
        "interpretation": (
            "compact_native_query_response_operator_supported"
            if selected is not None
            else (
                "signed_coreset_promising_but_below_operator_gate"
                if continuation
                else "fixed_midpoint_signed_coreset_family_retired"
            )
        ),
        "design1_gate": "not_tested_Current_Exact_cache_is_used_in_construction",
    }
    output.mkdir()
    atomic_json(output / "summary.json", result)
    edge_table.to_csv(output / "edge_table.csv", index=False)
    grid.to_csv(output / "grid_table.csv", index=False)

    report = [
        f"# Medium signed attention-response coreset: {args.scope}",
        "",
        "This is a fit-free Exact-state oracle. The complete Parent response is the control path; real held-out Current queries read paired positive-Current and negative-Parent midpoint atoms through the native attention kernel.",
        "",
        "| R | edge-equal recovery | minimum edge | positive edges | >=80% edges | stored scalars | full-KV ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grid.itertuples(index=False):
        report.append(
            f"| {row.sample_count} | {row.edge_equal_probability_recovery:.4f} | {row.minimum_edge_probability_recovery:.4f} | {row.positive_edges}/5 | {row.edges_at_or_above_0_80}/5 | {row.stored_scalars} | {row.stored_to_full_KV_ratio:.4%} |"
        )
    report.extend(
        [
            "",
            "## Adjudication",
            "",
            f"- Canary-to-discovery gate: {'PASS' if launch else 'FAIL'}.",
            f"- Compact operator gate: {'PASS' if selected is not None else 'FAIL'}; smallest passing R: {selected}.",
            f"- Interpretation: `{result['interpretation']}`.",
            "- No label, candidate-conditioned construction, response regression, confirmation user or executable-cost claim is used.",
            "- Passing this oracle can only unlock a separately contracted sparse causal-replay constructor; it cannot admit Design 1.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
