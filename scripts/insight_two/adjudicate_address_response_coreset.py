#!/usr/bin/env python3
"""Adjudicate the attention-address signed-response coreset oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_address_response_coreset_v1.yaml"
)
RESULT_ROOT = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_address_response_coreset_v1"
)
CHRONOLOGICAL_GRID = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_signed_response_coreset_v1/canary/analysis/grid_table.csv"
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _grid(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    address = metrics[metrics.sample_count > 0].copy()
    aggregations: dict[str, tuple[str, str]] = {
        "probability_recovery": ("probability_gap_recovery", "mean"),
        "logit_recovery": ("logit_gap_recovery", "mean"),
        "bernoulli_js": ("bernoulli_js_to_exact", "mean"),
        "top1_agreement": ("top1_agreement", "mean"),
        "top10_overlap": ("top10_overlap", "mean"),
        "rank_correlation": ("rank_correlation", "mean"),
        "stored_scalars": ("stored_scalars", "max"),
        "stored_to_full_KV_ratio": ("stored_to_full_KV_ratio", "max"),
        "covering_radius": ("layer0_address_covering_radius", "mean"),
        "cluster_mass_min": ("cluster_mass_min", "min"),
        "cluster_mass_max": ("cluster_mass_max", "max"),
        "cluster_mass_sum": ("cluster_mass_sum", "min"),
    }
    missing = {source for source, _ in aggregations.values()} - set(address.columns)
    if missing:
        raise RuntimeError(f"address-response metrics missing columns: {sorted(missing)}")
    edge_table = (
        address.groupby(["edge", "method", "sample_count"], as_index=False)
        .agg(**aggregations)
        .sort_values(["sample_count", "edge"])
    )
    records = []
    for sample_count, group in edge_table.groupby("sample_count", sort=True):
        records.append(
            {
                "sample_count": int(sample_count),
                "edge_equal_probability_recovery": float(
                    group.probability_recovery.mean()
                ),
                "minimum_edge_probability_recovery": float(
                    group.probability_recovery.min()
                ),
                "positive_edges": int((group.probability_recovery > 0).sum()),
                "edges_at_or_above_0_80": int(
                    (group.probability_recovery >= 0.80).sum()
                ),
                "edge_equal_logit_recovery": float(group.logit_recovery.mean()),
                "mean_covering_radius": float(group.covering_radius.mean()),
                "minimum_cluster_mass": int(group.cluster_mass_min.min()),
                "maximum_cluster_mass": int(group.cluster_mass_max.max()),
                "minimum_cluster_mass_sum": int(group.cluster_mass_sum.min()),
                "stored_scalars": int(group.stored_scalars.max()),
                "stored_to_full_KV_ratio": float(
                    group.stored_to_full_KV_ratio.max()
                ),
            }
        )
    return edge_table, pd.DataFrame(records).sort_values("sample_count")


def _smallest_passing(
    grid: pd.DataFrame,
    *,
    recovery: float,
    positive_edges: int,
) -> int | None:
    passing = grid[
        grid.sample_count.isin([64, 128])
        & (grid.edge_equal_probability_recovery >= recovery)
        & (grid.positive_edges >= positive_edges)
    ].sort_values("sample_count")
    return None if passing.empty else int(passing.iloc[0].sample_count)


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
        raise RuntimeError("cannot adjudicate failed address-response instrumentation")
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    metrics = pd.read_parquet(source / "metrics.parquet")
    edge_table, grid = _grid(metrics)

    chronological = pd.read_csv(CHRONOLOGICAL_GRID)[
        ["sample_count", "edge_equal_probability_recovery"]
    ].rename(
        columns={
            "edge_equal_probability_recovery": "chronological_canary_recovery"
        }
    )
    comparison = grid.merge(chronological, on="sample_count", how="left", validate="1:1")
    comparison["address_minus_chronological_canary"] = (
        comparison.edge_equal_probability_recovery
        - comparison.chronological_canary_recovery
    )

    launch_rule = contract["gates"]["canary_to_discovery"]
    launch_selected = _smallest_passing(
        grid,
        recovery=float(
            launch_rule[
                "one_of_R64_or_R128_edge_equal_probability_recovery_at_least"
            ]
        ),
        positive_edges=int(launch_rule["selected_R_positive_edges_minimum"]),
    )
    launch = bool(args.scope == "canary" and launch_selected is not None)

    support_rule = contract["gates"]["insight_support"]
    support_selected = _smallest_passing(
        grid,
        recovery=float(
            support_rule[
                "one_of_R64_or_R128_edge_equal_probability_recovery_at_least"
            ]
        ),
        positive_edges=int(support_rule["selected_R_positive_edges_minimum"]),
    )
    comparison_count = support_selected if support_selected is not None else launch_selected
    matched_improvement = None
    if comparison_count is not None:
        matched = comparison[comparison.sample_count == comparison_count]
        matched_improvement = float(
            matched.iloc[0].address_minus_chronological_canary
        )
    geometry_gate = bool(
        matched_improvement is not None
        and matched_improvement
        >= float(
            support_rule[
                "selected_R_improvement_over_sealed_chronological_control_at_least"
            ]
        )
    )
    # The storage-matched geometry contrast is frozen on odd-32 canary users.
    # Discovery extends the address rule to more users; it does not revive the
    # chronological family after its preregistered canary stop.
    if args.scope == "discovery":
        canary_analysis = json.loads(
            (RESULT_ROOT / "canary/analysis/summary.json").read_text(encoding="utf-8")
        )
        geometry_gate = bool(canary_analysis["address_geometry_gate_passed"])
        matched_improvement = canary_analysis[
            "selected_R_address_minus_chronological_canary"
        ]
    insight_support = bool(support_selected is not None and geometry_gate)

    result = {
        "status": "address_response_coreset_adjudicated",
        "scope": args.scope,
        "contract_sha256": run["contract_sha256"],
        "labels_read": False,
        "oracle_exact_cache_used": True,
        "discovery_launch_gate_passed": launch,
        "smallest_passing_canary_sample_count": launch_selected,
        "attention_address_insight_gate_passed": insight_support,
        "smallest_passing_insight_sample_count": support_selected,
        "address_geometry_gate_passed": geometry_gate,
        "selected_R_address_minus_chronological_canary": matched_improvement,
        "interpretation": (
            "attention_address_contraction_supported_oracle_only"
            if insight_support
            else (
                "attention_address_oracle_promising_below_insight_gate"
                if launch_selected is not None
                else "attention_address_coreset_hypothesis_retired"
            )
        ),
        "design1_gate": "not_tested_Current_Exact_upper_layer_cache_is_used",
    }
    output.mkdir()
    atomic_json(output / "summary.json", result)
    edge_table.to_csv(output / "edge_table.csv", index=False)
    grid.to_csv(output / "grid_table.csv", index=False)
    comparison.to_csv(output / "chronological_comparison.csv", index=False)

    report = [
        f"# Medium attention-address signed response coreset: {args.scope}",
        "",
        "This is a fit-free Exact-state oracle. It changes only landmark geometry: layer-0 paired Current/Parent key coverage replaces chronological midpoints; the signed native-query reader and held-out panel are unchanged.",
        "",
        "| R | edge-equal recovery | minimum edge | positive edges | >=80% edges | address radius | address - chronological canary | full-KV ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.itertuples(index=False):
        report.append(
            f"| {row.sample_count} | {row.edge_equal_probability_recovery:.4f} | {row.minimum_edge_probability_recovery:.4f} | {row.positive_edges}/5 | {row.edges_at_or_above_0_80}/5 | {row.mean_covering_radius:.4f} | {row.address_minus_chronological_canary:.4f} | {row.stored_to_full_KV_ratio:.4%} |"
        )
    report.extend(
        [
            "",
            "## Adjudication",
            "",
            f"- Canary-to-discovery gate: {'PASS' if launch else 'FAIL'}; smallest passing R: {launch_selected}.",
            f"- Attention-address oracle gate: {'PASS' if insight_support else 'FAIL'}; smallest passing R: {support_selected}.",
            f"- Storage-matched address-geometry gate: {'PASS' if geometry_gate else 'FAIL'}; selected improvement: {matched_improvement}.",
            f"- Interpretation: `{result['interpretation']}`.",
            "- Address clustering alone is not a Design contribution. This oracle cannot admit Design 1 because selected upper-layer Current K/V remain Exact.",
            "- No labels, candidates in construction, response fitting, confirmation users or executable-cost claims are used.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
