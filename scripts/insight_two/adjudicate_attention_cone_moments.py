#!/usr/bin/env python3
"""Adjudicate the user attention-cone response-moment diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_attention_cone_moments_v1.yaml"
)
RESULT_ROOT = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_attention_cone_moments_v1"
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _method_grid(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "edge",
        "method",
        "sample_kind",
        "sample_count",
        "probability_gap_recovery",
        "logit_gap_recovery",
        "bernoulli_js_to_exact",
        "top1_agreement",
        "top10_overlap",
        "rank_correlation",
        "persistent_moment_scalars",
        "persistent_storage_ratio_to_current_KV",
        "temporary_current_sample_KV_ratio",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise RuntimeError(f"cone-moment metrics missing columns: {sorted(missing)}")
    observed = metrics[metrics.method != "Current_Reuse"].copy()
    edge_table = (
        observed.groupby(
            ["edge", "method", "sample_kind", "sample_count"], as_index=False
        )
        .agg(
            probability_recovery=("probability_gap_recovery", "mean"),
            logit_recovery=("logit_gap_recovery", "mean"),
            bernoulli_js=("bernoulli_js_to_exact", "mean"),
            top1_agreement=("top1_agreement", "mean"),
            top10_overlap=("top10_overlap", "mean"),
            rank_correlation=("rank_correlation", "mean"),
            persistent_moment_scalars=("persistent_moment_scalars", "max"),
            persistent_storage_ratio_to_current_KV=(
                "persistent_storage_ratio_to_current_KV",
                "max",
            ),
            temporary_current_sample_KV_ratio=(
                "temporary_current_sample_KV_ratio",
                "max",
            ),
        )
        .sort_values(["sample_kind", "sample_count", "edge"])
    )
    records = []
    for keys, group in edge_table.groupby(
        ["method", "sample_kind", "sample_count"], sort=False
    ):
        method, sample_kind, sample_count = keys
        records.append(
            {
                "method": method,
                "sample_kind": sample_kind,
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
                "persistent_moment_scalars": int(
                    group.persistent_moment_scalars.max()
                ),
                "persistent_storage_ratio_to_current_KV": float(
                    group.persistent_storage_ratio_to_current_KV.max()
                ),
                "temporary_current_sample_KV_ratio": float(
                    group.temporary_current_sample_KV_ratio.max()
                ),
            }
        )
    grid = pd.DataFrame(records)
    order = {"full": 0, "chronological": 1, "address": 2}
    grid["_order"] = grid.sample_kind.map(order)
    grid = grid.sort_values(["_order", "sample_count"]).drop(columns="_order")
    return edge_table, grid


def _row(grid: pd.DataFrame, method: str) -> pd.Series:
    selected = grid[grid.method == method]
    if len(selected) != 1:
        raise RuntimeError(f"expected exactly one aggregate row for {method}")
    return selected.iloc[0]


def _smallest_sampled_passing(grid: pd.DataFrame, threshold: float) -> int | None:
    passing = grid[
        grid.method.isin(["address_R64", "address_R128"])
        & (grid.edge_equal_probability_recovery >= threshold)
        & (grid.positive_edges >= 4)
    ].sort_values("sample_count")
    return None if passing.empty else int(passing.iloc[0].sample_count)


def _cone_summary(cone: pd.DataFrame) -> dict[str, float]:
    expected = {
        "heldout_current_majority_agreement",
        "heldout_parent_majority_agreement",
        "current_parent_sign_crossing_fraction",
        "current_negative_activation_fraction",
        "parent_negative_activation_fraction",
        "current_negative_response_fraction",
        "parent_negative_response_fraction",
        "qk_abs_p50",
        "qk_abs_p95",
    }
    missing = expected - set(cone.columns)
    if missing:
        raise RuntimeError(f"cone diagnostics missing columns: {sorted(missing)}")
    return {name: float(cone[name].mean()) for name in sorted(expected)}


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
        raise RuntimeError("cannot adjudicate failed cone-moment instrumentation")
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    metrics = pd.read_parquet(source / "metrics.parquet")
    cone = pd.read_parquet(source / "cone_diagnostics.parquet")
    edge_table, grid = _method_grid(metrics)

    full = _row(grid, "full_cone_moment")
    representation_rule = contract["gates"]["representation_support"]
    representation = bool(
        full.edge_equal_probability_recovery
        >= float(
            representation_rule[
                "full_moment_edge_equal_probability_recovery_at_least"
            ]
        )
        and full.minimum_edge_probability_recovery
        >= float(
            representation_rule[
                "full_moment_minimum_edge_probability_recovery_at_least"
            ]
        )
        and full.positive_edges
        >= int(representation_rule["full_moment_positive_edges_minimum"])
    )
    address128 = _row(grid, "address_R128")
    chronological128 = _row(grid, "chronological_R128")
    address_advantage = float(
        address128.edge_equal_probability_recovery
        - chronological128.edge_equal_probability_recovery
    )
    launch_rule = contract["gates"]["canary_to_discovery"]
    sampled_canary = bool(
        address128.edge_equal_probability_recovery
        >= float(
            launch_rule[
                "address_R128_edge_equal_probability_recovery_at_least"
            ]
        )
        and address128.positive_edges
        >= int(launch_rule["address_R128_positive_edges_minimum"])
        and address_advantage
        >= float(launch_rule["address_R128_minus_chronological_R128_at_least"])
    )
    launch = bool(args.scope == "canary" and representation and sampled_canary)
    sampled_threshold = float(
        contract["gates"]["sampled_moment_support"][
            "one_of_address_R64_or_R128_edge_equal_probability_recovery_at_least"
        ]
    )
    sampled_selected = _smallest_sampled_passing(grid, sampled_threshold)
    sampled_support = sampled_selected is not None
    cone_summary = _cone_summary(cone)

    if not representation:
        interpretation = "attention_cone_functional_state_rejected"
    elif not sampled_canary:
        interpretation = "cone_moment_representation_supported_sampled_constructor_rejected"
    elif sampled_support:
        interpretation = "cone_moments_and_sampled_Current_oracle_supported"
    else:
        interpretation = "cone_moment_representation_supported_sampled_oracle_below_80"
    result = {
        "status": "attention_cone_moments_adjudicated",
        "scope": args.scope,
        "contract_sha256": run["contract_sha256"],
        "labels_read": False,
        "oracle_exact_cache_used": True,
        "attention_cone_representation_gate_passed": representation,
        "full_moment_edge_equal_probability_recovery": float(
            full.edge_equal_probability_recovery
        ),
        "full_moment_minimum_edge_probability_recovery": float(
            full.minimum_edge_probability_recovery
        ),
        "address_R128_edge_equal_probability_recovery": float(
            address128.edge_equal_probability_recovery
        ),
        "chronological_R128_edge_equal_probability_recovery": float(
            chronological128.edge_equal_probability_recovery
        ),
        "address_R128_minus_chronological_R128": address_advantage,
        "sampled_canary_gate_passed": sampled_canary,
        "discovery_launch_gate_passed": launch,
        "sampled_moment_support_gate_passed": sampled_support,
        "smallest_passing_address_sample_count": sampled_selected,
        "cone_diagnostics_edge_equal": cone_summary,
        "interpretation": interpretation,
        "design1_gate": "not_tested_sampled_Current_upper_KV_is_Exact",
    }
    output.mkdir()
    atomic_json(output / "summary.json", result)
    edge_table.to_csv(output / "edge_table.csv", index=False)
    grid.to_csv(output / "grid_table.csv", index=False)
    cone.groupby(["edge", "layer"], as_index=False).mean(numeric_only=True).to_csv(
        output / "cone_by_edge_layer.csv", index=False
    )

    report = [
        f"# Medium user attention-cone response moments: {args.scope}",
        "",
        "The full row is an Exact-state representation oracle. Compact rows keep the complete Parent moment but estimate the Current moment from Exact upper-layer samples; they are constructor oracles, not executable migration actions.",
        "",
        "| method | Current samples | edge-equal recovery | minimum edge | positive edges | >=80% edges | persistent KV ratio | temporary Current-sample ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grid.itertuples(index=False):
        report.append(
            f"| {row.method} | {row.sample_count} | {row.edge_equal_probability_recovery:.4f} | {row.minimum_edge_probability_recovery:.4f} | {row.positive_edges}/5 | {row.edges_at_or_above_0_80}/5 | {row.persistent_storage_ratio_to_current_KV:.4%} | {row.temporary_current_sample_KV_ratio:.4%} |"
        )
    report.extend(
        [
            "",
            "## Adjudication",
            "",
            f"- Full attention-cone representation gate: {'PASS' if representation else 'FAIL'}; recovery {full.edge_equal_probability_recovery:.4f}, minimum edge {full.minimum_edge_probability_recovery:.4f}.",
            f"- Address-R128 canary gate: {'PASS' if sampled_canary else 'FAIL'}; recovery {address128.edge_equal_probability_recovery:.4f}, advantage over chronological {address_advantage:.4f}.",
            f"- Canary-to-discovery launch: {'PASS' if launch else 'FAIL'}.",
            f"- Sampled-moment 80% gate: {'PASS' if sampled_support else 'FAIL'}; smallest passing address R: {sampled_selected}.",
            f"- Interpretation: `{interpretation}`.",
            "- A passing full moment establishes only a functional representation. Exact sampled Current upper-layer state prevents every compact row from admitting Design 1.",
            "- No labels, output fitting, ridge/MLP, confirmation users or executable-compute claims are used.",
            "",
            "## Cone diagnostics (row-equal over user/layer/head)",
            "",
        ]
    )
    for name, value in cone_summary.items():
        report.append(f"- `{name}`: {value:.6f}")
    report.append("")
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
