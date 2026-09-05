#!/usr/bin/env python3
"""Adjudicate the formal paired functional-delta canary.

The only Design-candidate contrast is native causal closure R64 versus its
same-selection, same-carrier Parent-conditioned ablation.  Exact-state rows
are oracles, R128 is diagnostic, and the affine compiler is a disposable
storage/compiler ablation.  None of those can substitute for closure gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "configs/contracts/"
    "yambda500m_medium_legacy_pointwise_insight2_paired_functional_delta_v1.yaml"
)
RESULT_ROOT = (
    ROOT / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/"
    "diagnostic_paired_functional_delta_v1"
)
PRIMARY = "native_causal_closure_R64"
MATCHED_CONTROL = "native_parent_conditioned_R64"
REPRESENTATION_P8 = "representation_full_affine_bulk_P8"
REPRESENTATION_P32 = "representation_full_affine_bulk_P32"
CARRIER_ORACLE_R64 = "carrier_oracle_native_R64"
EXACT_LAYER0_ABLATION = "native_exact_layer0_closure_R64"
AFFINE_COMPILER_R64 = "closure_affine_compiler_R64_P8"
EXPECTED_METHODS = {
    "Current_Reuse",
    REPRESENTATION_P8,
    REPRESENTATION_P32,
    CARRIER_ORACLE_R64,
    "carrier_oracle_native_R128",
    MATCHED_CONTROL,
    "native_parent_conditioned_R128",
    PRIMARY,
    "native_causal_closure_R128",
    EXACT_LAYER0_ABLATION,
    AFFINE_COMPILER_R64,
    "closure_affine_compiler_R128_P8",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def method_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "edge",
        "uid",
        "method",
        "evidence_class",
        "probe_count",
        "carrier_count",
        "probability_gap_recovery",
        "logit_gap_recovery",
        "bernoulli_js_to_exact",
        "top1_agreement",
        "top10_overlap",
        "rank_correlation",
        "persistent_ratio_to_full_KV",
        "materialized_intervention_ratio_to_full_KV",
        "current_exact_for_construction",
        "constructor_is_legal",
        "recursive_causal_closure",
        "theoretical_neural_compute_fraction",
        "theoretical_selection_compute_fraction",
        "theoretical_total_compute_fraction",
        "within_20_percent_total_compute",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise RuntimeError(f"paired-delta metrics missing columns: {sorted(missing)}")
    if set(metrics.method.unique()) != EXPECTED_METHODS:
        raise RuntimeError("paired-delta method set differs")

    edge = (
        metrics.groupby(
            [
                "edge",
                "method",
                "evidence_class",
                "probe_count",
                "carrier_count",
            ],
            as_index=False,
        )
        .agg(
            probability_recovery=("probability_gap_recovery", "mean"),
            logit_recovery=("logit_gap_recovery", "mean"),
            bernoulli_js=("bernoulli_js_to_exact", "mean"),
            top1_agreement=("top1_agreement", "mean"),
            top10_overlap=("top10_overlap", "mean"),
            rank_correlation=("rank_correlation", "mean"),
            persistent_ratio_to_full_KV=("persistent_ratio_to_full_KV", "max"),
            materialized_ratio_to_full_KV=(
                "materialized_intervention_ratio_to_full_KV",
                "max",
            ),
            neural_compute_fraction=(
                "theoretical_neural_compute_fraction",
                "max",
            ),
            selection_compute_fraction=(
                "theoretical_selection_compute_fraction",
                "max",
            ),
            total_compute_fraction=(
                "theoretical_total_compute_fraction",
                "max",
            ),
            users_within_20_percent=(
                "within_20_percent_total_compute",
                lambda values: bool(values.dropna().all()) if len(values.dropna()) else None,
            ),
        )
        .sort_values(["method", "edge"])
    )
    records: list[dict[str, Any]] = []
    for keys, group in edge.groupby(
        ["method", "evidence_class", "probe_count", "carrier_count"],
        sort=False,
    ):
        method, evidence, probes, carriers = keys
        records.append(
            {
                "method": method,
                "evidence_class": evidence,
                "probe_count": int(probes),
                "carrier_count": int(carriers),
                "edge_equal_probability_recovery": float(group.probability_recovery.mean()),
                "minimum_edge_probability_recovery": float(group.probability_recovery.min()),
                "positive_edges": int((group.probability_recovery > 0).sum()),
                "edges_at_or_above_0_70": int((group.probability_recovery >= 0.70).sum()),
                "edges_at_or_above_0_80": int((group.probability_recovery >= 0.80).sum()),
                "edges_at_or_above_0_90": int((group.probability_recovery >= 0.90).sum()),
                "edge_equal_logit_recovery": float(group.logit_recovery.mean()),
                "persistent_ratio_to_full_KV": float(group.persistent_ratio_to_full_KV.max()),
                "materialized_ratio_to_full_KV": float(group.materialized_ratio_to_full_KV.max()),
                "maximum_neural_compute_fraction": (
                    None
                    if group.neural_compute_fraction.isna().all()
                    else float(group.neural_compute_fraction.max())
                ),
                "maximum_selection_compute_fraction": (
                    None
                    if group.selection_compute_fraction.isna().all()
                    else float(group.selection_compute_fraction.max())
                ),
                "maximum_total_compute_fraction": (
                    None
                    if group.total_compute_fraction.isna().all()
                    else float(group.total_compute_fraction.max())
                ),
                "all_users_within_20_percent": (
                    None
                    if group.users_within_20_percent.isna().all()
                    else bool(group.users_within_20_percent.fillna(False).all())
                ),
            }
        )
    return edge, pd.DataFrame(records).sort_values(
        ["evidence_class", "carrier_count", "probe_count"]
    )


def _one(grid: pd.DataFrame, method: str) -> pd.Series:
    rows = grid[grid.method == method]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one aggregate row for {method}")
    return rows.iloc[0]


def matched_edge_comparisons(edge: pd.DataFrame) -> pd.DataFrame:
    selected = edge[
        edge.method.isin(
            [
                PRIMARY,
                MATCHED_CONTROL,
                EXACT_LAYER0_ABLATION,
                AFFINE_COMPILER_R64,
            ]
        )
    ][["edge", "method", "probability_recovery"]]
    pivot = selected.pivot(
        index="edge", columns="method", values="probability_recovery"
    ).reset_index()
    required = {PRIMARY, MATCHED_CONTROL, EXACT_LAYER0_ABLATION, AFFINE_COMPILER_R64}
    if required - set(pivot.columns):
        raise RuntimeError("paired-delta matched R64 comparison is incomplete")
    pivot["closure_minus_parent_conditioned"] = pivot[PRIMARY] - pivot[MATCHED_CONTROL]
    pivot["closure_minus_exact_layer0_ablation"] = pivot[PRIMARY] - pivot[EXACT_LAYER0_ABLATION]
    pivot["affine_compiler_minus_native_closure"] = pivot[AFFINE_COMPILER_R64] - pivot[PRIMARY]
    return pivot


def paired_user_edge_bootstrap(
    metrics: pd.DataFrame,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    """Bootstrap the matched closure gain over user-edge pairs."""

    if resamples < 1000:
        raise ValueError("paired bootstrap requires at least 1000 resamples")
    selected = metrics[metrics.method.isin([PRIMARY, MATCHED_CONTROL])][
        ["edge", "uid", "method", "probability_gap_recovery"]
    ]
    paired = selected.pivot(
        index=["edge", "uid"], columns="method", values="probability_gap_recovery"
    )
    if paired.isna().any().any() or len(paired) * 2 != len(selected):
        raise RuntimeError("closure bootstrap pairs are incomplete or duplicated")
    delta = (paired[PRIMARY] - paired[MATCHED_CONTROL]).to_numpy(dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 1000
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        indices = generator.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "seed": seed,
        "resamples": resamples,
        "pairs": len(delta),
        "observed_mean": float(delta.mean()),
        "CI95_lower": float(lower),
        "CI95_upper": float(upper),
    }


def adjudicate(
    *,
    grid: pd.DataFrame,
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    contract: dict[str, Any],
    bootstrap: dict[str, float | int],
) -> dict[str, Any]:
    representation8 = _one(grid, REPRESENTATION_P8)
    representation32 = _one(grid, REPRESENTATION_P32)
    carrier_oracle = _one(grid, CARRIER_ORACLE_R64)
    primary = _one(grid, PRIMARY)
    control = _one(grid, MATCHED_CONTROL)
    compiler = _one(grid, AFFINE_COMPILER_R64)
    exact_layer0 = _one(grid, EXACT_LAYER0_ABLATION)

    representation_rule = contract["gates"]["representation_support"]
    probe_difference = abs(
        float(representation8.edge_equal_probability_recovery)
        - float(representation32.edge_equal_probability_recovery)
    )
    representation_gate = bool(
        representation8.edge_equal_probability_recovery
        >= float(representation_rule["P8_edge_equal_recovery_at_least"])
        and representation8.minimum_edge_probability_recovery
        >= float(representation_rule["P8_minimum_edge_recovery_at_least"])
        and representation8.positive_edges >= int(representation_rule["P8_positive_edges_minimum"])
        and probe_difference <= float(representation_rule["P8_P32_absolute_difference_at_most"])
    )

    oracle_rule = contract["gates"]["carrier_oracle_support"]
    oracle_gate = bool(
        carrier_oracle.edge_equal_probability_recovery
        >= float(oracle_rule["R64_edge_equal_recovery_at_least"])
        and carrier_oracle.positive_edges >= int(oracle_rule["R64_positive_edges_minimum"])
    )

    closure_rule = contract["gates"]["causal_closure_design_candidate"]
    closure_gain = float(comparison.closure_minus_parent_conditioned.mean())
    closure_gain_minimum_edge = float(comparison.closure_minus_parent_conditioned.min())
    closure_winning_edges = int((comparison.closure_minus_parent_conditioned > 0).sum())
    all_position_sets_within_budget = bool(
        diagnostics.R64_closure_within_20_for_all_position_sets.all()
    )
    budget_gate = bool(
        primary.maximum_total_compute_fraction
        <= float(closure_rule["R64_total_compute_fraction_at_most"])
        and primary.all_users_within_20_percent
        and all_position_sets_within_budget
    )
    closure_quality_gate = bool(
        primary.edge_equal_probability_recovery
        >= float(closure_rule["R64_edge_equal_recovery_at_least"])
        and primary.positive_edges >= int(closure_rule["R64_positive_edges_minimum"])
    )
    closure_mechanism_gate = bool(
        closure_gain >= float(closure_rule["R64_closure_minus_control_at_least"])
        and closure_winning_edges >= int(closure_rule["R64_closure_winning_edges_minimum"])
        and float(bootstrap["CI95_lower"])
        > float(closure_rule["paired_bootstrap_CI_lower_strictly_above"])
    )
    design_candidate_gate = bool(
        representation_gate
        and oracle_gate
        and closure_quality_gate
        and closure_mechanism_gate
        and budget_gate
    )

    compiler_loss = float(
        compiler.edge_equal_probability_recovery - primary.edge_equal_probability_recovery
    )
    exact_layer0_difference = float(
        primary.edge_equal_probability_recovery - exact_layer0.edge_equal_probability_recovery
    )
    if design_candidate_gate:
        interpretation = "causal_functional_delta_closure_supported_as_design_candidate"
    elif representation_gate and oracle_gate and not closure_mechanism_gate:
        interpretation = "functional_delta_supported_but_recursive_closure_has_no_distinct_gain"
    elif representation_gate and oracle_gate and not budget_gate:
        interpretation = "causal_closure_promising_but_outside_primary_compute_budget"
    elif representation_gate:
        interpretation = "functional_representation_supported_constructor_not_admitted"
    else:
        interpretation = "paired_functional_boundary_not_supported"

    return {
        "representation_gate_passed": representation_gate,
        "representation_P8_edge_equal_recovery": float(
            representation8.edge_equal_probability_recovery
        ),
        "representation_P8_minimum_edge_recovery": float(
            representation8.minimum_edge_probability_recovery
        ),
        "representation_P8_P32_absolute_difference": probe_difference,
        "carrier_oracle_gate_passed": oracle_gate,
        "carrier_oracle_R64_edge_equal_recovery": float(
            carrier_oracle.edge_equal_probability_recovery
        ),
        "closure_quality_gate_passed": closure_quality_gate,
        "closure_mechanism_gate_passed": closure_mechanism_gate,
        "closure_R64_edge_equal_recovery": float(primary.edge_equal_probability_recovery),
        "parent_conditioned_R64_edge_equal_recovery": float(
            control.edge_equal_probability_recovery
        ),
        "closure_minus_parent_conditioned_edge_equal": closure_gain,
        "closure_minus_parent_conditioned_minimum_edge": closure_gain_minimum_edge,
        "closure_winning_edges": closure_winning_edges,
        "closure_gain_paired_bootstrap": bootstrap,
        "R64_budget_gate_passed": budget_gate,
        "R64_maximum_neural_compute_fraction": float(primary.maximum_neural_compute_fraction),
        "R64_maximum_selection_compute_fraction": float(primary.maximum_selection_compute_fraction),
        "R64_maximum_total_compute_fraction": float(primary.maximum_total_compute_fraction),
        "R64_all_users_within_20_percent": bool(primary.all_users_within_20_percent),
        "R64_all_unique_position_sets_within_20_percent": (all_position_sets_within_budget),
        "R64_persistent_incremental_ratio_to_full_KV": float(primary.persistent_ratio_to_full_KV),
        "exact_layer0_ablation_edge_equal_recovery": float(
            exact_layer0.edge_equal_probability_recovery
        ),
        "paired_closure_minus_exact_layer0_ablation": exact_layer0_difference,
        "affine_compiler_R64_edge_equal_recovery": float(compiler.edge_equal_probability_recovery),
        "affine_compiler_minus_native_closure": compiler_loss,
        "causal_closure_design_candidate_gate_passed": design_candidate_gate,
        "interpretation": interpretation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary",), default="canary")
    args = parser.parse_args()
    source = RESULT_ROOT / args.scope
    output = source / "analysis"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    run = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if not run.get("passed"):
        raise RuntimeError("cannot adjudicate failed paired-delta instrumentation")
    if not run.get("recursive_causal_closure_implemented"):
        raise RuntimeError("run does not contain the recursive causal closure")
    if run.get("legal_Parent_conditioned_path_uses_Current_Exact"):
        raise RuntimeError("legal paired-delta path used Current Exact")
    contract_hash = sha256_file(CONTRACT)
    if run.get("contract_sha256") != contract_hash:
        raise RuntimeError("paired-delta run and contract hashes differ")
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    metrics = pd.read_parquet(source / "metrics.parquet")
    diagnostics = pd.read_parquet(source / "diagnostics.parquet")
    edge, grid = method_tables(metrics)
    comparison = matched_edge_comparisons(edge)
    closure_rule = contract["gates"]["causal_closure_design_candidate"]
    bootstrap = paired_user_edge_bootstrap(
        metrics,
        seed=int(closure_rule["paired_bootstrap_seed"]),
        resamples=int(closure_rule["paired_bootstrap_resamples"]),
    )
    decision = adjudicate(
        grid=grid,
        comparison=comparison,
        diagnostics=diagnostics,
        contract=contract,
        bootstrap=bootstrap,
    )

    result = {
        "status": "paired_functional_delta_canary_adjudicated",
        "scope": args.scope,
        "contract_sha256": contract_hash,
        "labels_read": False,
        "construction_candidates_read": False,
        "Current_Exact_in_legal_construction": False,
        "primary_budget_point": "R64",
        "R128_role": "diagnostic_only_never_selected",
        "primary_layer0_prefix_mode": "paired_closure",
        "exact_Current_layer0_prefix_role": "consistency_ablation_only",
        "moments_sampling_or_clustering_is_novelty": False,
        "novelty_status": "causal_closure_mechanism_requires_separate_prior_art_audit",
        "design1_status": (
            "causal_closure_candidate_supported_not_paper_frozen"
            if decision["causal_closure_design_candidate_gate_passed"]
            else "causal_closure_candidate_not_admitted"
        ),
        "discovery_launch_gate_passed": False,
        "discovery_blocker": "new_contract_and_resource_estimate_required_after_canary",
        **decision,
    }
    output.mkdir()
    atomic_json(output / "summary.json", result)
    edge.to_csv(output / "edge_table.csv", index=False)
    grid.to_csv(output / "grid_table.csv", index=False)
    comparison.to_csv(output / "matched_R64_comparison.csv", index=False)
    diagnostics.describe(include="all").to_csv(output / "diagnostic_summary.csv")

    report = [
        "# Medium paired functional-delta canary",
        "",
        "The primary Design contrast is `native_causal_closure_R64` versus `native_parent_conditioned_R64`. They share the model, users, carrier positions, masses, Parent control and native serving reader; only causal propagation of earlier functional deltas changes.",
        "",
        "| method | evidence type | edge-equal recovery | minimum edge | positive edges | >=80% edges | incremental KV ratio | maximum total compute |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grid.itertuples(index=False):
        compute = (
            "n/a"
            if row.maximum_total_compute_fraction is None
            or pd.isna(row.maximum_total_compute_fraction)
            else f"{row.maximum_total_compute_fraction:.4%}"
        )
        report.append(
            f"| {row.method} | {row.evidence_class} | {row.edge_equal_probability_recovery:.4f} | {row.minimum_edge_probability_recovery:.4f} | {row.positive_edges}/5 | {row.edges_at_or_above_0_80}/5 | {row.persistent_ratio_to_full_KV:.4%} | {compute} |"
        )
    report.extend(
        [
            "",
            "## Adjudication",
            "",
            f"- Functional representation: {'PASS' if decision['representation_gate_passed'] else 'FAIL'}.",
            f"- Exact-carrier oracle: {'PASS' if decision['carrier_oracle_gate_passed'] else 'FAIL'}.",
            f"- R64 closure quality: {'PASS' if decision['closure_quality_gate_passed'] else 'FAIL'}; recovery {decision['closure_R64_edge_equal_recovery']:.4f}.",
            f"- Causal-closure mechanism: {'PASS' if decision['closure_mechanism_gate_passed'] else 'FAIL'}; gain over matched independent carriers {decision['closure_minus_parent_conditioned_edge_equal']:.4f}, wins {decision['closure_winning_edges']}/5 edges.",
            f"- Paired user-edge bootstrap: 95% CI [{decision['closure_gain_paired_bootstrap']['CI95_lower']:.4f}, {decision['closure_gain_paired_bootstrap']['CI95_upper']:.4f}] from {decision['closure_gain_paired_bootstrap']['resamples']} fixed-seed resamples.",
            f"- Honest R64 compute: {'PASS' if decision['R64_budget_gate_passed'] else 'FAIL'}; neural {decision['R64_maximum_neural_compute_fraction']:.4%}, selection {decision['R64_maximum_selection_compute_fraction']:.4%}, total {decision['R64_maximum_total_compute_fraction']:.4%}.",
            f"- Design-candidate gate: {'PASS' if decision['causal_closure_design_candidate_gate_passed'] else 'FAIL'}.",
            f"- Interpretation: `{decision['interpretation']}`.",
            "- R128 is diagnostic and cannot be selected even if its quality is higher.",
            "- The affine compiler may reduce persistent storage, but moments, sampling and landmark selection are not treated as novelty.",
            "- Passing this canary does not freeze Design 1 or authorize discovery; a resource estimate and a new prospective contract remain required.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
