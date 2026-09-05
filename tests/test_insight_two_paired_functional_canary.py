from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from insight_two.adjudicate_paired_functional_delta import (  # noqa: E402
    AFFINE_COMPILER_R64,
    CARRIER_ORACLE_R64,
    EXACT_LAYER0_ABLATION,
    EXPECTED_METHODS,
    MATCHED_CONTROL,
    PRIMARY,
    REPRESENTATION_P8,
    REPRESENTATION_P32,
    adjudicate,
    matched_edge_comparisons,
    method_tables,
    paired_user_edge_bootstrap,
)

EDGES = [f"v{index}_to_v{index + 1}" for index in range(5)]


def _recovery(method: str) -> float:
    return {
        "Current_Reuse": 0.0,
        REPRESENTATION_P8: 0.96,
        REPRESENTATION_P32: 0.955,
        CARRIER_ORACLE_R64: 0.86,
        "carrier_oracle_native_R128": 0.91,
        MATCHED_CONTROL: 0.72,
        "native_parent_conditioned_R128": 0.79,
        PRIMARY: 0.84,
        "native_causal_closure_R128": 0.90,
        EXACT_LAYER0_ABLATION: 0.82,
        AFFINE_COMPILER_R64: 0.80,
        "closure_affine_compiler_R128_P8": 0.85,
    }[method]


def _evidence(method: str) -> str:
    if method == "Current_Reuse":
        return "serving_baseline"
    if method.startswith("representation_"):
        return "representation_oracle"
    if method.startswith("carrier_oracle_"):
        return "carrier_state_oracle"
    if method.startswith("native_parent_"):
        return "legal_independent_ablation"
    if method.startswith("native_causal_"):
        return "legal_recursive_candidate"
    if method == EXACT_LAYER0_ABLATION:
        return "legal_layer0_consistency_ablation"
    return "legal_affine_compiler_ablation"


def _metrics(*, users: int = 8) -> pd.DataFrame:
    rows = []
    for edge in EDGES:
        for uid in range(users):
            for method in EXPECTED_METHODS:
                carriers = (
                    128
                    if "R128" in method
                    else 64
                    if "R64" in method
                    else 1024
                    if "representation" in method
                    else 0
                )
                probes = 32 if method.endswith("P32") else 8 if "affine" in method else 0
                legal_native = method in {
                    MATCHED_CONTROL,
                    "native_parent_conditioned_R128",
                    PRIMARY,
                    "native_causal_closure_R128",
                }
                recursive = method in {
                    PRIMARY,
                    "native_causal_closure_R128",
                    EXACT_LAYER0_ABLATION,
                    AFFINE_COMPILER_R64,
                    "closure_affine_compiler_R128_P8",
                }
                exact = _evidence(method) in {
                    "representation_oracle",
                    "carrier_state_oracle",
                }
                value = _recovery(method)
                rows.append(
                    {
                        "edge": edge,
                        "uid": uid,
                        "method": method,
                        "evidence_class": _evidence(method),
                        "probe_count": probes,
                        "carrier_count": carriers,
                        "probability_gap_recovery": value,
                        "logit_gap_recovery": value - 0.01,
                        "bernoulli_js_to_exact": 0.01,
                        "top1_agreement": 0.8,
                        "top10_overlap": 0.9,
                        "rank_correlation": 0.9,
                        "persistent_ratio_to_full_KV": 0.0626 if carriers == 64 else 0.02,
                        "materialized_intervention_ratio_to_full_KV": 0.125,
                        "current_exact_for_construction": exact,
                        "constructor_is_legal": not exact,
                        "recursive_causal_closure": recursive,
                        "theoretical_neural_compute_fraction": 0.15 if legal_native else None,
                        "theoretical_selection_compute_fraction": 0.03 if legal_native else None,
                        "theoretical_total_compute_fraction": 0.18 if legal_native else None,
                        "within_20_percent_total_compute": True if legal_native else None,
                    }
                )
    return pd.DataFrame(rows)


def _contract() -> dict:
    return {
        "gates": {
            "representation_support": {
                "P8_edge_equal_recovery_at_least": 0.90,
                "P8_minimum_edge_recovery_at_least": 0.80,
                "P8_positive_edges_minimum": 5,
                "P8_P32_absolute_difference_at_most": 0.02,
            },
            "carrier_oracle_support": {
                "R64_edge_equal_recovery_at_least": 0.70,
                "R64_positive_edges_minimum": 4,
            },
            "causal_closure_design_candidate": {
                "R64_edge_equal_recovery_at_least": 0.70,
                "R64_positive_edges_minimum": 4,
                "R64_closure_minus_control_at_least": 0.10,
                "R64_closure_winning_edges_minimum": 4,
                "R64_total_compute_fraction_at_most": 0.20,
                "paired_bootstrap_CI_lower_strictly_above": 0.0,
            },
        }
    }


def _diagnostics() -> pd.DataFrame:
    return pd.DataFrame({"R64_closure_within_20_for_all_position_sets": [True] * 40})


def test_design_gate_requires_matched_causal_closure_gain() -> None:
    metrics = _metrics()
    edge, grid = method_tables(metrics)
    comparison = matched_edge_comparisons(edge)
    bootstrap = paired_user_edge_bootstrap(metrics, seed=17037, resamples=2000)
    decision = adjudicate(
        grid=grid,
        comparison=comparison,
        diagnostics=_diagnostics(),
        contract=_contract(),
        bootstrap=bootstrap,
    )
    assert decision["causal_closure_design_candidate_gate_passed"]
    assert decision["closure_mechanism_gate_passed"]
    assert decision["closure_gain_paired_bootstrap"]["CI95_lower"] > 0

    failed = metrics.copy()
    failed.loc[failed.method == PRIMARY, "probability_gap_recovery"] = 0.72
    edge, grid = method_tables(failed)
    comparison = matched_edge_comparisons(edge)
    bootstrap = paired_user_edge_bootstrap(failed, seed=17037, resamples=2000)
    decision = adjudicate(
        grid=grid,
        comparison=comparison,
        diagnostics=_diagnostics(),
        contract=_contract(),
        bootstrap=bootstrap,
    )
    assert not decision["closure_mechanism_gate_passed"]
    assert not decision["causal_closure_design_candidate_gate_passed"]


def test_R128_and_affine_compiler_cannot_rescue_failed_primary_R64() -> None:
    metrics = _metrics()
    metrics.loc[metrics.method == PRIMARY, "probability_gap_recovery"] = 0.65
    metrics.loc[
        metrics.method.isin(["native_causal_closure_R128", "closure_affine_compiler_R128_P8"]),
        "probability_gap_recovery",
    ] = 0.99
    edge, grid = method_tables(metrics)
    comparison = matched_edge_comparisons(edge)
    bootstrap = paired_user_edge_bootstrap(metrics, seed=17037, resamples=2000)
    decision = adjudicate(
        grid=grid,
        comparison=comparison,
        diagnostics=_diagnostics(),
        contract=_contract(),
        bootstrap=bootstrap,
    )
    assert not decision["closure_quality_gate_passed"]
    assert not decision["causal_closure_design_candidate_gate_passed"]


def test_bootstrap_is_fixed_seed_and_user_edge_paired() -> None:
    metrics = _metrics(users=4)
    first = paired_user_edge_bootstrap(metrics, seed=91, resamples=2000)
    second = paired_user_edge_bootstrap(metrics, seed=91, resamples=2000)
    assert first == second
    assert first["pairs"] == 4 * len(EDGES)
    assert first["CI95_lower"] > 0
