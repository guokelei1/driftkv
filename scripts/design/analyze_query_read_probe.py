#!/usr/bin/env python3
"""Metadata-only arithmetic estimate for shared b+Aq and b+Aq+Tr probes.

Includes the pilot's one-time calibration even when evaluation loads its rules.
No checkpoint or rules loading, fitting, backbone execution, or timing conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from design.report_native_flops import D, H, L, d, prefix_literal, query_ops, rebuild, value


def correction(users, queries, use_response):
    """ONE layer's literal prepare_layer and callback; no folded optimization.

    prepare_layer constructs the same shared coefficients separately for each
    user through its existing all-ones latent. Count these repeated operations.
    """
    elements = users * queries * D
    result = {
        "shared_coefficients_expanded_per_user": 2 * users * H * (d + 1) * d,
        "slope_scale": users * H * d * d,
        "query_center_times_slope": 2 * users * H * d * d,
        "intercept_subtract": users * D,
        "query_times_slope": 2 * elements * d,
        "query_intercept_add": elements,
        "delta_scale_and_add": 2 * elements,
    }
    if use_response:
        result.update(
            native_rate_center_scale=3 * elements,
            native_times_shared_weights=2 * elements * D,
            native_term_add=elements,
        )
    return result


def fit_joint_cost(users, queries, use_response):
    """ONE retained fit_joint solve, rank=1 and read_dim=0 or D.

    Count explicit products including both residual evaluations and fitting
    prediction. Three-operand contractions use left-to-right contraction, as
    in this environment without opt_einsum; reductions and LU are estimates.
    Empty read columns perform no floating arithmetic.
    """
    m, j, a = users * queries, d + 1, D if use_response else 0
    p = j + a
    return {
        "query_and_native_statistics_normalization_approx": 8 * m * (D + a) + 4 * (D + a),
        "count_weights_and_latent_outer_approx": 6 * users + 1,
        "qgram": 2 * m * H * j * j,
        "qgram_scale": users * H * j * j,
        "gram_contract": 2 * users * H * j * j,
        "qty": 2 * m * H * j * d,
        "qty_scale_and_rhs_contract": 3 * users * H * j * d,
        "qta": 2 * m * H * j * a,
        "qta_scale_and_cross_contract": 3 * users * H * j * a,
        "read_gram": 2 * m * a * a + 2 * users * a * a + a * a,
        "read_rhs": 2 * m * a * D + 2 * users * a * D + a * D,
        "system_regularization": (H + 1) * p * p,
        "lu_and_solve_approx": H * ((2 / 3) * p**3 + 2 * p * p * d),
        "two_explicit_residual_products": 4 * H * p * p * d,
        "query_fitting_prediction": m * H * j + 2 * m * H * j * d,
        "native_fitting_prediction": 2 * m * a * D,
        "prediction_add": m * D,
        "error_and_ridge_diagnostics_approx": 8 * m * D + 8 * H * p * d + 2 * users + 4,
    }


def main(args):
    pilot = args.calibration_run.resolve()
    run = args.run.resolve()
    settings = json.loads((pilot / "configuration.json").read_text())
    calibration_result = json.loads((pilot / "summary.json").read_text())
    result = json.loads((run / "summary.json").read_text())
    full_settings = json.loads((run / "configuration.json").read_text())
    assert calibration_result["status"] == result["status"] == "completed"
    assert calibration_result["mode"] == "pilot" and result["mode"] == "diagnostic"
    assert settings["config_sha256"] == full_settings["config_sha256"]
    assert settings["source_sha256"] == full_settings["source_sha256"]
    config = settings["config"]
    assert settings["backend"] == "torch", "The literal prefill ledger models torch eager."
    n = config["history_length"]
    fit_users = len(config["calibration_uids"])
    fit_queries = len(config["calibration_candidate_indices"])
    queries = len(config["evaluation_candidate_indices"])
    users = len(full_settings["config"]["evaluation_uids"])
    assert users == 2560 and fit_users == 256 and fit_queries == 16 and queries == 32
    assert not set(config["calibration_uids"]) & set(full_settings["config"]["evaluation_uids"])
    for edge in calibration_result["edges"]:
        full_edge = next(row for row in result["edges"] if row["edge"] == edge["edge"])
        assert full_edge["calibration_rules_sha256"] == edge["calibration_rules_sha256"]
        assert full_edge["calibration_teacher_users_this_run"] == 0
        assert full_edge["calibration_reused_from"]
        assert full_edge["original_calibration_costs"] == edge["original_calibration_costs"]
        model = edge["model_pair"]["config"]
        assert (model["hidden_size"], model["num_layers"], model["num_heads"]) == (D, L, H)
        assert model["block_variant"] == "legacy" and model["activation"] == "elu_plus1"
        assert not model["relative_position_bias"] and model["gating"] == "silu_gate"
        for method, conditioned in (("shared_query", False), ("shared_query_response", True)):
            layers = edge["fitting"][method]["layers"]
            expected = H * (d + 1 + (D if conditioned else 0)) * d
            assert len(layers) == L and all(row["shared_parameters"] == expected for row in layers)

    baseline = {
        "dependency_closed_rebuild_per_user": value(rebuild(n)),
        "literal_eager_cache_prefill_per_user": value(prefix_literal(n)),
        "ordinary_read_per_user": value(query_ops(n, queries)),
    }
    branches = {}
    for method, conditioned in (("shared_query", False), ("shared_query_response", True)):
        parts = {
            "calibration_source_cache_prefill": fit_users * baseline["literal_eager_cache_prefill_per_user"],
            "calibration_teacher_cache_prefill": fit_users * baseline["literal_eager_cache_prefill_per_user"],
            "six_complete_source_reader_passes": L * fit_users * value(query_ops(n, fit_queries)),
            "six_single_layer_teacher_reads": L * fit_users * fit_queries * (4 * n * D + 3 * n * H),
            "target_and_observed_rate_preprocessing": L * 3 * fit_users * fit_queries * D,
            "fifteen_fitted_prefix_layer_corrections": L * (L - 1) // 2
                * sum(correction(fit_users, fit_queries, conditioned).values()),
            "six_joint_ridge_fits": L * sum(fit_joint_cost(fit_users, fit_queries, conditioned).values()),
        }
        k = sum(parts.values())
        extra = L * sum(correction(1, queries, conditioned).values())
        rebuild_one = baseline["dependency_closed_rebuild_per_user"]
        native_one = baseline["ordinary_read_per_user"]
        branches[method] = {
            "calibration_parts": parts,
            "pilot_calibration_K": k,
            "one_layer_fit_detail": fit_joint_cost(fit_users, fit_queries, conditioned),
            "one_layer_correction_detail": correction(1, queries, conditioned),
            "additional_read_per_user_C_read": extra,
            "additional_over_rebuild": extra / rebuild_one,
            "additional_over_ordinary_read": extra / native_one,
            "shared_parameter_count": L * H * (d + 1 + (D if conditioned else 0)) * d,
            "arithmetic_break_even_users": math.ceil(k / (rebuild_one - extra)),
            "one_32candidate_group_per_2560_users": {
                "total_additional_reads": users * extra,
                "calibration_plus_additional_reads": k + users * extra,
                "all_user_exact_rebuilds": users * rebuild_one,
                "incremental_compatibility_ratio": (k + users * extra) / (users * rebuild_one),
                "including_ordinary_reads_ratio": (k + users * (extra + native_one))
                    / (users * (rebuild_one + native_one)),
            },
        }
    files = (pilot / "configuration.json", pilot / "summary.json", run / "configuration.json",
             run / "summary.json", Path(__file__), ROOT / "scripts/design/query_read_probe.py",
             ROOT / "scripts/design/run_query_read_probe.py", ROOT / "scripts/design/shared_read_probe.py",
             ROOT / "scripts/design/diagnose_native_input.py", ROOT / "scripts/design/report_native_flops.py")
    ledger = {
        "scope": "Per-release scalar arithmetic estimate, not timing-derived FLOPs or an end-to-end speed benchmark.",
        "unit": "multiply-add=2; ELU, SiLU, rsqrt, sin/cos each assigned one; FP32 and FP64 operations summed",
        "shapes": dict(layers=L, heads=H, head_width=d, width=D, history=n,
                       calibration_users=fit_users, calibration_candidates=fit_queries,
                       evaluation_users=users, serving_candidates=queries),
        "baseline": baseline, "branches": branches,
        "calibration_accounting": "Charge the pilot calibration once per branch per release. The full run loads pilot rules; its zero repeated fit cost does not eliminate pilot K. Both branches share cache prefills in the actual two-arm research run, but each standalone K conservatively includes both.",
        "teacher_accounting": "K includes Current prefill on 256 calibration users and six one-layer teacher reads. Shared serving uses no evaluation-user teacher. Evaluation-user Exact measurements are research costs, excluded from K and serving. This diagnostic does not run a per-user oracle.",
        "evaluation_only_exact_measurement_per_release": {
            "users": users,
            "current_cache_prefills": users * baseline["literal_eager_cache_prefill_per_user"],
            "current_cache_endpoint_reads": users * baseline["ordinary_read_per_user"],
            "scope": "Diagnostic Exact reference, not input to either shared rule. Excludes the small first-batch reader canary and pilot evaluation.",
        },
        "limits": [
            "All 2560 users are assumed to retain 1024 events and score 32 candidates once. This is an arithmetic calculation on observed shapes, not measured population FLOPs.",
            "The denominator uses dependency-closed cache reconstruction (five full causal blocks, final norm/KV), below the actual eager compute_kv cost charged for calibration.",
            "Ordinary native reads are required by both paths. C_read is only the extra correction preparation/application. Repeated requests accumulate this cost.",
            "Query-only shared parameters still receive higher-layer queries that depend on user history; this arm is not user-information-free.",
            "All-ones latent expansion and repeated per-user coefficient preparation are charged as executed; no unimplemented folding or caching optimization is assumed.",
            "Matrix products dominate. Statistics, reductions, scalar diagnostics and LU are estimates; memory traffic, I/O, copies, lookup, conversion, comparisons and launch overhead are excluded.",
        ],
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cost.json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    lines = ["# Shared query/read correction arithmetic", "",
             "CPU metadata-only estimate per release. Pilot calibration is charged once even though full evaluation reuses its parameters.", "",
             f"Shapes: 6 layers, 6 heads × 32, 1,024 history events; {fit_users} calibration users × {fit_queries} candidates; {users} evaluation users × {queries} candidates.", "",
             f"One cache-only rebuild: {baseline['dependency_closed_rebuild_per_user'] / 1e9:.4f} G operations. One eager calibration prefill: {baseline['literal_eager_cache_prefill_per_user'] / 1e9:.4f} G. Ordinary 32-candidate read: {baseline['ordinary_read_per_user'] / 1e6:.3f} M.", "",
             "| Predictor | Pilot K (T ops) | Extra/user (M ops) | Extra/native read | Extra/rebuild | (K + 2560 extra)/(2560 rebuild) |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, row in branches.items():
        lines.append(f"| {method} | {row['pilot_calibration_K'] / 1e12:.4f} | {row['additional_read_per_user_C_read'] / 1e6:.3f} | {100 * row['additional_over_ordinary_read']:.3f}% | {100 * row['additional_over_rebuild']:.3f}% | {100 * row['one_32candidate_group_per_2560_users']['incremental_compatibility_ratio']:.3f}% |")
    lines += ["", ledger["teacher_accounting"], "", "The last column counts additional compatibility computation, with one 32-candidate group per user. It is not an end-to-end time ratio. Native reading remains necessary; the JSON also reports the arithmetic ratio when ordinary reads are included. Repeated requests add correction costs.", "",
              "Calibration includes both cache prefills, six full source reads, six one-layer teacher reads, 15 prior-layer corrections, and all six ridge fits including residual checks and prediction diagnostics. Empty read columns cost no floating arithmetic. Reductions/LU/special functions use the stated approximation; memory and execution overhead are excluded.", "",
              "Reproduce: `PYTHONPATH=src:scripts python scripts/design/analyze_query_read_probe.py`.", ""]
    (args.output / "cost.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    base = ROOT / "results/insight/query_read_6l_2560_20260920"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run", type=Path, default=base / "pilot")
    parser.add_argument("--run", type=Path, default=base / "diagnostic")
    parser.add_argument("--output", type=Path, default=base / "analysis")
    main(parser.parse_args())
