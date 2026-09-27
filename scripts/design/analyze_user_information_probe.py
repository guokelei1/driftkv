#!/usr/bin/env python3
"""CPU-only arithmetic ledger for the retained user-information Insight probe.

Reads metadata only: no checkpoint loading, fitting, model execution, or timing
conversion. The ledger estimates scalar arithmetic, not hardware instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from design.report_native_flops import D, H, L, prefix_literal, query_ops, rebuild, value


def correction(users, queries, use_response, batch_size):
    """Literal prepare_layer + apply_prepared_layer, for ONE layer.

    Charge native-weight folding once per batch, as the runner does. Candidate
    normalization costs two operations per element; a multiply-add costs two.
    Views/copies, comparisons, dtype/device conversions and lookup are excluded.
    """
    elements = users * queries * D
    parts = {
        "candidate_normalize": 2 * elements,
        "candidate_linear": 2 * elements * D,
        "candidate_bias": elements,
        "delta_scale_and_add": 2 * elements,
    }
    if use_response:
        parts.update(
            native_weight_fold=3 * math.ceil(users / batch_size) * D * D,
            native_center_subtract=elements,
            native_divide_by_count=elements,
            native_linear=2 * elements * D,
            native_add=elements,
        )
    return parts


def ridge(users, queries, use_response):
    """ONE FP64 fit_layer: explicit dense products and approximate scalars.

    LU factorization uses 2/3 p^3 and the D right-hand sides use 2 p^2 D.
    Both explicit system@weights evaluations are charged, as are prediction
    and error diagnostics. Reduction/normalization counts are an estimate,
    independent of PyTorch's reduction algorithm or vectorized instruction set.
    """
    rows = users * queries
    features = D * (2 if use_response else 1)
    p = features + 1
    return {
        "mean_std_normalize_approx": 8 * rows * features + 4 * features,
        "gram": 2 * rows * p * p,
        "rhs": 2 * rows * p * D,
        "gram_rhs_scale_and_regularize": 3 * p * p + p * D,
        "lu_and_solve_approx": (2 / 3) * p**3 + 2 * p * p * D,
        "two_residual_products": 4 * p * p * D,
        "fitting_prediction": 2 * rows * p * D,
        "fitting_and_residual_scalar_diagnostics_approx": 5 * rows * D + 8 * p * D,
    }


def analyze(run, output):
    settings = json.loads((run / "configuration.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    config = settings["config"]
    assert summary["status"] == "completed" and config["probe_kind"] == "user_information"
    assert settings["backend"] == "torch", "prefix_literal models the retained eager graph"
    n = config["history_length"]
    calibration = len(config["calibration_uids"])
    fit_queries = len(config["calibration_candidate_indices"])
    serve_queries = len(config["evaluation_candidate_indices"])
    batch = settings["batch_size"]
    assert not set(config["calibration_uids"]) & set(config["evaluation_uids"])
    for edge in summary["edges"]:
        model = edge["model_pair"]["config"]
        assert (model["hidden_size"], model["num_layers"], model["num_heads"]) == (D, L, H)
        assert model["block_variant"] == "legacy" and model["activation"] == "elu_plus1"
        assert model["gating"] == "silu_gate" and not model["relative_position_bias"]
        assert edge["calibration_users"] == calibration
        for name, use_response in (("shared_candidate", False), ("shared_user_response", True)):
            layers = edge["fitting"][name]["layers"]
            assert len(layers) == L
            assert all(row["shared_parameters"] == (D * (2 if use_response else 1) + 1) * D
                       and row["fitting_queries_per_user"] == fit_queries for row in layers)

    c_rebuild = value(rebuild(n))
    c_literal = value(prefix_literal(n))
    c_native = value(query_ops(n, serve_queries))
    branches = {}
    for name, use_response in (("shared_candidate", False), ("shared_user_response", True)):
        # Each fit layer recomputes the full reader, then reads only this layer's
        # teacher K/V at the branch's query. Previously fitted layers are applied.
        parts = {
            "both_calibration_cache_prefills_literal": 2 * calibration * c_literal,
            "six_complete_source_reader_passes": L * calibration * value(query_ops(n, fit_queries)),
            "six_single_layer_teacher_reads": L * calibration * fit_queries * (4 * n * D + 3 * n * H),
            "target_and_observed_rate_preprocessing": L * 3 * calibration * fit_queries * D,
            "previously_fitted_layer_corrections": L * (L - 1) // 2
                * sum(correction(calibration, fit_queries, use_response, batch).values()),
            "six_ridge_fits": L * sum(ridge(calibration, fit_queries, use_response).values()),
        }
        total = sum(parts.values())
        # Deliberately do not assume batching amortizes the per-call native fold.
        extra = L * sum(correction(1, serve_queries, use_response, 1).values())
        population = summary["edges"][0]["model_pair"]["population_users"]
        branches[name] = {
            "calibration_parts": parts,
            "calibration_total_K": total,
            "one_layer_ridge_detail": ridge(calibration, fit_queries, use_response),
            "one_layer_one_user_correction_detail": correction(1, serve_queries, use_response, 1),
            "extra_per_user_C_read": extra,
            "extra_read_over_cache_only_rebuild": extra / c_rebuild,
            "extra_read_over_native_32candidate_read": extra / c_native,
            "shared_weights": L * (D * (2 if use_response else 1) + 1) * D,
            "arithmetic_break_even_users": math.ceil(total / (c_rebuild - extra)),
            "conditional_population": {
                "users_U": population,
                "shared_K_plus_U_C_read": total + population * extra,
                "exact_U_C_rebuild": population * c_rebuild,
                "ratio": (total + population * extra) / (population * c_rebuild),
                "assumption": f"all U users have N={n} and each scores Q={serve_queries} candidates once",
                "measured_population_execution": False,
            },
        }
    report = {
        "scope": "Arithmetic estimate per release; not a hardware profiler or population measurement.",
        "unit": "scalar arithmetic; multiply-add=2; ELU/SiLU/rsqrt/sincos each assigned 1",
        "shapes": dict(layers=L, width=D, heads=H, history=n, calibration_users=calibration,
                       calibration_candidates=fit_queries, serving_candidates=serve_queries,
                       calibration_batch=batch, serving_batch_assumption=1),
        "one_user": {"dependency_closed_cache_rebuild": c_rebuild,
                     "literal_torch_compute_kv": c_literal,
                     "native_candidate_read": c_native},
        "formula": "incremental shared cost K + U*C_read versus exact U*C_rebuild; native reads cancel",
        "branches": branches,
        "accounting_notes": [
            "K includes constructing both Parent and Current calibration caches, even if Parent is already available.",
            "Each branch is accounted independently; the actual two-arm experiment reuses the two calibration caches.",
            "K charges six complete source reads, six one-layer teacher reads and all 15 prior-layer correction applications.",
            "The exact denominator is cache-only dependency closure (five complete causal blocks, last-layer norm and K/V), not the larger literal eager prefill.",
            "Serving C_read includes correction preparation and application only; ordinary native reads remain necessary.",
            "No evaluation-user teacher/oracle is needed to generate either shared prediction. Diagnostic Exact/oracle measurements are research costs, excluded from serving and calibration K.",
            "No data I/O, memory traffic, embedding lookup, copies, comparisons, dtype/device conversion, dispatch, or timings are converted to FLOPs.",
            "Ridge reduction/scalar counts and LU are arithmetic estimates; FP32 model and FP64 fitting arithmetic are summed without hardware weighting.",
            "The conditional 30,000-user calculation is shape extrapolation only. Repeated reads accumulate C_read; no lifecycle or measured population-speed claim follows.",
        ],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (run / "configuration.json", run / "summary.json", Path(__file__),
                                    ROOT / "scripts/design/report_native_flops.py")},
        "observed_timing_seconds": [
            dict(edge=e["edge"], calibration_cache_prefill=e["costs"]["fit_source_prefill_seconds"]
                 + e["costs"]["fit_teacher_prefill_seconds"],
                 fitting={name: {key: value for key, value in item.items() if key.endswith("seconds")}
                          for name, item in e["fitting"].items()},
                 score_paths=e["score_path_seconds"])
            for e in summary["edges"]
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "cost.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Read-correction diagnostic arithmetic", "",
             "This is an operation-count estimate from the retained runner and tensor shapes. It is not a population benchmark.", "",
             f"Six layers, width 192, history {n}; {calibration} calibration users × {fit_queries} candidates; {serve_queries} serving candidates per user.", "",
             f"Cache-only rebuild: {c_rebuild / 1e9:.4f} G operations per user; literal calibration prefill: {c_literal / 1e9:.4f} G; native 32-candidate read: {c_native / 1e6:.3f} M.", "",
             "| Shared predictor | Calibration K (T ops) | Extra read/user (M ops) | Extra/rebuild | Conditional U=30,000 ratio |",
             "|---|---:|---:|---:|---:|"]
    for name, row in branches.items():
        lines.append(f"| {name} | {row['calibration_total_K'] / 1e12:.4f} | {row['extra_per_user_C_read'] / 1e6:.3f} | {100 * row['extra_read_over_cache_only_rebuild']:.3f}% | {100 * row['conditional_population']['ratio']:.3f}% |")
    lines += ["", "The conditional ratio is `(K + U*C_read)/(U*C_rebuild)`, assuming every user has 1,024 retained events and scores 32 candidates once. It includes both calibration prefills, response acquisition, previous-layer corrections, and ridge fitting. It excludes the native read common to both paths. It does not establish measured population FLOPs, speed, repeated-read amortization, or useful quality for the candidate-only arm.", "",
              "Multiply-add counts as two; special functions count as one. Dense products follow the eager runner; reduction and LU costs are estimates. Memory traffic, transfers and dispatch are excluded. Evaluation-user teachers and per-user oracle fitting are research measurements, not inputs needed to serve either shared rule.", "",
              "Reproduce: `PYTHONPATH=src:scripts python scripts/design/analyze_user_information_probe.py`.", ""]
    (output / "cost.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/insight/user_information_6l_20260920/diagnostic")
    parser.add_argument("--output", type=Path, default=ROOT / "results/insight/user_information_6l_20260920/analysis")
    args = parser.parse_args()
    analyze(args.run.resolve(), args.output.resolve())
