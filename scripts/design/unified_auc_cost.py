"""Pure arithmetic ledger for the fixed-snapshot, six-layer AUC comparison.

No model/data imports or I/O. Multiply-add=2; each ELU, SiLU, rsqrt and
sin/cos evaluation counts as one operation. Cache construction uses causal
dependency closure and ends at the last needed K/V projection. This is useful
theoretical arithmetic, not the current eager executor's instruction count.

Parent caches, retained DroidSpeak entry states and ordinary evaluation reads
are pre-existing/common inputs. Teacher construction, calibration and extra
serving work are charged. Memory requirements are reported separately.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

D, L, H, d = 192, 6, 6, 32
ELEMENT_BYTES = 4


def _embedding(n):
    return 2 * n * (32 * D + D * D) + n * (16 + 2 * D + 32)


def _norm_kv(n):
    return 4 * n * D * D + n * (4 * D + 2)


def _block(n, pairs):
    # Q/K/V/output/gate projections; norm; unnormalized ELU+1 attention;
    # SiLU gate, elementwise product and residual. No final model norm.
    return (10 * n * D * D + 4 * D * pairs
            + n * (4 * D + 2 + 3 * D) + 3 * H * pairs)


def interval_rebuild(n, start, end):
    """K/V in inclusive [start,end], using stored Parent E_start if start>0.

    The final block output is unnecessary for these K/V. For start>0 this is
    a hybrid cache, not the corresponding Current-Exact layer subset.
    """
    count = end - start + 1
    pairs = n * (n + 1) // 2
    return (_embedding(n) if start == 0 else 0) + (count - 1) * _block(n, pairs) + _norm_kv(n)


def exact_rebuild(n):
    return interval_rebuild(n, 0, L - 1)


def tail_rebuild(n, width):
    """All-layer hybrid tail K/V, conditional on retained Parent prefix K/V."""
    m = min(n, width)
    pairs = m * (n - m) + m * (m + 1) // 2
    return _embedding(m) + (L - 1) * _block(m, pairs) + _norm_kv(m)


def native_read(n, queries):
    """Ordinary complete query scoring; used for calibration acquisition only."""
    gemm = queries * (2 * (32 * D + D * D + 30 * D * D) + 4 * L * D * n + 2 * D)
    scalar = queries * (16 + 3 * D + 7 * (4 * D + 1) + L * (6 * D + H + 2 * H * n) + 1)
    special = queries * (L * H * (n + 1) + L * D + 7 + 32)
    return gemm + scalar + special


def teacher_layer_read(n, queries):
    return queries * n * (4 * D + 3 * H)


def translate_apply(n, k):
    """Two full-head kD->D affine maps per layer; never substitute head d."""
    return 4 * L * n * k * D * D + 2 * L * n * D


def centered_ridge(rows, inputs, outputs=D):
    """One kv_translate.fit_affine_ridge, with LU and multiple RHS outputs."""
    p, o = inputs, outputs
    return {
        "centering": 2 * rows * (p + o) + p + o,
        "gram_and_rhs": 2 * rows * p * p + 2 * rows * p * o,
        "regularizer": p,
        "lu_and_solve_approx": (2 / 3) * p**3 + 2 * p * p * o,
        "bias": 2 * p * o + o,
    }


def source_layer_selection(fit_rows, evaluation_rows):
    """All 2*L*L*H single-head OLS probes and R2 evaluations, once.

    Leading QR arithmetic models the full-rank least-squares solve; pivot and
    rank-detection work is excluded. This is not a measured LAPACK cost. The
    caller identifies in-sample or UID-disjoint selector evaluation separately.
    """
    p = o = d
    per_probe = {
        "centering": 2 * fit_rows * (p + o) + p + o,
        "qr_and_rhs_approx": 2 * fit_rows * p * p - (2 / 3) * p**3 + 2 * fit_rows * p * o,
        "triangular_solve_approx": p * p * o,
        "bias": 2 * p * o + o,
        "prediction": 2 * evaluation_rows * p * o + evaluation_rows * o,
        "r2_approx": 7 * evaluation_rows * o + o + 4,
    }
    probes = 2 * L * L * H
    return {key: probes * value for key, value in per_probe.items()}


def shared_ridge(users, queries, use_response):
    """One bias_read_probe.fit_layer: weighted ridge, including its checks."""
    rows, p = users * queries, 1 + (D if use_response else 0)
    return {
        "response_normalization_approx": (8 * rows * D + 4 * D) if use_response else 0,
        "count_weights_and_diagnostics_approx": 6 * users + 2,
        "weighted_design_and_target": rows * (p + D),
        "gram_and_rhs": 2 * rows * p * p + 2 * rows * p * D,
        "divide_and_regularize": 3 * p * p + p * D,
        "lu_and_solve_approx": (2 / 3) * p**3 + 2 * p * p * D,
        "two_residual_products": 4 * p * p * D,
        "fitting_prediction": 2 * rows * p * D,
        "error_diagnostics_approx": 5 * rows * D + 2 * users + 8 * p * D,
    }


def shared_coefficient_fold(use_response):
    """One layer: standardized weights -> raw-response T and intercept."""
    return 3 * D * D + D if use_response else 0


def shared_layer_read(users, queries, use_response):
    """Canonical delta=N*b or N*b+T*r for ONE layer across fixed snapshots.

    `queries` is the total actual query count, not the per-user count. N*b is
    prepared once per snapshot. For the response arm, (r/N)T*N is algebraically
    folded to rT. No identity-matrix fusion or read-result reuse is assumed.
    """
    return users * D + queries * (2 * D * D + 2 * D if use_response else D)


def _result(method, lengths, query_counts, components, metadata):
    if len(lengths) != len(query_counts) or not lengths:
        raise ValueError("need aligned, nonempty per-user history and feedback counts")
    denominator = sum(exact_rebuild(int(n)) for n in lengths)
    total = sum(components.values())
    return {
        "method": method,
        "components": components,
        "extra_ops": total,
        "exact_rebuild_ops": denominator,
        "extra_over_exact": total / denominator,
        "evaluation_users": len(lengths),
        "evaluation_queries": sum(query_counts),
        "ordinary_evaluation_read_ops_not_charged": sum(native_read(int(n), int(q)) for n, q in zip(lengths, query_counts)),
        "metadata": metadata,
        "scope": "Useful theoretical arithmetic per release on one fixed cache per UID; not eager timing, cache-byte coverage, or lifecycle cost.",
    }


def droidspeak_cost(lengths, query_counts, calibration_lengths, interval, fit_queries=16):
    """Each selected interval pays the same complete 21-interval profiler."""
    lengths, query_counts = list(map(int, lengths)), list(map(int, query_counts))
    calibration_lengths = list(map(int, calibration_lengths))
    start, end = map(int, interval)
    intervals = tuple(combinations_with_replacement(range(L), 2))
    profile_users = len(calibration_lengths)
    parts = {
        "profile_teacher_cache": sum(exact_rebuild(n) for n in calibration_lengths),
        "profile_all_21_interval_caches": sum(interval_rebuild(n, a, b) for n in calibration_lengths for a, b in intervals),
        "profile_21_predictions_and_exact_read": 22 * sum(native_read(n, fit_queries) for n in calibration_lengths),
        "profile_logit_mse_approx": 21 * profile_users * (3 * fit_queries + 1),
        "evaluation_interval_refresh": sum(interval_rebuild(n, start, end) for n in lengths),
    }
    return _result("droidspeak", lengths, query_counts, parts, {
        "interval": [start, end], "profile_intervals": [list(x) for x in intervals],
        "calibration_users": profile_users, "teacher_users_over_evaluation_users": profile_users / len(lengths),
        "profile_candidates_per_user": fit_queries,
        "extra_persistent_entry_bytes": (L - 1) * sum(lengths) * D * ELEMENT_BYTES,
        "selected_entry_read_bytes": (sum(lengths) * D * ELEMENT_BYTES) if start else 0,
        "refreshed_kv_bytes": 2 * (end - start + 1) * sum(lengths) * D * ELEMENT_BYTES,
        "entry_state": "All E_1..E_5 were retained during Parent execution; no teacher boundary is supplied.",
        "quality_executor": "Existing complete interval blocks; theoretical cost omits the unused final block output, with identical installed K/V.",
        "selection": "One interval per layer-count budget selected using calibration logit MSE; no evaluation-label selection.",
    })


def tail_cost(lengths, query_counts, width):
    lengths, query_counts = list(map(int, lengths)), list(map(int, query_counts))
    width = int(width)
    tokens = sum(min(n, width) for n in lengths)
    return _result("tail", lengths, query_counts, {
        "evaluation_tail_refresh": sum(tail_rebuild(n, width) for n in lengths),
    }, {
        "tail_width": width, "calibration_users": 0,
        "teacher_users_over_evaluation_users": 0,
        "refreshed_kv_bytes": 2 * L * tokens * D * ELEMENT_BYTES,
        "extra_persistent_entry_bytes": 0,
        "semantics": "Replay real tail events through Current conditioned on Parent prefix K/V; preserves original event time deltas.",
        "quality_executor": "Existing complete tail blocks; theoretical cost omits unused final block output.",
    })


def translate_cost(lengths, query_counts, calibration_lengths, k, *, fitting_rows=None,
                   selection_fit_rows=None, selection_evaluation_rows=None):
    """Standalone k point: one selector, this k ridge, and this k application.

    Calibration lengths must include every distinct teacher cache used by the
    selector or final fit. Row counts allow the protocol's fixed token sample;
    omitted counts use every calibration token and in-sample source selection.
    """
    lengths, query_counts = list(map(int, lengths)), list(map(int, query_counts))
    calibration_lengths = list(map(int, calibration_lengths))
    k = int(k)
    rows = sum(calibration_lengths) if fitting_rows is None else int(fitting_rows)
    fit_rows = rows if selection_fit_rows is None else int(selection_fit_rows)
    eval_rows = fit_rows if selection_evaluation_rows is None else int(selection_evaluation_rows)
    selector = source_layer_selection(fit_rows, eval_rows)
    ridge = centered_ridge(rows, k * D)
    parts = {
        "calibration_teacher_cache": sum(exact_rebuild(n) for n in calibration_lengths),
        "source_layer_selection": sum(selector.values()),
        "this_k_final_ridge": 2 * L * sum(ridge.values()),
        "evaluation_translation": sum(translate_apply(n, k) for n in lengths),
    }
    return _result("translate", lengths, query_counts, parts, {
        "k": k, "calibration_users": len(calibration_lengths),
        "teacher_users_over_evaluation_users": len(calibration_lengths) / len(lengths),
        "fitting_rows": rows, "selector_fit_rows": fit_rows, "selector_evaluation_rows": eval_rows,
        "selector_cost_detail": selector, "one_target_layer_one_kv_ridge_detail": ridge,
        "shared_parameter_bytes": (2 * L * k * D * D + 2 * L * D) * ELEMENT_BYTES,
        "refreshed_kv_bytes": 2 * L * sum(lengths) * D * ELEMENT_BYTES,
        "extra_persistent_entry_bytes": 0,
        "mapping": "For each target layer and K/V, concatenate k full-width source layers and map kD to D.",
        "selection_accounting": "Selection is shared across k in execution; each standalone plotted point pays for it once, not for all other k fits.",
    })


def shared_read_cost(lengths, query_counts, calibration_lengths, use_response, fit_queries=16):
    """Shared bias or bias+Tr; one sequential calibration and all real reads."""
    lengths, query_counts = list(map(int, lengths)), list(map(int, query_counts))
    calibration_lengths = list(map(int, calibration_lengths))
    users = len(calibration_lengths)
    ridge = shared_ridge(users, fit_queries, use_response)
    prefix_applications = L * (L - 1) // 2
    parts = {
        "calibration_teacher_cache": sum(exact_rebuild(n) for n in calibration_lengths),
        "six_complete_source_reader_passes": L * sum(native_read(n, fit_queries) for n in calibration_lengths),
        "six_single_layer_teacher_reads": L * sum(teacher_layer_read(n, fit_queries) for n in calibration_lengths),
        "target_and_observed_rates": 3 * L * users * fit_queries * D,
        "fifteen_previous_layer_corrections": prefix_applications * shared_layer_read(users, users * fit_queries, use_response),
        "six_weighted_ridge_fits": L * sum(ridge.values()),
        "coefficient_folding_once": L * shared_coefficient_fold(use_response),
        "evaluation_extra_reads": L * shared_layer_read(len(lengths), sum(query_counts), use_response),
    }
    return _result("shared_bias_response" if use_response else "shared_bias", lengths, query_counts, parts, {
        "calibration_users": users, "teacher_users_over_evaluation_users": users / len(lengths),
        "fit_candidates_per_user": fit_queries, "one_layer_ridge_detail": ridge,
        "shared_parameter_bytes": L * D * (1 + (D if use_response else 0)) * ELEMENT_BYTES,
        "per_snapshot_scaled_bias_bytes_if_retained": L * D * ELEMENT_BYTES,
        "refreshed_kv_bytes": 0, "extra_persistent_entry_bytes": 0,
        "serving_arithmetic": "delta=N*b or N*b+T*r; normalization folded once, N*b once per snapshot, every actual feedback query charged.",
        "runtime_difference": "The current callback computes (r/N)T then multiplies by N; algebraic cancellation is included only in the theoretical ledger, not claimed as an implemented optimization.",
    })
