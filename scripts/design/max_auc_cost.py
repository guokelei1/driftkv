"""Sixteen-layer arithmetic, reusing an isolated ten-layer cost instance.

The Max model keeps Large's width/head layout (D320/H10/d32). All common
formulas read the isolated module's L; only closures capturing ten layers
need new reader/profiler implementations and descriptive component names.
Neither frozen source module nor its exported instance is changed.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

from design import large_auc_cost


def make_cost():
    module = large_auc_cost.make_cost()
    module.L = 16
    D, L, H = module.D, module.L, module.H

    def native_read(n, queries):
        norms = L + 1
        gemm = queries * (2 * (32 * D + D * D + 5 * L * D * D) + 4 * L * D * n + 2 * D)
        scalar = queries * (16 + 3 * D + norms * (4 * D + 1) + L * (6 * D + H + 2 * H * n) + 1)
        special = queries * (L * H * (n + 1) + L * D + norms + 32)
        return gemm + scalar + special

    def droidspeak_cost(lengths, query_counts, calibration_lengths, interval, fit_queries=16):
        lengths, query_counts = list(map(int, lengths)), list(map(int, query_counts))
        calibration_lengths = list(map(int, calibration_lengths))
        start, end = map(int, interval)
        if not 0 <= start <= end < L:
            raise ValueError("Selected DroidSpeak interval is outside the Max model")
        intervals = tuple(combinations_with_replacement(range(L), 2))
        count, users = len(intervals), len(calibration_lengths)
        parts = {
            "profile_teacher_cache": sum(module.exact_rebuild(n) for n in calibration_lengths),
            f"profile_all_{count}_interval_caches": sum(module.interval_rebuild(n, a, b)
                for n in calibration_lengths for a, b in intervals),
            f"profile_{count}_predictions_and_exact_read": (count + 1) * sum(native_read(n, fit_queries) for n in calibration_lengths),
            "profile_logit_mse_approx": count * users * (3 * fit_queries + 1),
            "evaluation_interval_refresh": sum(module.interval_rebuild(n, start, end) for n in lengths),
        }
        return module._result("droidspeak", lengths, query_counts, parts, {
            "interval": [start, end], "profile_intervals": [list(x) for x in intervals],
            "calibration_users": users, "teacher_users_over_evaluation_users": users / len(lengths),
            "profile_candidates_per_user": fit_queries,
            "extra_persistent_entry_bytes": (L - 1) * sum(lengths) * D * module.ELEMENT_BYTES,
            "selected_entry_read_bytes": sum(lengths) * D * module.ELEMENT_BYTES if start else 0,
            "refreshed_kv_bytes": 2 * (end - start + 1) * sum(lengths) * D * module.ELEMENT_BYTES,
            "entry_state": "All E_1..E_15 were retained during Parent execution; no teacher boundary is supplied.",
            "quality_executor": "Existing complete interval blocks; theoretical cost omits the unused final block output, with identical installed K/V.",
            "selection": "One interval per layer-count budget selected using calibration logit MSE; no evaluation-label selection.",
        })

    shared = module.shared_read_cost

    def shared_read_cost(lengths, query_counts, calibration_lengths, use_response, fit_queries=16):
        result = shared(lengths, query_counts, calibration_lengths, use_response, fit_queries)
        renamed = {
            "10_complete_source_reader_passes": "16_complete_source_reader_passes",
            "10_single_layer_teacher_reads": "16_single_layer_teacher_reads",
            "45_previous_layer_corrections": "120_previous_layer_corrections",
            "10_weighted_ridge_fits": "16_weighted_ridge_fits",
        }
        result["components"] = {renamed.get(key, key): value for key, value in result["components"].items()}
        return result

    module.native_read = native_read
    module.droidspeak_cost = droidspeak_cost
    module.shared_read_cost = shared_read_cost
    return module


MAX = make_cost()


def verify_reuse():
    """Confirm unchanged 6L/10L accounting and the Max-specific depth factors."""
    medium = large_auc_cost.verify_medium_equivalence()
    original, isolated = large_auc_cost.LARGE, large_auc_cost.make_cost()
    lengths, queries, teachers = [64, 1024], [3, 7], [64, 1024]
    cases = [("droidspeak_cost", (lengths, queries, teachers, interval))
             for interval in ((0, 0), (2, 4), (0, 9))]
    cases += [("tail_cost", (lengths, queries, width)) for width in (32, 1024)]
    cases += [("translate_cost", (lengths, queries, teachers, k)) for k in (1, 4)]
    cases += [("shared_read_cost", (lengths, queries, teachers, response)) for response in (False, True)]
    for name, arguments in cases:
        if getattr(original, name)(*arguments) != getattr(isolated, name)(*arguments):
            raise AssertionError(f"Ten-layer reference mismatch: {name}")
    if (original.L, isolated.L, MAX.L) != (10, 10, 16):
        raise AssertionError("Max changed another arithmetic instance")
    droid = MAX.droidspeak_cost(lengths, queries, teachers, (0, 15))
    if (len(droid["metadata"]["profile_intervals"]) != 136
            or droid["components"]["profile_136_predictions_and_exact_read"] != 137 * sum(MAX.native_read(n, 16) for n in teachers)
            or droid["components"]["evaluation_interval_refresh"] != droid["exact_rebuild_ops"]
            or MAX.tail_cost(lengths, queries, 1024)["extra_over_exact"] != 1):
        raise AssertionError("Max interval profiler or full reconstruction arithmetic differs")
    read = MAX.shared_read_cost(lengths, queries, teachers, True)
    if (read["components"]["16_weighted_ridge_fits"] != 16 * sum(MAX.shared_ridge(2, 16, True).values())
            or read["components"]["120_previous_layer_corrections"] != 120 * MAX.shared_layer_read(2, 32, True)):
        raise AssertionError("Max sequential calibration depth factors differ")
    return dict(status="passed", medium=medium, large_exact_method_cases=len(cases),
                max_intervals=136, max_previous_layer_corrections=120)


if __name__ == "__main__":
    print(verify_reuse())
    print({"architecture": [MAX.L, MAX.D, MAX.H, MAX.d],
           "exact_rebuild_flops_per_1024_event_user": MAX.exact_rebuild(1024)})
