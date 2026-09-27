"""Ten-layer arithmetic using an isolated instance of the frozen six-layer ledger.

The common formulas read D/L/H/d from their own module namespace. Four small
adapters remove six-layer assumptions: complete-reader projection/norm counts,
the ridge output default, DroidSpeak's interval count, and component labels.
The original module and its globals are never changed.
"""

from __future__ import annotations

import importlib.util
from itertools import combinations_with_replacement
from pathlib import Path


SOURCE = Path(__file__).with_name("unified_auc_cost.py")


def make_cost(hidden_size=320, num_layers=10, num_heads=10):
    if (hidden_size, num_layers, num_heads) not in ((320, 10, 10), (192, 6, 6)):
        raise ValueError("This ledger supports the frozen Large model and its Medium reference check")
    spec = importlib.util.spec_from_file_location("_isolated_auc_arithmetic", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.D, module.L, module.H = hidden_size, num_layers, num_heads
    module.d = hidden_size // num_heads
    D, L, H = module.D, module.L, module.H

    def native_read(n, queries):
        # Five D-to-D projections per block; one RMSNorm per block plus final norm.
        norms = L + 1
        gemm = queries * (2 * (32 * D + D * D + 5 * L * D * D) + 4 * L * D * n + 2 * D)
        scalar = queries * (16 + 3 * D + norms * (4 * D + 1) + L * (6 * D + H + 2 * H * n) + 1)
        special = queries * (L * H * (n + 1) + L * D + norms + 32)
        return gemm + scalar + special

    original_ridge = module.centered_ridge

    def centered_ridge(rows, inputs, outputs=None):
        # Python's original outputs=D default remains 192 after changing globals.
        return original_ridge(rows, inputs, D if outputs is None else outputs)

    def droidspeak_cost(lengths, query_counts, calibration_lengths, interval, fit_queries=16):
        lengths, query_counts = list(map(int, lengths)), list(map(int, query_counts))
        calibration_lengths = list(map(int, calibration_lengths))
        start, end = map(int, interval)
        if not 0 <= start <= end < L:
            raise ValueError("Selected DroidSpeak interval is outside the model")
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
            "entry_state": f"All E_1..E_{L - 1} were retained during Parent execution; no teacher boundary is supplied.",
            "quality_executor": "Existing complete interval blocks; theoretical cost omits the unused final block output, with identical installed K/V.",
            "selection": "One interval per layer-count budget selected using calibration logit MSE; no evaluation-label selection.",
        })

    original_shared = module.shared_read_cost

    def shared_read_cost(lengths, query_counts, calibration_lengths, use_response, fit_queries=16):
        result = original_shared(lengths, query_counts, calibration_lengths, use_response, fit_queries)
        if L != 6:
            renames = {
                "six_complete_source_reader_passes": f"{L}_complete_source_reader_passes",
                "six_single_layer_teacher_reads": f"{L}_single_layer_teacher_reads",
                "fifteen_previous_layer_corrections": f"{L * (L - 1) // 2}_previous_layer_corrections",
                "six_weighted_ridge_fits": f"{L}_weighted_ridge_fits",
            }
            result["components"] = {renames.get(key, key): value for key, value in result["components"].items()}
        return result

    module.native_read = native_read
    module.centered_ridge = centered_ridge
    module.droidspeak_cost = droidspeak_cost
    module.shared_read_cost = shared_read_cost
    return module


def verify_medium_equivalence():
    """Small exact check for all methods, including partial intervals and widths."""
    spec = importlib.util.spec_from_file_location("_original_medium_arithmetic", SOURCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    adapted = make_cost(192, 6, 6)
    lengths, queries, teachers = [64, 1024], [3, 7], [64, 1024]
    cases = [("droidspeak_cost", (lengths, queries, teachers, interval))
             for interval in ((0, 0), (2, 4), (0, 5))]
    cases += [("tail_cost", (lengths, queries, width)) for width in (32, 1024)]
    cases += [("translate_cost", (lengths, queries, teachers, k)) for k in (1, 4)]
    cases += [("shared_read_cost", (lengths, queries, teachers, response)) for response in (False, True)]
    for name, arguments in cases:
        if getattr(original, name)(*arguments) != getattr(adapted, name)(*arguments):
            raise AssertionError(f"Medium reference mismatch: {name}")
    if (original.D, original.L, original.H, original.d) != (192, 6, 6, 32):
        raise AssertionError("Isolated adaptation changed the original module")
    return dict(status="passed", exact_method_cases=len(cases))


LARGE = make_cost()


if __name__ == "__main__":
    print(verify_medium_equivalence())
    print({"architecture": [LARGE.L, LARGE.D, LARGE.H, LARGE.d],
           "exact_rebuild_flops_per_1024_event_user": LARGE.exact_rebuild(1024)})
