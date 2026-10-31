#!/usr/bin/env python3
"""Combine actual event shapes with measured API times; never FLOPs/peak seconds."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OLD_LEDGER = ROOT / "results/query_relay_2026_10/workloads/stride16_candidates16_ledger.json"
DEFAULT_SHAPES = ROOT / "results/system_study_2026_10/service/time_model/ledger30k/attempt2/summary.json"
DEFAULT_NATIVE = ROOT / "results/system_study_2026_10/service/model_group_probe/attempt1/summary.json"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def joint_shapes(events, requests, releases, end, cap=1024, band_cap=32):
    """Same causal timeline as the retained cost ledger, retaining joint shapes."""
    import numpy as np
    timestamps = events[0]
    cursor = int(np.searchsorted(timestamps, releases[0][1], side="left"))
    length, previously_materialized = min(cursor, cap), False
    queries = defaultdict(Counter)
    for row in requests:
        queries[row["version"]][int(row["query_timestamp"])] += row["candidates"]
    shapes, consumers, counts = Counter(), Counter(), {}
    for index, (version, at) in enumerate(releases):
        until = releases[index + 1][1] if index + 1 < len(releases) else end
        c = Counter(initial_history_rows=length)
        def record(operation, width=0, code=None, source=None):
            shapes[(version, source or version, operation, length, width,
                    coded if code is None else code)] += 1
        if previously_materialized:
            c["current_to_anchor_rows"] += length
            # All rows, including the current-born zero suffix, are inverse-converted.
            record("anchorize", code=length, source=releases[index - 1][0])
        coded, materialized = length, False
        def consume(role):
            nonlocal materialized
            if not materialized:
                c["first_consumptions"] += 1
                c["first_consumption_" + role] += 1
                c["anchor_to_current_rows"] += length
                record("materialize", code=length)
                materialized = True
        def advance(timestamp):
            nonlocal cursor, length, coded
            stop = int(np.searchsorted(timestamps, timestamp, side="left"))
            while cursor < stop:
                width = 1 << (min(band_cap, stop - cursor).bit_length() - 1)
                consume("writer")
                record("append", width)
                c["writer_bands"] += 1
                c["writer_queries"] += width
                c["writer_dense_pairs"] += width * (length + width)
                c["writer_prefix_pairs"] += width * coded
                if coded:
                    c["writer_nonzero_prefix_queries"] += width
                dropped = np.maximum(0, length + np.arange(1, width + 1) - cap)
                c["writer_valid_prefix_pairs"] += int(np.maximum(0, coded - dropped).sum())
                c[f"native_append_shape_{length}:{width}"] += 1
                coded = max(0, coded - max(0, length + width - cap))
                length = min(cap, length + width)
                cursor += width
                c["appends"] += width
        for query_index, (timestamp, width) in enumerate(sorted(queries[version].items())):
            entering_length, entering_cursor, had_code = length, cursor, materialized
            advance(timestamp)
            consume("query")
            c["ranking_queries"] += width
            c["ranking_dense_pairs"] += width * length
            c["ranking_prefix_pairs"] += width * coded
            if coded:
                c["ranking_nonzero_prefix_queries"] += width
            if query_index == 0:
                c[f"once_rebuild_length_{length}"] += 1
                record("rebuild", code=0)
            for offset in range(0, width, 32):
                c[f"native_query_shape_{length}:{min(32, width-offset)}"] += 1
            for offset in range(0, width, 16):
                record("score", min(16, width - offset))
                first = offset == 0
                changed = first and cursor > entering_cursor
                start_length = entering_length if first else length
                for method in ("relay", "rebuild"):
                    rebuilt = method == "rebuild" and first and query_index == 0
                    # A first request with no append can reconstruct from item
                    # history without paging the obsolete native K/V in.
                    load_native = not (rebuilt and not changed)
                    consumers[(method, version, "recommendation", start_length, length,
                        load_native, method == "relay", changed or rebuilt,
                        method == "relay" and (changed or (first and not had_code)))] += 1
        entering_length, entering_cursor = length, cursor
        advance(until)
        changed = cursor > entering_cursor
        anchor_at_boundary = index + 1 < len(releases) and materialized
        for method in ("relay", "rebuild"):
            if changed or (method == "relay" and anchor_at_boundary):
                consumers[(method, version, "publication_or_final_drain", entering_length, length,
                    True, method == "relay", changed, method == "relay")] += 1
        c.update(final_history_rows=length, final_coded_rows=coded,
                 materialized_users=int(materialized))
        counts[version] = c
        previously_materialized = materialized
    return shapes, counts, consumers


def build_shapes(output, threads=4):
    import numpy as np
    import pyarrow.parquet as pq
    from hstu_kvcache.evaluation.history import load_histories_semijoin
    from hstu_kvcache.evaluation.release_context import RELEASES, END
    from hstu_kvcache.methods.cachetide.cost import CONFIG, PANEL
    from hstu_kvcache.evaluation.cachetide_workload import compact_rounds
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    old, config, panel = read(OLD_LEDGER), read(CONFIG), read(PANEL)
    dataset_path = ROOT / config["data"]["dataset"]["path"]
    population_path = ROOT / config["data"]["population"]["path"]
    dataset = read(dataset_path)
    users = sorted(pq.read_table(population_path, columns=["uid"])["uid"].to_pylist())
    assert len(users) == len(set(users)) == old["users"] == 30000
    print(json.dumps(dict(status="loading_30k_CPU_histories", users=len(users))), flush=True)
    histories = load_histories_semijoin(users, dataset_path=dataset_path,
        known_vocab_size=dataset["foundation_items"],
        oov_buckets=panel["model_config"]["num_items"] - dataset["foundation_items"],
        start_timestamp=RELEASES[0][1], end_timestamp=END, max_history=1024, threads=threads)
    shapes, consumers = Counter(), Counter()
    counts = {version: Counter() for version, _ in RELEASES}
    for number, uid in enumerate(users):
        events = histories.rows.get(uid, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
        requests = compact_rounds(events, stride=16, candidates=16, known_vocab_size=dataset["foundation_items"])
        local_shapes, local_counts, local_consumers = joint_shapes(events, requests, RELEASES, END)
        shapes.update(local_shapes)
        consumers.update(local_consumers)
        for version, local in local_counts.items():
            counts[version].update(local)
        if (number + 1) % 5000 == 0:
            print(json.dumps(dict(status="counting_joint_shapes", users=number+1,
                elapsed_seconds=time.perf_counter()-start)), flush=True)
    checks = {}
    for version, c in counts.items():
        expected = dict(old["by_release"][version])
        expected.pop("ranking_rounds")
        if Counter(expected) != c:
            different = {k: (c[k], expected.get(k, 0)) for k in set(c) | set(expected)
                         if c[k] != expected.get(k, 0)}
            raise AssertionError((version, different))
        checks[version] = "all_original_native_and_code_counters_match"
    rows = [dict(version=v, parameter_version=pv, operation=op, history_length=h,
                 width=w, coded_rows=c, calls=n)
            for (v, pv, op, h, w, c), n in sorted(shapes.items())]
    consumer_rows = [dict(method=m, version=v, kind=k, input_rows=before, output_rows=after,
        load_native=ln, load_codes=lc, write_native=wn, write_codes=wc, calls=n,
        h2d_bytes=n * before * (9216*ln + 1152*lc),
        d2h_bytes=n * after * (9216*wn + 1152*wc))
        for (m, v, k, before, after, ln, lc, wn, wc), n in sorted(consumers.items())]
    result = dict(status="complete_cpu_joint_shape_ledger", users=len(users), model="medium",
        workload=old["workload"], shapes=rows, host_consumer_shapes=consumer_rows,
        by_release={v: dict(c) for v, c in counts.items()},
        checks=checks, elapsed_CPU_builder_seconds=time.perf_counter()-start,
        source_bindings=[binding(p) for p in (OLD_LEDGER, CONFIG, PANEL, dataset_path, population_path, Path(__file__))],
        scope="Real 30k causal behavior shapes only; no model execution, quality labels, GPU or arrival timing. Query32 split into two Query16 APIs; appends keep original power-of-two bands.",
        batching_scope="Individual call histogram preserves prefix joint shapes but not a global arrival/batch schedule.",
        host_scenario_scope="One missed history consumer keeps state resident across its append bands, preparation and score; writes back changed native/code state once. Publication old-tail advance and anchorization share a page. Query16 rounds at tied timestamps are separate consumers. This is a conditional host placement policy, not an observed hit rate or implemented backing tier.")
    write(output / "summary.json", result)
    (output / "summary.md").write_text("# 30k真实事件联合形状\n\n" +
        f"共{len(rows):,}种联合形状；三版本全部原始native与码计数逐项一致。增加码前缀长度、物化和反锚形状，未运行模型。\n" +
        "推荐以16候选API计时；同时间32候选拆为两次。形状计数不规定跨用户批次或到达排队。\n")
    print(json.dumps(dict(status=result["status"], shapes=len(rows), output=str(output))), flush=True)
    return result


def length_bin(n):
    return 0 if n == 0 else 1 if n <= 128 else 2 if n <= 512 else 3 if n < 1024 else 4


def prefix_bin(n, coded):
    return 0 if coded == 0 else min(4, max(1, math.ceil(4*coded/max(1, n))))


def session_observations(directory):
    """Every timed call has three non-overlapping, synchronous phase records."""
    import pyarrow.parquet as pq
    summary = read(directory / "summary.json")
    if summary.get("status") != "complete_distinct_user_batched_session_probe":
        raise ValueError("session measurements are not complete")
    diagnostics = read(directory / "diagnostics.json")
    summary["time_model_phase_scope"] = dict(
        separately_warmed=all(r.get("phase_is_after_separate_backend_warmup", False)
                             for r in (diagnostics.values() if isinstance(diagnostics, dict) else diagnostics)))
    rows = pq.read_table(directory / "phases.parquet").to_pylist()
    if len(rows) % 3:
        raise ValueError("incomplete call phases")
    execution_config = read(directory / "configuration.json")["config"]
    graph_enabled = execution_config.get("graph_compute", False)
    summary["time_model_graph_policy"] = graph_policy(execution_config)
    # Older Parquet writers inferred columns from the first pack row, before
    # compute_backend was assigned. Their retained ordered call ledgers carry
    # the actual execution choice; do not infer measured backends from shapes.
    ledgers = {method:read(directory / f"{method}_calls.json")
               for method in {row["method"] for row in rows}
               if (directory / f"{method}_calls.json").exists()}
    positions = Counter()
    calls = []
    for offset in range(0, len(rows), 3):
        group = rows[offset:offset+3]
        if [r["phase"] for r in group[:2]] != ["pack_h2d", "compute"] or not group[-1]["phase"].startswith("scatter"):
            raise ValueError("phase ordering changed; do not add overlapping intervals")
        first = group[0]
        for row in group[1:]:
            if any(row[k] != first[k] for k in ("method", "version", "operation", "uids", "history_length", "query_width")):
                raise ValueError("phase call identity mismatch")
        method = first["method"]
        metadata = None
        if method in ledgers:
            metadata = ledgers[method][positions[method]]
            if any(metadata[k] != first[k] for k in ("version", "operation", "uids", "history_length", "query_width")):
                raise ValueError("ordered call ledger does not match measured phases")
        positions[method] += 1
        backend = group[1].get("compute_backend")
        recorded_backend = metadata.get("compute_backend") if metadata else None
        if backend and recorded_backend and backend != recorded_backend:
            raise ValueError("phase and call ledger disagree on compute backend")
        backend = backend or recorded_backend
        if backend is None:
            if graph_enabled and first["operation"] in ("append", "score", "rebuild"):
                raise ValueError("graph measurement lacks actual compute_backend evidence")
            backend = "eager"
        calls.append(dict(method=first["method"], version=first["version"], operation=first["operation"],
            history_length=first["history_length"], width=first["query_width"], coded_rows=first["coded_rows"],
            compute_backend=backend,
            session_coded_rows=first.get("session_coded_rows"),
            batch=first["batch"], stage_seconds={r["phase"]: r["wall_seconds"] for r in group},
            wall_seconds=sum(r["wall_seconds"] for r in group)))
    if any(positions[method] != len(ledger) for method,ledger in ledgers.items()):
        raise ValueError("measured phases do not cover the complete retained call ledger")
    return summary, calls


def graph_policy(config):
    """Bind dispatch to the measured run, retaining the original default policy."""
    return dict(batch_sizes=tuple(config.get("graph_batch_sizes",(1,8))),
                append_widths=tuple(config.get("graph_append_widths",(1,16,32))))


def planned_backend(row, graph_enabled=False, *, batch_sizes=(1,8), append_widths=(1,16,32)):
    """The captured run's policy; never consult later GraphCompute source."""
    if not graph_enabled or row["history_length"] != 1024 or row["batch"] not in batch_sizes:
        return "eager"
    op,width = row["operation"],row["width"]
    # Rebuild's API ids width is history_length; its logical query width is 0.
    supported = ((op == "score" and width == 16) or
                 (op == "append" and width in append_widths) or op == "rebuild")
    return "cuda_graph" if supported else "eager"


def coefficient_keys(row):
    method, version, op = row["method"], row["version"], row["operation"]
    h, w = row["history_length"], row["width"]
    coded = row["coded_rows"] if method == "relay" and op in ("score", "append") else -1
    band = prefix_bin(h, coded) if coded >= 0 else -1
    # A fallback is an explicit empirical scenario, never an exact measurement
    # for a previously unmeasured shape. No scale or method is crossed.
    keys = [(method, version, op, h, w, coded),
            (method, version, op, h, w, band),
            (method, version, op, length_bin(h), w, band),
            (method, version, op, h == 1024, w, coded > 0),
            (method, op, h == 1024, w, coded > 0),
            (method, op, h == 1024, w), (method, op, h == 1024), (method, op)]
    # Even the broadest fallback must retain the measured execution backend.
    return [(row.get("compute_backend", "eager"),*key) for key in keys]


class Coefficients:
    def __init__(self, calls, *, graph_enabled=None, measured_graph_policy=None):
        self.graph_enabled = (any(row.get("compute_backend") == "cuda_graph" for row in calls)
                              if graph_enabled is None else graph_enabled)
        self.graph_policy = measured_graph_policy or graph_policy({})
        self.pools = [defaultdict(lambda: defaultdict(list)) for _ in range(8)]
        self.cache = {}
        self.aggregates = {}
        for row in calls:
            for pool, key in zip(self.pools, coefficient_keys(row)):
                pool[key][row["batch"]].append(row)

    def lookup(self, row, maximum=8, forced_batch=None):
        if "compute_backend" not in row:
            row = dict(row, compute_backend=planned_backend(
                dict(row,batch=row.get("batch", forced_batch or maximum)),self.graph_enabled,**self.graph_policy))
        cache_key = (tuple(coefficient_keys(row)[0]), maximum, forced_batch)
        if cache_key in self.cache:
            return self.cache[cache_key]
        for tier, (pool, key) in enumerate(zip(self.pools, coefficient_keys(row))):
            batches = [b for b in pool.get(key, {}) if b <= maximum and (forced_batch is None or b == forced_batch)]
            if not batches:
                continue
            batch = max(batches)
            aggregate_key = (tier,key,batch)
            if aggregate_key in self.aggregates:
                result = self.aggregates[aggregate_key]
                self.cache[cache_key] = result
                return result
            rows = pool[key][batch]
            times = [r["wall_seconds"]/batch for r in rows]
            stages = defaultdict(list)
            for r in rows:
                for stage, value in r["stage_seconds"].items():
                    stages[stage].append(value/batch)
            result = dict(tier=tier, batch=batch, compute_backend=row["compute_backend"], observations=len(rows),
                seconds_per_user_operation=statistics.mean(times),
                low_seconds_per_user_operation=min(times), high_seconds_per_user_operation=max(times),
                stage_seconds_per_user_operation={s: statistics.mean(v) for s,v in stages.items()})
            self.cache[cache_key] = result
            self.aggregates[aggregate_key] = result
            return result
        raise ValueError(f"no measured coefficient even for operation: {row}")

    def requested(self, row, batch):
        """Preserve the actual deployment batch when observed at any shape tier."""
        try:
            result = self.lookup(row, 8, forced_batch=batch)
        except ValueError:
            try:
                result = self.lookup(row, batch)
            except ValueError:
                # A small development trace may contain no B1 graph rebuild.
                # Keep the same backend, charge the *whole* measured larger
                # batch call, and expose this empirical proxy instead of
                # linearly extrapolating a better large-batch per-user rate.
                for larger in range(batch+1,9):
                    try:
                        result = dict(self.lookup(row,8,forced_batch=larger))
                        factor = larger/batch
                        for key in ("seconds_per_user_operation","low_seconds_per_user_operation","high_seconds_per_user_operation"):
                            result[key] *= factor
                        result["stage_seconds_per_user_operation"] = {k:v*factor for k,v in result["stage_seconds_per_user_operation"].items()}
                        result["batch_proxy_kind"] = "larger_batch_full_call_empirical_proxy"
                        break
                    except ValueError:
                        continue
                else:
                    raise
        return dict(result, requested_batch=batch, batch_matched=result["batch"] == batch)


def project_calls(shapes, coefficients, method, maximum=8):
    if coefficients.graph_enabled:
        raise ValueError("graph projection requires the real batch ledger for shape-dependent backend dispatch")
    totals = Counter()
    stages, operations, coverage = Counter(), defaultdict(Counter), Counter()
    for shape in shapes:
        op = shape["operation"]
        if (method == "relay" and op == "rebuild") or (method == "rebuild" and op in ("materialize", "anchorize")):
            continue
        row = dict(shape, method=method, version=shape.get("parameter_version", shape["version"]))
        coeff = coefficients.lookup(row, maximum)
        n = shape["calls"]
        for target, source in (("point_seconds", "seconds_per_user_operation"),
                               ("empirical_low_seconds", "low_seconds_per_user_operation"),
                               ("empirical_high_seconds", "high_seconds_per_user_operation")):
            amount = n * coeff[source]
            totals[target] += amount
            operations[op][target] += amount
        totals["user_operations"] += n
        operations[op]["user_operations"] += n
        coverage[f"tier{coeff['tier']}"] += n
        coverage[f"batch{coeff['batch']}"] += n
        for stage, value in coeff["stage_seconds_per_user_operation"].items():
            stages[stage] += n*value
    return dict(**dict(totals), stages=dict(stages), operations=dict(operations),
        weighted_coefficient_coverage=dict(coverage),
        range_scope="Sum of empirical observed per-call minima/maxima in selected shape pools, not confidence limits or guaranteed bounds")


def project_batches(shapes, coefficients, method):
    totals, stages, coverage = Counter(), Counter(), Counter()
    larger_batch_proxy = Counter()
    operations = defaultdict(Counter)
    for row in shapes:
        if row["method"] != method:
            continue
        coeff = coefficients.requested(row, row["batch"])
        n = row["calls"] * row["batch"]
        for target, source in (("point_seconds", "seconds_per_user_operation"),
                               ("empirical_low_seconds", "low_seconds_per_user_operation"),
                               ("empirical_high_seconds", "high_seconds_per_user_operation")):
            amount = n * coeff[source]
            totals[target] += amount
            operations[row["operation"]][target] += amount
        totals["user_operations"] += n
        totals["batch_calls"] += row["calls"]
        operations[row["operation"]]["user_operations"] += n
        coverage[f"tier{coeff['tier']}"] += n
        coverage[f"target_batch{row['batch']}"] += n
        coverage["exact_batch" if coeff["batch_matched"] else "batch_proxy"] += n
        if coeff.get("batch_proxy_kind"):
            coverage[coeff["batch_proxy_kind"]] += n
            larger_batch_proxy["user_operations"] += n
            larger_batch_proxy["batch_calls"] += row["calls"]
            larger_batch_proxy["point_seconds"] += n*coeff["seconds_per_user_operation"]
        coverage["backend_"+coeff["compute_backend"]] += n
        for stage, value in coeff["stage_seconds_per_user_operation"].items():
            stages[stage] += n*value
    return dict(**dict(totals), stages=dict(stages), operations=dict(operations),
        weighted_coefficient_coverage=dict(coverage),
        larger_batch_full_call_proxy=dict(larger_batch_proxy),
        range_scope="Exact real-population frontier batches; times use observed same-batch shape pools or flagged empirical proxies. Min/max sums are neither bounds nor confidence intervals.")


def sample_accounting(summary, calls, coefficients, method, maximum):
    selected = [r for r in calls if r["method"] == method]
    phase_sum = sum(r["wall_seconds"] for r in selected)
    if maximum == 8:
        predicted = sum(coefficients.lookup(r, maximum, forced_batch=r["batch"])["seconds_per_user_operation"] * r["batch"] for r in selected)
        if not math.isclose(predicted, phase_sum, rel_tol=1e-12):
            raise AssertionError("sample phase accounting failed")
    else:
        predicted = sum(coefficients.lookup(r, 1)["seconds_per_user_operation"] * r["batch"] for r in selected)
    execution = "batch8" if maximum == 8 else "serial"
    observed = [r["service_wall_seconds"] for r in summary["time_units"]
                if r["method"] == method and r["execution"] == execution]
    return dict(sample_users=len(summary["users"]), diagnostic_phase_sum_seconds=phase_sum,
        coefficient_prediction_before_normalization_seconds=predicted,
        uninstrumented_service_seconds=observed,
        normalization_to_bulk_median=statistics.median(observed)/predicted,
        normalization_low=min(observed)/predicted, normalization_high=max(observed)/predicted,
        scope="Same-sample accounting and observed barrier/orchestration normalization, not independent prediction validation; original scores and state checks are in the session report")


def shared_preparation(directory):
    measured, calibration = read(directory / "summary.json"), read(directory / "calibration_timing.json")
    known = sum(r["wall_seconds"] or 0 for r in calibration["phases"])
    low = known + sum(r.get("lower_seconds", 0) for r in calibration["phases"])
    high = known + sum(r.get("upper_seconds", 0) for r in calibration["phases"])
    shared = measured["aggregates"]["resident_shared_compute_seconds"]
    distribution_four = measured["aggregates"]["distribution_serial_seconds"]
    distribution = sum(r["wall_seconds"] for r in measured["phases"] if r["category"] == "distribution"
                       and (r["name"].endswith("d2h") or r.get("replica_index") == 0))
    costs = read(ROOT / "results/system_study_2026_10/cost/scaling/attempt1/summary.json")["branches"]["current"]["decomposition"]
    fixed = costs["fixed_components_flops"]
    one_compile = (costs["catalogue_linear_preparation_flops"] + costs["small_matrix_preparation_flops"]) / costs["replicas"]
    return dict(calibration_recorded_workflow_lower_seconds=low, calibration_recorded_workflow_upper_seconds=high,
        fresh_shared_compute_seconds=shared, shared_distribution_serial_seconds=distribution,
        four_copy_distribution_serial_seconds_separate=distribution_four,
        fixed_S_lower_seconds=low+shared+distribution, fixed_S_upper_seconds=high+shared+distribution,
        preparation_strategies=dict(
            publisher_once_one_worker=dict(fixed_S_lower_seconds=low+shared+distribution,
                fixed_S_upper_seconds=high+shared+distribution,
                fixed_preparation_flops=fixed["correction_calibration_including_origin_once"]+one_compile,
                source_reconstruction_sensitivity_flops=costs["source_reconstruction_fixed_flops"],
                measured_or_projected="One measured GPU compile, each artifact D2H and replica0 H2D once. Conservative if publisher and worker share device and could retain state directly."),
            publisher_once_four_copies=dict(fixed_S_lower_seconds=low+shared+distribution_four,
                fixed_S_upper_seconds=high+shared+distribution_four,
                fixed_preparation_flops=fixed["correction_calibration_including_origin_once"]+one_compile,
                source_reconstruction_sensitivity_flops=costs["source_reconstruction_fixed_flops"],
                measured_or_projected="One actual GPU compile then four measured serialized host copies; no network or concurrent replica claim"),
            four_independent_preparers=dict(fixed_S_lower_seconds=low+4*shared,
                fixed_S_upper_seconds=high+4*shared,
                fixed_preparation_flops=costs["fixed_flops"],
                source_reconstruction_sensitivity_flops=costs["source_reconstruction_fixed_flops"],
                measured_or_projected="Four times measured resident shared compute, calibration once; method input/bank/model distribution to independent workers is not measured and must be added. Not an alternative complete deployment winner.")),
        standalone_method_input_load_seconds=measured["aggregates"]["method_input_load_seconds"],
        common_foundation_startup_seconds_separate=measured["aggregates"]["common_model_startup_seconds"],
        scope="Primary one A40 worker: retained offline calibration including causal source/teacher generation and one origin, fresh compile once, then measured D2H and first H2D copy per shared artifact. Historical calibration loads are not counted twice. Four workers are a separate sensitivity.",
        missing="Final relay checkpoint serialization and artifact/network persistence are not separately measured; add their nonnegative fixed delta_S if required. Initial V2 cache and foundation training are common outside the window.",
        distribution_scope="Primary uses D2H+replica0 H2D, conservatively retaining a host round trip even when publisher and serving share the GPU. The separate four-copy scenario uses all four actually sequential copies, never concurrent cluster wall time.")


def graph_preparation(session):
    """One setup of each method's graph set; repeated benchmark setups are samples."""
    output = {}
    if any(row.get("graph_captures_during_timing",0) for row in session["time_units"]):
        raise ValueError("steady service coefficients include new graph captures")
    for method,owner in (("relay","relay"),("rebuild","native")):
        samples = [row["graph_state_before"] for row in session["time_units"]
                   if row["method"] == method and row["execution"] == "batch8"
                   and row.get("graph_state_before")]
        if not samples:
            output[method] = dict(point_seconds=0.,low_seconds=0.,high_seconds=0.,samples_seconds=[],shapes=[],
                                  scope="No captured compute graph used by this measured method.")
            continue
        values = [s["preparation_seconds_by_method"].get(owner,0.) for s in samples]
        shapes = [r for r in samples[0].get("graph_shapes",[]) if r["owner"] == owner]
        shape_keys = lambda rows: sorted((r["operation"],r.get("version"),r["batch"],r["width"],r["initial_coded_rows"])
                                        for r in rows if r["owner"] == owner)
        if any(shape_keys(s.get("graph_shapes",[])) != shape_keys(shapes) for s in samples):
            raise ValueError("repeated graph preparations have different actual shape sets")
        output[method] = dict(point_seconds=statistics.median(values),low_seconds=min(values),high_seconds=max(values),
            samples_seconds=values,shapes=shapes,
            allocated_live_delta_bytes=max(s["allocated_delta_bytes_by_method"].get(owner,0) for s in samples),
            scope="Median of actual one-set setup samples, counted once per method. Includes static allocation, three warm compute calls and capture, not the full warm trace. Stable kernel-cache setup; pure first JIT is not separately measured.")
    return output


def graph_preparation_flops(setups):
    from hstu_kvcache.evaluation.cost import CostModel, eager_read
    from hstu_kvcache.methods.cachetide.cost import read_flops
    cm = CostModel(192,6,6,"triton")
    dense = CostModel(192,6,6,"torch")
    result = {}
    for method,setup in setups.items():
        if setup["point_seconds"] and not setup["shapes"]:
            result[method] = dict(total_flops=None,components={},priced_graph_entries=0,
                                 missing="Historical graph preparation lacks retained per-entry shape metadata.")
            continue
        components = Counter()
        for row in setup["shapes"]:
            op,n,w,b,c = (row[k] for k in ("operation","history_length","width","batch","initial_coded_rows"))
            # Legacy metadata counted the Python call inside capture. CUDA stream
            # capture records kernels for later replay; only the three warm calls
            # execute here. The first replay is already a normal service call.
            repeats = row.get("initial_prepare_executed_model_calls",3)
            if op == "rebuild":
                native = cm.full_cache(n)
            elif op == "append":
                native = cm.band_append(n,w,window_size=1024)
            else:
                native = eager_read(dense,n,queries=w)
            components[op+"_native_model_flops"] += repeats*b*native
            if method == "relay" and c and op in ("append","score"):
                correction = read_flops(pairs=w*c,queries=w,head_rank=6,gate_normalization="rms")["total"]
                correction -= (6-1)*6*(4*32+2)*w  # item-query RMS shared across layers
                components[op+"_correction_flops"] += repeats*b*correction
        result[method] = dict(total_flops=sum(components.values()),components=dict(components),
                             priced_graph_entries=len(setup["shapes"]))
    known = all(result[m]["total_flops"] is not None for m in result)
    return dict(methods=result,method_minus_native_setup_flops=result["relay"]["total_flops"]-result["rebuild"]["total_flops"] if known else None,
        scope="Three executed warm model calls per graph entry, with its recorded initial code prefix. The fourth Python call records the graph without executing device kernels; the following replay belongs to normal service. Legacy initial_prepare_model_calls=4 is interpreted using the audited implementation, not rewritten. Mathematical effective operations only; copies/capture metadata are not FLOPs, tile padding and TF32x3 machine instructions excluded. Repeated benchmark graph sets are not summed.",
        execution_semantics_source="https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html",
        audited_implementation=binding(ROOT/"experiments/archive/system_study/graph_compute.py"))


def graph_shape_coverage(batches, setups, *, graph_enabled, policy):
    """Graph memory is per operation/version/shape, not per user or call."""
    def key(row):
        op = row["operation"]
        return (op,row.get("version"),row["batch"],1024 if op == "rebuild" else row["width"],
                bool(row.get("coded_rows",row.get("initial_coded_rows",0))))
    def unpack(k):
        return dict(operation=k[0],version=k[1],batch=k[2],width=k[3],coded=k[4])
    output = {}
    for method in ("relay","rebuild"):
        needed = {key(r) for r in batches if r["method"] == method
                  and planned_backend(r,graph_enabled,**policy) == "cuda_graph"}
        observed = {key(r) for r in setups[method]["shapes"]}
        missing = needed-observed
        setup = setups[method]
        seconds_per_shape = setup["point_seconds"]/len(observed) if observed else None
        memory_per_shape = setup.get("allocated_live_delta_bytes",0)/len(observed) if observed else None
        output[method] = dict(required_shapes=len(needed),observed_shapes=len(observed),
            required_shapes_by_version=dict(Counter(k[1] for k in needed)),
            observed_shapes_by_version=dict(Counter(k[1] for k in observed)),
            covered_required_shapes=len(needed&observed),missing_shapes=[unpack(k) for k in sorted(missing)],
            measured_shapes_unused_by_population=[unpack(k) for k in sorted(observed-needed)],
            unmeasured_additional_preparation_required=bool(missing),
            additional_setup_mean_shape_seconds_proxy=len(missing)*seconds_per_shape if missing and seconds_per_shape is not None else 0.,
            additional_live_graph_bytes_mean_shape_proxy=len(missing)*memory_per_shape if missing and memory_per_shape is not None else 0.,
            scope="Main fixed preparation retains the actually measured graph set. Missing population shapes add unknown delta_S_graph and workspace; listed means per measured shape are diagnostics, not measured preparation or reliable bounds. The benchmark retains all releases' graph sets. Releasing old-version graphs could lower live peak but has not been measured here. Graph sets depend on model/adapter release and operation/B/W/has-code, not UID count. Positive code prefixes share one dynamic-count graph. No cold preparation is assigned to unsupported eager shapes.")
    return output


def v4_calibration_substitution(report_path, shared_path):
    report = read(report_path)
    if (report.get("status") != "complete" or not report["logical_operation_ledger_equal"]
            or not report["frozen_state_equal"] or not report["backbone_gradients_absent"]
            or not report["retained_reference_check"]["state_dict_bitwise_equal"]):
        raise ValueError("V4 execution comparison has not passed its required controls")
    old = next(r["wall_seconds"] for r in read(shared_path/"calibration_timing.json")["phases"]
               if r["name"] == "relay_calibration" and r["version"] == "v4")
    new = report["branches"]["batched"]["recorded_calibration_seconds"]
    return dict(old_V4_seconds=old,measured_batched_V4_seconds=new,change_in_method_fixed_seconds=new-old,
        evidence=binding(report_path),scope="Replace only V4's whole calibration with this measured common+batched-joint+post+save workflow. Keep original origin, V3, V5, all source/teacher generation, shared generation/distribution and graph setup; no propagation of a joint-only speed ratio. Single execution comparison, not population service measurement.")


def transfer_keys(row):
    return [(row["history_length"], row["batch"]),
            (length_bin(row["history_length"]), row["batch"]),
            (row["history_length"] == 1024, row["batch"]),
            (row["batch"],), ()]


def operation_paging_shapes(batches):
    """Capacity-feasible pages at actual API boundaries of frontier execution."""
    counts = Counter()
    for r in batches:
        m, v, n, b, op = (r[k] for k in ("method", "version", "history_length", "batch", "operation"))
        if op != "rebuild":
            counts[(m,v,"h2d",n,True,m == "relay",b)] += r["calls"]
        if op in ("materialize", "anchorize"):
            counts[(m,v,"d2h",n,False,True,b)] += r["calls"]
        elif op in ("append", "rebuild"):
            out_n = min(1024,n+r["width"]) if op == "append" else n
            counts[(m,v,"d2h",out_n,True,m == "relay",b)] += r["calls"]
    return [dict(method=m,version=v,direction=d,history_length=n,native=na,code=co,batch=b,calls=k,
                 payload_bytes=k*b*n*(9216*na+1152*co))
            for (m,v,d,n,na,co,b),k in sorted(counts.items())]


def project_host_traffic(batch_ledger, transfer_path, *, operation_boundary=True, window_boundary=False):
    """Directional real bytes times measured real-state visit coefficients.

    Transfer measurements include a complete round trip. This policy omits
    unneeded clean writeback and rebuild input K/V; directional field subsets
    therefore use explicitly approximate byte-scaled observed phase times.
    """
    actual = read(transfer_path)
    directional_shapes = batch_ledger["window_directional_batches"] if window_boundary else operation_paging_shapes(batch_ledger["batches"]) if operation_boundary else batch_ledger["host_directional_batches"]
    result = {}
    for method in ("relay", "rebuild"):
        pools = [defaultdict(list) for _ in range(5)]
        for observation in actual[method]["rows"]:
            for pool, key in zip(pools, transfer_keys(observation)):
                pool[key].append(observation)
        totals, stages, coverage = Counter(), Counter(), Counter()
        directions = defaultdict(Counter)
        for shape in directional_shapes:
            if shape["method"] != method or not shape["payload_bytes"]:
                continue
            for tier, (pool, key) in enumerate(zip(pools, transfer_keys(shape))):
                if key in pool:
                    observations = pool[key]
                    break
            amounts, phase_samples = [], defaultdict(list)
            direction = shape["direction"]
            phases = ("host_pack_seconds", "h2d_seconds") if direction == "h2d" else ("d2h_seconds", "host_scatter_seconds")
            byte_count = shape["payload_bytes"] / shape["calls"]
            for r in observations:
                byte_ratio = byte_count / r[direction+"_bytes"]
                for phase in phases:
                    phase_samples[phase].append(r[phase]*byte_ratio)
                # Half of observed round-trip Python/orchestration gap belongs
                # to each direction; this is a declared policy estimate.
                gap = max(0., r["wall_seconds"] - sum(r[p] for p in
                    ("host_pack_seconds", "h2d_seconds", "d2h_seconds", "host_scatter_seconds")))
                gap = .5*gap*shape["batch"]/r["batch"]
                phase_samples["direction_orchestration_gap_proxy_seconds"].append(gap)
                amounts.append(sum(r[p]*byte_ratio for p in phases)+gap)
            for field, value in (("point_seconds", statistics.median(amounts)),
                                 ("empirical_low_seconds", min(amounts)),
                                 ("empirical_high_seconds", max(amounts))):
                totals[field] += shape["calls"]*value
                directions[direction][field] += shape["calls"]*value
            totals[direction+"_bytes"] += shape["payload_bytes"]
            totals["directional_batch_visits"] += shape["calls"]
            coverage[f"tier{tier}"] += shape["batch"]*shape["calls"]
            for phase, values in phase_samples.items():
                stages[phase] += shape["calls"]*statistics.median(values)
        result[method] = dict(**dict(totals), stages=dict(stages), directions=dict(directions),
            coefficient_coverage_user_direction_visits=dict(coverage),
            measured_staging_allocation_max_seconds=max(r["stage_allocation_seconds"] for r in actual[method]["rows"]),
            measured_sample_backing_creation_seconds=actual[method]["backing_prepare_seconds"],
            measured_sample_backing_bytes=actual[method]["resident_backing_bytes"])
    return dict(methods=result, scope="Real 30k frontier paging byte counts, real final V5 state measurements. Complete-page phases scale by requested bytes for field subsets/unseen N; empirical observations are not bounds. Round-trip orchestration gap split equally between directions. GPU allocation is already inside H2D; pinned staging allocation is once, not each visit.",
        placement=("Bounded fixed-UID release windows, complete native+code input and output every UID/release including idle users, initial input and final drain. Transfer batches <=8; resident window size is explicit. Compute follows the same causal per-user path; reordering all requests in a release is an offline/saturated bulk condition, not online latency or an observed cache hit rate. No transfer/compute overlap." if window_boundary else
            "Primary capacity-feasible conditional policy pages at each actual operation boundary (batch<=8): append writes native+codes, conversions only codes, score read-only, rebuild generates native without loading obsolete KV. No overlap. Existing warm API packing retained, not removed as a speculative optimization. Miss fraction uniformly scales operation visits; not observed cache behavior." if operation_boundary else
            "Unimplemented request-local optimization only: page retained across append/prepare/score. Actual frontier advances the whole wave before scoring and cannot in general retain that wave on GPU. These smaller bytes must not be used as primary capacity-feasible deployment results."),
        missing="First creation of host backing for the whole population is not a measured deployment cost. Existing native backing belongs to common B; allocating the extra code backing requires an additional nonnegative delta_backing. One cold 32-user allocation is retained as evidence, not extrapolated as a proven bandwidth.")


def break_even(fixed, per_user_c, per_user_r):
    margin = per_user_r-per_user_c
    if margin <= 0:
        return dict(minimum_users=None, reason="C>=R: no finite population amortizes positive S", per_user_margin_seconds=margin)
    return dict(minimum_users=math.floor(fixed/margin)+1, per_user_margin_seconds=margin,
                reason="conditional S+N*C < N*R, fixed catalogue and workload mix")


def zero_extra_read_sensitivity(scenario, population=30000):
    """Counterfactual: method append/score cost equals matched native, not zero."""
    methods = scenario["methods"]
    method, native = methods["relay"], methods["rebuild"]
    method_factor = method["sample_accounting"]["normalization_to_bulk_median"]
    native_factor = native["sample_accounting"]["normalization_to_bulk_median"]
    conversion = sum(method["operations"][op]["point_seconds"]
                     for op in ("materialize", "anchorize")) * method_factor
    reconstruction = native["operations"]["rebuild"]["point_seconds"] * native_factor
    full = scenario["primary_complete_scenario"]
    extra_page = full["additional_state_paging_seconds"]["relay"] - full["additional_state_paging_seconds"]["rebuild"]
    variable_saving = reconstruction-conversion-extra_page
    method_fixed = scenario["fixed_S_midpoint_seconds"]
    native_fixed = scenario.get("fixed_baseline_seconds",0.)
    fixed = method_fixed-native_fixed
    native_total = full["serial_resource_rebuild_seconds"]
    total = native_total-variable_saving+fixed
    return dict(scope="Optimistic counterfactual within the existing scheduler and measured coefficient model: Relay append and score take exactly matched-native time, while all conversion phases, fixed preparation and actual paging policy remain. Not an achieved implementation, hardware lower bound or statistical bound.",
        population=population, fixed_S_seconds=method_fixed, fixed_native_seconds=native_fixed,
        fixed_difference_seconds=fixed,
        avoided_full_rebuild_seconds=reconstruction,
        retained_full_conversion_seconds=conversion,
        retained_extra_paging_seconds=extra_page,
        maximum_allowed_fixed_S_for_reference_break_even_seconds=variable_saving+native_fixed,
        method_total_seconds=total, native_total_seconds=native_total,
        saved_percent=100*(native_total-total)/native_total,
        conditional_break_even=break_even(fixed,(native_total-native_fixed-variable_saving)/population,(native_total-native_fixed)/population),
        asymptotic_total_time_saving_percent=100*variable_saving/(native_total-native_fixed),
        assumptions="Same Medium model, catalogue, 3-release horizon and measured window/batch/activity distribution at every population; not measured Large/Max extrapolation.")


def native_catalog(path):
    source = read(path)
    return dict(scope="Measured native-only Medium/Large/Max V2/V5 operator references. Large/Max correction, full service traces and end-to-end method comparisons are absent.",
        units=[{k:r[k] for k in ("scale","version","operation","history_length","batch","median_seconds","min_seconds","max_seconds")}
               for r in source["units"]],
        batch8_scope="Native probe B8 repeats one history; not actual eight-user scheduling. Not substituted into corrected session totals.")


def jit_boundary(directory):
    warmup_path = directory/"backend_warmup.json"
    return dict(stable_cycle="The main S+service prediction assumes required backend kernels already compiled for the unchanged model/kernel geometry. Parameter-only publication does not by itself require compilation.",
        measured_backend_warmup=read(warmup_path) if warmup_path.exists() else None,
        scope="Warmup intervals execute full reference trajectories and may contain first compilation, allocations, caches and normal model work. They are not isolated compilation costs and are not added once per model release.",
        cold_start="Add a separately reported nonnegative delta_JIT to both affected method setups if the required kernel cache is unavailable. Pure compile seconds are not separated; stable-cycle estimates do not establish cold-start break-even.",
        known_failure="The earlier projection32 attempt1 was interrupted during shape-specialization warmup (about291s without completion). This is retained engineering evidence, not a warm service coefficient or a finite measured cold compile budget.")


def reader_padding(batches):
    """Extra method-only logical dot geometry of the current fused projection reader."""
    totals, operations = Counter(), defaultdict(Counter)
    for row in batches:
        if row["method"] != "relay" or row["operation"] not in ("score","append") or not row["coded_rows"]:
            continue
        b,m,c,k = (row[name] for name in ("batch","width","coded_rows","calls"))
        scale = k*6*6*b
        if m <= 4:
            # These are explicit products/reductions, not tensor-dot tiles.
            totals["vector_coded_queries"] += k*b*m
            totals["vector_valid_pair_multiply_accumulate_equivalent_flops"] += 4*scale*m*c*6
            totals["vector_padded_pair_multiply_accumulate_equivalent_flops"] += 4*scale*m*((c+63)//64)*64*8
            continue
        mp,cp = ((m+15)//16)*16,((c+63)//64)*64
        pair_valid, pair_tile = 4*scale*m*c*6,4*scale*mp*cp*16
        project_valid,project_tile = 4*scale*m*32*6,4*scale*mp*32*16
        for name,value in (("history_code_dot_useful_flops",pair_valid),
                           ("history_code_dot_tile_flops",pair_tile),
                           ("key_projection_and_value_decode_useful_flops",project_valid),
                           ("key_projection_and_value_decode_tile_flops",project_tile),
                           ("additional_reader_dot_padding_flops",pair_tile+project_tile-pair_valid-project_valid)):
            totals[name] += value
            operations[row["operation"]][name] += value
    return dict(**dict(totals),by_operation=dict(operations),
        geometry="M>4: BM16,BN64,BR16,BD32; key projection and value decode once per query tile, score/latent code dots only ceil(maxcoded/64) blocks. Count all six layers/heads and actual batch/query geometry. M<=4 uses vector products/reductions with rank8 masks; shown separately, no tensor-dot claim.",
        boundary="Only method-specific logical dots added to the mathematical ledger. Does not count TF32x3 instruction expansion, scalar/reduction padding, gate GEMM backend padding, common native attention/model padding, or machine instruction FLOPs.")


def arithmetic_views(batch_ledger):
    parts = read(ROOT/"results/system_study_2026_10/cost/scaling/attempt1/summary.json")["branches"]["current"]["decomposition"]
    one_shared = (parts["catalogue_linear_preparation_flops"]+parts["small_matrix_preparation_flops"])/parts["replicas"]
    fixed = parts["fixed_components_flops"]["correction_calibration_including_origin_once"]+one_shared
    mathematical = fixed+parts["variable_at_reference_flops"]+batch_ledger["execution_extra"]["additional_coalescing_flops"]
    conversion = batch_ledger["execution_extra"]["conversion_padding_extra_dot_flops"]
    reader = reader_padding(batch_ledger["batches"])
    native,rebuilt,source = (parts[k] for k in ("common_native_at_reference_flops","rebuild_at_reference_flops","source_reconstruction_fixed_flops"))
    views = {}
    for name,added in (("mathematical_effective",mathematical),("with_conversion_tile_sensitivity",mathematical+conversion),
                       ("with_conversion_and_reader_dot_tile_sensitivity",mathematical+conversion+reader["additional_reader_dot_padding_flops"])):
        views[name] = dict(added_flops=added,added_percent_of_mathematical_rebuild=100*added/rebuilt,
            with_source_reconstruction_added_percent=100*(added+source)/rebuilt,
            saved_percent_against_mathematical_full_service=100*(rebuilt-added)/(native+rebuilt))
    return dict(preparation="single publisher/one serving worker, shared compile once; calibration64fit and fixed origin unchanged",
        reference_population=30000,views=views,reader_padding=reader,
        additional_coalescing_mathematical_flops=batch_ledger["execution_extra"]["additional_coalescing_flops"],
        additional_conversion_dot_padding_flops=conversion,
        boundary="Tile sensitivity adds known method dot padding to the mathematical numerator and keeps the mathematical Rebuild denominator. It is not complete executed hardware FLOPs. Logical coalescing is included before tile padding; no double counting.")


def cross_sample_prediction(reference_path, target_summary, target_calls):
    """Check smaller-sample coefficients on a separately timed larger trace."""
    reference, reference_calls = session_observations(reference_path)
    coefficients = Coefficients(reference_calls,measured_graph_policy=reference["time_model_graph_policy"])
    methods = {}
    for method in ("relay","rebuild"):
        shapes = [dict(row,calls=1) for row in target_calls if row["method"]==method]
        prediction = project_batches(shapes,coefficients,method)
        norm = sample_accounting(reference,reference_calls,coefficients,method,8)
        observed = [r["service_wall_seconds"] for r in target_summary["time_units"] if r["method"]==method and r["execution"]=="batch8"]
        predicted = prediction["point_seconds"]*norm["normalization_to_bulk_median"]
        methods[method] = dict(predicted_target_phase_seconds=prediction["point_seconds"],
            actual_target_phase_seconds=sum(r["wall_seconds"] for r in target_calls if r["method"]==method),
            predicted_target_bulk_seconds=predicted,actual_target_bulk_seconds=observed,
            relative_error_to_target_bulk_median_percent=100*(predicted/statistics.median(observed)-1),
            coverage=prediction["weighted_coefficient_coverage"])
    return dict(reference_users=len(reference["users"]),target_users=len(target_summary["users"]),
        overlapping_user_count=len(set(reference["users"]) & set(target_summary["users"])),methods=methods,
        scope="Small-trace timing coefficients and its normalization are frozen for this check; only larger-trace executed shapes are supplied. No target timings are used to refit this validation prediction. Distinct run/environment effects remain; this is not quality evaluation.",
        source_bindings=[binding(reference_path/"summary.json"),binding(reference_path/"phases.parquet")])


def compose(output, shape_path, session_path, shared_path, native_path, *, batch_path=None, transfer_path=None, validation_reference=None, v4_calibration_report=None, development_only=False):
    shape_ledger = read(shape_path)
    session, calls = session_observations(session_path)
    if not session["time_model_phase_scope"]["separately_warmed"]:
        raise ValueError("population steady-cycle projection requires separately warmed phase measurements")
    session_config = read(session_path/"configuration.json")
    if validation_reference:
        reference_config = read(validation_reference/"configuration.json")
        # An earlier native append backend must not validate a newer matched
        # fused append implementation. Source changes require explicit review.
        path = "experiments/archive/system_study/batched_session.py"
        if reference_config.get("source_sha256",{}).get(path) != session_config.get("source_sha256",{}).get(path):
            raise ValueError("cross-sample timing check uses different batch execution source; omit or supply matched-backend reference")
    coefficients = Coefficients(calls,graph_enabled=session_config["config"].get("graph_compute",False),
                                measured_graph_policy=session["time_model_graph_policy"])
    graph_setups = graph_preparation(session)
    shared = shared_preparation(shared_path)
    v4_alternative = v4_calibration_substitution(v4_calibration_report,shared_path) if v4_calibration_report else None
    shared["measured_conversion_weight_wrap_seconds"] = session.get("conversion_wrap_seconds_included_in_shared_load", 0.)
    shared["method_input_load_included_once_seconds"] = shared["standalone_method_input_load_seconds"]
    supplement = shared["measured_conversion_weight_wrap_seconds"]+shared["method_input_load_included_once_seconds"]
    for key in ("fixed_S_lower_seconds", "fixed_S_upper_seconds"):
        shared[key] += supplement
    for name,strategy in shared["preparation_strategies"].items():
        copies = 4 if name == "four_independent_preparers" else 1
        strategy["method_inputs_and_wrap_supplement_seconds"] = copies*supplement
        for key in ("fixed_S_lower_seconds","fixed_S_upper_seconds"):
            strategy[key] += copies*supplement
    batch_ledger = read(batch_path) if batch_path else None
    if batch_ledger and batch_ledger["users"] != shape_ledger["users"]:
        raise ValueError("batch and individual population differ")
    graph_coverage = graph_shape_coverage(batch_ledger["batches"],graph_setups,
        graph_enabled=coefficients.graph_enabled,policy=session["time_model_graph_policy"]) if batch_ledger else None
    window_users = batch_ledger.get("window_users") if batch_ledger else None
    host = project_host_traffic(batch_ledger, transfer_path,window_boundary=bool(window_users)) if batch_ledger and transfer_path else None
    consumer_host = project_host_traffic(batch_ledger, transfer_path, operation_boundary=False) if host else None
    scenarios = {}
    for maximum in ((8,) if batch_ledger else (1, 8)):
        methods = {}
        for method in ("relay", "rebuild"):
            projection = project_batches(batch_ledger["batches"], coefficients, method) if batch_ledger else project_calls(shape_ledger["shapes"], coefficients, method, maximum)
            sample = sample_accounting(session, calls, coefficients, method, maximum)
            projection["sample_accounting"] = sample
            projection["normalized_point_seconds"] = projection["point_seconds"] * sample["normalization_to_bulk_median"]
            projection["normalized_empirical_low_seconds"] = projection["empirical_low_seconds"] * sample["normalization_low"]
            projection["normalized_empirical_high_seconds"] = projection["empirical_high_seconds"] * sample["normalization_high"]
            methods[method] = projection
        n0 = shape_ledger["users"]
        c, r = (methods[m]["normalized_point_seconds"]/n0 for m in ("relay", "rebuild"))
        s = (shared["fixed_S_lower_seconds"]+shared["fixed_S_upper_seconds"])/2 + graph_setups["relay"]["point_seconds"]
        native_s = graph_setups["rebuild"]["point_seconds"]
        difference_s = s-native_s
        grid = [dict(users=n, relay_S_plus_NC_seconds=s+n*c, rebuild_NR_seconds=native_s+n*r,
            saved_seconds=n*(r-c)-difference_s, saved_percent=100*(n*(r-c)-difference_s)/(native_s+n*r))
            for n in (1000,4000,10000,30000,79681,200000)]
        scenarios[f"maximum_batch{maximum}"] = dict(methods=methods, per_user_C_seconds=c,
            per_user_R_seconds=r, fixed_S_midpoint_seconds=s, fixed_baseline_seconds=native_s,
            fixed_difference_seconds=difference_s, break_even=break_even(difference_s,c,r), population_grid=grid)
        point = scenarios[f"maximum_batch{maximum}"]
        raw_c, raw_r = (methods[m]["point_seconds"]/n0 for m in ("relay", "rebuild"))
        point["unnormalized_phase_sensitivity"] = dict(per_user_C_seconds=raw_c,per_user_R_seconds=raw_r,
            relay_S_plus_service_seconds=s+n0*raw_c,rebuild_service_seconds=native_s+n0*raw_r,
            saved_percent=100*(n0*(raw_r-raw_c)-difference_s)/(native_s+n0*raw_r),
            break_even=break_even(difference_s,raw_c,raw_r),
            scope="Direct measured synchronous-phase coefficients before bulk normalization. Difference from main prediction exposes sample orchestration/measurement-order sensitivity; neither is independent population validation.")
        point["paging_scenarios"] = []
        for miss in ((1.,) if window_users and host else (0.,.1,1.) if host else (0.,)):
            page = {m: miss*host["methods"][m]["point_seconds"] if host else 0. for m in methods}
            cp, rp = c+page["relay"]/n0, r+page["rebuild"]/n0
            totalc, totalr = s+n0*cp, native_s+n0*rp
            low_c = shared["fixed_S_lower_seconds"]+graph_setups["relay"]["low_seconds"]+methods["relay"]["normalized_empirical_low_seconds"]
            high_c = shared["fixed_S_upper_seconds"]+graph_setups["relay"]["high_seconds"]+methods["relay"]["normalized_empirical_high_seconds"]
            low_r, high_r = (graph_setups["rebuild"][b+"_seconds"]+methods["rebuild"]["normalized_empirical_"+b+"_seconds"] for b in ("low","high"))
            if host:
                low_c += miss*host["methods"]["relay"]["empirical_low_seconds"]
                high_c += miss*host["methods"]["relay"]["empirical_high_seconds"]
                low_r += miss*host["methods"]["rebuild"]["empirical_low_seconds"]
                high_r += miss*host["methods"]["rebuild"]["empirical_high_seconds"]
            four = shared["preparation_strategies"]["publisher_once_four_copies"]
            four_s = (four["fixed_S_lower_seconds"]+four["fixed_S_upper_seconds"])/2
            point["paging_scenarios"].append(dict(assumed_uniform_missed_operation_fraction=miss,
                paging_policy=f"mandatory_complete_release_window_{window_users}" if window_users else "uniform_operation_miss_sensitivity",
                resident_service_seconds={m:methods[m]["normalized_point_seconds"] for m in methods},
                additional_state_paging_seconds=page, per_user_C_seconds=cp, per_user_R_seconds=rp,
                serial_resource_relay_seconds=totalc, serial_resource_rebuild_seconds=totalr,
                saved_percent=100*(totalr-totalc)/totalr,
                empirical_relay_interval_seconds=[low_c,high_c], empirical_rebuild_interval_seconds=[low_r,high_r],
                method_fixed_setup_seconds=s,native_fixed_setup_seconds=native_s,
                break_even=break_even(difference_s,cp,rp),
                ideal_four_balanced_workers=None if coefficients.graph_enabled else dict(relay_seconds=four_s+n0*cp/4, rebuild_seconds=n0*rp/4,
                    scope="Conditional no-overlap upper scheduling ideal: equal service work, no CPU/PCIe contention; offline calibration S remains once and is not divided. Not a distributed measurement or loaded wall-clock bound.")))
        point["primary_complete_scenario"] = point["paging_scenarios"][0 if window_users else -1]
        if graph_coverage and any(v["unmeasured_additional_preparation_required"] for v in graph_coverage.values()):
            primary = point["primary_complete_scenario"]
            extra = {m:graph_coverage[m]["additional_setup_mean_shape_seconds_proxy"] for m in methods}
            tc = primary["serial_resource_relay_seconds"]+extra["relay"]
            tr = primary["serial_resource_rebuild_seconds"]+extra["rebuild"]
            point["missing_graph_setup_sensitivity"] = dict(additional_setup_seconds_mean_shape_proxy=extra,
                serial_resource_relay_seconds=tc,serial_resource_rebuild_seconds=tr,saved_percent=100*(tr-tc)/tr,
                scope="Main totals exclude unmeasured additional shape setup. This separate diagnostic estimates it using each method's mean measured preparation per shape, not a measured full-population setup or reliable bound. Service shape proxies remain separate.")
        point["zero_extra_read_counterfactual"] = zero_extra_read_sensitivity(point,n0)
        if v4_alternative:
            primary = point["primary_complete_scenario"]
            change = v4_alternative["change_in_method_fixed_seconds"]
            new_total = primary["serial_resource_relay_seconds"]+change
            native_total = primary["serial_resource_rebuild_seconds"]
            point["v4_only_calibration_sensitivity"] = dict(**v4_alternative,
                method_fixed_setup_seconds=s+change,native_fixed_setup_seconds=native_s,
                serial_resource_relay_seconds=new_total,serial_resource_rebuild_seconds=native_total,
                saved_percent=100*(native_total-new_total)/native_total,
                break_even=break_even(difference_s+change,primary["per_user_C_seconds"],primary["per_user_R_seconds"]))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="development_only_time_prescreen" if development_only else "conditional_time_projection_real_frontier" if batch_ledger else "conditional_time_projection_resident_coefficients", created_utc=datetime.now(timezone.utc).isoformat(),
        reference_population=shape_ledger["users"], task="Medium V3/V4/V5 next-item stride16 candidates16",
        scenarios=scenarios, shared_preparation=shared, native_model_references=native_catalog(native_path),
        sample_users=len(session["users"]), phase_measurement_scope=session["time_model_phase_scope"],
        measured_execution_configuration=session_config["config"],
        measured_graph_dispatch_policy=session["time_model_graph_policy"],
        graph_preparation=graph_setups,
        population_graph_shape_coverage=graph_coverage,
        graph_preparation_flops_sensitivity=graph_preparation_flops(graph_setups),
        recorded_execution_source_sha256=session_config.get("source_sha256"),
        real_frontier_geometry=dict(users=batch_ledger["users"],checks=batch_ledger["checks"],batch_mix=batch_ledger["batch_mix"],execution_extra=batch_ledger["execution_extra"]) if batch_ledger else None,
        resident_window_users=window_users,
        host_paging_projection=host,
        unimplemented_request_local_paging_potential=consumer_host,
        compilation_boundary=jit_boundary(session_path),
        arithmetic_views=arithmetic_views(batch_ledger) if batch_ledger else None,
        cross_sample_timing_check=cross_sample_prediction(validation_reference,session,calls) if validation_reference else None,
        source_bindings=[binding(p) for p in (shape_path, session_path/"summary.json",session_path/"phases.parquet",session_path/"configuration.json",
            shared_path/"summary.json",shared_path/"calibration_timing.json",native_path,Path(__file__))],
        distinctions=["Actual phase/body wall times include CPU work, launch gaps and synchronization; neither they nor CUDA event spans are GPU busy seconds.",
            "Pack_h2d combines CPU packing, device gathers and small input copies; it cannot be relabelled as pure CPU or transfer time. All three phase intervals are sequential and non-overlapping.",
            "30k population counts are exact, hardware time coefficients are sampled and shape-bucket fallbacks are explicit empirical projections.",
            "With frontier input, deployment batch occupancy and max-coded prefix are exactly replayed on CPU. Unmeasured hardware shapes use flagged same-batch nearby pools. Saturated frontier scheduling does not establish an arrival-time batch or queue policy.",
            "30k final K/V+code do not fit four48GB cards. Host paging sensitivity adds measured directional transfer/packing coefficients to exact byte counts; this is not integrated backing-tier service or an observed miss rate.",
            "S plus population service is additive API-work seconds under serialized execution assumptions, not loaded multi-GPU makespan or p99.",
            "Foundation training, initial V2 cache construction and common foundation deployment add the same B to both sides; absolute savings unchanged, percentages diluted.",
            "Population expansion holds Medium model, catalogue, calibration users, three-release horizon and activity distribution fixed; 79681/200000 rows are not Large/Max workload evidence."])
    if batch_path:
        result["source_bindings"].append(binding(batch_path))
    if transfer_path:
        result["source_bindings"].append(binding(transfer_path))
    if v4_calibration_report:
        result["source_bindings"].append(binding(v4_calibration_report))
    if (session_path/"backend_warmup.json").exists():
        result["source_bindings"].append(binding(session_path/"backend_warmup.json"))
    for method in ("relay","rebuild"):
        if (session_path/f"{method}_calls.json").exists():
            result["source_bindings"].append(binding(session_path/f"{method}_calls.json"))
    write(output/"summary.json",result)
    lines=["# 开发预筛：真实操作账与小样本时间系数" if development_only else "# 真实操作账与服务时间条件预测", "", f"30k用户使用重新核对的真实联合形状；服务系数来自真实{len(session['users'])}用户会话。计时样本为实测，人口总时间为条件预测。" + (" 本次仅供定位瓶颈，不能作为最终128人系数或30k完整性能结论。" if development_only else ""), "",
        "主资源口径为单个A40 worker与host backing；页策略在下表明确区分。S计入历史校准、来源/教师生成、原起点、新测全目录编译，以及共享状态一次D2H和该worker一次H2D；即使同GPU可直接保留也保守计这次host往返。训练模型加载已在历史命令内，不重复加独立探针的共同模型加载。", "",
        f"固定S约{shared['fixed_S_lower_seconds']:.3f}–{shared['fixed_S_upper_seconds']:.3f}秒；本范围仅来自原起点前缀计时边界，是工作流构成口径范围，不是测量置信区间；另有未单列的最终checkpoint保存δS。", "",
        f"图准备分别计入两边固定项：Relay {graph_setups['relay']['point_seconds']:.3f}秒、Rebuild {graph_setups['rebuild']['point_seconds']:.3f}秒。每法一次图集准备取实际重复样本的中位数，整条预热trace没有作为S相加；图准备包含三次warm模型调用与一次capture。", "",
        "| 场景 | 每用户C秒 | 每用户R秒 | 30k方法固定项+NC秒 | 30k重建固定项+NR秒 | 净省% | 盈亏人数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name,s in scenarios.items():
        row=next(r for r in s["population_grid"] if r["users"]==30000)
        lines.append(f"| {name} | {s['per_user_C_seconds']:.6f} | {s['per_user_R_seconds']:.6f} | {row['relay_S_plus_NC_seconds']:.2f} | {row['rebuild_NR_seconds']:.2f} | {row['saved_percent']:.2f}% | {s['break_even']['minimum_users'] or '无有限盈亏点'} |")
    lines += ["", "归一敏感性：" + "；".join(
        f"{name}阶段原值不归一时净省{s['unnormalized_phase_sensitivity']['saved_percent']:.2f}%，relay/rebuild的样本bulk归一系数分别{s['methods']['relay']['sample_accounting']['normalization_to_bulk_median']:.4f}/{s['methods']['rebuild']['sample_accounting']['normalization_to_bulk_median']:.4f}"
        for name,s in scenarios.items()) + "。若两种口径结论明显不同，应先检查交错计时与环境波动，不能只选有利口径。"]
    lines += ["", "C、R均包含本方法完整原生追加/查询与所有方法特有工作，因此共同服务已在两边计入。若C≥R，扩大用户数不能消除逐用户时间劣势。", "",
        "每个预测保留compute、pack/H2D、scatter各项，及精确匹配/近似桶覆盖。近似区间由实测调用范围构成，不是数学上下界或统计置信区间。用同批实测用户完整未插桩时间核对并调整阶段同步开销；这是样本内计量核对，不是独立预测验证。", "",
        "这些值是串行API工作量预测，不能当GPU忙时、四卡墙钟或排队延迟。原生Large/Max V2/V5仅作为算子计时参照保留，尚无其corrector全方法结果。", "",
        "30k全量状态超过四张48GB卡容量。小输入H2D已在pack计时里；主host情景按实际operation边界迁移K/V与码，每批≤8：转换只写回码、追加写回K/V和码、评分只读、重建不读旧K/V。不能假定整个frontier wave都能驻GPU。", ""]
    if batch_ledger:
        lines += [f"CPU直接执行30k的frontier调度形状，逐用户操作与旧账一致；实际批尾、最大码前缀及零尾工作来自30k，未按{len(session['users'])}人平均放大。不同人口的表格仍假设沿用30k的活动与批次分布；没有重放其他人口。", ""]
    if host:
        if window_users:
            lines += [f"主执行口径为固定{window_users} UID驻留窗口。每release在窗口内完成用户全部因果请求、drain和码交接后卸载，发布间有屏障。它适用于离线或饱和bulk处理；不能把这类跨用户重排宣称为实时到达延迟收益。每用户每release完整K/V与码入出均计，含空闲用户、首次输入与最终输出。", ""]
        else:
            lines += ["完整保守口径为每operation全量页入、dirty页出（100% operation miss），不依赖全wave驻留。0%/10%仅为尚未测得的命中率敏感性；其更好结果不能替代容量可行口径。全frontier批填充也以饱和可重排请求为条件，不代表真实到达下的批率与延迟。", ""]
        lines += ["| host策略 | 30k方法 S+服务+搬运秒 | 重建服务+搬运秒 | 净省% | 盈亏人数 |", "|---|---:|---:|---:|---:|"]
        for row in scenarios["maximum_batch8"]["paging_scenarios"]:
            policy = f"{window_users} UID窗口全量入出" if window_users else f"operation miss {row['assumed_uniform_missed_operation_fraction']:.0%}"
            lines.append(f"| {policy} | {row['serial_resource_relay_seconds']:.2f} | {row['serial_resource_rebuild_seconds']:.2f} | {row['saved_percent']:.2f}% | {row['break_even']['minimum_users'] or '无有限盈亏点'} |")
        lines += ["", "真实V5状态page-in/out、CPU pack/scatter和GPU分配均已有实测；子字段/未覆盖形状按字节和邻桶作经验预测，无搬运重叠。全人口host backing的初始分配尚无实测：共同原生backing属于B，额外码backing另加非负δbacking；不把32人冷分配耗时冒充大规模带宽。operation miss敏感性（若列出）只是均匀假设，不是观测命中率。", "",
            "一次请求的追加与评分共享驻留页可进一步减少迁移，但需要改变当前frontier的执行局部性。该尚未实现的潜力另列在JSON，未用于主盈亏结论。", ""]
    if v4_alternative:
        alternative=scenarios["maximum_batch8"]["v4_only_calibration_sensitivity"]
        lines += [f"仅替换V4校准的独立敏感性：旧{alternative['old_V4_seconds']:.3f}秒换为实际合批完整路径{alternative['measured_batched_V4_seconds']:.3f}秒，其他release、来源/教师、原起点和图准备保持原值。主分页场景方法{alternative['serial_resource_relay_seconds']:.2f}秒、Rebuild {alternative['serial_resource_rebuild_seconds']:.2f}秒，净省{alternative['saved_percent']:.2f}%。没有把joint阶段加速倍数推广到整个S。", ""]
    lines += ["此处S保留实际离线重放来源的保守工作流；它与数学账source_available的默认情景不同。若从已维护的线上用户状态直接抽样，可避免重复来源replay，但仍需教师重建、快照、必要码物化和搬运；当前整包时间不能直接扣除，也没有按FLOPs分摊秒数。", ""]
    if coefficients.graph_enabled:
        lines += ["本次图结果只给单worker。图驻留空间按version/op/B/W/有无码的有限形状组合增长，不按UID数增长；正码前缀作为动态图输入共享。没有把时间直接除4称为四worker墙钟。每版模型/adapter地址绑定需要capture，内核JIT可复用与图无需重建是两回事。CUDA capture记录工作但不执行设备内核，因此冷准备FLOPs只计三次warm前向，后续正常replay不重复计。", ""]
        if graph_coverage:
            lines += ["| 图形状 | 30k所需 | 样本已测 | 所需而未测 |", "|---|---:|---:|---:|"]
            for m,g in graph_coverage.items():
                lines.append(f"| {m} | {g['required_shapes']} | {g['observed_shapes']} | {len(g['missing_shapes'])} |")
            lines += ["", "若存在所需而未测的新shape，主表固定项尚需额外δS_graph与workspace，不能视为完整冷准备。JSON按实际shape列出缺口，并提供每图平均准备/空间的粗略敏感性，后者不是实测或严格界。未覆盖服务shape仍只能使用同backend的显式邻桶；短历史eager不会借用graph系数。", "",
                "若目标小批没有同backend的小批证据，开发预筛将一个已测较大批的整次耗时赋给目标调用，并另列两方命中数与时间份额。GPU batch延迟并非严格单调，该代理不是保守上界；若高估Native，方法相对收益会偏乐观。正式结论需要检查其敏感性或补足关键缺测。", ""]
    else:
        lines += ["主口径1worker仅生成与加载一份共享状态，FLOPs已按一次编译重价。旧4独立副本FLOPs账、publisher编译一次分发4份、4份独立生成resident时间均另列，不能混称同一部署。四worker服务/4仅为均衡且无CPU/PCIe争用的敏感性，并加四份实际串行分发；S中的校准不除4，也不声称实测群集墙钟。", ""]
    lines += ["主时间对应kernel已预编译的稳定周期。首次JIT另加δJIT；同结构权重发布不重复编译。backend_warmup保留完整实测，但其中包含正常整条服务、分配和编译，不能把它全部算作纯编译或每release开销。先前projection32约291秒未完成的形状特化暖机已保留中止证据，不用于warm系数。", ""]
    (output/"summary.md").write_text("\n".join(lines))
    print(json.dumps(dict(status=result["status"],output=str(output),scenarios={k:v["break_even"] for k,v in scenarios.items()})),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-ledger", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--session", type=Path, default=ROOT/"results/system_study_2026_10/service/batched_session32/attempt1")
    parser.add_argument("--shared", type=Path, default=ROOT/"results/system_study_2026_10/service/shared_timing/attempt1")
    parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--batches", type=Path)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--arithmetic-only",action="store_true")
    parser.add_argument("--validation-reference",type=Path)
    parser.add_argument("--v4-calibration-report",type=Path)
    parser.add_argument("--development-only",action="store_true")
    args = parser.parse_args()
    if args.arithmetic_only:
        args.output.mkdir(parents=True,exist_ok=False)
        value=arithmetic_views(read(args.batches))
        value["source_bindings"]=[binding(p) for p in (args.batches,Path(__file__),ROOT/"experiments/archive/system_study/fused_query_projection_kernel.py")]
        write(args.output/"summary.json",value)
        (args.output/"summary.md").write_text("# 数学FLOPs与显式tile敏感性\n\n"+"\n".join(f"- {name}: 新增/数学Rebuild={r['added_percent_of_mathematical_rebuild']:.6f}%；含来源={r['with_source_reconstruction_added_percent']:.6f}%。" for name,r in value["views"].items())+"\n\n后两项为已知method dot补齐敏感性，未计全部算子/机器指令；不称完整执行FLOPs。原生共同工作与Rebuild分母保持数学有效口径。\n")
        print(value["views"])
    elif args.build_ledger:
        build_shapes(args.output, args.threads)
    else:
        compose(args.output,args.shapes,args.session,args.shared,args.native,batch_path=args.batches,transfer_path=args.transfer,validation_reference=args.validation_reference,
                v4_calibration_report=args.v4_calibration_report,development_only=args.development_only)
