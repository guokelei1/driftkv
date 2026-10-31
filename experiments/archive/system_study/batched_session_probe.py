#!/usr/bin/env python3
"""Causal distinct-user batching against the existing synchronous sessions."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from functools import partial
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import weakref

ROOT = Path(__file__).resolve().parents[3]

import torch

from experiments.archive.system_study import arrival_replay as arrival
from experiments.archive.system_study.batched_session import BatchedSessions, PhaseRecorder
from experiments.archive.system_study.model_replay import compare_snapshot, state_snapshot, counter_rows


class SessionPrefix:
    """Adapt the existing CPU-ID session contract to the CUDA-only fused core."""
    def __init__(self, prefix):
        self.prefix = prefix

    def __getattr__(self, name):
        return getattr(self.prefix, name)

    def _run(self, name, cache, state, item_ids):
        ids = torch.as_tensor(item_ids, device=self.prefix.P.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids[None]
        return getattr(self.prefix, name)(cache, state, ids)

    def materialize(self, cache, state, item_ids):
        return self._run("materialize", cache, state, item_ids)

    def anchorize(self, cache, state, item_ids):
        return self._run("anchorize", cache, state, item_ids)


def frontiers(trace, version):
    """One next request per distinct uid; never cross a publication boundary."""
    queues = defaultdict(deque)
    for event in trace:
        if event.kind == "recommendation" and event.version == version:
            queues[event.uid].append(event)
    while any(queues.values()):
        yield [queue.popleft() for queue in queues.values() if queue]


def snapshot(session):
    value = state_snapshot(session)
    value["metadata"]["native_query_ledger"] = counter_rows(getattr(session, "query_ledger", {}))
    return value


def run_serial(handler, trace, *, on_boundary=None):
    for event in trace:
        if event.kind != "behavior":
            handler(event)
        if on_boundary is not None and event.kind in ("publication", "drain"):
            on_boundary(event.sequence, handler)


def run_batched(handler, trace, max_batch, *, recorder=None, on_boundary=None, coalesce_zero_tail=False,
                projection_fused=False, native_fused_append=False, fused_code_append=False, graph_compute=None):
    engine = BatchedSessions(handler.sessions, max_batch=max_batch, recorder=recorder,
                             coalesce_zero_tail=coalesce_zero_tail, projection_fused=projection_fused,
                             native_fused_append=native_fused_append, fused_code_append=fused_code_append,
                             graph_compute=graph_compute)
    for event in trace:
        if event.kind == "publication":
            version = event.version
            engine.publish(handler.models[version], int(version[1:]), event.logical_time,
                adapter=handler.adapters[version] if handler.method == "relay" else None,
                prefix=handler.prefixes[version] if handler.method == "relay" else None,
                rebuild=handler.method == "rebuild", event_sequence=event.sequence)
            handler.current_version = version
            if on_boundary is not None:
                on_boundary(event.sequence, handler)
            for wave in frontiers(trace, version):
                engine.request_context = {e.uid: dict(logical_time=e.logical_time, event_sequence=e.sequence) for e in wave}
                scores = engine.score({e.uid: (e.logical_time, e.requests) for e in wave})
                if set(scores) & set(handler.scores):
                    raise ValueError("duplicate candidate identity")
                handler.scores.update(scores)
        elif event.kind == "drain":
            engine.request_context = {u: dict(logical_time=event.logical_time, event_sequence=event.sequence)
                                      for u in handler.sessions}
            engine.advance({u: event.logical_time for u in handler.sessions})
            if on_boundary is not None:
                on_boundary(event.sequence, handler)
    return engine


def summarize_calls(calls):
    counts = Counter((r["operation"], r["batch"]) for r in calls)
    return [dict(operation=operation, batch=batch, calls=count, user_operations=batch*count)
            for (operation, batch), count in sorted(counts.items())]


def summarize_graph_coverage(calls):
    counts = Counter((r["operation"], r.get("compute_backend", "eager"), r["batch"]) for r in calls
                     if r["operation"] in ("append", "score", "rebuild"))
    return [dict(operation=op, backend=backend, batch=batch, calls=n, user_operations=n*batch)
            for (op, backend, batch), n in sorted(counts.items())]


@torch.no_grad()
def run(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config = json.loads(args.config.read_text())
    selection_path = ROOT / config["selection"]
    selection = json.loads(selection_path.read_text())
    users = selection["selected_uids"]
    panel_path, candidate = ROOT / config["panel"], ROOT / config["candidate"]
    if len(users) != config["expected_users"] or \
            arrival.sha256(panel_path / "binding.json") != selection["panel_sha256"] or \
            arrival.sha256(candidate / "freeze.json") != selection["freeze_sha256"]:
        raise ValueError("registered panel bindings changed")
    panel, roles, _ = arrival.load_panel(panel_path)
    if not set(users).issubset(roles["evaluation"]):
        raise ValueError("batch probe must reuse engineering development users")
    device = arrival.setup_device(args.gpu, threads=2)
    if "gpu_memory_fraction" in config:
        torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device)
    settings = dict(created_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), command=sys.argv,
        config=config, users=users, device=torch.cuda.get_device_name(device),
        source_sha256={str(p.relative_to(ROOT)): arrival.sha256(p) for p in (
            Path(__file__), Path(__file__).with_name("batched_session.py"),
            ROOT / "src/hstu_kvcache/methods/cachetide/session.py", ROOT / "src/hstu_kvcache/methods/cachetide/state.py")},
        input_sha256=dict(config=arrival.sha256(args.config), selection=arrival.sha256(selection_path),
            panel=arrival.sha256(panel_path / "binding.json"), freeze=arrival.sha256(candidate / "freeze.json")),
        scope="Real distinct-user bulk execution; multiple model versions resident; no online arrival latency, capacity, quality-improvement or full deployment time claim",
        scheduling="Within each version, one pending request per uid; reorder independent users only; publications are barriers",
        phase_scope="Extra synchronization per pack/compute/scatter; separate uninstrumented full-trace timing")
    arrival.write_json(args.output / "configuration.json", settings)
    arrival.progress(args.output, "load_assets")
    started = time.perf_counter()
    models = arrival.load_models(panel, device)
    versions = [v for v, _ in arrival.RELEASES]
    adapters, weights, _ = arrival.load_candidates(candidate / "calibration", device, versions, users)
    options = arrival.ImplementationConfig(reader_backend=config["reader_backend"], row_chunk=config["row_chunk"])
    oracle_wrapped = {v: arrival.InferenceAdapter(a, options) for v, a in adapters.items()}
    wrapped = oracle_wrapped
    if config.get("projection_fused", False):
        from experiments.archive.system_study.fused_query_projection import QueryProjectionAdapter
        wrapped = {v: QueryProjectionAdapter(a, options) for v, a in adapters.items()}
    bank = arrival.load_bank(adapters["v3"].feature_bank_path, adapters["v3"].feature_bank_sha256,
                             device=device, dtype=torch.float32)
    prefixes, oracle_prefixes, publications = {}, {}, {}
    wrap_seconds = 0.
    for version in versions:
        path = candidate / "publication" / f"{version}.pt"
        metadata = json.loads(path.with_suffix(".json").read_text())
        prefix, binding = arrival.load_prefix(path, adapters[version], device=device,
                                               expected_sha256=metadata["artifact_sha256"])
        if binding["bindings"]["calibration_sha256"] != weights[version]["sha256"] or \
                binding["bindings"]["current_model_sha256"] != panel["models"][version]["checkpoint_sha256"]:
            raise ValueError("model/calibration/publication mismatch")
        oracle_prefixes[version] = arrival.configure_prefix(prefix, options,
            feature_fn=partial(arrival.fixed_features_from_bank, bank=bank))
        prefixes[version] = oracle_prefixes[version]
        if config.get("conversion_backend", "inplace") == "fused":
            from experiments.archive.system_study.fused_conversion import FusedReleasePrefix
            torch.cuda.synchronize(device)
            begin_wrap = time.perf_counter()
            prefixes[version] = SessionPrefix(FusedReleasePrefix.wrap(oracle_prefixes[version], bank=bank))
            torch.cuda.synchronize(device)
            wrap_seconds += time.perf_counter()-begin_wrap
        publications[version] = dict(path=str(path), sha256=metadata["artifact_sha256"])
    torch.cuda.synchronize(device)
    settings.update(shared_load_seconds=time.perf_counter()-started, publications=publications, calibration=weights,
                    conversion_wrap_seconds_included_in_shared_load=wrap_seconds)
    if config.get("conversion_backend") == "fused":
        settings["source_sha256"]["experiments/archive/system_study/fused_conversion.py"] = arrival.sha256(
            ROOT / "experiments/archive/system_study/fused_conversion.py")
    if config.get("projection_fused", False):
        for filename in ("fused_query_projection.py", "fused_query_projection_kernel.py"):
            path = ROOT / "experiments/archive/system_study" / filename
            settings["source_sha256"][str(path.relative_to(ROOT))] = arrival.sha256(path)
    arrival.write_json(args.output / "configuration.json", settings)
    histories = arrival.load_history(panel_path, requested_uids=users).rows
    requests = {u: arrival.candidate_rows(u, histories[u], stride=config["stride"],
        candidates=config["candidates"], known_vocab_size=panel["known_vocab_size"]) for u in users}
    trace = arrival.build_trace(histories, requests, arrival.RELEASES, arrival.END)
    trace_audit = arrival.audit_trace(trace, histories, requests)
    retained = {}
    for method in config["methods"]:
        if config.get("previous_batch_reference"):
            path = ROOT / config["previous_batch_reference"]
            rows = pq.read_table(path).to_pylist()
            retained[method] = {r["request_id"]: r["score"] for r in rows
                if r["method"] == method and r["execution"] == "batch8" and r["repeat"] == 0}
            settings["input_sha256"]["previous_batch_reference"] = arrival.sha256(path)
            continue
        reference_key = "relay_reference" if method == "relay" else "native_reference"
        if not config.get(reference_key):
            retained[method] = None
            continue
        path = ROOT / config[reference_key]
        rows = pq.read_table(path).to_pylist()
        retained[method] = {r["request_id"]: r["all_optimized" if method == "relay" else "score"] for r in rows
            if r["protocol"] == config["protocol"] and (method == "relay" or r["method"] == method)}
    make = lambda method, oracle=False: arrival.ArrivalHandler(method, models, oracle_wrapped if oracle else wrapped,
        oracle_prefixes if oracle else prefixes, histories, users, query_batch=config["query_batch"])
    diagnostics, phase_rows, score_rows, time_units = [], [], [], []
    transfer_profiles = {}
    warmups = {}
    graph_compute = None
    single_method_graph = config.get("graph_residency") == "single_method"
    graph_diagnostics, graph_pass_warmups, graph_releases = {}, [], []
    def new_graph_compute():
        from experiments.archive.system_study.graph_compute import GraphCompute
        return GraphCompute(batch_sizes=config.get("graph_batch_sizes", [1, 8]),
            append_widths=config.get("graph_append_widths", [1, 16, 32]),
            model_versions={id(model): version for version, model in models.items()},
            share_history_inputs=config.get("graph_share_history_inputs", False))
    if config.get("graph_compute", False):
        graph_compute = new_graph_compute()
        for filename in ("graph_compute.py", "fused_query_projection_kernel.py"):
            path = Path(__file__).with_name(filename)
            settings["source_sha256"][str(path.relative_to(ROOT))] = arrival.sha256(path)
    batch_options = dict(coalesce_zero_tail=config.get("coalesce_zero_tail", False),
                         projection_fused=config.get("projection_fused", False),
                         native_fused_append=config.get("native_fused_append", False),
                         fused_code_append=config.get("fused_code_append", False), graph_compute=graph_compute)
    arrival.write_json(args.output / "configuration.json", settings)
    def reset_graph_compute():
        nonlocal graph_compute
        # Benchmark only one method's shared graph workspace at a time. User
        # sessions have already been released; the model/table inputs stay put.
        torch.cuda.synchronize(device)
        before = dict(entries=0 if graph_compute is None else len(graph_compute.entries),
            allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device))
        old_owner = None if graph_compute is None else weakref.ref(graph_compute)
        batch_options["graph_compute"] = None
        graph_compute = None
        gc.collect()
        if old_owner is not None and old_owner() is not None:
            raise AssertionError("previous method's graph cache still has an owner")
        torch.cuda.empty_cache()
        graph_releases.append(dict(before=before, after=dict(allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device))))
        arrival.write_json(args.output / "graph_releases.json", graph_releases)
        graph_compute = new_graph_compute()
        batch_options["graph_compute"] = graph_compute
    timed_references = {}
    def checked_state(session, expected, *, method, uid, sequence):
        actual = snapshot(session)
        try:
            return compare_snapshot(actual, expected, atol=config["atol"], rtol=config["rtol"])
        except AssertionError as error:
            tensor_errors = {}
            for name, target in expected["tensors"].items():
                value = actual["tensors"][name]
                delta = value.double()-target.double()
                tensor_errors[name] = dict(max_abs=float(delta.abs().max()),
                    relative_l2=float(delta.norm()/target.double().norm().clamp_min(1e-30)),
                    elements=delta.numel(), outside_tolerance=int((delta.abs() >
                        config["atol"]+config["rtol"]*target.double().abs()).sum()))
            arrival.write_json(args.output / "failed_boundary.json", dict(method=method, uid=uid,
                event_sequence=sequence, version=session.trajectory.version,
                logical_time=session.trajectory.time, metadata_exact=actual["metadata"] == expected["metadata"],
                tensor_errors=tensor_errors, error=str(error)))
            raise
    def timed_one(method, execution, repeat, oracle_scores, reference_states):
        if single_method_graph:
            reset_graph_compute()
            arrival.progress(args.output, "single_method_graph_warmup", method=method, execution=execution, repeat=repeat)
            initial = time.perf_counter()
            warmer = make(method)
            torch.cuda.synchronize(device)
            initial = time.perf_counter()-initial
            started = time.perf_counter()
            run_batched(warmer, trace, config["max_batch"], **batch_options)
            torch.cuda.synchronize(device)
            warm_seconds = time.perf_counter()-started
            warm_comparison = arrival.compare_scores(warmer.scores, oracle_scores,
                atol=config["atol"], rtol=config["rtol"])
            graph_pass_warmups.append(dict(method=method, execution=execution, repeat=repeat,
                initial_v2_cache_seconds=initial, full_trace_warm_seconds=warm_seconds,
                graph_preparation_included_in_warm=graph_compute.describe(), paired_scores=warm_comparison,
                scope="Repeated benchmark preparation after releasing the previous method's graph pools; full trace warm time includes normal service plus graph preparation, and is not all capture cost."))
            arrival.write_json(args.output / "graph_pass_warmups.json", graph_pass_warmups)
            del warmer
            gc.collect()
        arrival.progress(args.output, "uninstrumented_bulk", method=method, execution=execution, repeat=repeat)
        begin = time.perf_counter()
        handler = make(method)
        torch.cuda.synchronize(device)
        initial_seconds = time.perf_counter()-begin
        torch.cuda.reset_peak_memory_stats(device)
        graphs_before = None if graph_compute is None else graph_compute.describe()
        begin = time.perf_counter()
        if execution == "serial":
            run_serial(handler, trace)
            engine = None
        else:
            execution_options = dict(batch_options)
            if execution == "batch8_original_code":
                execution_options["fused_code_append"] = False
            if execution == "batch8_eager":
                execution_options["graph_compute"] = None
            engine = run_batched(handler, trace, config["max_batch"], **execution_options)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-begin
        peak = torch.cuda.max_memory_allocated(device)
        graphs_after = None if graph_compute is None else graph_compute.describe()
        comparison = arrival.compare_scores(handler.scores, oracle_scores,
            atol=config["atol"], rtol=config["rtol"])
        final = []
        sequence = trace[-1].sequence
        for uid, session in handler.sessions.items():
            check = checked_state(session, reference_states[sequence, uid],
                                  method=method, uid=uid, sequence=sequence)
            final.append(dict(uid=uid, **check))
        time_units.append(dict(method=method, execution=execution, repeat=repeat,
            initial_v2_cache_seconds=initial_seconds, service_wall_seconds=elapsed,
            peak_allocated_bytes=peak, paired_scores=comparison, final_states=final,
            graph_state_before=graphs_before, graph_state_after=graphs_after,
            graph_captures_during_timing=0 if graph_compute is None else graphs_after["entries"]-graphs_before["entries"],
            calls=None if engine is None else summarize_calls(engine.calls),
            graph_coverage=None if engine is None else summarize_graph_coverage(engine.calls),
            timing_scope="Complete synchronous resident bulk service including CPU pack/H2D/D2H and ledger work; excludes initialization, shared loading, checks and phase barriers"))
        for uid in users:
            for row in requests[uid]:
                score_rows.append(dict(method=method, execution=execution, repeat=repeat,
                                      **row, score=handler.scores[row["request_id"]]))
        arrival.write_json(args.output / "completed_timings.json", time_units)
        pq.write_table(pa.Table.from_pylist(score_rows), args.output / "scores.parquet", compression="zstd")
        if config.get("profile_state_transfer", False) and repeat == config["timing_repeats"]-1 \
                and execution == config.get("timed_executions", ["serial", "batch8"])[-1]:
            from experiments.archive.system_study.state_transfer import profile_state_transfer
            arrival.progress(args.output, "independent_full_state_transfer", method=method)
            transfer_profiles[method] = profile_state_transfer(handler.sessions, device)
            arrival.write_json(args.output / "state_transfer.json", transfer_profiles)
        del handler, engine
        gc.collect()

    for method in config["methods"]:
        if single_method_graph:
            reset_graph_compute()
        arrival.progress(args.output, "previous_batch_reference" if config.get("oracle_execution") == "previous_batch"
                         else "serial_oracle_and_warmup", method=method)
        reference_states = {}
        def save_boundary(sequence, handler):
            for uid, session in handler.sessions.items():
                reference_states[sequence, uid] = snapshot(session)
        batch_reference = config.get("oracle_execution") == "previous_batch"
        oracle = make(method, oracle=not batch_reference)
        if batch_reference:
            # Reuse the preceding actual batch path: only native append stays
            # on its original backend. Independent relay checks are retained
            # in the preceding run, whose scores are matched below.
            reference_options = dict(batch_options,
                native_fused_append=config.get("reference_native_fused_append", False),
                fused_code_append=config.get("reference_fused_code_append", False), graph_compute=None)
            run_batched(oracle, trace, config["max_batch"], on_boundary=save_boundary, **reference_options)
        else:
            run_serial(oracle, trace, on_boundary=save_boundary)
        torch.cuda.synchronize(device)
        retained_check = (arrival.compare_scores(oracle.scores, retained[method], atol=config["atol"], rtol=config["rtol"])
                          if retained[method] is not None else dict(status="new_dynamic_oracle_no_retained_population_scores"))
        oracle_scores = dict(oracle.scores)
        del oracle
        gc.collect()
        if config.get("separate_phase_warmup", False):
            warmups[method] = {}
            for mode in config.get("timed_executions", ["serial", "batch8"]):
                arrival.progress(args.output, "separate_backend_warmup", method=method, execution=mode)
                warmer = make(method)
                begin = time.perf_counter()
                if mode == "serial":
                    run_serial(warmer, trace)
                else:
                    run_batched(warmer, trace, config["max_batch"], **batch_options)
                torch.cuda.synchronize(device)
                warmups[method][mode] = dict(seconds=time.perf_counter()-begin,
                    comparison=arrival.compare_scores(warmer.scores, oracle_scores,
                        atol=config["atol"], rtol=config["rtol"]))
                del warmer
                gc.collect()
            arrival.write_json(args.output / "backend_warmup.json", warmups)
            if graph_compute is not None:
                arrival.write_json(args.output / "graph_preparation.json", graph_compute.describe())
        arrival.progress(args.output, "batched_correctness_and_phase_warmup", method=method)
        comparisons = []
        def check_boundary(sequence, handler):
            for uid, session in handler.sessions.items():
                comparison = checked_state(session, reference_states[sequence, uid],
                    method=method, uid=uid, sequence=sequence)
                comparisons.append(dict(event_sequence=sequence, uid=uid, **comparison))
        recorder = PhaseRecorder(device)
        recorder.context = dict(method=method, max_batch=config["max_batch"], execution="diagnostic")
        batch = make(method)
        engine = run_batched(batch, trace, config["max_batch"], recorder=recorder,
                             on_boundary=check_boundary, **batch_options)
        comparison = arrival.compare_scores(batch.scores, oracle_scores, atol=config["atol"], rtol=config["rtol"])
        diagnostics.append(dict(method=method, retained_reference=retained_check,
            reference_execution=config.get("oracle_execution", "serial_original"),
            paired_scores=comparison, boundary_comparisons=comparisons, call_histogram=summarize_calls(engine.calls),
            graph_coverage=summarize_graph_coverage(engine.calls),
            additional_coalescing_flops=sum(r.get("additional_coalescing_flops", 0) for r in engine.calls),
            additional_zero_code_pairs=sum(r.get("additional_zero_code_pairs", 0) for r in engine.calls),
            additional_active_code_queries=sum(r.get("additional_active_code_queries", 0) for r in engine.calls),
            phase_is_after_separate_backend_warmup=config.get("separate_phase_warmup", False)))
        phase_rows.extend(recorder.rows)
        pq.write_table(pa.Table.from_pylist(phase_rows), args.output / "phases.parquet", compression="zstd")
        arrival.write_json(args.output / f"{method}_calls.json", engine.calls)
        arrival.write_json(args.output / "diagnostics.json", diagnostics)
        if graph_compute is not None:
            graph_diagnostics[method] = graph_compute.describe()
            arrival.write_json(args.output / "graph_diagnostic_preparation.json", graph_diagnostics)
        del batch, engine, recorder
        gc.collect()
        if config.get("interleave_methods", False):
            # Only the final boundary is needed after the complete diagnostic.
            final_sequence = trace[-1].sequence
            timed_references[method] = (oracle_scores,
                {(final_sequence, uid): reference_states[final_sequence, uid] for uid in users})
        else:
            for repeat in range(config["timing_repeats"]):
                order = list(config.get("timed_executions", ["serial", "batch8"]))
                if repeat % 2:
                    order.reverse()
                for execution in order:
                    timed_one(method, execution, repeat, oracle_scores, reference_states)
        del reference_states
        gc.collect()
    if config.get("interleave_methods", False):
        if config.get("defer_measurement", False):
            arrival.progress(args.output, "prepared_waiting_measurement_window")
            begin = time.perf_counter()
            while not (args.output / "measurement.ready").exists():
                time.sleep(2)
            settings["measurement_window_wait_seconds_excluded"] = time.perf_counter()-begin
            arrival.write_json(args.output / "configuration.json", settings)
        # Both methods have finished warmup/checks before any measured pass.
        for repeat in range(config["timing_repeats"]):
            if config.get("timing_graph_control", False):
                passes = [(m, e) for m in config["methods"] for e in ("batch8_eager", "batch8")]
                if repeat % 2:
                    passes.reverse()
                for method, execution in passes:
                    timed_one(method, execution, repeat, *timed_references[method])
                continue
            if config.get("timing_code_append_control", False):
                passes = [("relay", "batch8_original_code"), ("relay", "batch8"), ("rebuild", "batch8")]
                if repeat % 2:
                    passes.reverse()
                for method, execution in passes:
                    timed_one(method, execution, repeat, *timed_references[method])
                continue
            methods = list(config["methods"])
            executions = list(config.get("timed_executions", ["serial", "batch8"]))
            if repeat % 2:
                methods.reverse()
                executions.reverse()
            for execution in executions:
                for method in methods:
                    timed_one(method, execution, repeat, *timed_references[method])
        timed_references.clear()
    summary = dict(status="complete_distinct_user_batched_session_probe", users=users,
        trace_audit=trace_audit, diagnostics=diagnostics, time_units=time_units,
        scope=settings["scope"], shared_load_seconds=settings["shared_load_seconds"],
        conversion_wrap_seconds_included_in_shared_load=wrap_seconds,
        graph_compute=None if graph_compute is None else graph_compute.describe(),
        graph_residency=config.get("graph_residency", "all_methods"),
        graph_diagnostic_preparation=graph_diagnostics, graph_pass_warmups=graph_pass_warmups,
        graph_releases=graph_releases,
        state_transfer_file="state_transfer.json" if transfer_profiles else None)
    arrival.write_json(args.output / "summary.json", summary)
    arrival.progress(args.output, "complete", methods=config["methods"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.stdout = sys.stderr = (args.output / "run.log").open("a", buffering=1)
    try:
        run(args)
    except BaseException as error:
        arrival.write_json(args.output / "summary.json", dict(status="failed", pid=os.getpid(),
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        raise
