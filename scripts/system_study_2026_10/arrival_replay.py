#!/usr/bin/env python3
"""Finite real-GPU arrivals over the validated four-user causal trace.

Logical histories are unchanged. Arrival scaling is a declared synthetic load.
All versions are resident for this pilot; neither its memory peak nor its small
sample latency is a production-capacity or sustainable-throughput result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import torch

from design_two.common import END, RELEASES, load_models, setup_device
from design_two.prepare import load_history, load_panel
from implementation_2026_10.backend import ImplementationConfig, InferenceAdapter, configure_prefix
from implementation_2026_10.prepare import fixed_features_from_bank
from query_relay_2026_10.evaluate import load_candidates
from query_relay_2026_10.fixed_features import load_bank
from query_relay_2026_10.publish import load_prefix
from query_relay_2026_10.workload import candidate_rows
from system_study_2026_10.budgeted_rebuild import BudgetedRebuildSession, native_history_read
from system_study_2026_10.events import build_trace
from system_study_2026_10.model_replay import ModelReplayHandler, choose_users, counter_rows
from system_study_2026_10.prepare_trace import audit_trace
from system_study_2026_10.queueing import run_queue
from unified_reuse_2026_09.common import sha256, write_json


class ArrivalHandler(ModelReplayHandler):
    """Same relay path; native controls omit historical support bookkeeping."""

    def __init__(self, method, *args, **kwargs):
        super().__init__("relay" if method == "relay" else "reuse", *args, **kwargs)
        self.method = method
        if method != "relay":
            self.sessions = {uid: BudgetedRebuildSession(trajectory, uid, native_history_read)
                             for uid, trajectory in self.sessions.items()}

    def publish(self, version, timestamp):
        if self.method == "relay":
            return super().publish(version, timestamp)
        for session in self.sessions.values():
            session.advance(timestamp)
        for session in self.sessions.values():
            session.publish(self.models[version], int(version[1:]), timestamp,
                            selected=self.method == "rebuild")
        self.current_version = version

    def recommend(self, uid, timestamp, rows):
        if self.method == "relay":
            return super().recommend(uid, timestamp, rows)
        session = self.sessions[uid]
        for offset in range(0, len(rows), self.query_batch):
            batch = rows[offset:offset + self.query_batch]
            candidates = torch.tensor([[row["item_idx"] for row in batch]], device=session.cache.k.device)
            logits = session.score(timestamp, candidates).detach().flatten().cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite native scores")
            for row, value in zip(batch, logits.tolist(), strict=True):
                if row["request_id"] in self.scores:
                    raise RuntimeError("duplicate scored request")
                self.scores[row["request_id"]] = value


def annotate_trace(trace):
    """First recommendation is distinguished from actual code preparation."""
    seen, annotations = set(), {}
    for event in trace:
        fields = dict(version=event.version, candidate_count=len(event.requests), request_class=None)
        if event.kind == "recommendation":
            key = (event.uid, event.version)
            fields["request_class"] = "first_recommendation_in_release" if key not in seen else "warm_recommendation"
            seen.add(key)
        annotations[event.sequence] = fields
    return annotations


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), mean=float(values.mean()), p50=float(np.quantile(values, .5)),
                p95=float(np.quantile(values, .95)), p99=float(np.quantile(values, .99)),
                maximum=float(values.max()))


def summarize_timings(rows):
    groups = {}
    for label in ("recommendation", "first_recommendation_in_release", "warm_recommendation", "publication", "drain"):
        selected = [r for r in rows if r["kind"] == label or r.get("request_class") == label]
        groups[label] = {key: distribution([r[key] for r in selected])
                         for key in ("queue_seconds", "service_seconds", "latency_seconds")}
    queries = [r for r in rows if r["kind"] == "recommendation"]
    drain = next(r for r in rows if r["kind"] == "drain")
    return dict(groups=groups, scheduled_span_seconds=drain["arrival_time"],
        actual_finish_seconds=drain["finish_time"],
        end_queue_seconds=drain["queue_seconds"],
        recommendations_waiting_over_1ms=sum(r["queue_seconds"] > .001 for r in queries),
        finite_trace_rounds_per_wall_second=len(queries) / drain["finish_time"],
        publication_events=[r for r in rows if r["kind"] == "publication"],
        scope="Measured finite four-user synthetic arrival diagnostics; no stable throughput or production p99 claim")


def compare_scores(scores, reference, *, atol, rtol):
    if set(scores) != set(reference):
        raise AssertionError("measured trace and frozen reference request identities differ")
    maximum = 0.
    for key, expected in reference.items():
        error = abs(scores[key] - expected)
        if error > atol + rtol * abs(expected):
            raise AssertionError(f"measured-score mismatch: {key}")
        maximum = max(maximum, error)
    return dict(candidate_scores=len(scores), max_abs=maximum, status="passed")


def progress(output, stage, **fields):
    value = dict(status="running", stage=stage, pid=os.getpid(), **fields)
    write_json(output / "progress.json", value)
    print(json.dumps(value), flush=True)


@torch.no_grad()
def run(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config = json.loads(args.config.read_text())
    if config["protocol"] != "behavior_stride16_candidates16" or config["behavior_processing"] != "on_demand":
        raise ValueError("pilot preserves the registered next-item/on-demand protocol")
    selection_path = ROOT / config["selection"]
    selection = json.loads(selection_path.read_text())
    users = choose_users(selection)
    if len(users) != config["expected_users"]:
        raise ValueError("unexpected smoke population")
    panel_path, candidate = ROOT / config["panel"], ROOT / config["candidate"]
    if sha256(panel_path / "binding.json") != selection["panel_sha256"] or \
            sha256(candidate / "freeze.json") != selection["freeze_sha256"]:
        raise ValueError("frozen panel/candidate changed")
    panel, groups, _ = load_panel(panel_path)
    if not set(users).issubset(groups["evaluation"]):
        raise ValueError("changed user roles")
    reference_path = ROOT / config["reference_scores"]
    reference_rows = pq.read_table(reference_path).to_pylist()
    references = {method: {r["request_id"]: r["score"] for r in reference_rows
        if r["method"] == method and r["protocol"] == config["protocol"]} for method in config["methods"]}
    source_files = [Path(__file__), Path(__file__).with_name("model_replay.py"),
        Path(__file__).with_name("events.py"), Path(__file__).with_name("queueing.py"),
        Path(__file__).with_name("budgeted_rebuild.py")]
    settings = dict(created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        config=config, users=users, pid=os.getpid(),
        inputs={"config": sha256(args.config), "selection": sha256(selection_path),
            "panel": sha256(panel_path / "binding.json"), "freeze": sha256(candidate / "freeze.json"),
            "reference_scores": sha256(reference_path)},
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        model_load_scope="All V2--V5 models and V3--V5 published tables resident for this pilot",
        behavior_policy="Markers incur no GPU work; appends charged to next recommendation/publication/drain",
        measurement="Real perf_counter arrival/start/finish, GPU completion synchronized; no synthetic service durations",
        warmup=config["warmup"], catalogue_recompiled=False, training=False,
        limitations=["Four existing development users and synthetic arrival compression",
                     "Single FIFO worker with synchronous GPU calls; no network or multi-worker claim",
                     "No steady-state capacity comparison while all versions share GPU residency",
                     "Observed p99 is descriptive for this finite trace, not production tail evidence"])
    write_json(args.output / "configuration.json", settings)
    progress(args.output, "load_assets")
    began_load = time.perf_counter()
    device = setup_device(args.gpu, threads=2)
    models = load_models(panel, device)
    versions = [v for v, _ in RELEASES]
    adapters, weights, _ = load_candidates(candidate / "calibration", device, versions, users)
    impl = ImplementationConfig(reader_backend=config["reader_backend"], row_chunk=config["row_chunk"])
    wrapped = {v: InferenceAdapter(a, impl) for v, a in adapters.items()}
    bank = load_bank(adapters["v3"].feature_bank_path, adapters["v3"].feature_bank_sha256,
                     device=device, dtype=torch.float32)
    prefixes, publications = {}, {}
    for version in versions:
        path = candidate / "publication" / f"{version}.pt"
        metadata = json.loads(path.with_suffix(".json").read_text())
        prefix, record = load_prefix(path, adapters[version], device=device,
                                     expected_sha256=metadata["artifact_sha256"])
        if record["bindings"]["calibration_sha256"] != weights[version]["sha256"] or \
                record["bindings"]["current_model_sha256"] != panel["models"][version]["checkpoint_sha256"]:
            raise ValueError("model/calibration/publication mismatch")
        prefixes[version] = configure_prefix(prefix, impl,
            feature_fn=partial(fixed_features_from_bank, bank=bank))
        publications[version] = dict(path=str(path), sha256=metadata["artifact_sha256"],
            model_sha256=panel["models"][version]["checkpoint_sha256"])
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - began_load
    settings.update(device=torch.cuda.get_device_name(device), calibration=weights,
                    publications=publications, shared_load_and_inverse_seconds=load_seconds)
    write_json(args.output / "configuration.json", settings)
    histories = load_history(panel_path, requested_uids=users).rows
    requests = {u: candidate_rows(u, histories[u], stride=config["stride"],
        candidates=config["candidates"], known_vocab_size=panel["known_vocab_size"]) for u in users}
    base_trace = build_trace(histories, requests, RELEASES, END)
    trace_audit = audit_trace(base_trace, histories, requests)
    warmups, units, timing_rows, score_rows = {}, [], [], []
    make = lambda method: ArrivalHandler(method, models, wrapped, prefixes, histories,
                                         users, query_batch=config["query_batch"])
    for method in config["methods"]:
        progress(args.output, "warmup_full_trace", method=method)
        warmup_begin = time.perf_counter()
        warmer = make(method)
        for event in base_trace:
            if event.kind != "behavior":
                warmer(event)
        torch.cuda.synchronize(device)
        warmups[method] = dict(elapsed_seconds=time.perf_counter() - warmup_begin,
            comparison=compare_scores(warmer.scores, references[method], atol=config["atol"], rtol=config["rtol"]))
        del warmer
        gc.collect()
        write_json(args.output / "warmup.json", warmups)
        for scale in config["arrival_scales"]:
            trace = build_trace(histories, requests, RELEASES, END, arrival_scale=scale)
            annotations = annotate_trace(trace)
            start = time.perf_counter()
            handler = make(method)
            torch.cuda.synchronize(device)
            init_seconds = time.perf_counter() - start
            progress(args.output, "measured_arrivals", method=method, arrival_scale=scale,
                     scheduled_seconds=trace[-1].arrival_time)
            torch.cuda.reset_peak_memory_stats(device)
            measured = run_queue(trace, handler, synchronize=lambda: torch.cuda.synchronize(device))
            rows = [dict(method=method, arrival_scale=scale, **asdict(t), **annotations[t.sequence])
                    for t in measured]
            comparison = compare_scores(handler.scores, references[method], atol=config["atol"], rtol=config["rtol"])
            unit = dict(method=method, arrival_scale=scale, initial_v2_cache_seconds=init_seconds,
                comparison=comparison, measurements=summarize_timings(rows),
                measured_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_scope="All-version-resident pilot; not service capacity",
                final_operation_ledgers=[dict(uid=uid,
                    native=counter_rows(session.trajectory.ledger),
                    service=counter_rows(session.costs),
                    corrected_reads=counter_rows(getattr(session, "coded_read_ledger", {})),
                    queries=counter_rows(getattr(session, "query_ledger", {})))
                    for uid, session in handler.sessions.items()])
            units.append(unit)
            timing_rows.extend(rows)
            for uid in users:
                for row in requests[uid]:
                    score_rows.append(dict(method=method, arrival_scale=scale, **row,
                                           score=handler.scores[row["request_id"]]))
            write_json(args.output / "completed_units.json", units)
            pq.write_table(pa.Table.from_pylist(timing_rows), args.output / "timings.parquet", compression="zstd")
            pq.write_table(pa.Table.from_pylist(score_rows), args.output / "scores.parquet", compression="zstd")
            print(json.dumps(dict(status="measured_unit_complete", method=method, arrival_scale=scale,
                recommendation_p95=unit["measurements"]["groups"]["recommendation"]["latency_seconds"]["p95"],
                end_queue_seconds=unit["measurements"]["end_queue_seconds"], score_max_abs=comparison["max_abs"])), flush=True)
            del handler, measured
            gc.collect()
    summary = dict(status="complete_real_gpu_arrival_pilot", users=users, trace_audit=trace_audit,
        shared_load_and_inverse_seconds=load_seconds, warmups=warmups, units=units,
        timing_rows=len(timing_rows), measured_candidate_scores=len(score_rows),
        GPU_model_execution=True, real_arrival_measurement=True, production_performance_claim=False,
        scope=settings["limitations"])
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "progress.json", dict(status="complete", pid=os.getpid()))
    print(json.dumps(dict(status=summary["status"], units=len(units))), flush=True)


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
        write_json(args.output / "summary.json", dict(status="failed", pid=os.getpid(),
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            production_performance_claim=False))
        raise
