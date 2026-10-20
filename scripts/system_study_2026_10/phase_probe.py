#!/usr/bin/env python3
"""Locate real replay cost without changing equations or historical results.

Additional synchronization makes these phase diagnostics distinct from the
arrival experiment. Small row-chunk comparisons change execution geometry,
not the code update or logical FLOPs. All quality comparisons remain paired.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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

from system_study_2026_10 import arrival_replay as arrival
from system_study_2026_10.model_replay import compare_snapshot, state_snapshot


class PhaseTimer:
    def __init__(self, device):
        self.device, self.rows, self.stack = device, [], []
        self.context = {}

    def call(self, phase, function, *args, memory=False, **kwargs):
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device) if memory else None
        if memory:
            torch.cuda.reset_peak_memory_stats(self.device)
        row = dict(phase=phase, **self.context, children_seconds=0.)
        if memory:
            row["history_rows"] = args[0].seq_len
        self.stack.append(row)
        started = time.perf_counter()
        result = function(*args, **kwargs)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.stack.pop()
        row.update(inclusive_seconds=elapsed,
                   exclusive_seconds=elapsed - row["children_seconds"])
        if memory:
            row["temporary_peak_bytes"] = max(0, torch.cuda.max_memory_allocated(self.device) - allocated)
        if self.stack:
            self.stack[-1]["children_seconds"] += elapsed
        self.rows.append(row)
        return result

    def instrument_prefix(self, prefix):
        for method, phase in (("materialize", "code_materialize"), ("anchorize", "code_anchorize")):
            original = getattr(prefix, method)
            def measured(*args, _original=original, _phase=phase, **kwargs):
                return self.call(_phase, _original, *args, memory=True, **kwargs)
            setattr(prefix, method, measured)


def aggregate_phases(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["phase"]].append(row)
    return {phase: dict(calls=len(values),
        inclusive_total_seconds=sum(r["inclusive_seconds"] for r in values),
        exclusive_total_seconds=sum(r["exclusive_seconds"] for r in values),
        inclusive_mean_seconds=float(np.mean([r["inclusive_seconds"] for r in values])),
        inclusive_p95_seconds=float(np.quantile([r["inclusive_seconds"] for r in values], .95)),
        temporary_peak_bytes=max((r.get("temporary_peak_bytes", 0) for r in values), default=0))
        for phase, values in grouped.items()}


@torch.no_grad()
def run(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    plan = json.loads(args.config.read_text())
    base_path = ROOT / plan["base_config"]
    config = json.loads(base_path.read_text())
    selection_path = ROOT / config["selection"]
    selection = json.loads(selection_path.read_text())
    users = arrival.choose_users(selection)
    panel_path, candidate = ROOT / config["panel"], ROOT / config["candidate"]
    if arrival.sha256(panel_path / "binding.json") != selection["panel_sha256"] or \
            arrival.sha256(candidate / "freeze.json") != selection["freeze_sha256"]:
        raise ValueError("frozen inputs changed")
    panel, roles, _ = arrival.load_panel(panel_path)
    if len(users) != 4 or not set(users).issubset(roles["evaluation"]):
        raise ValueError("probe uses the same four development users")
    settings = dict(created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        plan=plan, base_config=config, users=users, pid=os.getpid(),
        input_sha256=dict(config=arrival.sha256(args.config), base_config=arrival.sha256(base_path),
            selection=arrival.sha256(selection_path), panel=arrival.sha256(panel_path / "binding.json"),
            freeze=arrival.sha256(candidate / "freeze.json")),
        source_sha256={str(path.relative_to(ROOT)): arrival.sha256(path) for path in (
            Path(__file__), Path(arrival.__file__), ROOT / "scripts/implementation_2026_10/state.py",
            ROOT / "scripts/query_relay_2026_10/session.py")},
        scope="Four-user phase diagnosis with additional barriers; no production latency, capacity or FLOPs-saving claim")
    arrival.write_json(args.output / "configuration.json", settings)
    arrival.progress(args.output, "load_assets")
    device = arrival.setup_device(args.gpu, threads=2)
    models = arrival.load_models(panel, device)
    versions = [v for v, _ in arrival.RELEASES]
    adapters, weights, _ = arrival.load_candidates(candidate / "calibration", device, versions, users)
    impl = arrival.ImplementationConfig(reader_backend=config["reader_backend"], row_chunk=128)
    wrapped = {v: arrival.InferenceAdapter(a, impl) for v, a in adapters.items()}
    bank = arrival.load_bank(adapters["v3"].feature_bank_path, adapters["v3"].feature_bank_sha256,
                             device=device, dtype=torch.float32)
    raw_prefixes, publications = {}, {}
    for version in versions:
        path = candidate / "publication" / f"{version}.pt"
        metadata = json.loads(path.with_suffix(".json").read_text())
        prefix, record = arrival.load_prefix(path, adapters[version], device=device,
                                             expected_sha256=metadata["artifact_sha256"])
        if record["bindings"]["calibration_sha256"] != weights[version]["sha256"] or \
                record["bindings"]["current_model_sha256"] != panel["models"][version]["checkpoint_sha256"]:
            raise ValueError("publication binding changed")
        raw_prefixes[version] = prefix
        publications[version] = dict(path=str(path), sha256=metadata["artifact_sha256"],
                                     model_sha256=panel["models"][version]["checkpoint_sha256"])
    settings.update(calibration=weights, publications=publications, device=torch.cuda.get_device_name(device))
    arrival.write_json(args.output / "configuration.json", settings)
    histories = arrival.load_history(panel_path, requested_uids=users).rows
    requests = {u: arrival.candidate_rows(u, histories[u], stride=16, candidates=16,
        known_vocab_size=panel["known_vocab_size"]) for u in users}
    trace = arrival.build_trace(histories, requests, arrival.RELEASES, arrival.END)
    all_reference = pq.read_table(ROOT / config["reference_scores"]).to_pylist()
    references = {method: {r["request_id"]: r["score"] for r in all_reference
        if r["method"] == method and r["protocol"] == config["protocol"]}
        for method in ("relay", "rebuild")}
    baseline_states, units, raw_rows = {}, [], []
    branches = [("relay", chunk) for chunk in plan["relay_row_chunks"]] + [("rebuild", None)]
    for method, chunk in branches:
        label = f"{method}_chunk{chunk}" if chunk is not None else method
        options = arrival.ImplementationConfig(reader_backend=config["reader_backend"], row_chunk=chunk or 128)
        prefixes = {v: arrival.configure_prefix(prefix, options,
            feature_fn=partial(arrival.fixed_features_from_bank, bank=bank)) for v, prefix in raw_prefixes.items()}
        make = lambda: arrival.ArrivalHandler(method, models, wrapped, prefixes, histories, users, query_batch=16)
        arrival.progress(args.output, "warmup", branch=label)
        warm = make()
        for event in trace:
            if event.kind != "behavior":
                warm(event)
        torch.cuda.synchronize(device)
        arrival.compare_scores(warm.scores, references[method], atol=config["atol"], rtol=config["rtol"])
        del warm
        gc.collect()
        timer = PhaseTimer(device)
        if method == "relay":
            for prefix in prefixes.values():
                timer.instrument_prefix(prefix)
        handler = make()
        comparisons = []
        arrival.progress(args.output, "phase_measurement", branch=label)
        for event in trace:
            timer.context = dict(branch=label, event_sequence=event.sequence,
                                 event_kind=event.kind, version=event.version, uid=event.uid)
            if event.kind == "publication":
                for uid in users:
                    timer.context["uid"] = uid
                    timer.call("publication_old_advance", handler.advance, uid, event.logical_time)
                timer.context["uid"] = None
                timer.call("publication_switch", handler.publish, event.version, event.logical_time)
            elif event.kind == "recommendation":
                timer.call("recommendation_advance", handler.advance, event.uid, event.logical_time)
                timer.call("recommendation_read", handler.recommend, event.uid, event.logical_time, event.requests)
            elif event.kind == "drain":
                for uid in users:
                    timer.context["uid"] = uid
                    timer.call("drain_advance", handler.advance, uid, event.logical_time)
            # Boundary copies/checks occur outside all measured phase calls.
            if method == "relay" and event.kind in ("publication", "drain"):
                for uid in users:
                    key = (event.sequence, uid)
                    state = state_snapshot(handler.sessions[uid])
                    if chunk == plan["relay_row_chunks"][0]:
                        baseline_states[key] = state
                    else:
                        compared = compare_snapshot(state, baseline_states[key],
                            atol=config["atol"], rtol=config["rtol"])
                        comparisons.append(dict(event_sequence=event.sequence, uid=uid, **compared))
        score_comparison = arrival.compare_scores(handler.scores, references[method],
            atol=config["atol"], rtol=config["rtol"])
        unit = dict(branch=label, row_chunk=chunk, score_comparison=score_comparison,
                    boundary_comparisons=comparisons, phases=aggregate_phases(timer.rows),
                    logical_update_unchanged=True, extra_synchronization=True,
                    flop_saving_claim=False)
        units.append(unit)
        raw_rows.extend(timer.rows)
        arrival.write_json(args.output / "completed_units.json", units)
        pq.write_table(pa.Table.from_pylist(raw_rows), args.output / "phases.parquet", compression="zstd")
        print(json.dumps(dict(status="branch_complete", branch=label, phases=unit["phases"],
                              score_max_abs=score_comparison["max_abs"])), flush=True)
        del handler, prefixes, timer
        gc.collect()
    summary = dict(status="complete_phase_probe", users=users, units=units,
                   scope=settings["scope"], paired_score_checks=True,
                   dataflow="Original on-demand path; redundant empty advance calls only separate stage boundaries")
    arrival.write_json(args.output / "summary.json", summary)
    arrival.write_json(args.output / "progress.json", dict(status="complete", pid=os.getpid()))
    print(json.dumps(dict(status=summary["status"], branches=len(units))), flush=True)


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
