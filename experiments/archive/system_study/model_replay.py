#!/usr/bin/env python3
"""Actual-model smoke test of global causal scheduling against per-user replay.

All fixed releases may reside together for this correctness check. Observed
elapsed time and peak allocation are diagnostics, not throughput/capacity
claims. Existing full-catalogue tables are loaded, never recompiled here.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]

import torch

from hstu_kvcache.evaluation.release_context import END, RELEASES, load_models, setup_device
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.methods.cachetide.lifecycle.lifecycle import LifecycleSession
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.methods.cachetide.backend import ImplementationConfig, InferenceAdapter, configure_prefix
from hstu_kvcache.methods.cachetide.prepare import fixed_features_from_bank
from experiments.archive.query_relay.evaluate import load_candidates
from hstu_kvcache.methods.cachetide.features import load_bank
from hstu_kvcache.methods.cachetide.publication import load_prefix
from hstu_kvcache.methods.cachetide.session import RelaySession
from hstu_kvcache.evaluation.cachetide_workload import candidate_rows
from experiments.archive.system_study.events import build_trace
from experiments.archive.system_study.prepare_trace import audit_trace
from hstu_kvcache.common.artifacts import sha256, write_json


def choose_users(selection):
    chosen, seen = [], set()
    for profile in selection["profiles"]:
        if profile["stratum"] not in seen:
            chosen.append(int(profile["uid"]))
            seen.add(profile["stratum"])
    return chosen


def counter_rows(counter):
    return [[list(key) if isinstance(key, tuple) else key, int(value)]
            for key, value in sorted(counter.items(), key=lambda item: str(item[0]))]


def state_snapshot(session):
    trajectory = session.trajectory if hasattr(session, "trajectory") else session
    state = getattr(session, "state", None)
    tensors = dict(k=trajectory.cache.k.detach().cpu().clone(),
                   v=trajectory.cache.v.detach().cpu().clone())
    if state is not None:
        tensors["codes"] = state.codes.detach().cpu().clone()
    metadata = dict(version=trajectory.version, cursor=trajectory.cursor,
        logical_time=trajectory.time, history_rows=trajectory.cache.seq_len,
        producers=trajectory.producers.tolist(),
        appended_tokens=trajectory.appended_tokens, evicted_tokens=trajectory.evicted_tokens,
        native_ledger=counter_rows(trajectory.ledger),
        service_costs=counter_rows(getattr(session, "costs", {})),
        read_ledger=counter_rows(getattr(session, "read_ledger", {})),
        coded_read_ledger=counter_rows(getattr(session, "coded_read_ledger", {})),
        state_format=getattr(session, "state_format", None),
        coded_rows=getattr(session, "coded_rows", None),
        pending_rebuild=getattr(session, "pending_rebuild", None),
        events=copy.deepcopy(getattr(session, "events", []))
            if hasattr(session, "trajectory") else [])
    if hasattr(session, "support"):
        metadata["support_flags"] = session.support.flags.tolist()
    return dict(metadata=metadata, tensors=tensors)


def compare_snapshot(actual, expected, *, atol, rtol):
    if actual["metadata"] != expected["metadata"]:
        different = [key for key in expected["metadata"]
                     if actual["metadata"].get(key) != expected["metadata"][key]]
        raise AssertionError(f"state/operation metadata differs: {different}")
    if set(actual["tensors"]) != set(expected["tensors"]):
        raise AssertionError("persistent tensor kinds differ")
    errors = {}
    for name, target in expected["tensors"].items():
        value = actual["tensors"][name]
        torch.testing.assert_close(value, target, atol=atol, rtol=rtol)
        delta = value.double() - target.double()
        errors[name] = dict(max_abs=float(delta.abs().max()) if delta.numel() else 0.,
            relative_l2=float(delta.norm() / target.double().norm().clamp_min(1e-30)))
    return dict(status="passed", metadata_exact=True, tensor_errors=errors,
                cursor=expected["metadata"]["cursor"], rows=expected["metadata"]["history_rows"])


class ModelReplayHandler:
    """Synchronous real-model consumer, reusable by the single-worker queue.

    The boundary callback is for correctness snapshots, and must be omitted
    from later service measurements. Behavior markers never run GPU work.
    """

    def __init__(self, method, models, adapters, prefixes, histories, users,
                 *, query_batch=16, on_boundary=None):
        self.method, self.models = method, models
        self.adapters, self.prefixes = adapters, prefixes
        self.query_batch, self.on_boundary = query_batch, on_boundary
        self.scores, self.sessions = {}, {}
        for uid in users:
            trajectory = NativeTrajectory(models["v2"], 2, histories[uid], RELEASES[0][1])
            self.sessions[uid] = (RelaySession(trajectory) if method == "relay" else
                                  LifecycleSession(trajectory) if method == "rebuild" else trajectory)
        self.current_version = "v2"

    def advance(self, uid, timestamp):
        self.sessions[uid].advance(timestamp)

    def capture(self, uid):
        if self.on_boundary is not None and self.current_version != "v2":
            self.on_boundary(uid, self.current_version, state_snapshot(self.sessions[uid]))

    def publish(self, version, timestamp):
        # Finish old-version work for every resident user before switching any.
        for uid in self.sessions:
            self.advance(uid, timestamp)
            self.capture(uid)
        for session in self.sessions.values():
            if self.method == "relay":
                session.publish(self.models[version], int(version[1:]), timestamp,
                                self.adapters[version], prefix=self.prefixes[version])
            elif self.method == "rebuild":
                session.publish(self.models[version], int(version[1:]), timestamp, None,
                                policy="rebuild_on_first_request")
            else:
                session.switch(self.models[version], int(version[1:]), timestamp)
        self.current_version = version

    def recommend(self, uid, timestamp, rows):
        session = self.sessions[uid]
        trajectory = session.trajectory if hasattr(session, "trajectory") else session
        device = trajectory.cache.k.device
        for offset in range(0, len(rows), self.query_batch):
            batch = rows[offset:offset + self.query_batch]
            ids = torch.tensor([[int(row["item_idx"]) for row in batch]], device=device)
            if self.method == "reuse":
                trajectory.advance(timestamp)
                delta = 0 if trajectory.last_timestamp is None else timestamp - trajectory.last_timestamp
                scores = native_score(trajectory.model, trajectory.cache, ids,
                                      torch.tensor([delta], device=device, dtype=torch.float32))[0]
            else:
                scores = session.score(timestamp, ids)
            values = scores.detach().reshape(-1).cpu()
            if not torch.isfinite(values).all():
                raise RuntimeError("nonfinite model scores")
            for row, value in zip(batch, values.tolist(), strict=True):
                request_id = row["request_id"]
                if request_id in self.scores:
                    raise RuntimeError("duplicate scored request")
                self.scores[request_id] = value

    def drain(self, timestamp):
        for uid in self.sessions:
            self.advance(uid, timestamp)
            self.capture(uid)

    @torch.no_grad()
    def __call__(self, event):
        if event.kind == "publication":
            self.publish(event.version, event.logical_time)
        elif event.kind == "recommendation":
            if event.version != self.current_version:
                raise RuntimeError("request read under the wrong release")
            self.recommend(event.uid, event.logical_time, event.requests)
        elif event.kind == "drain":
            self.drain(event.logical_time)
        elif event.kind != "behavior":
            raise ValueError(event.kind)


def report_progress(output, stage, **fields):
    value = dict(status="running", stage=stage, pid=os.getpid(), **fields)
    write_json(output / "progress.json", value)
    print(json.dumps(value), flush=True)


@torch.no_grad()
def run(args):
    began = time.perf_counter()
    config = json.loads(args.config.read_text())
    if config["behavior_processing"] != "on_demand" or config["query_batch"] != 16:
        raise ValueError("smoke run preserves on-demand history and query batch16")
    panel_path, candidate = ROOT / config["panel"], ROOT / config["candidate"]
    selection_path = ROOT / config["selection"]
    selection = json.loads(selection_path.read_text())
    freeze = json.loads((candidate / "freeze.json").read_text())
    if selection["panel_sha256"] != sha256(panel_path / "binding.json") or \
            selection["freeze_sha256"] != sha256(candidate / "freeze.json"):
        raise ValueError("retained selection/panel/candidate binding mismatch")
    users = choose_users(selection)
    if len(users) != config["expected_users"]:
        raise ValueError("unexpected stratum count")
    panel, groups, feedback = load_panel(panel_path)
    if not set(users).issubset(groups["evaluation"]):
        raise ValueError("smoke users left the existing development role")
    device = setup_device(args.gpu, threads=2)
    settings = dict(created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        config=config, users=users, pid=os.getpid(), device=torch.cuda.get_device_name(device),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        input_sha256=dict(config=sha256(args.config), selection=sha256(selection_path),
            panel=sha256(panel_path / "binding.json"), freeze=sha256(candidate / "freeze.json")),
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), Path(__file__).with_name("events.py"),
            ROOT / "src/hstu_kvcache/methods/cachetide/session.py",
            ROOT / "src/hstu_kvcache/methods/cachetide/backend.py",
            ROOT / "src/hstu_kvcache/methods/cachetide/read.py",
            ROOT / "src/hstu_kvcache/methods/cachetide/kernels/reader.py",
            ROOT / "src/hstu_kvcache/methods/cachetide/state.py")},
        scope="GPU scheduling correctness; all releases resident; no load, capacity, quality-gain or fresh confirmation claim",
        training=False, catalogue_recompiled=False)
    write_json(args.output / "configuration.json", settings)
    report_progress(args.output, "load_fixed_models", users=users)
    models = load_models(panel, device)
    versions = [version for version, _ in RELEASES]
    adapters, weights, _ = load_candidates(candidate / "calibration", device, versions, users)
    impl = ImplementationConfig(reader_backend=config["reader_backend"], row_chunk=config["row_chunk"])
    wrapped = {version: InferenceAdapter(adapter, impl) for version, adapter in adapters.items()}
    bank = load_bank(adapters["v3"].feature_bank_path, adapters["v3"].feature_bank_sha256,
                     device=device, dtype=torch.float32)
    feature_fn = partial(fixed_features_from_bank, bank=bank)
    prefixes, publications = {}, {}
    for version in versions:
        path = candidate / "publication" / f"{version}.pt"
        retained = json.loads(path.with_suffix(".json").read_text())
        prefix, metadata = load_prefix(path, adapters[version], device=device,
                                      expected_sha256=retained["artifact_sha256"])
        if weights[version]["sha256"] != freeze["release_bindings"][version]["calibration_sha256"] or \
                metadata["bindings"]["calibration_sha256"] != weights[version]["sha256"] or \
                metadata["bindings"]["current_model_sha256"] != panel["models"][version]["checkpoint_sha256"]:
            raise ValueError("publication/model/calibration binding mismatch")
        prefixes[version] = configure_prefix(prefix, impl, feature_fn=feature_fn)
        publications[version] = dict(path=str(path), sha256=retained["artifact_sha256"],
                                     model_sha256=panel["models"][version]["checkpoint_sha256"])
    settings.update(calibration=weights, publications=publications)
    write_json(args.output / "configuration.json", settings)
    histories = load_history(panel_path, requested_uids=users).rows
    completed, score_rows = [], []
    for protocol in config["protocols"]:
        requests = ({uid: candidate_rows(uid, histories[uid], stride=config["stride"],
            candidates=config["candidates"], known_vocab_size=panel["known_vocab_size"]) for uid in users}
            if protocol == "behavior_stride16_candidates16" else {uid: feedback[uid] for uid in users})
        trace = build_trace(histories, requests, RELEASES, END)
        audit = audit_trace(trace, histories, requests)
        for method in config["methods"]:
            report_progress(args.output, "global_replay", protocol=protocol, method=method,
                            completed_units=len(completed))
            snapshots = {}
            def retain(uid, version, value):
                snapshots[uid, version] = value
            global_handler = ModelReplayHandler(method, models, wrapped, prefixes, histories,
                users, query_batch=config["query_batch"], on_boundary=retain)
            event_counts = Counter()
            for event in trace:
                global_handler(event)
                event_counts[event.kind] += 1
                if event.kind == "publication":
                    report_progress(args.output, "global_release", protocol=protocol,
                                    method=method, version=event.version)
            torch.cuda.synchronize(device)
            global_scores = global_handler.scores
            del global_handler
            comparisons, max_score_error = [], 0.
            for uid in users:
                report_progress(args.output, "per_user_oracle", protocol=protocol, method=method, uid=uid)
                oracle = ModelReplayHandler(method, models, wrapped, prefixes, histories,
                                           [uid], query_batch=config["query_batch"])
                for index, (version, at) in enumerate(RELEASES):
                    oracle.publish(version, at)
                    by_time = defaultdict(list)
                    for row in requests[uid]:
                        if row["version"] == version:
                            by_time[int(row["query_timestamp"])].append(row)
                    for timestamp, pending in sorted(by_time.items()):
                        oracle.recommend(uid, timestamp, sorted(pending, key=lambda row: row["request_id"]))
                    until = RELEASES[index + 1][1] if index + 1 < len(RELEASES) else END
                    oracle.advance(uid, until)
                    comparison = compare_snapshot(state_snapshot(oracle.sessions[uid]),
                        snapshots[uid, version], atol=config["atol"], rtol=config["rtol"])
                    comparison.update(uid=uid, version=version)
                    comparisons.append(comparison)
                for row in requests[uid]:
                    rid = row["request_id"]
                    expected, actual = oracle.scores[rid], global_scores[rid]
                    if abs(actual - expected) > config["atol"] + config["rtol"] * abs(expected):
                        raise AssertionError(f"score mismatch {protocol}/{method}/{rid}")
                    max_score_error = max(max_score_error, abs(actual - expected))
                    score_rows.append(dict(protocol=protocol, method=method, **row,
                                           score=actual, global_score=actual, oracle_score=expected))
                del oracle
            unit = dict(protocol=protocol, method=method, candidate_scores=len(global_scores),
                        max_score_abs_error=max_score_error, boundary_comparisons=comparisons,
                        trace_audit=audit, event_counts=dict(event_counts), status="passed",
                        operation_ledgers=[dict(uid=uid, version=version, **{
                            key: value["metadata"][key] for key in (
                                "native_ledger", "service_costs", "read_ledger", "coded_read_ledger")})
                            for (uid, version), value in snapshots.items()])
            completed.append(unit)
            write_json(args.output / "completed_units.json", completed)
            del snapshots
            print(json.dumps(dict(status="unit_passed", protocol=protocol, method=method,
                                  scores=len(global_scores), max_abs=max_score_error)), flush=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(score_rows), args.output / "scores.parquet", compression="zstd")
    summary = dict(status="passed_gpu_global_replay_correctness", users=users,
        completed_units=len(completed), units=completed,
        total_paired_candidate_scores=len(score_rows),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        elapsed_seconds=time.perf_counter() - began,
        GPU_model_execution=True, loaded_service_measurement=False, quality_claim=False,
        scope=settings["scope"], old_results_modified=False)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "progress.json", dict(status="complete", pid=os.getpid()))
    print(json.dumps(dict(status=summary["status"], scores=len(score_rows),
                          elapsed_seconds=summary["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    log_stream = (args.output / "run.log").open("a", buffering=1)
    sys.stdout = sys.stderr = log_stream
    try:
        run(args)
    except BaseException as error:
        write_json(args.output / "summary.json", dict(status="failed", pid=os.getpid(),
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            GPU_model_execution="see progress.json", loaded_service_measurement=False))
        raise
