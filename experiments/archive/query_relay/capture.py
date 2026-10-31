#!/usr/bin/env python3
"""Pre-release calibration sources from this candidate's corrected trajectory.

Only frozen fit/validation users and events strictly before the target release
are used. Transient recommendation scores do not persist; their known past
timestamps are replay boundaries, without loading labels into the session.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import torch

from hstu_kvcache.evaluation.release_context import RELEASES, ledger_flops, load_models, setup_device
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.methods.cachetide.adapter import UnifiedRelayAdapter
from hstu_kvcache.methods.cachetide.cost import read_flops
from hstu_kvcache.methods.cachetide.session import RelaySession
from hstu_kvcache.evaluation.cachetide_workload import trigger_indices
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.methods.read_correction.calibration.v2.calibrate import mixed_candidates
from hstu_kvcache.common.artifacts import sha256, write_json

PANEL = ROOT / "results/design_two_2026_10/benchmark_panel"


def ledger_records(counter, names):
    return [dict(**dict(zip(names, key, strict=True)), count=int(value))
            for key, value in sorted(counter.items())]


def load_prior_candidates(folder, versions, device):
    adapters, records = {}, {}
    for version in versions:
        path = folder / version / "calibration.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["kind"] != UnifiedRelayAdapter.kind:
            raise ValueError(f"not a unified relay checkpoint: {path}")
        adapter = UnifiedRelayAdapter(**payload["config"]).to(device).eval().requires_grad_(False)
        adapter.load_state_dict(payload["state_dict"])
        adapters[version] = adapter
        records[version] = dict(path=str(path), sha256=sha256(path), config=payload["config"])
        if adapter.feature_mode == "origin_encoder":
            from hstu_kvcache.methods.cachetide.features import load_bank
            bank = load_bank(adapter.feature_bank_path, adapter.feature_bank_sha256,
                             device="cpu", dtype=next(adapter.parameters()).dtype)
            records[version]["feature_bank"] = dict(path=adapter.feature_bank_path,
                sha256=adapter.feature_bank_sha256, storage_bytes=bank.storage_bytes(),
                item_table_bytes=bank.table.numel() * bank.table.element_size(),
                scope="Immutable shared asset already built; no per-release preparation charge")
    return adapters, records


@torch.no_grad()
def capture_candidate_rows(models, adapters, histories, requests, uids, target, known,
                           *, workload="feedback", stride=16):
    """Return the source-pack schema consumed by calibrate.prepare."""
    current = models[target]
    device = next(current.parameters()).device
    cutover = dict(RELEASES)[target]
    prior_releases = tuple(row for row in RELEASES if row[1] < cutover)
    teacher_cost = {backend: CostModel(current.cfg.hidden_size, len(current.blocks),
                    current.cfg.num_heads, backend) for backend in ("torch", "triton")}
    rows, costs, records = {}, Counter(), []
    native_geometry, code_geometry, update_geometry = Counter(), Counter(), Counter()
    normalization_flops = 0
    for number, uid in enumerate(uids):
        cap = models["v2"].cfg.max_seq_len
        trajectory = NativeTrajectory(models["v2"], 2, histories.rows[uid], RELEASES[0][1],
                                      max_length=cap, append_band_size=min(32, cap))
        session = RelaySession(trajectory)
        behavior_boundaries = None
        if workload == "behavior":
            behavior_boundaries = trigger_indices(histories.rows[uid], stride=stride,
                known_vocab_size=known, releases=tuple(row for row in RELEASES if row[1] < cutover),
                end=cutover)
        for index, (version, at) in enumerate(prior_releases):
            session.publish(models[version], int(version[1:]), at, adapters[version])
            until = prior_releases[index + 1][1] if index + 1 < len(prior_releases) else cutover
            if workload == "behavior":
                timestamps = sorted({int(histories.rows[uid][0][position])
                                     for position in behavior_boundaries[version]})
            else:
                timestamps = sorted({int(row["query_timestamp"]) for row in requests.get(uid, [])
                                     if row["version"] == version and at <= int(row["query_timestamp"]) < until})
            for timestamp in timestamps:
                # Candidate reads have no persistent result. Preparation and
                # chronological append boundaries suffice to reproduce state.
                session.prepare_read(timestamp)
                costs["past_request_boundaries_without_score_execution"] += 1
            session.advance(until)
        parent, previous = session.source_snapshot(cutover)
        if not parent.seq_len:
            raise ValueError(f"calibration UID {uid} has no pre-release history")
        if trajectory.last_timestamp >= cutover:
            raise ValueError("source history includes a target-release event")
        teacher = trajectory.fresh_cache(current)
        items = torch.as_tensor(trajectory.retained_events[1], device=device, dtype=torch.long)[None]
        row = dict(parent=parent.to("cpu"), teacher=teacher.to("cpu"),
                   candidates=torch.tensor(mixed_candidates(uid, trajectory.retained_events[1],
                                            known, 16, seed=17)),
                   query_delta=float(cutover - trajectory.last_timestamp),
                   item_ids=items.detach().cpu(),
                   item_features=current.lookup_item_embeddings(items).detach().cpu())
        if previous is not None:
            row["previous_codes"] = previous.codes.detach().cpu()
        rows[uid] = row
        costs["source_trajectory_flops"] += sum(ledger_flops(trajectory.ledger, current, "triton").values())
        costs["teacher_cache_capture_flops"] += teacher_cost[
            "triton" if teacher.seq_len == current.cfg.max_seq_len else "torch"].full_cache(teacher.seq_len)
        for (kind, version, old_length, queries, coded_rows, native_rows), count in session.coded_read_ledger.items():
            adapter = adapters[f"v{version}"]
            if coded_rows:
                work = read_flops(
                    pairs=queries * coded_rows, queries=queries, head_rank=adapter.accessible_rank,
                    layers=adapter.num_layers, heads=adapter.heads, head_dim=adapter.head_dim,
                    gate_normalization=adapter.gate_normalization)
                costs["source_corrected_reader_added_flops"] += count * work["total"]
                normalization_flops += count * work["gate_query_normalization_flops"]
        for event in session.events:
            if event["action"] == "explicit_update":
                update_geometry[event["version"], event["rows"]] += 1
        native_geometry.update(trajectory.ledger)
        code_geometry.update(session.coded_read_ledger)
        records.append(dict(uid=int(uid), history_length=parent.seq_len,
            last_history_timestamp=int(trajectory.last_timestamp), target_cutover=cutover,
            producer_counts={str(int(p)): int((trajectory.producers == p).sum())
                             for p in set(trajectory.producers.tolist())},
            previous_code_version=trajectory.version, coded_rows=session.coded_rows,
            source_code_counters=dict(session.costs),
            source_writes="current model with corrected history", teacher="current raw-prefix reconstruction",
            memory=dict(session.memory)))
        if (number + 1) % 16 == 0 or number + 1 == len(uids):
            print(json.dumps(dict(status="candidate_source_capture", target=target,
                                  users=number + 1, total=len(uids))), flush=True)
    geometry = dict(native=ledger_records(native_geometry, ("action", "version", "before", "width")),
        code_reads=ledger_records(code_geometry,
            ("kind", "version", "old_length", "queries", "coded_rows", "native_rows")),
        explicit_updates=ledger_records(update_geometry, ("version", "rows")),
        gate_query_normalization_flops_included_in_reader_total=normalization_flops,
        scope="Actual capture execution; row/item update arithmetic is retained as shapes and still needs complete pricing.")
    return rows, dict(costs), records, geometry


def run(args):
    began = time.perf_counter()
    panel, groups, requests = load_panel(args.panel)
    fit = list(groups["fit"][:args.fit_users])
    validation = list(groups["validation"][:args.validation_users])
    if not fit or not validation or set(fit) & set(validation):
        raise ValueError("nonempty disjoint fit and validation groups are required")
    if (set(fit) | set(validation)) & set(groups["evaluation"]):
        raise ValueError("calibration users overlap evaluation")
    target = f"v{args.target_version}"
    versions = [version for version, at in RELEASES if at < dict(RELEASES)[target]]
    if versions and args.calibration_root is None:
        raise ValueError("later-release capture needs prior candidate checkpoints")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = dict(target=target, target_cutover=dict(RELEASES)[target], fit_uids=fit,
        validation_uids=validation, panel=str(args.panel), panel_sha256=sha256(args.panel / "binding.json"),
        calibration_root=None if args.calibration_root is None else str(args.calibration_root),
        role="pre_release_candidate_source_capture", seed=17, queries_per_user=16,
        no_future_events_or_request_labels=True, source_mode="candidate_continuous_corrected_writes",
        execution_sources={name: sha256(Path(__file__).parent / name)
                           for name in ("capture.py", "session.py", "adapter.py")})
    settings["workload"] = dict(kind=args.workload,
        stride=args.stride if args.workload == "behavior" else None,
        candidates=args.candidates if args.workload == "behavior" else None,
        boundary_scope="Recommendation timestamps only; transient scores and candidate count do not change persistent state")
    write_json(args.output / "configuration.json", settings)
    device = setup_device(args.gpu, args.threads)
    models = load_models(panel, device, last_version=args.target_version)
    adapters, checkpoints = load_prior_candidates(args.calibration_root, versions, device)
    histories = load_history(args.panel, requested_uids=fit + validation)
    rows, costs, records, geometry = capture_candidate_rows(models, adapters, histories, requests,
        fit + validation, target, panel["known_vocab_size"], workload=args.workload, stride=args.stride)
    payload = dict(rows=rows, costs=costs, records=records, geometry=geometry, settings=settings,
                   prior_checkpoints=checkpoints)
    path = args.output / "source.pt"
    torch.save(payload, path)
    summary = dict(status="complete", **settings, weights=checkpoints,
        shared_feature_banks={record["feature_bank"]["sha256"]: record["feature_bank"]
                              for record in checkpoints.values() if "feature_bank" in record},
        source_pack=str(path), source_pack_sha256=sha256(path), costs=costs, records=records,
        geometry=geometry, elapsed_seconds=time.perf_counter() - began,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        cost_limit="Native source replay and teacher prefix are priced. Added reader arithmetic is analytic; source row/item update shapes require final pricing. No complete-FLOPs claim.")
    write_json(args.output / "summary.json", summary)
    print(json.dumps(dict(status="complete", target=target, users=len(rows),
                         source_pack=str(path), costs=costs)), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory; pack is written as source.pt")
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--target-version", type=int, choices=(3, 4, 5), required=True)
    p.add_argument("--calibration-root", type=Path)
    p.add_argument("--fit-users", type=int, default=64)
    p.add_argument("--validation-users", type=int, default=16)
    p.add_argument("--workload", choices=("feedback", "behavior"), default="feedback")
    p.add_argument("--stride", type=int, choices=(8, 16, 32), default=16)
    p.add_argument("--candidates", type=int, choices=(2, 16), default=16)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
