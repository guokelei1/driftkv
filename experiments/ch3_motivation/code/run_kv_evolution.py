#!/usr/bin/env python3
"""Bounded single-GPU KV-map evolution observations, without a probe ladder."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import ROOT, load_edge, load_models, quality_role, write_json
from experiments.ch3_motivation.code.candidate_residual import candidate_residual
from experiments.ch3_motivation.code.kv_evolution import (
    checkpoint_metadata, clone_native_trajectory, teacher_forced_cache,
)
from experiments.ch3_motivation.code.kv_score_batch import score_cache_group
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.methods.read_correction.calibration.v2.calibrate import mixed_candidates
from hstu_kvcache.common.artifacts import sha256
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

DEFAULT_THRESHOLDS = (8, 32, 128)
METHODS = ("reuse", "rebuild", "f", "teacher_forced")
FEEDBACK_SCHEMA = pa.schema([
    ("uid", pa.uint64()), ("query_timestamp", pa.int64()),
    ("raw_item_id", pa.int64()), ("item_idx", pa.int64()), ("label", pa.int64()),
    ("phase", pa.string()), ("method", pa.string()), ("score", pa.float64()),
    ("appended_events", pa.int64()), ("inherited_fraction", pa.float64()),
    ("history_rows", pa.int64()),
])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def observation_timeline(events, cursor, requests, at, end, thresholds):
    """Merge synthetic checkpoints with true request endpoints, preserving ties."""
    timeline = {int(at): dict(checkpoints=[("cutover", 0)], request_roles=[])}
    for threshold in thresholds:
        position = cursor + threshold - 1
        if position >= len(events[0]):
            continue
        until = int(events[0][position]) + 1
        if until >= end:
            continue
        timeline.setdefault(until, dict(checkpoints=[], request_roles=[]))["checkpoints"].append(
            (f"append_{threshold}", threshold))
    real_times = sorted(int(t) for t, rows in requests.items() if rows)
    if any(not at <= t < end for t in real_times):
        raise ValueError("real request outside the frozen edge window")
    for role, timestamp in (("first", real_times[0]), ("last", real_times[-1])) if real_times else ():
        timeline.setdefault(timestamp, dict(checkpoints=[], request_roles=[]))["request_roles"].append(role)
    return sorted(timeline.items()), real_times


@torch.no_grad()
def replay_user(parent, current, probe, events, requests, uid, cfg,
                *, thresholds=DEFAULT_THRESHOLDS):
    """One cutover prefill per backbone; all three native paths evolve causally."""
    at, end = int(cfg["release_day"]) * 86400, int(cfg["end_day"]) * 86400
    device = next(current.parameters()).device
    base = NativeTrajectory(parent, int(cfg["edge"]) - 1, events, at,
        max_length=int(cfg["history_length"]), append_band_size=int(cfg["append_band_size"]))
    initial_length = base.cache.seq_len
    item_ids = torch.as_tensor(base.retained_events[1], dtype=torch.long, device=device)[None]
    candidates = mixed_candidates(int(uid), base.retained_events[1], current.cfg.num_items - 256,
        count=16, seed=int(cfg.get("seed", 17)))
    panel = torch.as_tensor(candidates, dtype=torch.long, device=device)[None]
    reuse, teacher = clone_native_trajectory(base), clone_native_trajectory(base)
    reuse.switch(current, int(cfg["edge"]), at)
    teacher.switch(current, int(cfg["edge"]), at, rebuild=True)
    mapped = clone_native_trajectory(reuse)
    mapped.cache = probe.prepare_cache(current, base.cache, item_ids)
    initial_mapped = mapped.cache
    timeline, real_times = observation_timeline(events, base.cursor, requests, at, end, thresholds)
    record = dict(uid=int(uid), initial_length=initial_length,
        panel_candidates=[int(x) for x in candidates], snapshots=[],
        first_request_timestamp=real_times[0] if real_times else None,
        last_request_timestamp=real_times[-1] if real_times else None)
    feedback = []
    for until, roles in timeline:
        for path in (reuse, teacher, mapped):
            path.advance(until)
        if (len({p.cursor for p in (reuse, teacher, mapped)}) != 1
                or len({p.cache.seq_len for p in (reuse, teacher, mapped)}) != 1
                or len({p.evicted_tokens for p in (reuse, teacher, mapped)}) != 1):
            raise ValueError("paired trajectory event/FIFO alignment changed")
        metadata = checkpoint_metadata(mapped, initial_length)
        metadata["seconds_since_release"] = int(until - at)
        gap = 0 if mapped.last_timestamp is None else until - mapped.last_timestamp
        delta = torch.tensor([gap], dtype=torch.float32, device=device)
        forced = teacher_forced_cache(initial_mapped, teacher.cache, mapped.evicted_tokens)
        caches = dict(reuse=reuse.cache, rebuild=teacher.cache, f=mapped.cache, teacher_forced=forced)
        if roles["checkpoints"]:
            residual = candidate_residual(current, mapped.cache, teacher.cache, panel, delta)
            scores = {"f": residual["mapped_scores"]}
            controls = ("reuse", "rebuild", "teacher_forced")
            values = score_cache_group(current, [caches[m] for m in controls], panel, delta).cpu().tolist()
            scores.update(zip(controls, values, strict=True))
            target = np.asarray(scores["rebuild"], dtype=np.float64)
            score_mse = {method: float(np.mean((np.asarray(scores[method]) - target) ** 2))
                         for method in ("reuse", "f", "teacher_forced")}
            compact = dict(summary=residual["summary"],
                candidate_raw_mse=np.asarray(residual["candidate_raw_mse_by_layer_head"]).mean((0, 1)).tolist())
            for checkpoint, threshold in roles["checkpoints"]:
                candidate_info = dict(compact)
                if checkpoint == "cutover":
                    for key in ("raw_mse_by_layer_head", "length_normalized_mse_by_layer_head",
                                "centered_energy_ratio_by_layer_head"):
                        candidate_info[key] = residual[key]
                record["snapshots"].append(dict(metadata, checkpoint=checkpoint,
                    target_appends=int(threshold), candidate_residual=candidate_info,
                    score_mse=score_mse, fixed_panel_scores=scores))
        if roles["request_roles"]:
            phase = "first_last" if len(roles["request_roles"]) == 2 else roles["request_roles"][0]
            rows = requests[until]
            for start in range(0, len(rows), int(cfg.get("query_batch", 16))):
                batch = rows[start:start + int(cfg.get("query_batch", 16))]
                ids = torch.tensor([[int(row["item_idx"]) for row in batch]], device=device)
                scores = score_cache_group(current, [caches[m] for m in METHODS], ids, delta).cpu().tolist()
                for method, values in zip(METHODS, scores, strict=True):
                    for row, value in zip(batch, values, strict=True):
                        feedback.append(dict(uid=int(uid), query_timestamp=int(until),
                            raw_item_id=int(row["raw_item_id"]), item_idx=int(row["item_idx"]),
                            label=int(row["label"]), phase=phase, method=method, score=float(value),
                            appended_events=metadata["appended_events"],
                            inherited_fraction=metadata["inherited_fraction"],
                            history_rows=metadata["history_rows"]))
    return record, feedback


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    root = Path(args.input_root).resolve()
    population_path = root / "population.json"
    population = json.loads(population_path.read_text())
    users = population["main"][:args.users]
    if args.users < 1 or len(users) != args.users or not 1 <= args.block_users <= 128:
        raise ValueError("requested fixed main population or bounded block size unavailable")
    role = quality_role(users, diagnostic=args.diagnostic)
    binding, histories, requests = load_edge(root / f"edge_{args.edge}")
    if binding["population_sha256"] != sha256(population_path):
        raise ValueError("population input binding changed")
    for key in ("history", "requests"):
        if sha256(root / f"edge_{args.edge}" / binding[key]["path"]) != binding[key]["sha256"]:
            raise ValueError("frozen input data changed")
    checkpoint = Path(args.checkpoint).resolve()
    probe, metadata = load_variant(checkpoint, device)
    if probe.stage != "f" or int(metadata["identity"]["edge"]) != args.edge:
        raise ValueError("evolution starts from this edge's trained F checkpoint")
    reserved = metadata["identity"]["fit_uids"] + metadata["identity"]["validation_users"]
    if set(users) & set(reserved):
        raise ValueError("evaluation overlaps fitting/validation users")
    probe.eval().requires_grad_(False)
    parent, current, models = load_models(args.edge, device)
    if metadata["identity"]["models"] != models:
        raise ValueError("fitting and evolution backbones differ")
    cfg = binding["settings"]
    source_paths = [Path(__file__), Path(__file__).with_name("kv_evolution.py"),
        Path(__file__).with_name("candidate_residual.py"), Path(__file__).with_name("common.py"),
        Path(__file__).with_name("kv_score_batch.py"),
        Path(__file__).with_name("probe_variants.py"), Path(__file__).with_name("probes.py"),
        ROOT / "src/hstu_kvcache/evaluation/trajectory.py",
        ROOT / "src/hstu_kvcache/models/state_transition.py",
        ROOT / "src/hstu_kvcache/adaptation/reader.py",
        ROOT / "src/hstu_kvcache/models/hstu.py",
        ROOT / "src/hstu_kvcache/models/attention.py",
        ROOT / "src/hstu_kvcache/methods/read_correction/calibration/v2/calibrate.py"]
    sources = {str(path.resolve()): sha256(path) for path in source_paths}
    proof_path = ROOT / "experiments/ch3_motivation/runs/insight_mechanisms/attempt_1/numeric_batch/summary.json"
    proof = json.loads(proof_path.read_text())
    if (proof["status"] != "passed"
            or proof["helper_sha256"] != sha256(Path(__file__).with_name("kv_score_batch.py"))):
        raise ValueError("actual-model batched scoring proof changed")
    cost_path = checkpoint.parent / "cost_preflight.json"
    configuration = dict(format="kv_evolution_population_v1", edge=args.edge,
        input_root=str(root), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        population_sha256=sha256(population_path), main_uid_sha256=digest(users), users=users,
        input_binding_sha256=sha256(root / f"edge_{args.edge}/inputs.json"), input_data=binding,
        models=models, sources=sources, physical_gpu=args.gpu, hardware=torch.cuda.get_device_name(0),
        role=role, diagnostic=bool(args.diagnostic), block_users=args.block_users,
        append_thresholds=list(args.append_thresholds), fixed_panel=dict(count=16, seed=cfg.get("seed", 17),
            rule="mixed_candidates: only prerelease retained item IDs plus uniform candidates; no labels"),
        teacher="current-model cutover rebuild then causal native appends; not fresh per checkpoint",
        teacher_forcing="diagnostic only: same mapped initial inherited rows, teacher-produced new rows",
        score_batch_proof=dict(path=str(proof_path), sha256=sha256(proof_path)),
        cost_preflight_reference=dict(path=str(cost_path), sha256=sha256(cost_path)),
        metric_scope="synthetic panel has only teacher residuals; real first/last request labels only for AUC")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configuration_path = output / "configuration.json"
    if configuration_path.exists() and json.loads(configuration_path.read_text()) != configuration:
        raise ValueError("resume configuration, source, input or checkpoint identity changed")
    write_json(configuration_path, configuration)
    identity = digest(configuration)
    started = time.perf_counter()
    all_records, all_feedback, artifacts = [], [], []
    reused_users = 0
    for offset in range(0, len(users), args.block_users):
        selected = users[offset:offset + args.block_users]
        block_path = output / f"blocks/part_{offset:06d}.json"
        feedback_path = output / f"feedback/part_{offset:06d}.parquet"
        if block_path.exists():
            block = json.loads(block_path.read_text())
            if (block["status"] != "complete" or block["identity_sha256"] != identity or block["uids"] != selected
                    or block["feedback_sha256"] != sha256(feedback_path)):
                raise ValueError("committed block binding or feedback hash changed")
            records, feedback = block["users"], pq.read_table(feedback_path).to_pylist()
            reused_users += len(selected)
        else:
            records, feedback = [], []
            for uid in selected:
                record, values = replay_user(parent, current, probe, histories.rows[uid],
                    requests.get(uid, {}), uid, cfg, thresholds=args.append_thresholds)
                records.append(record)
                feedback.extend(values)
            if any(sha256(Path(path)) != value for path, value in sources.items()):
                raise ValueError("numerical source changed before committing population block")
            feedback_path.parent.mkdir(exist_ok=True)
            temporary = feedback_path.with_suffix(".parquet.writing")
            pq.write_table(pa.Table.from_pylist(feedback, schema=FEEDBACK_SCHEMA), temporary)
            temporary.replace(feedback_path)
            block = dict(status="complete", offset=offset, uids=selected, identity_sha256=identity,
                users=records, feedback_path=str(feedback_path.relative_to(output)),
                feedback_sha256=sha256(feedback_path))
            write_json(block_path, block)
        all_records.extend(records)
        all_feedback.extend(feedback)
        artifacts.append(dict(path=str(block_path.relative_to(output)), sha256=sha256(block_path),
                              feedback_path=str(feedback_path.relative_to(output)),
                              feedback_sha256=sha256(feedback_path)))
        progress = dict(status="running", edge=args.edge, completed_users=len(all_records),
            total_users=len(users), reused_users=reused_users, seconds=time.perf_counter() - started)
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    support = Counter(point["checkpoint"] for user in all_records for point in user["snapshots"])
    feedback_uids = {row["uid"] for row in all_feedback}
    phase_support = {phase: len({row["uid"] for row in all_feedback
        if row["phase"] in (phase, "first_last")}) for phase in ("first", "last")}
    summary = dict(status="complete", edge=args.edge, users=len(users), diagnostic=bool(args.diagnostic),
        role=role, physical_gpu=args.gpu, checkpoint_support=dict(support),
        feedback_users=len(feedback_uids), real_phase_uid_support=phase_support,
        distinct_first_last_users=sum(u["first_request_timestamp"] is not None
            and u["first_request_timestamp"] != u["last_request_timestamp"] for u in all_records),
        feedback_rows_all_methods=len(all_feedback), methods=list(METHODS), artifacts=artifacts,
        cost_preflight_reference=configuration["cost_preflight_reference"],
        seconds=time.perf_counter() - started, resumed_users=reused_users,
        limitations=["Teacher-forced new rows are a diagnostic, not a serving method.",
            "Synthetic checkpoints have no observed-label AUC.",
            "Main resident count does not imply every checkpoint/feedback phase has that support.",
            "Each edge is independent; this run does not establish cross-version quality."],
        statistics="Aggregate paired feedback phases and synthetic residuals separately.")
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", dict(progress, status="complete"))
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--edge", type=int, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--gpu", type=int, default=1)
    result.add_argument("--users", type=int, default=2000)
    result.add_argument("--diagnostic", action="store_true")
    result.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs")
    result.add_argument("--block-users", type=int, default=64)
    result.add_argument("--append-thresholds", type=int, nargs="+", default=list(DEFAULT_THRESHOLDS))
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if any(x <= 0 for x in args.append_thresholds) or len(set(args.append_thresholds)) != len(args.append_thresholds):
        raise ValueError("append thresholds must be distinct positive counts")
    try:
        result = run(args)
        write_json(args.output / "exit.json", dict(exit_code=0))
        print(json.dumps(dict(status="complete", users=result["users"], output=str(args.output))), flush=True)
    except BaseException as error:
        write_json(args.output / "exit.json", dict(exit_code=1, error=str(error)))
        raise
