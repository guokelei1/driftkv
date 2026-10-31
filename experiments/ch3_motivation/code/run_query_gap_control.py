#!/usr/bin/env python3
"""Separate append-context and query-gap changes in fixed KV-map snapshots."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import ROOT, load_edge, load_models, quality_role, write_json
from experiments.ch3_motivation.code.kv_evolution import (
    clone_native_trajectory, checkpoint_metadata, teacher_forced_cache,
)
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.run_kv_evolution import observation_timeline
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.methods.read_correction.calibration.v2.calibrate import mixed_candidates
from hstu_kvcache.common.artifacts import sha256
import numpy as np
import torch

METHODS = ("reuse", "f", "teacher_forced", "rebuild")
GAPS = ("actual_gap", "cutover_gap")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@torch.no_grad()
def score_gap_grid(model, caches, candidates, gaps):
    """Independent cache/gap pairs in one native scoring batch; no KV writes."""
    if len({c.seq_len for c in caches}) != 1 or candidates.shape[0] != 1:
        raise ValueError("one user's aligned caches and candidate panel required")
    grid = list(caches) * len(gaps)
    merged = HSTUKVCache(torch.cat([c.k for c in grid], dim=1),
                         torch.cat([c.v for c in grid], dim=1), caches[0].seq_len)
    delta = torch.tensor([g for g in gaps for _ in caches], device=candidates.device,
                         dtype=torch.float32)
    values, _ = native_score(model, merged, candidates.expand(len(grid), -1), delta)
    return values.reshape(len(gaps), len(caches), candidates.shape[1])


def metrics(values, teacher):
    values, teacher = np.asarray(values), np.asarray(teacher)
    error = values - teacher
    centered = error - error.mean()
    i, j = np.triu_indices(len(teacher), 1)
    margins = teacher[i] - teacher[j]
    valid = np.abs(margins) > 1e-6
    disagreements = np.sign(values[i] - values[j]) != np.sign(margins)
    return dict(total_mse=float(np.mean(error ** 2)),
                centered_mse=float(np.mean(centered ** 2)),
                common_offset_mse=float(error.mean() ** 2),
                teacher_candidate_variance=float(np.var(teacher)),
                pair_discord=float(disagreements[valid].mean()) if valid.any() else None)


@torch.no_grad()
def replay_user(parent, current, probe, events, requests, uid, cfg, *, check_batch=False):
    at, end = cfg["release_day"] * 86400, cfg["end_day"] * 86400
    device = next(current.parameters()).device
    base = NativeTrajectory(parent, cfg["edge"] - 1, events, at,
        max_length=cfg["history_length"], append_band_size=cfg["append_band_size"])
    initial_length = base.cache.seq_len
    initial_gap = 0 if base.last_timestamp is None else at - base.last_timestamp
    ids = torch.as_tensor(base.retained_events[1], device=device, dtype=torch.long)[None]
    candidates = mixed_candidates(uid, base.retained_events[1], current.cfg.num_items - 256,
                                  count=16, seed=cfg.get("seed", 17))
    panel = torch.as_tensor(candidates, device=device, dtype=torch.long)[None]
    reuse, teacher = clone_native_trajectory(base), clone_native_trajectory(base)
    reuse.switch(current, cfg["edge"], at)
    teacher.switch(current, cfg["edge"], at, rebuild=True)
    mapped = clone_native_trajectory(reuse)
    mapped.cache = probe.prepare_cache(current, base.cache, ids)
    initial_mapped = mapped.cache
    # Preserve the original advance boundaries (first/last requests included).
    timeline, _ = observation_timeline(events, base.cursor, requests, at, end, (8, 32, 128))
    result = dict(uid=int(uid), candidates=[int(c) for c in candidates], snapshots=[])
    max_batch_error = 0.
    for until, roles in timeline:
        for path in (reuse, teacher, mapped):
            path.advance(until)
        if not roles["checkpoints"]:
            continue
        actual_gap = 0 if mapped.last_timestamp is None else until - mapped.last_timestamp
        forced = teacher_forced_cache(initial_mapped, teacher.cache, mapped.evicted_tokens)
        caches = [reuse.cache, mapped.cache, forced, teacher.cache]
        values = score_gap_grid(current, caches, panel, [actual_gap, initial_gap])
        if check_batch:
            separate = torch.stack([torch.stack([native_score(current, c, panel,
                torch.tensor([gap], device=device, dtype=torch.float32))[0][0] for c in caches])
                for gap in [actual_gap, initial_gap]])
            max_batch_error = max(max_batch_error, float((values - separate).abs().max()))
            if not torch.allclose(values, separate, atol=1e-6, rtol=1e-5):
                raise ValueError("gap/cache batch differs from independent native scores")
        values = values.cpu().tolist()
        table = {gap: {m: row for m, row in zip(METHODS, rows, strict=True)}
                 for gap, rows in zip(GAPS, values, strict=True)}
        measured = {gap: {m: metrics(rows[m], rows["rebuild"]) for m in METHODS}
                    for gap, rows in table.items()}
        for checkpoint, threshold in roles["checkpoints"]:
            result["snapshots"].append(dict(checkpoint_metadata(mapped, initial_length),
                checkpoint=checkpoint, target_appends=threshold, actual_gap=int(actual_gap),
                cutover_gap=int(initial_gap), scores=table, metrics=measured))
    result["max_batch_abs_error"] = max_batch_error if check_batch else None
    return result


def estimate(array, weights):
    a = np.asarray(array, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite paired query-gap metric")
    draws = weights @ a / a.size
    return dict(mean=float(a.mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist())


def analyze(records, *, diagnostic, replicates=500):
    index = {u["uid"]: {p["checkpoint"]: p for p in u["snapshots"]} for u in records}
    out = []
    for checkpoint in ("cutover", "append_8", "append_32", "append_128"):
        uids = [u for u, rows in index.items() if checkpoint in rows]
        if not uids:
            out.append(dict(checkpoint=checkpoint, unique_users=0, mechanism_use_permitted=False))
            continue
        rng = np.random.default_rng(17)
        weights = np.asarray([np.bincount(rng.integers(len(uids), size=len(uids)),
                              minlength=len(uids)) for _ in range(replicates)], dtype=np.uint16)
        point = dict(checkpoint=checkpoint, unique_users=len(uids),
                     mechanism_use_permitted=not diagnostic and len(uids) >= 512, gaps={})
        for gap in GAPS:
            point["gaps"][gap] = {}
            for metric in ("total_mse", "centered_mse", "common_offset_mse"):
                now = [index[u][checkpoint]["metrics"][gap]["f"][metric] for u in uids]
                start = [index[u]["cutover"]["metrics"][gap]["f"][metric] for u in uids]
                forced = [index[u][checkpoint]["metrics"][gap]["teacher_forced"][metric] for u in uids]
                point["gaps"][gap][metric] = dict(current=estimate(now, weights),
                    matched_cutover=estimate(start, weights),
                    paired_change=estimate(np.asarray(now) - start, weights),
                    teacher_forcing_improvement=estimate(np.asarray(now) - forced, weights))
        point["actual_minus_fixed_gap_centered_mse"] = estimate([
            index[u][checkpoint]["metrics"]["actual_gap"]["f"]["centered_mse"] -
            index[u][checkpoint]["metrics"]["cutover_gap"]["f"]["centered_mse"] for u in uids], weights)
        out.append(point)
    return dict(checkpoints=out, paired_uid_bootstrap_replicates=replicates,
        scope="synthetic teacher-logit agreement; not observed-feedback AUC; fixed-gap controls time input, not all context changes")


def run(args):
    if args.gpu != 1:
        raise ValueError("this extension uses only physical GPU1")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    root = ROOT / "experiments/ch3_motivation/inputs"
    population = json.loads((root / "population.json").read_text())
    users = population["main"][:args.users]
    role = quality_role(users, diagnostic=args.diagnostic)
    if len(users) != args.users:
        raise ValueError("unavailable frozen main users")
    binding, histories, requests = load_edge(root / f"edge_{args.edge}")
    for key in ("history", "requests"):
        if sha256(root / f"edge_{args.edge}" / binding[key]["path"]) != binding[key]["sha256"]:
            raise ValueError("frozen edge data changed")
    if binding["population_sha256"] != sha256(root / "population.json"):
        raise ValueError("frozen population binding changed")
    probe, meta = load_variant(args.checkpoint, device)
    parent, current, models = load_models(args.edge, device)
    if (probe.stage != "f" or meta["identity"]["edge"] != args.edge
            or meta["identity"]["models"] != models
            or set(users) & set(meta["identity"]["fit_uids"] + meta["identity"]["validation_users"])):
        raise ValueError("F identity/model/role mismatch")
    probe.eval().requires_grad_(False)
    paths = [Path(__file__), *[Path(__file__).with_name(n) for n in
        ("common.py", "run_kv_evolution.py", "kv_evolution.py", "probe_variants.py", "probes.py")],
        ROOT / "src/hstu_kvcache/adaptation/reader.py", ROOT / "src/hstu_kvcache/evaluation/trajectory.py",
        ROOT / "src/hstu_kvcache/models/hstu.py", ROOT / "src/hstu_kvcache/models/attention.py"]
    sources = {str(p.resolve()): sha256(p) for p in paths}
    config = dict(format="kv_query_gap_control_v1", edge=args.edge, users=users, physical_gpu=1,
        models=models, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        sources=sources, input_binding_sha256=sha256(root / f"edge_{args.edge}/inputs.json"),
        population_sha256=sha256(root / "population.json"), role=role, diagnostic=args.diagnostic,
        block_users=args.block_users, check_batch=args.check_batch,
        scope="same original checkpoint/advance timeline; each cache scored at actual gap and its own cutover gap",
        reference="insight_mechanisms/attempt_1 main_edge; source weights/population unchanged; research-only extra reads")
    args.output.mkdir(parents=True, exist_ok=True)
    p = args.output / "configuration.json"
    if p.exists() and json.loads(p.read_text()) != config:
        raise ValueError("new source/input/settings require a separate output")
    write_json(p, config)
    identity = digest(config)
    records, artifacts = [], []
    started = time.perf_counter()
    for offset in range(0, len(users), args.block_users):
        selected = users[offset:offset + args.block_users]
        path = args.output / f"blocks/part_{offset:06d}.json"
        if path.exists():
            block = json.loads(path.read_text())
            if block["identity_sha256"] != identity or block["uids"] != selected:
                raise ValueError("committed query-gap block changed")
        else:
            rows = [replay_user(parent, current, probe, histories.rows[u], requests.get(u, {}),
                u, binding["settings"], check_batch=args.check_batch) for u in selected]
            if any(sha256(Path(p)) != h for p, h in sources.items()):
                raise ValueError("numerical source changed during run")
            block = dict(identity_sha256=identity, uids=selected, users=rows)
            write_json(path, block)
        records.extend(block["users"])
        artifacts.append(dict(path=str(path.relative_to(args.output)), sha256=sha256(path)))
        progress = dict(completed_users=len(records), total_users=len(users), seconds=time.perf_counter()-started)
        write_json(args.output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    summary = dict(status="complete", users=len(records), edge=args.edge, diagnostic=args.diagnostic,
        role=role, physical_gpu=1, artifacts=artifacts, seconds=time.perf_counter()-started,
        max_batch_abs_error=max((r["max_batch_abs_error"] or 0.) for r in records) if args.check_batch else None,
        **analyze(records, diagnostic=args.diagnostic))
    write_json(args.output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edge", type=int, choices=(2, 3), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--users", type=int, default=2000)
    p.add_argument("--block-users", type=int, default=64)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--check-batch", action="store_true")
    args = p.parse_args()
    try:
        run(args)
        write_json(args.output / "exit.json", dict(exit_code=0))
    except BaseException as error:
        write_json(args.output / "exit.json", dict(exit_code=1, error=str(error)))
        raise
