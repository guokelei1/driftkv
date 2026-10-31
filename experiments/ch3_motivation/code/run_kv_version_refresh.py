#!/usr/bin/env python3
"""Fixed-history version adaptation diagnostic; no feedback labels or AUC."""
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

from experiments.ch3_motivation.code.common import ROOT, load_models, quality_role, restore_pack, write_json
from experiments.ch3_motivation.code.kv_score_batch import score_cache_group
from experiments.ch3_motivation.code.probe_cost import cached_preparation_flops
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.evaluation.yambda import load_model
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.methods.read_correction.calibration.v2.calibrate import mixed_candidates
from hstu_kvcache.common.artifacts import sha256
import numpy as np
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def validate_refresh_pair(previous, prior, refresh, metadata, *, history_edge,
                          target_version, models, uids, predecessor_sha256, history_cutoff):
    """A refresh of actual mapped inputs cannot be replaced by native-adjacent F."""
    old, new = prior["identity"], metadata["identity"]
    if (previous.stage != "f" or refresh.stage != "f"
            or metadata.get("method_label") != "refresh_map"
            or new.get("protocol") != "frozen_mapped_source_version_refresh_v1"
            or new.get("source_kind") != "actual_predecessor_mapped_cache_not_native_old_target"):
        raise ValueError("requires the trained actual-mapped-input refresh protocol")
    if (target_version != history_edge + 1 or old["edge"] != history_edge
            or new["history_edge"] != history_edge
            or new["source_native_version"] != history_edge - 1
            or new["predecessor_target_version"] != history_edge
            or new["target_version"] != target_version
            or new["history_cutoff"] != history_cutoff
            or new["predecessor_checkpoint_sha256"] != predecessor_sha256
            or new["models"] != models):
        raise ValueError("refresh source/target/model/predecessor identity differs")
    if any(models.get(key) != value for key, value in old["models"].items()) or len(old["models"]) != 2:
        raise ValueError("previous F is not the paired source-to-old-target map")
    for field in ("input_binding_sha256", "population_sha256", "fit_uids", "validation_users"):
        if old[field] != new[field]:
            raise ValueError("previous/refresh fitting inputs or roles differ")
    if set(uids) & set(new["fit_uids"] + new["validation_users"]):
        raise ValueError("version evaluation overlaps fitting or validation users")
    geometry = ("num_layers", "num_heads", "head_dim", "item_dim", "include_item")
    if any(getattr(previous.cfg, name) != getattr(refresh.cfg, name) for name in geometry):
        raise ValueError("previous/refresh head or item geometry differs")


def endpoint(version, names, scores):
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite version diagnostic scores")
    table = {name: row.tolist() for name, row in zip(names, values, strict=True)}
    teacher = np.asarray(table["teacher"])
    mse = {name: float(np.mean((np.asarray(row) - teacher) ** 2)) for name, row in table.items()}
    base = mse["source_unmapped"]
    relative = {name: value / base if base > 0 else None for name, value in mse.items()}
    return dict(version=int(version), scores=table, score_mse=mse, relative_to_unmapped=relative)


@torch.no_grad()
def replay_user(source_model, old_target, new_target, previous, refresh, events, uid, cfg):
    """Freeze one true cutover prefix and one Q16 panel under both targets."""
    at = int(cfg["release_day"]) * 86400
    device = next(new_target.parameters()).device
    source = NativeTrajectory(source_model, int(cfg["edge"]) - 1, events, at,
        max_length=int(cfg["history_length"]), append_band_size=int(cfg["append_band_size"]))
    ids = torch.as_tensor(source.retained_events[1], dtype=torch.long, device=device)[None]
    # OLD target embeddings are essential: this reproduces what was stored.
    unchanged = previous.prepare_cache(old_target, source.cache, ids)
    old_teacher = source.fresh_cache(old_target)
    new_teacher = source.fresh_cache(new_target)
    refreshed = refresh.prepare_cache(new_target, unchanged, ids)
    candidates = mixed_candidates(int(uid), source.retained_events[1], new_target.cfg.num_items - 256,
                                  count=16, seed=int(cfg.get("seed", 17)))
    panel = torch.as_tensor(candidates, dtype=torch.long, device=device)[None]
    gap = 0 if source.last_timestamp is None else at - source.last_timestamp
    delta = torch.tensor([gap], dtype=torch.float32, device=device)
    old_scores = score_cache_group(old_target, [source.cache, unchanged, old_teacher], panel, delta).cpu().tolist()
    new_scores = score_cache_group(new_target, [source.cache, unchanged, refreshed, new_teacher], panel, delta).cpu().tolist()
    return dict(uid=int(uid), history_length=int(source.cache.seq_len), query_delta=int(gap),
        history_cutoff=at, panel_candidates=[int(x) for x in candidates],
        endpoints=dict(old_target=endpoint(cfg["edge"], ("source_unmapped", "unchanged", "teacher"), old_scores),
            new_target=endpoint(cfg["edge"] + 1, ("source_unmapped", "unchanged", "refresh", "teacher"), new_scores)))


def run(args):
    if args.gpu != 1 or args.target_version != args.history_edge + 1:
        raise ValueError("use GPU1 and the next target version for this two-case study")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    root = Path(args.input_root).resolve()
    population_path = root / "population.json"
    population = json.loads(population_path.read_text())
    users = population["main"][:args.users]
    if args.users < 1 or len(users) != args.users or not 1 <= args.block_users <= 128:
        raise ValueError("fixed main population or bounded block size unavailable")
    role = quality_role(users, diagnostic=args.diagnostic)
    directory = root / f"edge_{args.history_edge}"
    binding_path = directory / "inputs.json"
    binding = json.loads(binding_path.read_text())
    history_path = directory / binding["history"]["path"]
    if binding["population_sha256"] != sha256(population_path) or sha256(history_path) != binding["history"]["sha256"]:
        raise ValueError("frozen main roles or history pack changed")
    histories = restore_pack(history_path)  # Feedback parquet is never read.
    cfg = binding["settings"]
    source, old_target, models = load_models(args.history_edge, device)
    target_path = ROOT / f"results/unified_training_2026_09/large/seed17/checkpoints/v{args.target_version}/checkpoint_100.pt"
    new_target, payload = load_model(target_path, device)
    del payload
    new_target.eval().requires_grad_(False)
    models[str(target_path.relative_to(ROOT))] = sha256(target_path)
    if any((m.cfg.hidden_size, len(m.blocks), m.cfg.num_heads, m.cfg.num_items) !=
           (old_target.cfg.hidden_size, len(old_target.blocks), old_target.cfg.num_heads, old_target.cfg.num_items)
           for m in (source, new_target)):
        raise ValueError("version diagnostic requires one stable model/catalog geometry")
    previous_path, refresh_path = Path(args.predecessor_checkpoint).resolve(), Path(args.refresh_checkpoint).resolve()
    previous, prior = load_variant(previous_path, device)
    refresh, metadata = load_variant(refresh_path, device)
    previous.eval().requires_grad_(False)
    refresh.eval().requires_grad_(False)
    validate_refresh_pair(previous, prior, refresh, metadata, history_edge=args.history_edge,
        target_version=args.target_version, models=models, uids=users,
        predecessor_sha256=sha256(previous_path), history_cutoff=int(cfg["release_day"]) * 86400)
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ("common.py", "kv_score_batch.py", "probe_variants.py", "probes.py", "probe_cost.py")],
        ROOT / "src/hstu_kvcache/evaluation/trajectory.py", ROOT / "src/hstu_kvcache/models/state_transition.py",
        ROOT / "src/hstu_kvcache/adaptation/reader.py", ROOT / "src/hstu_kvcache/models/hstu.py",
        ROOT / "src/hstu_kvcache/models/attention.py", ROOT / "src/hstu_kvcache/evaluation/yambda.py",
        ROOT / "src/hstu_kvcache/methods/read_correction/calibration/v2/calibrate.py", ROOT / "src/hstu_kvcache/evaluation/cost.py"]
    sources = {str(p.resolve()): sha256(p) for p in paths}
    configuration = dict(format="fixed_history_version_refresh_population_v1", history_edge=args.history_edge,
        target_version=args.target_version, input_root=str(root), users=users, main_uid_sha256=digest(users),
        population_sha256=sha256(population_path), input_binding_sha256=sha256(binding_path),
        history_pack=dict(path=str(history_path), sha256=sha256(history_path)), settings=cfg, models=models,
        predecessor_checkpoint=str(previous_path), predecessor_checkpoint_sha256=sha256(previous_path),
        refresh_checkpoint=str(refresh_path), refresh_checkpoint_sha256=sha256(refresh_path),
        refresh_fitting_identity=metadata["identity"], sources=sources, physical_gpu=1,
        hardware=torch.cuda.get_device_name(0), diagnostic=bool(args.diagnostic), role=role,
        block_users=args.block_users,
        panel="Q16 prerelease history+uniform, same IDs and cutover query gap under both target models",
        source_state="previous F materialized with OLD target embeddings; G input is that actual state",
        scope="fixed-history synthetic teacher diagnostic; no feedback labels, serving trajectory or AUC",
        limitation="New G also adds capacity and optimization; no matched old-target G control is claimed.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("resume version configuration/source/input/checkpoint identity changed")
    write_json(config_path, configuration)
    identity = digest(configuration)
    records, artifacts = [], []
    reused = 0
    started = time.perf_counter()
    for offset in range(0, len(users), args.block_users):
        selected = users[offset:offset + args.block_users]
        block_path = output / f"blocks/part_{offset:06d}.json"
        if block_path.exists():
            block = json.loads(block_path.read_text())
            if block["status"] != "complete" or block["identity_sha256"] != identity or block["uids"] != selected:
                raise ValueError("committed version block identity changed")
            values = block["users"]
            reused += len(selected)
        else:
            values = [replay_user(source, old_target, new_target, previous, refresh,
                histories.rows[uid], uid, cfg) for uid in selected]
            if any(sha256(Path(path)) != value for path, value in sources.items()):
                raise ValueError("numerical source changed before version block commit")
            write_json(block_path, dict(status="complete", identity_sha256=identity,
                offset=offset, uids=selected, users=values))
        records.extend(values)
        artifacts.append(dict(path=str(block_path.relative_to(output)), sha256=sha256(block_path)))
        progress = dict(status="running", history_edge=args.history_edge, target_version=args.target_version,
            completed_users=len(records), total_users=len(users), reused_users=reused, seconds=time.perf_counter() - started)
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    cm = CostModel(new_target.cfg.hidden_size, len(new_target.blocks), new_target.cfg.num_heads, cfg["attention_backend"])
    denominator = sum((cm if row["history_length"] == 1024 else CostModel(cm.hidden_size, cm.num_layers,
        cm.num_heads, "torch")).full_cache(row["history_length"]) for row in records)
    population_refresh = sum(cached_preparation_flops(refresh, row["history_length"]) for row in records)
    population_previous = sum(cached_preparation_flops(previous, row["history_length"]) for row in records)
    calibration = int(metadata["cumulative_calibration_flops_estimate"])
    if calibration != metadata["source_teacher_and_target_preparation_flops"] + metadata["total_calibration_flops_estimate"]:
        raise ValueError("actual refresh calibration accounts do not sum")
    sunk = int(prior["cumulative_calibration_flops_estimate"])
    costs = dict(once_rebuild_flops=int(denominator), current_refresh_calibration_flops=calibration,
        current_refresh_population_preparation_flops=int(population_refresh),
        current_incremental_refresh_flops=int(calibration + population_refresh),
        current_incremental_percent_of_once_rebuild=100 * (calibration + population_refresh) / denominator if denominator else None,
        prior_sunk_calibration_flops=sunk, prior_sunk_population_preparation_flops=int(population_previous),
        prior_sunk_total_flops=int(sunk + population_previous),
        refresh_calibration_components=dict(preparation=metadata["preparation"],
            predecessor_source_materialization=metadata["predecessor_source_materialization"],
            training_and_validation_flops=int(metadata["total_calibration_flops_estimate"])),
        scope="Current deployment input already exists as C_previous. Prior F costs are sunk and separate; G calibration includes its actual prior-source materialization. Population teacher/source research replay is excluded from deployment incremental cost.")
    summary = dict(status="complete", history_edge=args.history_edge, target_version=args.target_version,
        users=len(records), diagnostic=bool(args.diagnostic), role=role, physical_gpu=1,
        artifacts=artifacts, costs=costs, seconds=time.perf_counter() - started, resumed_users=reused,
        feedback_labels_used=False, quality_auc_computed=False,
        scope="Fixed history and query panel compare each path to its own target teacher; normalized scores do not prove actual task accuracy or gate necessity.")
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", dict(progress, status="complete"))
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--history-edge", type=int, choices=(2, 3), required=True)
    result.add_argument("--target-version", type=int, choices=(3, 4), required=True)
    result.add_argument("--predecessor-checkpoint", type=Path, required=True)
    result.add_argument("--refresh-checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--gpu", type=int, default=1)
    result.add_argument("--users", type=int, default=2000)
    result.add_argument("--diagnostic", action="store_true")
    result.add_argument("--block-users", type=int, default=64)
    result.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        value = run(args)
        write_json(args.output / "exit.json", dict(exit_code=0))
        print(json.dumps(dict(status="complete", users=value["users"], output=str(args.output))), flush=True)
    except BaseException as error:
        write_json(args.output / "exit.json", dict(exit_code=1, error=str(error)))
        raise
