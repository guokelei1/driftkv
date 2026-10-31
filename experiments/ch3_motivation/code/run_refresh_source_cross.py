#!/usr/bin/env python3
"""Cross native/actual calibration sources on fixed-history held-out users.

Both maps have the same architecture, fitting population and budget. Every
score is compared with one NEW-target teacher on the same causal history.
The native-old-target input is a diagnostic rebuild, not retained serving state.
"""
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
from experiments.ch3_motivation.code.run_kv_version_refresh import validate_refresh_pair
from hstu_kvcache.evaluation.yambda import load_model
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.methods.read_correction.calibration.v2.calibrate import mixed_candidates
from hstu_kvcache.common.artifacts import sha256
import numpy as np
import torch

METHODS = ("native_unadapted", "actual_unadapted", "train_native_on_native",
           "train_native_on_actual", "train_actual_on_native", "train_actual_on_actual", "teacher")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def validate_native_pair(native, metadata, actual_metadata, *, actual_sha256,
                         previous_sha256, models):
    n, a = metadata["identity"], actual_metadata["identity"]
    if (native.stage != "f" or metadata.get("method_label") != "native_refresh_control"
            or n.get("protocol") != "frozen_native_source_refresh_control_v1"
            or n.get("source_kind") != "native_old_target_same_history"
            or n.get("paired_actual_refresh_checkpoint_sha256") != actual_sha256
            or n.get("paired_predecessor_checkpoint_sha256") != previous_sha256):
        raise ValueError("native control is not paired to this actual-source refresh")
    for key in ("history_edge", "target_version", "history_cutoff", "input_binding_sha256",
                "population_sha256", "fit_uids", "validation_users", "fit_users", "variant",
                "width", "depth", "queries", "seed", "candidate_seed", "candidate_source",
                "recent_fraction", "calibration_candidate_sha256", "steps", "learning_rate",
                "read_weight", "score_weight", "validation_every", "patience", "query_conditioning_mode"):
        if n[key] != a[key]:
            raise ValueError(f"native/actual fitting controls differ in {key}")
    if (n["native_source_version"] != a["predecessor_target_version"]
            or len(n["models"]) != 2
            or any(models.get(path) != value for path, value in n["models"].items())):
        raise ValueError("native control used a different source/target model pair")


@torch.no_grad()
def replay_user(source_model, old_target, new_target, previous, actual_map,
                native_map, events, uid, cfg):
    """Build native N and actual A once, then cross both refresh maps on both."""
    at = int(cfg["release_day"]) * 86400
    device = next(new_target.parameters()).device
    source = NativeTrajectory(source_model, int(cfg["edge"]) - 1, events, at,
        max_length=int(cfg["history_length"]), append_band_size=int(cfg["append_band_size"]))
    ids = torch.as_tensor(source.retained_events[1], dtype=torch.long, device=device)[None]
    actual = previous.prepare_cache(old_target, source.cache, ids)
    native = source.fresh_cache(old_target)
    teacher = source.fresh_cache(new_target)
    caches = (native, actual, native_map.prepare_cache(new_target, native, ids),
        native_map.prepare_cache(new_target, actual, ids),
        actual_map.prepare_cache(new_target, native, ids),
        actual_map.prepare_cache(new_target, actual, ids), teacher)
    candidates = mixed_candidates(int(uid), source.retained_events[1], new_target.cfg.num_items - 256,
                                  count=16, seed=int(cfg.get("seed", 17)))
    panel = torch.as_tensor(candidates, dtype=torch.long, device=device)[None]
    gap = 0 if source.last_timestamp is None else at - source.last_timestamp
    delta = torch.tensor([gap], dtype=torch.float32, device=device)
    values = score_cache_group(new_target, caches, panel, delta).cpu().double().numpy()
    if not np.isfinite(values).all():
        raise ValueError("nonfinite crossed source-state scores")
    scores = dict(zip(METHODS, values.tolist(), strict=True))
    target = values[-1]
    mse = {method: float(np.mean((value-target)**2)) for method, value in zip(METHODS, values, strict=True)}
    return dict(uid=int(uid), history_length=int(source.cache.seq_len), history_cutoff=at,
        query_delta=int(gap), panel_candidates=[int(x) for x in candidates],
        target_version=int(cfg["edge"]) + 1, scores=scores, score_mse=mse)


def run(args):
    if args.gpu != 1 or args.target_version != args.history_edge + 1:
        raise ValueError("this two-case source study uses GPU1 and the next target version")
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
    if (binding["population_sha256"] != sha256(population_path)
            or sha256(history_path) != binding["history"]["sha256"]):
        raise ValueError("frozen evaluation role/history binding changed")
    histories = restore_pack(history_path)
    cfg = binding["settings"]
    source, old_target, models = load_models(args.history_edge, device)
    target_path = ROOT / f"results/unified_training_2026_09/large/seed17/checkpoints/v{args.target_version}/checkpoint_100.pt"
    new_target, payload = load_model(target_path, device)
    del payload
    new_target.eval().requires_grad_(False)
    models[str(target_path.relative_to(ROOT))] = sha256(target_path)
    geometry = (old_target.cfg.hidden_size, len(old_target.blocks), old_target.cfg.num_heads, old_target.cfg.num_items)
    if any((m.cfg.hidden_size, len(m.blocks), m.cfg.num_heads, m.cfg.num_items) != geometry
           for m in (source, new_target)):
        raise ValueError("cross-source study needs stable model and item geometry")
    paths = {name: Path(value).resolve() for name, value in dict(previous=args.predecessor_checkpoint,
        actual=args.actual_refresh_checkpoint, native=args.native_refresh_checkpoint).items()}
    previous, prior = load_variant(paths["previous"], device)
    actual_map, actual_metadata = load_variant(paths["actual"], device)
    native_map, native_metadata = load_variant(paths["native"], device)
    for probe in (previous, actual_map, native_map):
        probe.eval().requires_grad_(False)
    validate_refresh_pair(previous, prior, actual_map, actual_metadata,
        history_edge=args.history_edge, target_version=args.target_version,
        models=models, uids=users, predecessor_sha256=sha256(paths["previous"]),
        history_cutoff=int(cfg["release_day"])*86400)
    validate_native_pair(native_map, native_metadata, actual_metadata,
        actual_sha256=sha256(paths["actual"]), previous_sha256=sha256(paths["previous"]), models=models)
    sources_to_bind = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ("common.py", "kv_score_batch.py", "run_kv_version_refresh.py", "probe_variants.py", "probes.py", "probe_cost.py")],
        ROOT / "src/hstu_kvcache/evaluation/trajectory.py", ROOT / "src/hstu_kvcache/models/state_transition.py",
        ROOT / "src/hstu_kvcache/adaptation/reader.py", ROOT / "src/hstu_kvcache/models/hstu.py",
        ROOT / "src/hstu_kvcache/models/attention.py", ROOT / "src/hstu_kvcache/evaluation/yambda.py",
        ROOT / "src/hstu_kvcache/methods/read_correction/calibration/v2/calibrate.py", ROOT / "src/hstu_kvcache/evaluation/cost.py"]
    sources = {str(path.resolve()): sha256(path) for path in sources_to_bind}
    configuration = dict(format="crossed_refresh_source_population_v1", history_edge=args.history_edge,
        target_version=args.target_version, users=users, main_uid_sha256=digest(users),
        population_sha256=sha256(population_path), input_binding_sha256=sha256(binding_path),
        input_root=str(root), history_pack=dict(path=str(history_path), sha256=sha256(history_path)),
        settings=cfg, models=models, checkpoints={name: dict(path=str(path), sha256=sha256(path))
        for name, path in paths.items()}, fitting_identities=dict(actual=actual_metadata["identity"],
            native=native_metadata["identity"]), sources=sources, physical_gpu=1,
        hardware=torch.cuda.get_device_name(0), diagnostic=bool(args.diagnostic), role=role,
        block_users=args.block_users, methods=list(METHODS),
        scope="Fixed history, candidates, time and one new-target teacher; no feedback or continuous serving.",
        normalization_scope="Each fitter computes fit-only scales on its own source pool using the same algorithm.",
        native_input_scope="Native old-target rebuild is a diagnostic input, not a free deployment cache.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("cross-source configuration/input/weight/source identity changed")
    write_json(config_path, configuration)
    identity = digest(configuration)
    records, artifacts = [], []
    reused = 0
    started = time.perf_counter()
    for offset in range(0, len(users), args.block_users):
        selected = users[offset:offset+args.block_users]
        block_path = output / f"blocks/part_{offset:06d}.json"
        if block_path.exists():
            block = json.loads(block_path.read_text())
            if (block["status"] != "complete" or block["identity_sha256"] != identity
                    or block["uids"] != selected):
                raise ValueError("committed cross-source block binding changed")
            values = block["users"]
            reused += len(values)
        else:
            values = [replay_user(source, old_target, new_target, previous, actual_map, native_map,
                histories.rows[uid], uid, cfg) for uid in selected]
            if any(sha256(Path(path)) != digest for path, digest in sources.items()):
                raise ValueError("cross-source numerical code changed before block commit")
            write_json(block_path, dict(status="complete", identity_sha256=identity,
                offset=offset, uids=selected, users=values))
        records.extend(values)
        artifacts.append(dict(path=str(block_path.relative_to(output)), sha256=sha256(block_path)))
        progress = dict(status="running", history_edge=args.history_edge, target_version=args.target_version,
            completed_users=len(records), total_users=len(users), reused_users=reused,
            seconds=time.perf_counter()-started)
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    cm = CostModel(new_target.cfg.hidden_size, len(new_target.blocks), new_target.cfg.num_heads, cfg["attention_backend"])
    rebuild = sum((cm if row["history_length"] == 1024 else CostModel(cm.hidden_size, cm.num_layers,
        cm.num_heads, "torch")).full_cache(row["history_length"]) for row in records)
    costs = dict(once_new_target_rebuild_flops=int(rebuild), controls={})
    for name, probe, metadata in (("train_actual", actual_map, actual_metadata),
                                  ("train_native", native_map, native_metadata)):
        calibration = int(metadata["cumulative_calibration_flops_estimate"])
        if calibration != (metadata["source_teacher_and_target_preparation_flops"]
                           + metadata["total_calibration_flops_estimate"]):
            raise ValueError("control calibration ledger does not sum")
        population_map = sum(cached_preparation_flops(probe, row["history_length"]) for row in records)
        costs["controls"][name] = dict(calibration_flops=calibration,
            population_mapping_flops=int(population_map), total_flops=int(calibration+population_map),
            percent_once_new_target_rebuild=100*(calibration+population_map)/rebuild if rebuild else None)
    costs.update(prior_sunk_calibration_flops=int(prior["cumulative_calibration_flops_estimate"]),
        prior_sunk_population_mapping_flops=int(sum(cached_preparation_flops(previous,
            row["history_length"]) for row in records)),
        diagnostic_native_input_reconstruction_flops=int(rebuild),
        scope="Each control separately includes its own calibration and one population mapping, not all four research cells. A inputs already exist; native N reconstruction and research teachers are not free deployment inputs. Prior F work is sunk. Common reads are excluded; no full230-read serving-cost claim.")
    summary = dict(status="complete", format=configuration["format"], history_edge=args.history_edge,
        target_version=args.target_version, users=len(records), diagnostic=bool(args.diagnostic),
        role=role, physical_gpu=1, artifacts=artifacts, costs=costs, resumed_users=reused,
        seconds=time.perf_counter()-started, feedback_labels_used=False, quality_auc_computed=False,
        score_mse_means={method: float(np.mean([row["score_mse"][method] for row in records])) for method in METHODS},
        scope=configuration["scope"])
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", dict(progress, status="complete"))
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--history-edge", type=int, choices=(2, 3), required=True)
    result.add_argument("--target-version", type=int, choices=(3, 4), required=True)
    result.add_argument("--predecessor-checkpoint", type=Path, required=True)
    result.add_argument("--actual-refresh-checkpoint", type=Path, required=True)
    result.add_argument("--native-refresh-checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs")
    result.add_argument("--gpu", type=int, default=1)
    result.add_argument("--users", type=int, default=2000)
    result.add_argument("--block-users", type=int, default=64)
    result.add_argument("--diagnostic", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        summary = run(args)
        write_json(args.output / "exit.json", dict(exit_code=0))
        print(json.dumps(dict(status="complete", users=summary["users"], output=str(args.output))), flush=True)
    except BaseException as error:
        write_json(args.output / "exit.json", dict(exit_code=1, error=str(error)))
        raise
