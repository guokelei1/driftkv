#!/usr/bin/env python3
"""Collect fixed-snapshot candidate residuals for cheap CPU-only probes.

F is frozen. Native queries always receive the true candidate. A near-score
panel is selected using ONLY the teacher's scores in a fixed 64-item pool.
No feedback file, label, future event, or final Design module is evaluated.
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
from experiments.ch3_motivation.code.probe_cost import cached_preparation_flops
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.probes import native_query_flops
from hstu_kvcache.adaptation.reader import history_read, score as native_score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.methods.read_correction.calibration.v2.calibrate import mixed_candidates
from hstu_kvcache.common.artifacts import sha256
import numpy as np
import torch

PANELS = ("broad16", "near16")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def panel_indices(teacher_scores):
    """Select the narrowest contiguous sorted-teacher-score window, ties stable."""
    scores = np.asarray(teacher_scores, dtype=np.float64)
    if scores.shape != (64,) or not np.isfinite(scores).all():
        raise ValueError("near16 requires exactly64 finite teacher pool scores")
    order = np.argsort(scores, kind="stable")
    start = int(np.argmin(scores[order[15:]] - scores[order[:-15]]))
    return np.stack((np.r_[0:8, 56:64], order[start:start + 16])).astype(np.int16)


def centered_panels(features, read_residual, teacher_scores, f_scores, indices):
    """All targets retain their actual native candidate; centering is descriptive."""
    x = np.asarray(features, dtype=np.float32)[indices]
    read = np.asarray(read_residual, dtype=np.float64)[indices]  # [P,Q,L,H,D]
    t = np.asarray(teacher_scores, dtype=np.float64)[indices]
    f = np.asarray(f_scores, dtype=np.float64)[indices]
    logit = t - f
    return dict(features=x,
        read_target=(read - read.mean(1, keepdims=True)).astype(np.float32),
        logit_target=(logit - logit.mean(1, keepdims=True)).astype(np.float32),
        read_total_mse=np.mean(read ** 2, axis=(1, 4)),  # [P,L,H]
        logit_total_mse=np.mean(logit ** 2, axis=1),
        teacher_scores=t.astype(np.float32), f_scores=f.astype(np.float32))


@torch.no_grad()
def collect_user(parent, current, probe, projection, events, uid, cfg):
    at = int(cfg["release_day"]) * 86400
    device = next(current.parameters()).device
    source = NativeTrajectory(parent, int(cfg["edge"]) - 1, events, at,
        max_length=int(cfg["history_length"]), append_band_size=int(cfg["append_band_size"]))
    ids = torch.as_tensor(source.retained_events[1], dtype=torch.long, device=device)[None]
    mapped = probe.prepare_cache(current, source.cache, ids)
    teacher = source.fresh_cache(current)
    pool = mixed_candidates(int(uid), source.retained_events[1], current.cfg.num_items - 256,
        count=64, seed=17)
    candidates = torch.as_tensor(pool, dtype=torch.long, device=device)[None]
    gap = 0 if source.last_timestamp is None else at - source.last_timestamp
    delta = torch.tensor([gap], dtype=torch.float32, device=device)
    actual, trace = native_score(current, mapped, candidates, delta, trace=True)
    wanted, _ = native_score(current, teacher, candidates, delta)
    residual = []
    for layer, (q, heads) in enumerate(zip(trace.queries, trace.history_heads, strict=True)):
        target = history_read(current.blocks[layer].attn, q, teacher.k[layer], teacher.v[layer])
        residual.append((target - heads) / max(1, source.cache.seq_len))
    # [L,1,H,Q,D] -> [Q,L,H,D], compact CPU transfer once per user.
    read = torch.stack(residual)[:, 0].permute(2, 0, 1, 3).cpu().numpy()
    features = (current.lookup_item_embeddings(candidates)[0] @ projection).cpu().numpy()
    teacher_scores, f_scores = wanted[0].cpu().numpy(), actual[0].cpu().numpy()
    indices = panel_indices(teacher_scores)
    values = centered_panels(features, read, teacher_scores, f_scores, indices)
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("nonfinite collected candidate feature or target")
    return dict(uid=np.uint64(uid), history_length=np.int32(source.cache.seq_len),
        query_delta=np.int64(gap), pool_ids=np.asarray(pool, dtype=np.int64),
        panel_indices=indices, **values)


def collection_cost(model, probe, lengths, cfg):
    """Actual operation shapes; research preparation, never serving cost."""
    cm = CostModel(model.cfg.hidden_size, len(model.blocks), model.cfg.num_heads, cfg["attention_backend"])
    parts = dict(source_and_teacher_prefill=0, frozen_f_materialization=0,
                 f_and_teacher_pool_queries=0, same_query_teacher_reads=0,
                 item_projection=0, target_subtract_and_length_normalize=0,
                 panel_target_centering=0, panel_target_statistics=0,
                 teacher_window_span_subtractions=0)
    width, layers, heads = model.cfg.hidden_size, len(model.blocks), model.cfg.num_heads
    for n in lengths:
        cache_cm = cm if n == 1024 else CostModel(width, layers, heads, "torch")
        parts["source_and_teacher_prefill"] += 2 * cache_cm.full_cache(int(n))
        parts["frozen_f_materialization"] += cached_preparation_flops(probe, int(n))
        parts["f_and_teacher_pool_queries"] += 2 * native_query_flops(model, history_length=int(n), queries=64, batch=1)
        parts["same_query_teacher_reads"] += 64 * int(n) * layers * (4 * width + 3 * heads)
        parts["item_projection"] += 2 * 64 * width * 16
        parts["target_subtract_and_length_normalize"] += 2 * 64 * layers * width + 64
        # Each of two panels: mean (15 adds + one division),16 subtractions.
        parts["panel_target_centering"] += 2 * 32 * (layers * width + 1)
        parts["panel_target_statistics"] += 2 * 32 * (layers * width + 1)
        parts["teacher_window_span_subtractions"] += 49
    parts = {key: int(value) for key, value in parts.items()}
    return dict(components=parts, total_estimated_flops=sum(parts.values()), users=len(lengths),
        scope="Research collection only: both native prefills, F materialization, both64-query paths, same-q teacher reads and projection/target arithmetic. Sorting/gathers/bytes are not floating-point arithmetic; no deployment or final-Design cost claim.")


def run(args):
    if args.gpu != 1:
        raise ValueError("this study uses physical GPU1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    main_root, fit_root = Path(args.input_root).resolve(), Path(args.fit_input_root).resolve()
    main_pop = json.loads((main_root / "population.json").read_text())
    fit_pop = json.loads((fit_root / "population.json").read_text())
    roles = dict(fit=fit_pop["fit"][:100], validation=fit_pop["validation"], main=main_pop["main"][:args.users])
    if len(roles["main"]) != args.users or args.users < 1 or not 1 <= args.block_users <= 64:
        raise ValueError("fixed main population or block size unavailable")
    quality_role(roles["main"], diagnostic=args.diagnostic)
    if args.diagnostic:
        roles["fit"], roles["validation"] = roles["fit"][:8], roles["validation"][:2]
    if len(set(sum(roles.values(), []))) != sum(map(len, roles.values())):
        raise ValueError("fit, validation and main roles overlap")
    parent, current, models = load_models(args.edge, device)
    probe, metadata = load_variant(Path(args.checkpoint), device)
    identity = metadata["identity"]
    if (probe.stage != "f" or identity["edge"] != args.edge or identity["models"] != models
            or identity["fit_uids"] != fit_pop["fit"][:100]
            or identity["validation_users"] != fit_pop["validation"]):
        raise ValueError("requires this case's actual frozen F100/val16")
    probe.eval().requires_grad_(False)
    inputs = {}
    histories = {}
    for name, root in (("fitting", fit_root), ("main", main_root)):
        binding_path = root / f"edge_{args.edge}/inputs.json"
        binding = json.loads(binding_path.read_text())
        history_path = binding_path.parent / binding["history"]["path"]
        if binding["population_sha256"] != sha256(root / "population.json") or sha256(history_path) != binding["history"]["sha256"]:
            raise ValueError("frozen population/history changed")
        inputs[name] = dict(root=str(root), population_sha256=sha256(root / "population.json"),
            binding_sha256=sha256(binding_path), history_path=str(history_path),
            history_sha256=sha256(history_path), settings=binding["settings"])
        histories[name] = restore_pack(history_path)  # No request/feedback parquet read.
    if (inputs["fitting"]["binding_sha256"] != identity["input_binding_sha256"]
            or inputs["fitting"]["settings"]["release_day"] != inputs["main"]["settings"]["release_day"]):
        raise ValueError("F input history/cutover identity differs")
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ("common.py", "probe_variants.py", "probes.py", "probe_cost.py")],
        ROOT / "src/hstu_kvcache/evaluation/trajectory.py", ROOT / "src/hstu_kvcache/models/state_transition.py",
        ROOT / "src/hstu_kvcache/adaptation/reader.py", ROOT / "src/hstu_kvcache/models/hstu.py",
        ROOT / "src/hstu_kvcache/models/attention.py", ROOT / "src/hstu_kvcache/evaluation/yambda.py",
        ROOT / "src/hstu_kvcache/methods/read_correction/calibration/v2/calibrate.py", ROOT / "src/hstu_kvcache/evaluation/cost.py"]
    sources = {str(p.resolve()): sha256(p) for p in paths}
    rng = np.random.default_rng(17)
    projection_array = (rng.integers(0, 2, size=(current.cfg.hidden_size, 16)) * 2 - 1).astype(np.float32) / np.sqrt(current.cfg.hidden_size)
    projection_array = projection_array.astype(np.float32)
    projection = torch.as_tensor(projection_array, device=device)
    config = dict(format="candidate_residual_probe_v1", edge=args.edge, roles=roles, models=models,
        inputs=inputs, checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256(args.checkpoint),
        sources=sources, physical_gpu=1, diagnostic=bool(args.diagnostic), block_users=args.block_users,
        hardware=torch.cuda.get_device_name(0), panels=list(PANELS), pool_count=64, panel_count=16,
        panel_rules=dict(broad16="pool positions0..7 and56..63, preserving recent-history plus uniform; NOT the historicalQ16 replica",
            near16="smallest teacher-score span over all contiguous sorted16-windows in the SAME64 pool; stable ties"),
        projection=dict(seed=17, kind="fixed Rademacher / sqrt(item_width)", shape=list(projection_array.shape),
            sha256=hashlib.sha256(projection_array.tobytes()).hexdigest()),
        geometry=dict(layers=len(current.blocks), heads=current.cfg.num_heads, head_dim=current.blocks[0].attn.head_dim),
        target="same-F-query teacher history read minus F read, divided by aligned history length; panel-centered",
        scope="Frozen cutover teacher residual predictability only; no future labels, AUC, read intervention or Design-effectiveness claim")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("resume configuration/source/input/checkpoint differs")
    write_json(config_path, config)
    np.save(output / "projection.npy", projection_array)
    configuration_sha = digest(config)
    started, artifacts, costs = time.perf_counter(), [], {}
    for role, uids in roles.items():
        name = "main" if role == "main" else "fitting"
        lengths = []
        for offset in range(0, len(uids), args.block_users):
            selected = uids[offset:offset + args.block_users]
            block_path = output / role / f"part_{offset:06d}.npz"
            manifest_path = block_path.with_suffix(".json")
            if manifest_path.exists():
                m = json.loads(manifest_path.read_text())
                if m["configuration_sha256"] != configuration_sha or m["uids"] != selected or m["npz_sha256"] != sha256(block_path):
                    raise ValueError("committed candidate block identity changed")
                with np.load(block_path, allow_pickle=False) as b:
                    lengths.extend(b["history_length"].tolist())
            else:
                rows = [collect_user(parent, current, probe, projection, histories[name].rows[uid], uid,
                    inputs[name]["settings"]) for uid in selected]
                if any(sha256(Path(p)) != value for p, value in sources.items()):
                    raise ValueError("numerical source changed before candidate block commit")
                arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
                lengths.extend(arrays["history_length"].tolist())
                block_path.parent.mkdir(exist_ok=True)
                with block_path.with_suffix(".npz.writing").open("wb") as handle:
                    np.savez(handle, **arrays)  # Uncompressed: avoid slow compression of floating targets.
                block_path.with_suffix(".npz.writing").replace(block_path)
                write_json(manifest_path, dict(configuration_sha256=configuration_sha, role=role,
                    offset=offset, uids=selected, npz_sha256=sha256(block_path)))
            artifacts.append(dict(path=str(block_path.relative_to(output)), sha256=sha256(block_path),
                manifest=str(manifest_path.relative_to(output)), manifest_sha256=sha256(manifest_path)))
            progress = dict(status="running", edge=args.edge, role=role, completed_users=offset + len(selected),
                total_users=len(uids), seconds=time.perf_counter() - started)
            write_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        costs[role] = collection_cost(current, probe, lengths, inputs[name]["settings"])
    summary = dict(status="complete", edge=args.edge, roles={k:len(v) for k,v in roles.items()},
        diagnostic=bool(args.diagnostic), physical_gpu=1, artifacts=artifacts,
        research_collection_costs=costs, total_research_collection_flops=sum(v["total_estimated_flops"] for v in costs.values()),
        prior_f_sunk_calibration_flops=metadata["cumulative_calibration_flops_estimate"],
        seconds=time.perf_counter()-started, feedback_labels_used=False, quality_auc_computed=False,
        feature_target_bytes=sum((output/x["path"]).stat().st_size for x in artifacts))
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", dict(progress, status="complete"))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edge", type=int, choices=(2,3), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--users", type=int, default=2000)
    p.add_argument("--block-users", type=int, default=64)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--input-root", type=Path, default=ROOT/"experiments/ch3_motivation/inputs")
    p.add_argument("--fit-input-root", type=Path, default=ROOT/"experiments/ch3_motivation/inputs_feedback_pool/fit256")
    args = p.parse_args()
    try:
        result = run(args)
        write_json(args.output/"exit.json", dict(exit_code=0))
        print(json.dumps(dict(status="complete", roles=result["roles"])), flush=True)
    except BaseException as error:
        write_json(args.output/"exit.json", dict(exit_code=1, error=str(error)))
        raise
