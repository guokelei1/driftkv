#!/usr/bin/env python3
"""Full-rank affine KV fit on fixed pre-release Design 1 scenes.

Only the fixed fitting users' aligned valid tokens supervise the ridge.
The original sixteen queries per user are reserved for read/logit diagnostics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from math import ceil
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src")]

import torch

from hstu_kvcache.design_one.affine import AffineKVReadViewAdapter
from design_one import calibrate_nonlinear as shared
from design_one.rolling_calibration import build_rolling_scenes

RIDGE = .001


def token_batches(rows, scene_ids, layer, batch_size, device):
    """Equal-length batches contain only aligned real rows, without padding."""
    for selected in shared.groups(scene_ids, rows, batch_size):
        source, target, context = [], [], []
        for key in selected:
            row = rows[key]
            old, full = row["parent"], row["teacher"]
            if old.seq_len != full.seq_len or row["context"].shape[1] != old.seq_len:
                raise ValueError("source, teacher and immutable context token positions differ")
            source.append(torch.cat((old.k[layer, 0, :old.seq_len], old.v[layer, 0, :old.seq_len]), -1))
            target.append(torch.cat((full.k[layer, 0, :full.seq_len], full.v[layer, 0, :full.seq_len]), -1))
            context.append(row["context"][0, :old.seq_len])
        native = torch.cat(source).to(device=device, dtype=torch.float64)
        full = torch.cat(target).to(device=device, dtype=torch.float64)
        features = torch.cat((native, torch.cat(context).to(device=device, dtype=torch.float64)), -1)
        yield features, full-native


@torch.no_grad()
def fit_layer(module, rows, scene_ids, layer, *, batch_size, device, costs):
    """Two FP64 streaming passes, average-token ridge and an unpenalized bias."""
    dimensions, outputs = module.map_weight.shape
    sx = torch.zeros(dimensions, device=device, dtype=torch.float64)
    sy = torch.zeros(outputs, device=device, dtype=torch.float64)
    xx, yy = torch.zeros_like(sx), torch.zeros_like(sy)
    count = 0
    for x, y in token_batches(rows, scene_ids, layer, batch_size, device):
        count += len(x)
        sx += x.sum(0)
        xx += x.square().sum(0)
        sy += y.sum(0)
        yy += y.square().sum(0)
        costs["teacher_kv_residual_flops"] += y.numel()
        costs["token_statistics_flops"] += 3*(x.numel()+y.numel())+2*(dimensions+outputs)
    center, bias = sx/count, sy/count
    variance_x = (xx/count-center.square()).clamp_min(0)
    variance_y = (yy/count-bias.square()).clamp_min(0)
    scale = variance_x.sqrt().clamp_min(1e-4)
    target_scale = variance_y.sqrt().clamp_min(1e-6)
    costs["token_statistics_flops"] += 6*(dimensions+outputs)
    gram = torch.zeros(dimensions, dimensions, device=device, dtype=torch.float64)
    cross = torch.zeros(dimensions, outputs, device=device, dtype=torch.float64)
    for x, y in token_batches(rows, scene_ids, layer, batch_size, device):
        n = len(x)
        x, y = (x-center)/scale, (y-bias)/target_scale
        gram.addmm_(x.T, x)
        cross.addmm_(x.T, y)
        costs["teacher_kv_residual_flops"] += y.numel()
        costs["ridge_normalization_flops"] += 2*(x.numel()+y.numel())
        costs["ridge_gram_cross_flops"] += 2*n*dimensions*(dimensions+outputs)+dimensions*(dimensions+outputs)
    gram /= count
    cross /= count
    system = gram.clone()
    system.diagonal().add_(RIDGE)
    # Keep the CPU solve isolated from the retained multithreaded LU issue.
    previous_threads = torch.get_num_threads()
    if device.type == "cpu":
        torch.set_num_threads(1)
    try:
        solution = torch.linalg.solve(system, cross)
    finally:
        if device.type == "cpu":
            torch.set_num_threads(previous_threads)
    checked = system@solution
    torch.testing.assert_close(checked, cross, atol=1e-8, rtol=1e-7)
    physical_weight = solution*target_scale[None]
    module.set_parameters(center, scale, physical_weight, bias)
    # Full fitting-token residual MSE from the accumulated sufficient statistics.
    error = variance_y+(solution*(gram@solution-2*cross)).sum(0)*target_scale.square()
    fit_error = float(error.clamp_min(0).mean())
    baseline = float((yy/count).mean())
    costs["ridge_solve_flops_estimate"] += ceil((2/3)*dimensions**3+2*dimensions**2*outputs)
    costs["ridge_diagnostic_flops_estimate"] += 4*dimensions**2*outputs+12*dimensions*outputs+8*outputs
    costs["ridge_publication_flops"] += dimensions*(dimensions+outputs)+dimensions+dimensions*outputs
    return dict(layer=layer, fitting_users=len({rows[key]["uid"] for key in scene_ids}),
        fitting_scenes=len(scene_ids), valid_tokens=count, input_width=dimensions, output_width=outputs,
        ridge=RIDGE, intercept_penalized=False, fit_solver_kv_mse=fit_error,
        uncorrected_kv_mse=baseline, relative_solver_kv_mse=fit_error/max(baseline, 1e-30),
        normal_relative_residual=float((checked-cross).abs().max()/cross.abs().max().clamp_min(1e-30)),
        target_scale_min=float(target_scale.min()), target_scale_max=float(target_scale.max()),
        parameter_count=dimensions*outputs+outputs)


def fit(current, rows, train, validation, *, batch_size, device, costs, timings):
    if {rows[key]["uid"] for key in train} & {rows[key]["uid"] for key in validation}:
        raise ValueError("fitting and validation scenes overlap by UID")
    current.eval().requires_grad_(False)
    adapter = AffineKVReadViewAdapter(len(current.blocks), current.cfg.num_heads,
        current.cfg.hidden_size//current.cfg.num_heads, current.cfg.max_seq_len).to(device)
    records = []
    for layer, module in enumerate(adapter.layers):
        record = shared.timed(lambda: fit_layer(module, rows, train, layer,
            batch_size=batch_size, device=device, costs=costs), timings, "streaming_ridge_fit", device)
        records.append(record)
        print(json.dumps({"status": "affine_layer_fitted", **record}), flush=True)
    # No query has participated in the fit. Both diagnostics now use the
    # actual fully corrected branch queries, including all corrected prefixes.
    diagnostics = {"layers": records}
    for name, keys in (("fitting_reads", train), ("validation", validation)):
        diagnostics[name] = shared.timed(lambda: shared.validate(current, adapter, rows, keys,
            "kv_context", batch_size=batch_size, device=device, costs=costs), timings, name, device)
    return adapter, diagnostics


def capture_inputs(args, device, costs, timings):
    """Reuse fixed reservations, the existing paired capture and four-scene builder."""
    config, stage, versions, train, validation = shared.inputs(args.config, 4 if args.small else None,
                                                              2 if args.small else None)
    parent, previous = shared.timed(lambda: shared.load_model(ROOT/versions["v4"]["checkpoint"], device), timings, "model_load", device)
    current, payload = shared.timed(lambda: shared.load_model(ROOT/versions["v5"]["checkpoint"], device), timings, "model_load", device)
    model_config = payload["config"]
    expected = dict(num_layers=6, hidden_size=192, num_heads=6, max_seq_len=1024,
        activation="elu_plus1", block_variant="legacy", relative_position_bias=False,
        gating="silu_gate", causal_diagonal="inclusive")
    if previous["config"] != model_config or any(model_config[key] != value for key, value in expected.items()):
        raise ValueError("expected frozen six-layer Medium endpoints")
    parent.eval().requires_grad_(False)
    current.eval().requires_grad_(False)
    dataset_path = ROOT/config["data"]["dataset"]["path"]
    known = int(payload.get("known_vocab_size", json.loads(dataset_path.read_text())["foundation_items"]))
    del previous, payload
    uids, cutover, maximum = train+validation, stage["days_half_open"][0]*shared.DAY, model_config["max_seq_len"]

    def histories(limit):
        raw = shared.load_histories(uids, dataset_path=dataset_path, known_vocab_size=known,
            oov_buckets=model_config["num_items"]-known, start_timestamp=cutover, end_timestamp=cutover+1,
            max_history=limit, threads=args.history_threads)
        return {uid: shared.snapshot(raw, uid, cutover, limit) for uid in uids}

    bounded = shared.timed(lambda: histories(maximum), timings, "history_io", device)
    rows = shared.timed(lambda: shared.capture(parent, current, bounded, uids, cutover=cutover,
        known=known, queries=16, device=device, batch_size=args.batch_size,
        history_length=maximum, attention_backend=args.attention_backend), timings, "paired_cache_capture", device)
    for uid in uids:
        rows[uid]["candidates"] = torch.from_numpy(shared.mixed_candidates(uid, bounded[uid][1], known, 16, seed=17))
    models = {name: shared.CostModel.for_scale("medium", name) for name in ("torch", "triton")}
    capture_cost = sum(models[args.attention_backend if rows[batch[0]]["parent"].seq_len == maximum else "torch"].full_cache(
        rows[batch[0]]["parent"].seq_len, batch=len(batch)) for batch in shared.groups(uids, rows, args.batch_size))
    costs["parent_cache_capture_flops"] = costs["teacher_cache_capture_flops"] = capture_cost
    expanded = shared.timed(lambda: histories(maximum+768), timings, "expanded_history_io", device)
    rows, records = shared.timed(lambda: build_rolling_scenes(parent, current, rows, bounded, expanded,
        uids, cutover=cutover, batch_size=args.batch_size, device=device, max_length=maximum,
        attention_backend=args.attention_backend, costs=costs), timings, "rolling_scene_capture", device)
    metadata = dict(configuration=dict(path=str(args.config), sha256=shared.sha256_file(args.config)),
        users_file_sha256=config["users"]["sha256"], uids=train, fit_uids=train, validation_uids=validation,
        users=len(train), budget=len(train), queries_per_user=16, cutover=cutover, history_length=maximum,
        latest_history_timestamp=max(int(bounded[uid][0][-1]) for uid in uids),
        evaluation_users_excluded=config["users"]["groups"][stage["evaluation_group"]]["count"],
        model_config=model_config, checkpoint_hashes={key: versions[key]["checkpoint_sha256"] for key in ("v4", "v5")},
        checkpoint_verification="sealed manifest/seal bindings; weight payload hashes inherited",
        scene_protocol=dict(append_targets=[0, 128, 384, 768], queries_per_scene=4,
            total_queries_per_uid=16, pre_release_history_limit=maximum+768, scenes=records))
    del parent, bounded, expanded
    gc.collect()
    return current, rows, train, validation, metadata


def run(args):
    began = time.perf_counter()
    if (args.output/"calibration.json").exists() or (args.output/"calibration.pt").exists():
        raise FileExistsError("choose a new calibration output directory")
    torch.set_num_threads(args.threads)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.70, device)
        torch.cuda.reset_peak_memory_stats(device)
    costs, timings = defaultdict(int), defaultdict(float)
    current, rows, train, validation, metadata = capture_inputs(args, device, costs, timings)
    train_users, validation_users = set(train), set(validation)
    train_scenes = [key for key, row in rows.items() if row["uid"] in train_users]
    validation_scenes = [key for key, row in rows.items() if row["uid"] in validation_users]
    adapter, diagnostics = fit(current, rows, train_scenes, validation_scenes,
        batch_size=args.batch_size, device=device, costs=costs, timings=timings)
    sources = shared.source_hashes()
    for path in (Path(__file__).resolve(), Path(sys.modules[AffineKVReadViewAdapter.__module__].__file__).resolve(),
                 ROOT/"scripts/design_one/calibrate_nonlinear.py", ROOT/"scripts/design_one/rolling_calibration.py"):
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        sources[name] = shared.sha256_file(path)
    caches = {id(row[key]): row[key] for row in rows.values() for key in ("parent", "teacher")}
    metadata.update(status="complete", kind=adapter.kind, method="affine_kv_view", scale="medium", edge="v4_to_v5",
        role="development_canary" if args.small else "development_calibration",
        adapter_config=adapter.get_config(), input_width=2*current.cfg.hidden_size+2,
        context_features=["is_parent", "old_fraction_at_write"], fit_scene_ids=train_scenes,
        validation_scene_ids=validation_scenes, execution_sources=sources, diagnostics=diagnostics,
        fitting="aligned teacherKV-sourceKV; uniform valid-token average; centered standardized X/Y; ridge1e-3 on slopes, unpenalized residual-mean intercept; output scales folded into physical weights",
        normalization="fit-user scene tokens only, population std; input std floor1e-4, residual target std floor1e-6; no padding tokens",
        candidate_rule="fixed mixed_recent_uniform sixteen per UID split state_index::4; all queries used only for post-fit diagnostics",
        teacher="Current Full KV on each scene's identical actual terminal history; no teacher query enters KV fitting",
        validation_rule="held-out UIDs, diagnostics after the one closed-form fit; no model selection",
        cache_state="same four pure/rolling scenes as round8, actual native writes and evictions, immutable row birth context",
        settings=dict(ridge=RIDGE, batch_size=args.batch_size, solver_dtype="float64", published_dtype="float32",
            attention_backend=args.attention_backend, torch_threads=args.threads, history_threads=args.history_threads,
            device=str(device), gpu_memory_fraction=.70, optimizer="none"),
        cost={**dict(costs), "calibration_flops": sum(costs.values()),
            "teacher_flops": costs["teacher_cache_capture_flops"]+costs["rolling_teacher_cache_flops"]
                +costs["teacher_same_query_flops"]+costs["teacher_full_query_flops"],
            "scope": "all Parent/teacher captures, native replay/context, streaming token statistics and Gram/cross/solve, publication, complete fitting/held-out query diagnostics",
            "convention": "multiply-add=2; actual valid-token matrix shapes; LU and metric arithmetic estimated; FP64 has no extra multiplier; no SGD/backward"},
        timings_seconds=dict(timings), elapsed_seconds=time.perf_counter()-began,
        paired_cache_bytes=sum(cache.k.nbytes+cache.v.nbytes for cache in caches.values()),
        cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1<<20),
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/(1<<30) if device.type == "cuda" else 0)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind=adapter.kind, adapter=adapter.export_state(), metadata=metadata), args.output/"calibration.pt")
    metadata["weights_sha256"] = shared.sha256_file(args.output/"calibration.pt")
    (args.output/"calibration.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(dict(status="calibration_complete", output=str(args.output),
        calibration_flops=metadata["cost"]["calibration_flops"], elapsed_seconds=metadata["elapsed_seconds"])), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=shared.DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(4))
    parser.add_argument("--small", action="store_true", help="fixed first4 fitting/2 validation UIDs, same four scenes")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--history-threads", type=int, default=8)
    parser.add_argument("--attention-backend", choices=("torch", "triton"), default="triton")
    run(parser.parse_args())

