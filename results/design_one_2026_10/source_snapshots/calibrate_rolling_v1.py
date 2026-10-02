#!/usr/bin/env python3
"""Fit Design1 shared read correction on frozen, pre-release Medium users.

PCA and ridge see fitting UIDs only. Each layer's targets use the queries
produced by the already corrected lower layers. Validation never selects
weights; it reports same-query response error after the complete fit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from math import ceil
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import torch

from design.shared_read_probe import fit_layer
from evaluate_yambda500m_foundation_raw import load_histories, load_model, sha256_file
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.design_one import ProducerSummary, SharedReadAdapter, SummaryProjection
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band, truncate_cache
from read_correction_2026_09.calibrate import capture, groups, stacked
from read_correction_2026_09.cost import CostModel, eager_read, teacher_history_read
from read_correction_2026_09.v2.calibrate import mixed_candidates
from read_correction_v5.history_conditioned.capture import prepare_timeline
from selective_recompute_2026_09.calibrate import snapshot
from selective_recompute_2026_09.evaluate import native_backend

DEFAULT_CONFIG = ROOT / "configs/design/medium_v0_v5_development.json"
PRODUCERS = (4, 5)
DAY = 86400


def checked_json(record):
    path = ROOT / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"frozen input changed: {path}")
    return json.loads(path.read_text())


def inputs(path, fit_users=None, validation_users=None):
    """Bind the requested prefix of the frozen reservation, including canaries."""
    config = json.loads(path.read_text())
    stage = config["stages"]["design1"]
    if stage["edges"] != ["v4_to_v5"]:
        raise ValueError("this first prototype implements frozen Medium V4 -> V5")
    users = checked_json(config["users"])
    train = users[stage["fit_group"]]
    validation = users[stage["validation_group"]]
    for value, group in ((fit_users, train), (validation_users, validation)):
        if value is not None and not 1 <= value <= len(group):
            raise ValueError("canary counts must select a nonempty frozen UID prefix")
    train = train[:fit_users]
    validation = validation[:validation_users]
    evaluation = set(users[stage["evaluation_group"]])
    if set(train) & (set(validation) | evaluation) or set(validation) & evaluation:
        raise RuntimeError("fitting, validation and evaluation users overlap")
    if len(train) < 2:
        raise ValueError("PCA calibration requires at least two fitting users")
    manifest = checked_json(config["model"]["manifest"])
    versions = {entry["version"]: entry for entry in config["model"]["versions"]}
    admitted = {entry["version"]: entry for entry in manifest["versions"]}
    for name in ("v4", "v5"):
        record = versions[name]
        seal = checked_json(record["seal"])
        checked_json(record["full_only_admission"])
        if not record["full_only_gates_pass"] or any(
            record[key] != admitted[name][key] for key in ("checkpoint", "checkpoint_sha256")
        ) or seal["checkpoint_sha256"] != record["checkpoint_sha256"]:
            raise RuntimeError(f"frozen checkpoint binding differs: {name}")
    checked_json(config["data"]["dataset"])
    return config, stage, versions, train, validation


def source_hashes():
    paths = [Path(__file__), ROOT / "scripts/design/shared_read_probe.py",
             ROOT / "scripts/design/diagnose_native_input.py",
             ROOT / "scripts/evaluate_yambda500m_foundation_raw.py",
             ROOT / "scripts/read_correction_2026_09/calibrate.py",
             ROOT / "scripts/read_correction_2026_09/v2/calibrate.py",
             ROOT / "scripts/read_correction_2026_09/cost.py",
             ROOT / "scripts/read_correction_v5/history_conditioned/capture.py",
             ROOT / "scripts/selective_recompute_2026_09/calibrate.py",
             ROOT / "scripts/selective_recompute_2026_09/cost.py",
             ROOT / "scripts/selective_recompute_2026_09/evaluate.py",
             ROOT / "src/hstu_kvcache/adaptation/reader.py",
             ROOT / "src/hstu_kvcache/data/yambda_history.py",
             ROOT / "src/hstu_kvcache/data/oov.py",
             ROOT / "src/hstu_kvcache/training/foundation.py",
             *sorted((ROOT / "src/hstu_kvcache/models").glob("*.py")),
             *sorted((ROOT / "src/hstu_kvcache/design_one").glob("*.py"))]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def timed(function, timings, name, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings[name] += time.perf_counter() - start
    return result


def ridge_flops(users, queries, heads, dim, latent, read_dim):
    """Executed factorized normal equations; LU and statistics are estimates."""
    s, q, h, d, r, a = users, queries, heads, dim, latent, read_dim
    j, p = d + 1, latent * (dim + 1) + read_dim
    gram = (2*s*h*q*j*j + s*r*r + 2*r*r*s*h*j*j
            + 2*s*h*q*j*d + 2*s*r*h*j*d
            + 2*s*h*q*j*a + 2*s*r*h*j*a
            + 2*s*q*a*a + 2*s*q*a*h*d)
    solve = ceil(h * ((2/3)*p**3 + 2*p*p*d))
    checks_and_prediction = 4*h*p*p*d + 2*s*h*q*r*j*d + 2*s*q*a*h*d
    statistics = 12*s*q*(h*d+a) + 6*s*r + 3*h*p*p + 8*s*h*q*d
    return int(gram + solve + checks_and_prediction + statistics)


def projection_flops(users, features, rank):
    """Gram PCA estimate, including standardization, eigensolve and whitening."""
    return int(8*users*features + 2*users*users*features + 9*users**3
               + 2*features*users*rank + 2*users*features*rank)


def midpoint_boundary(timestamps):
    """Nearest interior timestamp-group boundary; ties choose the earlier one."""
    boundaries = [index for index in range(1, len(timestamps))
                  if int(timestamps[index-1]) < int(timestamps[index])]
    if not boundaries:
        raise ValueError("mixed calibration needs two distinct pre-release timestamp groups")
    return min(boundaries, key=lambda index: (abs(2*index-len(timestamps)), index))


def terminal_tuple_hash(events):
    """Chronological (timestamp, mapped item, behavior) rows, little-endian int64."""
    return hashlib.sha256(np.column_stack(events).astype("<i8").tobytes()).hexdigest()


@torch.no_grad()
def mixed_scenes(current, rows, histories, uids, *, cutover, batch_size, device,
                 max_length, costs):
    """Two real states per UID, sixteen total queries and one shared Full teacher.

    The causal prefix of Parent Full is already its exact prefix cache, so no
    second Parent or teacher materialization is needed. Current appends the
    remaining observed events through the native bounded reader. The initial
    and terminal windows share the same first event; this first mixed probe
    covers producer mixing, not eviction or a multi-release lifetime.
    """
    buckets, scenes, records = defaultdict(list), {}, []
    for uid in uids:
        times = histories[uid][0]
        if int(times[-1]) >= cutover:
            raise ValueError("mixed scenes must use strictly pre-release history")
        split = midpoint_boundary(times)
        if len(rows[uid]["candidates"]) != 16:
            raise ValueError("mixed scenes require the fixed sixteen-query user budget")
        buckets[(len(times), split)].append(uid)
        scenes[f"{uid}:pure"] = {**rows[uid], "uid": uid, "scene_kind": "pure",
                                 "candidates": rows[uid]["candidates"][::2]}
        records.append(dict(scene_id=f"{uid}:pure", uid=uid, kind="pure", queries=8,
                            inherited_count=len(times), native_writes=0))
    cost_model = CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads, "torch")
    for (length, split), bucket in sorted(buckets.items()):
        for start in range(0, len(bucket), batch_size):
            selected = bucket[start:start+batch_size]
            parent_prefix = HSTUKVCache(
                torch.cat([truncate_cache(rows[uid]["parent"], split).k for uid in selected], 1).to(device),
                torch.cat([truncate_cache(rows[uid]["parent"], split).v for uid in selected], 1).to(device), split)
            items = torch.tensor([histories[uid][1][split:].tolist() for uid in selected],
                                 dtype=torch.long, device=device)
            behaviors = torch.tensor([histories[uid][2][split:].tolist() for uid in selected],
                                     dtype=torch.long, device=device)
            deltas = torch.tensor([(histories[uid][0][split:]-histories[uid][0][split-1:-1]).tolist()
                                   for uid in selected], dtype=torch.float32, device=device).clamp(0, 7*DAY)
            # Torch avoids a separate Triton specialization per partial prefix.
            with native_backend([current], "torch"):
                cache = append_with_rolling_band(current, parent_prefix, items, behaviors,
                                                 deltas, max_length)
            appended = length-split
            costs["mixed_native_append_flops"] += cost_model.band_append(
                split, appended, batch=len(selected), window_size=max_length)
            for index, uid in enumerate(selected):
                base = rows[uid]
                scenes[f"{uid}:mixed"] = {
                    **base, "uid": uid, "scene_kind": "mixed",
                    "parent": HSTUKVCache(cache.k[:, index:index+1].cpu(),
                                          cache.v[:, index:index+1].cpu(), length),
                    "candidates": base["candidates"][1::2],
                    "producer": [4]*split+[5]*appended, "native_writes": appended,
                }
                records.append(dict(scene_id=f"{uid}:mixed", uid=uid, kind="mixed", queries=8,
                                    inherited_count=split, native_writes=appended,
                                    parent_last_timestamp=int(histories[uid][0][split-1]),
                                    first_append_timestamp=int(histories[uid][0][split]),
                                    terminal_last_timestamp=int(histories[uid][0][-1])))
    costs["mixed_extra_parent_capture_flops"] = 0
    costs["mixed_extra_teacher_capture_flops"] = 0
    return scenes, records


@torch.no_grad()
def rolling_scenes(parent, current, rows, histories, expanded_histories, uids, *,
                   cutover, batch_size, device, max_length, attention_backend, costs):
    """Parent full window, real Current writes and evictions, one terminal teacher.

    The expanded pre-release history supplies older Parent-row dependencies.
    Pure scenes keep their original window. A rolling window with different
    oldest-timestamp boundary rows receives its own aligned Full teacher.
    """
    timelines, buckets, scenes, records = {}, defaultdict(list), {}, []
    costs.setdefault("mixed_extra_teacher_capture_flops", 0)
    for uid in uids:
        terminal = histories[uid]
        timeline = prepare_timeline(expanded_histories[uid], cutover=cutover,
            history_length=max_length, append_target=len(terminal[0])//2)
        timeline["reuse_teacher"] = all(np.array_equal(left, right) for left, right in zip(
            timeline["terminal"], terminal, strict=True))
        if len(rows[uid]["candidates"]) != 16:
            raise ValueError("rolling scenes require the fixed sixteen-query user budget")
        timelines[uid] = timeline
        buckets[(len(timeline["prefix"][0]), timeline["actual_append"], len(timeline["terminal"][0]))].append(uid)
        scenes[f"{uid}:pure"] = {**rows[uid], "uid": uid, "scene_kind": "pure",
                                 "candidates": rows[uid]["candidates"][::2]}
        records.append(dict(scene_id=f"{uid}:pure", uid=uid, kind="pure", queries=8,
                            inherited_count=len(terminal[0]), native_writes=0, rolling_evictions=0,
                            terminal_tuple_sha256=terminal_tuple_hash(terminal)))
    models = {backend: CostModel(current.cfg.hidden_size, len(current.blocks),
                                 current.cfg.num_heads, backend) for backend in ("torch", "triton")}
    for (prefix_length, suffix_length, terminal_length), bucket in sorted(buckets.items()):
        backend = attention_backend if prefix_length == max_length else "torch"
        cost_model = models[backend]
        for start in range(0, len(bucket), batch_size):
            selected = bucket[start:start+batch_size]
            times, items, behaviors = [torch.as_tensor(np.stack([
                timelines[uid]["prefix"][column] for uid in selected]), device=device)
                for column in range(3)]
            deltas = torch.zeros_like(times, dtype=torch.float32)
            deltas[:, 1:] = times[:, 1:]-times[:, :-1]
            with native_backend([parent, current], backend):
                cache = parent.compute_kv(items.long(), behaviors.long(), deltas)
                costs["mixed_extra_parent_capture_flops"] += cost_model.full_cache(
                    prefix_length, batch=len(selected))
                if suffix_length:
                    new_times, new_items, new_behaviors = [torch.as_tensor(np.stack([
                        timelines[uid]["suffix"][column] for uid in selected]), device=device)
                        for column in range(3)]
                    previous_times = torch.cat((times[:, -1:], new_times[:, :-1]), dim=1)
                    new_deltas = (new_times-previous_times).clamp(0, 7*DAY).float()
                    offset = 0
                    while offset < suffix_length:
                        available = min(32, suffix_length-offset)
                        width = 1 << (available.bit_length()-1)
                        end = offset+width
                        before = cache.seq_len
                        cache = append_with_rolling_band(current, cache, new_items[:, offset:end].long(),
                            new_behaviors[:, offset:end].long(), new_deltas[:, offset:end], max_length)
                        costs["mixed_native_append_flops"] += cost_model.band_append(
                            before, width, batch=len(selected), window_size=max_length)
                        offset = end
            if cache.seq_len != terminal_length:
                raise RuntimeError("rolling cache and fixed Full teacher positions differ")
            for index, uid in enumerate(selected):
                timeline, base = timelines[uid], rows[uid]
                inherited = timeline["inherited_count"]
                teacher = base["teacher"]
                if not timeline["reuse_teacher"]:
                    # Expanded reads select different oldest tie-boundary rows
                    # for two frozen fitting users. Keep the pure teacher and
                    # charge a fresh teacher aligned to this real rolling state.
                    ts, ids, bs = [torch.as_tensor(values[None], device=device) for values in timeline["terminal"]]
                    dt = torch.zeros_like(ts, dtype=torch.float32)
                    dt[:, 1:] = ts[:, 1:]-ts[:, :-1]
                    teacher_backend = attention_backend if terminal_length == max_length else "torch"
                    with native_backend([current], teacher_backend):
                        teacher = current.compute_kv(ids.long(), bs.long(), dt).to("cpu")
                    costs["mixed_extra_teacher_capture_flops"] += models[teacher_backend].full_cache(terminal_length)
                scenes[f"{uid}:mixed"] = {
                    **base, "uid": uid, "scene_kind": "mixed",
                    "parent": HSTUKVCache(cache.k[:, index:index+1].cpu(),
                                          cache.v[:, index:index+1].cpu(), terminal_length),
                    "candidates": base["candidates"][1::2],
                    "teacher": teacher, "query_delta": float(cutover-timeline["terminal_last_timestamp"]),
                    "producer": [4]*inherited+[5]*(terminal_length-inherited),
                    "native_writes": suffix_length,
                }
                records.append(dict(scene_id=f"{uid}:mixed", uid=uid, kind="mixed", queries=8,
                    parent_length=prefix_length, terminal_length=terminal_length,
                    inherited_count=inherited, native_writes=suffix_length,
                    rolling_evictions=prefix_length+suffix_length-terminal_length,
                    parent_first_timestamp=int(timeline["prefix"][0][0]),
                    parent_last_timestamp=timeline["parent_last_timestamp"],
                    first_append_timestamp=timeline["pseudo_release_timestamp"],
                    terminal_last_timestamp=timeline["terminal_last_timestamp"],
                    native_backend=backend, teacher_reused=timeline["reuse_teacher"],
                    terminal_tuple_sha256=terminal_tuple_hash(timeline["terminal"])))
    return scenes, records


@torch.no_grad()
def collect_layer(current, rows, uids, features, projection, parameters, layer,
                  *, batch_size, device, max_length, costs, summary_mode="producer_mean"):
    device_parameters = [{key: value.to(device=device, dtype=torch.float32)
                          for key, value in layer.items()} for layer in parameters]
    adapter = SharedReadAdapter(projection.to(device), device_parameters, producer_ids=PRODUCERS,
                               target=5, max_length=max_length, summary_mode=summary_mode)
    collected = {}
    cost_model = CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads, "torch")
    costs["shared_response_preparation_flops"] += adapter.estimate_flops()["shared_response_preparation"]
    for selected in groups(uids, rows, batch_size):
        cache = stacked(rows, selected, "parent", device)
        counts = torch.full((len(selected),), cache.seq_len, device=device)
        source_features = torch.stack([features[uid] for uid in selected]).to(device)
        view = adapter.prepare_features(source_features, counts)
        candidates = torch.stack([rows[uid]["candidates"] for uid in selected]).to(device)
        delta = torch.tensor([rows[uid]["query_delta"] for uid in selected], device=device)
        _, trace = score(current, cache, candidates, delta, trace=True, history_override=view)
        query, native = trace.queries[layer], trace.history_heads[layer]
        teacher_k = torch.cat([rows[uid]["teacher"].k[layer] for uid in selected]).to(device)
        teacher_v = torch.cat([rows[uid]["teacher"].v[layer] for uid in selected]).to(device)
        teacher = history_read(current.blocks[layer].attn, query, teacher_k, teacher_v)
        wanted = (teacher-native)/counts[:, None, None, None]
        observed = native.transpose(1, 2).flatten(2)/counts[:, None, None]
        prediction = (view(layer, query, native)-native)/counts[:, None, None, None]
        if not all(torch.isfinite(value).all() for value in (query, wanted, observed, prediction)):
            raise RuntimeError(f"nonfinite calibration tensors at layer {layer}")
        for index, uid in enumerate(selected):
            collected[uid] = tuple(value[index].cpu() for value in (query, wanted, observed, prediction))
        n, q, b = cache.seq_len, candidates.shape[1], len(selected)
        costs["query_trace_flops"] += eager_read(cost_model, n, queries=q, batch=b)
        costs["teacher_same_query_flops"] += teacher_history_read(cost_model, n, queries=q, batch=b)
        overhead = adapter.estimate_flops(batch=b, candidates=q)
        costs["corrected_prefix_flops"] += overhead["candidate_reads"]
        if layer < len(parameters):
            # The second call to this fitted layer above extracts its residual.
            costs["diagnostic_correction_flops"] += overhead["candidate_reads"]//len(parameters)
        costs["summary_publication_flops_estimate"] += overhead["summary_projection"]+overhead["view_generation"]
        costs["normalization_and_metrics_flops_estimate"] += 8*wanted.numel() + observed.numel()
    return tuple(torch.stack([collected[uid][index] for uid in uids]) for index in range(4))


@torch.no_grad()
def fit(current, rows, train, validation, *, batch_size, device, max_length,
        rank=32, timings=None, costs=None, summary_mode="producer_mean"):
    """Train/validation contain row keys (scene IDs in mixed mode), never split scenes across UIDs."""
    timings = defaultdict(float) if timings is None else timings
    costs = defaultdict(int) if costs is None else costs
    def summaries():
        result = {}
        for uid in train+validation:
            cache = rows[uid]["parent"]
            summary = ProducerSummary.from_cache(cache, rows[uid].get("producer", 4),
                                                 producer_ids=PRODUCERS, max_length=max_length)
            summary.native_writes.fill_(rows[uid].get("native_writes", 0))
            result[uid] = summary.features(mode=summary_mode)[0].cpu()
            costs["source_summary_flops_estimate"] += summary.estimate_flops("scan", length=cache.seq_len)
            costs["source_summary_flops_estimate"] += summary.estimate_flops("features")
        return result
    features = timed(summaries, timings, "source_summary", device)
    matrix = torch.stack([features[uid] for uid in train])
    projection = timed(lambda: SummaryProjection.fit(matrix, rank=rank), timings, "source_pca", device)
    fitted_rank = projection.projection.shape[-1]
    costs["source_pca_flops_estimate"] += projection_flops(len(train), matrix.shape[-1], fitted_rank)
    latent = projection.encode_features(matrix).cpu().double()
    costs["source_pca_encode_flops"] += len(train)*(2*matrix.shape[-1]*fitted_rank + 2*matrix.shape[-1])
    counts = torch.tensor([rows[uid]["parent"].seq_len for uid in train])
    parameters, layer_records = [], []
    for layer, block in enumerate(current.blocks):
        query, wanted, observed, _ = timed(lambda: collect_layer(
            current, rows, train, features, projection, parameters, layer,
            batch_size=batch_size, device=device, max_length=max_length, costs=costs,
            summary_mode=summary_mode),
            timings, "fitting_teacher_and_query_reads", device)
        p, record = timed(lambda: fit_layer(latent, query, wanted, observed, counts),
                          timings, "shared_ridge_fit", device)
        parameters.append({key: value.detach().cpu() for key, value in p.items()})
        costs["ridge_fit_flops_estimate"] += ridge_flops(
            len(train), query.shape[2], block.attn.num_heads, block.attn.head_dim,
            latent.shape[-1], observed.shape[-1])
        layer_records.append({"layer": layer, **record,
                              "target_rate_mse": float(wanted.double().square().mean())})
        print(json.dumps({"status": "layer_fitted", **layer_records[-1]}), flush=True)
    # Reporting on separate users occurs only after all parameters are fixed.
    validation_records = []
    validation_by_scene = defaultdict(list)
    validation_counts = torch.tensor([rows[uid]["parent"].seq_len for uid in validation], dtype=torch.double)
    for layer in range(len(current.blocks)):
        _, wanted, _, predicted = timed(lambda: collect_layer(
            current, rows, validation, features, projection, parameters, layer,
            batch_size=batch_size, device=device, max_length=max_length, costs=costs,
            summary_mode=summary_mode),
            timings, "validation_teacher_and_query_reads", device)
        error = float((predicted.double()-wanted.double()).square().mean())
        baseline = float(wanted.double().square().mean())
        aggregate_error = float(((predicted.double()-wanted.double()).square().mean((1, 2, 3))
                                 * validation_counts.square()).mean())
        aggregate_baseline = float((wanted.double().square().mean((1, 2, 3))*validation_counts.square()).mean())
        validation_records.append({"layer": layer, "rate_mse": error, "uncorrected_rate_mse": baseline,
                                   "relative_rate_mse": error/max(baseline, 1e-30),
                                   "aggregate_response_mse": aggregate_error,
                                   "uncorrected_aggregate_response_mse": aggregate_baseline,
                                   "relative_aggregate_response_mse": aggregate_error/max(aggregate_baseline, 1e-30)})
        if any("scene_kind" in rows[key] for key in validation):
            for kind in ("pure", "mixed"):
                indices = [index for index, key in enumerate(validation) if rows[key]["scene_kind"] == kind]
                e = (predicted[indices].double()-wanted[indices].double()).square().mean((1, 2, 3))
                b = wanted[indices].double().square().mean((1, 2, 3))
                counts_squared = validation_counts[indices].square()
                validation_by_scene[kind].append(dict(layer=layer, scenes=len(indices),
                    rate_mse=float(e.mean()), uncorrected_rate_mse=float(b.mean()),
                    relative_rate_mse=float(e.mean()/b.mean().clamp_min(1e-30)),
                    aggregate_response_mse=float((e*counts_squared).mean()),
                    uncorrected_aggregate_response_mse=float((b*counts_squared).mean()),
                    relative_aggregate_response_mse=float((e*counts_squared).sum()
                                                         /(b*counts_squared).sum().clamp_min(1e-30))))
    adapter = SharedReadAdapter(projection, parameters, producer_ids=PRODUCERS,
                                target=5, max_length=max_length, summary_mode=summary_mode)
    costs["shared_response_preparation_flops"] += adapter.estimate_flops()["shared_response_preparation"]
    diagnostics = {"layers": layer_records, "validation": validation_records,
                   "projection": projection.diagnostics}
    if validation_by_scene:
        diagnostics["validation_by_scene"] = dict(validation_by_scene)
    return adapter, diagnostics


def run(args):
    started = time.perf_counter()
    config, stage, versions, train, validation = inputs(args.config, args.fit_users, args.validation_users)
    if (args.output / "calibration.pt").exists() or (args.output / "calibration.json").exists():
        raise FileExistsError("calibration output already exists; choose a new output directory")
    torch.set_num_threads(args.threads)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.70, device)
        torch.cuda.reset_peak_memory_stats(device)
    timings, costs = defaultdict(float), defaultdict(int)
    parent, pp = timed(lambda: load_model(ROOT/versions["v4"]["checkpoint"], device),
                       timings, "model_load", device)
    model_config = pp["config"]
    del pp
    current, cp = timed(lambda: load_model(ROOT/versions["v5"]["checkpoint"], device),
                        timings, "model_load", device)
    if cp["config"] != model_config:
        raise RuntimeError("parent and current architectures differ")
    expected = {"num_layers": 6, "hidden_size": 192, "num_heads": 6, "max_seq_len": 1024,
                "activation": "elu_plus1", "block_variant": "legacy", "relative_position_bias": False,
                "gating": "silu_gate", "causal_diagonal": "inclusive"}
    if any(model_config[key] != value for key, value in expected.items()):
        raise RuntimeError("checkpoint architecture differs from the frozen Medium read operator")
    parent.requires_grad_(False)
    current.requires_grad_(False)
    dataset_path = ROOT/config["data"]["dataset"]["path"]
    dataset = json.loads(dataset_path.read_text())
    known = int(cp.get("known_vocab_size", dataset["foundation_items"]))
    del cp
    cutover = stage["days_half_open"][0]*DAY
    max_length = model_config["max_seq_len"]
    uids = train+validation
    history = timed(lambda: load_histories(
        uids, dataset_path=dataset_path, known_vocab_size=known,
        oov_buckets=model_config["num_items"]-known, start_timestamp=cutover,
        end_timestamp=cutover+1, max_history=max_length, threads=args.history_threads),
        timings, "history_io", device)
    histories = {uid: snapshot(history, uid, cutover, max_length) for uid in uids}
    del history
    rows = timed(lambda: capture(parent, current, histories, uids, cutover=cutover, known=known,
        queries=stage["calibration_queries_per_user"], device=device, batch_size=args.batch_size,
        history_length=max_length, attention_backend=args.attention_backend),
        timings, "parent_and_teacher_cache_capture", device)
    for uid in uids:
        rows[uid]["candidates"] = torch.from_numpy(mixed_candidates(
            uid, histories[uid][1], known, stage["calibration_queries_per_user"], seed=17))
    latest_history_timestamp = max(int(histories[uid][0][-1]) for uid in uids)
    history_length_histogram = dict(Counter(rows[uid]["parent"].seq_len for uid in uids))
    sparse = CostModel.for_scale("medium", args.attention_backend)
    dense = CostModel.for_scale("medium", "torch")
    one_capture = sum((sparse if rows[batch[0]]["parent"].seq_len == max_length else dense).full_cache(
        rows[batch[0]]["parent"].seq_len, batch=len(batch)) for batch in groups(uids, rows, args.batch_size))
    costs["parent_cache_capture_flops"] = one_capture
    costs["teacher_cache_capture_flops"] = one_capture
    train_scenes, validation_scenes, scene_records = train, validation, None
    if args.rolling_scenes:
        expanded = timed(lambda: load_histories(
            uids, dataset_path=dataset_path, known_vocab_size=known,
            oov_buckets=model_config["num_items"]-known, start_timestamp=cutover,
            end_timestamp=cutover+1, max_history=max_length+max_length//2, threads=args.history_threads),
            timings, "expanded_history_io", device)
        expanded_histories = {uid: snapshot(expanded, uid, cutover, max_length+max_length//2) for uid in uids}
        del expanded
        rows, scene_records = timed(lambda: rolling_scenes(
            parent, current, rows, histories, expanded_histories, uids, cutover=cutover,
            batch_size=args.batch_size, device=device, max_length=max_length,
            attention_backend=args.attention_backend, costs=costs),
            timings, "rolling_native_state_capture", device)
        del expanded_histories
    elif args.mixed_scenes:
        rows, scene_records = timed(lambda: mixed_scenes(
            current, rows, histories, uids, cutover=cutover, batch_size=args.batch_size,
            device=device, max_length=max_length, costs=costs),
            timings, "mixed_native_state_capture", device)
    if args.mixed_scenes or args.rolling_scenes:
        train_scenes = [f"{uid}:{kind}" for uid in train for kind in ("pure", "mixed")]
        validation_scenes = [f"{uid}:{kind}" for uid in validation for kind in ("pure", "mixed")]
    del parent, histories
    gc.collect()
    adapter, diagnostics = fit(current, rows, train_scenes, validation_scenes, batch_size=args.batch_size,
        device=device, max_length=max_length, rank=args.rank, timings=timings, costs=costs,
        summary_mode=args.summary_mode)
    distinct_caches = {id(row[key]): row[key] for row in rows.values() for key in ("parent", "teacher")}
    cache_bytes = sum(cache.k.nbytes+cache.v.nbytes for cache in distinct_caches.values())
    total_flops = sum(costs.values())
    metadata = {
        "status": "complete", "kind": "design_one_shared_read_v1", "scale": "medium", "edge": "v4_to_v5",
        "role": "development_canary" if len(train) != 256 or len(validation) != 16 else "development_calibration",
        "configuration": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "users_file_sha256": config["users"]["sha256"], "uids": train, "fit_uids": train,
        "users": len(train), "budget": len(train), "validation_uids": validation,
        "evaluation_users_excluded": config["users"]["groups"][stage["evaluation_group"]]["count"],
        "queries_per_user": stage["calibration_queries_per_user"], "cutover": cutover,
        "latest_history_timestamp": latest_history_timestamp, "history_length": max_length,
        "history_length_histogram": history_length_histogram,
        "candidate_rule": "8 most recent unique known items; remaining uniform known catalog without replacement; SeedSequence([17,uid]); no feedback labels",
        "history_tie_order": "timestamp_mapped_item_behavior, aligned Parent and Current Full histories",
        "cache_state": ("pure V4 plus full-prefix native rolling scenes with actual evictions, strictly before cutover"
                        if args.rolling_scenes else "pure V4 and native V4-prefix/V5-suffix scenes, all events strictly before cutover"
                        if args.mixed_scenes else "pure V4 at cutover; no post-release or mixed-producer fitting scenes"),
        "summary": ("producer-wise compact K/V sums divided by total retained count, plus counts/fractions/native writes; no influence sketch"
                    if args.summary_mode == "producer_mass" else
                    "producer-wise compact K/V means, counts, fractions and native write count; no influence sketch"),
        "summary_mode": args.summary_mode,
        "fitting": "shared aggregate-response ridge .01; fit-only requested PCA rank bounded by fit sample rank; lower fitted layers frozen before next actual-query capture",
        "rank_requested": args.rank, "teacher": "Current Full cache on strictly pre-release history; same actual corrected-branch q at each fitted layer",
        "validation_rule": "held-out fitting-reservation UIDs, diagnostics only after all parameters fixed; no tuning or checkpoint selection",
        "checkpoint_hashes": {name: versions[name]["checkpoint_sha256"] for name in ("v4", "v5")},
        "checkpoint_verification": "sealed manifest and seal bindings checked; weight payload hashes inherited, not rehashed",
        "model_config": model_config, "execution_sources": source_hashes(), "diagnostics": diagnostics,
        "cost": {**dict(costs), "calibration_flops": total_flops,
                 "teacher_flops": costs["teacher_cache_capture_flops"]+costs["teacher_same_query_flops"]
                                  +costs.get("mixed_extra_teacher_capture_flops", 0),
                 "scope": "includes Parent capture, all Current Full teachers, fit and validation reads, source summary/PCA/publication, ridge and diagnostics",
                 "convention": "multiply-add=2; analytical executed-shape cache/read costs; PCA eigensolve, ridge LU/statistics and summary publication estimates; no FP64 multiplier; CPU/GPU time reported separately"},
        "timings_seconds": dict(timings), "elapsed_seconds": time.perf_counter()-started,
        "paired_cache_bytes": cache_bytes,
        "cpu_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1<<20),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device)/(1<<30) if device.type == "cuda" else 0,
        "settings": {"batch_size": args.batch_size, "attention_backend": args.attention_backend,
                     "torch_threads": args.threads, "history_threads": args.history_threads,
                     "device": str(device), "gpu_memory_fraction": .70},
    }
    if args.mixed_scenes or args.rolling_scenes:
        metadata["scene_protocol"] = dict(
            mode="pure_plus_native_rolling" if args.rolling_scenes else "pure_plus_midpoint_native",
            fitting_scenes=len(train_scenes),
            validation_scenes=len(validation_scenes), queries_per_scene=8, total_queries_per_uid=16,
            candidate_partition="even candidate indices for pure, odd indices for mixed; no candidate repeated across scenes",
            split=("half the terminal events assigned to the native tail, at most512; move split after any intersected timestamp group"
                   if args.rolling_scenes else "nearest interior complete timestamp-group boundary to midpoint; equal distance selects earlier boundary"),
            uid_separation="both scenes of a UID stay in its frozen fitting or validation role; PCA uses fitting scenes only",
            prefix=("extra Parent Full on the last1024 events before the synthetic cutover, using expanded pre-release history"
                    if args.rolling_scenes else "causal prefix sliced from the same Parent Full cache; no additional Parent recomputation"),
            teacher=("pure teacher reused when chronological terminal tuples match; changed oldest tie-boundary rows receive a separately charged aligned Current Full teacher"
                     if args.rolling_scenes else "one terminal Current Full cache per UID, shared by its two scenes; materialization charged once"),
            replay=("native bounded appends in at most32-event bands, enforcing real rolling evictions; no correction on write tokens"
                    if args.rolling_scenes else "one native Torch bounded append of the complete observed suffix; no correction on write tokens"),
            native_delta_rule="adjacent observed-event timestamp gaps clamped to [0, 7 days], matching serving appends; Full teacher retains its canonical raw gaps",
            scope=("same terminal history window; single-edge actual native append/eviction lifetime, not multiple releases"
                   if args.rolling_scenes else "same terminal history window; covers producer mixing without evictions or multiple releases"),
            pre_release_history_limit=max_length+max_length//2 if args.rolling_scenes else max_length,
            scenes=scene_records,
        )
        metadata["settings"]["mixed_scenes"] = True
        metadata["settings"]["rolling_scenes"] = args.rolling_scenes
        if args.rolling_scenes:
            metadata["scene_protocol"]["extra_teacher_uids"] = [record["uid"] for record in scene_records
                if record["kind"] == "mixed" and not record["teacher_reused"]]
            metadata["scene_protocol"]["terminal_tuple_hash_rule"] = "SHA256 of chronological timestamp/mapped-item/behavior rows encoded as little-endian int64"
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": metadata["kind"], "adapter": adapter.state_dict(), "metadata": metadata},
               args.output / "calibration.pt")
    metadata["weights_sha256"] = sha256_file(args.output / "calibration.pt")
    (args.output / "calibration.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"status": "calibration_complete", "output": str(args.output),
                      "elapsed_seconds": metadata["elapsed_seconds"], "calibration_flops": total_flops}), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(4))
    parser.add_argument("--fit-users", type=int, help="first frozen fitting UIDs for a small canary")
    parser.add_argument("--validation-users", type=int, help="first frozen validation UIDs for a small canary")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=32, choices=(0, 8, 32))
    parser.add_argument("--summary-mode", choices=("producer_mean", "producer_mass"), default="producer_mean")
    scene_mode = parser.add_mutually_exclusive_group()
    scene_mode.add_argument("--mixed-scenes", action="store_true",
                        help="fixed pure/midpoint-native scenes, eight disjoint queries each per UID")
    scene_mode.add_argument("--rolling-scenes", action="store_true",
                           help="fixed pure/native-rolling scenes; extended prehistory and actual evictions")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--history-threads", type=int, default=8)
    parser.add_argument("--attention-backend", choices=("torch", "triton"), default="triton")
    run(parser.parse_args())
