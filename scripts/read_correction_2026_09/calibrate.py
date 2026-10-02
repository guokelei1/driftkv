#!/usr/bin/env python3
"""Fit request-local corrections on independent, strictly pre-release histories."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import torch

from evaluate_yambda500m_foundation_raw import load_histories, load_model
from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.read_correction import score_corrected
from read_correction_2026_09.common import (
    METHODS, OUTPUT, PANEL_ROOT, edge_name, method_directory, plan, sha256, signature,
    sources, write_json,
)
from read_correction_2026_09.cost import (
    CostModel, correction_forward, eager_read, query_ridge_fit, teacher_history_read,
)
from read_correction_2026_09.history_conditioned.fit import fit as fit_history
from read_correction_2026_09.query_only.fit import fit as fit_query
from selective_recompute_2026_09.calibrate import snapshot
from selective_recompute_2026_09.evaluate import native_backend


def candidates(uid, known, count):
    return np.random.default_rng(np.random.SeedSequence([17, int(uid)])).choice(
        known - 1, count, replace=False) + 1


def groups(uids, rows, batch_size):
    lengths = defaultdict(list)
    for uid in uids:
        lengths[rows[uid]["parent"].seq_len].append(uid)
    for _, group in sorted(lengths.items()):
        for start in range(0, len(group), batch_size):
            yield group[start:start + batch_size]


def stacked(rows, uids, key, device):
    return HSTUKVCache(
        torch.cat([rows[uid][key].k for uid in uids], 1).to(device),
        torch.cat([rows[uid][key].v for uid in uids], 1).to(device),
        rows[uids[0]][key].seq_len,
    )


@torch.no_grad()
def capture(parent, current, histories, uids, *, cutover, known, queries, device,
            batch_size, history_length, attention_backend):
    """CPU retains a bounded pair of caches per fitting user; GPU holds one batch."""
    by_length, rows = defaultdict(list), {}
    for uid in uids:
        by_length[len(histories[uid][0])].append(uid)
    for length, group in sorted(by_length.items()):
        backend = attention_backend if length == history_length else "torch"
        for start in range(0, len(group), batch_size):
            selected = group[start:start + batch_size]
            raw = [histories[uid] for uid in selected]
            timestamps = torch.as_tensor(np.stack([x[0] for x in raw]), device=device)
            items = torch.as_tensor(np.stack([x[1] for x in raw]), dtype=torch.long, device=device)
            behaviors = torch.as_tensor(np.stack([x[2] for x in raw]), dtype=torch.long, device=device)
            deltas = torch.zeros_like(timestamps, dtype=torch.float32)
            deltas[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
            with native_backend([parent, current], backend):
                old, full = parent.compute_kv(items, behaviors, deltas), current.compute_kv(items, behaviors, deltas)
            for index, uid in enumerate(selected):
                rows[uid] = {
                    "parent": HSTUKVCache(old.k[:, index:index+1].cpu(), old.v[:, index:index+1].cpu(), length),
                    "teacher": HSTUKVCache(full.k[:, index:index+1].cpu(), full.v[:, index:index+1].cpu(), length),
                    "candidates": torch.tensor(candidates(uid, known, queries), dtype=torch.long),
                    "query_delta": float(cutover - histories[uid][0][-1]),
                }
            print(json.dumps({"status": "cache_capture", "users": len(rows), "total": len(uids)}), flush=True)
            del old, full
    return rows


@torch.no_grad()
def layer_targets(current, rows, uids, modules, layer, *, batch_size, device):
    """Teacher sees the actual already-corrected branch query at this layer."""
    collected = {}
    for selected in groups(uids, rows, batch_size):
        cache = stacked(rows, selected, "parent", device)
        teacher_k = torch.cat([rows[uid]["teacher"].k[layer] for uid in selected], 0).to(device)
        teacher_v = torch.cat([rows[uid]["teacher"].v[layer] for uid in selected], 0).to(device)
        counts = torch.full((len(selected),), cache.seq_len, device=device, dtype=torch.long)
        ids = torch.stack([rows[uid]["candidates"] for uid in selected]).to(device)
        dt = torch.tensor([rows[uid]["query_delta"] for uid in selected], device=device)
        _, trace = score_corrected(current, cache, ids, dt, modules, counts, trace=True)
        query = trace.queries[layer]
        target = (history_read(current.blocks[layer].attn, query, teacher_k, teacher_v)
                  - trace.history_heads[layer]) / counts[:, None, None, None]
        for index, uid in enumerate(selected):
            collected[uid] = (query[index].cpu(), target[index].cpu())
    return (torch.stack([collected[uid][0] for uid in uids]),
            torch.stack([collected[uid][1] for uid in uids]))


def padded_layer(rows, uids, layer):
    counts = torch.tensor([rows[uid]["parent"].seq_len for uid in uids], dtype=torch.long)
    width = rows[uids[0]]["parent"].k.shape[-1]
    keys = torch.zeros((len(uids), int(counts.max()), width))
    values = torch.zeros_like(keys)
    for i, uid in enumerate(uids):
        n = int(counts[i])
        keys[i, :n] = rows[uid]["parent"].k[layer, 0]
        values[i, :n] = rows[uid]["parent"].v[layer, 0]
    return keys, values, counts


def fit_budget(current, rows, uids, method, config, *, device, scale, batch_size):
    modules, records = [None] * len(current.blocks), []
    queries = config["calibration_queries_per_user"]
    lengths = [rows[uid]["parent"].seq_len for uid in uids]
    cost = CostModel.for_scale(scale, config["attention_backend"])
    torch_cost = CostModel.for_scale(scale, "torch")
    costs = {"parent_and_teacher_cache_flops": 2 * sum(
        (cost if n == config["history_length"] else torch_cost).full_cache(n) for n in lengths),
        "query_trace_flops": 0, "same_query_teacher_flops": 0,
        "ridge_fit_flops_estimate": 0, "history_fit_flops_estimate": 0,
        "normalization_and_loss_flops": 0}
    for layer, block in enumerate(current.blocks):
        began = time.perf_counter()
        q, targets = layer_targets(current, rows, uids, modules, layer,
                                   batch_size=batch_size, device=device)
        qmodule, qstats = fit_query(q, targets, ridge=config["ridge"])
        costs["query_trace_flops"] += sum(eager_read(cost, n, queries=queries) for n in lengths)
        costs["same_query_teacher_flops"] += sum(teacher_history_read(cost, n, queries=queries) for n in lengths)
        # Teacher-minus-native and divide by N; target mean-square diagnostic.
        costs["normalization_and_loss_flops"] += 4 * targets.numel()
        costs["query_trace_flops"] += sum(correction_forward(m.get_config(), n, queries=queries)
                                            for m in modules if m is not None for n in lengths)
        costs["ridge_fit_flops_estimate"] += query_ridge_fit(block.attn.num_heads, block.attn.head_dim, len(uids)*queries)
        if method == "history_conditioned":
            k, v, counts = padded_layer(rows, uids, layer)
            module, stats = fit_history(
                q, targets, k, v, counts, initial_query=qmodule,
                width=config["history_width"], epochs=config["history_epochs"],
                learning_rate=config["history_learning_rate"], weight_decay=config["history_weight_decay"],
                batch_size=batch_size, token_chunk=config["history_token_chunk"],
                query_chunk=config["history_query_chunk"], seed=config["seed"]+layer, device=device,
            )
            forward = correction_forward(module.get_config(), k.shape[1], queries=queries,
                                         batch=len(uids), include_scale_add=False)
            costs["history_fit_flops_estimate"] += forward * (3*config["history_epochs"] + 2)
            costs["normalization_and_loss_flops"] += (int(counts.sum())*k.shape[-1]*12
                + targets.numel()*(config["history_epochs"]+2)*4
                + targets.numel()*config["history_epochs"]*2
                + stats["parameter_count"]*stats["optimizer_steps"]*14
                + stats["output_normalization_flops"])
            stats["initial_query_ridge"] = qstats
            stats["history_weight_norm"] = float(module.history_weight.detach().norm())
            del k, v, counts
        else:
            module, stats = qmodule.to(device), qstats
        module.eval()
        modules[layer] = module
        records.append({"layer": layer, "config": module.get_config(), "fit": stats,
                        "elapsed_seconds": time.perf_counter()-began,
                        "target_rate_mean_square": float(targets.double().square().mean())})
        print(json.dumps({"status": "layer_fit", "method": method, "users": len(uids),
                          "layer": layer, "layers": len(modules), **records[-1]}), flush=True)
    costs["calibration_flops"] = sum(costs.values())
    costs["convention"] = "analytical arithmetic; ridge solve and backward/Adam estimated, not measured hardware instructions"
    costs["history_backward_estimate"] = "training forward plus backward = 3 * forward; optimizer estimate 14 operations per parameter per step"
    return modules, records, costs


def run(args):
    config = plan()
    budgets = sorted(set(args.budgets or config["calibration_budgets"]))
    if args.limit_users:
        budgets = [min(value, args.limit_users) for value in budgets]
        budgets = sorted(set(budgets))
    if args.epochs is not None:
        config["history_epochs"] = args.epochs
    methods = args.method or list(METHODS)
    binding_path = args.panel_root / args.scale / edge_name(args.edge) / "binding.json"
    binding = json.loads(binding_path.read_text())
    users_path = args.calibration_root / args.scale / edge_name(args.edge) / "calibration_users.json"
    reservation = json.loads(users_path.read_text())
    if reservation["source_binding"]["sha256"] != sha256(binding_path):
        raise RuntimeError("calibration reservation panel binding changed")
    if min(budgets) < 1 or max(budgets) > len(reservation["uids"]):
        raise ValueError("calibration budgets exceed reserved users")
    all_uids = reservation["uids"][:max(budgets)]
    torch.set_num_threads(config["torch_threads"])
    pa.set_cpu_count(config["history_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < config["initial_free_fraction"]*total:
        raise RuntimeError("GPU lacks the configured initial free-memory margin")
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"], device)
    torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    execution_sources = sources()
    os.environ["EVOKV_ATTENTION_BACKEND"] = config["attention_backend"]
    parent, payload = load_model(ROOT/binding["sources"]["parent"]["path"], device)
    parent_config = payload["config"]
    del payload
    current, payload = load_model(ROOT/binding["sources"]["current"]["path"], device)
    if parent_config != payload["config"]:
        raise RuntimeError("parent/current architecture differs")
    model_config = payload["config"]
    dataset_path = ROOT/binding["sources"]["dataset"]["path"]
    dataset = json.loads(dataset_path.read_text())
    known = int(payload.get("known_vocab_size", dataset["foundation_items"]))
    oov = int(model_config["num_items"]) - known
    del payload
    parent.requires_grad_(False)
    current.requires_grad_(False)
    cutover = int(binding["cutover"])
    history = load_histories(all_uids, dataset_path=dataset_path, known_vocab_size=known,
                            oov_buckets=oov, start_timestamp=cutover, end_timestamp=cutover+1,
                            max_history=config["history_length"], threads=config["history_threads"])
    histories = {uid: snapshot(history, uid, cutover, config["history_length"]) for uid in all_uids}
    del history
    width = current.blocks[0].attn.num_heads * current.blocks[0].attn.head_dim
    cache_bytes = 4*4*len(current.blocks)*width*sum(len(histories[uid][0]) for uid in all_uids)
    if cache_bytes > 128*(1<<30):
        raise RuntimeError("calibration CPU cache exceeds the 128 GiB per-process bound")
    batch_size = args.batch_size or config["calibration_batches"][args.scale]
    rows = capture(parent, current, histories, all_uids, cutover=cutover, known=known,
                   queries=config["calibration_queries_per_user"], device=device, batch_size=batch_size,
                   history_length=config["history_length"], attention_backend=config["attention_backend"])
    del parent, histories
    gc.collect()
    cache_seconds = time.perf_counter()-began
    results = []
    for method in methods:
        for budget in budgets:
            selected = all_uids[:budget]
            directory = method_directory(method, args.scale, args.edge, output_root=args.output_root, revision=args.revision)
            output = directory/f"calibration_c{budget}.json"
            model_path = output.with_suffix(".pt")
            provenance = signature({"sources": execution_sources, "config": config,
                                    "users_sha256": sha256(users_path), "uids": selected,
                                    "method": method, "batch_size": batch_size})
            if output.exists():
                record = json.loads(output.read_text())
                if record["source_signature"] != provenance or sha256(model_path) != record["weights_sha256"]:
                    raise RuntimeError(f"existing calibration differs; use a new candidate revision: {output}")
                results.append(record)
                continue
            start = time.perf_counter()
            modules, layer_records, costs = fit_budget(current, rows, selected, method, config,
                                                       device=device, scale=args.scale, batch_size=batch_size)
            torch.cuda.synchronize(device)
            record = {
                "status": "complete", "kind": method, "scale": args.scale, "edge": edge_name(args.edge),
                "revision": args.revision or config["revision"], "evaluation_role": "development_exploration",
                "users": budget, "uids": selected, "queries_per_user": config["calibration_queries_per_user"],
                "cutover": cutover, "layers": layer_records, "cost": costs,
                "history_length_histogram": dict(Counter(rows[uid]["parent"].seq_len for uid in selected)),
                "cache_bytes_actual_shared": cache_bytes, "shared_cache_users": len(all_uids),
                "fit_seconds": time.perf_counter()-start, "shared_cache_prepare_seconds": cache_seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/(1<<30),
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/(1<<30),
                "cpu_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1<<20),
                "batch_size": batch_size, "settings": config, "model_config": model_config,
                "source_signature": provenance, "execution_source_hashes": execution_sources,
                "calibration_users_sha256": sha256(users_path), "panel_binding_sha256": sha256(binding_path),
                "checkpoint_hashes": {key: binding["sources"][key]["sha256"] for key in ("parent", "current")},
                "target": "same corrected-prefix query: (Current Full history response - native inherited history response)/N",
                "history_tie_order": "timestamp_mapped_item_behavior; teacher uses same aligned rows as inherited cache",
                "full_anchor_note": "saved Full retains its original tie order; it is not silently regenerated",
                "fitting_labels": "none; uniform known-catalog candidates SeedSequence([17,uid])",
            }
            directory.mkdir(parents=True, exist_ok=True)
            payload = {"kind": method, "modules": [{"config": m.get_config(),
                         "state_dict": {key: value.detach().cpu() for key, value in m.state_dict().items()}}
                        for m in modules], "metadata": record}
            temporary = model_path.with_suffix(".pt.partial")
            torch.save(payload, temporary)
            temporary.replace(model_path)
            record["weights_sha256"] = sha256(model_path)
            write_json(output, record)
            results.append(record)
            print(json.dumps({"status": "calibration_complete", "output": str(output),
                              "fit_seconds": record["fit_seconds"]}), flush=True)
            del modules, payload
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1,6), required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--method", choices=METHODS, nargs="+")
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--limit-users", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--revision")
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=OUTPUT/"panels")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    run(parser.parse_args())
