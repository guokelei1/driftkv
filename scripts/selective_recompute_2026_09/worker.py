#!/usr/bin/env python3
"""One GPU's bounded user shards, sharing stream replay across four baselines."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from evaluate_yambda500m_foundation_raw import balanced_users, load_histories, load_model
from selective_recompute_2026_09.common import (
    OUTPUT, METHODS, edge_name, plan, sha256, signature, sources, write_json,
)
from selective_recompute_2026_09.cost import CostModel
from selective_recompute_2026_09.evaluate import score_unit
from selective_recompute_2026_09.scheduling import ordered_uids


def run(args):
    config = plan()
    name = edge_name(args.edge)
    panel_dir = args.panel_root / args.scale / name
    panel = json.loads((panel_dir / "binding.json").read_text())
    requests_path = panel_dir / f"{args.partition}_requests.parquet"
    all_rows = pq.read_table(requests_path).to_pylist()
    assignment = balanced_users(all_rows, args.world_size)
    by_user = defaultdict(list)
    for request in all_rows:
        if assignment[int(request["uid"])] == args.rank:
            by_user[int(request["uid"])].append(request)
    uids = sorted(by_user, key=lambda uid: (-len(by_user[uid]), uid))
    if args.limit_users:
        uids = uids[:args.limit_users]
    if not uids:
        raise ValueError("empty rank panel")
    uids = ordered_uids(by_user, max_length=config["history_length"], uids=uids)
    cal_path = args.calibration_root / "layer" / args.scale / name / "calibration.json"
    calibration = json.loads(cal_path.read_text())
    intervals = calibration["selected_intervals"]
    source_hashes = sources()
    bound = {"execution_sources": source_hashes, "panel_binding": sha256(panel_dir / "binding.json"),
             "requests": sha256(requests_path), "calibration": sha256(cal_path), "uids": uids,
             "partition": args.partition, "scale": args.scale, "edge": name}
    source_signature = signature(bound)
    shared_dir = args.output_root / "runtime" / args.scale / name / f"rank{args.rank}"
    complete_path = shared_dir / "complete.json"
    if complete_path.exists():
        previous = json.loads(complete_path.read_text())
        if previous["source_signature"] != source_signature:
            raise RuntimeError("finished rank has different source/panel inputs")
        print(json.dumps(previous), flush=True)
        return previous
    os.environ["EVOKV_ATTENTION_BACKEND"] = config["attention_backend"]
    torch.set_num_threads(config["torch_threads"])
    pa.set_cpu_count(config["history_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    free, total = torch.cuda.mem_get_info(device)
    if free / total < config["initial_free_fraction"]:
        raise RuntimeError("GPU has less than the configured free-memory reserve")
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"], device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    source = panel["sources"]
    parent, pp = load_model(Path(source["parent"]["path"]), device)
    current, cp = load_model(Path(source["current"]["path"]), device)
    if pp["config"] != cp["config"]:
        raise RuntimeError("adjacent model configurations differ")
    model_seconds = time.perf_counter() - started
    stress_report = None
    stress_seconds = 0.0
    if args.stress:
        from selective_recompute_2026_09.stress import stress
        stress_start = time.perf_counter()
        stress_report = stress(parent, current, args.scale, device, config)
        stress_seconds = time.perf_counter() - stress_start
        write_json(shared_dir / "stress.json", stress_report)
    dataset_path = Path(source["dataset"]["path"])
    dataset = json.loads(dataset_path.read_text())
    known = int(cp.get("known_vocab_size", dataset["foundation_items"]))
    cutover = int(panel["days"][0]) * 86400
    history = load_histories(uids, dataset_path=dataset_path, known_vocab_size=known,
                            oov_buckets=int(cp["config"]["num_items"]) - known,
                            start_timestamp=cutover, end_timestamp=int(panel["days"][1]) * 86400,
                            max_history=1024, threads=config["history_threads"])
    history_seconds = time.perf_counter() - started - model_seconds - stress_seconds
    cost = CostModel.for_scale(args.scale, config["attention_backend"])
    if (cost.hidden_size, cost.num_layers, cost.num_heads) != (current.cfg.hidden_size, current.cfg.num_layers, current.cfg.num_heads):
        raise RuntimeError("cost architecture differs from loaded checkpoint")
    cohort = args.cohort_size or config["cohort_sizes"][args.scale]
    query_batch = args.query_batch or config["query_batches"][args.scale]
    unit_users = args.unit_users or config["unit_users"]
    reductions, units = [], []
    for unit, offset in enumerate(range(0, len(uids), unit_users)):
        selected = uids[offset:offset + unit_users]
        seal_path = shared_dir / f"unit_{unit:05d}.json"
        if seal_path.exists():
            seal = json.loads(seal_path.read_text())
            if seal["source_signature"] != source_signature or seal["uids"] != selected:
                raise RuntimeError("resumable unit has changed inputs")
            for entry in seal["outputs"].values():
                if sha256(Path(entry["path"])) != entry["sha256"]:
                    raise RuntimeError("resumable baseline shard changed")
            units.append(seal)
            continue
        begin = time.perf_counter()
        while True:
            try:
                rows, stats, controls = score_unit(selected, by_user, history, parent, current, cutover,
                    intervals=intervals, cost_model=cost, cohort_size=cohort, query_batch=query_batch,
                    sparse_query_chunk=config["sparse_query_chunk"], verify=args.verify,
                    append_band_size=config["append_band_size"])
                break
            except torch.cuda.OutOfMemoryError:
                if cohort == 1 and query_batch == 1:
                    raise
                reductions.append({"unit": unit, "old_cohort": cohort, "old_query_batch": query_batch})
                cohort, query_batch = max(1, cohort // 2), max(1, query_batch // 2)
                gc.collect()
                torch.cuda.empty_cache()
        expected_ids = {r["request_id"] for uid in selected for r in by_user[uid]}
        outputs = {}
        for method in METHODS:
            grouped = defaultdict(list)
            for row in rows[method]:
                grouped[row["budget"]].append(row["request_id"])
            if not grouped or any(len(ids) != len(expected_ids) or set(ids) != expected_ids for ids in grouped.values()):
                raise RuntimeError("baseline budget missing/duplicating evaluated requests")
            path = args.output_root / method / args.scale / name / f"rank{args.rank}" / f"shard_{unit:05d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".parquet.partial")
            pq.write_table(pa.Table.from_pylist(rows[method]), temporary, compression="zstd")
            os.replace(temporary, path)
            outputs[method] = {"path": str(path), "sha256": sha256(path), "rows": len(rows[method])}
        seal = {"source_signature": source_signature, "uids": selected, "requests": len(expected_ids),
                "outputs": outputs, "stats": stats, "controls": controls,
                "seconds": time.perf_counter() - begin, "cohort_size": cohort, "query_batch": query_batch}
        write_json(seal_path, seal)
        units.append(seal)
        progress = {"scale": args.scale, "edge": name, "rank": args.rank,
                    "users": offset + len(selected), "total_users": len(uids),
                    "requests": sum(u["requests"] for u in units),
                    "elapsed_seconds": time.perf_counter() - started,
                    "compute_seconds": sum(u["seconds"] for u in units),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1 << 30),
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1 << 30),
                    "gpu_total_gib": total / (1 << 30), "batch_reductions": reductions}
        write_json(shared_dir / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    final = {"status": "complete", "source_signature": source_signature, "inputs": bound,
             "scale": args.scale, "edge": name, "rank": args.rank, "users": len(uids),
             "requests": sum(u["requests"] for u in units), "partition": args.partition,
             "model_load_seconds": model_seconds, "history_load_seconds": history_seconds,
             "stress_seconds": stress_seconds,
             "compute_seconds": sum(u["seconds"] for u in units),
             "elapsed_seconds": time.perf_counter() - started,
             "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1 << 30),
             "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1 << 30),
             "gpu_total_gib": total / (1 << 30), "batch_reductions": reductions,
             "units": len(units), "cohort_size": cohort, "query_batch": query_batch}
    write_json(complete_path, final)
    print(json.dumps({k: v for k, v in final.items() if k != "inputs"}), flush=True)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--panel-root", type=Path, default=OUTPUT / "panels")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--calibration-root", type=Path, default=OUTPUT)
    parser.add_argument("--partition", choices=("evaluation", "canary"), default="canary")
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--cohort-size", type=int, default=0)
    parser.add_argument("--query-batch", type=int, default=0)
    parser.add_argument("--unit-users", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    if args.partition == "canary" and args.output_root.resolve() == OUTPUT.resolve():
        parser.error("canary requires a separate --output-root under probes/")
    run(args)
