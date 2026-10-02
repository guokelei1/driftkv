#!/usr/bin/env python3
"""One GPU's resumable shards for both development correction methods."""
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
from hstu_kvcache.read_correction import build_correction
from read_correction_2026_09.common import (OUTPUT, PANEL_ROOT, METHODS, edge_name,
    method_directory, plan, sha256, signature, sources, write_json)
from read_correction_2026_09.cost import CostModel
from read_correction_2026_09.evaluate import score_unit
from selective_recompute_2026_09.scheduling import ordered_uids


def runtime_directory(output_root, scale, edge):
    return Path(output_root) / "runtime" / "development" / plan()["revision"] / scale / edge_name(edge)


def calibration_files(calibration_root, scale, edge, budgets):
    result = {}
    for method in METHODS:
        folder = method_directory(method, scale, edge, output_root=calibration_root)
        result[method] = {str(c): {suffix: {"path": str(folder / f"calibration_c{c}.{suffix}"),
            "sha256": sha256(folder / f"calibration_c{c}.{suffix}")}
            for suffix in ("pt", "json")} for c in budgets}
    return result


def verify_unit(seal, source_signature, selected):
    if seal["source_signature"] != source_signature or seal["uids"] != selected:
        raise RuntimeError("resumable shard inputs changed")
    if set(seal["outputs"]) != set(METHODS):
        raise RuntimeError("sealed unit does not contain both correction methods")
    for entry in seal["outputs"].values():
        if sha256(entry["path"]) != entry["sha256"]:
            raise RuntimeError("sealed correction shard changed")


def run(args):
    config = plan()
    args.budgets = sorted(set(args.budgets or config["budgets"]))
    name = edge_name(args.edge)
    panel_dir = args.panel_root / args.scale / name
    panel_path = panel_dir / "binding.json"
    panel = json.loads(panel_path.read_text())
    requests_path = panel_dir / f"{args.partition}_requests.parquet"
    if sha256(requests_path) != panel["files"][f"{args.partition}_requests"]["sha256"]:
        raise RuntimeError("frozen request panel changed")
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
    cal_files = calibration_files(args.calibration_root, args.scale, args.edge, args.budgets)
    panel_users = {int(r["uid"]) for r in all_rows}
    for method, variants in cal_files.items():
        for budget, files in variants.items():
            record = json.loads(Path(files["json"]["path"]).read_text())
            if (record["status"], record["kind"], record["scale"], record["edge"], record["users"]) != (
                    "complete", method, args.scale, name, int(budget)):
                raise RuntimeError("calibration method/edge/budget differs")
            if record["panel_binding_sha256"] != sha256(panel_path) or record["weights_sha256"] != files["pt"]["sha256"]:
                raise RuntimeError("calibration weights/panel binding differs")
            if panel_users.intersection(record["uids"]):
                raise RuntimeError("fitting and scored panel users overlap")
    bound = {"execution_sources": sources(), "panel_binding": sha256(panel_path),
             "requests": sha256(requests_path), "calibrations": cal_files, "uids": uids,
             "partition": args.partition, "scale": args.scale, "edge": name,
             "budgets": args.budgets, "world_size": args.world_size,
             "verify": args.verify, "evaluation_role": "development_exploration"}
    shared_dir = runtime_directory(args.output_root, args.scale, args.edge) / f"rank{args.rank}"
    unit_users = args.unit_users or config["unit_users"]
    # The unit partition is part of resume identity, including empty interrupted runs.
    bound["unit_users"] = unit_users
    source_signature = signature(bound)
    complete_path = shared_dir / "complete.json"
    if complete_path.exists():
        previous = json.loads(complete_path.read_text())
        if previous["source_signature"] != source_signature:
            raise RuntimeError("finished rank has different source/panel/calibration inputs")
        for unit, offset in enumerate(range(0, len(uids), unit_users)):
            seal = json.loads((shared_dir / f"unit_{unit:05d}.json").read_text())
            verify_unit(seal, source_signature, uids[offset:offset + unit_users])
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
    corrections = {}
    for method, variants in cal_files.items():
        corrections[method] = {}
        for budget, files in variants.items():
            artifact = torch.load(files["pt"]["path"], map_location="cpu", weights_only=False)
            if artifact["kind"] != method or len(artifact["modules"]) != len(current.blocks):
                raise RuntimeError("calibration artifact method/layer count differs")
            corrections[method][int(budget)] = [build_correction(method, item["config"], item["state_dict"])
                .to(device=device, dtype=next(current.parameters()).dtype).eval()
                for item in artifact["modules"]]
    model_seconds = time.perf_counter() - started
    dataset_path = Path(source["dataset"]["path"])
    dataset = json.loads(dataset_path.read_text())
    known = int(cp.get("known_vocab_size", dataset["foundation_items"]))
    cutover = int(panel["days"][0]) * 86400
    history = load_histories(uids, dataset_path=dataset_path, known_vocab_size=known,
        oov_buckets=int(cp["config"]["num_items"]) - known, start_timestamp=cutover,
        end_timestamp=int(panel["days"][1]) * 86400, max_history=config["history_length"],
        threads=config["history_threads"])
    history_seconds = time.perf_counter() - started - model_seconds
    cost = CostModel.for_scale(args.scale, config["attention_backend"])
    if (cost.hidden_size, cost.num_layers, cost.num_heads) != (current.cfg.hidden_size,
            current.cfg.num_layers, current.cfg.num_heads):
        raise RuntimeError("cost architecture differs from checkpoint")
    cohort = args.cohort_size or config["cohort_sizes"][args.scale]
    query_batch = args.query_batch or config["query_batches"][args.scale]
    reductions, units = [], []
    for unit, offset in enumerate(range(0, len(uids), unit_users)):
        selected = uids[offset:offset + unit_users]
        seal_path = shared_dir / f"unit_{unit:05d}.json"
        if seal_path.exists():
            seal = json.loads(seal_path.read_text())
            verify_unit(seal, source_signature, selected)
            units.append(seal)
            continue
        begin = time.perf_counter()
        while True:
            try:
                rows, stats, controls = score_unit(selected, by_user, history, parent, current, cutover,
                    corrections=corrections, cost_model=cost, cohort_size=cohort, query_batch=query_batch,
                    verify=args.verify, append_band_size=config["append_band_size"])
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
            if set(grouped) != set(args.budgets) or any(len(ids) != len(expected_ids)
                    or set(ids) != expected_ids for ids in grouped.values()):
                raise RuntimeError("correction budget missing/duplicating evaluated requests")
            path = method_directory(method, args.scale, args.edge, output_root=args.output_root) / f"rank{args.rank}" / f"shard_{unit:05d}.parquet"
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
            "requests": sum(u["requests"] for u in units), "elapsed_seconds": time.perf_counter() - started,
            "compute_seconds": sum(u["seconds"] for u in units),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1 << 30),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1 << 30),
            "gpu_total_gib": total / (1 << 30), "batch_reductions": reductions}
        write_json(shared_dir / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    final = {"status": "complete", "source_signature": source_signature, "inputs": bound,
        "scale": args.scale, "edge": name, "rank": args.rank, "users": len(uids),
        "requests": sum(u["requests"] for u in units), "partition": args.partition,
        "evaluation_role": "development_exploration", "model_load_seconds": model_seconds,
        "history_load_seconds": history_seconds, "compute_seconds": sum(u["seconds"] for u in units),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1 << 30),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1 << 30),
        "gpu_total_gib": total / (1 << 30), "batch_reductions": reductions,
        "units": len(units), "cohort_size": cohort, "query_batch": query_batch}
    write_json(complete_path, final)
    print(json.dumps({k: v for k, v in final.items() if k != "inputs"}), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--partition", choices=("canary", "evaluation"), default="canary")
    parser.add_argument("--limit-users", type=int)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--calibration-root", type=Path, default=OUTPUT)
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--cohort-size", type=int)
    parser.add_argument("--query-batch", type=int)
    parser.add_argument("--unit-users", type=int)
    parser.add_argument("--verify", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
