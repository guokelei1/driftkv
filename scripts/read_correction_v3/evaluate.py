#!/usr/bin/env python3
"""Resumable v3 H evaluation; fixed hash probes or the complete retained panel."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from evaluate_yambda500m_foundation_raw import load_histories, load_model
from hstu_kvcache.evaluation.binary_metrics import binary_metrics
from hstu_kvcache.read_correction import build_correction
from read_correction_2026_09.cost import CostModel, eager_read, normalize_metrics
from read_correction_2026_09.v2.worker import score_unit
from read_correction_v3.common import (
    OUTPUT, PANEL_ROOT, edge_name, method_directory, plan, sha256, sources, write_json,
)
from selective_recompute_2026_09.common import signature
from selective_recompute_2026_09.scheduling import ordered_uids

METHOD = "history_conditioned"


def cost_record(scale, hist, inference, calibration, *, history_length=1024):
    """Expose denominators without choosing a new normalization convention.

    The canonical eager query read is common to all policies. Parent cache
    initialization is shown separately because steady serving inherits it.
    Calibration is charged once, including the inherited v1 fit.
    """
    cost, dense = CostModel.for_scale(scale), CostModel.for_scale(scale, "torch")
    result = cost.workload_denominator(hist["full_history_hist"], hist["append_prefix_hist"],
        append_new_kv_only=False, torch_full_history_hist=hist["torch_full_history_hist"],
        torch_append_prefix_hist=hist["torch_append_prefix_hist"],
        band_append_hist=hist["band_append_hist"], torch_band_append_hist=hist["torch_band_append_hist"])
    query = sum(int(count)*eager_read(cost, int(n)) for n, count in hist["full_history_hist"].items())
    initial = sum(int(count)*(cost if int(n) == history_length else dense).full_cache(int(n))
                  for n, count in hist["initial_history_hist"].items())
    steady_reuse = result["reuse_append_flops"] + query
    steady_full = result["full_history_flops"] + query
    extra = int(inference) + int(calibration)
    ratio = lambda n, d: 100*n/d if d else None
    return {**result, "common_eager_query_flops": query,
        "initial_parent_cache_flops": initial, "reuse_steady_flops": steady_reuse,
        "reuse_with_initial_cache_flops": steady_reuse + initial,
        "full_history_plus_query_flops": steady_full,
        "correction_flops": int(inference), "calibration_flops": int(calibration),
        "extra_flops": extra, "method_steady_including_calibration_flops": steady_reuse + extra,
        "relative_flops_percent": ratio(extra, result["full_minus_reuse_flops"]),
        "extra_over_reuse_steady_percent": ratio(extra, steady_reuse),
        "extra_over_reuse_with_initial_cache_percent": ratio(extra, steady_reuse + initial),
        "method_over_reuse_steady_percent": ratio(steady_reuse + extra, steady_reuse),
        "method_over_full_history_plus_query_percent": ratio(steady_reuse + extra, steady_full)}


def projected_cost(scale, edge, budget, calibration, configs):
    """Exact old-panel inference arithmetic plus the new measured fit ledger."""
    old = ROOT / "results/read_correction_2026_09"
    directory = old / "history_conditioned/development/v1" / scale / edge_name(edge)
    old_calibration = directory / f"calibration_c{budget}.json"
    old_cal = json.loads(old_calibration.read_text())
    if configs != [layer["config"] for layer in old_cal["layers"]]:
        raise RuntimeError("v3 structure differs; old H inference cost cannot be reused")
    prior_path = old / "runtime/development/v1" / scale / edge_name(edge) / "summary.json"
    prior = json.loads(prior_path.read_text())
    point = next(p for p in prior["points"] if p["method"] == METHOD and p["budget"] == budget)
    return {"scope": "cost projection on the unchanged complete 3000-user panel; no projected quality",
        "users": point["users"], "requests": point["requests"],
        "cost": cost_record(scale, prior["histograms"], point["correction_flops"], calibration),
        "source": {"path": str(prior_path), "sha256": sha256(prior_path)},
        "structure_source": {"path": str(old_calibration), "sha256": sha256(old_calibration)}}


def check_unit(record, binding, uids):
    if record["input_signature"] != binding or record["uids"] != uids:
        raise RuntimeError("saved unit uses different sources, calibration or users")
    if sha256(Path(record["scores"]["path"])) != record["scores"]["sha256"]:
        raise RuntimeError("saved score unit changed")


def run(args):
    config = plan()
    panel_path = args.panel_root / args.scale / edge_name(args.edge) / "binding.json"
    panel = json.loads(panel_path.read_text())
    request_path = panel_path.parent / f"{args.partition}_requests.parquet"
    if sha256(request_path) != panel["files"][f"{args.partition}_requests"]["sha256"]:
        raise RuntimeError("frozen request panel changed")
    by_user = defaultdict(list)
    for row in pq.read_table(request_path).to_pylist():
        by_user[int(row["uid"])].append(row)
    selected = sorted(by_user, key=lambda uid: hashlib.sha256(
        f"read-correction-v2-probe:{config['seed']}:{uid}".encode()).digest())
    if args.limit_users:
        selected = selected[:args.limit_users]
    uids = ordered_uids(by_user, max_length=config["history_length"], uids=selected)
    cal_path = method_directory(args.calibration_root, args.scale, args.edge) / f"calibration_c{args.budget}.json"
    weights_path = cal_path.with_suffix(".pt")
    calibration = json.loads(cal_path.read_text())
    if (calibration["status"], calibration["kind"], calibration["scale"], calibration["edge"], calibration["users"]) != (
            "complete", METHOD, args.scale, edge_name(args.edge), args.budget):
        raise RuntimeError("calibration method/edge/budget differs")
    if calibration["panel_binding_sha256"] != sha256(panel_path) or calibration["weights_sha256"] != sha256(weights_path):
        raise RuntimeError("calibration panel/weights binding differs")
    fit_uids = set(calibration["uids"]) | set(calibration["fit"]["validation_uids"])
    if fit_uids.intersection(by_user):
        raise RuntimeError("fit/validation users overlap the scored partition")
    unit_users = int(config["unit_users"])
    inputs = {"execution_sources": sources(), "panel_binding_sha256": sha256(panel_path),
        "requests_sha256": sha256(request_path), "calibration_sha256": sha256(cal_path),
        "weights_sha256": sha256(weights_path), "calibration_path": str(cal_path),
        "uid_selection": "smallest sha256(read-correction-v2-probe:seed:uid); same v2 probe membership",
        "seed": config["seed"], "uids": uids, "verify": args.verify,
        "partition": args.partition, "unit_users": unit_users}
    binding = signature(inputs)
    units = []
    for index, offset in enumerate(range(0, len(uids), unit_users)):
        path = args.output / "units" / f"unit_{index:05d}.json"
        if path.exists():
            record = json.loads(path.read_text())
            check_unit(record, binding, uids[offset:offset + unit_users])
            units.append(record)
        else:
            units.append(None)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous["input_signature"] != binding or any(unit is None for unit in units):
            raise RuntimeError("completed evaluation inputs or units changed")
        if sha256(Path(previous["scores"]["path"])) != previous["scores"]["sha256"]:
            raise RuntimeError("completed combined scores changed")
        print(json.dumps({"status": "already_complete", "output": str(summary_path)}), flush=True)
        return previous
    write_json(args.output / "users.json", {"uids": uids, "selection": inputs["uid_selection"], "seed": config["seed"]})
    started = time.perf_counter()
    model_seconds = history_seconds = 0.
    reductions = []
    if any(unit is None for unit in units):
        os.environ["EVOKV_ATTENTION_BACKEND"] = config["attention_backend"]
        torch.set_num_threads(config["torch_threads"])
        pa.set_cpu_count(config["history_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        free, total = torch.cuda.mem_get_info(device)
        if free / total < config["initial_free_fraction"]:
            raise RuntimeError("GPU lacks the configured initial free-memory reserve")
        torch.cuda.set_per_process_memory_fraction(config["memory_fraction"], device)
        torch.cuda.reset_peak_memory_stats(device)
        source = panel["sources"]
        parent, pp = load_model(ROOT / source["parent"]["path"], device)
        current, cp = load_model(ROOT / source["current"]["path"], device)
        if pp["config"] != cp["config"]:
            raise RuntimeError("parent/current architecture differs")
        artifact = torch.load(weights_path, map_location="cpu", weights_only=False)
        if artifact["kind"] != METHOD or len(artifact["modules"]) != len(current.blocks):
            raise RuntimeError("correction architecture differs")
        modules = [build_correction(METHOD, item["config"], item["state_dict"]).to(
            device=device, dtype=next(current.parameters()).dtype).eval() for item in artifact["modules"]]
        parent.requires_grad_(False); current.requires_grad_(False)
        dataset_path = ROOT / source["dataset"]["path"]
        dataset = json.loads(dataset_path.read_text())
        known = int(cp.get("known_vocab_size", dataset["foundation_items"]))
        oov = int(cp["config"]["num_items"]) - known
        del pp, cp, artifact
        model_seconds = time.perf_counter() - started
        history = load_histories(uids, dataset_path=dataset_path, known_vocab_size=known, oov_buckets=oov,
            start_timestamp=int(panel["cutover"]), end_timestamp=int(panel["days"][1])*86400,
            max_history=config["history_length"], threads=config["history_threads"])
        history_seconds = time.perf_counter() - started - model_seconds
        cost = CostModel.for_scale(args.scale, config["attention_backend"])
        cohort, query_batch = config["cohort_sizes"][args.scale], config["query_batches"][args.scale]
        for index, offset in enumerate(range(0, len(uids), unit_users)):
            if units[index] is not None:
                continue
            selected = uids[offset:offset + unit_users]
            begin = time.perf_counter()
            while True:
                try:
                    rows, hist, controls = score_unit(selected, by_user, history, parent, current, int(panel["cutover"]),
                        corrections={METHOD: {args.budget: modules}}, cost_model=cost,
                        cohort_size=cohort, query_batch=query_batch, verify=args.verify,
                        append_band_size=config["append_band_size"])
                    break
                except torch.cuda.OutOfMemoryError:
                    if cohort == 1 and query_batch == 1:
                        raise
                    reductions.append({"unit": index, "cohort": cohort, "query_batch": query_batch})
                    cohort, query_batch = max(1, cohort//2), max(1, query_batch//2)
                    gc.collect(); torch.cuda.empty_cache()
            scores = rows[METHOD]
            expected = sorted(r["request_id"] for uid in selected for r in by_user[uid])
            if sorted(r["request_id"] for r in scores) != expected or len(set(expected)) != len(expected):
                raise RuntimeError("unit scores do not match the retained requests")
            path = args.output / "units" / f"unit_{index:05d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".parquet.partial")
            pq.write_table(pa.Table.from_pylist(scores), temporary, compression="zstd")
            temporary.replace(path)
            record = {"input_signature": binding, "uids": selected, "histograms": hist,
                "controls": controls, "seconds": time.perf_counter()-begin,
                "scores": {"path": str(path), "sha256": sha256(path)},
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/2**30,
                "gpu_total_gib": total/2**30, "batch_reductions": list(reductions)}
            write_json(path.with_suffix(".json"), record)
            units[index] = record
            print(json.dumps({"status": "scoring", "scale": args.scale, "edge": edge_name(args.edge),
                "budget": args.budget, "users": offset+len(selected), "total_users": len(uids),
                "unit_seconds": record["seconds"]}), flush=True)
    scores = sorted([row for unit in units for row in pq.read_table(unit["scores"]["path"]).to_pylist()],
                    key=lambda row: row["request_id"])
    reference = sorted([row for uid in uids for row in by_user[uid]], key=lambda row: row["request_id"])
    if [row["request_id"] for row in scores] != [row["request_id"] for row in reference]:
        raise RuntimeError("combined scores do not match retained requests")
    hist = {key: Counter() for key in units[0]["histograms"]}
    controls = {key: 0 for key in units[0]["controls"]}
    for unit in units:
        for key, value in unit["histograms"].items():
            hist[key].update(value)
        for key, value in unit["controls"].items():
            controls[key] = controls[key]+value if key == "requests" else max(controls[key], value)
    hist = {key: dict(value) for key, value in hist.items()}
    labels = np.asarray([r["label"] for r in reference])
    full = binary_metrics(labels, np.asarray([r["full_logit"] for r in reference]))
    reuse = binary_metrics(labels, np.asarray([r["reuse_logit"] for r in reference]))
    measured = binary_metrics(labels, np.asarray([r["hstu_logit"] for r in scores]))
    ledger = cost_record(args.scale, hist, sum(r["correction_flops"] for r in scores),
                         calibration["cost"]["calibration_flops"], history_length=config["history_length"])
    artifact = torch.load(weights_path, map_location="cpu", weights_only=False)
    projection = projected_cost(args.scale, args.edge, args.budget, ledger["calibration_flops"],
                                [item["config"] for item in artifact["modules"]])
    metrics = normalize_metrics(full_auc=full["ROC_AUC"], reuse_auc=reuse["ROC_AUC"],
        baseline_auc=measured["ROC_AUC"], extra_flops=ledger["extra_flops"],
        full_minus_reuse_flops=ledger["full_minus_reuse_flops"])
    point = {"method": METHOD, "baseline": METHOD, "scale": args.scale,
        "edge": edge_name(args.edge), "budget": args.budget, "kind": "measurement",
        "users": len(uids), "requests": len(scores), "full_auc": full["ROC_AUC"],
        "reuse_auc": reuse["ROC_AUC"], "baseline_auc": measured["ROC_AUC"],
        "evaluation_role": "development_exploration", "partition": args.partition,
        "metrics": measured, **ledger, **metrics}
    scores_path = args.output / "scores.parquet"
    pq.write_table(pa.Table.from_pylist(scores), scores_path, compression="zstd")
    report = {"status": "complete", "revision": "v3", "evaluation_role": "development_exploration",
        "probe_only": len(uids) != len(by_user), "method": METHOD, "scale": args.scale,
        "edge": edge_name(args.edge), "budget": args.budget, "partition": args.partition,
        "users": len(uids), "requests": len(scores), "current_full": full, "current_reuse": reuse,
        "measured": measured, "points": [point], "cost": ledger, "projected_full_panel_cost": projection,
        "cost_scope": "complete inherited and new calibration charged once to scored requests; full-panel projection separate",
        "histograms": hist, "controls": controls, "inputs": inputs, "input_signature": binding,
        "scores": {"path": str(scores_path), "sha256": sha256(scores_path)},
        "model_load_seconds": model_seconds, "history_load_seconds": history_seconds,
        "scoring_seconds": sum(unit["seconds"] for unit in units), "elapsed_seconds": time.perf_counter()-started,
        "peak_allocated_gib": max(unit["peak_allocated_gib"] for unit in units),
        "peak_reserved_gib": max(unit["peak_reserved_gib"] for unit in units),
        "gpu_total_gib": max(unit["gpu_total_gib"] for unit in units), "batch_reductions": reductions}
    write_json(summary_path, report)
    print(json.dumps({key: report[key] for key in ("status", "scale", "edge", "budget", "users", "requests",
        "current_full", "current_reuse", "measured", "controls", "scoring_seconds", "peak_allocated_gib")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=(32, 128, 512), default=128)
    parser.add_argument("--partition", choices=("canary", "evaluation"), default="evaluation")
    parser.add_argument("--limit-users", type=int, default=128, help="0 evaluates the entire retained panel")
    parser.add_argument("--calibration-root", type=Path, default=OUTPUT)
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    run(parser.parse_args())
