#!/usr/bin/env python3
"""Complete-panel v4 evaluation with atomic user-unit continuation.

Calibration is an explicit input: its historical source signature stays
retained independently of this evaluator.
"""
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
from hstu_kvcache.read_correction_v4.token_read import TokenReadCorrection, score_token_corrected
from hstu_kvcache.read_correction_v4.nonlinear import NonlinearTokenReadCorrection
from hstu_kvcache.training import collate_foundation_batch
from read_correction_v3.evaluate import cost_record
from read_correction_v4.common import PANEL_ROOT, edge_name, sources, sha256, write_json
from read_correction_v4.cost import CostModel, correction_forward as affine_cost, normalize_metrics
from read_correction_v4.cost_nonlinear import correction_forward as nonlinear_cost
from selective_recompute_2026_09.evaluate import all_snapshots, prefix_events
from selective_recompute_2026_09.scheduling import ordered_uids

VARIANTS = ("map_all", "map_old_prefix")
HISTOGRAMS = ("full_history_hist", "append_prefix_hist", "initial_history_hist",
    "torch_full_history_hist", "torch_append_prefix_hist", "band_append_hist", "torch_band_append_hist")


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_modules(artifact, device):
    builders = {"token_read_affine": (TokenReadCorrection, affine_cost),
                "token_read_nonlinear_v4": (NonlinearTokenReadCorrection, nonlinear_cost)}
    cls, cost_fn = builders[artifact["kind"]]
    modules = []
    for row in artifact["modules"]:
        module = cls(**row["config"]).to(device)
        module.load_state_dict(row["state_dict"])
        modules.append(module.eval().requires_grad_(False))
    return modules, cost_fn


def validate_rows(rows, reference):
    rows = sorted(rows, key=lambda row: row["request_id"])
    if len(rows) != len(reference):
        raise RuntimeError("unit request count differs from the frozen panel")
    for row, expected in zip(rows, reference, strict=True):
        for key in ("request_id", "uid", "label", "query_timestamp", "full_logit", "reuse_logit"):
            if row[key] != expected[key]:
                raise RuntimeError(f"unit request differs from the frozen panel in {key}")
        if not np.isfinite(row["hstu_logit"]):
            raise RuntimeError("nonfinite v4 prediction")
    return rows


def verify_saved_unit(record, binding, selected, variants):
    if record["input_signature"] != binding or record["uids"] != selected or set(record["outputs"]) != set(variants):
        raise RuntimeError("saved v4 unit has different inputs, users or variants")
    for output in record["outputs"].values():
        if sha256(output["path"]) != output["sha256"]:
            raise RuntimeError("saved v4 unit scores changed")


@torch.inference_mode()
def score_unit(current, parent, history, by_user, uids, cutover, modules, cfg,
               scale, device, *, variants, correction_cost, verify):
    records = {name: [] for name in variants}
    stats = {key: Counter() for key in HISTOGRAMS}
    controls = {"requests": 0, "reuse_max_abs_logit_error": 0.,
                "full_max_abs_logit_error": 0., "identity_max_abs_logit_error": 0.}
    full = [uid for uid in uids if len(prefix_events(history.rows[uid], cutover, cfg["history_length"])) == cfg["history_length"]]
    full_set = set(full)
    cohort_size = cfg["cohort_sizes"][scale]
    cohorts = [full[i:i + cohort_size] for i in range(0, len(full), cohort_size)] + [[uid] for uid in uids if uid not in full_set]
    cost = CostModel.for_scale(scale, cfg["attention_backend"])
    for snap, _ in all_snapshots(cohorts, by_user, history, parent, current, cutover,
        cfg["query_batches"][scale], stats, cost, cfg["append_band_size"]):
        cache, requests = snap.state.cache, snap.requests
        n = cache.seq_len
        counts = torch.full((len(requests),), n, device=device)
        inherited = [max(0, n - int(row["append_count_since_cutover"])) for row in requests]
        old_counts = torch.tensor(inherited, device=device)
        if verify:
            native = current.observe_cc_reuse(cache, snap.candidates, snap.query_deltas)[0][:, 0]
            identity = score_token_corrected(current, cache, snap.candidates, snap.query_deltas,
                [None] * len(current.blocks), counts)[0][:, 0]
            full_batch = collate_foundation_batch([{**row, "weight": row.get("weight", 1.)} for row in requests],
                history, device=device, max_history=cfg["history_length"])
            full_scores = current.observe_cc_full(full_batch.item_ids, full_batch.behaviors, full_batch.time_deltas,
                full_batch.candidate_ids, full_batch.query_time_deltas, lengths=full_batch.lengths)[0][:, 0]
            for key, actual, expected in (
                ("reuse_max_abs_logit_error", native, native.new_tensor([row["reuse_logit"] for row in requests])),
                ("full_max_abs_logit_error", full_scores, native.new_tensor([row["full_logit"] for row in requests])),
                ("identity_max_abs_logit_error", identity, native)):
                controls[key] = max(controls[key], float((actual - expected).abs().max()))
            controls["requests"] += len(requests)
        extra = sum(correction_cost(module.get_config(), n) for module in modules)
        for name in variants:
            old = old_counts if name == "map_old_prefix" else None
            values = score_token_corrected(current, cache, snap.candidates, snap.query_deltas,
                modules, counts, old_counts=old)[0][:, 0].float().cpu().tolist()
            for row, value, count in zip(requests, values, inherited, strict=True):
                records[name].append({key: row[key] for key in (
                    "request_id", "uid", "label", "query_timestamp", "full_logit", "reuse_logit")}
                    | {"hstu_logit": value, "correction_flops": extra, "history_length": n,
                       "inherited_count": count, "append_count_since_cutover": row["append_count_since_cutover"]})
    if any(value > 2e-5 for key, value in controls.items() if key.endswith("error")):
        raise RuntimeError(f"v4 numerical control mismatch: {controls}")
    return records, {key: {str(k): int(v) for k, v in value.items()} for key, value in stats.items()}, controls


def run(args):
    output = args.output.resolve()
    variants = tuple(args.variants)
    panel_path = args.panel_root / args.scale / edge_name(args.edge) / "binding.json"
    panel = json.loads(panel_path.read_text())
    requests_path = panel_path.parent / "evaluation_requests.parquet"
    if sha256(requests_path) != panel["files"]["evaluation_requests"]["sha256"]:
        raise RuntimeError("frozen evaluation requests changed")
    cal_path, weights_path = args.calibration_dir / "calibration.json", args.calibration_dir / "calibration.pt"
    calibration = json.loads(cal_path.read_text())
    if calibration["status"] != "complete" or sha256(weights_path) != calibration["weights_sha256"]:
        raise RuntimeError("calibration is unfinished or its weights changed")
    if calibration["panel_binding_sha256"] != sha256(panel_path):
        raise RuntimeError("calibration and evaluation use different panels")
    cfg = json.loads(json.dumps(calibration["settings"]))
    fitting = calibration["uids"] + calibration["validation_uids"]
    if len(set(fitting)) != len(fitting):
        raise RuntimeError("fitting and validation users overlap")
    fit_keys = ("calibration_users", "validation_users", "tokens_per_user", "calibration_queries_per_user",
        "ridge", "seed", "history_length", "attention_backend", "candidate_mode", "nonlinear_epochs",
        "nonlinear_learning_rate", "weight_decay", "nonlinear_width_rule", "nonlinear_width", "fit_objective", "selection",
        "query_compensation", "query_ridge")
    fit_settings = {key: cfg[key] for key in fit_keys if key in cfg}
    by_user = defaultdict(list)
    for row in pq.read_table(requests_path).to_pylist():
        by_user[int(row["uid"])].append(row)
    if len(by_user) != 3000 or set(fitting).intersection(by_user):
        raise RuntimeError("the fixed panel must contain 3000 users disjoint from fitting/validation")
    selected = sorted(by_user, key=lambda uid: hashlib.sha256(f"read-correction-v2-probe:17:{uid}".encode()).digest())
    if args.limit_users:
        selected = selected[:args.limit_users]
    uids = ordered_uids(by_user, max_length=cfg["history_length"], uids=selected)
    if args.cohort_size:
        cfg["cohort_sizes"][args.scale] = args.cohort_size
    if args.query_batch:
        cfg["query_batches"][args.scale] = args.query_batch
    inputs = {"execution_sources": sources(), "settings": json.loads(json.dumps(cfg)), "fit_signature": signature(fit_settings),
        "calibration_sources": calibration["execution_sources"], "calibration_settings": fit_settings,
        "calibration": {"path": str(cal_path.resolve()), "sha256": sha256(cal_path), "weights_sha256": sha256(weights_path)},
        "panel_sha256": sha256(panel_path), "requests_sha256": sha256(requests_path), "uids": uids,
        "unit_users": args.unit_users, "variants": list(variants), "verification": "first unit Full/Reuse/identity"}
    binding = signature(inputs)
    units = []
    for index, start in enumerate(range(0, len(uids), args.unit_users)):
        path = output / "units" / f"unit_{index:05d}.json"
        record = json.loads(path.read_text()) if path.exists() else None
        if record is not None:
            verify_saved_unit(record, binding, uids[start:start + args.unit_users], variants)
        units.append(record)
    # Keep declared inputs immutable while carrying observed safe batches into
    # the next process; operational OOM reductions do not change method FLOPs.
    for unit in units:
        if unit is not None:
            cfg["cohort_sizes"][args.scale] = min(cfg["cohort_sizes"][args.scale], unit["cohort_size"])
            cfg["query_batches"][args.scale] = min(cfg["query_batches"][args.scale], unit["query_batch"])
    summary_path = output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous["input_signature"] != binding or any(unit is None for unit in units):
            raise RuntimeError("completed v4 evaluation has different inputs or missing units")
        for record in previous["scores"].values():
            if sha256(record["path"]) != record["sha256"]:
                raise RuntimeError("completed combined scores changed")
        print(json.dumps({"status": "already_complete", "output": str(summary_path)}), flush=True)
        return previous
    write_json(output / "inputs.json", dict(inputs, input_signature=binding))
    started = time.perf_counter()
    reductions = []
    if any(unit is None for unit in units):
        os.environ["EVOKV_ATTENTION_BACKEND"] = cfg["attention_backend"]
        torch.set_num_threads(cfg["torch_threads"])
        pa.set_cpu_count(cfg["history_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        free, total = torch.cuda.mem_get_info(device)
        while free / total < cfg["initial_free_fraction"]:
            print(json.dumps({"status": "waiting_for_memory", "gpu": args.gpu, "free_fraction": free / total}), flush=True)
            time.sleep(30)
            free, total = torch.cuda.mem_get_info(device)
        torch.cuda.set_per_process_memory_fraction(cfg["memory_fraction"], device)
        torch.cuda.reset_peak_memory_stats(device)
        parent, parent_payload = load_model(ROOT / panel["sources"]["parent"]["path"], device)
        current, current_payload = load_model(ROOT / panel["sources"]["current"]["path"], device)
        if parent_payload["config"] != current_payload["config"]:
            raise RuntimeError("parent and current architectures differ")
        dataset_path = ROOT / panel["sources"]["dataset"]["path"]
        dataset = json.loads(dataset_path.read_text())
        known = int(current_payload.get("known_vocab_size", dataset["foundation_items"]))
        parent.requires_grad_(False); current.requires_grad_(False)
        artifact = torch.load(weights_path, map_location="cpu", weights_only=False)
        modules, cost_fn = build_modules(artifact, device)
        if len(modules) != len(current.blocks):
            raise RuntimeError("correction and model layer counts differ")
        del parent_payload, current_payload, artifact
        remaining = [uid for index, start in enumerate(range(0, len(uids), args.unit_users)) if units[index] is None
                     for uid in uids[start:start + args.unit_users]]
        history = load_histories(remaining, dataset_path=dataset_path, known_vocab_size=known,
            oov_buckets=current.cfg.num_items - known, start_timestamp=int(panel["cutover"]),
            end_timestamp=int(panel["days"][1]) * 86400, max_history=cfg["history_length"], threads=cfg["history_threads"])
        for index, start in enumerate(range(0, len(uids), args.unit_users)):
            if units[index] is not None:
                continue
            selected = uids[start:start + args.unit_users]
            reference = sorted([row for uid in selected for row in by_user[uid]], key=lambda row: row["request_id"])
            beginning = time.perf_counter()
            reduction_start = len(reductions)
            while True:
                try:
                    scores, histograms, controls = score_unit(current, parent, history, by_user, selected,
                        int(panel["cutover"]), modules, cfg, args.scale, device, variants=variants,
                        correction_cost=cost_fn, verify=index == 0)
                    break
                except torch.cuda.OutOfMemoryError:
                    cohort, queries = cfg["cohort_sizes"][args.scale], cfg["query_batches"][args.scale]
                    if cohort == queries == 1:
                        raise
                    reductions.append({"unit": index, "cohort_size": cohort, "query_batch": queries})
                    cfg["cohort_sizes"][args.scale], cfg["query_batches"][args.scale] = max(1, cohort // 2), max(1, queries // 2)
                    gc.collect(); torch.cuda.empty_cache()
            paths = {}
            for name in variants:
                rows = validate_rows(scores[name], reference)
                path = output / "units" / f"unit_{index:05d}.{name}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".parquet.partial")
                pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
                temporary.replace(path)
                paths[name] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}
            record = {"input_signature": binding, "uids": selected, "requests": len(reference), "outputs": paths,
                "histograms": histograms, "controls": controls, "verified": index == 0,
                "seconds": time.perf_counter() - beginning, "batch_reductions": reductions[reduction_start:],
                "cohort_size": cfg["cohort_sizes"][args.scale], "query_batch": cfg["query_batches"][args.scale],
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
            write_json(output / "units" / f"unit_{index:05d}.json", record)
            units[index] = record
            print(json.dumps({"status": "scoring", "scale": args.scale, "edge": edge_name(args.edge),
                "users": start + len(selected), "total_users": len(uids), "unit_seconds": record["seconds"]}), flush=True)
    histograms = {key: Counter() for key in HISTOGRAMS}
    controls = {key: 0 for key in units[0]["controls"]}
    for unit in units:
        for key in HISTOGRAMS:
            histograms[key].update(unit["histograms"][key])
        for key, value in unit["controls"].items():
            controls[key] = controls[key] + value if key == "requests" else max(controls[key], value)
    histograms = {key: dict(value) for key, value in histograms.items()}
    reference = sorted([row for uid in uids for row in by_user[uid]], key=lambda row: row["request_id"])
    labels = np.asarray([row["label"] for row in reference])
    full = binary_metrics(labels, np.asarray([row["full_logit"] for row in reference]))
    reuse = binary_metrics(labels, np.asarray([row["reuse_logit"] for row in reference]))
    points, combined = [], {}
    for name in variants:
        table = pa.concat_tables([pq.read_table(unit["outputs"][name]["path"]) for unit in units]).sort_by([("request_id", "ascending")])
        rows = validate_rows(table.to_pylist(), reference)
        measured = binary_metrics(labels, table["hstu_logit"].to_numpy())
        ledger = cost_record(args.scale, histograms, sum(row["correction_flops"] for row in rows),
                             calibration["cost"]["calibration_flops"])
        normalized = normalize_metrics(full_auc=full["ROC_AUC"], reuse_auc=reuse["ROC_AUC"],
            baseline_auc=measured["ROC_AUC"], extra_flops=ledger["extra_flops"], full_minus_reuse_flops=ledger["full_minus_reuse_flops"])
        path = output / f"{name}.parquet"
        pq.write_table(table, path, compression="zstd")
        combined[name] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}
        points.append({"variant": name, "method": "history_conditioned", "baseline": "history_conditioned",
            "scale": args.scale, "edge": edge_name(args.edge), "kind": "measurement", "budget": name,
            "calibration_users": len(calibration["uids"]), "users": len(uids), "requests": len(rows),
            "full_auc": full["ROC_AUC"], "reuse_auc": reuse["ROC_AUC"], "baseline_auc": measured["ROC_AUC"],
            "metrics": measured, "partition": "evaluation", "evaluation_role": "development_exploration",
            **ledger, **normalized})
    summary = {"status": "complete", "revision": "v4", "evaluation_role": "development_exploration",
        "probe_only": len(uids) != len(by_user), "scale": args.scale, "edge": edge_name(args.edge),
        "users": len(uids), "requests": len(reference), "partition": "evaluation", "variants": list(variants),
        "points": points, "current_full": full, "current_reuse": reuse, "inputs": inputs,
        "input_signature": binding, "scores": combined, "histograms": histograms, "controls": controls,
        "calibration_cost_scope": "original complete fit charged once per reported variant; reuse of an existing fitted artifact does not erase its calibration cost",
        "scoring_seconds": sum(unit["seconds"] for unit in units), "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": max(unit["peak_allocated_gib"] for unit in units),
        "peak_reserved_gib": max(unit["peak_reserved_gib"] for unit in units),
        "batch_reductions": [item for unit in units for item in unit["batch_reductions"]]}
    write_json(summary_path, summary)
    print(json.dumps({"status": "complete", "output": str(summary_path), "users": len(uids),
        "points": [{key: point[key] for key in ("variant", "baseline_auc", "recovery_percent", "relative_flops_percent")} for point in points]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--unit-users", type=int, default=256)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--cohort-size", type=int)
    parser.add_argument("--query-batch", type=int)
    run(parser.parse_args())
