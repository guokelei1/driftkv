#!/usr/bin/env python3
"""Small causal rolling probe for cross-head, per-token read-time correction."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
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
from hstu_kvcache.read_correction_v4.token_read import TokenReadCorrection, fit_affine_tokens, score_token_corrected
from hstu_kvcache.training import collate_foundation_batch
from read_correction_2026_09.calibrate import candidates
from read_correction_2026_09.cost import CostModel
from read_correction_2026_09.v2.calibrate import load_data
from read_correction_v3.evaluate import cost_record
from read_correction_v4.common import OUTPUT, PANEL_ROOT, RESERVATIONS, configuration, edge_name, sources, sha256, write_json
from read_correction_v4.cost import correction_forward, token_ridge_fit
from selective_recompute_2026_09.evaluate import all_snapshots, prefix_events
from selective_recompute_2026_09.scheduling import ordered_uids


def token_pairs(rows, uids, layer, cap):
    inputs, targets = [], []
    for uid in uids:
        row = rows[uid]
        n = row["parent"].seq_len
        index = torch.linspace(0, n - 1, min(n, cap)).long()
        x = torch.cat((row["parent"].k[layer, 0, index], row["parent"].v[layer, 0, index]), -1)
        y = torch.cat((row["teacher"].k[layer, 0, index], row["teacher"].v[layer, 0, index]), -1)
        inputs.append(x)
        targets.append(y - x)
    return torch.cat(inputs), torch.cat(targets)


def fit_modules(current, rows, train, validation, cfg, device, scale):
    modules, records = [], []
    sparse = CostModel.for_scale(scale, cfg["attention_backend"])
    dense = CostModel.for_scale(scale, "torch")
    ledger = {"parent_and_teacher_cache_flops": 2 * sum(
        (sparse if rows[u]["parent"].seq_len == cfg["history_length"] else dense).full_cache(rows[u]["parent"].seq_len)
        for u in train + validation), "token_ridge_fit_flops": 0, "teacher_residual_flops": 0,
        "validation_mapping_flops": 0}
    for layer, block in enumerate(current.blocks):
        x, y = token_pairs(rows, train, layer, cfg["tokens_per_user"])
        vx, vy = token_pairs(rows, validation, layer, cfg["tokens_per_user"])
        started = time.perf_counter()
        params, stats = fit_affine_tokens(x.to(device), y.to(device), ridge=cfg["ridge"])
        module = TokenReadCorrection(block.attn.num_heads, block.attn.head_dim).to(device)
        with torch.no_grad():
            for key, value in params.items():
                getattr(module, key).copy_(value)
            xn = (vx.to(device) - module.input_mean) / module.input_scale
            error = module.token_delta(xn) - vy.to(device)
            validation_mse = float(error.double().square().mean())
        module.eval().requires_grad_(False)
        modules.append(module)
        p = x.shape[1]
        ledger["token_ridge_fit_flops"] += token_ridge_fit(observations=len(x), input_width=p)
        ledger["teacher_residual_flops"] += (len(x) + len(vx)) * p
        ledger["validation_mapping_flops"] += len(vx) * (2*p*p + 6*p)
        records.append({"layer": layer, "train": stats, "validation_delta_mse": validation_mse,
            "validation_zero_delta_mse": float(vy.double().square().mean()),
            "seconds": time.perf_counter()-started})
        print(json.dumps({"status": "layer_fit", **records[-1]}), flush=True)
    ledger["calibration_flops"] = sum(ledger.values())
    ledger["convention"] = "analytical arithmetic, token ridge solve estimated; all teacher-cache preparation and fitting charged"
    return modules, records, ledger


@torch.inference_mode()
def evaluate(current, parent, history, by_user, uids, cutover, modules, cfg, scale, device,
             *, correction_cost=correction_forward):
    records = {"map_all": [], "map_old_prefix": []}
    stats = {key: Counter() for key in ("full_history_hist", "append_prefix_hist", "initial_history_hist",
        "torch_full_history_hist", "torch_append_prefix_hist", "band_append_hist", "torch_band_append_hist")}
    controls = {"requests": 0, "reuse_max_abs_logit_error": 0., "full_max_abs_logit_error": 0., "identity_max_abs_logit_error": 0.}
    full = [u for u in uids if len(prefix_events(history.rows[u], cutover, cfg["history_length"])) == cfg["history_length"]]
    full_set = set(full)
    size = cfg["cohort_sizes"][scale]
    cohorts = [full[i:i+size] for i in range(0, len(full), size)] + [[u] for u in uids if u not in full_set]
    cost = CostModel.for_scale(scale, cfg["attention_backend"])
    for snap, _ in all_snapshots(cohorts, by_user, history, parent, current, cutover,
        cfg["query_batches"][scale], stats, cost, cfg["append_band_size"]):
        cache = snap.state.cache
        n = cache.seq_len
        counts = torch.full((len(snap.requests),), n, device=device)
        old_counts = torch.tensor([max(0, n-int(r["append_count_since_cutover"])) for r in snap.requests], device=device)
        native = current.observe_cc_reuse(cache, snap.candidates, snap.query_deltas)[0][:, 0]
        identity = score_token_corrected(current, cache, snap.candidates, snap.query_deltas,
            [None] * len(current.blocks), counts)[0][:, 0]
        full_batch = collate_foundation_batch([{**r, "weight": r.get("weight", 1.)} for r in snap.requests],
            history, device=device, max_history=cfg["history_length"])
        full_scores = current.observe_cc_full(full_batch.item_ids, full_batch.behaviors, full_batch.time_deltas,
            full_batch.candidate_ids, full_batch.query_time_deltas, lengths=full_batch.lengths)[0][:, 0]
        for key, actual, expected in (
            ("reuse_max_abs_logit_error", native, native.new_tensor([r["reuse_logit"] for r in snap.requests])),
            ("full_max_abs_logit_error", full_scores, native.new_tensor([r["full_logit"] for r in snap.requests])),
            ("identity_max_abs_logit_error", identity, native)):
            controls[key] = max(controls[key], float((actual-expected).abs().max()))
        controls["requests"] += len(snap.requests)
        extra = sum(correction_cost(m.get_config(), n) for m in modules)
        for name, old in (("map_all", None), ("map_old_prefix", old_counts)):
            values = score_token_corrected(current, cache, snap.candidates, snap.query_deltas,
                modules, counts, old_counts=old)[0][:, 0].float().cpu().tolist()
            if not np.isfinite(values).all():
                raise RuntimeError("nonfinite token-map predictions")
            for request, value, inherited in zip(snap.requests, values, old_counts.tolist()):
                records[name].append({"request_id": request["request_id"], "uid": request["uid"],
                    "label": request["label"], "full_logit": request["full_logit"], "reuse_logit": request["reuse_logit"],
                    "hstu_logit": value, "correction_flops": extra, "history_length": n,
                    "inherited_count": inherited, "append_count_since_cutover": request["append_count_since_cutover"]})
    if any(value > 2e-5 for key, value in controls.items() if key.endswith("error")):
        raise RuntimeError(f"control mismatch: {controls}")
    return records, {key: dict(value) for key, value in stats.items()}, controls


def run(args, *, fitter=fit_modules, config=None, artifact_kind="token_read_affine",
        correction_cost=correction_forward):
    cfg = configuration() if config is None else config
    output = args.output_root / args.scale / edge_name(args.edge)
    if (output / "summary.json").exists():
        raise RuntimeError("probe already exists; retain it and use a new output root")
    output.mkdir(parents=True, exist_ok=True)
    binding_path = PANEL_ROOT / args.scale / edge_name(args.edge) / "binding.json"
    panel = json.loads(binding_path.read_text())
    started = time.perf_counter()
    source_hashes = sources()
    load_args = argparse.Namespace(scale=args.scale, edge=args.edge, gpu=args.gpu,
        panel_root=PANEL_ROOT, reservations=RESERVATIONS, budgets=[cfg["calibration_users"]+cfg["validation_users"]])
    current, rows, uids, device, metadata = load_data(load_args, cfg)
    train, validation = uids[:cfg["calibration_users"]], uids[cfg["calibration_users"]:]
    if cfg.get("candidate_mode") == "uniform_known":
        checkpoint = torch.load(ROOT/panel["sources"]["current"]["path"], map_location="cpu", weights_only=False, mmap=True)
        dataset = json.loads((ROOT/panel["sources"]["dataset"]["path"]).read_text())
        known = int(checkpoint.get("known_vocab_size", dataset["foundation_items"]))
        del checkpoint
        for uid in uids:
            rows[uid]["candidates"] = torch.from_numpy(candidates(uid, known, cfg["calibration_queries_per_user"]))
        metadata["candidate_rule"] = "uniform known catalog without replacement; SeedSequence([17,uid]); no feedback labels"
    modules, layers, ledger = fitter(current, rows, train, validation, cfg, device, args.scale)
    torch.save({"kind": artifact_kind, "modules": [{"config": m.get_config(),
        "state_dict": {k: v.cpu() for k,v in m.state_dict().items()}} for m in modules]}, output/"calibration.pt")
    write_json(output/"calibration.json", {"status": "complete", "settings": cfg, "uids": train,
        "validation_uids": validation, "execution_sources": source_hashes, "layers": layers, "cost": ledger,
        "weights_sha256": sha256(output/"calibration.pt"), **metadata})
    if getattr(args, "calibration_only", False):
        print(json.dumps({"status": "calibration_complete", "scale": args.scale,
            "edge": edge_name(args.edge), "output": str(output),
            "elapsed_seconds": time.perf_counter()-started}), flush=True)
        return
    del rows
    gc.collect(); torch.cuda.empty_cache()
    by_user = defaultdict(list)
    for row in pq.read_table(binding_path.parent/"evaluation_requests.parquet").to_pylist():
        by_user[int(row["uid"])].append(row)
    selected = sorted(by_user, key=lambda u: hashlib.sha256(f"read-correction-v2-probe:17:{u}".encode()).digest())[:128]
    selected = ordered_uids(by_user, max_length=cfg["history_length"], uids=selected)
    if set(train+validation).intersection(by_user):
        raise RuntimeError("fitting users overlap evaluation population")
    parent, payload = load_model(ROOT/panel["sources"]["parent"]["path"], device)
    del payload
    parent.requires_grad_(False)
    checkpoint = torch.load(ROOT/panel["sources"]["current"]["path"], map_location="cpu", weights_only=False, mmap=True)
    dataset_path = ROOT/panel["sources"]["dataset"]["path"]
    dataset = json.loads(dataset_path.read_text())
    known = int(checkpoint.get("known_vocab_size", dataset["foundation_items"]))
    del checkpoint
    history = load_histories(selected, dataset_path=dataset_path, known_vocab_size=known,
        oov_buckets=current.cfg.num_items-known, start_timestamp=int(panel["cutover"]),
        end_timestamp=int(panel["days"][1])*86400, max_history=cfg["history_length"], threads=cfg["history_threads"])
    inference_started = time.perf_counter()
    predictions, hist, controls = evaluate(current, parent, history, by_user, selected, int(panel["cutover"]),
        modules, cfg, args.scale, device, correction_cost=correction_cost)
    inference_seconds = time.perf_counter()-inference_started
    prior_path = ROOT/"results/read_correction_2026_09/runtime/development/v1"/args.scale/edge_name(args.edge)/"summary.json"
    prior = json.loads(prior_path.read_text())
    projected_inference = sum(int(count)*sum(correction_cost(m.get_config(), int(n)) for m in modules)
        for n,count in prior["histograms"]["full_history_hist"].items())
    projected = cost_record(args.scale, prior["histograms"], projected_inference, ledger["calibration_flops"])
    points = []
    for name, records in predictions.items():
        records.sort(key=lambda r: r["request_id"])
        labels = np.asarray([r["label"] for r in records])
        metrics = {key: binary_metrics(labels, np.asarray([r[field] for r in records])) for key,field in
            (("full","full_logit"),("reuse","reuse_logit"),("method","hstu_logit"))}
        full, reuse, auc = (metrics[k]["ROC_AUC"] for k in ("full","reuse","method"))
        pq.write_table(pa.Table.from_pylist(records), output/f"{name}.parquet", compression="zstd")
        point = {"variant": name, "full_auc": full, "reuse_auc": reuse, "auc": auc,
            "recovery_percent": 100*(auc-reuse)/(full-reuse), "requests": len(records),
            "projected_full_panel_cost": projected,
            "probe_cost": cost_record(args.scale, hist, sum(r["correction_flops"] for r in records), ledger["calibration_flops"])}
        points.append(point)
    result = {"status": "complete", "role": "small development probe", "scale": args.scale,
        "edge": edge_name(args.edge), "users": len(selected), "uids": selected, "settings": cfg,
        "execution_sources": source_hashes, "controls": controls, "points": points,
        "scoring_seconds": inference_seconds, "elapsed_seconds": time.perf_counter()-started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/2**30,
        "gpu_total_gib": torch.cuda.get_device_properties(device).total_memory/2**30}
    write_json(output/"summary.json", result)
    print(json.dumps({k: result[k] for k in ("status","scale","edge","controls","points","elapsed_seconds","peak_reserved_gib")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1,6), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT/"affine_probe")
    run(parser.parse_args())
