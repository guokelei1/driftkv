#!/usr/bin/env python3
"""Shared read corrections on real feedback, with paired request-local prefixes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from design import bias_read_probe, query_read_probe
from design.competitor_data import history_arrays, make_candidate_panel
from design.competitor_models import load_model_pair, sha256_file
from design.run_shared_read_probe import (
    cache_slice, clock, collect_cache, device_parameters, reader_canary, write_json,
)
from evaluate_yambda500m_foundation_raw import load_histories, load_model
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.evaluation.binary_metrics import binary_metrics

NEW_METHODS = ("shared_bias", "shared_bias_response")
OLD_METHODS = ("shared_query", "shared_query_response")
PATHS = ("reuse", "exact", *NEW_METHODS, *OLD_METHODS)
REQUEST_ROOT = ROOT / "data/manifests/yambda500m_medium_hstu_native_d7_d14_v1"


def requests(uids, cutover):
    """Use the frozen real-feedback candidate panel without reading its labels."""
    return pq.read_table(REQUEST_ROOT / "requests_fidelity.parquet", filters=[
        ("uid", "in", uids), ("time_block", "=", "matrix_horizon"),
        ("target_known", "=", True), ("query_timestamp", ">=", cutover),
        ("query_timestamp", "<", cutover + 14 * 86400),
    ], columns=["request_id", "uid", "query_timestamp", "item_idx"]).to_pandas().sort_values(
        ["uid", "query_timestamp", "item_idx", "request_id"]).reset_index(drop=True)


def fit_rule(current, source, teacher, panel, deltas, use_response, batch_size, device):
    users, length = source.k.shape[1], source.seq_len
    counts = torch.full((users,), float(length), device=device)
    parameters, records = [], []
    for layer, block in enumerate(current.blocks):
        targets, observations = [], []
        installed = device_parameters(parameters, device)
        for start in range(0, users, batch_size):
            stop = min(start + batch_size, users)
            src, dst = cache_slice(source, start, stop), cache_slice(teacher, start, stop)
            override = bias_read_probe.make_history_override(installed, counts[start:stop])
            _, trace = score(current, src, panel[start:stop], deltas[start:stop],
                             trace=True, history_override=override)
            q, native = trace.queries[layer], trace.history_heads[layer]
            target = history_read(block.attn, q, dst.k[layer], dst.v[layer])
            targets.append(((target - native) / length).cpu())
            observations.append((native.transpose(1, 2).flatten(2) / length).cpu())
        p, record = bias_read_probe.fit_layer(torch.cat(targets), torch.cat(observations),
                                              counts.cpu(), use_response)
        parameters.append(p)
        records.append(record)
        print(f"  fit {'b+Tr' if use_response else 'b'} layer {layer+1}", flush=True)
    return parameters, records


def evaluate(parent, current, history, rows, rules, batch_size, device, max_length):
    """Isolate read compatibility at each request's own strictly causal prefix."""
    groups = defaultdict(list)
    for i, row in enumerate(rows.itertuples(index=False)):
        timestamps = history.rows[int(row.uid)][0]
        length = min(int(np.searchsorted(timestamps, row.query_timestamp, side="left")), max_length)
        if length == 0:
            raise ValueError(f"Empty causal prefix for request {row.request_id}")
        groups[length].append(i)
    values = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in PATHS}
    prefix_lengths = np.zeros(len(rows), dtype=np.int32)
    prefix_last = np.zeros(len(rows), dtype=np.int64)
    checks = []
    installed = {name: device_parameters(parameters, device) for name, parameters in rules.items()}
    ledger = defaultdict(float)
    completed = 0
    for length, indices in sorted(groups.items(), reverse=True):
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start+batch_size]
            part = rows.iloc[chosen]
            prefixes = [history.prefix(int(r.uid), int(r.query_timestamp), max_length)
                        for r in part.itertuples(index=False)]
            stamps = np.stack([p[2] for p in prefixes])
            assert np.all(stamps[:, -1] < part.query_timestamp.to_numpy())
            deltas = np.zeros_like(stamps, dtype=np.float32)
            deltas[:, 1:] = np.diff(stamps, axis=1)
            events = (torch.as_tensor(np.stack([p[0] for p in prefixes]), device=device),
                      torch.as_tensor(np.stack([p[1] for p in prefixes]), device=device),
                      torch.as_tensor(deltas, device=device))
            began = clock(device)
            source = parent.compute_kv(*events)
            teacher = current.compute_kv(*events)
            ledger["paired_prefix_seconds"] += clock(device) - began
            panel = torch.tensor(part.item_idx.to_numpy(dtype=np.int64)[:, None], device=device)
            delta = torch.as_tensor((part.query_timestamp.to_numpy()-stamps[:, -1]).astype(np.float32), device=device)
            counts = torch.full((len(chosen),), float(length), device=device)
            if not checks or (length < max_length and all(c["length"] == max_length for c in checks)):
                checks.append(dict(length=length, **reader_canary(current, source, teacher, panel, delta)))
            for name in PATHS:
                began = clock(device)
                if name in ("reuse", "exact"):
                    prediction, _ = score(current, source if name == "reuse" else teacher, panel, delta)
                else:
                    factory = bias_read_probe if name in NEW_METHODS else query_read_probe
                    override = factory.make_history_override(installed[name], counts)
                    prediction, _ = score(current, source, panel, delta, history_override=override)
                values[name][chosen] = prediction[:, 0].cpu().numpy()
                ledger[name+"_read_seconds"] += clock(device) - began
            prefix_lengths[chosen], prefix_last[chosen] = length, stamps[:, -1]
            completed += len(chosen)
            if completed % 1024 < len(chosen) or completed == len(rows):
                print(f"  scored {completed}/{len(rows)} feedback requests", flush=True)
            del source, teacher, events, panel, delta, counts, prediction, _
    assert all(np.isfinite(v).all() for v in values.values())
    return rows.assign(prefix_length=prefix_lengths, prefix_last_timestamp=prefix_last, **values), checks, dict(ledger)


def score_and_join(raw, output, edge):
    """Seal all candidate scores before associating the existing observed labels."""
    raw_path = output / f"{edge}_scores.parquet"
    raw.to_parquet(raw_path, index=False)
    write_json(output / f"{edge}_scores.seal.json", dict(
        score_sha256=sha256_file(raw_path), requests=len(raw), columns=list(raw.columns),
        label_join_started=False))
    labels = pq.read_table(REQUEST_ROOT / "requests_quality.parquet", filters=[
        ("request_id", "in", raw.request_id.tolist()),
    ], columns=["request_id", "uid", "query_timestamp", "item_idx", "label"]).to_pandas()
    joined = raw.merge(labels, on=["request_id", "uid", "query_timestamp", "item_idx"],
                       how="left", validate="one_to_one")
    assert len(joined) == len(raw) and joined.label.notna().all()
    assert joined.label.isin([0, 1]).all()
    joined.to_parquet(output / f"{edge}_quality.parquet", index=False)
    measures = {name: binary_metrics(joined.label.to_numpy(), joined[name].to_numpy()) for name in PATHS}
    gap = measures["exact"]["ROC_AUC"] - measures["reuse"]["ROC_AUC"]
    quality = {name: dict(auc=measures[name]["ROC_AUC"],
        delta_auc_pp=100*(measures[name]["ROC_AUC"]-measures["reuse"]["ROC_AUC"]),
        auc_gap_recovery=(measures[name]["ROC_AUC"]-measures["reuse"]["ROC_AUC"])/gap if gap>1e-4 else None,
        binary_metrics=measures[name]) for name in PATHS}
    return dict(requests=len(joined), feedback_users=int(joined.uid.nunique()),
        positives=int(joined.label.sum()), negatives=int((joined.label==0).sum()),
        exact_minus_reuse_auc=gap, quality=quality,
        scores_sha256=sha256_file(raw_path), quality_sha256=sha256_file(output/f"{edge}_quality.parquet"))


@torch.inference_mode()
def run(args):
    config = json.loads(args.config.read_text())
    fit_uids = config["calibration_uids"]
    eval_uids = config["pilot_uids"] if args.pilot else config["evaluation_uids"]
    assert not set(fit_uids) & set(eval_uids)
    assert not set(config["pilot_uids"]) & set(config["evaluation_uids"])
    assert sha256_file(ROOT/config["model_chain"]) == config["model_chain_sha256"]
    assert sha256_file(ROOT/config["source_split"]["path"]) == config["source_split"]["sha256"]
    assert config["ridge"] == .01 and config["edges"] == [1, 2]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    source_paths = [Path(__file__), ROOT/"scripts/design/bias_read_probe.py",
        ROOT/"scripts/design/query_read_probe.py", ROOT/"scripts/design/run_shared_read_probe.py",
        ROOT/"scripts/design/shared_read_probe.py", ROOT/"scripts/design/diagnose_native_input.py",
        ROOT/"scripts/design/competitor_models.py", ROOT/"scripts/design/competitor_data.py",
        ROOT/"scripts/evaluate_yambda500m_foundation_raw.py",
        *[ROOT/"src/hstu_kvcache"/p for p in ("adaptation/reader.py", "models/hstu.py",
            "models/attention.py", "models/kv_cache.py", "data/yambda_history.py",
            "training/foundation.py", "evaluation/binary_metrics.py")]]
    settings = dict(config=config, config_sha256=sha256_file(args.config),
        mode="pilot" if args.pilot else "diagnostic", device=str(device),
        threads=args.threads, batch_size=args.batch_size, calibration_uids=fit_uids,
        evaluation_uids=eval_uids, torch_version=torch.__version__,
        source_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        request_fidelity_sha256=sha256_file(REQUEST_ROOT/"requests_fidelity.parquet"),
        request_quality_sha256=sha256_file(REQUEST_ROOT/"requests_quality.parquet"))
    assert settings["request_fidelity_sha256"] == config["request_manifest"]["requests_fidelity"]["sha256"]
    assert settings["request_quality_sha256"] == config["request_manifest"]["requests_quality"]["sha256"]
    write_json(args.output/"configuration.json", settings)
    started = time.perf_counter()
    try:
        numerical = bias_read_probe.self_check()
        pairs = [load_model_pair("medium", edge, verify_hashes=True) for edge in config["edges"]]
        assert all(p["admission"]["reuse_eligible"] for p in pairs)
        data = pairs[0]["dataset"]
        needed = fit_uids+eval_uids if args.pilot else eval_uids
        history = load_histories(needed, dataset_path=Path(data["manifest"]),
            known_vocab_size=data["known_items"], oov_buckets=data["oov_buckets"],
            start_timestamp=min(p["cutover"] for p in pairs)-1,
            end_timestamp=max(p["cutover"] for p in pairs)+14*86400,
            max_history=config["history_length"], threads=args.threads)
        history_seconds = time.perf_counter()-started
        print(f"History loaded in {history_seconds:.1f}s", flush=True)
        results = []
        for pair in pairs:
            edge = pair["edge"]
            edge_start = clock(device)
            print(f"Starting {edge}", flush=True)
            parent, p = load_model(Path(pair["parent"]["checkpoint"]), device)
            assert p["config"] == pair["config"]
            del p
            current, p = load_model(Path(pair["current"]["checkpoint"]), device)
            assert p["config"] == pair["config"]
            del p
            parent.requires_grad_(False)
            current.requires_grad_(False)
            binding = {role: pair[role]["checkpoint_sha256"] for role in ("parent", "current")}
            old_dir = ROOT/"results/insight/query_read_6l_2560_20260920/pilot"
            old_path = old_dir/f"{edge}_rules.pt"
            old_record = next(r for r in json.loads((old_dir/"summary.json").read_text())["edges"] if r["edge"]==edge)
            assert sha256_file(old_path) == old_record["calibration_rules_sha256"]
            old = torch.load(old_path, map_location="cpu", weights_only=True)
            assert old["calibration_uids"] == fit_uids and old["model_binding"] == binding
            rules = dict(old["rules"])
            rule_path = args.output/f"{edge}_rules.pt"
            if args.pilot:
                arrays = history_arrays(history, np.asarray(fit_uids), pair["cutover"], config["history_length"])
                panel, _ = make_candidate_panel(arrays[1], data["known_items"])
                panel = torch.as_tensor(panel[:, config["calibration_candidate_indices"]], device=device)
                delta = torch.as_tensor(arrays[4], device=device)
                source = collect_cache(parent, arrays, len(fit_uids), args.batch_size, device)
                teacher = collect_cache(current, arrays, len(fit_uids), args.batch_size, device)
                fitting = {}
                for name in NEW_METHODS:
                    rules[name], fitting[name] = fit_rule(current, source, teacher, panel, delta,
                        name=="shared_bias_response", args.batch_size, device)
                del source, teacher, panel, delta
                torch.save(dict(rules=rules, calibration_uids=fit_uids, fitting=fitting,
                    model_binding=binding, config_sha256=settings["config_sha256"],
                    old_rules_sha256=sha256_file(old_path)), rule_path)
            else:
                previous = json.loads((args.calibration_from/"configuration.json").read_text())
                prior_summary = json.loads((args.calibration_from/"summary.json").read_text())
                assert previous["config_sha256"] == settings["config_sha256"]
                assert previous["source_sha256"] == settings["source_sha256"]
                assert prior_summary["status"] == "completed" and prior_summary["mode"] == "pilot"
                prior_record = next(r for r in prior_summary["edges"] if r["edge"]==edge)
                prior_path = args.calibration_from/f"{edge}_rules.pt"
                assert sha256_file(prior_path) == prior_record["rules_sha256"]
                artifact = torch.load(prior_path, map_location="cpu", weights_only=True)
                assert artifact["calibration_uids"] == fit_uids and artifact["model_binding"] == binding
                rules, fitting = artifact["rules"], artifact["fitting"]
                shutil.copyfile(prior_path, rule_path)
            frame = requests(eval_uids, pair["cutover"])
            metadata = next(r for r in config["prospective_request_metadata"][
                "pilot" if args.pilot else "evaluation"] if r["edge"]==edge)
            assert len(frame) == metadata["known_requests"]
            assert frame.uid.nunique() == metadata["known_feedback_users"]
            raw, checks, ledger = evaluate(parent, current, history, frame, rules,
                args.batch_size, device, config["history_length"])
            record = dict(edge=edge, pair=pair, selected_users=len(eval_uids), calibration_users=len(fit_uids),
                rules_sha256=sha256_file(rule_path), old_rules_sha256=sha256_file(old_path),
                fitting=fitting, numerical_checks=checks, ledger_seconds=ledger,
                **score_and_join(raw, args.output, edge))
            record["no_feedback_users"] = len(eval_uids)-record["feedback_users"]
            record["excluded_oov_requests"] = metadata["excluded_oov_requests"]
            record["elapsed_seconds"] = clock(device)-edge_start
            results.append(record)
            write_json(args.output/f"{edge}_summary.json", record)
            print(json.dumps({k:record[k] for k in ("edge", "feedback_users", "requests", "quality", "elapsed_seconds")}), flush=True)
            del parent, current, rules, old
        summary = dict(status="completed", mode=settings["mode"], numerical_check=numerical,
            history_seconds=history_seconds, elapsed_seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1<<30), edges=results)
        write_json(args.output/"summary.json", summary)
    except Exception as exc:
        write_json(args.output/"summary.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/insight/auc_read_6l_20260920.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--calibration-from", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not args.pilot and args.calibration_from is None:
        parser.error("the expanded diagnostic reuses --calibration-from unchanged")
    run(args)
