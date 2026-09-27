#!/usr/bin/env python3
"""Extend the frozen AUC curves with additional, disjoint teacher users."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design import bias_read_probe
from design.competitor_models import load_model_pair, sha256_file
from design.expanded_read_calibration import calibrate_expanded
from design.run_shared_read_probe import device_parameters
from design.run_unified_auc import join_quality, requests, take_events, write_json
from design.run_unified_auc_parallel import (
    ARRAY_NAMES, REQUEST_COLUMNS, batch_spans, load_pair, prepared_arrays,
    setup_gpu, source_hashes,
)
from hstu_kvcache.adaptation.reader import score

DEFAULT_ROOT = ROOT / "results/insight/unified_auc_10k_teachers7144_20260920"
BASE_ROOT = ROOT / "results/insight/unified_auc_10k_20260920"


@torch.inference_mode()
def evaluate_rules(parent, current, arrays, uids, rows, rules, batch_size, chunk, device):
    """Score only the new rules; Reuse independently checks the frozen reader."""
    values = {name: np.full(len(rows), np.nan, dtype=np.float32)
              for name in ("reuse", *rules)}
    installed = {name: device_parameters(parameters, device) for name, parameters in rules.items()}
    by_uid = {int(uid): frame.index.to_numpy() for uid, frame in rows.groupby("uid", sort=False)}
    total_requests = np.zeros(len(uids), dtype=np.int64)
    began = time.perf_counter()
    for begin in range(0, len(uids), batch_size):
        end = min(begin + batch_size, len(uids))
        source = parent.compute_kv(*take_events(arrays, begin, end, device))
        groups = [by_uid.get(int(uid), np.empty(0, dtype=np.int64)) for uid in uids[begin:end]]
        total_requests[begin:end] = [len(group) for group in groups]
        counts = torch.full((end-begin,), float(source.seq_len), device=device)
        overrides = {name: bias_read_probe.make_history_override(parameters, counts)
                     for name, parameters in installed.items()}
        maximum = max(map(len, groups), default=0)
        for offset in range(0, maximum, chunk):
            width = min(chunk, maximum-offset)
            item = np.ones((end-begin, width), dtype=np.int64)
            delta = np.ones((end-begin, width), dtype=np.float32)
            indices = np.full((end-begin, width), -1, dtype=np.int64)
            for local, group in enumerate(groups):
                take = group[offset:offset+width]
                indices[local, :len(take)] = take
                if len(take):
                    item[local, :len(take)] = rows.loc[take, "item_idx"].to_numpy(dtype=np.int64)
                    delta[local, :len(take)] = rows.loc[take, "query_timestamp"].to_numpy()-arrays[0][begin+local, -1]
            panel = torch.as_tensor(item, device=device)
            query_delta = torch.as_tensor(delta, device=device)
            mask = indices >= 0
            for name in values:
                prediction, _ = score(current, source, panel, query_delta,
                                      history_override=overrides.get(name))
                values[name][indices[mask]] = prediction.cpu().numpy()[mask]
        del source, counts, overrides
        if end % 512 < batch_size or end == len(uids):
            print(f"  {end}/{len(uids)} snapshots; {time.perf_counter()-began:.1f}s", flush=True)
    assert sum(total_requests) == len(rows)
    assert all(np.isfinite(value).all() for value in values.values())
    prefix_last = {int(uid): int(stamps[-1]) for uid, stamps in zip(uids, arrays[0], strict=True)}
    raw = rows.assign(prefix_last_timestamp=rows.uid.map(prefix_last), prefix_length=1024, **values)
    assert (raw.prefix_last_timestamp < raw.query_timestamp).all()
    return raw, dict(per_user_requests=total_requests.tolist(), requests=len(rows),
                     elapsed_seconds=time.perf_counter()-began)


@torch.inference_mode()
def worker(args):
    settings = json.loads((args.output / "configuration.json").read_text())
    pair = settings["model_pairs"][args.edge]
    edge, all_uids = pair["edge"], settings["evaluation_uids"]
    begin, end = batch_spans(len(all_uids), args.batch_size)[args.rank]
    uids = all_uids[begin:end]
    arrays = tuple(array[begin:end] for array in prepared_arrays(
        args.base / "prepared", edge, "evaluation", all_uids))
    rows = requests(uids, pair["cutover"])
    artifact_path = args.output / f"{edge}_calibration.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert artifact["config_sha256"] == settings["config_sha256"]
    assert artifact["model_binding"] == {role: pair[role]["checkpoint_sha256"] for role in ("parent", "current")}
    names = [f"{arm}_{budget}" for budget in settings["config"]["new_budgets"] for arm in ("shared", "personal")]
    directory = args.output / edge / f"rank{args.rank}"
    device = setup_gpu(args.rank, args.threads)
    parent, current = load_pair(pair, device)
    raw, execution = evaluate_rules(parent, current, arrays, uids, rows,
        {name: artifact["rules"][name] for name in names}, args.batch_size, args.candidate_chunk, device)
    path = directory / "scores.parquet"
    raw.to_parquet(path, index=False)
    write_json(directory / "execution.json", dict(status="completed", **execution,
        evaluation_uids=uids, scores_sha256=sha256_file(path), rules_sha256=sha256_file(artifact_path),
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30), device=str(device)))


def evaluate_four_gpus(args, edge_index, edge):
    children = []
    try:
        for rank in range(4):
            directory = args.output / edge / f"rank{rank}"
            directory.mkdir(parents=True)
            log = (directory / "runtime.log").open("w")
            command = [sys.executable, str(Path(__file__)), "--worker", "--edge", str(edge_index),
                "--rank", str(rank), "--output", str(args.output), "--config", str(args.config),
                "--base", str(args.base), "--batch-size", str(args.batch_size),
                "--candidate-chunk", str(args.candidate_chunk), "--threads", str(args.threads)]
            env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
            children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        for child, _ in children:
            if child.wait() != 0:
                raise RuntimeError(f"Evaluation worker failed; see {args.output / edge}")
    finally:
        for child, log in children:
            if child.poll() is None:
                child.terminate()
            log.close()
        for child, _ in children:
            child.wait()
    records, frames = [], []
    for rank in range(4):
        directory = args.output / edge / f"rank{rank}"
        record = json.loads((directory / "execution.json").read_text())
        assert record["status"] == "completed"
        assert record["scores_sha256"] == sha256_file(directory / "scores.parquet")
        records.append(record)
        frames.append(pd.read_parquet(directory / "scores.parquet"))
    raw = pd.concat(frames, ignore_index=True).sort_values(
        ["uid", "query_timestamp", "item_idx", "request_id"]).reset_index(drop=True)
    return raw, records


def run(args):
    config = json.loads(args.config.read_text())
    base_config_path = ROOT / config["base_config"]
    base_config = json.loads(base_config_path.read_text())
    base_result = args.base / "diagnostic"
    base_summary = json.loads((base_result / "summary.json").read_text())
    assert base_summary["status"] == "completed"
    assert config["base_config_sha256"] == sha256_file(base_config_path)
    assert config["evaluation_uids"] == base_config["evaluation_uids"]
    assert config["calibration_uids"][:256] == base_config["calibration_uids"]
    assert not set(config["calibration_uids"]) & set(config["evaluation_uids"])
    assert config["new_budgets"] == [512, 1024, 2048, 4096, 7144]
    assert args.batch_size == 64 and args.candidate_chunk == 32
    prepared = json.loads((args.prepared / "metadata.json").read_text())
    assert prepared["status"] == "completed" and prepared["config_sha256"] == sha256_file(args.config)
    canary = json.loads(args.canary.read_text())
    assert canary["status"] == "completed" and canary["reference"]["status"] == "passed"
    canary_settings = json.loads((args.canary.parent / "configuration.json").read_text())
    assert canary_settings["config_sha256"] == sha256_file(args.config)
    assert canary_settings["module_sha256"] == sha256_file(ROOT / "scripts/design/expanded_read_calibration.py")
    evaluation_check = args.canary.parent / "evaluation_canary.json"
    assert json.loads(evaluation_check.read_text())["status"] == "passed"
    for edge in config["edge_names"]:
        for name, details in prepared["outputs"][edge]["fit"]["files"].items():
            assert sha256_file(args.prepared / edge / "fit" / f"{name}.npy") == details["sha256"]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    began = time.perf_counter()
    torch.set_num_threads(args.threads)
    try:
        pairs = [load_model_pair("medium", edge, verify_hashes=True) for edge in base_config["edges"]]
        assert all(pair["admission"]["reuse_eligible"] for pair in pairs)
        files = source_hashes()
        for path in (Path(__file__), ROOT / "scripts/design/expanded_read_calibration.py",
                     ROOT / "scripts/design/prepare_expanded_auc_inputs.py"):
            files[str(path.relative_to(ROOT))] = sha256_file(path)
        settings = dict(mode="diagnostic", config=config, config_sha256=sha256_file(args.config),
            base_config_sha256=sha256_file(base_config_path),
            base_summary_sha256=sha256_file(base_result / "summary.json"),
            evaluation_uids=config["evaluation_uids"], model_pairs=pairs, source_sha256=files,
            prepared_metadata_sha256=sha256_file(args.prepared / "metadata.json"),
            canary_sha256=sha256_file(args.canary), evaluation_canary_sha256=sha256_file(evaluation_check),
            batch_size=64, candidate_chunk=32,
            execution_overlay=dict(gpu_ids=[0, 1, 2, 3], worker_torch_threads=args.threads,
                calibration_cache="CPU memory", serial_edges=True),
            aggregation="new raw scores joined one-to-one to unchanged base scores; pooled AUC per edge")
        write_json(args.output / "configuration.json", settings)
        summaries = []
        for index, pair in enumerate(pairs):
            started = time.perf_counter()
            edge = pair["edge"]
            directory = args.prepared / edge / "fit"
            assert np.array_equal(np.load(directory / "uids.npy"), config["calibration_uids"])
            print(f"Calibrate {edge}: {config['new_budgets']} teachers on GPU0/1/2/3", flush=True)
            extension = calibrate_expanded(pair, directory, directory / "panel.npy", config["new_budgets"],
                batch_size=64, gpus=(0, 1, 2, 3), threads=args.threads)
            base_record = base_summary["edges"][index]
            base_artifact_path = base_result / f"{edge}_calibration.pt"
            assert sha256_file(base_artifact_path) == base_record["rules_sha256"]
            artifact = torch.load(base_artifact_path, map_location="cpu", weights_only=False)
            artifact["rules"].update(extension["rules"])
            artifact["fitting"].update(extension["fitting"])
            artifact.update(calibration_uids=config["calibration_uids"], teacher_budgets=config["teacher_budgets"],
                model_binding=extension["model_binding"], config_sha256=settings["config_sha256"],
                extension_execution=extension["execution"], panel_sha256=extension["panel_sha256"],
                base_artifact_sha256=base_record["rules_sha256"])
            rule_path = args.output / f"{edge}_calibration.pt"
            torch.save(artifact, rule_path)
            print(f"Evaluate {edge}: new rules on the unchanged 10000 users", flush=True)
            new_raw, workers = evaluate_four_gpus(args, index, edge)
            assert [uid for worker_record in workers for uid in worker_record["evaluation_uids"]] == config["evaluation_uids"]
            base_raw_path = base_result / f"{edge}_scores.parquet"
            assert sha256_file(base_raw_path) == base_record["scores_sha256"]
            old_raw = pd.read_parquet(base_raw_path)
            columns = [*REQUEST_COLUMNS, "prefix_last_timestamp", "prefix_length"]
            assert old_raw[columns].equals(new_raw[columns])
            difference = float(np.max(np.abs(old_raw.reuse.to_numpy()-new_raw.reuse.to_numpy())))
            np.testing.assert_allclose(new_raw.reuse, old_raw.reuse, atol=2e-5, rtol=2e-5)
            merged = old_raw.copy()
            for name in extension["rules"]:
                merged[name] = new_raw[name].to_numpy()
            assert merged[old_raw.columns].equals(old_raw)
            assert base_record["per_user_requests"] == [n for worker_record in workers for n in worker_record["per_user_requests"]]
            record = dict(base_record)
            for key in ("calibration_seconds", "calibration_peak_allocated_gib", "numerical"):
                record[f"base_{key}"] = record.pop(key)
            record.update(teacher_budgets=config["teacher_budgets"], rules_sha256=sha256_file(rule_path),
                workers=workers, calibration_execution=extension["execution"],
                base_scores_sha256=base_record["scores_sha256"], maximum_reuse_logit_difference=difference,
                **join_quality(merged, args.output, edge))
            record["elapsed_seconds"] = time.perf_counter()-started
            summaries.append(record)
            write_json(args.output / f"{edge}_summary.json", record)
            write_json(args.output / "summary.json", dict(status="running", mode="diagnostic", edges=summaries))
            print(f"Completed {edge} in {record['elapsed_seconds']:.1f}s; Reuse difference {difference}", flush=True)
            del extension, artifact, merged, old_raw, new_raw
        write_json(args.output / "summary.json", dict(status="completed", mode="diagnostic", edges=summaries,
            base_summary_sha256=settings["base_summary_sha256"], elapsed_seconds=time.perf_counter()-began))
    except Exception as exc:
        write_json(args.output / "summary.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-began))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/unified_auc_10k_teachers7144_20260920.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "diagnostic")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_ROOT / "prepared")
    parser.add_argument("--base", type=Path, default=BASE_ROOT)
    parser.add_argument("--canary", type=Path, default=DEFAULT_ROOT / "calibration_canary/summary.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--edge", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else run(arguments)
