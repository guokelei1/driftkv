#!/usr/bin/env python3
"""Run the frozen AUC protocol on four GPUs, one release at a time."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design.competitor_models import load_model_pair, sha256_file
from design.run_unified_auc import (
    bias_read_probe, calibration, evaluate, join_quality, load_model, requests, write_json,
)
from hstu_kvcache.baselines.kv_translate import KVTranslator
from hstu_kvcache.training.foundation import FoundationHistoryIndex

ARRAY_NAMES = ("timestamps", "items", "behaviors", "deltas", "query_deltas")
REQUEST_COLUMNS = ("request_id", "uid", "query_timestamp", "item_idx")
DEFAULT_RESULT = ROOT / "results/insight/unified_auc_10k_20260920"


def batch_spans(users, batch_size, workers=4):
    """Split whole global batches; each user's matrix shape stays unchanged."""
    blocks = (users + batch_size - 1) // batch_size
    if blocks < workers:
        raise ValueError("parallel comparison needs at least one batch per worker")
    spans, begin = [], 0
    for rank in range(workers):
        count = blocks // workers + (rank < blocks % workers)
        end = min(users, begin + count * batch_size)
        spans.append((begin, end))
        begin = end
    assert begin == users
    return spans


def prepared_arrays(root, edge, group, expected_uids):
    directory = root / edge / group
    uids = np.load(directory / "uids.npy", mmap_mode="r")
    assert np.array_equal(uids[:len(expected_uids)], expected_uids)
    # Copy-on-write mmap is writable for torch.as_tensor without changing disk.
    arrays = tuple(np.load(directory / f"{name}.npy", mmap_mode="c")[:len(expected_uids)]
                   for name in ARRAY_NAMES)
    assert all(len(array) == len(expected_uids) for array in arrays)
    return arrays


def verify_prepared(root, metadata, config, edge_names, canary):
    groups = {"pilot": config["canary"]["mature_pilot_uids"]} if canary else {
        "fit": config["calibration_uids"], "evaluation": config["evaluation_uids"]}
    for edge in edge_names:
        for group, uids in groups.items():
            directory = root / edge / group
            record = metadata["outputs"][edge][group]
            assert record["users"] == len(uids)
            assert np.array_equal(np.load(directory / "uids.npy"), uids)
            for name, details in record["files"].items():
                assert sha256_file(directory / f"{name}.npy") == details["sha256"]


def setup_gpu(gpu, threads):
    torch.set_num_threads(threads)
    pa.set_cpu_count(threads)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    return device


def load_pair(pair, device):
    models = []
    for role in ("parent", "current"):
        model, payload = load_model(Path(pair[role]["checkpoint"]), device)
        assert payload["config"] == pair["config"]
        models.append(model)
        del payload
    return models


def source_hashes():
    files = [Path(__file__), ROOT / "scripts/design/run_unified_auc.py",
             ROOT / "scripts/design/prepare_unified_auc_inputs.py",
             ROOT / "scripts/design/unified_auc_baselines.py",
             ROOT / "scripts/design/run_auc_read_probe.py", ROOT / "scripts/design/bias_read_probe.py",
             ROOT / "scripts/design/run_shared_read_probe.py", ROOT / "scripts/design/competitor_data.py",
             ROOT / "scripts/design/competitor_models.py", ROOT / "scripts/evaluate_yambda500m_foundation_raw.py",
             ROOT / "src/hstu_kvcache/data/yambda_history.py", ROOT / "src/hstu_kvcache/training/foundation.py",
             ROOT / "src/hstu_kvcache/evaluation/binary_metrics.py", ROOT / "src/hstu_kvcache/adaptation/reader.py",
             *sorted((ROOT / "src/hstu_kvcache/models").glob("*.py")),
             *sorted((ROOT / "src/hstu_kvcache/baselines").glob("*/core.py"))]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in files}


@torch.inference_mode()
def worker(args):
    settings = json.loads((args.output / "configuration.json").read_text())
    pair = settings["model_pairs"][args.edge]
    edge, all_uids = pair["edge"], settings["evaluation_uids"]
    span = batch_spans(len(all_uids), args.batch_size)[args.rank]
    begin, end = (0, len(all_uids)) if args.reference else span
    uids = all_uids[begin:end]
    group = "pilot" if settings["mode"] == "canary_parallel" else "evaluation"
    arrays = tuple(array[begin:end] for array in prepared_arrays(args.prepared, edge, group, all_uids))
    rows = requests(uids, pair["cutover"])
    rule_path = args.output / f"{edge}_calibration.pt"
    artifact = torch.load(rule_path, map_location="cpu", weights_only=False)
    assert artifact["config_sha256"] == settings["config_sha256"]
    assert artifact["model_binding"] == {r: pair[r]["checkpoint_sha256"] for r in ("parent", "current")}
    translators = {int(k): KVTranslator(**value) for k, value in artifact["translators"].items()}
    directory = args.output / edge / ("reference" if args.reference else f"rank{args.rank}")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = setup_gpu(args.gpu, args.threads)
    try:
        parent, current = load_pair(pair, device)
        raw, execution = evaluate(parent, current, arrays, uids, rows, artifact, translators,
                                  args.batch_size, args.candidate_chunk, device)
        raw_path = directory / "scores.parquet"
        raw.to_parquet(raw_path, index=False)
        write_json(directory / "execution.json", dict(status="completed", **execution,
            uid_begin=begin, uid_end=end, evaluation_uids=uids, rules_sha256=sha256_file(rule_path),
            scores_sha256=sha256_file(raw_path), total_seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30), device=str(device)))
    except Exception as exc:
        write_json(directory / "execution.json", dict(status="failed", error=repr(exc)))
        raise


def launch_workers(args, edge_index, *, reference=False):
    edge = json.loads((args.output / "configuration.json").read_text())["model_pairs"][edge_index]["edge"]
    children = []
    try:
        for rank in ([0] if reference else range(4)):
            directory = args.output / edge / ("reference" if reference else f"rank{rank}")
            directory.mkdir(parents=True)
            log = (directory / "runtime.log").open("w")
            command = [sys.executable, str(Path(__file__)), "--worker", "--config", str(args.config),
                "--output", str(args.output), "--prepared", str(args.prepared), "--edge", str(edge_index),
                "--rank", str(rank), "--gpu", str(rank), "--batch-size", str(args.batch_size),
                "--candidate-chunk", str(args.candidate_chunk), "--threads", str(args.threads)]
            if reference:
                command.append("--reference")
            env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            children.append((process, log))
        for process, _ in children:
            if process.wait() != 0:
                raise RuntimeError(f"GPU worker failed; see {args.output / edge}")
    finally:
        for process, log in children:
            if process.poll() is None:
                process.terminate()
            log.close()
        for process, _ in children:
            process.wait()


def concatenate(args, pair, uids):
    edge = pair["edge"]
    directories = [args.output / edge / f"rank{rank}" for rank in range(4)]
    records = [json.loads((directory / "execution.json").read_text()) for directory in directories]
    assert all(record["status"] == "completed" for record in records)
    assert [uid for record in records for uid in record["evaluation_uids"]] == uids
    raw = pd.concat([pd.read_parquet(directory / "scores.parquet") for directory in directories], ignore_index=True)
    raw = raw.sort_values(["uid", "query_timestamp", "item_idx", "request_id"]).reset_index(drop=True)
    expected = requests(uids, pair["cutover"])
    assert raw[list(REQUEST_COLUMNS)].equals(expected[list(REQUEST_COLUMNS)])
    assert not raw.request_id.duplicated().any()
    counts = raw.groupby("uid").size().reindex(uids, fill_value=0).to_numpy(dtype=np.int64)
    assert np.array_equal(counts, np.concatenate([record["per_user_requests"] for record in records]))
    rule_hash = sha256_file(args.output / f"{edge}_calibration.pt")
    assert all(record["rules_sha256"] == rule_hash for record in records)
    return raw, dict(snapshot_users=len(uids), feedback_users=int((counts > 0).sum()),
        no_feedback_users=int((counts == 0).sum()), requests=len(raw), per_user_requests=counts.tolist(),
        checks=sorted({check for record in records for check in record["checks"]}), workers=records)


def parallel_canary(args, pair, uids):
    artifact_path = args.calibration_from / f"{pair['edge']}_calibration.pt"
    shutil.copyfile(artifact_path, args.output / artifact_path.name)
    launch_workers(args, 0, reference=True)
    launch_workers(args, 0)
    raw, execution = concatenate(args, pair, uids)
    reference_dir = args.output / pair["edge"] / "reference"
    reference = pd.read_parquet(reference_dir / "scores.parquet")
    assert raw[list(REQUEST_COLUMNS)].equals(reference[list(REQUEST_COLUMNS)])
    paths = [name for name in raw if name not in (*REQUEST_COLUMNS, "prefix_last_timestamp", "prefix_length")]
    differences = {}
    for name in paths:
        actual, expected = raw[name].to_numpy(), reference[name].to_numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        differences[name] = float(np.max(np.abs(actual-expected)))
    reference_quality = join_quality(reference, reference_dir, pair["edge"])
    pooled_quality = join_quality(raw, args.output, pair["edge"])
    auc_differences = {name: abs(pooled_quality["quality"][name]["auc"] -
                                  reference_quality["quality"][name]["auc"]) for name in paths}
    return dict(status="completed", mode="canary_parallel", users=len(uids), paths=len(paths),
        batch_size=args.batch_size, global_batch_spans=batch_spans(len(uids), args.batch_size),
        source_calibration_sha256=sha256_file(artifact_path),
        maximum_absolute_logit_difference=max(differences.values()), per_path_logit_difference=differences,
        maximum_absolute_auc_difference=max(auc_differences.values()), per_path_auc_difference=auc_differences,
        logit_atol=2e-5, logit_rtol=2e-5, execution=execution, quality=pooled_quality,
        reference_execution=json.loads((reference_dir / "execution.json").read_text()))


@torch.inference_mode()
def run(args):
    config = json.loads(args.config.read_text())
    assert config["edges"] == [0, 1, 2, 3] and config["history_length"] == 1024
    assert len(config["calibration_uids"]) == 256 and len(config["evaluation_uids"]) == 10000
    assert not set(config["calibration_uids"]) & set(config["evaluation_uids"])
    assert not set(config["pilot_uids"]) & set(config["evaluation_uids"])
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.canary_parallel:
        assert args.batch_size == 16
    else:
        assert args.batch_size == 64
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        for name in ("requests_fidelity", "requests_quality"):
            record = config["request_manifest"][name]
            assert sha256_file(ROOT / record["path"]) == record["sha256"]
        assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
        prepared = json.loads((args.prepared / "metadata.json").read_text())
        assert prepared["status"] == "completed" and prepared["config_sha256"] == sha256_file(args.config)
        pairs = [load_model_pair("medium", i, verify_hashes=True)
                 for i in (config["edges"][:1] if args.canary_parallel else config["edges"])]
        assert all(pair["admission"]["reuse_eligible"] for pair in pairs)
        verify_prepared(args.prepared, prepared, config, [pair["edge"] for pair in pairs], args.canary_parallel)
        canary_record = None
        if not args.canary_parallel:
            canary_record = json.loads((args.parallel_canary / "summary.json").read_text())
            assert canary_record["status"] == "completed" and canary_record["mode"] == "canary_parallel"
        uids = config["canary"]["mature_pilot_uids"][:96] if args.canary_parallel else config["evaluation_uids"]
        settings = dict(config=config, config_sha256=sha256_file(args.config),
            mode="canary_parallel" if args.canary_parallel else "diagnostic", evaluation_uids=uids,
            batch_size=args.batch_size, candidate_chunk=args.candidate_chunk, torch_version=torch.__version__,
            devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"], worker_threads=args.threads,
            model_pairs=pairs, source_sha256=source_hashes(), prepared_metadata_sha256=sha256_file(args.prepared / "metadata.json"),
            global_batch_spans=batch_spans(len(uids), args.batch_size),
            execution_overlay=dict(gpu_ids=[0, 1, 2, 3], data_threads=prepared["threads"],
                worker_torch_threads=args.threads, serial_edges=True, global_batch_size=args.batch_size,
                shard_alignment="contiguous complete global batches; final partial batch belongs to last rank",
                completed_parallel_canary=None if args.canary_parallel else dict(
                    path=str(args.parallel_canary), summary_sha256=sha256_file(args.parallel_canary / "summary.json"),
                    configuration_sha256=sha256_file(args.parallel_canary / "configuration.json"))),
            aggregation="concatenate all raw scores, then compute pooled AUC once per edge")
        write_json(args.output / "configuration.json", settings)
        if args.canary_parallel:
            result = parallel_canary(args, pairs[0], uids)
            result["elapsed_seconds"] = time.perf_counter()-started
            write_json(args.output / "summary.json", result)
            return
        device = setup_gpu(0, args.threads)
        check = bias_read_probe.self_check()
        summaries = []
        for edge_index, pair in enumerate(pairs):
            began = time.perf_counter()
            edge = pair["edge"]
            print(f"Calibrate {edge} once on GPU0", flush=True)
            arrays = prepared_arrays(args.prepared, edge, "fit", config["calibration_uids"])
            history = FoundationHistoryIndex({uid: (arrays[0][i], arrays[1][i], arrays[2][i])
                for i, uid in enumerate(config["calibration_uids"])})
            parent, current = load_pair(pair, device)
            artifact, translators = calibration(parent, current, history, pair, config,
                                                 args.batch_size, device, False)
            artifact.update(model_binding={r: pair[r]["checkpoint_sha256"] for r in ("parent", "current")},
                            config_sha256=settings["config_sha256"])
            rule_path = args.output / f"{edge}_calibration.pt"
            torch.save(artifact, rule_path)
            calibration_seconds = time.perf_counter()-began
            calibration_peak = torch.cuda.max_memory_allocated(device)/(1 << 30)
            del parent, current, translators, arrays, history
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Evaluate {edge} across GPU0/1/2/3", flush=True)
            launch_workers(args, edge_index)
            raw, execution = concatenate(args, pair, uids)
            record = dict(edge=edge, model_pair=pair, rules_sha256=sha256_file(rule_path),
                teacher_budgets=artifact["teacher_budgets"], profile_users=artifact["profile_users"],
                translate_users=artifact["translate_users"], intervals=artifact["intervals"],
                interval_profiles=artifact["interval_profiles"], numerical=artifact["numerical"],
                calibration_seconds=calibration_seconds, calibration_peak_allocated_gib=calibration_peak,
                **execution, **join_quality(raw, args.output, edge))
            record["elapsed_seconds"] = time.perf_counter()-began
            summaries.append(record)
            write_json(args.output / f"{edge}_summary.json", record)
            write_json(args.output / "summary.json", dict(status="running", mode="diagnostic", edges=summaries))
            print(json.dumps(dict(edge=edge, elapsed_seconds=record["elapsed_seconds"],
                                  quality=record["quality"])), flush=True)
            del raw, artifact
            torch.cuda.reset_peak_memory_stats(device)
        write_json(args.output / "summary.json", dict(status="completed", mode="diagnostic", self_check=check,
            elapsed_seconds=time.perf_counter()-started,
            peak_allocated_gib=max(max(r["calibration_peak_allocated_gib"],
                *(w["peak_allocated_gib"] for w in r["workers"])) for r in summaries), edges=summaries))
    except Exception as exc:
        write_json(args.output / "summary.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/unified_auc_10k_20260920.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_RESULT / "prepared")
    parser.add_argument("--canary-parallel", action="store_true")
    parser.add_argument("--calibration-from", type=Path, default=DEFAULT_RESULT / "canary")
    parser.add_argument("--parallel-canary", type=Path, default=DEFAULT_RESULT / "canary_parallel")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--edge", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else run(arguments)
