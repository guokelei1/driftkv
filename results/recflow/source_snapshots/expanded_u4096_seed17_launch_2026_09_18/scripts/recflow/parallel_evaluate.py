#!/usr/bin/env python3
"""Evaluate a frozen RecFlow window in independent GPU request shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from development_probe import (  # noqa: E402
    ROOT,
    HSTUConfig,
    PreparedRecFlow,
    ProbeData,
    RecFlowGenerator,
    evaluate,
    save_json,
)

from hstu_kvcache.recflow.metrics import (  # noqa: E402
    aggregate_metrics,
    random_expected_metrics,
    request_metrics,
)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _merge_arrays(paths, indices):
    """Restore panel order, including when a shard has no sampled requests."""
    arrays = [dict(np.load(path)) for path in paths]
    nonempty = [array for array in arrays if len(array["indices"])]
    if not nonempty:
        if len(indices):
            raise ValueError("Workers did not produce the requested panel.")
        return arrays[0]
    fields = set(nonempty[0])
    if any(set(array) != fields for array in nonempty):
        raise ValueError("Worker metric columns differ.")
    combined = {key: np.concatenate([array[key] for array in nonempty]) for key in fields}
    found = combined["indices"]
    if len(np.unique(found)) != len(found) or not np.array_equal(np.sort(found), np.sort(indices)):
        raise ValueError("Worker request indices do not partition the requested panel.")
    positions = {int(index): row for row, index in enumerate(found)}
    order = [positions[int(index)] for index in indices]
    return {key: values[order] for key, values in combined.items()}


def _rows(arrays, strategy=None):
    keys = [key for key in arrays if "@" in key and "__" not in key]
    if strategy is not None:
        prefix = strategy + "__"
        keys = [key[len(prefix):] for key in arrays if key.startswith(prefix) and "@" in key]
    return [dict(positives=float(arrays["positives"][row]),
                 known_positives=float(arrays["known_positives"][row]),
                 **{key: float(arrays[(strategy + "__" if strategy else "") + key][row])
                    for key in keys}) for row in range(len(arrays["indices"]))]


def merge_evaluations(dataset, indices, sampled_indices, args, phase, worker_dirs, out,
                      parallel_wall_seconds):
    """Recompute pooled/user/day metrics from request rows, never shard means."""
    indices = np.asarray(indices, dtype=np.int64)
    sampled_set = set(map(int, sampled_indices))
    sampled_order = np.asarray([i for i in indices if int(i) in sampled_set], dtype=np.int64)
    if not args.sampled_distractors:
        sampled_order = np.empty(0, dtype=np.int64)
    full = _merge_arrays([d / f"{phase}_request_metrics.npz" for d in worker_dirs], indices)
    sample = (_merge_arrays([d / f"{phase}_sampled_request_metrics.npz" for d in worker_dirs],
                           sampled_order) if args.sampled_distractors else None)
    rankings = [row for directory in worker_dirs
                for row in json.loads((directory / f"{phase}_rankings.json").read_text())]
    by_index = {row["index"]: row for row in rankings}
    if len(by_index) != len(rankings) or set(by_index) != set(map(int, indices)):
        raise ValueError("Worker raw rankings do not partition the requested panel.")
    rankings = [by_index[int(index)] for index in indices]
    if not np.array_equal([row["uid"] for row in rankings], full["uids"]):
        raise ValueError("Raw rankings and metric UID rows differ.")
    rows = _rows(full)
    baseline = [request_metrics(dataset.popular_ranking, row["positive_ids"], dataset.catalog_set)
                for row in rankings]
    random_rows = [random_expected_metrics(int(row["positives"]), int(row["known_positives"]),
                                          len(dataset.raw_ids)) for row in rows]
    sampled_rows, sampled_random = {}, {}
    if sample is not None and len(sampled_order):
        for count in args.sampled_distractors:
            for name in ("uniform", "popularity"):
                key = f"{name}_{count}"
                sampled_rows[key] = _rows(sample, key)
                sampled_random[key] = [random_expected_metrics(
                    int(row["positives"]), int(row["known_positives"]), int(pool_size))
                    for row, pool_size in zip(sampled_rows[key], sample[key + "__candidate_count"], strict=True)]
    uids = full["uids"]
    sampled_uids = sample["uids"] if sample is not None else np.empty(0, dtype=np.int64)
    days = np.asarray([dataset.prepared.requests[int(i)]["day"] for i in indices])
    sampled_days = np.asarray([dataset.prepared.requests[int(i)]["day"] for i in sampled_order])

    def aggregate_selected(full_selected, sample_selected):
        full_uids, subset_uids = uids[full_selected], sampled_uids[sample_selected]
        return dict(
            full_catalog=aggregate_metrics([rows[i] for i in full_selected], full_uids),
            initial_popularity=aggregate_metrics([baseline[i] for i in full_selected], full_uids),
            sampled_candidate_diagnostics={key: aggregate_metrics([value[i] for i in sample_selected], subset_uids)
                for key, value in sampled_rows.items()},
            random_expected=dict(
                full_catalog=aggregate_metrics([random_rows[i] for i in full_selected], full_uids),
                sampled_candidate_diagnostics={key: aggregate_metrics([value[i] for i in sample_selected], subset_uids)
                    for key, value in sampled_random.items()}))

    worker_results = [json.loads((directory / f"{phase}_evaluation.json").read_text())
                      for directory in worker_dirs]
    # Preserve serial evaluator definitions; numerical aggregates are rebuilt.
    definition_fields = ("phase", "beam_width", "retrieval_scope", "evaluation_precision",
                         "request_selection", "candidate_seed", "sampled_tie_break", "random_definition")
    summary = {key: worker_results[0][key] for key in definition_fields}
    if getattr(args, "evaluation_request_selection", None) is not None:
        summary["request_selection"] = args.evaluation_request_selection
    if getattr(args, "evaluation_panel_scope", None) is not None:
        summary["evaluation_panel_scope"] = args.evaluation_panel_scope
    summary.update(aggregate_selected(np.arange(len(indices)), np.arange(len(sampled_order))))
    summary["seconds"] = parallel_wall_seconds
    summary["full_request_panel"] = dict(requests=len(indices),
        indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
    summary["sampled_request_panel"] = dict(requests=len(sampled_order),
        indices_sha256=hashlib.sha256(sampled_order.tobytes()).hexdigest(),
        selection="Supplied fixed subset of the full panel")
    summary["per_day"] = {str(day): aggregate_selected(np.flatnonzero(days == day),
                                                       np.flatnonzero(sampled_days == day))
                          for day in sorted(set(days))}
    workers = []
    for rank, (directory, result) in enumerate(zip(worker_dirs, worker_results, strict=True)):
        metadata_path = directory / "worker.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else dict(rank=rank)
        workers.append(dict(metadata, evaluation_seconds=result["seconds"],
            full_requests=result["full_request_panel"]["requests"],
            sampled_requests=result["sampled_request_panel"]["requests"],
            peak_allocated_bytes=result["peak_allocated_bytes"],
            peak_reserved_bytes=result["peak_reserved_bytes"]))
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        values = [worker[key] for worker in workers if worker[key] is not None]
        summary[key] = max(values) if values else None
    summary["parallel_evaluation"] = dict(
        sharding="Strided full-panel positions; sampled diagnostics use each shard's intersection.",
        parallel_wall_seconds=parallel_wall_seconds,
        wall_time_scope="Worker dispatch through completion, including process/data/model startup; excludes coordinator merge.",
        sum_worker_evaluation_seconds=sum(worker["evaluation_seconds"] for worker in workers),
        cost_note="Request sharding reduces elapsed time; this is not a claim of reduced GPU-seconds.",
        peak_memory_scope="Top-level peaks are the maximum per GPU, not the sum across GPUs.",
        workers=workers)
    np.savez_compressed(out / f"{phase}_request_metrics.npz", **full)
    if sample is not None:
        np.savez_compressed(out / f"{phase}_sampled_request_metrics.npz", **sample)
    (out / f"{phase}_rankings.json").write_text(json.dumps(rankings) + "\n")
    save_json(out / f"{phase}_evaluation.json", summary)
    return summary


def _dataset(configuration):
    dataset = ProbeData(PreparedRecFlow(configuration["data"]), configuration["catalog_size"],
                        configuration["cohort_users"], configuration["context"])
    for name, values in (("catalog", dataset.raw_ids), ("cohort", np.asarray(dataset.uids))):
        if hashlib.sha256(values.tobytes()).hexdigest() != configuration[name + "_sha256"]:
            raise ValueError(f"Checkpoint {name} differs from the prepared data.")
    return dataset


def _worker(job_path):
    begin = time.monotonic()
    job = json.loads(Path(job_path).read_text())
    out = Path(job["output"])
    torch.set_num_threads(4)
    torch.cuda.set_device(job["device"])
    loaded = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
    configuration = loaded["configuration"]
    dataset = _dataset(configuration)
    model = RecFlowGenerator(HSTUConfig(**configuration["model"]), torch.tensor(dataset.paths),
        history_categories=configuration.get("history_categories", False)).to(job["device"])
    model.load_state_dict(loaded["model"])
    del loaded  # The training optimizer remains off GPU and is not needed here.
    args = argparse.Namespace(**configuration)
    args.device = job["device"]
    evaluate(model, dataset, np.asarray(job["indices"], dtype=np.int64), args, job["phase"], out,
             sampled_indices=np.asarray(job["sampled_indices"], dtype=np.int64))
    save_json(out / "worker.json", dict(rank=job["rank"], device=job["device"],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device_name=torch.cuda.get_device_name(job["device"]),
        total_seconds=time.monotonic() - begin))


def parallel_evaluate(checkpoint, panels, window, output, phase, devices, limit=None):
    """Blocking coordinator; use a fresh output directory, with no training."""
    checkpoint, panels, out = Path(checkpoint).resolve(), Path(panels).resolve(), Path(output).resolve()
    if out.exists():
        raise ValueError("Use a fresh evaluation directory to preserve prior evidence.")
    devices = [f"cuda:{value}" if str(value).isdigit() else str(value) for value in devices]
    if not devices or len(set(devices)) != len(devices) or not all(d.startswith("cuda:") for d in devices):
        raise ValueError("Use distinct CUDA devices, for example 0 1 2 3.")
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configuration = dict(loaded["configuration"])
    checkpoint_phase, checkpoint_epoch = loaded.get("phase"), loaded.get("epoch")
    del loaded
    if configuration["eval_precision"] != "fp32" or configuration["decoder"] != "beam":
        raise ValueError("This fixed-window helper implements the declared FP32 beam evaluation.")
    manifest = json.loads(panels.with_name("summary.json").read_text())
    if digest(panels) != manifest["arrays_file_sha256"]:
        raise ValueError("Frozen panel file changed.")
    for key in ("catalog_sha256", "cohort_sha256"):
        if configuration[key] != manifest[key]:
            raise ValueError("Checkpoint and frozen window cohort/catalog differ.")
    with np.load(panels) as frozen:
        indices = np.asarray(frozen[f"eval_{window}"], dtype=np.int64)
        sampled = np.asarray(frozen[f"sampled_{window}"], dtype=np.int64)
    if limit is not None:
        if limit < 1:
            raise ValueError("A canary panel limit must be positive.")
        indices = indices[:limit]
        sampled = sampled[np.isin(sampled, indices)]
    if not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("The full panel must be nonempty with unique request indices.")
    if len(np.unique(sampled)) != len(sampled) or not np.isin(sampled, indices).all():
        raise ValueError("Sampled requests must be a unique subset of the frozen full panel.")
    devices = devices[:len(indices)]
    out.mkdir(parents=True)
    sources = [Path(__file__), ROOT / "scripts/recflow/development_probe.py",
               ROOT / "src/hstu_kvcache/recflow/data.py", ROOT / "src/hstu_kvcache/recflow/model.py",
               ROOT / "src/hstu_kvcache/recflow/metrics.py", ROOT / "src/hstu_kvcache/models/hstu.py"]
    configuration.update(evaluation_checkpoint=str(checkpoint), evaluation_checkpoint_sha256=digest(checkpoint),
        evaluation_checkpoint_phase=checkpoint_phase, evaluation_checkpoint_epoch=checkpoint_epoch,
        evaluation_panels=str(panels), evaluation_panels_sha256=digest(panels), evaluation_window=window,
        evaluation_panel_scope=manifest.get("scope"),
        evaluation_request_selection=manifest.get("request_selection"),
        evaluation_devices=devices, evaluation_canary_limit=limit,
        evaluation_cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        evaluation_source_sha256={str(path.resolve().relative_to(ROOT)): digest(path) for path in sources})
    save_json(out / "configuration.json", configuration)
    workers, processes, logs = [], [], []
    begin = time.monotonic()
    try:
        for rank, device in enumerate(devices):
            worker_out = out / "workers" / str(rank)
            worker_out.mkdir(parents=True)
            shard = indices[rank::len(devices)]
            job_path = worker_out / "job.json"
            save_json(job_path, dict(checkpoint=str(checkpoint), output=str(worker_out), phase=phase,
                rank=rank, device=device, indices=shard.tolist(),
                sampled_indices=sampled[np.isin(sampled, shard)].tolist()))
            log = (worker_out / "run.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                               "--worker", str(job_path)], stdout=log, stderr=subprocess.STDOUT))
            workers.append(worker_out)
        while any(process.poll() is None for process in processes):
            failed = [rank for rank, process in enumerate(processes) if process.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Evaluation worker(s) {failed} failed; inspect retained worker logs.")
            time.sleep(0.5)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("Evaluation worker failed; inspect retained worker logs.")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    elapsed = time.monotonic() - begin
    dataset = _dataset(configuration)
    args = argparse.Namespace(**configuration)
    summary = merge_evaluations(dataset, indices, sampled, args, phase, workers, out, elapsed)
    save_json(out / "summary.json", dict(configuration=configuration, **{phase: summary}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--panels", type=Path)
    parser.add_argument("--window", choices=("19_19", "19_21", "22_24", "25_27", "20_20", "21_21", "22_22", "23_23", "24_24"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase")
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--limit", type=int, help="First N frozen requests for a correctness canary only")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
        return
    if any(getattr(args, key) is None for key in ("checkpoint", "panels", "window", "output", "phase")):
        parser.error("Supply --checkpoint, --panels, --window, --output and --phase.")
    result = parallel_evaluate(args.checkpoint, args.panels, args.window, args.output,
                               args.phase, args.devices, args.limit)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
