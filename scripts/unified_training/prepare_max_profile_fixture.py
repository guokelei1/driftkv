#!/usr/bin/env python3
"""Prepare the exact first Max V2 canary batches on CPU for bounded profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from train_yambda500m_foundation_fsdp import (
    balanced_uid_assignment,
    contract_model_config,
    load_histories,
    load_rows,
    sha256_file,
)
from hstu_kvcache.training import collate_foundation_batch


def identity_digest(rows: list[dict]) -> str:
    identities = [
        [row["request_id"], int(row["uid"]), int(row["query_timestamp"])]
        for row in rows
    ]
    encoded = json.dumps(identities, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch-contract", type=Path,
        default=ROOT / "configs/contracts/yambda5b_max_v2_1epoch_4gpu_b80_20260919.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/backend_acceleration/max_profile_2026_09_19/fixture",
    )
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batches < 1 or not 1 <= args.threads <= 4:
        raise ValueError("a positive batch count and one to four CPU threads are required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pa.set_cpu_count(args.threads)
    pa.set_io_thread_count(args.threads)
    torch.set_num_threads(args.threads)
    launch = yaml.safe_load(args.launch_contract.read_text())
    frozen = launch["frozen_inputs"]
    start_day, end_day = launch["scope"]["training_days_half_open"]
    global_batch = int(launch["training"]["global_batch_size"])
    world_size = 4
    local_batch = global_batch // world_size
    oov_buckets = int(launch["model"]["oov_buckets"])
    cfg, dataset_path, known = contract_model_config(launch, oov_buckets=oov_buckets)
    assert cfg.num_layers == 16 and global_batch == 80 and [start_day, end_day] == [231, 245]
    assert sha256_file(dataset_path) == frozen["dataset_manifest_sha256"]
    request_path = ROOT / frozen["requests_quality"]
    block = launch["training"]["window"]
    print(json.dumps({"stage": "load_training_rows", "request_file": str(request_path)}), flush=True)
    read_started = time.perf_counter()
    # world=1 obtains exactly the trainer's filtered/sorted rows and user-equal
    # weights with one file read. The same assignment function then splits them.
    all_rows, total_requests, total_users = load_rows(
        request_path, block, rank=0, world=1, start_day=start_day, end_day=end_day,
    )
    uids = np.asarray([int(row["uid"]) for row in all_rows], dtype=np.int64)
    unique, counts = np.unique(uids, return_counts=True)
    assignment = balanced_uid_assignment(unique, counts, world_size)
    selected = [[] for _ in range(world_size)]
    rank_totals = [0] * world_size
    for row in all_rows:
        rank = assignment[int(row["uid"])]
        rank_totals[rank] += 1
        if len(selected[rank]) < args.batches * local_batch:
            selected[rank].append(row)
    assert all(len(rows) == args.batches * local_batch for rows in selected)
    rows_seconds = time.perf_counter() - read_started
    del all_rows, uids, unique, counts, assignment
    selected_uids = sorted({int(row["uid"]) for rows in selected for row in rows})
    print(json.dumps({"stage": "load_causal_histories", "selected_users": len(selected_uids),
                      "rows_seconds": rows_seconds}), flush=True)
    history_started = time.perf_counter()
    histories = load_histories(
        selected_uids, oov_buckets=oov_buckets, dataset_path=dataset_path,
        known_vocab_size=known, start_timestamp=start_day * 86_400,
        end_timestamp=end_day * 86_400, max_history=cfg.max_seq_len,
        threads=args.threads,
    )
    history_seconds = time.perf_counter() - history_started
    rank_summaries = []
    for rank, rows in enumerate(selected):
        arrays = {}
        batch_summaries = []
        for step in range(args.batches):
            batch_rows = rows[step * local_batch:(step + 1) * local_batch]
            collate_started = time.perf_counter()
            batch = collate_foundation_batch(
                batch_rows, histories, device=torch.device("cpu"), max_history=cfg.max_seq_len,
            )
            collate_seconds = time.perf_counter() - collate_started
            prefix = f"b{step:03d}_"
            for field in fields(batch):
                value = getattr(batch, field.name)
                if value is not None:
                    arrays[prefix + field.name] = value.numpy()
            arrays[prefix + "request_ids"] = np.asarray([str(row["request_id"]) for row in batch_rows])
            arrays[prefix + "uids"] = np.asarray([row["uid"] for row in batch_rows], dtype=np.int64)
            arrays[prefix + "query_timestamps"] = np.asarray(
                [row["query_timestamp"] for row in batch_rows], dtype=np.int64,
            )
            lengths, group_counts = np.unique(batch.lengths.numpy(), return_counts=True)
            batch_summaries.append({
                "step": step, "width": int(batch.item_ids.shape[1]),
                "lengths": batch.lengths.tolist(),
                "length_groups": {str(int(n)): int(c) for n, c in zip(lengths, group_counts, strict=True)},
                "cpu_collate_seconds": collate_seconds,
                "selected_request_identity_sha256": identity_digest(batch_rows),
            })
        path = args.output / f"rank{rank}.npz"
        np.savez(path, **arrays)
        timestamps = [int(row["query_timestamp"]) for row in rows]
        rank_summaries.append({
            "rank": rank, "file": path.name, "file_sha256": sha256_file(path),
            "payload_bytes": path.stat().st_size,
            "full_window_requests": rank_totals[rank], "selected_requests": len(rows),
            "selected_users": len({int(row["uid"]) for row in rows}),
            "query_timestamp_min": min(timestamps), "query_timestamp_max": max(timestamps),
            "selected_request_identity_sha256": identity_digest(rows), "batches": batch_summaries,
        })
    assert not torch.cuda.is_initialized(), "fixture preparation must remain CPU-only"
    summary = {
        "status": "complete_cpu_fixture", "format": "npz_bNNN_field_arrays",
        "model_config": asdict(cfg), "world_size": world_size,
        "global_batch_size": global_batch, "local_batch_size": local_batch,
        "num_batches": args.batches, "first_batch_index": 0,
        "seed": int(launch["scope"]["seed"]), "training_days_half_open": [start_day, end_day],
        "training_block": block, "total_window_requests": total_requests,
        "total_window_users": total_users, "selected_users": len(selected_uids),
        "row_selection": "first chronological rank-local batches; exact trainer balanced_uid_assignment and user-equal weights; same selection as original Max V2 canary",
        "representativeness": "startup batches reproduce the prior controlled backend comparison; they do not establish the distribution of later steady-training batches",
        "history": "existing bounded history loader: last 1024 prior listens plus [231,245); each batch prefix strictly precedes its query timestamp",
        "parent_checkpoint": frozen["parent_v1_checkpoint"],
        "parent_checkpoint_sha256": frozen["parent_v1_checkpoint_sha256"],
        "parent_hash_source": "existing frozen launch contract; checkpoint not opened or rehashed during CPU fixture preparation",
        "launch_contract": str(args.launch_contract.resolve().relative_to(ROOT)),
        "launch_contract_sha256": sha256_file(args.launch_contract),
        "requests_quality": frozen["requests_quality"],
        "requests_quality_sha256_from_contract": frozen["requests_quality_sha256"],
        "dataset_manifest": frozen["dataset_manifest"],
        "dataset_manifest_sha256": frozen["dataset_manifest_sha256"],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in [
                Path(__file__).resolve(), ROOT / "scripts/train_yambda500m_foundation_fsdp.py",
                ROOT / "src/hstu_kvcache/training/foundation.py",
                ROOT / "src/hstu_kvcache/data/yambda_history.py",
            ]
        },
        "cpu_threads": args.threads, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "read_and_partition_rows_seconds": rows_seconds,
        "load_histories_seconds": history_seconds,
        "cpu_collate_timing_scope": "one call per actual batch with existing history index; excludes data loading and host-to-device transfer; process restricted to spare CPUs",
        "history_events_retained": sum(len(value[0]) for value in histories.rows.values()),
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cuda_initialized": torch.cuda.is_initialized(), "quality_metrics_read": False,
        "ranks": rank_summaries,
    }
    (args.output / "configuration.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "output": str(args.output),
                      "wall_seconds": summary["wall_seconds"],
                      "peak_process_rss_mib": summary["peak_process_rss_mib"]}), flush=True)


if __name__ == "__main__":
    main()
