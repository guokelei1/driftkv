#!/usr/bin/env python3
"""Four-rank checkpoint/Full-cache/NVMe I/O canary for the Large point.

This script is deliberately label-free.  It materializes Parent and Current
full-context K/V for a small real-user cohort, writes both states to the
workspace NVMe, verifies file and tensor checksums after reload, then removes
the redundant tensor payloads while retaining an auditable JSON summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from evaluate_yambda500m_foundation_raw import balanced_users, load_histories, load_model


DAY = 86_400


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def materialize_batch(model, histories, uids: list[int], cutover: int, max_length: int):
    timestamps, items, behaviors = [], [], []
    for uid in uids:
        raw_times, raw_items, raw_behaviors = histories.rows[uid]
        stop = int(np.searchsorted(raw_times, cutover, side="left"))
        if stop < max_length:
            raise RuntimeError(f"uid={uid} lacks a full cutover prefix")
        timestamps.append(raw_times[stop - max_length:stop])
        items.append(raw_items[stop - max_length:stop])
        behaviors.append(raw_behaviors[stop - max_length:stop])
    device = next(model.parameters()).device
    times = torch.tensor(np.stack(timestamps), dtype=torch.long, device=device)
    item_ids = torch.tensor(np.stack(items), dtype=torch.long, device=device)
    behavior_ids = torch.tensor(np.stack(behaviors), dtype=torch.long, device=device)
    deltas = torch.zeros_like(times, dtype=torch.float32)
    deltas[:, 1:] = times[:, 1:] - times[:, :-1]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cache = model.compute_kv(item_ids, behavior_ids, deltas)
    return {
        "k": cache.k.detach().to(device="cpu", dtype=torch.bfloat16),
        "v": cache.v.detach().to(device="cpu", dtype=torch.bfloat16),
        "seq_len": int(cache.seq_len),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutover-day", type=int, required=True)
    parser.add_argument("--end-day", type=int, required=True)
    parser.add_argument("--users-per-rank", type=int, default=8)
    parser.add_argument("--history-threads", type=int, default=14)
    parser.add_argument("--arrow-cpu-threads", type=int, default=14)
    parser.add_argument("--arrow-io-threads", type=int, default=4)
    parser.add_argument("--torch-cpu-threads", type=int, default=4)
    parser.add_argument("--cpu-affinity-by-rank")
    parser.add_argument("--allow-canary-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.users_per_rank < 1 or args.end_day <= args.cutover_day:
        raise ValueError("invalid state-I/O cohort or day range")

    local_rank = int(os.environ["LOCAL_RANK"])
    if args.cpu_affinity_by_rank:
        groups = args.cpu_affinity_by_rank.split(";")
        os.sched_setaffinity(0, {int(value) for value in groups[local_rank].split(",")})
    pa.set_cpu_count(args.arrow_cpu_threads)
    pa.set_io_thread_count(args.arrow_io_threads)
    torch.set_num_threads(args.torch_cpu_threads)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if world != 4:
            raise RuntimeError("Large state-I/O canary is frozen to four ranks")
        if rank == 0:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite {args.output}")
            args.output.mkdir(parents=True)
        dist.barrier()

        source = args.manifest_dir / "requests_fidelity.parquet"
        rows = pq.read_table(
            source,
            filters=[
                ("time_block", "=", "matrix_horizon"),
                ("target_known", "=", True),
                ("query_timestamp", ">=", args.cutover_day * DAY),
                ("query_timestamp", "<", args.end_day * DAY),
            ],
            columns=["request_id", "uid", "query_timestamp"],
        ).to_pylist()
        assignment = balanced_users(rows, world)
        candidates = sorted(uid for uid, owner in assignment.items() if owner == rank)
        # Read extra candidates because a real request user may have an underfull
        # prefix at this cutover.  Selection is UID-only and label-free.
        candidates = candidates[: max(args.users_per_rank * 16, 128)]

        parent, parent_payload = load_model(
            args.parent, device, allow_canary=args.allow_canary_checkpoints
        )
        current, current_payload = load_model(
            args.current, device, allow_canary=args.allow_canary_checkpoints
        )
        if parent_payload["config"] != current_payload["config"]:
            raise RuntimeError("Parent and Current configurations differ")
        max_length = int(current_payload["config"]["max_seq_len"])
        known = int(current_payload["known_vocab_size"])
        oov = int(current_payload["config"]["num_items"]) - known
        histories = load_histories(
            candidates,
            oov_buckets=oov,
            dataset_path=args.dataset_manifest.resolve(),
            known_vocab_size=known,
            start_timestamp=args.cutover_day * DAY,
            end_timestamp=args.end_day * DAY,
            max_history=max_length,
            threads=args.history_threads,
        )
        cutover = args.cutover_day * DAY
        selected = [
            uid for uid in candidates
            if int(np.searchsorted(histories.rows[uid][0], cutover, side="left")) >= max_length
        ][:args.users_per_rank]
        if len(selected) != args.users_per_rank:
            raise RuntimeError(
                f"rank {rank} found only {len(selected)} full-context users; "
                f"requires {args.users_per_rank}"
            )

        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        parent_cache = materialize_batch(parent, histories, selected, cutover, max_length)
        parent_seconds = time.perf_counter() - started
        started = time.perf_counter()
        current_cache = materialize_batch(current, histories, selected, cutover, max_length)
        current_seconds = time.perf_counter() - started
        del parent, current
        torch.cuda.empty_cache()

        tensor_hashes = {
            "parent_k": tensor_sha256(parent_cache["k"]),
            "parent_v": tensor_sha256(parent_cache["v"]),
            "current_k": tensor_sha256(current_cache["k"]),
            "current_v": tensor_sha256(current_cache["v"]),
        }
        sample = args.output / f"state_rank{rank}.pt"
        temporary = sample.with_suffix(".pt.partial")
        started = time.perf_counter()
        torch.save({
            "rank": rank,
            "uids": selected,
            "parent": parent_cache,
            "current": current_cache,
            "tensor_sha256": tensor_hashes,
        }, temporary)
        os.replace(temporary, sample)
        write_seconds = time.perf_counter() - started
        file_bytes = sample.stat().st_size
        file_hash = sha256_file(sample)
        del parent_cache, current_cache

        started = time.perf_counter()
        restored = torch.load(sample, map_location="cpu", weights_only=False)
        read_seconds = time.perf_counter() - started
        restored_hashes = {
            "parent_k": tensor_sha256(restored["parent"]["k"]),
            "parent_v": tensor_sha256(restored["parent"]["v"]),
            "current_k": tensor_sha256(restored["current"]["k"]),
            "current_v": tensor_sha256(restored["current"]["v"]),
        }
        checksum_pass = restored_hashes == tensor_hashes and sha256_file(sample) == file_hash
        shapes = {
            key: list(restored[state][tensor].shape)
            for key, state, tensor in (
                ("parent_k", "parent", "k"), ("parent_v", "parent", "v"),
                ("current_k", "current", "k"), ("current_v", "current", "v"),
            )
        }
        del restored
        # The tensors are generated canary payloads, not evidence.  Retain
        # checksums and measured sizes, but avoid consuming ~0.8 GiB forever.
        sample.unlink()
        rank_result = {
            "rank": rank,
            "users": len(selected),
            "uids": selected,
            "max_length": max_length,
            "tensor_shapes": shapes,
            "tensor_sha256": tensor_hashes,
            "file_sha256_before_deletion": file_hash,
            "file_bytes_before_deletion": file_bytes,
            "write_seconds": write_seconds,
            "read_seconds": read_seconds,
            "write_mib_per_second": file_bytes / 2**20 / write_seconds,
            "read_mib_per_second": file_bytes / 2**20 / read_seconds,
            "parent_materialization_seconds": parent_seconds,
            "current_materialization_seconds": current_seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "checksum_pass": checksum_pass,
            "sample_payload_deleted_after_verification": True,
        }
        gathered: list[dict | None] = [None] * world
        dist.all_gather_object(gathered, rank_result)
        if rank == 0:
            values = [value for value in gathered if value is not None]
            passed = len(values) == world and all(value["checksum_pass"] for value in values)
            total_bytes = sum(int(value["file_bytes_before_deletion"]) for value in values)
            atomic_json(args.output / "summary.json", {
                "status": "large_state_io_canary_passed" if passed else "large_state_io_canary_failed",
                "quality_labels_read": False,
                "world_size": world,
                "users_per_rank": args.users_per_rank,
                "cutover_day": args.cutover_day,
                "total_serialized_bytes": total_bytes,
                "effective_parallel_write_mib_per_second": total_bytes / 2**20 / max(value["write_seconds"] for value in values),
                "effective_parallel_read_mib_per_second": total_bytes / 2**20 / max(value["read_seconds"] for value in values),
                "rank_results": values,
            })
            if not passed:
                raise RuntimeError("state-I/O checksum canary failed")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
