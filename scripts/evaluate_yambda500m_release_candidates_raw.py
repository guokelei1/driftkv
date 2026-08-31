#!/usr/bin/env python3
"""Four-GPU, raw-first Parent/Current Full evaluation for release candidates.

This entry point intentionally contains no cache producer, Reuse, rolling,
readout-drift, or JS path. It is the upstream release-only evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from evaluate_yambda500m_foundation_raw import balanced_users, load_histories
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.training import collate_foundation_batch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def named_paths(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("model arguments must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"invalid or duplicate model name: {name}")
        output[name] = Path(raw_path).resolve()
    return output


def load_candidate_model(
    path: Path, device: torch.device, *, allow_canary: bool = False,
) -> tuple[HSTU, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    status = str(payload.get("status", ""))
    status_ok = status.startswith("formal_") and status.endswith("_checkpoint")
    status_ok = status_ok or (
        allow_canary and status in {"four_gpu_canary_checkpoint", "distributed_canary_checkpoint"}
    )
    if not status_ok:
        raise RuntimeError(f"candidate evaluator requires a formal checkpoint: {path}")
    progress = float(payload.get("progress", -1.0))
    if progress != 1.0:
        raise RuntimeError(f"release candidates must be complete contract endpoints: {path}")
    model = HSTU(HSTUConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    return model.eval(), payload


def score_batch(*, models: dict[str, HSTU], batch) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        output = {}
        for name, model in models.items():
            scores, _ = model.observe_cc_full(
                batch.item_ids, batch.behaviors, batch.time_deltas,
                batch.candidate_ids, batch.query_time_deltas, lengths=batch.lengths,
            )
            output[name] = scores[:, 0].float().cpu().numpy()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--training-block", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--parent", required=True, help="NAME=PATH")
    parser.add_argument("--current", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--start-day", type=int)
    parser.add_argument("--end-day", type=int)
    parser.add_argument("--training-start-day", type=int)
    parser.add_argument("--training-end-day", type=int)
    parser.add_argument("--allow-canary-checkpoints", action="store_true")
    parser.add_argument("--history-threads", type=int, default=4)
    parser.add_argument("--arrow-cpu-threads", type=int, default=4)
    parser.add_argument("--arrow-io-threads", type=int, default=4)
    parser.add_argument("--torch-cpu-threads", type=int, default=2)
    parser.add_argument("--cpu-affinity-by-rank", help="semicolon-separated comma lists")
    args = parser.parse_args()
    parents = named_paths([args.parent])
    if len(parents) != 1:
        raise ValueError("exactly one parent is required")
    parent_name, parent_path = next(iter(parents.items()))
    current_paths = named_paths(args.current)

    local_rank = int(os.environ["LOCAL_RANK"])
    if args.cpu_affinity_by_rank:
        groups = args.cpu_affinity_by_rank.split(";")
        if local_rank >= len(groups):
            raise ValueError("CPU affinity does not define every local rank")
        os.sched_setaffinity(0, {int(value) for value in groups[local_rank].split(",")})
    if min(args.history_threads, args.arrow_cpu_threads, args.arrow_io_threads, args.torch_cpu_threads) < 1:
        raise ValueError("CPU thread counts must be positive")
    pa.set_cpu_count(args.arrow_cpu_threads)
    pa.set_io_thread_count(args.arrow_io_threads)
    torch.set_num_threads(args.torch_cpu_threads)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if world < 1:
            raise RuntimeError("release-only evaluation requires at least one rank")
        if not 1 <= args.batch_size <= 128:
            raise ValueError("batch-size must be in [1,128]")
        if rank == 0:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite {args.output}")
            args.output.mkdir(parents=True)
        dist.barrier()

        parent, parent_payload = load_candidate_model(
            parent_path, device, allow_canary=args.allow_canary_checkpoints
        )
        currents: dict[str, HSTU] = {}
        current_payloads: dict[str, dict] = {}
        for name, path in current_paths.items():
            currents[name], current_payloads[name] = load_candidate_model(
                path, device, allow_canary=args.allow_canary_checkpoints
            )
        models = {parent_name: parent, **currents}
        model_vocab_sizes = {int(payload["config"]["num_items"]) for payload in [parent_payload, *current_payloads.values()]}
        if len(model_vocab_sizes) != 1:
            raise RuntimeError("all Parent/Current candidates must share one item mapping dimension")
        dataset_value = args.dataset_manifest or parent_payload.get("dataset_manifest")
        if dataset_value is None:
            dataset_path = Path("data/processed/yambda500m_unified_v1/scales/small/dataset.json").resolve()
        else:
            dataset_path = Path(dataset_value)
            if not dataset_path.is_absolute():
                dataset_path = (Path(__file__).resolve().parents[1] / dataset_path).resolve()
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        known_vocab_size = int(parent_payload.get("known_vocab_size", dataset["foundation_items"]))
        oov_buckets = model_vocab_sizes.pop() - known_vocab_size
        if oov_buckets < 0:
            raise RuntimeError("candidate vocabulary is smaller than the frozen known-item vocabulary")

        if (args.start_day is None) != (args.end_day is None):
            raise ValueError("start-day and end-day must be supplied together")
        source = args.manifest_dir / "requests_fidelity.parquet"
        filters = [("time_block", "=", args.block), ("target_known", "=", True)]
        if args.start_day is not None:
            if args.start_day < 0 or args.end_day <= args.start_day:
                raise ValueError("invalid half-open evaluation day range")
            filters.extend([("query_timestamp", ">=", args.start_day * 86_400), ("query_timestamp", "<", args.end_day * 86_400)])
        request_table = pq.read_table(
            source,
            filters=filters,
            columns=[
                "request_id", "uid", "query_timestamp", "item_idx",
            ],
        ).sort_by([("uid", "ascending"), ("query_timestamp", "ascending"), ("request_id", "ascending")])
        rows = request_table.to_pylist()
        assignment = balanced_users(rows, world)
        selected_uids = sorted(uid for uid, assigned in assignment.items() if assigned == rank)
        if args.max_users:
            selected_uids = selected_uids[:args.max_users]
        selected = set(selected_uids)
        rows = [row for row in rows if int(row["uid"]) in selected]

        training_filters = [("time_block", "=", args.training_block), ("target_known", "=", True)]
        if (args.training_start_day is None) != (args.training_end_day is None):
            raise ValueError("training-start-day and training-end-day must be supplied together")
        if args.training_start_day is not None:
            training_filters.extend([
                ("query_timestamp", ">=", args.training_start_day * 86_400),
                ("query_timestamp", "<", args.training_end_day * 86_400),
            ])
        training_users = set(
            int(value) for value in pq.read_table(
                source, filters=training_filters,
                columns=["uid"],
            )["uid"].to_pylist()
        )
        for row in rows:
            # The collator requires these fields, but labels are intentionally absent here.
            row["label"] = 0.0
            row["weight"] = 1.0
            row["recurring_user"] = int(row["uid"]) in training_users

        max_history = int(parent_payload["config"]["max_seq_len"])
        history = load_histories(
            selected_uids, oov_buckets=oov_buckets, dataset_path=dataset_path,
            known_vocab_size=known_vocab_size,
            start_timestamp=args.start_day * 86_400 if args.start_day is not None else None,
            end_timestamp=args.end_day * 86_400 if args.end_day is not None else None,
            max_history=max_history if args.start_day is not None else None,
            threads=args.history_threads,
        )
        hashes = {name: sha256_file(path) for name, path in {parent_name: parent_path, **current_paths}.items()}

        output: list[dict] = []
        torch.cuda.reset_peak_memory_stats(device)
        for start in range(0, len(rows), args.batch_size):
            request_batch = rows[start:start + args.batch_size]
            batch = collate_foundation_batch(
                request_batch, history, device=device, max_history=max_history
            )
            scores = score_batch(models=models, batch=batch)
            for index, request in enumerate(request_batch):
                for name in models:
                    payload = parent_payload if name == parent_name else current_payloads[name]
                    output.append({
                        "request_id": request["request_id"], "uid": int(request["uid"]),
                        "query_timestamp": int(request["query_timestamp"]), "stage": args.stage,
                        "evaluation_block": args.block, "training_block": args.training_block,
                        "evaluation_day_range": [args.start_day, args.end_day],
                        "model_name": name, "is_parent": name == parent_name,
                        "checkpoint_progress": float(payload["progress"]),
                        "training_epochs_completed": float(
                            payload.get("training_epochs_completed", payload.get("passes", 1.0))
                        ),
                        "hstu_logit": float(scores[name][index]),
                        "history_length": int(batch.lengths[index]),
                        "history_oov_fraction": float(request.get("history_oov_fraction", 0.0)),
                        "recurring_user": bool(request["recurring_user"]),
                        "checkpoint_sha256": hashes[name],
                        "parent_checkpoint_sha256": hashes[parent_name],
                        "architecture": "hstu_native_cc",
                    })
            if start % (args.batch_size * 16) == 0:
                (args.output / f"progress_rank{rank}.json").write_text(json.dumps({
                    "rank": rank, "completed_requests": min(start + len(request_batch), len(rows)),
                    "assigned_requests": len(rows),
                }) + "\n")

        shard = args.output / f"raw_rank{rank}.parquet"
        pq.write_table(pa.Table.from_pylist(output), shard, compression="zstd")
        torch.cuda.synchronize(device)
        local_peak = {
            "rank": rank,
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
        }
        peak_memory_by_rank: list[dict | None] = [None] * world
        dist.all_gather_object(peak_memory_by_rank, local_peak)
        dist.barrier()
        if rank == 0:
            merged = pa.concat_tables([pq.read_table(args.output / f"raw_rank{value}.parquet") for value in range(world)])
            models_count = len(models)
            expected = request_table.num_rows * models_count
            if args.max_users == 0 and merged.num_rows != expected:
                raise RuntimeError(f"row conservation failed: {merged.num_rows} != {expected}")
            raw_path = args.output / "raw.parquet"
            pq.write_table(merged, raw_path, compression="zstd")
            seal = {
                "status": "release_candidate_full_only_raw_sealed_before_label_join",
                "raw_sha256": sha256_file(raw_path), "rows": merged.num_rows,
                "stage": args.stage, "evaluation_block": args.block,
                "evaluation_day_range": [args.start_day, args.end_day],
                "training_block": args.training_block, "parent": parent_name,
                "currents": list(currents), "contains_reuse": False,
                "oov_buckets": oov_buckets, "known_vocab_size": known_vocab_size,
                "max_seq_len": max_history, "dataset_manifest": str(dataset_path),
                "execution_runtime": {
                    "world_size": world,
                    "batch_size_per_rank": args.batch_size,
                    "max_users_per_rank": args.max_users,
                    "peak_memory_by_rank": peak_memory_by_rank,
                },
                "cpu_runtime": {
                    "history_threads": args.history_threads,
                    "arrow_cpu_threads": args.arrow_cpu_threads,
                    "arrow_io_threads": args.arrow_io_threads,
                    "torch_cpu_threads": args.torch_cpu_threads,
                    "affinity_by_rank": args.cpu_affinity_by_rank,
                },
                "architecture": "hstu_native_cc",
            }
            (args.output / "raw.seal.json").write_text(json.dumps(seal, indent=2) + "\n")
            print(json.dumps(seal, indent=2))
        dist.barrier()
        if rank == 0:
            for value in range(world):
                (args.output / f"raw_rank{value}.parquet").unlink()
                progress = args.output / f"progress_rank{value}.json"
                if progress.exists():
                    progress.unlink()
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
