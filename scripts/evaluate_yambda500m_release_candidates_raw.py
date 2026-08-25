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


def load_candidate_model(path: Path, device: torch.device) -> tuple[HSTU, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("status") != "formal_small_seed17_checkpoint":
        raise RuntimeError(f"candidate evaluator requires a formal checkpoint: {path}")
    progress = float(payload.get("progress", -1.0))
    if progress != 1.0:
        raise RuntimeError(f"release candidates must be complete whole-epoch checkpoints: {path}")
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
    parser.add_argument("--parent", required=True, help="NAME=PATH")
    parser.add_argument("--current", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--start-day", type=int)
    parser.add_argument("--end-day", type=int)
    args = parser.parse_args()
    parents = named_paths([args.parent])
    if len(parents) != 1:
        raise ValueError("exactly one parent is required")
    parent_name, parent_path = next(iter(parents.items()))
    current_paths = named_paths(args.current)

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    try:
        if world != 4:
            raise RuntimeError("formal release-only evaluation requires four ranks")
        if not 1 <= args.batch_size <= 128:
            raise ValueError("batch-size must be in [1,128]")
        if rank == 0:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite {args.output}")
            args.output.mkdir(parents=True)
        dist.barrier()

        parent, parent_payload = load_candidate_model(parent_path, device)
        currents: dict[str, HSTU] = {}
        current_payloads: dict[str, dict] = {}
        for name, path in current_paths.items():
            currents[name], current_payloads[name] = load_candidate_model(path, device)
        models = {parent_name: parent, **currents}
        model_vocab_sizes = {int(payload["config"]["num_items"]) for payload in [parent_payload, *current_payloads.values()]}
        if len(model_vocab_sizes) != 1:
            raise RuntimeError("all Parent/Current candidates must share one item mapping dimension")
        oov_buckets = model_vocab_sizes.pop() - 781678
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

        training_users = set(
            int(value) for value in pq.read_table(
                source, filters=[("time_block", "=", args.training_block), ("target_known", "=", True)],
                columns=["uid"],
            )["uid"].to_pylist()
        )
        for row in rows:
            # The collator requires these fields, but labels are intentionally absent here.
            row["label"] = 0.0
            row["weight"] = 1.0
            row["recurring_user"] = int(row["uid"]) in training_users

        history = load_histories(selected_uids, oov_buckets=oov_buckets)
        hashes = {name: sha256_file(path) for name, path in {parent_name: parent_path, **current_paths}.items()}

        output: list[dict] = []
        for start in range(0, len(rows), args.batch_size):
            request_batch = rows[start:start + args.batch_size]
            batch = collate_foundation_batch(request_batch, history, device=device)
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
                        "hstu_logit": float(scores[name][index]),
                        "history_length": int(request.get("history_length", 0)),
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
                "oov_buckets": oov_buckets, "architecture": "hstu_native_cc",
            }
            (args.output / "raw.seal.json").write_text(json.dumps(seal, indent=2) + "\n")
            print(json.dumps(seal, indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
