#!/usr/bin/env python3
"""Four-rank FSDP trainer for HSTU-native Small release candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.distributed.fsdp import (
    FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision,
    ShardingStrategy, StateDictType,
)

from hstu_kvcache.data import apply_stable_oov_buckets
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.training import FoundationHistoryIndex, cache_producer_sha256, collate_foundation_batch


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "configs/contracts/yambda500m_small_seed17_launch_v1.yaml"
PARENT = ROOT / "configs/contracts/yambda500m_small_foundation_chain_v1.yaml"
DATASET = ROOT / "data/processed/yambda500m_unified_v1/scales/small/dataset.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_launch(version: str, launch_path: Path) -> dict:
    launch = yaml.safe_load(launch_path.read_text())
    if launch.get("status") == "prospective_release_chain_candidate_v1" or str(launch.get("status", "")).startswith("prospective_release_recipe_matrix_"):
        frozen = launch["frozen_inputs"]
        for key in ("dataset_manifest", "item_mapping"):
            if sha256_file(ROOT / frozen[key]) != frozen[f"{key}_sha256"]:
                raise RuntimeError(f"release-chain launch input hash mismatch: {key}")
        scope = launch["scope"]
        if "versions" in scope:
            allowed = set(scope["versions"])
        else:
            total_versions = scope["total_versions_including_v0_by_training_days"].values()
            allowed = {f"v{index}" for index in range(max(map(int, total_versions)))}
    elif "parent_contract_sha256" in launch:
        if launch["parent_contract_sha256"] != sha256_file(PARENT):
            raise RuntimeError("launch contract parent hash mismatch")
        allowed = set(launch["scope"]["default_versions"] + launch["scope"]["optional_same_recipe_extension"] + ["r0"])
    else:
        frozen = launch["frozen_inputs"]
        for key in ("foundation_contract", "original_launch_contract", "dataset_manifest", "item_mapping", "frozen_base", "v0_checkpoint", "v1_checkpoint"):
            if sha256_file(ROOT / frozen[key]) != frozen[f"{key}_sha256"]:
                raise RuntimeError(f"five-version launch input hash mismatch: {key}")
        if "observed_trajectory_seal" in frozen and sha256_file(ROOT / frozen["observed_trajectory_seal"]) != frozen["observed_trajectory_seal_sha256"]:
            raise RuntimeError("training-strength diagnostic input hash mismatch")
        allowed = set(launch["scope"]["versions"])
    if version not in allowed:
        raise RuntimeError(f"version {version} is not authorized")
    return launch


class FoundationForward(nn.Module):
    def __init__(self, hstu: HSTU) -> None:
        super().__init__(); self.hstu = hstu

    def forward(self, items, behaviors, deltas, candidates, query_deltas, lengths):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.hstu.score_cc_full(
                items, behaviors, deltas, candidates, query_deltas, lengths=lengths
            ).float()


def full_hstu_state(model: FSDP, rank: int) -> dict[str, torch.Tensor]:
    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
        wrapped = model.state_dict()
    if rank:
        return {}
    return {
        (name[5:] if name.startswith("hstu.") else name): value.cpu()
        for name, value in wrapped.items()
    }


def balanced_uid_assignment(uids: np.ndarray, counts: np.ndarray, world: int) -> dict[int, int]:
    loads = [0] * world; assignment = {}
    ordered = sorted(zip(uids.tolist(), counts.tolist(), strict=True), key=lambda value: (-value[1], value[0]))
    for uid, count in ordered:
        rank = min(range(world), key=lambda value: (loads[value], value))
        assignment[int(uid)] = rank; loads[rank] += int(count)
    return assignment


def load_rows(
    path: Path, block: str | list[str], rank: int, world: int,
    *, start_day: int | None = None, end_day: int | None = None,
) -> tuple[list[dict], int, int]:
    block_filter = ("time_block", "in", block) if isinstance(block, list) else ("time_block", "=", block)
    filters = [block_filter, ("target_known", "=", True)]
    if (start_day is None) != (end_day is None):
        raise ValueError("train-start-day and train-end-day must be supplied together")
    if start_day is not None:
        if start_day < 0 or end_day <= start_day:
            raise ValueError("invalid half-open training day range")
        filters.extend([("query_timestamp", ">=", start_day * 86_400), ("query_timestamp", "<", end_day * 86_400)])
    table = pq.read_table(
        path, filters=filters,
        columns=["request_id", "uid", "query_timestamp", "item_idx", "label"],
    ).sort_by([("query_timestamp", "ascending"), ("uid", "ascending"), ("request_id", "ascending")])
    all_uids = table["uid"].to_numpy().astype(np.int64)
    unique, counts = np.unique(all_uids, return_counts=True)
    assignment = balanced_uid_assignment(unique, counts, world)
    total = table.num_rows; users = len(unique)
    rows = [row for row in table.to_pylist() if assignment[int(row["uid"])] == rank]
    count_by_uid = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    for row in rows:
        row["weight"] = total / (users * count_by_uid[int(row["uid"])])
    return rows, total, users


def load_histories(uids: list[int], *, oov_buckets: int = 0) -> FoundationHistoryIndex:
    dataset = json.loads(DATASET.read_text()); root = DATASET.parent
    listens = (root / dataset["shared_listens_glob"]).resolve()
    mapping = (root / dataset["item_mapping_path"]).resolve()
    placeholders = ",".join("?" for _ in uids)
    connection = duckdb.connect()
    table = connection.execute(
        f"""SELECT l.uid,l.timestamp,l.raw_item_id,coalesce(m.item_idx,0) item_idx,l.behavior
             FROM read_parquet(?) l LEFT JOIN read_parquet(?) m
             ON l.raw_item_id=m.raw_item_id WHERE l.uid IN ({placeholders})
             ORDER BY l.uid,l.timestamp,l.raw_item_id,l.behavior""",
        [str(listens), str(mapping), *uids],
    ).fetch_arrow_table()
    connection.close()
    item_ids = apply_stable_oov_buckets(
        table["raw_item_id"].to_numpy(), table["item_idx"].to_numpy(),
        known_vocab_size=781678, buckets=oov_buckets,
    )
    return FoundationHistoryIndex.from_columns(
        table["uid"].to_numpy(), table["timestamp"].to_numpy(),
        item_ids, table["behavior"].to_numpy(),
    )


def save_checkpoint(model, rank, output, payload, progress, expected_producer_sha=None):
    state = full_hstu_state(model, rank)
    if rank == 0:
        producer_sha = cache_producer_sha256(state)
        if expected_producer_sha is not None and producer_sha != expected_producer_sha:
            raise RuntimeError("R0 changed cache-producing parameters")
        state_payload = {**payload, "model": state, "progress": progress,
                         "cache_producer_sha256": producer_sha,
                         "r0_same_producer_invariant": expected_producer_sha is None or producer_sha == expected_producer_sha}
        target = output / f"checkpoint_{int(progress * 100):03d}.pt"
        temporary = target.with_suffix(".pt.partial")
        torch.save(state_payload, temporary); os.replace(temporary, target)
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--launch-contract", type=Path, default=LAUNCH)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--canary-steps", type=int, default=0)
    parser.add_argument("--oov-buckets", type=int, default=0)
    parser.add_argument("--train-start-day", type=int)
    parser.add_argument("--train-end-day", type=int)
    parser.add_argument("--passes", type=int)
    parser.add_argument("--training-block")
    args = parser.parse_args()
    launch_path = args.launch_contract.resolve()
    launch = validate_launch(args.version, launch_path)
    frozen = launch.get("frozen_inputs", {})
    if "parent_v1_checkpoint" in frozen:
        expected_parent = (ROOT / frozen["parent_v1_checkpoint"]).resolve()
        if args.parent is None or args.parent.resolve() != expected_parent:
            raise RuntimeError("v2 candidate must use the contract-frozen accepted v1 parent")
        if sha256_file(expected_parent) != frozen["parent_v1_checkpoint_sha256"]:
            raise RuntimeError("v2 contract parent checkpoint hash mismatch")
    if (args.version == "v0") != (args.parent is None):
        raise SystemExit("v0 has no parent; every release requires its direct parent checkpoint")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    try:
        if world != 4:
            raise RuntimeError("Small formal training requires exactly four FSDP ranks")
        if rank == 0:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite {args.output}")
            args.output.mkdir(parents=True)
        dist.barrier()
        if args.training_block:
            block = args.training_block
        elif "window" in launch.get("training", {}):
            block = launch["training"]["window"]
        elif "training_windows" in launch.get("windows_days_half_open", {}):
            block = launch["windows_days_half_open"]["training_windows"][args.version]
        else:
            block = {"v0": "foundation", "v1": "update1", "v2": "update2", "r0": "update1"}[args.version]
        request_path = args.manifest_dir / "requests_quality.parquet"
        rows, total_requests, total_users = load_rows(
            request_path, block, rank, world,
            start_day=args.train_start_day, end_day=args.train_end_day,
        )
        recipe = launch.get("training", launch.get("inputs_and_recipe"))
        passes = int(args.passes if args.passes is not None else recipe["passes"])
        allowed_passes = set(recipe.get("candidate_epoch_options", [passes]))
        if passes not in allowed_passes:
            raise RuntimeError(f"passes={passes} is not authorized by the prospective contract")
        if passes < 1:
            raise RuntimeError("training passes must be positive")
        if args.canary_steps:
            rows = rows[: max(args.batch_size, args.batch_size * args.canary_steps)]
        if args.oov_buckets < 0:
            raise ValueError("oov-buckets must be non-negative")
        histories = load_histories(sorted({int(row["uid"]) for row in rows}), oov_buckets=args.oov_buckets)
        training_rows = rows * passes
        cfg = HSTUConfig(
            num_items=781678 + args.oov_buckets, num_behaviors=4, hidden_size=128, num_layers=4,
            num_heads=4, max_seq_len=512, num_query_types=3, query_type_id=2,
            num_query_actions=1,
        )
        raw = HSTU(cfg)
        parent_hash = None; parent_producer_sha = None
        if args.parent:
            parent = torch.load(args.parent, map_location="cpu", weights_only=False)
            if parent.get("progress") != 1.0:
                raise RuntimeError("recursive release parent must be the final checkpoint")
            if args.version.startswith("v") and int(args.version[1:]) > 0:
                expected_parent = launch.get("scope", {}).get("expected_parent_version", f"v{int(args.version[1:]) - 1}")
                if parent.get("version") != expected_parent:
                    raise RuntimeError(f"{args.version} requires direct parent {expected_parent}")
            if int(parent["config"]["num_items"]) != cfg.num_items:
                raise RuntimeError("parent and candidate must use the same OOV mapping dimension")
            raw.load_state_dict(parent["model"]); parent_hash = sha256_file(args.parent)
            parent_producer_sha = parent["cache_producer_sha256"]
        if args.version == "r0":
            for name, parameter in raw.named_parameters():
                parameter.requires_grad_(name.startswith(("query_encoder.", "cc_score_head.")))
        wrapped = FoundationForward(raw)
        model = FSDP(
            wrapped, device_id=device, use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
            limit_all_gathers=True, sync_module_states=True,
        )
        learning_rate = float(recipe.get("learning_rates", {}).get(args.version, recipe.get("learning_rate", 2e-4)))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate, weight_decay=float(recipe.get("weight_decay", 1e-4)),
        )
        local_steps = math.ceil(len(training_rows) / args.batch_size)
        step_tensor = torch.tensor(local_steps, device=device)
        dist.all_reduce(step_tensor, op=dist.ReduceOp.MAX)
        steps = int(step_tensor); steps = min(steps, args.canary_steps) if args.canary_steps else steps
        losses = []
        for step in range(steps):
            start = step * args.batch_size
            batch_rows = training_rows[start:start + args.batch_size]
            if len(batch_rows) < args.batch_size:
                source = training_rows or batch_rows
                while len(batch_rows) < args.batch_size:
                    duplicate = dict(source[len(batch_rows) % len(source)]); duplicate["weight"] = 0.0
                    batch_rows.append(duplicate)
            batch = collate_foundation_batch(batch_rows, histories, device=device)
            hstu_logits = model(batch.item_ids, batch.behaviors, batch.time_deltas, batch.candidate_ids, batch.query_time_deltas, batch.lengths)[:, 0]
            per_request = F.binary_cross_entropy_with_logits(hstu_logits, batch.labels, reduction="none")
            loss = (per_request * batch.weights).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
            if step + 1 == steps:
                payload = {
                    "status": "four_gpu_canary_checkpoint" if args.canary_steps else "formal_small_seed17_checkpoint",
                    "contract": str(launch_path.relative_to(ROOT)), "contract_sha256": sha256_file(launch_path),
                    "architecture": "hstu_native_cc", "version": args.version, "window": block, "seed": 17,
                    "training_day_range": [args.train_start_day, args.train_end_day],
                    "config": asdict(cfg), "parent_checkpoint_sha256": parent_hash,
                    "request_manifest_sha256": sha256_file(request_path),
                    "world_size": world, "batch_size_per_rank": args.batch_size,
                    "total_requests": total_requests, "total_users": total_users,
                    "passes": passes, "effective_training_examples": total_requests * passes,
                    "oov_buckets": args.oov_buckets,
                    "learning_rate": learning_rate,
                }
                save_checkpoint(model, rank, args.output, payload, 1.0,
                                parent_producer_sha if args.version == "r0" else None)
        if rank == 0:
            result = {
                "status": "four_gpu_canary_passed" if args.canary_steps else "formal_training_complete",
                "contract_sha256": sha256_file(launch_path),
                "version": args.version, "steps": steps, "mean_rank0_loss": float(np.mean(losses)),
                "final_checkpoint": str(args.output / "checkpoint_100.pt"),
                "theta3_read_or_trained": bool(launch.get("scope", {}).get("theta3_read_or_trained", False)),
            }
            (args.output / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
