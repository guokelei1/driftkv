#!/usr/bin/env python3
"""FSDP trainer for HSTU-native release candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pyarrow as pa
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

from hstu_kvcache.data.yambda_history import load_yambda_histories
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


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_launch(version: str, launch_path: Path) -> dict:
    launch = yaml.safe_load(launch_path.read_text())
    if "branches" in launch.get("scope", {}):
        frozen = launch["frozen_inputs"]
        for key in ("dataset_manifest", "item_mapping", "unified_scale_contract"):
            if sha256_file(ROOT / frozen[key]) != frozen[f"{key}_sha256"]:
                raise RuntimeError(f"release-chain launch input hash mismatch: {key}")
        allowed = {
            name
            for branch in launch["scope"]["branches"].values()
            for name in branch["versions"]
        }
    elif launch.get("status") == "prospective_release_chain_candidate_v1" or str(launch.get("status", "")).startswith("prospective_release_recipe_matrix_"):
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


def validate_execution_contract(path: Path | None, launch_path: Path, world: int) -> dict | None:
    if path is None:
        if world != 4:
            raise RuntimeError("non-four-rank execution requires a prospective execution contract")
        return None
    execution = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = execution["frozen_parent"]
    if parent["contract_sha256"] != sha256_file(launch_path):
        raise RuntimeError("execution supplement does not bind the launch contract")
    if world != int(execution["execution_amendment"]["world_size"]):
        raise RuntimeError(f"world size {world} is not authorized by the execution supplement")
    return execution


def local_batch_size(global_batch_size: int, world: int, rank: int) -> int:
    if global_batch_size < world:
        raise ValueError("global batch size must be at least the world size")
    quotient, remainder = divmod(global_batch_size, world)
    return quotient + int(rank < remainder)


def parse_checkpoint_epochs(
    raw: str | None, recipe: dict, passes: int,
) -> tuple[float, ...]:
    """Return contract-bound cumulative epoch endpoints for one continuous run."""
    frozen = tuple(float(value) for value in recipe.get("checkpoint_epochs", []))
    if raw is None:
        requested = frozen
    else:
        requested = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if requested and not frozen:
        raise RuntimeError("staged checkpoint epochs require a prospective contract")
    if frozen and requested != frozen:
        raise RuntimeError(
            f"checkpoint epochs differ from contract: requested={requested}, frozen={frozen}"
        )
    if any(not math.isfinite(value) or value <= 0 for value in requested):
        raise RuntimeError("checkpoint epochs must be finite and positive")
    if any(right <= left for left, right in zip(requested, requested[1:])):
        raise RuntimeError("checkpoint epochs must be strictly increasing")
    if requested and abs(requested[-1] - float(passes)) > 1e-12:
        raise RuntimeError("the final staged checkpoint must equal --passes")
    return requested


def checkpoint_step_schedule(
    steps_per_pass: int, checkpoint_epochs: tuple[float, ...],
) -> dict[int, float]:
    """Map synchronized optimizer-step endpoints to their cumulative epochs."""
    if steps_per_pass < 1:
        raise ValueError("steps_per_pass must be positive")
    schedule: dict[int, float] = {}
    for epoch in checkpoint_epochs:
        step = int(math.ceil(epoch * steps_per_pass - 1e-12))
        if step in schedule:
            raise RuntimeError("two epoch endpoints round to the same optimizer step")
        schedule[step] = epoch
    return schedule


def epoch_checkpoint_name(epoch: float) -> str:
    label = f"{epoch:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"checkpoint_epoch_{label}.pt"


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


def load_histories(
    uids: list[int], *, oov_buckets: int = 0, dataset_path: Path = DATASET,
    known_vocab_size: int | None = None, start_timestamp: int | None = None,
    end_timestamp: int | None = None, max_history: int | None = None,
    threads: int = 4,
) -> FoundationHistoryIndex:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    known = int(known_vocab_size or dataset["foundation_items"])
    return load_yambda_histories(
        dataset_path, uids, known_vocab_size=known, oov_buckets=oov_buckets,
        start_timestamp=start_timestamp,
        max_pre_events=max_history if start_timestamp is not None else None,
        end_timestamp=int(end_timestamp or (2**63 - 1)),
        threads=threads,
    )


def contract_model_config(launch: dict, *, oov_buckets: int) -> tuple[HSTUConfig, Path, int]:
    frozen = launch.get("frozen_inputs", {})
    dataset_path = (ROOT / frozen.get("dataset_manifest", DATASET)).resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    known = int(dataset["foundation_items"])
    if "model" not in launch:
        return HSTUConfig(
            num_items=known + oov_buckets, num_behaviors=4, hidden_size=128,
            num_layers=4, num_heads=4, max_seq_len=512, num_query_types=3,
            query_type_id=2, num_query_actions=1,
        ), dataset_path, known
    values = dict(launch["model"])
    expected_known = int(values.pop("known_items_from_dataset_manifest"))
    frozen_oov = int(values.pop("oov_buckets"))
    reporting_only = {
        "head_dimension", "total_parameters", "contextual_block_parameters",
        "total_parameter_reporting_must_separate_embedding_and_contextual_blocks",
    }
    unknown = set(values) - {field.name for field in fields(HSTUConfig)} - reporting_only
    if unknown:
        raise RuntimeError(f"unrecognized model contract fields: {sorted(unknown)}")
    for key in reporting_only:
        values.pop(key, None)
    if known != expected_known:
        raise RuntimeError("dataset known vocabulary differs from scale contract")
    if oov_buckets != frozen_oov:
        raise RuntimeError("requested OOV buckets differ from scale contract")
    return HSTUConfig(num_items=known + oov_buckets, **values), dataset_path, known


def save_checkpoint(
    model, rank, output, payload, progress, expected_producer_sha=None,
    *, target_name: str | None = None,
):
    state = full_hstu_state(model, rank)
    if rank == 0:
        producer_sha = cache_producer_sha256(state)
        if expected_producer_sha is not None and producer_sha != expected_producer_sha:
            raise RuntimeError("R0 changed cache-producing parameters")
        state_payload = {**payload, "model": state, "progress": progress,
                         "cache_producer_sha256": producer_sha,
                         "r0_same_producer_invariant": expected_producer_sha is None or producer_sha == expected_producer_sha}
        target = output / (target_name or f"checkpoint_{int(progress * 100):03d}.pt")
        temporary = target.with_suffix(".pt.partial")
        torch.save(state_payload, temporary); os.replace(temporary, target)
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--launch-contract", type=Path, default=LAUNCH)
    parser.add_argument("--execution-contract", type=Path)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--canary-steps", type=int, default=0)
    parser.add_argument("--oov-buckets", type=int, default=0)
    parser.add_argument("--train-start-day", type=int)
    parser.add_argument("--train-end-day", type=int)
    parser.add_argument("--passes", type=int)
    parser.add_argument(
        "--checkpoint-epochs",
        help="comma-separated cumulative endpoints; must exactly match the launch contract",
    )
    parser.add_argument("--training-block")
    parser.add_argument("--branch", choices=("shared", "D7", "D14"), default="shared")
    parser.add_argument("--history-threads", type=int, default=4)
    parser.add_argument("--arrow-cpu-threads", type=int, default=4)
    parser.add_argument("--arrow-io-threads", type=int, default=4)
    parser.add_argument("--torch-cpu-threads", type=int, default=2)
    parser.add_argument("--cpu-affinity-by-rank", help="semicolon-separated comma lists")
    parser.add_argument("--progress-interval", type=int, default=500)
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
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.cpu_affinity_by_rank:
        groups = args.cpu_affinity_by_rank.split(";")
        if local_rank >= len(groups):
            raise ValueError("CPU affinity does not define every local rank")
        os.sched_setaffinity(0, {int(value) for value in groups[local_rank].split(",")})
    if min(args.history_threads, args.arrow_cpu_threads, args.arrow_io_threads, args.torch_cpu_threads) < 1:
        raise ValueError("CPU thread counts must be positive")
    if args.progress_interval < 1:
        raise ValueError("progress-interval must be positive")
    pa.set_cpu_count(args.arrow_cpu_threads)
    pa.set_io_thread_count(args.arrow_io_threads)
    torch.set_num_threads(args.torch_cpu_threads)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        execution_path = args.execution_contract.resolve() if args.execution_contract else None
        execution = validate_execution_contract(execution_path, launch_path, world)
        if args.global_batch_size is not None:
            batch_size = local_batch_size(args.global_batch_size, world, rank)
            expected = execution["execution_amendment"]["local_batch_sizes_by_rank"] if execution else None
            if expected is not None and expected != [local_batch_size(args.global_batch_size, world, value) for value in range(world)]:
                raise RuntimeError("runtime batch partition differs from the execution contract")
            global_batch_size = int(args.global_batch_size)
        else:
            batch_size = int(args.batch_size)
            global_batch_size = batch_size * world
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
        checkpoint_epochs = parse_checkpoint_epochs(args.checkpoint_epochs, recipe, passes)
        if args.canary_steps:
            rows = rows[: max(batch_size, batch_size * args.canary_steps)]
        if args.oov_buckets < 0:
            raise ValueError("oov-buckets must be non-negative")
        cfg, dataset_path, known_vocab_size = contract_model_config(
            launch, oov_buckets=args.oov_buckets
        )
        bounded_start = args.train_start_day * 86_400 if args.train_start_day is not None else None
        bounded_end = args.train_end_day * 86_400 if args.train_end_day is not None else None
        histories = load_histories(
            sorted({int(row["uid"]) for row in rows}), oov_buckets=args.oov_buckets,
            dataset_path=dataset_path, known_vocab_size=known_vocab_size,
            start_timestamp=bounded_start, end_timestamp=bounded_end,
            max_history=cfg.max_seq_len if bounded_start is not None else None,
            threads=args.history_threads,
        )
        training_rows = rows * passes
        seed = int(launch.get("scope", {}).get("seed", 17))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
        if args.version == "v0" and "foundation_learning_rate" in recipe:
            learning_rate = float(recipe["foundation_learning_rate"])
        elif args.version != "v0" and "update_learning_rate" in recipe:
            learning_rate = float(recipe["update_learning_rate"])
        else:
            learning_rate = float(recipe.get("learning_rates", {}).get(args.version, recipe.get("learning_rate", 2e-4)))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate, weight_decay=float(recipe.get("weight_decay", 1e-4)),
        )
        staged_formal = bool(checkpoint_epochs) and not args.canary_steps
        if staged_formal:
            local_steps_per_pass = math.ceil(len(rows) / batch_size)
            step_tensor = torch.tensor(local_steps_per_pass, device=device)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.MAX)
            steps_per_pass = int(step_tensor)
            steps = steps_per_pass * passes
            checkpoint_steps = checkpoint_step_schedule(steps_per_pass, checkpoint_epochs)
        else:
            local_steps = math.ceil(len(training_rows) / batch_size)
            step_tensor = torch.tensor(local_steps, device=device)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.MAX)
            steps = int(step_tensor)
            steps_per_pass = None
            checkpoint_steps = {}
        steps = min(steps, args.canary_steps) if args.canary_steps else steps
        losses = []
        step_seconds: list[float] = []
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(steps):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            if staged_formal:
                assert steps_per_pass is not None
                step_within_pass = step % steps_per_pass
                start = step_within_pass * batch_size
                batch_rows = rows[start:start + batch_size]
                padding_source = rows or batch_rows
            else:
                start = step * batch_size
                batch_rows = training_rows[start:start + batch_size]
                padding_source = training_rows or batch_rows
            if len(batch_rows) < batch_size:
                while len(batch_rows) < batch_size:
                    duplicate = dict(padding_source[len(batch_rows) % len(padding_source)]); duplicate["weight"] = 0.0
                    batch_rows.append(duplicate)
            batch = collate_foundation_batch(
                batch_rows, histories, device=device, max_history=cfg.max_seq_len
            )
            hstu_logits = model(batch.item_ids, batch.behaviors, batch.time_deltas, batch.candidate_ids, batch.query_time_deltas, batch.lengths)[:, 0]
            per_request = F.binary_cross_entropy_with_logits(hstu_logits, batch.labels, reduction="none")
            # FSDP averages gradients across ranks. This scaling produces the
            # same global-batch mean for any contract-authorized partition,
            # while zero-weight padding remains neutral.
            loss = (per_request * batch.weights).sum() * (world / global_batch_size)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - started)
            losses.append(float(loss.detach()))
            completed = step + 1
            is_checkpoint_step = completed in checkpoint_steps or completed == steps
            if rank == 0 and (
                completed % args.progress_interval == 0 or is_checkpoint_step
            ):
                timed = step_seconds[1:] if len(step_seconds) > 1 else step_seconds
                median_step = float(statistics.median(timed))
                progress = {
                    "status": "saving_checkpoint" if is_checkpoint_step else "formal_training_in_progress",
                    "contract_sha256": sha256_file(launch_path),
                    "execution_contract_sha256": sha256_file(execution_path) if execution_path else None,
                    "version": args.version,
                    "branch": args.branch,
                    "completed_steps": completed,
                    "total_steps": steps,
                    "progress_fraction": completed / steps,
                    "global_batch_size": global_batch_size,
                    "completed_global_requests_including_padding": completed * global_batch_size,
                    "effective_training_examples": total_requests * passes,
                    "checkpoint_epochs": list(checkpoint_epochs),
                    "latest_rank0_loss": losses[-1],
                    "mean_rank0_loss_so_far": float(np.mean(losses)),
                    "median_rank0_step_seconds": median_step,
                    "estimated_remaining_seconds": (steps - completed) * median_step,
                    "peak_reserved_mib_rank0": float(torch.cuda.max_memory_reserved(device) / 2**20),
                    "updated_at_unix_seconds": time.time(),
                }
                atomic_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if is_checkpoint_step:
                completed_epoch = (
                    0.0 if args.canary_steps else
                    checkpoint_steps[completed] if completed in checkpoint_steps else
                    float(passes)
                )
                checkpoint_name = (
                    epoch_checkpoint_name(completed_epoch)
                    if staged_formal else "checkpoint_100.pt"
                )
                payload = {
                    "status": (
                        "distributed_canary_checkpoint" if args.canary_steps
                        else "formal_staged_epoch_checkpoint" if staged_formal
                        else "formal_scale_seed17_checkpoint" if "model" in launch
                        else "formal_small_seed17_checkpoint"
                    ),
                    "contract": str(launch_path.relative_to(ROOT)), "contract_sha256": sha256_file(launch_path),
                    "architecture": "hstu_native_cc", "version": args.version,
                    "branch": args.branch, "window": block, "seed": seed,
                    "training_day_range": [args.train_start_day, args.train_end_day],
                    "config": asdict(cfg), "parent_checkpoint_sha256": parent_hash,
                    "request_manifest_sha256": sha256_file(request_path),
                    "world_size": world, "batch_size_per_rank": batch_size,
                    "local_batch_sizes_by_rank": [local_batch_size(global_batch_size, world, value) for value in range(world)],
                    "global_batch_size": global_batch_size,
                    "execution_contract": str(execution_path.relative_to(ROOT)) if execution_path else None,
                    "execution_contract_sha256": sha256_file(execution_path) if execution_path else None,
                    "total_requests": total_requests, "total_users": total_users,
                    "passes": passes, "effective_training_examples": total_requests * passes,
                    "training_epochs_completed": completed_epoch,
                    "training_epoch_target": float(passes),
                    "staged_checkpoint_epochs": list(checkpoint_epochs),
                    "completed_steps": completed,
                    "steps_per_pass": steps_per_pass,
                    "effective_training_examples_completed": int(round(total_requests * completed_epoch)),
                    "oov_buckets": args.oov_buckets,
                    "known_vocab_size": known_vocab_size,
                    "dataset_manifest": str(dataset_path.relative_to(ROOT)),
                    "dataset_manifest_sha256": sha256_file(dataset_path),
                    "learning_rate": learning_rate,
                }
                save_checkpoint(model, rank, args.output, payload, 1.0,
                                parent_producer_sha if args.version == "r0" else None,
                                target_name=checkpoint_name)
        warmup = 1 if len(step_seconds) > 1 else 0
        rank_metrics = {
            "rank": rank,
            "local_batch_size": batch_size,
            "median_timed_step_seconds": float(statistics.median(step_seconds[warmup:])),
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
        }
        gathered: list[dict | None] = [None] * world
        dist.all_gather_object(gathered, rank_metrics)
        if rank == 0:
            synchronized_step_seconds = max(float(value["median_timed_step_seconds"]) for value in gathered if value is not None)
            final_checkpoint = (
                args.output / epoch_checkpoint_name(checkpoint_epochs[-1])
                if staged_formal else args.output / "checkpoint_100.pt"
            )
            result = {
                "status": "distributed_canary_passed" if args.canary_steps else "formal_training_complete",
                "contract_sha256": sha256_file(launch_path),
                "version": args.version, "steps": steps, "mean_rank0_loss": float(np.mean(losses)),
                "final_checkpoint": str(final_checkpoint),
                "checkpoint_epochs": list(checkpoint_epochs),
                "checkpoints": (
                    {
                        str(epoch): str(args.output / epoch_checkpoint_name(epoch))
                        for epoch in checkpoint_epochs
                    }
                    if staged_formal else {"final": str(final_checkpoint)}
                ),
                "world_size": world, "global_batch_size": global_batch_size,
                "local_batch_sizes_by_rank": [local_batch_size(global_batch_size, world, value) for value in range(world)],
                "median_synchronized_step_seconds": synchronized_step_seconds,
                "global_requests_per_second": global_batch_size / synchronized_step_seconds,
                "rank_metrics": gathered,
                "execution_contract_sha256": sha256_file(execution_path) if execution_path else None,
                "theta3_read_or_trained": bool(launch.get("scope", {}).get("theta3_read_or_trained", False)),
            }
            (args.output / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
            atomic_json(args.output / "progress.json", {
                "status": "formal_training_complete" if not args.canary_steps else "distributed_canary_complete",
                "contract_sha256": sha256_file(launch_path),
                "execution_contract_sha256": sha256_file(execution_path) if execution_path else None,
                "version": args.version, "branch": args.branch,
                "completed_steps": steps, "total_steps": steps, "progress_fraction": 1.0,
                "median_synchronized_step_seconds": synchronized_step_seconds,
                "global_requests_per_second": global_batch_size / synchronized_step_seconds,
                "updated_at_unix_seconds": time.time(),
            })
            print(json.dumps(result, indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
