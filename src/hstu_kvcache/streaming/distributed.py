from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .trainer import build_next_item_targets, train_step


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    initialized: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def init_distributed_runtime(
    device: str,
    backend: str = "nccl",
) -> DistributedRuntime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        resolved = torch.device(device)
        if resolved.type == "cuda":
            torch.cuda.set_device(resolved)
        return DistributedRuntime(rank, world_size, local_rank, resolved, False)
    if not torch.cuda.is_available():
        raise RuntimeError("distributed execution requires CUDA")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedRuntime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
        initialized=True,
    )


def close_distributed_runtime(runtime: DistributedRuntime) -> None:
    if runtime.initialized:
        dist.destroy_process_group()


def primary_log(runtime: DistributedRuntime, message: str) -> None:
    if runtime.is_primary:
        print(message, flush=True)


def batch_target_count(batch: dict) -> int:
    _, valid = build_next_item_targets(
        batch["item_ids"],
        batch["lengths"],
        batch.get("labels"),
        batch.get("train_mask"),
    )
    return int(valid.sum())


def batch_coverage(batches: list[dict]) -> dict[str, int]:
    return {
        "batches": len(batches),
        "sequences": sum(len(batch["lengths"]) for batch in batches),
        "tokens": sum(int(batch["lengths"].sum()) for batch in batches),
        "eligible_targets": sum(batch_target_count(batch) for batch in batches),
    }


def shard_train_batches(
    batches: list[dict],
    runtime: DistributedRuntime,
) -> tuple[list[tuple[dict, float]], dict]:
    eligible = [batch for batch in batches if batch_target_count(batch) > 0]
    if not eligible:
        raise RuntimeError("no eligible training batches")
    steps = (len(eligible) + runtime.world_size - 1) // runtime.world_size
    local = []
    for step in range(steps):
        index = step * runtime.world_size + runtime.rank
        active = min(
            runtime.world_size,
            len(eligible) - step * runtime.world_size,
        )
        if index < len(eligible):
            local.append((eligible[index], runtime.world_size / active))
        else:
            source = eligible[0]
            dummy = dict(source)
            dummy["labels"] = torch.zeros_like(source["item_ids"])
            if "train_mask" in source:
                dummy["train_mask"] = torch.zeros_like(
                    source["train_mask"],
                    dtype=torch.bool,
                )
            local.append((dummy, 1.0))
    coverage = batch_coverage(eligible)
    coverage["zero_target_batches_removed"] = len(batches) - len(eligible)
    coverage["distributed_padding_batches"] = (
        steps * runtime.world_size - len(eligible)
    )
    coverage["steps_per_rank"] = len(local)
    return local, coverage


def reduce_loss(losses: list[float], runtime: DistributedRuntime) -> float:
    values = torch.tensor(
        [sum(losses), len(losses)],
        dtype=torch.float64,
        device=runtime.device,
    )
    if runtime.initialized:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if values[1].item() == 0:
        return float("nan")
    return float((values[0] / values[1]).item())


def train_distributed_batches(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[tuple[dict, float]],
    runtime: DistributedRuntime,
    micro_batch_size: int | None = None,
) -> float:
    if micro_batch_size is not None and micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive")
    losses = []
    for batch, loss_scale in batches:
        logical_batch_size = len(batch["item_ids"])
        if micro_batch_size is None or micro_batch_size >= logical_batch_size:
            losses.append(
                train_step(
                    model,
                    batch,
                    optimizer,
                    runtime.device,
                    loss_scale=loss_scale,
                )
            )
            continue
        micro_batches = []
        for start in range(0, logical_batch_size, micro_batch_size):
            end = min(start + micro_batch_size, logical_batch_size)
            micro_batches.append(
                {
                    name: (
                        value[start:end]
                        if isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and len(value) == logical_batch_size
                        else value
                    )
                    for name, value in batch.items()
                }
            )
        target_counts = [batch_target_count(value) for value in micro_batches]
        total_targets = sum(target_counts)
        weighted_loss = 0.0
        for index, (micro_batch, target_count) in enumerate(
            zip(micro_batches, target_counts, strict=True)
        ):
            final_micro_batch = index == len(micro_batches) - 1
            sync_context = (
                nullcontext()
                if final_micro_batch or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            gradient_weight = (
                target_count / total_targets
                if total_targets > 0
                else 0.0
            )
            with sync_context:
                loss = train_step(
                    model,
                    micro_batch,
                    optimizer,
                    runtime.device,
                    loss_scale=loss_scale * gradient_weight,
                    zero_grad=index == 0,
                    optimizer_step=final_micro_batch,
                )
            weighted_loss += loss * gradient_weight
        losses.append(weighted_loss)
    return reduce_loss([loss for loss in losses if loss > 0], runtime)


def gather_records(
    local_records: list[dict],
    runtime: DistributedRuntime,
    sort_key: str = "user_id",
) -> list[dict] | None:
    if not runtime.initialized:
        return sorted(local_records, key=lambda record: record[sort_key])
    gathered: list[list[dict] | None] | None
    if runtime.is_primary:
        gathered = [None] * runtime.world_size
    else:
        gathered = None
    dist.gather_object(local_records, gathered, dst=0)
    if not runtime.is_primary:
        return None
    records = [
        record
        for shard in gathered
        if shard is not None
        for record in shard
    ]
    return sorted(records, key=lambda record: record[sort_key])
