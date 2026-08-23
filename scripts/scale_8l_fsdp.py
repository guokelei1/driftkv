#!/usr/bin/env python3
"""FSDP primitives for the 8L scale reproduction."""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)


class CandidateConditionedForward(nn.Module):
    """Expose the frozen CC scoring path as an FSDP-compatible forward."""

    def __init__(self, hstu: nn.Module) -> None:
        super().__init__()
        self.hstu = hstu

    def forward(
        self,
        items: torch.Tensor,
        behaviors: torch.Tensor,
        deltas: torch.Tensor,
        candidates: torch.Tensor,
        query_deltas: torch.Tensor,
        lengths: torch.Tensor,
        query_types: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chunks = self.hstu.score_cc_full_chunked(
                    items, behaviors, deltas, candidates, query_deltas,
                    chunk_size=chunk_size, lengths=lengths,
                    query_type_ids=query_types,
                )
        return torch.cat(chunks, dim=1).float()


def initialize() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch distributed scale jobs with torchrun")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return rank, world, torch.device(f"cuda:{local}")


def wrap(model: nn.Module, device: torch.device) -> FSDP:
    return FSDP(
        CandidateConditionedForward(model),
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=True,
    )


def finish() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def full_hstu_state_dict(model: FSDP) -> dict[str, torch.Tensor]:
    """Collect a CPU full HSTU state dict on rank zero only."""
    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
        wrapped = model.state_dict()
    if dist.get_rank() != 0:
        return {}
    prefix = "hstu."
    output = {
        (name[len(prefix):] if name.startswith(prefix) else name): value.cpu()
        for name, value in wrapped.items()
    }
    if not output or any(name.startswith("_fsdp") for name in output):
        raise RuntimeError("unexpected FSDP checkpoint key layout")
    return output
