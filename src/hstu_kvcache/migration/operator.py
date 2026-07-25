from __future__ import annotations

from typing import Protocol

import torch

from ..models import HSTUKVCache
from .capsule import MigratedKVBatch, MigrationCapsuleBatch
from .low_rank import CompiledCacheAdapter
from .program import MigrationProgram, execute_migration_program


class MigrationOperator(Protocol):
    @property
    def name(self) -> str: ...

    def prepare_program(
        self,
        program: MigrationProgram,
        device: torch.device | str,
    ) -> MigrationProgram: ...

    def execute(
        self,
        program: MigrationProgram,
        capsule: MigrationCapsuleBatch,
    ) -> MigratedKVBatch: ...


class ReferenceMigrationOperator:
    @property
    def name(self) -> str:
        return "reference_fp32"

    def prepare_program(
        self,
        program: MigrationProgram,
        device: torch.device | str,
    ) -> MigrationProgram:
        return program.to(device, dtype=torch.float32)

    @torch.no_grad()
    def execute(
        self,
        program: MigrationProgram,
        capsule: MigrationCapsuleBatch,
    ) -> MigratedKVBatch:
        return execute_migration_program(program, capsule)


class PackedMigrationOperator:
    def __init__(self, execution_dtype: torch.dtype = torch.float16) -> None:
        if execution_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("packed operator dtype must be float16, bfloat16, or float32")
        self.execution_dtype = execution_dtype

    @property
    def name(self) -> str:
        return f"packed_{str(self.execution_dtype).removeprefix('torch.')}"

    def prepare_program(
        self,
        program: MigrationProgram,
        device: torch.device | str,
    ) -> MigrationProgram:
        return program.to(device, dtype=self.execution_dtype)

    @torch.no_grad()
    def execute(
        self,
        program: MigrationProgram,
        capsule: MigrationCapsuleBatch,
    ) -> MigratedKVBatch:
        program.validate_capsule(capsule)
        cache = migrate_packed_cache_capsule(capsule, program.adapter)
        return MigratedKVBatch(
            record_ids=capsule.record_ids,
            migration_anchor_version=capsule.migration_anchor_version,
            served_kv_target=program.target_version,
            cache=cache,
            lengths=capsule.lengths,
        )


@torch.no_grad()
def migrate_packed_cache_capsule(
    capsule: MigrationCapsuleBatch,
    adapter: CompiledCacheAdapter,
) -> HSTUKVCache:
    normed = capsule.normed
    if normed.shape[0] != adapter.weights.shape[0]:
        raise ValueError("capsule and packed adapter depths differ")
    if normed.shape[-1] != adapter.weights.shape[1]:
        raise ValueError("capsule and packed adapter widths differ")
    if normed.device != adapter.weights.device or normed.device != adapter.biases.device:
        raise ValueError("capsule and packed adapter must share a device")
    flattened = normed.to(adapter.weights.dtype).flatten(1, 2)
    bias = adapter.biases[:, None, :].expand(
        -1,
        flattened.shape[1],
        -1,
    )
    projected = torch.baddbmm(bias, flattened, adapter.weights)
    projected = projected.unflatten(1, normed.shape[1:3])
    positions = torch.arange(normed.shape[2], device=normed.device)
    invalid = positions.unsqueeze(0) >= capsule.lengths.unsqueeze(1)
    projected.masked_fill_(invalid.unsqueeze(0).unsqueeze(-1), 0)
    projected = projected.to(normed.dtype)
    width = projected.shape[-1] // 2
    return HSTUKVCache(
        k=projected[..., :width],
        v=projected[..., width:],
        seq_len=normed.shape[2],
    )
