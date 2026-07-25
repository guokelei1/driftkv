from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models import HSTU
from .capsule import MigratedKVBatch, MigrationCapsuleBatch
from .low_rank import (
    CompiledCacheAdapter,
    LowRankCacheAdapter,
    compile_low_rank_cache_adapter,
    compile_projection_cache_adapter,
    migrate_compiled_cache_capsule,
)


@dataclass(frozen=True)
class MigrationProgram:
    source_version: str
    target_version: str
    adapter: CompiledCacheAdapter

    def __post_init__(self) -> None:
        if not self.source_version or not self.target_version:
            raise ValueError("source_version and target_version must be nonempty")
        if self.adapter.weights.ndim != 3:
            raise ValueError("program weights must have shape [layers, hidden, 2 * kv_width]")
        if self.adapter.biases.ndim != 2:
            raise ValueError("program biases must have shape [layers, 2 * kv_width]")
        if self.adapter.weights.shape[0] == 0:
            raise ValueError("program must contain at least one layer")
        if self.adapter.weights.shape[1] == 0 or self.adapter.weights.shape[2] == 0:
            raise ValueError("program projection dimensions must be positive")
        if self.adapter.weights.shape[2] % 2:
            raise ValueError("program output width must split evenly into K and V")
        if self.adapter.biases.shape != (
            self.adapter.weights.shape[0],
            self.adapter.weights.shape[2],
        ):
            raise ValueError("program weight and bias shapes differ")
        if self.adapter.weights.device != self.adapter.biases.device:
            raise ValueError("program weights and biases must be on the same device")

    @property
    def cohort_key(self) -> tuple[str, str]:
        return self.source_version, self.target_version

    @property
    def device(self) -> torch.device:
        return self.adapter.weights.device

    @property
    def num_layers(self) -> int:
        return self.adapter.weights.shape[0]

    @property
    def input_width(self) -> int:
        return self.adapter.weights.shape[1]

    @property
    def kv_width(self) -> int:
        return self.adapter.weights.shape[2] // 2

    @property
    def nbytes(self) -> int:
        return self.adapter.nbytes

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
        dtype: torch.dtype | None = None,
    ) -> MigrationProgram:
        return MigrationProgram(
            source_version=self.source_version,
            target_version=self.target_version,
            adapter=CompiledCacheAdapter(
                weights=self.adapter.weights.to(
                    device=device,
                    dtype=dtype,
                    non_blocking=non_blocking,
                ),
                biases=self.adapter.biases.to(
                    device=device,
                    dtype=dtype,
                    non_blocking=non_blocking,
                ),
                source_rank=self.adapter.source_rank,
                ridge=self.adapter.ridge,
            ),
        )

    def validate_capsule(self, capsule: MigrationCapsuleBatch) -> None:
        if capsule.migration_anchor_version != self.source_version:
            raise ValueError("capsule migration anchor does not match program source")
        if capsule.num_layers != self.num_layers:
            raise ValueError("capsule and program depths differ")
        if capsule.hidden_size != self.input_width:
            raise ValueError("capsule hidden width and program input width differ")
        if capsule.device != self.device:
            raise ValueError("capsule and program must be on the same device")


@torch.no_grad()
def compile_migration_program(
    model: HSTU,
    source_version: str,
    target_version: str,
    adapter: LowRankCacheAdapter | None = None,
) -> MigrationProgram:
    compiled = (
        compile_projection_cache_adapter(model)
        if adapter is None
        else compile_low_rank_cache_adapter(model, adapter)
    )
    return MigrationProgram(
        source_version=source_version,
        target_version=target_version,
        adapter=compiled,
    )


@torch.no_grad()
def execute_migration_program(
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
) -> MigratedKVBatch:
    program.validate_capsule(capsule)
    cache = migrate_compiled_cache_capsule(capsule, program.adapter)
    return MigratedKVBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        served_kv_target=program.target_version,
        cache=cache,
        lengths=capsule.lengths,
    )
