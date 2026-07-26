from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from ..models import HSTUKVCache
from .layerwise import LayerwiseCacheState


class KVOutputExtent(Protocol):
    @property
    def record_ids(self) -> tuple[int, ...]: ...

    @property
    def migration_anchor_version(self) -> str: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def seq_len(self) -> int: ...

    @property
    def lengths(self) -> torch.Tensor: ...


@dataclass(frozen=True)
class MigrationCapsuleBatch:
    record_ids: tuple[int, ...]
    migration_anchor_version: str
    normed: torch.Tensor
    lengths: torch.Tensor

    def __post_init__(self) -> None:
        if not self.migration_anchor_version:
            raise ValueError("migration_anchor_version must be nonempty")
        if self.normed.ndim != 4:
            raise ValueError("normed must have shape [layers, batch, sequence, hidden]")
        if self.lengths.ndim != 1:
            raise ValueError("lengths must be one-dimensional")
        if not self.normed.is_floating_point():
            raise ValueError("normed must have a floating-point dtype")
        if self.lengths.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("lengths must have an integer dtype")
        if self.normed.shape[0] == 0:
            raise ValueError("capsule must contain at least one layer")
        if self.normed.shape[1] == 0:
            raise ValueError("capsule must contain at least one record")
        if self.normed.shape[2] == 0 or self.normed.shape[3] == 0:
            raise ValueError("sequence and hidden dimensions must be positive")
        if self.normed.shape[1] != self.lengths.shape[0]:
            raise ValueError("normed batch dimension and lengths differ")
        if len(self.record_ids) != self.normed.shape[1]:
            raise ValueError("record_ids and normed batch dimension differ")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("record_ids must be unique within a batch")
        if self.normed.device != self.lengths.device:
            raise ValueError("normed and lengths must be on the same device")
        if self.device.type == "cpu":
            if bool(torch.any(self.lengths < 0)) or bool(
                torch.any(self.lengths > self.normed.shape[2])
            ):
                raise ValueError("lengths must be within the padded sequence width")

    @classmethod
    def from_layerwise_state(
        cls,
        state: LayerwiseCacheState,
        migration_anchor_version: str,
        record_ids: Sequence[int] | None = None,
    ) -> MigrationCapsuleBatch:
        normed = torch.stack(state.normed_states)
        if normed.shape[2] != state.kv.seq_len:
            raise ValueError("normalized state and K/V sequence widths differ")
        if record_ids is None:
            record_ids = range(normed.shape[1])
        return cls(
            record_ids=tuple(record_ids),
            migration_anchor_version=migration_anchor_version,
            normed=normed,
            lengths=state.lengths,
        )

    @property
    def device(self) -> torch.device:
        return self.normed.device

    @property
    def num_layers(self) -> int:
        return self.normed.shape[0]

    @property
    def batch_size(self) -> int:
        return self.normed.shape[1]

    @property
    def seq_len(self) -> int:
        return self.normed.shape[2]

    @property
    def hidden_size(self) -> int:
        return self.normed.shape[3]

    @property
    def nbytes(self) -> int:
        return (
            self.normed.numel() * self.normed.element_size()
            + self.lengths.numel() * self.lengths.element_size()
        )

    @property
    def is_pinned(self) -> bool:
        return (
            self.device.type == "cpu"
            and self.normed.is_pinned()
            and self.lengths.is_pinned()
        )

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> MigrationCapsuleBatch:
        return MigrationCapsuleBatch(
            record_ids=self.record_ids,
            migration_anchor_version=self.migration_anchor_version,
            normed=self.normed.to(device, non_blocking=non_blocking),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> MigrationCapsuleBatch:
        if self.device.type != "cpu":
            raise ValueError("only CPU capsules can be pinned")
        if self.is_pinned:
            return self
        return MigrationCapsuleBatch(
            record_ids=self.record_ids,
            migration_anchor_version=self.migration_anchor_version,
            normed=self.normed.pin_memory(),
            lengths=self.lengths.pin_memory(),
        )

    def split(self, max_records: int) -> tuple[MigrationCapsuleBatch, ...]:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        return tuple(
            MigrationCapsuleBatch(
                record_ids=self.record_ids[start : start + max_records],
                migration_anchor_version=self.migration_anchor_version,
                normed=self.normed[:, start : start + max_records].contiguous(),
                lengths=self.lengths[start : start + max_records].contiguous(),
            )
            for start in range(0, self.batch_size, max_records)
        )


@dataclass(frozen=True)
class MigratedKVBatch:
    record_ids: tuple[int, ...]
    migration_anchor_version: str
    served_kv_target: str
    cache: HSTUKVCache
    lengths: torch.Tensor

    def __post_init__(self) -> None:
        if not self.migration_anchor_version or not self.served_kv_target:
            raise ValueError("migration and serving versions must be nonempty")
        if self.cache.k.ndim != 4 or self.cache.v.ndim != 4:
            raise ValueError("K/V tensors must have shape [layers, batch, sequence, width]")
        if self.cache.k.shape != self.cache.v.shape:
            raise ValueError("K and V shapes differ")
        if len(self.record_ids) != self.cache.k.shape[1]:
            raise ValueError("record_ids and K/V batch dimension differ")
        if self.lengths.shape != (self.cache.k.shape[1],):
            raise ValueError("lengths and K/V batch dimension differ")
        if self.cache.seq_len != self.cache.k.shape[2]:
            raise ValueError("cache seq_len and tensor sequence width differ")
        if self.cache.k.device != self.lengths.device:
            raise ValueError("cache and lengths must be on the same device")

    @property
    def batch_size(self) -> int:
        return self.cache.k.shape[1]

    @property
    def nbytes(self) -> int:
        return (
            self.cache.k.numel() * self.cache.k.element_size()
            + self.cache.v.numel() * self.cache.v.element_size()
            + self.lengths.numel() * self.lengths.element_size()
        )

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> MigratedKVBatch:
        return MigratedKVBatch(
            record_ids=self.record_ids,
            migration_anchor_version=self.migration_anchor_version,
            served_kv_target=self.served_kv_target,
            cache=HSTUKVCache(
                k=self.cache.k.to(device, non_blocking=non_blocking),
                v=self.cache.v.to(device, non_blocking=non_blocking),
                seq_len=self.cache.seq_len,
            ),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class PinnedKVOutputPool:
    served_kv_target: str
    outputs: dict[tuple[int, ...], MigratedKVBatch]

    def __post_init__(self) -> None:
        if not self.served_kv_target:
            raise ValueError("served_kv_target must be nonempty")
        if not self.outputs:
            raise ValueError("output pool must contain at least one extent")
        seen: set[int] = set()
        for record_ids, output in self.outputs.items():
            if record_ids != output.record_ids:
                raise ValueError("output-pool key and record IDs differ")
            if output.served_kv_target != self.served_kv_target:
                raise ValueError("output-pool target versions differ")
            overlap = seen.intersection(record_ids)
            if overlap:
                raise ValueError("output-pool record IDs must be globally unique")
            seen.update(record_ids)
            if output.cache.k.device.type != "cpu":
                raise ValueError("output-pool K/V must be CPU resident")
            if not output.cache.k.is_pinned() or not output.cache.v.is_pinned():
                raise ValueError("output-pool K/V must be pinned")

    @classmethod
    def allocate(
        cls,
        batches: Sequence[KVOutputExtent],
        served_kv_target: str,
        num_layers: int,
        kv_width: int,
        dtype: torch.dtype = torch.float16,
    ) -> PinnedKVOutputPool:
        if not batches:
            raise ValueError("cannot allocate an empty output pool")
        if num_layers < 1 or kv_width < 1:
            raise ValueError("output-pool dimensions must be positive")
        outputs = {}
        for batch in batches:
            if batch.device.type != "cpu":
                raise ValueError("output-pool source batches must be CPU resident")
            shape = (
                num_layers,
                batch.batch_size,
                batch.seq_len,
                kv_width,
            )
            k = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            v = torch.empty_like(k, pin_memory=True)
            lengths = (
                batch.lengths
                if batch.lengths.is_pinned()
                else batch.lengths.pin_memory()
            )
            outputs[batch.record_ids] = MigratedKVBatch(
                record_ids=batch.record_ids,
                migration_anchor_version=batch.migration_anchor_version,
                served_kv_target=served_kv_target,
                cache=HSTUKVCache(k=k, v=v, seq_len=batch.seq_len),
                lengths=lengths,
            )
        return cls(
            served_kv_target=served_kv_target,
            outputs=outputs,
        )

    @property
    def nbytes(self) -> int:
        return sum(output.nbytes for output in self.outputs.values())

    def acquire(
        self,
        capsule: KVOutputExtent,
    ) -> MigratedKVBatch:
        try:
            output = self.outputs[capsule.record_ids]
        except KeyError as exc:
            raise KeyError("capsule extent is absent from output pool") from exc
        if output.migration_anchor_version != capsule.migration_anchor_version:
            raise ValueError("output-pool migration anchor differs from capsule")
        if output.batch_size != capsule.batch_size:
            raise ValueError("output-pool batch size differs from capsule")
        if output.cache.seq_len != capsule.seq_len:
            raise ValueError("output-pool sequence width differs from capsule")
        if not torch.equal(output.lengths, capsule.lengths):
            raise ValueError("output-pool lengths differ from source extent")
        return output
