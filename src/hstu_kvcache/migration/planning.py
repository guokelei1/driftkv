from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..models import HSTUKVCache
from .capsule import MigratedKVBatch, MigrationCapsuleBatch


@dataclass(frozen=True)
class CohortBatchPlan:
    organization: str
    source_record_ids: tuple[int, ...]
    source_seq_len: int
    source_lengths: torch.Tensor
    batches: tuple[MigrationCapsuleBatch, ...]

    def __post_init__(self) -> None:
        if not self.organization:
            raise ValueError("organization must be nonempty")
        if not self.batches:
            raise ValueError("cohort plan must contain at least one batch")
        if self.source_lengths.shape != (len(self.source_record_ids),):
            raise ValueError("source lengths and record IDs differ")
        planned_ids = tuple(
            record_id
            for batch in self.batches
            for record_id in batch.record_ids
        )
        if len(planned_ids) != len(set(planned_ids)):
            raise ValueError("planned record IDs must be unique")
        if set(planned_ids) != set(self.source_record_ids):
            raise ValueError("planned record IDs do not cover the source cohort")
        if any(batch.seq_len > self.source_seq_len for batch in self.batches):
            raise ValueError("planned batch exceeds the source sequence width")
        anchors = {
            batch.migration_anchor_version
            for batch in self.batches
        }
        if len(anchors) != 1:
            raise ValueError("all planned batches must share one migration anchor")

    @property
    def record_count(self) -> int:
        return len(self.source_record_ids)

    @property
    def payload_nbytes(self) -> int:
        return sum(batch.nbytes for batch in self.batches)

    @property
    def padded_tokens(self) -> int:
        return sum(
            batch.batch_size * batch.seq_len - int(batch.lengths.sum().item())
            for batch in self.batches
        )

    @torch.no_grad()
    def restore_logical_order(
        self,
        results: tuple[MigratedKVBatch, ...],
    ) -> MigratedKVBatch:
        if len(results) != len(self.batches):
            raise ValueError("result count and planned batch count differ")
        first = results[0]
        layers, _, _, width = first.cache.k.shape
        device = first.cache.k.device
        dtype = first.cache.k.dtype
        restored_k = torch.zeros(
            layers,
            self.record_count,
            self.source_seq_len,
            width,
            device=device,
            dtype=dtype,
        )
        restored_v = torch.zeros_like(restored_k)
        logical_positions = {
            record_id: index
            for index, record_id in enumerate(self.source_record_ids)
        }
        for planned, result in zip(self.batches, results, strict=True):
            if result.record_ids != planned.record_ids:
                raise ValueError("result record IDs differ from the cohort plan")
            if result.cache.k.device != device or result.cache.k.dtype != dtype:
                raise ValueError("result batches must share device and dtype")
            positions = torch.tensor(
                [logical_positions[value] for value in result.record_ids],
                device=device,
            )
            restored_k[:, positions, : result.cache.seq_len] = result.cache.k
            restored_v[:, positions, : result.cache.seq_len] = result.cache.v
        return MigratedKVBatch(
            record_ids=self.source_record_ids,
            migration_anchor_version=first.migration_anchor_version,
            served_kv_target=first.served_kv_target,
            cache=HSTUKVCache(
                k=restored_k,
                v=restored_v,
                seq_len=self.source_seq_len,
            ),
            lengths=self.source_lengths.to(device),
        )


def build_contiguous_cohort_plan(
    capsule: MigrationCapsuleBatch,
    max_records: int,
) -> CohortBatchPlan:
    return CohortBatchPlan(
        organization="contiguous",
        source_record_ids=capsule.record_ids,
        source_seq_len=capsule.seq_len,
        source_lengths=capsule.lengths,
        batches=capsule.split(max_records),
    )


def build_length_bucketed_cohort_plan(
    capsule: MigrationCapsuleBatch,
    max_records: int,
    bucket_width: int,
    trim_padding: bool = True,
) -> CohortBatchPlan:
    if max_records < 1 or bucket_width < 1:
        raise ValueError("max_records and bucket_width must be positive")
    lengths = capsule.lengths.detach().cpu().tolist()
    order = sorted(
        range(capsule.batch_size),
        key=lambda index: (
            math.ceil(lengths[index] / bucket_width),
            lengths[index],
            index,
        ),
    )
    batches = []
    for start in range(0, len(order), max_records):
        selected = order[start : start + max_records]
        indices = torch.tensor(selected, device=capsule.device)
        selected_lengths = capsule.lengths.index_select(0, indices)
        seq_len = capsule.seq_len
        if trim_padding:
            maximum = max(1, int(selected_lengths.max().item()))
            seq_len = min(
                capsule.seq_len,
                math.ceil(maximum / bucket_width) * bucket_width,
            )
        batches.append(
            MigrationCapsuleBatch(
                record_ids=tuple(capsule.record_ids[index] for index in selected),
                migration_anchor_version=capsule.migration_anchor_version,
                normed=capsule.normed.index_select(1, indices)[
                    :, :, :seq_len
                ].contiguous(),
                lengths=selected_lengths.contiguous(),
            )
        )
    return CohortBatchPlan(
        organization=f"length_bucketed_{bucket_width}",
        source_record_ids=capsule.record_ids,
        source_seq_len=capsule.seq_len,
        source_lengths=capsule.lengths,
        batches=tuple(batches),
    )
