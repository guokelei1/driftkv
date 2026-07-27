from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from ..models import HSTUKVCache
from .cohort_jagged import JaggedMigratedKVBatch
from .lifecycle import (
    CacheLifecycleState,
    LifecycleDecision,
    LifecyclePolicy,
)

_T = TypeVar("_T")


def _offsets(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.long,
                device=lengths.device,
            ),
            lengths.long().cumsum(0),
        )
    )


def pack_padded_cache(
    cache: HSTUKVCache,
    lengths: torch.Tensor,
    record_ids: tuple[int, ...],
    source_version: str,
    target_version: str,
    dtype: torch.dtype = torch.float16,
) -> JaggedMigratedKVBatch:
    prepared_lengths = lengths.to(
        device=cache.k.device,
        dtype=torch.long,
    )
    if (
        cache.k.ndim != 4
        or cache.k.shape != cache.v.shape
        or cache.k.shape[1] != len(record_ids)
        or prepared_lengths.shape != (len(record_ids),)
        or bool(torch.any(prepared_lengths < 1))
        or bool(torch.any(prepared_lengths > cache.k.shape[2]))
    ):
        raise ValueError("padded cache inputs differ")
    packed_k = torch.cat(
        [
            cache.k[:, row, : int(length)].to(dtype=dtype)
            for row, length in enumerate(prepared_lengths.tolist())
        ],
        dim=1,
    ).contiguous()
    packed_v = torch.cat(
        [
            cache.v[:, row, : int(length)].to(dtype=dtype)
            for row, length in enumerate(prepared_lengths.tolist())
        ],
        dim=1,
    ).contiguous()
    return JaggedMigratedKVBatch(
        record_ids=record_ids,
        migration_anchor_version=source_version,
        served_kv_target=target_version,
        k=packed_k,
        v=packed_v,
        lengths=prepared_lengths,
        offsets=_offsets(prepared_lengths),
    )


def unpack_jagged_cache(
    cache: JaggedMigratedKVBatch,
    dtype: torch.dtype = torch.float32,
) -> HSTUKVCache:
    width = int(cache.lengths.max())
    shape = (
        cache.k.shape[0],
        cache.batch_size,
        width,
        cache.k.shape[2],
    )
    k = torch.zeros(shape, dtype=dtype, device=cache.k.device)
    v = torch.zeros_like(k)
    for row in range(cache.batch_size):
        start = int(cache.offsets[row])
        end = int(cache.offsets[row + 1])
        length = end - start
        k[:, row, :length].copy_(cache.k[:, start:end])
        v[:, row, :length].copy_(cache.v[:, start:end])
    return HSTUKVCache(k=k, v=v, seq_len=width)


def select_jagged_rows(
    cache: JaggedMigratedKVBatch,
    rows: tuple[int, ...],
) -> JaggedMigratedKVBatch:
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(not 0 <= row < cache.batch_size for row in rows)
    ):
        raise ValueError("jagged row selection is invalid")
    selected_k = []
    selected_v = []
    selected_lengths = []
    selected_ids = []
    for row in rows:
        start = int(cache.offsets[row])
        end = int(cache.offsets[row + 1])
        selected_k.append(cache.k[:, start:end])
        selected_v.append(cache.v[:, start:end])
        selected_lengths.append(end - start)
        selected_ids.append(cache.record_ids[row])
    lengths = torch.tensor(
        selected_lengths,
        dtype=torch.long,
        device=cache.k.device,
    )
    return JaggedMigratedKVBatch(
        record_ids=tuple(selected_ids),
        migration_anchor_version=cache.migration_anchor_version,
        served_kv_target=cache.served_kv_target,
        k=torch.cat(selected_k, dim=1).contiguous(),
        v=torch.cat(selected_v, dim=1).contiguous(),
        lengths=lengths,
        offsets=_offsets(lengths),
    )


def assemble_jagged_rows(
    layout: JaggedMigratedKVBatch,
    sources: tuple[JaggedMigratedKVBatch, ...],
    target_version: int,
) -> JaggedMigratedKVBatch:
    source_by_record = {}
    for source in sources:
        for row, record_id in enumerate(source.record_ids):
            if record_id in source_by_record:
                raise ValueError("assembled record appears more than once")
            source_by_record[record_id] = (source, row)
    if set(source_by_record) != set(layout.record_ids):
        raise ValueError("assembled records do not cover the layout")
    k = torch.empty_like(layout.k)
    v = torch.empty_like(layout.v)
    for target_row, record_id in enumerate(layout.record_ids):
        source, source_row = source_by_record[record_id]
        target_start = int(layout.offsets[target_row])
        target_stop = int(layout.offsets[target_row + 1])
        source_start = int(source.offsets[source_row])
        source_stop = int(source.offsets[source_row + 1])
        if target_stop - target_start != source_stop - source_start:
            raise ValueError("assembled record length differs")
        k[:, target_start:target_stop].copy_(
            source.k[:, source_start:source_stop]
        )
        v[:, target_start:target_stop].copy_(
            source.v[:, source_start:source_stop]
        )
    return JaggedMigratedKVBatch(
        record_ids=layout.record_ids,
        migration_anchor_version=f"theta{target_version}",
        served_kv_target=f"theta{target_version}",
        k=k,
        v=v,
        lengths=layout.lengths.clone(),
        offsets=layout.offsets.clone(),
    )


def relative_cache_values(
    actual: JaggedMigratedKVBatch,
    reference: JaggedMigratedKVBatch,
) -> torch.Tensor:
    if (
        actual.record_ids != reference.record_ids
        or actual.k.shape != reference.k.shape
        or not torch.equal(actual.lengths, reference.lengths)
        or not torch.equal(actual.offsets, reference.offsets)
    ):
        raise ValueError("cache comparison layouts differ")
    output = []
    for row in range(actual.batch_size):
        start = int(actual.offsets[row])
        end = int(actual.offsets[row + 1])
        delta = torch.cat(
            (
                actual.k[:, start:end].float()
                - reference.k[:, start:end].float(),
                actual.v[:, start:end].float()
                - reference.v[:, start:end].float(),
            ),
            dim=-1,
        )
        denominator = torch.cat(
            (
                reference.k[:, start:end].float(),
                reference.v[:, start:end].float(),
            ),
            dim=-1,
        )
        numerator_norm = delta.square().sum(dim=(1, 2)).sqrt()
        denominator_norm = denominator.square().sum(dim=(1, 2)).sqrt()
        output.append(
            numerator_norm
            / denominator_norm.clamp_min(torch.finfo(torch.float32).eps)
        )
    return torch.stack(output)


def transition_sketch_values(
    source: JaggedMigratedKVBatch,
    candidate: JaggedMigratedKVBatch,
) -> dict[str, torch.Tensor]:
    if (
        source.record_ids != candidate.record_ids
        or source.k.shape != candidate.k.shape
        or not torch.equal(source.lengths, candidate.lengths)
        or not torch.equal(source.offsets, candidate.offsets)
    ):
        raise ValueError("transition sketch layouts differ")
    correction = []
    norm_change = []
    cosine_distance = []
    relative_peak = []
    epsilon = torch.finfo(torch.float32).eps
    for row in range(source.batch_size):
        start = int(source.offsets[row])
        end = int(source.offsets[row + 1])
        old = torch.cat(
            (
                source.k[:, start:end].float(),
                source.v[:, start:end].float(),
            ),
            dim=-1,
        )
        new = torch.cat(
            (
                candidate.k[:, start:end].float(),
                candidate.v[:, start:end].float(),
            ),
            dim=-1,
        )
        delta = new - old
        old_norm = old.square().sum(dim=(1, 2)).sqrt().clamp_min(epsilon)
        new_norm = new.square().sum(dim=(1, 2)).sqrt().clamp_min(epsilon)
        correction.append(delta.square().sum(dim=(1, 2)).sqrt() / old_norm)
        norm_change.append((new_norm / old_norm).log().abs())
        cosine = (old * new).sum(dim=(1, 2)) / (old_norm * new_norm)
        cosine_distance.append((1.0 - cosine).clamp(0.0, 2.0))
        old_rms = old.square().mean(dim=(1, 2)).sqrt().clamp_min(epsilon)
        relative_peak.append(
            delta.abs().amax(dim=(1, 2)) / old_rms
        )
    return {
        "relative_correction": torch.stack(correction),
        "absolute_log_norm_ratio": torch.stack(norm_change),
        "cosine_distance": torch.stack(cosine_distance),
        "relative_peak_correction": torch.stack(relative_peak),
    }


if triton is not None:

    @triton.jit
    def _cache_norm_sums_kernel(
        source_k,
        source_v,
        candidate_k,
        candidate_v,
        offsets,
        output,
        total_tokens,
        num_layers: tl.constexpr,
        batch_size: tl.constexpr,
        width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        layer = tl.program_id(0)
        record = tl.program_id(1)
        block = tl.program_id(2)
        record_start = tl.load(offsets + record)
        record_stop = tl.load(offsets + record + 1)
        record_elements = (record_stop - record_start) * width
        element = block * BLOCK + tl.arange(0, BLOCK)
        mask = element < record_elements
        token = element // width
        channel = element - token * width
        index = (
            layer * total_tokens * width
            + (record_start + token) * width
            + channel
        )
        old_k = tl.load(source_k + index, mask=mask, other=0.0).to(
            tl.float32
        )
        old_v = tl.load(source_v + index, mask=mask, other=0.0).to(
            tl.float32
        )
        new_k = tl.load(candidate_k + index, mask=mask, other=0.0).to(
            tl.float32
        )
        new_v = tl.load(candidate_v + index, mask=mask, other=0.0).to(
            tl.float32
        )
        old_sum = tl.sum(old_k * old_k + old_v * old_v)
        new_sum = tl.sum(new_k * new_k + new_v * new_v)
        base = (record * num_layers + layer) * 2
        tl.atomic_add(output + base, old_sum)
        tl.atomic_add(output + base + 1, new_sum)

else:
    _cache_norm_sums_kernel = None


def absolute_log_norm_ratio_values(
    source: JaggedMigratedKVBatch,
    candidate: JaggedMigratedKVBatch,
    block: int = 1024,
) -> torch.Tensor:
    if (
        source.record_ids != candidate.record_ids
        or source.k.shape != candidate.k.shape
        or not torch.equal(source.lengths, candidate.lengths)
        or not torch.equal(source.offsets, candidate.offsets)
        or block not in {256, 512, 1024, 2048, 4096}
    ):
        raise ValueError("norm-ratio sketch layouts differ")
    if source.k.device.type == "cuda" and triton is not None:
        sums = torch.zeros(
            (source.batch_size, source.k.shape[0], 2),
            dtype=torch.float32,
            device=source.k.device,
        )
        maximum = int(source.lengths.max()) * source.k.shape[2]
        grid = (
            source.k.shape[0],
            source.batch_size,
            triton.cdiv(maximum, block),
        )
        _cache_norm_sums_kernel[grid](
            source.k,
            source.v,
            candidate.k,
            candidate.v,
            source.offsets,
            sums,
            source.token_count,
            num_layers=source.k.shape[0],
            batch_size=source.batch_size,
            width=source.k.shape[2],
            BLOCK=block,
        )
        epsilon = torch.finfo(torch.float32).eps
        return (
            0.5
            * (
                sums[..., 1].clamp_min(epsilon).log()
                - sums[..., 0].clamp_min(epsilon).log()
            ).abs()
        )
    return transition_sketch_values(
        source,
        candidate,
    )["absolute_log_norm_ratio"]


def aggregate_layer_values(
    values: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if (
        values.ndim != 2
        or values.shape[1] < 1
        or not 0.5 <= quantile <= 1.0
        or not bool(torch.isfinite(values).all())
        or bool(torch.any(values < 0))
    ):
        raise ValueError("layer values are invalid")
    return torch.quantile(values.float(), quantile, dim=1)


@dataclass
class Stage46KVStore:
    record_ids: tuple[int, ...]
    served_version: int
    lengths: torch.Tensor
    offsets: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not self.record_ids
            or len(set(self.record_ids)) != len(self.record_ids)
            or self.served_version < 0
            or self.k.ndim != 3
            or self.k.shape != self.v.shape
            or self.k.dtype != torch.float16
            or not self.k.is_contiguous()
            or not self.v.is_contiguous()
            or self.lengths.shape != (len(self.record_ids),)
            or self.offsets.shape != (len(self.record_ids) + 1,)
            or self.k.device != self.v.device
            or self.k.device != self.lengths.device
            or self.k.device != self.offsets.device
        ):
            raise ValueError("Stage 4.6 K/V store is invalid")

    @classmethod
    def from_batch(
        cls,
        batch: JaggedMigratedKVBatch,
        served_version: int,
    ) -> Stage46KVStore:
        return cls(
            record_ids=batch.record_ids,
            served_version=served_version,
            lengths=batch.lengths.clone(),
            offsets=batch.offsets.clone(),
            k=batch.k.clone().contiguous(),
            v=batch.v.clone().contiguous(),
        )

    @property
    def nbytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (self.k, self.v, self.lengths, self.offsets)
        )

    def read_extent(self, start: int, stop: int) -> JaggedMigratedKVBatch:
        if not 0 <= start < stop <= len(self.record_ids):
            raise ValueError("Stage 4.6 extent bounds are invalid")
        token_start = int(self.offsets[start])
        token_stop = int(self.offsets[stop])
        lengths = self.lengths[start:stop].clone()
        return JaggedMigratedKVBatch(
            record_ids=self.record_ids[start:stop],
            migration_anchor_version=f"theta{self.served_version}",
            served_kv_target=f"theta{self.served_version}",
            k=self.k[:, token_start:token_stop].clone().contiguous(),
            v=self.v[:, token_start:token_stop].clone().contiguous(),
            lengths=lengths,
            offsets=_offsets(lengths),
        )

    def write_records(
        self,
        start: int,
        source: JaggedMigratedKVBatch,
        rows: tuple[int, ...],
    ) -> None:
        if (
            source.record_ids != self.record_ids[start : start + source.batch_size]
            or any(not 0 <= row < source.batch_size for row in rows)
            or len(set(rows)) != len(rows)
        ):
            raise ValueError("Stage 4.6 publication rows differ")
        for row in rows:
            target_row = start + row
            target_start = int(self.offsets[target_row])
            target_stop = int(self.offsets[target_row + 1])
            source_start = int(source.offsets[row])
            source_stop = int(source.offsets[row + 1])
            if target_stop - target_start != source_stop - source_start:
                raise ValueError("Stage 4.6 publication lengths differ")
            self.k[:, target_start:target_stop].copy_(
                source.k[:, source_start:source_stop]
            )
            self.v[:, target_start:target_stop].copy_(
                source.v[:, source_start:source_stop]
            )

    def advance_version(self, target_version: int) -> None:
        if target_version != self.served_version + 1:
            raise ValueError("Stage 4.6 store target must be adjacent")
        self.served_version = target_version


@dataclass(frozen=True)
class RoutedExtent:
    decisions: tuple[LifecycleDecision, ...]
    correction_magnitudes: tuple[float | None, ...]

    def __post_init__(self) -> None:
        if len(self.decisions) != len(self.correction_magnitudes):
            raise ValueError("routed extent fields differ")


def route_extent(
    policy: LifecyclePolicy,
    states: tuple[CacheLifecycleState, ...],
    target_version: int,
    source: JaggedMigratedKVBatch,
    candidate: JaggedMigratedKVBatch | None,
    layer_quantile: float,
) -> RoutedExtent:
    if (
        tuple(state.record_id for state in states) != source.record_ids
        or target_version != states[0].served_version + 1
        or any(state.served_version != states[0].served_version for state in states)
    ):
        raise ValueError("router state and source extent differ")
    candidate_required = tuple(policy.requires_candidate(state) for state in states)
    if any(candidate_required):
        if candidate is None:
            raise ValueError("router candidate is missing")
        correction = aggregate_layer_values(
            relative_cache_values(candidate, source),
            layer_quantile,
        ).cpu()
    else:
        if candidate is not None:
            raise ValueError("unneeded router candidate was supplied")
        correction = torch.zeros(len(states))
    decisions = []
    magnitudes: list[float | None] = []
    for row, state in enumerate(states):
        magnitude = float(correction[row]) if candidate_required[row] else None
        decisions.append(
            policy.decide(
                state,
                target_version,
                magnitude,
            )
        )
        magnitudes.append(magnitude)
    return RoutedExtent(
        decisions=tuple(decisions),
        correction_magnitudes=tuple(magnitudes),
    )


@dataclass
class Stage46CostLedger:
    migration_ms: float = 0.0
    exact_refresh_ms: float = 0.0
    router_ms: float = 0.0
    publication_ms: float = 0.0
    all_exact_reference_ms: float = 0.0
    discarded_migration_records: int = 0

    @property
    def mixed_policy_ms(self) -> float:
        return (
            self.migration_ms
            + self.exact_refresh_ms
            + self.router_ms
            + self.publication_ms
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "migration_ms": self.migration_ms,
            "exact_refresh_ms": self.exact_refresh_ms,
            "router_ms": self.router_ms,
            "publication_ms": self.publication_ms,
            "mixed_policy_ms": self.mixed_policy_ms,
            "all_exact_reference_ms": self.all_exact_reference_ms,
            "discarded_migration_records": (
                self.discarded_migration_records
            ),
        }


@dataclass
class CudaTimer:
    device: torch.device
    samples_ms: list[float] = field(default_factory=list)

    def measure(self, action: Callable[[], _T]) -> _T:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = action()
        end.record()
        end.synchronize()
        self.samples_ms.append(float(start.elapsed_time(end)))
        return value

    @property
    def total_ms(self) -> float:
        return sum(self.samples_ms)
