from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from ..models import HSTUKVCache
from .cohort_jagged import JaggedMigratedKVBatch
from .design2_embedding import (
    D2ShardedHSTU,
    ModuloRowShardedEmbedding,
    modulo_embedding_local_id,
    modulo_embedding_owner,
)
from .design2_plan import D2ActionPlan, D2ActionRecord
from .recompute import RawHistoryBatch, exact_hidden_and_kv_from_item_embeddings
from .stage46_chain import pack_padded_cache

INTEGRATED_ROUTES = ("compiled", "scheduled_exact", "natural_exact")
PILOT_W3_ROUTE_QUOTAS = {
    0: {"compiled": 51, "scheduled_exact": 5, "natural_exact": 8},
    1: {"compiled": 51, "scheduled_exact": 4, "natural_exact": 9},
    2: {"compiled": 51, "scheduled_exact": 4, "natural_exact": 9},
}


def integrated_route(record: D2ActionRecord) -> str:
    if record.requested_action == "compiled":
        return "compiled"
    return record.requested_reason


def integrated_lookup_token_ledger(
    records: tuple[D2ActionRecord, ...],
    organization: str,
) -> dict[str, int]:
    if not records:
        raise ValueError("integrated lookup ledger requires records")
    if organization == "staged":
        phases = {
            "scheduled_exact_retained": sum(
                value.retained_tokens
                for value in records
                if integrated_route(value) == "scheduled_exact"
            ),
            "natural_exact_prefix": sum(
                value.target_prefix_tokens
                for value in records
                if integrated_route(value) == "natural_exact"
            ),
            "delta_append": sum(
                value.delta_tokens
                for value in records
                if integrated_route(value)
                in {"compiled", "scheduled_exact"}
            ),
            "latest_append": sum(
                value.latest_tokens for value in records
            ),
        }
    elif organization == "fused_finalization":
        phases = {
            "compiled_finalization_append": sum(
                value.delta_tokens + value.latest_tokens
                for value in records
                if integrated_route(value) == "compiled"
            ),
            "scheduled_final_exact": sum(
                value.final_tokens
                for value in records
                if integrated_route(value) == "scheduled_exact"
            ),
            "natural_final_exact": sum(
                value.final_tokens
                for value in records
                if integrated_route(value) == "natural_exact"
            ),
        }
    else:
        raise ValueError("integrated lookup organization differs")
    phases["total"] = sum(phases.values())
    return phases


def _quantile_sample(
    records: tuple[D2ActionRecord, ...],
    count: int,
) -> tuple[D2ActionRecord, ...]:
    if count < 0 or count > len(records):
        raise ValueError("integrated sample quota exceeds its stratum")
    if count == len(records):
        return tuple(sorted(records, key=lambda value: value.record_id))
    ordered = sorted(
        records,
        key=lambda value: (
            value.final_tokens,
            value.old_tokens,
            value.record_id,
        ),
    )
    selected = {
        min(len(ordered) - 1, math.floor((index + 0.5) * len(ordered) / count))
        for index in range(count)
    } if count else set()
    if len(selected) != count:
        raise RuntimeError("integrated quantile sample is not unique")
    return tuple(
        sorted(
            (ordered[index] for index in selected),
            key=lambda value: value.record_id,
        )
    )


def select_integrated_records(
    plan: D2ActionPlan,
    owner_map: dict[int, int],
    world_size: int,
    cohort: str,
) -> tuple[D2ActionRecord, ...]:
    if world_size != 3:
        raise ValueError("integrated benchmark currently requires world size three")
    if set(owner_map) != {value.record_id for value in plan.records}:
        raise ValueError("integrated owner map does not cover the action plan")
    if cohort == "full682":
        return plan.records
    if cohort != "pilot192":
        raise ValueError("integrated cohort must be pilot192 or full682")
    selected = []
    for rank in range(world_size):
        for route in INTEGRATED_ROUTES:
            candidates = tuple(
                value
                for value in plan.records
                if owner_map[value.record_id] == rank
                and integrated_route(value) == route
            )
            selected.extend(
                _quantile_sample(
                    candidates,
                    PILOT_W3_ROUTE_QUOTAS[rank][route],
                )
            )
    output = tuple(sorted(selected, key=lambda value: value.record_id))
    if (
        len(output) != 192
        or len({value.record_id for value in output}) != 192
        or any(
            sum(owner_map[value.record_id] == rank for value in output) != 64
            for rank in range(world_size)
        )
    ):
        raise RuntimeError("integrated pilot selection is not owner balanced")
    return output


@dataclass(frozen=True)
class D2IntegratedExtent:
    route: str
    ordinal: int
    record_ids_by_rank: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            self.route not in {*INTEGRATED_ROUTES, "all"}
            or self.ordinal < 0
            or not self.record_ids_by_rank
            or any(
                len(set(record_ids)) != len(record_ids)
                for record_ids in self.record_ids_by_rank
            )
        ):
            raise ValueError("integrated extent is invalid")

    def local_record_ids(self, rank: int) -> tuple[int, ...]:
        return self.record_ids_by_rank[rank]


def build_integrated_schedule(
    records: tuple[D2ActionRecord, ...],
    owner_map: dict[int, int],
    world_size: int,
    extent_size: int,
    *,
    route_major: bool,
    compiled_order: str = "final_length",
) -> tuple[D2IntegratedExtent, ...]:
    if (
        world_size < 1
        or extent_size < 1
        or not records
        or compiled_order not in {"final_length", "suffix_retained"}
    ):
        raise ValueError("integrated schedule inputs are invalid")
    if any(
        value.record_id not in owner_map
        or not 0 <= owner_map[value.record_id] < world_size
        for value in records
    ):
        raise ValueError("integrated schedule owner is invalid")
    routes = INTEGRATED_ROUTES if route_major else ("all",)
    extents = []
    for route in routes:
        per_rank = []
        for rank in range(world_size):
            candidates = [
                value
                for value in records
                if owner_map[value.record_id] == rank
                and (route == "all" or integrated_route(value) == route)
            ]
            if route == "compiled" and compiled_order == "suffix_retained":
                candidates.sort(
                    key=lambda value: (
                        value.delta_tokens + value.latest_tokens,
                        value.retained_tokens,
                        value.final_tokens,
                        value.record_id,
                    )
                )
            else:
                candidates.sort(
                    key=lambda value: (
                        value.final_tokens,
                        value.old_tokens,
                        value.record_id,
                    )
                )
            per_rank.append(candidates)
        steps = max(
            math.ceil(len(values) / extent_size)
            for values in per_rank
        )
        for ordinal in range(steps):
            start = ordinal * extent_size
            stop = start + extent_size
            extents.append(
                D2IntegratedExtent(
                    route=route,
                    ordinal=ordinal,
                    record_ids_by_rank=tuple(
                        tuple(value.record_id for value in values[start:stop])
                        for values in per_rank
                    ),
                )
            )
    observed = [
        record_id
        for extent in extents
        for rank_ids in extent.record_ids_by_rank
        for record_id in rank_ids
    ]
    expected_multiplier = 1
    if (
        len(observed) != len(records) * expected_multiplier
        or set(observed) != {value.record_id for value in records}
        or len(observed) != len(set(observed))
    ):
        raise RuntimeError("integrated schedule coverage differs")
    return tuple(extents)


def build_integrated_exact_pool_schedule(
    records: tuple[D2ActionRecord, ...],
    owner_map: dict[int, int],
    world_size: int,
    extent_size: int,
) -> tuple[D2IntegratedExtent, ...]:
    exact_records = tuple(
        value
        for value in records
        if integrated_route(value) in {"scheduled_exact", "natural_exact"}
    )
    if not exact_records:
        raise ValueError("integrated exact pool requires exact records")
    return build_integrated_schedule(
        exact_records,
        owner_map,
        world_size,
        extent_size,
        route_major=False,
    )


def integrated_exact_reason_counts(
    extent: D2IntegratedExtent,
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
) -> dict[str, int]:
    actions = tuple(
        actions_by_id[value]
        for value in extent.local_record_ids(rank)
    )
    counts = {
        reason: sum(integrated_route(value) == reason for value in actions)
        for reason in ("scheduled_exact", "natural_exact")
    }
    if sum(counts.values()) != len(actions):
        raise ValueError("integrated exact pool contains a compiled action")
    return counts


@dataclass(frozen=True)
class IntegratedLookupMetrics:
    rank: int
    world_size: int
    requested_tokens: int
    local_requested_tokens: int
    remote_requested_tokens: int
    served_remote_requested_tokens: int
    counts_collective_input_bytes: int
    counts_collective_output_bytes: int
    id_collective_input_bytes: int
    id_collective_output_bytes: int
    vector_collective_input_bytes: int
    vector_collective_output_bytes: int
    off_diagonal_send_bytes: int
    off_diagonal_receive_bytes: int
    collective_calls: int
    counts_collective_seconds: float
    id_collective_seconds: float
    vector_collective_seconds: float

    @property
    def actual_collective_tensor_payload_bytes(self) -> int:
        return sum(
            (
                self.counts_collective_input_bytes,
                self.counts_collective_output_bytes,
                self.id_collective_input_bytes,
                self.id_collective_output_bytes,
                self.vector_collective_input_bytes,
                self.vector_collective_output_bytes,
            )
        )

    @property
    def off_diagonal_bytes(self) -> int:
        return self.off_diagonal_send_bytes + self.off_diagonal_receive_bytes

    @property
    def collective_seconds(self) -> float:
        return (
            self.counts_collective_seconds
            + self.id_collective_seconds
            + self.vector_collective_seconds
        )

    def to_dict(self) -> dict[str, int | float]:
        output = asdict(self)
        output["actual_collective_tensor_payload_bytes"] = (
            self.actual_collective_tensor_payload_bytes
        )
        output["off_diagonal_bytes"] = self.off_diagonal_bytes
        output["collective_seconds"] = self.collective_seconds
        return output


@dataclass(frozen=True)
class IntegratedLookupResult:
    item_vectors: torch.Tensor
    metrics: IntegratedLookupMetrics


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_collective(
    device: torch.device,
    action,
    timing_mode: str,
) -> float:
    if timing_mode not in {"device", "current_stream"}:
        raise ValueError("integrated collective timing mode differs")
    if device.type != "cuda" or timing_mode == "device":
        _synchronize(device)
        started = time.perf_counter()
        action()
        _synchronize(device)
        return time.perf_counter() - started
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    action()
    finished.record()
    finished.synchronize()
    return started.elapsed_time(finished) / 1000.0


@torch.inference_mode()
def fast_modulo_sharded_lookup(
    embedding: ModuloRowShardedEmbedding,
    item_ids: torch.Tensor,
    lengths: torch.Tensor,
    collective_timing: str = "device",
) -> IntegratedLookupResult:
    embedding._validate_group()
    if (
        item_ids.ndim != 2
        or lengths.shape != (item_ids.shape[0],)
        or item_ids.device != embedding.local_weight.device
        or lengths.device != item_ids.device
    ):
        raise ValueError("integrated sharded lookup inputs differ")
    width = item_ids.shape[1]
    prepared_lengths = lengths.long()
    if bool(torch.any(prepared_lengths < 0)) or bool(
        torch.any(prepared_lengths > width)
    ):
        raise ValueError("integrated sharded lookup lengths differ")
    valid = (
        torch.arange(width, device=item_ids.device).unsqueeze(0)
        < prepared_lengths.unsqueeze(1)
    )
    positions = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    requested_ids = item_ids.reshape(-1).index_select(0, positions).long()
    if requested_ids.numel() and (
        bool(torch.any(requested_ids < 0))
        or bool(torch.any(requested_ids >= embedding.num_embeddings))
    ):
        raise ValueError("integrated item id exceeds the vocabulary")
    owners = modulo_embedding_owner(requested_ids, embedding.world_size)
    local_mask = owners == embedding.rank
    remote_mask = ~local_mask
    local_ids = requested_ids[local_mask]
    remote_ids = requested_ids[remote_mask]
    local_positions = positions[local_mask]
    remote_positions = positions[remote_mask]
    vectors = torch.zeros(
        (item_ids.numel(), embedding.hidden_size),
        dtype=torch.float32,
        device=item_ids.device,
    )
    if local_ids.numel():
        vectors.index_copy_(
            0,
            local_positions,
            embedding.local_weight.index_select(
                0,
                modulo_embedding_local_id(local_ids, embedding.world_size),
            ),
        )
    if embedding.world_size == 1:
        return IntegratedLookupResult(
            item_vectors=vectors.reshape(
                *item_ids.shape,
                embedding.hidden_size,
            ),
            metrics=IntegratedLookupMetrics(
                rank=embedding.rank,
                world_size=1,
                requested_tokens=requested_ids.numel(),
                local_requested_tokens=local_ids.numel(),
                remote_requested_tokens=0,
                served_remote_requested_tokens=0,
                counts_collective_input_bytes=0,
                counts_collective_output_bytes=0,
                id_collective_input_bytes=0,
                id_collective_output_bytes=0,
                vector_collective_input_bytes=0,
                vector_collective_output_bytes=0,
                off_diagonal_send_bytes=0,
                off_diagonal_receive_bytes=0,
                collective_calls=0,
                counts_collective_seconds=0.0,
                id_collective_seconds=0.0,
                vector_collective_seconds=0.0,
            ),
        )
    send_counts = torch.bincount(
        owners[remote_mask],
        minlength=embedding.world_size,
    ).to(device=item_ids.device, dtype=torch.int64)
    receive_counts = torch.empty_like(send_counts)
    counts_seconds = _timed_collective(
        item_ids.device,
        lambda: dist.all_to_all_single(
            receive_counts,
            send_counts,
            group=embedding.process_group,
        ),
        collective_timing,
    )
    order = torch.argsort(owners[remote_mask], stable=True)
    ordered_ids = remote_ids.index_select(0, order)
    ordered_positions = remote_positions.index_select(0, order)
    send_local_ids = modulo_embedding_local_id(
        ordered_ids,
        embedding.world_size,
    ).contiguous()
    send_splits = tuple(int(value) for value in send_counts.tolist())
    receive_splits = tuple(int(value) for value in receive_counts.tolist())
    received_local_ids = torch.empty(
        sum(receive_splits),
        dtype=torch.int64,
        device=item_ids.device,
    )
    id_seconds = _timed_collective(
        item_ids.device,
        lambda: dist.all_to_all_single(
            received_local_ids,
            send_local_ids,
            output_split_sizes=receive_splits,
            input_split_sizes=send_splits,
            group=embedding.process_group,
        ),
        collective_timing,
    )
    if received_local_ids.numel() and (
        bool(torch.any(received_local_ids < 0))
        or bool(torch.any(received_local_ids >= embedding.local_weight.shape[0]))
    ):
        raise RuntimeError("integrated received item id exceeds its shard")
    response_vectors = embedding.local_weight.index_select(
        0,
        received_local_ids,
    )
    received_vectors = torch.empty(
        (send_local_ids.numel(), embedding.hidden_size),
        dtype=torch.float32,
        device=item_ids.device,
    )
    vector_seconds = _timed_collective(
        item_ids.device,
        lambda: dist.all_to_all_single(
            received_vectors,
            response_vectors,
            output_split_sizes=send_splits,
            input_split_sizes=receive_splits,
            group=embedding.process_group,
        ),
        collective_timing,
    )
    if ordered_positions.numel():
        vectors.index_copy_(0, ordered_positions, received_vectors)
    count_bytes = send_counts.element_size()
    id_bytes = send_local_ids.element_size()
    vector_bytes = embedding.local_weight.element_size()
    id_input_bytes = send_local_ids.numel() * id_bytes
    id_output_bytes = received_local_ids.numel() * id_bytes
    vector_input_bytes = response_vectors.numel() * vector_bytes
    vector_output_bytes = received_vectors.numel() * vector_bytes
    count_off_diagonal_bytes = (embedding.world_size - 1) * count_bytes
    return IntegratedLookupResult(
        item_vectors=vectors.reshape(
            *item_ids.shape,
            embedding.hidden_size,
        ),
        metrics=IntegratedLookupMetrics(
            rank=embedding.rank,
            world_size=embedding.world_size,
            requested_tokens=requested_ids.numel(),
            local_requested_tokens=local_ids.numel(),
            remote_requested_tokens=remote_ids.numel(),
            served_remote_requested_tokens=received_local_ids.numel(),
            counts_collective_input_bytes=send_counts.numel() * count_bytes,
            counts_collective_output_bytes=receive_counts.numel() * count_bytes,
            id_collective_input_bytes=id_input_bytes,
            id_collective_output_bytes=id_output_bytes,
            vector_collective_input_bytes=vector_input_bytes,
            vector_collective_output_bytes=vector_output_bytes,
            off_diagonal_send_bytes=(
                count_off_diagonal_bytes + id_input_bytes + vector_input_bytes
            ),
            off_diagonal_receive_bytes=(
                count_off_diagonal_bytes + id_output_bytes + vector_output_bytes
            ),
            collective_calls=3,
            counts_collective_seconds=counts_seconds,
            id_collective_seconds=id_seconds,
            vector_collective_seconds=vector_seconds,
        ),
    )


@dataclass(frozen=True)
class IntegratedExactResult:
    fragment: JaggedMigratedKVBatch | None
    last_hidden: torch.Tensor
    lookup_metrics: IntegratedLookupMetrics


@torch.inference_mode()
def integrated_sharded_exact(
    model: D2ShardedHSTU,
    batch: RawHistoryBatch,
    target_version: str,
    dtype: torch.dtype = torch.float16,
    collective_timing: str = "device",
) -> IntegratedExactResult:
    lookup = fast_modulo_sharded_lookup(
        model.item_embedding,
        batch.item_ids,
        batch.lengths,
        collective_timing=collective_timing,
    )
    if batch.batch_size == 0:
        return IntegratedExactResult(
            fragment=None,
            last_hidden=torch.empty(
                (0, model.dense_model.cfg.hidden_size),
                dtype=torch.float32,
                device=batch.device,
            ),
            lookup_metrics=lookup.metrics,
        )
    if bool(torch.any(batch.lengths < 1)):
        raise ValueError("integrated exact records must be nonempty")
    hidden, cache = exact_hidden_and_kv_from_item_embeddings(
        model.dense_model,
        lookup.item_vectors,
        batch.behaviors,
        batch.time_deltas,
        lengths=batch.lengths,
    )
    return IntegratedExactResult(
        fragment=pack_padded_cache(
            cache,
            batch.lengths,
            batch.record_ids,
            target_version,
            target_version,
            dtype=dtype,
        ),
        last_hidden=model.dense_model.last_hidden(hidden, batch.lengths),
        lookup_metrics=lookup.metrics,
    )


def slice_integrated_jagged_ranges(
    cache: JaggedMigratedKVBatch,
    starts: tuple[int, ...],
    stops: tuple[int, ...],
) -> JaggedMigratedKVBatch:
    if (
        len(starts) != cache.batch_size
        or len(stops) != cache.batch_size
        or any(start < 0 or stop <= start for start, stop in zip(starts, stops, strict=True))
    ):
        raise ValueError("integrated jagged ranges are invalid")
    source_lengths = tuple(int(value) for value in cache.lengths.tolist())
    if any(
        stop > length
        for stop, length in zip(stops, source_lengths, strict=True)
    ):
        raise ValueError("integrated jagged range exceeds its source")
    source_offsets = [0]
    for length in source_lengths:
        source_offsets.append(source_offsets[-1] + length)
    lengths = torch.tensor(
        [stop - start for start, stop in zip(starts, stops, strict=True)],
        dtype=torch.long,
        device=cache.k.device,
    )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=cache.k.device),
            lengths.cumsum(0),
        )
    )
    return JaggedMigratedKVBatch(
        record_ids=cache.record_ids,
        migration_anchor_version=cache.migration_anchor_version,
        served_kv_target=cache.served_kv_target,
        k=torch.cat(
            [
                cache.k[:, source_offsets[row] + start : source_offsets[row] + stop]
                for row, (start, stop) in enumerate(zip(starts, stops, strict=True))
            ],
            dim=1,
        ).contiguous(),
        v=torch.cat(
            [
                cache.v[:, source_offsets[row] + start : source_offsets[row] + stop]
                for row, (start, stop) in enumerate(zip(starts, stops, strict=True))
            ],
            dim=1,
        ).contiguous(),
        lengths=lengths,
        offsets=offsets,
    )


@dataclass(frozen=True)
class IntegratedAppendResult:
    fragment: JaggedMigratedKVBatch | None
    last_hidden: torch.Tensor
    lookup_metrics: IntegratedLookupMetrics


@dataclass(frozen=True)
class IntegratedAppendOnlyKVBatch:
    retained: JaggedMigratedKVBatch
    suffix: JaggedMigratedKVBatch
    lengths: torch.Tensor
    offsets: torch.Tensor
    target_version: str

    def __post_init__(self) -> None:
        if (
            self.retained.record_ids != self.suffix.record_ids
            or self.retained.k.shape[0] != self.suffix.k.shape[0]
            or self.retained.k.shape[2] != self.suffix.k.shape[2]
            or self.retained.k.dtype != self.suffix.k.dtype
            or self.retained.k.device != self.suffix.k.device
            or self.suffix.served_kv_target != self.target_version
            or self.lengths.shape != self.retained.lengths.shape
            or self.offsets.shape != self.retained.offsets.shape
            or self.lengths.device != self.retained.k.device
            or self.offsets.device != self.retained.k.device
            or (
                self.retained.k.device.type == "cpu"
                and not torch.equal(
                    self.lengths,
                    self.retained.lengths + self.suffix.lengths,
                )
            )
        ):
            raise ValueError("integrated append-only destination differs")

    @property
    def record_ids(self) -> tuple[int, ...]:
        return self.retained.record_ids

    @property
    def batch_size(self) -> int:
        return len(self.record_ids)

    @property
    def token_count(self) -> int:
        return self.retained.token_count + self.suffix.token_count

    @property
    def nbytes(self) -> int:
        return (
            self.retained.nbytes
            + self.suffix.nbytes
            + self.lengths.numel() * self.lengths.element_size()
            + self.offsets.numel() * self.offsets.element_size()
        )


@dataclass(frozen=True)
class IntegratedAppendOnlyResult:
    fragment: IntegratedAppendOnlyKVBatch | None
    last_hidden: torch.Tensor
    lookup_metrics: IntegratedLookupMetrics


def materialize_integrated_append_only(
    batch: IntegratedAppendOnlyKVBatch,
) -> JaggedMigratedKVBatch:
    retained_lengths = tuple(
        int(value) for value in batch.retained.lengths.tolist()
    )
    suffix_lengths = tuple(
        int(value) for value in batch.suffix.lengths.tolist()
    )
    retained_offsets = [0]
    suffix_offsets = [0]
    for retained_length, suffix_length in zip(
        retained_lengths,
        suffix_lengths,
        strict=True,
    ):
        retained_offsets.append(
            retained_offsets[-1] + retained_length
        )
        suffix_offsets.append(suffix_offsets[-1] + suffix_length)
    row_k = []
    row_v = []
    for row in range(batch.batch_size):
        row_k.append(
            torch.cat(
                (
                    batch.retained.k[
                        :,
                        retained_offsets[row] : retained_offsets[row + 1],
                    ],
                    batch.suffix.k[
                        :,
                        suffix_offsets[row] : suffix_offsets[row + 1],
                    ],
                ),
                dim=1,
            )
        )
        row_v.append(
            torch.cat(
                (
                    batch.retained.v[
                        :,
                        retained_offsets[row] : retained_offsets[row + 1],
                    ],
                    batch.suffix.v[
                        :,
                        suffix_offsets[row] : suffix_offsets[row + 1],
                    ],
                ),
                dim=1,
            )
        )
    return JaggedMigratedKVBatch(
        record_ids=batch.record_ids,
        migration_anchor_version=batch.target_version,
        served_kv_target=batch.target_version,
        k=torch.cat(row_k, dim=1).contiguous(),
        v=torch.cat(row_v, dim=1).contiguous(),
        lengths=batch.lengths,
        offsets=batch.offsets,
    )


@torch.inference_mode()
def integrated_sharded_append(
    model: D2ShardedHSTU,
    retained: JaggedMigratedKVBatch | None,
    suffix: RawHistoryBatch,
    target_version: str,
    dtype: torch.dtype = torch.float16,
) -> IntegratedAppendResult:
    lookup = fast_modulo_sharded_lookup(
        model.item_embedding,
        suffix.item_ids,
        suffix.lengths,
    )
    if suffix.batch_size == 0:
        if retained is not None:
            raise ValueError("empty integrated append has a retained cache")
        return IntegratedAppendResult(
            fragment=None,
            last_hidden=torch.empty(
                (0, model.dense_model.cfg.hidden_size),
                dtype=torch.float32,
                device=suffix.device,
            ),
            lookup_metrics=lookup.metrics,
        )
    if (
        retained is None
        or retained.record_ids != suffix.record_ids
        or retained.k.device != suffix.device
        or bool(torch.any(suffix.lengths < 0))
    ):
        raise ValueError("integrated append inputs differ")
    retained_lengths = tuple(int(value) for value in retained.lengths.tolist())
    suffix_lengths = tuple(int(value) for value in suffix.lengths.tolist())
    if not any(suffix_lengths):
        return IntegratedAppendResult(
            fragment=JaggedMigratedKVBatch(
                record_ids=retained.record_ids,
                migration_anchor_version=target_version,
                served_kv_target=target_version,
                k=retained.k,
                v=retained.v,
                lengths=retained.lengths,
                offsets=retained.offsets,
            ),
            last_hidden=torch.zeros(
                (
                    suffix.batch_size,
                    model.dense_model.cfg.hidden_size,
                ),
                dtype=torch.float32,
                device=suffix.device,
            ),
            lookup_metrics=lookup.metrics,
        )
    retained_offsets = [0]
    for length in retained_lengths:
        retained_offsets.append(retained_offsets[-1] + length)
    retained_width = max(retained_lengths)
    shape = (
        retained.k.shape[0],
        retained.batch_size,
        retained_width,
        retained.k.shape[2],
    )
    padded_k = torch.zeros(
        shape,
        dtype=torch.float32,
        device=retained.k.device,
    )
    padded_v = torch.zeros_like(padded_k)
    for row, length in enumerate(retained_lengths):
        start = retained_offsets[row]
        stop = retained_offsets[row + 1]
        padded_k[:, row, :length].copy_(retained.k[:, start:stop])
        padded_v[:, row, :length].copy_(retained.v[:, start:stop])
    hidden, updated = model.dense_model.forward_with_cache_from_item_embeddings(
        cached_kv=HSTUKVCache(
            k=padded_k,
            v=padded_v,
            seq_len=retained_width,
        ),
        new_item_vectors=lookup.item_vectors,
        new_behaviors=suffix.behaviors,
        new_time_deltas=suffix.time_deltas,
    )
    row_k = []
    row_v = []
    for row, (retained_length, suffix_length) in enumerate(
        zip(retained_lengths, suffix_lengths, strict=True)
    ):
        row_k.append(
            torch.cat(
                (
                    updated.k[:, row, :retained_length],
                    updated.k[
                        :,
                        row,
                        retained_width : retained_width + suffix_length,
                    ],
                ),
                dim=1,
            ).to(dtype=dtype)
        )
        row_v.append(
            torch.cat(
                (
                    updated.v[:, row, :retained_length],
                    updated.v[
                        :,
                        row,
                        retained_width : retained_width + suffix_length,
                    ],
                ),
                dim=1,
            ).to(dtype=dtype)
        )
    lengths = torch.tensor(
        [
            retained_length + suffix_length
            for retained_length, suffix_length in zip(
                retained_lengths,
                suffix_lengths,
                strict=True,
            )
        ],
        dtype=torch.long,
        device=retained.k.device,
    )
    offsets = torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.long,
                device=retained.k.device,
            ),
            lengths.cumsum(0),
        )
    )
    last_hidden = torch.zeros(
        (suffix.batch_size, model.dense_model.cfg.hidden_size),
        dtype=hidden.dtype,
        device=suffix.device,
    )
    positive_rows = torch.nonzero(
        suffix.lengths > 0,
        as_tuple=False,
    ).flatten()
    last_hidden.index_copy_(
        0,
        positive_rows,
        hidden[
            positive_rows,
            suffix.lengths.index_select(0, positive_rows).long() - 1,
        ],
    )
    return IntegratedAppendResult(
        fragment=JaggedMigratedKVBatch(
            record_ids=retained.record_ids,
            migration_anchor_version=target_version,
            served_kv_target=target_version,
            k=torch.cat(row_k, dim=1).contiguous(),
            v=torch.cat(row_v, dim=1).contiguous(),
            lengths=lengths,
            offsets=offsets,
        ),
        last_hidden=last_hidden,
        lookup_metrics=lookup.metrics,
    )


@torch.inference_mode()
def integrated_sharded_append_only(
    model: D2ShardedHSTU,
    retained: JaggedMigratedKVBatch | None,
    suffix: RawHistoryBatch,
    target_version: str,
    dtype: torch.dtype = torch.float16,
    collective_timing: str = "device",
) -> IntegratedAppendOnlyResult:
    lookup = fast_modulo_sharded_lookup(
        model.item_embedding,
        suffix.item_ids,
        suffix.lengths,
        collective_timing=collective_timing,
    )
    if suffix.batch_size == 0:
        if retained is not None:
            raise ValueError(
                "empty integrated append-only has a retained cache"
            )
        return IntegratedAppendOnlyResult(
            fragment=None,
            last_hidden=torch.empty(
                (0, model.dense_model.cfg.hidden_size),
                dtype=torch.float32,
                device=suffix.device,
            ),
            lookup_metrics=lookup.metrics,
        )
    if (
        retained is None
        or retained.record_ids != suffix.record_ids
        or retained.k.device != suffix.device
        or bool(torch.any(suffix.lengths < 1))
    ):
        raise ValueError("integrated append-only inputs differ")
    retained_lengths = tuple(
        int(value) for value in retained.lengths.tolist()
    )
    retained_offsets = [0]
    for length in retained_lengths:
        retained_offsets.append(retained_offsets[-1] + length)
    retained_width = max(retained_lengths)
    shape = (
        retained.k.shape[0],
        retained.batch_size,
        retained_width,
        retained.k.shape[2],
    )
    padded_k = torch.zeros(
        shape,
        dtype=torch.float32,
        device=retained.k.device,
    )
    padded_v = torch.zeros_like(padded_k)
    for row, length in enumerate(retained_lengths):
        start = retained_offsets[row]
        stop = retained_offsets[row + 1]
        padded_k[:, row, :length].copy_(retained.k[:, start:stop])
        padded_v[:, row, :length].copy_(retained.v[:, start:stop])
    hidden, new_cache = (
        model.dense_model.forward_with_cache_from_item_embeddings_new_kv(
            cached_kv=HSTUKVCache(
                k=padded_k,
                v=padded_v,
                seq_len=retained_width,
            ),
            new_item_vectors=lookup.item_vectors,
            new_behaviors=suffix.behaviors,
            new_time_deltas=suffix.time_deltas,
        )
    )
    suffix_fragment = pack_padded_cache(
        new_cache,
        suffix.lengths,
        suffix.record_ids,
        target_version,
        target_version,
        dtype=dtype,
    )
    lengths = retained.lengths + suffix.lengths.long()
    offsets = torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.long,
                device=retained.k.device,
            ),
            lengths.cumsum(0),
        )
    )
    rows = torch.arange(suffix.batch_size, device=suffix.device)
    return IntegratedAppendOnlyResult(
        fragment=IntegratedAppendOnlyKVBatch(
            retained=retained,
            suffix=suffix_fragment,
            lengths=lengths,
            offsets=offsets,
            target_version=target_version,
        ),
        last_hidden=hidden[rows, suffix.lengths.long() - 1],
        lookup_metrics=lookup.metrics,
    )
