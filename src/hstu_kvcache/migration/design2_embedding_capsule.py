from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.distributed as dist

from .design2_embedding import modulo_embedding_local_rows

D2_EMBEDDING_CAPSULE_PROTOCOL = (
    "cohortkv_d2_plan_compiled_embedding_capsule_v1"
)
_INDEX_BYTES = 8
_DENSE_UNIQUE_MAX_BYTES = 64 * 1024 * 1024
_DENSE_UNIQUE_REQUEST_RATIO = 4


@dataclass(frozen=True)
class D2EmbeddingCapsuleRankPlan:
    rank: int
    world_size: int
    unique_item_ids: tuple[int, ...]
    inverse_slots: tuple[int, ...]
    local_rows: tuple[int, ...]
    local_capsule_slots: tuple[int, ...]
    send_local_rows_by_requester: tuple[tuple[int, ...], ...]
    receive_capsule_slots_by_owner: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        unique_count = len(self.unique_item_ids)
        if (
            self.world_size < 1
            or not 0 <= self.rank < self.world_size
            or tuple(sorted(set(self.unique_item_ids)))
            != self.unique_item_ids
            or any(value < 0 for value in self.unique_item_ids)
            or any(
                not 0 <= value < unique_count
                for value in self.inverse_slots
            )
            or len(self.local_rows) != len(self.local_capsule_slots)
            or any(value < 0 for value in self.local_rows)
            or any(
                not 0 <= value < unique_count
                for value in self.local_capsule_slots
            )
            or len(self.send_local_rows_by_requester)
            != self.world_size
            or len(self.receive_capsule_slots_by_owner)
            != self.world_size
            or self.send_local_rows_by_requester[self.rank]
            or self.receive_capsule_slots_by_owner[self.rank]
            or any(
                value < 0
                for values in self.send_local_rows_by_requester
                for value in values
            )
            or any(
                not 0 <= value < unique_count
                for values in self.receive_capsule_slots_by_owner
                for value in values
            )
        ):
            raise ValueError("D2 embedding capsule rank plan is invalid")
        covered = (
            *self.local_capsule_slots,
            *(
                value
                for values in self.receive_capsule_slots_by_owner
                for value in values
            ),
        )
        if tuple(sorted(covered)) != tuple(range(unique_count)):
            raise ValueError(
                "D2 embedding capsule slots are not covered exactly once"
            )

    @property
    def requested_tokens(self) -> int:
        return len(self.inverse_slots)

    @property
    def unique_tokens(self) -> int:
        return len(self.unique_item_ids)

    @property
    def local_unique_tokens(self) -> int:
        return len(self.local_rows)

    @property
    def remote_unique_tokens(self) -> int:
        return sum(self.receive_splits)

    @property
    def served_remote_unique_tokens(self) -> int:
        return sum(self.send_splits)

    @property
    def send_splits(self) -> tuple[int, ...]:
        return tuple(
            len(values) for values in self.send_local_rows_by_requester
        )

    @property
    def receive_splits(self) -> tuple[int, ...]:
        return tuple(
            len(values) for values in self.receive_capsule_slots_by_owner
        )

    @property
    def plan_nbytes(self) -> int:
        index_count = (
            len(self.unique_item_ids)
            + len(self.inverse_slots)
            + len(self.local_rows)
            + len(self.local_capsule_slots)
            + sum(self.send_splits)
            + sum(self.receive_splits)
            + 2 * self.world_size
        )
        return index_count * _INDEX_BYTES


@dataclass(frozen=True)
class D2EmbeddingCapsulePlan:
    num_embeddings: int
    world_size: int
    ranks: tuple[D2EmbeddingCapsuleRankPlan, ...]
    compile_seconds: float
    protocol: str = D2_EMBEDDING_CAPSULE_PROTOCOL

    def __post_init__(self) -> None:
        if (
            self.protocol != D2_EMBEDDING_CAPSULE_PROTOCOL
            or self.num_embeddings < 1
            or self.world_size < 1
            or not math.isfinite(self.compile_seconds)
            or self.compile_seconds < 0
            or len(self.ranks) != self.world_size
            or tuple(value.rank for value in self.ranks)
            != tuple(range(self.world_size))
            or any(value.world_size != self.world_size for value in self.ranks)
            or any(
                item_id >= self.num_embeddings
                for value in self.ranks
                for item_id in value.unique_item_ids
            )
        ):
            raise ValueError("D2 embedding capsule plan is invalid")
        for source in range(self.world_size):
            maximum_row = modulo_embedding_local_rows(
                self.num_embeddings,
                source,
                self.world_size,
            )
            for destination in range(self.world_size):
                rows = self.ranks[source].send_local_rows_by_requester[
                    destination
                ]
                slots = self.ranks[
                    destination
                ].receive_capsule_slots_by_owner[source]
                expected = tuple(
                    self.ranks[destination].unique_item_ids[slot]
                    // self.world_size
                    for slot in slots
                )
                if (
                    len(rows) != len(slots)
                    or rows != expected
                    or any(row >= maximum_row for row in rows)
                ):
                    raise ValueError(
                        "D2 embedding capsule source manifest differs"
                    )

    @property
    def plan_nbytes(self) -> int:
        return sum(value.plan_nbytes for value in self.ranks)

    def rank_plan(self, rank: int) -> D2EmbeddingCapsuleRankPlan:
        if not 0 <= rank < self.world_size:
            raise ValueError("D2 embedding capsule rank is invalid")
        return self.ranks[rank]


@dataclass(frozen=True)
class D2MaterializedEmbeddingCapsuleRankPlan:
    rank: int
    world_size: int
    num_embeddings: int
    requested_tokens: int
    unique_tokens: int
    local_unique_tokens: int
    remote_unique_tokens: int
    served_remote_unique_tokens: int
    rank_plan_bytes: int
    global_plan_bytes: int
    plan_compile_seconds: float
    materialization_seconds: float
    inverse_slots: torch.Tensor
    local_rows: torch.Tensor
    local_capsule_slots: torch.Tensor
    send_local_rows: torch.Tensor
    receive_capsule_slots: torch.Tensor
    send_splits: tuple[int, ...]
    receive_splits: tuple[int, ...]

    def __post_init__(self) -> None:
        tensors = (
            self.inverse_slots,
            self.local_rows,
            self.local_capsule_slots,
            self.send_local_rows,
            self.receive_capsule_slots,
        )
        integer_values = (
            self.rank,
            self.world_size,
            self.num_embeddings,
            self.requested_tokens,
            self.unique_tokens,
            self.local_unique_tokens,
            self.remote_unique_tokens,
            self.served_remote_unique_tokens,
            self.rank_plan_bytes,
            self.global_plan_bytes,
        )
        if (
            self.world_size < 1
            or self.num_embeddings < 1
            or not 0 <= self.rank < self.world_size
            or any(value < 0 for value in integer_values)
            or len(self.send_splits) != self.world_size
            or len(self.receive_splits) != self.world_size
            or any(value < 0 for value in self.send_splits)
            or any(value < 0 for value in self.receive_splits)
            or sum(self.send_splits) != self.send_local_rows.numel()
            or sum(self.receive_splits)
            != self.receive_capsule_slots.numel()
            or self.inverse_slots.numel() != self.requested_tokens
            or self.local_rows.numel() != self.local_unique_tokens
            or self.local_capsule_slots.numel()
            != self.local_unique_tokens
            or self.receive_capsule_slots.numel()
            != self.remote_unique_tokens
            or self.send_local_rows.numel()
            != self.served_remote_unique_tokens
            or self.local_unique_tokens + self.remote_unique_tokens
            != self.unique_tokens
            or any(
                value.ndim != 1
                or value.dtype != torch.long
                or not value.is_contiguous()
                for value in tensors
            )
            or len({value.device for value in tensors}) != 1
            or any(
                not math.isfinite(value) or value < 0
                for value in (
                    self.plan_compile_seconds,
                    self.materialization_seconds,
                )
            )
        ):
            raise ValueError(
                "materialized D2 embedding capsule plan is invalid"
            )

    @property
    def device(self) -> torch.device:
        return self.inverse_slots.device

    @property
    def materialized_plan_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.inverse_slots,
                self.local_rows,
                self.local_capsule_slots,
                self.send_local_rows,
                self.receive_capsule_slots,
            )
        )


@dataclass(frozen=True)
class D2EmbeddingCapsuleMetrics:
    rank: int
    world_size: int
    requested_tokens: int
    unique_tokens: int
    local_unique_tokens: int
    remote_unique_tokens: int
    served_remote_unique_tokens: int
    rank_plan_bytes: int
    global_plan_bytes: int
    materialized_plan_bytes: int
    counts_collective_bytes: int
    id_collective_bytes: int
    vector_collective_input_bytes: int
    vector_collective_output_bytes: int
    off_diagonal_send_bytes: int
    off_diagonal_receive_bytes: int
    collective_calls: int
    plan_compile_seconds: float
    plan_materialization_seconds: float
    collective_seconds: float
    execution_seconds: float

    def __post_init__(self) -> None:
        integers = (
            self.rank,
            self.world_size,
            self.requested_tokens,
            self.unique_tokens,
            self.local_unique_tokens,
            self.remote_unique_tokens,
            self.served_remote_unique_tokens,
            self.rank_plan_bytes,
            self.global_plan_bytes,
            self.materialized_plan_bytes,
            self.counts_collective_bytes,
            self.id_collective_bytes,
            self.vector_collective_input_bytes,
            self.vector_collective_output_bytes,
            self.off_diagonal_send_bytes,
            self.off_diagonal_receive_bytes,
            self.collective_calls,
        )
        if (
            self.world_size < 1
            or not 0 <= self.rank < self.world_size
            or any(value < 0 for value in integers)
            or self.local_unique_tokens + self.remote_unique_tokens
            != self.unique_tokens
            or self.counts_collective_bytes != 0
            or self.id_collective_bytes != 0
            or self.off_diagonal_send_bytes
            != self.vector_collective_input_bytes
            or self.off_diagonal_receive_bytes
            != self.vector_collective_output_bytes
            or self.collective_calls != (0 if self.world_size == 1 else 1)
            or any(
                not math.isfinite(value) or value < 0
                for value in (
                    self.plan_compile_seconds,
                    self.plan_materialization_seconds,
                    self.collective_seconds,
                    self.execution_seconds,
                )
            )
            or self.execution_seconds < self.collective_seconds
        ):
            raise ValueError("D2 embedding capsule metrics are invalid")

    @property
    def vector_collective_payload_bytes(self) -> int:
        return (
            self.vector_collective_input_bytes
            + self.vector_collective_output_bytes
        )

    @property
    def off_diagonal_bytes(self) -> int:
        return (
            self.off_diagonal_send_bytes
            + self.off_diagonal_receive_bytes
        )

    def to_dict(self) -> dict[str, int | float]:
        values = asdict(self)
        values["vector_collective_payload_bytes"] = (
            self.vector_collective_payload_bytes
        )
        values["off_diagonal_bytes"] = self.off_diagonal_bytes
        return values


@dataclass(frozen=True)
class D2EmbeddingCapsuleLookup:
    item_vectors: torch.Tensor
    unique_vectors: torch.Tensor
    metrics: D2EmbeddingCapsuleMetrics


def _compiled_rank_plan(
    *,
    rank: int,
    world_size: int,
    unique_item_ids: tuple[int, ...],
    inverse_slots: tuple[int, ...],
    local_rows: tuple[int, ...],
    local_capsule_slots: tuple[int, ...],
    send_local_rows_by_requester: tuple[tuple[int, ...], ...],
    receive_capsule_slots_by_owner: tuple[tuple[int, ...], ...],
) -> D2EmbeddingCapsuleRankPlan:
    value = object.__new__(D2EmbeddingCapsuleRankPlan)
    object.__setattr__(value, "rank", rank)
    object.__setattr__(value, "world_size", world_size)
    object.__setattr__(value, "unique_item_ids", unique_item_ids)
    object.__setattr__(value, "inverse_slots", inverse_slots)
    object.__setattr__(value, "local_rows", local_rows)
    object.__setattr__(
        value,
        "local_capsule_slots",
        local_capsule_slots,
    )
    object.__setattr__(
        value,
        "send_local_rows_by_requester",
        send_local_rows_by_requester,
    )
    object.__setattr__(
        value,
        "receive_capsule_slots_by_owner",
        receive_capsule_slots_by_owner,
    )
    return value


def _compiled_plan(
    *,
    num_embeddings: int,
    world_size: int,
    ranks: tuple[D2EmbeddingCapsuleRankPlan, ...],
    compile_seconds: float,
) -> D2EmbeddingCapsulePlan:
    value = object.__new__(D2EmbeddingCapsulePlan)
    object.__setattr__(value, "num_embeddings", num_embeddings)
    object.__setattr__(value, "world_size", world_size)
    object.__setattr__(value, "ranks", ranks)
    object.__setattr__(value, "compile_seconds", compile_seconds)
    object.__setattr__(
        value,
        "protocol",
        D2_EMBEDDING_CAPSULE_PROTOCOL,
    )
    return value


def compile_d2_embedding_capsule(
    requester_item_ids: Sequence[Sequence[int]],
    num_embeddings: int,
    world_size: int,
) -> D2EmbeddingCapsulePlan:
    started = time.perf_counter()
    if (
        num_embeddings < 1
        or world_size < 1
        or len(requester_item_ids) != world_size
    ):
        raise ValueError("D2 embedding capsule compiler input is invalid")
    requests = []
    try:
        for values in requester_item_ids:
            request = np.asarray(values, dtype=np.int64)
            if request.ndim != 1:
                raise ValueError
            requests.append(request)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "D2 embedding capsule item ID is invalid"
        ) from error
    if any(
        values.size
        and (
            int(values.min()) < 0
            or int(values.max()) >= num_embeddings
        )
        for values in requests
    ):
        raise ValueError("D2 embedding capsule item ID is invalid")
    unique_item_ids: list[tuple[int, ...]] = [()] * world_size
    inverse_slots: list[tuple[int, ...]] = [()] * world_size
    local_rows: list[tuple[int, ...]] = [()] * world_size
    local_slots: list[tuple[int, ...]] = [()] * world_size
    send_rows = [
        [() for _ in range(world_size)] for _ in range(world_size)
    ]
    receive_slots = [
        [() for _ in range(world_size)] for _ in range(world_size)
    ]
    dense_workspace_bytes = num_embeddings * (
        np.dtype(np.bool_).itemsize + np.dtype(np.int64).itemsize
    )
    for requester, request in enumerate(requests):
        use_dense = (
            request.size > 0
            and dense_workspace_bytes <= _DENSE_UNIQUE_MAX_BYTES
            and num_embeddings
            <= _DENSE_UNIQUE_REQUEST_RATIO * request.size
        )
        if use_dense:
            present = np.zeros(num_embeddings, dtype=np.bool_)
            present[request] = True
            unique = np.flatnonzero(present)
            slot_by_item = np.empty(num_embeddings, dtype=np.int64)
            slot_by_item[unique] = np.arange(
                unique.size,
                dtype=np.int64,
            )
            inverse = slot_by_item[request]
        else:
            unique, inverse = np.unique(
                request,
                return_inverse=True,
            )
        owners = np.remainder(unique, world_size)
        rows = np.floor_divide(unique, world_size)
        for owner in range(world_size):
            slots = np.flatnonzero(owners == owner)
            owner_rows = rows[slots]
            row_tuple = tuple(owner_rows.tolist())
            slot_tuple = tuple(slots.tolist())
            if owner == requester:
                local_rows[requester] = row_tuple
                local_slots[requester] = slot_tuple
            else:
                send_rows[owner][requester] = row_tuple
                receive_slots[requester][owner] = slot_tuple
        unique_item_ids[requester] = tuple(unique.tolist())
        inverse_slots[requester] = tuple(inverse.tolist())
    ranks = tuple(
        _compiled_rank_plan(
            rank=rank,
            world_size=world_size,
            unique_item_ids=unique_item_ids[rank],
            inverse_slots=inverse_slots[rank],
            local_rows=local_rows[rank],
            local_capsule_slots=local_slots[rank],
            send_local_rows_by_requester=tuple(send_rows[rank]),
            receive_capsule_slots_by_owner=tuple(
                receive_slots[rank]
            ),
        )
        for rank in range(world_size)
    )
    compile_seconds = time.perf_counter() - started
    return _compiled_plan(
        num_embeddings=num_embeddings,
        world_size=world_size,
        ranks=ranks,
        compile_seconds=compile_seconds,
    )


def _index_tensor(
    values: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long, device=device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_process_group(
    world_size: int,
    rank: int,
    process_group: dist.ProcessGroup | None,
) -> None:
    if world_size == 1:
        if dist.is_initialized() and (
            dist.get_world_size(group=process_group) != 1
            or dist.get_rank(group=process_group) != rank
        ):
            raise ValueError(
                "process group differs from D2 embedding capsule"
            )
        return
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size(group=process_group) != world_size
        or dist.get_rank(group=process_group) != rank
    ):
        raise RuntimeError(
            "distributed process group is not initialized for capsule"
        )


def materialize_d2_embedding_capsule(
    plan: D2EmbeddingCapsulePlan,
    rank: int,
    device: torch.device | str,
) -> D2MaterializedEmbeddingCapsuleRankPlan:
    rank_plan = plan.rank_plan(rank)
    target = torch.device(device)
    _synchronize(target)
    started = time.perf_counter()
    inverse_slots = _index_tensor(rank_plan.inverse_slots, target)
    local_rows = _index_tensor(rank_plan.local_rows, target)
    local_capsule_slots = _index_tensor(
        rank_plan.local_capsule_slots,
        target,
    )
    send_local_rows = _index_tensor(
        tuple(
            value
            for values in rank_plan.send_local_rows_by_requester
            for value in values
        ),
        target,
    )
    receive_capsule_slots = _index_tensor(
        tuple(
            value
            for values in rank_plan.receive_capsule_slots_by_owner
            for value in values
        ),
        target,
    )
    _synchronize(target)
    materialization_seconds = time.perf_counter() - started
    return D2MaterializedEmbeddingCapsuleRankPlan(
        rank=rank,
        world_size=plan.world_size,
        num_embeddings=plan.num_embeddings,
        requested_tokens=rank_plan.requested_tokens,
        unique_tokens=rank_plan.unique_tokens,
        local_unique_tokens=rank_plan.local_unique_tokens,
        remote_unique_tokens=rank_plan.remote_unique_tokens,
        served_remote_unique_tokens=(
            rank_plan.served_remote_unique_tokens
        ),
        rank_plan_bytes=rank_plan.plan_nbytes,
        global_plan_bytes=plan.plan_nbytes,
        plan_compile_seconds=plan.compile_seconds,
        materialization_seconds=materialization_seconds,
        inverse_slots=inverse_slots,
        local_rows=local_rows,
        local_capsule_slots=local_capsule_slots,
        send_local_rows=send_local_rows,
        receive_capsule_slots=receive_capsule_slots,
        send_splits=rank_plan.send_splits,
        receive_splits=rank_plan.receive_splits,
    )


def execute_d2_embedding_capsule(
    materialized: D2MaterializedEmbeddingCapsuleRankPlan,
    local_weight: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
) -> D2EmbeddingCapsuleLookup:
    _validate_process_group(
        materialized.world_size,
        materialized.rank,
        process_group,
    )
    expected_rows = modulo_embedding_local_rows(
        materialized.num_embeddings,
        materialized.rank,
        materialized.world_size,
    )
    if (
        local_weight.ndim != 2
        or not local_weight.is_floating_point()
        or local_weight.shape[0] != expected_rows
        or local_weight.shape[1] < 1
        or local_weight.device != materialized.device
    ):
        raise ValueError(
            "local weight differs from D2 embedding capsule layout"
        )
    device = local_weight.device
    hidden_size = local_weight.shape[1]
    _synchronize(device)
    execution_started = time.perf_counter()
    unique_vectors = torch.empty(
        (materialized.unique_tokens, hidden_size),
        dtype=local_weight.dtype,
        device=device,
    )
    if materialized.local_unique_tokens:
        unique_vectors.index_copy_(
            0,
            materialized.local_capsule_slots,
            local_weight.index_select(
                0,
                materialized.local_rows,
            ),
        )
    response_vectors = local_weight.index_select(
        0,
        materialized.send_local_rows,
    )
    received_vectors = torch.empty(
        (materialized.remote_unique_tokens, hidden_size),
        dtype=local_weight.dtype,
        device=device,
    )
    collective_seconds = 0.0
    if materialized.world_size > 1:
        _synchronize(device)
        started = time.perf_counter()
        dist.all_to_all_single(
            received_vectors,
            response_vectors,
            output_split_sizes=list(materialized.receive_splits),
            input_split_sizes=list(materialized.send_splits),
            group=process_group,
        )
        _synchronize(device)
        collective_seconds = time.perf_counter() - started
    if materialized.remote_unique_tokens:
        unique_vectors.index_copy_(
            0,
            materialized.receive_capsule_slots,
            received_vectors,
        )
    item_vectors = unique_vectors.index_select(
        0,
        materialized.inverse_slots,
    )
    _synchronize(device)
    execution_seconds = time.perf_counter() - execution_started
    element_bytes = local_weight.element_size()
    input_bytes = response_vectors.numel() * element_bytes
    output_bytes = received_vectors.numel() * element_bytes
    metrics = D2EmbeddingCapsuleMetrics(
        rank=materialized.rank,
        world_size=materialized.world_size,
        requested_tokens=materialized.requested_tokens,
        unique_tokens=materialized.unique_tokens,
        local_unique_tokens=materialized.local_unique_tokens,
        remote_unique_tokens=materialized.remote_unique_tokens,
        served_remote_unique_tokens=(
            materialized.served_remote_unique_tokens
        ),
        rank_plan_bytes=materialized.rank_plan_bytes,
        global_plan_bytes=materialized.global_plan_bytes,
        materialized_plan_bytes=(
            materialized.materialized_plan_bytes
        ),
        counts_collective_bytes=0,
        id_collective_bytes=0,
        vector_collective_input_bytes=input_bytes,
        vector_collective_output_bytes=output_bytes,
        off_diagonal_send_bytes=input_bytes,
        off_diagonal_receive_bytes=output_bytes,
        collective_calls=(
            0 if materialized.world_size == 1 else 1
        ),
        plan_compile_seconds=materialized.plan_compile_seconds,
        plan_materialization_seconds=(
            materialized.materialization_seconds
        ),
        collective_seconds=collective_seconds,
        execution_seconds=execution_seconds,
    )
    return D2EmbeddingCapsuleLookup(
        item_vectors=item_vectors,
        unique_vectors=unique_vectors,
        metrics=metrics,
    )
