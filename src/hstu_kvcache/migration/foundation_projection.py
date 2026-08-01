from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from .design2_embedding import (
    modulo_embedding_local_id,
    modulo_embedding_local_rows,
    modulo_embedding_owner,
)

CollectivePhaseGuard = Callable[
    [str],
    AbstractContextManager[object] | None,
]


def _tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


@dataclass(frozen=True)
class FoundationProjectionCapacity:
    num_embeddings: int
    embedding_width: int
    output_width: int
    rank: int
    world_size: int
    local_embedding_rows: int
    global_embedding_parameter_bytes: int
    local_embedding_parameter_bytes: int
    projection_parameter_bytes: int

    def __post_init__(self) -> None:
        if (
            self.num_embeddings < 1
            or self.embedding_width < 1
            or self.output_width < 1
            or self.world_size < 1
            or not 0 <= self.rank < self.world_size
            or self.local_embedding_rows
            != modulo_embedding_local_rows(
                self.num_embeddings,
                self.rank,
                self.world_size,
            )
            or self.global_embedding_parameter_bytes
            != self.num_embeddings * self.embedding_width * 4
            or self.local_embedding_parameter_bytes
            != self.local_embedding_rows * self.embedding_width * 4
            or self.projection_parameter_bytes
            != self.embedding_width * self.output_width * 4
        ):
            raise ValueError("foundation projection capacity is invalid")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def foundation_projection_capacity(
    *,
    num_embeddings: int,
    embedding_width: int,
    output_width: int,
    rank: int,
    world_size: int,
) -> FoundationProjectionCapacity:
    local_rows = modulo_embedding_local_rows(
        num_embeddings,
        rank,
        world_size,
    )
    return FoundationProjectionCapacity(
        num_embeddings=num_embeddings,
        embedding_width=embedding_width,
        output_width=output_width,
        rank=rank,
        world_size=world_size,
        local_embedding_rows=local_rows,
        global_embedding_parameter_bytes=(
            num_embeddings * embedding_width * 4
        ),
        local_embedding_parameter_bytes=local_rows * embedding_width * 4,
        projection_parameter_bytes=embedding_width * output_width * 4,
    )


@dataclass(frozen=True)
class FoundationProjectedLookupMetrics:
    rank: int
    world_size: int
    requested_tokens: int
    local_requested_tokens: int
    remote_requested_tokens: int
    served_remote_tokens: int
    response_element_bytes: int
    returned_tensor_bytes: int
    returned_valid_vector_bytes: int
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
    local_projection_seconds: float
    remote_projection_seconds: float
    lookup_seconds: float

    def __post_init__(self) -> None:
        integer_values = (
            self.rank,
            self.world_size,
            self.requested_tokens,
            self.local_requested_tokens,
            self.remote_requested_tokens,
            self.served_remote_tokens,
            self.response_element_bytes,
            self.returned_tensor_bytes,
            self.returned_valid_vector_bytes,
            self.counts_collective_input_bytes,
            self.counts_collective_output_bytes,
            self.id_collective_input_bytes,
            self.id_collective_output_bytes,
            self.vector_collective_input_bytes,
            self.vector_collective_output_bytes,
            self.off_diagonal_send_bytes,
            self.off_diagonal_receive_bytes,
            self.collective_calls,
        )
        timing_values = (
            self.counts_collective_seconds,
            self.id_collective_seconds,
            self.vector_collective_seconds,
            self.local_projection_seconds,
            self.remote_projection_seconds,
            self.lookup_seconds,
        )
        if (
            self.world_size < 1
            or not 0 <= self.rank < self.world_size
            or any(value < 0 for value in integer_values)
            or self.local_requested_tokens
            + self.remote_requested_tokens
            != self.requested_tokens
            or any(
                not math.isfinite(value) or value < 0
                for value in timing_values
            )
        ):
            raise ValueError("foundation projected lookup metrics are invalid")

    @property
    def counts_collective_tensor_bytes(self) -> int:
        return (
            self.counts_collective_input_bytes
            + self.counts_collective_output_bytes
        )

    @property
    def id_collective_tensor_bytes(self) -> int:
        return (
            self.id_collective_input_bytes
            + self.id_collective_output_bytes
        )

    @property
    def vector_collective_tensor_bytes(self) -> int:
        return (
            self.vector_collective_input_bytes
            + self.vector_collective_output_bytes
        )

    @property
    def collective_tensor_bytes(self) -> int:
        return (
            self.counts_collective_tensor_bytes
            + self.id_collective_tensor_bytes
            + self.vector_collective_tensor_bytes
        )

    @property
    def off_diagonal_bytes(self) -> int:
        return (
            self.off_diagonal_send_bytes
            + self.off_diagonal_receive_bytes
        )

    @property
    def collective_seconds(self) -> float:
        return (
            self.counts_collective_seconds
            + self.id_collective_seconds
            + self.vector_collective_seconds
        )

    @property
    def projection_seconds(self) -> float:
        return (
            self.local_projection_seconds
            + self.remote_projection_seconds
        )

    def to_dict(self) -> dict[str, int | float]:
        values = asdict(self)
        values["counts_collective_tensor_bytes"] = (
            self.counts_collective_tensor_bytes
        )
        values["id_collective_tensor_bytes"] = (
            self.id_collective_tensor_bytes
        )
        values["vector_collective_tensor_bytes"] = (
            self.vector_collective_tensor_bytes
        )
        values["collective_tensor_bytes"] = self.collective_tensor_bytes
        values["off_diagonal_bytes"] = self.off_diagonal_bytes
        values["collective_seconds"] = self.collective_seconds
        values["projection_seconds"] = self.projection_seconds
        return values


@dataclass(frozen=True)
class FoundationProjectedLookup:
    item_vectors: torch.Tensor
    metrics: FoundationProjectedLookupMetrics


class FoundationProjectedModuloEmbedding(nn.Module):
    def __init__(
        self,
        *,
        local_weight: torch.Tensor,
        projection_weight: torch.Tensor,
        num_embeddings: int,
        rank: int,
        world_size: int,
        response_dtype: torch.dtype = torch.float16,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        if (
            local_weight.ndim != 2
            or local_weight.dtype != torch.float32
            or projection_weight.ndim != 2
            or projection_weight.dtype != torch.float32
            or projection_weight.device != local_weight.device
            or local_weight.shape[1] < 1
            or projection_weight.shape[0] < 1
            or projection_weight.shape[1] != local_weight.shape[1]
            or local_weight.shape[0]
            != modulo_embedding_local_rows(
                num_embeddings,
                rank,
                world_size,
            )
            or response_dtype
            not in {torch.float16, torch.bfloat16, torch.float32}
        ):
            raise ValueError("foundation projected embedding layout differs")
        self.num_embeddings = num_embeddings
        self.rank = rank
        self.world_size = world_size
        self.response_dtype = response_dtype
        self.process_group = process_group
        self.register_buffer(
            "local_weight",
            local_weight.detach().contiguous(),
        )
        self.register_buffer(
            "projection_weight",
            projection_weight.detach().contiguous(),
        )

    @property
    def embedding_width(self) -> int:
        return self.local_weight.shape[1]

    @property
    def output_width(self) -> int:
        return self.projection_weight.shape[0]

    @property
    def capacity(self) -> FoundationProjectionCapacity:
        return foundation_projection_capacity(
            num_embeddings=self.num_embeddings,
            embedding_width=self.embedding_width,
            output_width=self.output_width,
            rank=self.rank,
            world_size=self.world_size,
        )

    def _validate_group(self) -> None:
        if self.world_size == 1:
            if dist.is_initialized() and (
                dist.get_world_size(group=self.process_group) != 1
                or dist.get_rank(group=self.process_group) != self.rank
            ):
                raise ValueError(
                    "process group differs from foundation projection"
                )
            return
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size(group=self.process_group)
            != self.world_size
            or dist.get_rank(group=self.process_group) != self.rank
        ):
            raise RuntimeError("distributed process group is not initialized")

    def _synchronize(self) -> None:
        if self.local_weight.device.type == "cuda":
            torch.cuda.synchronize(self.local_weight.device)

    def _phase(
        self,
        guard: CollectivePhaseGuard | None,
        name: str,
    ) -> AbstractContextManager[object]:
        if guard is None:
            return nullcontext()
        value = guard(name)
        return nullcontext() if value is None else value

    def _collective(
        self,
        *,
        name: str,
        guard: CollectivePhaseGuard | None,
        action: Callable[[], None],
    ) -> float:
        with self._phase(guard, name):
            self._synchronize()
            started = time.perf_counter()
            action()
            self._synchronize()
            return time.perf_counter() - started

    def _project(self, rows: torch.Tensor) -> tuple[torch.Tensor, float]:
        self._synchronize()
        started = time.perf_counter()
        vectors = F.linear(rows, self.projection_weight).to(
            dtype=self.response_dtype
        )
        self._synchronize()
        return vectors, time.perf_counter() - started

    def lookup(
        self,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
        collective_phase_guard: CollectivePhaseGuard | None = None,
    ) -> FoundationProjectedLookup:
        self._validate_group()
        if (
            item_ids.ndim != 2
            or lengths.shape != (item_ids.shape[0],)
            or item_ids.device != self.local_weight.device
            or lengths.device != item_ids.device
            or item_ids.dtype
            not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
            or lengths.dtype
            not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
        ):
            raise ValueError("foundation projected lookup inputs differ")
        width = item_ids.shape[1]
        lengths_long = lengths.long()
        if bool(torch.any(lengths_long < 0)) or bool(
            torch.any(lengths_long > width)
        ):
            raise ValueError("lengths exceed the padded item width")
        self._synchronize()
        lookup_started = time.perf_counter()
        valid = (
            torch.arange(width, device=item_ids.device).unsqueeze(0)
            < lengths_long.unsqueeze(1)
        )
        positions = torch.nonzero(
            valid.reshape(-1),
            as_tuple=False,
        ).flatten()
        requested_ids = item_ids.reshape(-1).index_select(
            0,
            positions,
        ).long()
        if requested_ids.numel() and (
            bool(torch.any(requested_ids < 0))
            or bool(torch.any(requested_ids >= self.num_embeddings))
        ):
            raise ValueError("valid item id exceeds the embedding vocabulary")
        owners = modulo_embedding_owner(
            requested_ids,
            self.world_size,
        )
        local_mask = owners == self.rank
        remote_mask = ~local_mask
        local_ids = requested_ids[local_mask]
        remote_ids = requested_ids[remote_mask]
        local_positions = positions[local_mask]
        remote_positions = positions[remote_mask]
        output = torch.zeros(
            (item_ids.numel(), self.output_width),
            dtype=self.response_dtype,
            device=item_ids.device,
        )
        local_vectors, local_projection_seconds = self._project(
            self.local_weight.index_select(
                0,
                modulo_embedding_local_id(
                    local_ids,
                    self.world_size,
                ),
            )
        )
        if local_positions.numel():
            output.index_copy_(0, local_positions, local_vectors)
        requested_tokens = requested_ids.numel()
        local_requested_tokens = local_ids.numel()
        remote_requested_tokens = remote_ids.numel()
        if self.world_size == 1:
            self._synchronize()
            metrics = FoundationProjectedLookupMetrics(
                rank=self.rank,
                world_size=self.world_size,
                requested_tokens=requested_tokens,
                local_requested_tokens=local_requested_tokens,
                remote_requested_tokens=0,
                served_remote_tokens=0,
                response_element_bytes=output.element_size(),
                returned_tensor_bytes=_tensor_bytes(output),
                returned_valid_vector_bytes=(
                    requested_tokens
                    * self.output_width
                    * output.element_size()
                ),
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
                local_projection_seconds=local_projection_seconds,
                remote_projection_seconds=0.0,
                lookup_seconds=time.perf_counter() - lookup_started,
            )
            return FoundationProjectedLookup(
                item_vectors=output.reshape(
                    *item_ids.shape,
                    self.output_width,
                ),
                metrics=metrics,
            )
        send_counts = torch.bincount(
            owners[remote_mask],
            minlength=self.world_size,
        ).to(device=item_ids.device, dtype=torch.int64)
        receive_counts = torch.empty_like(send_counts)
        counts_seconds = self._collective(
            name="foundation_projection_counts",
            guard=collective_phase_guard,
            action=lambda: dist.all_to_all_single(
                receive_counts,
                send_counts,
                group=self.process_group,
            ),
        )
        remote_owners = owners[remote_mask]
        order = torch.argsort(remote_owners, stable=True)
        ordered_ids = remote_ids.index_select(0, order)
        ordered_positions = remote_positions.index_select(0, order)
        send_local_ids = modulo_embedding_local_id(
            ordered_ids,
            self.world_size,
        ).contiguous()
        send_splits = tuple(int(value) for value in send_counts.tolist())
        receive_splits = tuple(
            int(value) for value in receive_counts.tolist()
        )
        received_local_ids = torch.empty(
            int(receive_counts.sum().item()),
            dtype=torch.int64,
            device=item_ids.device,
        )
        id_seconds = self._collective(
            name="foundation_projection_ids",
            guard=collective_phase_guard,
            action=lambda: dist.all_to_all_single(
                received_local_ids,
                send_local_ids,
                output_split_sizes=receive_splits,
                input_split_sizes=send_splits,
                group=self.process_group,
            ),
        )
        if received_local_ids.numel() and (
            bool(torch.any(received_local_ids < 0))
            or bool(
                torch.any(
                    received_local_ids >= self.local_weight.shape[0]
                )
            )
        ):
            raise RuntimeError("received local item id exceeds shard")
        response_vectors, remote_projection_seconds = self._project(
            self.local_weight.index_select(0, received_local_ids)
        )
        received_vectors = torch.empty(
            (send_local_ids.numel(), self.output_width),
            dtype=self.response_dtype,
            device=item_ids.device,
        )
        vector_seconds = self._collective(
            name="foundation_projection_vectors",
            guard=collective_phase_guard,
            action=lambda: dist.all_to_all_single(
                received_vectors,
                response_vectors,
                output_split_sizes=send_splits,
                input_split_sizes=receive_splits,
                group=self.process_group,
            ),
        )
        if ordered_positions.numel():
            output.index_copy_(
                0,
                ordered_positions,
                received_vectors,
            )
        count_element_bytes = send_counts.element_size()
        response_element_bytes = output.element_size()
        counts_input_bytes = _tensor_bytes(send_counts)
        counts_output_bytes = _tensor_bytes(receive_counts)
        id_input_bytes = _tensor_bytes(send_local_ids)
        id_output_bytes = _tensor_bytes(received_local_ids)
        vector_input_bytes = _tensor_bytes(response_vectors)
        vector_output_bytes = _tensor_bytes(received_vectors)
        count_off_diagonal_bytes = (
            self.world_size - 1
        ) * count_element_bytes
        self._synchronize()
        metrics = FoundationProjectedLookupMetrics(
            rank=self.rank,
            world_size=self.world_size,
            requested_tokens=requested_tokens,
            local_requested_tokens=local_requested_tokens,
            remote_requested_tokens=remote_requested_tokens,
            served_remote_tokens=received_local_ids.numel(),
            response_element_bytes=response_element_bytes,
            returned_tensor_bytes=_tensor_bytes(output),
            returned_valid_vector_bytes=(
                requested_tokens
                * self.output_width
                * response_element_bytes
            ),
            counts_collective_input_bytes=counts_input_bytes,
            counts_collective_output_bytes=counts_output_bytes,
            id_collective_input_bytes=id_input_bytes,
            id_collective_output_bytes=id_output_bytes,
            vector_collective_input_bytes=vector_input_bytes,
            vector_collective_output_bytes=vector_output_bytes,
            off_diagonal_send_bytes=(
                count_off_diagonal_bytes
                + id_input_bytes
                + vector_input_bytes
            ),
            off_diagonal_receive_bytes=(
                count_off_diagonal_bytes
                + id_output_bytes
                + vector_output_bytes
            ),
            collective_calls=3,
            counts_collective_seconds=counts_seconds,
            id_collective_seconds=id_seconds,
            vector_collective_seconds=vector_seconds,
            local_projection_seconds=local_projection_seconds,
            remote_projection_seconds=remote_projection_seconds,
            lookup_seconds=time.perf_counter() - lookup_started,
        )
        return FoundationProjectedLookup(
            item_vectors=output.reshape(
                *item_ids.shape,
                self.output_width,
            ),
            metrics=metrics,
        )
