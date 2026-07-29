from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch import nn

from ..models import HSTU, HSTUKVCache
from .cohort_jagged import JaggedMigratedKVBatch
from .design2_plan import canonical_sha256
from .recompute import (
    RawHistoryBatch,
    exact_hidden_and_kv_from_item_embeddings,
)
from .stage46_chain import pack_padded_cache

CollectivePhaseGuard = Callable[
    [str],
    AbstractContextManager[object] | None,
]


def _item_ids_sha256(item_ids: torch.Tensor) -> str:
    return canonical_sha256(
        {"item_ids": item_ids.detach().cpu().tolist()}
    )


def _unique_item_ids_sha256(item_ids: torch.Tensor) -> str:
    return _item_ids_sha256(torch.unique(item_ids, sorted=True))


def _split_item_ids_sha256(
    item_ids: torch.Tensor,
    counts: tuple[int, ...],
) -> tuple[str, ...]:
    values = []
    offset = 0
    for count in counts:
        values.append(_item_ids_sha256(item_ids[offset : offset + count]))
        offset += count
    if offset != item_ids.numel():
        raise ValueError("item ID split counts differ from payload")
    return tuple(values)


def modulo_embedding_owner(
    item_ids: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if item_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("item_ids must have an integer dtype")
    return torch.remainder(item_ids, world_size)


def modulo_embedding_local_id(
    item_ids: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if item_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("item_ids must have an integer dtype")
    return torch.div(item_ids, world_size, rounding_mode="floor")


def modulo_embedding_local_rows(
    num_embeddings: int,
    rank: int,
    world_size: int,
) -> int:
    if (
        num_embeddings < 1
        or world_size < 1
        or not 0 <= rank < world_size
    ):
        raise ValueError("invalid modulo embedding layout")
    if rank >= num_embeddings:
        return 0
    return (num_embeddings - 1 - rank) // world_size + 1


@dataclass(frozen=True)
class D2ShardedEmbeddingMetrics:
    rank: int
    world_size: int
    requested_tokens: int
    unique_tokens: int
    local_requested_tokens: int
    local_unique_tokens: int
    remote_requested_tokens: int
    remote_unique_tokens: int
    served_remote_requested_tokens: int
    served_remote_unique_tokens: int
    requested_ids_sha256: str
    requested_unique_ids_sha256: str
    local_requested_ids_sha256: str
    local_unique_ids_sha256: str
    remote_requested_ids_sha256: str
    remote_unique_ids_sha256: str
    served_remote_ids_sha256: str
    served_remote_unique_ids_sha256: str
    remote_send_ids_sha256: tuple[str, ...]
    remote_receive_ids_sha256: tuple[str, ...]
    remote_send_counts: tuple[int, ...]
    remote_receive_counts: tuple[int, ...]
    counts_collective_input_bytes: int
    counts_collective_output_bytes: int
    id_collective_input_bytes: int
    id_collective_output_bytes: int
    vector_collective_input_bytes: int
    vector_collective_output_bytes: int
    off_diagonal_send_bytes: int
    off_diagonal_receive_bytes: int
    collective_calls: int
    off_diagonal_collective_calls: int
    counts_collective_seconds: float
    id_collective_seconds: float
    vector_collective_seconds: float

    def __post_init__(self) -> None:
        integer_values = (
            self.rank,
            self.world_size,
            self.requested_tokens,
            self.unique_tokens,
            self.local_requested_tokens,
            self.local_unique_tokens,
            self.remote_requested_tokens,
            self.remote_unique_tokens,
            self.served_remote_requested_tokens,
            self.served_remote_unique_tokens,
            self.counts_collective_input_bytes,
            self.counts_collective_output_bytes,
            self.id_collective_input_bytes,
            self.id_collective_output_bytes,
            self.vector_collective_input_bytes,
            self.vector_collective_output_bytes,
            self.off_diagonal_send_bytes,
            self.off_diagonal_receive_bytes,
            self.collective_calls,
            self.off_diagonal_collective_calls,
        )
        if (
            self.world_size < 1
            or not 0 <= self.rank < self.world_size
            or any(value < 0 for value in integer_values)
            or len(self.remote_send_counts) != self.world_size
            or len(self.remote_receive_counts) != self.world_size
            or len(self.remote_send_ids_sha256) != self.world_size
            or len(self.remote_receive_ids_sha256) != self.world_size
            or any(value < 0 for value in self.remote_send_counts)
            or any(value < 0 for value in self.remote_receive_counts)
            or sum(self.remote_send_counts)
            != self.remote_requested_tokens
            or sum(self.remote_receive_counts)
            != self.served_remote_requested_tokens
            or any(
                len(value) != 64
                for value in (
                    self.requested_ids_sha256,
                    self.requested_unique_ids_sha256,
                    self.local_requested_ids_sha256,
                    self.local_unique_ids_sha256,
                    self.remote_requested_ids_sha256,
                    self.remote_unique_ids_sha256,
                    self.served_remote_ids_sha256,
                    self.served_remote_unique_ids_sha256,
                    *self.remote_send_ids_sha256,
                    *self.remote_receive_ids_sha256,
                )
            )
            or self.local_requested_tokens
            + self.remote_requested_tokens
            != self.requested_tokens
            or any(
                not math.isfinite(value) or value < 0
                for value in (
                    self.counts_collective_seconds,
                    self.id_collective_seconds,
                    self.vector_collective_seconds,
                )
            )
        ):
            raise ValueError("sharded embedding metrics are invalid")

    @property
    def counts_collective_payload_bytes(self) -> int:
        return (
            self.counts_collective_input_bytes
            + self.counts_collective_output_bytes
        )

    @property
    def id_collective_payload_bytes(self) -> int:
        return (
            self.id_collective_input_bytes
            + self.id_collective_output_bytes
        )

    @property
    def vector_collective_payload_bytes(self) -> int:
        return (
            self.vector_collective_input_bytes
            + self.vector_collective_output_bytes
        )

    @property
    def actual_collective_tensor_payload_bytes(self) -> int:
        return (
            self.counts_collective_payload_bytes
            + self.id_collective_payload_bytes
            + self.vector_collective_payload_bytes
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
    def off_diagonal_collective_seconds(self) -> float:
        return self.collective_seconds

    def to_dict(self) -> dict[str, int | float]:
        values = asdict(self)
        values["counts_collective_payload_bytes"] = (
            self.counts_collective_payload_bytes
        )
        values["id_collective_payload_bytes"] = (
            self.id_collective_payload_bytes
        )
        values["vector_collective_payload_bytes"] = (
            self.vector_collective_payload_bytes
        )
        values["actual_collective_tensor_payload_bytes"] = (
            self.actual_collective_tensor_payload_bytes
        )
        values["off_diagonal_bytes"] = self.off_diagonal_bytes
        values["collective_seconds"] = self.collective_seconds
        values["off_diagonal_collective_seconds"] = (
            self.off_diagonal_collective_seconds
        )
        return values


@dataclass(frozen=True)
class D2ShardedEmbeddingLookup:
    item_vectors: torch.Tensor
    metrics: D2ShardedEmbeddingMetrics


class ModuloRowShardedEmbedding(nn.Module):
    def __init__(
        self,
        local_weight: torch.Tensor,
        num_embeddings: int,
        rank: int,
        world_size: int,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        if (
            local_weight.ndim != 2
            or not local_weight.is_floating_point()
            or local_weight.shape[1] < 1
            or local_weight.shape[0]
            != modulo_embedding_local_rows(
                num_embeddings,
                rank,
                world_size,
            )
        ):
            raise ValueError("local embedding shard differs from modulo layout")
        self.num_embeddings = num_embeddings
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.register_buffer(
            "local_weight",
            local_weight.detach().to(dtype=torch.float32).contiguous(),
        )

    @property
    def hidden_size(self) -> int:
        return self.local_weight.shape[1]

    def _validate_group(self) -> None:
        if self.world_size == 1:
            if dist.is_initialized():
                if (
                    dist.get_world_size(group=self.process_group) != 1
                    or dist.get_rank(group=self.process_group) != self.rank
                ):
                    raise ValueError("process group differs from embedding layout")
            return
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size(group=self.process_group)
            != self.world_size
            or dist.get_rank(group=self.process_group) != self.rank
        ):
            raise RuntimeError("distributed process group is not initialized")

    def _phase(
        self,
        guard: CollectivePhaseGuard | None,
        name: str,
    ) -> AbstractContextManager[object]:
        if guard is None:
            return nullcontext()
        value = guard(name)
        return nullcontext() if value is None else value

    def _synchronize(self) -> None:
        if self.local_weight.device.type == "cuda":
            torch.cuda.synchronize(self.local_weight.device)

    def _collective(
        self,
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

    def lookup(
        self,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
        collective_phase_guard: CollectivePhaseGuard | None = None,
    ) -> D2ShardedEmbeddingLookup:
        self._validate_group()
        if (
            item_ids.ndim != 2
            or lengths.shape != (item_ids.shape[0],)
            or item_ids.device != self.local_weight.device
            or lengths.device != item_ids.device
            or lengths.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
        ):
            raise ValueError("sharded embedding lookup inputs differ")
        width = item_ids.shape[1]
        if bool(torch.any(lengths < 0)) or bool(
            torch.any(lengths > width)
        ):
            raise ValueError("lengths exceed the padded item width")
        valid = (
            torch.arange(width, device=item_ids.device).unsqueeze(0)
            < lengths.long().unsqueeze(1)
        )
        positions = torch.nonzero(
            valid.reshape(-1),
            as_tuple=False,
        ).flatten()
        flat_ids = item_ids.reshape(-1)
        requested_ids = flat_ids.index_select(0, positions).long()
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
        vectors = torch.zeros(
            (item_ids.numel(), self.hidden_size),
            dtype=torch.float32,
            device=item_ids.device,
        )
        if local_ids.numel():
            local_rows = modulo_embedding_local_id(
                local_ids,
                self.world_size,
            )
            vectors.index_copy_(
                0,
                local_positions,
                self.local_weight.index_select(0, local_rows),
            )
        requested = requested_ids.numel()
        unique = torch.unique(requested_ids).numel()
        local_requested = local_ids.numel()
        local_unique = torch.unique(local_ids).numel()
        remote_requested = remote_ids.numel()
        remote_unique = torch.unique(remote_ids).numel()
        requested_ids_sha256 = _item_ids_sha256(requested_ids)
        requested_unique_ids_sha256 = _unique_item_ids_sha256(
            requested_ids
        )
        local_requested_ids_sha256 = _item_ids_sha256(local_ids)
        local_unique_ids_sha256 = _unique_item_ids_sha256(local_ids)
        remote_requested_ids_sha256 = _item_ids_sha256(remote_ids)
        remote_unique_ids_sha256 = _unique_item_ids_sha256(remote_ids)
        if self.world_size == 1:
            empty_ids_sha256 = _item_ids_sha256(requested_ids[:0])
            metrics = D2ShardedEmbeddingMetrics(
                rank=self.rank,
                world_size=self.world_size,
                requested_tokens=requested,
                unique_tokens=unique,
                local_requested_tokens=local_requested,
                local_unique_tokens=local_unique,
                remote_requested_tokens=remote_requested,
                remote_unique_tokens=remote_unique,
                served_remote_requested_tokens=0,
                served_remote_unique_tokens=0,
                requested_ids_sha256=requested_ids_sha256,
                requested_unique_ids_sha256=requested_unique_ids_sha256,
                local_requested_ids_sha256=local_requested_ids_sha256,
                local_unique_ids_sha256=local_unique_ids_sha256,
                remote_requested_ids_sha256=remote_requested_ids_sha256,
                remote_unique_ids_sha256=remote_unique_ids_sha256,
                served_remote_ids_sha256=empty_ids_sha256,
                served_remote_unique_ids_sha256=empty_ids_sha256,
                remote_send_ids_sha256=(empty_ids_sha256,),
                remote_receive_ids_sha256=(empty_ids_sha256,),
                remote_send_counts=(0,),
                remote_receive_counts=(0,),
                counts_collective_input_bytes=0,
                counts_collective_output_bytes=0,
                id_collective_input_bytes=0,
                id_collective_output_bytes=0,
                vector_collective_input_bytes=0,
                vector_collective_output_bytes=0,
                off_diagonal_send_bytes=0,
                off_diagonal_receive_bytes=0,
                collective_calls=0,
                off_diagonal_collective_calls=0,
                counts_collective_seconds=0.0,
                id_collective_seconds=0.0,
                vector_collective_seconds=0.0,
            )
            return D2ShardedEmbeddingLookup(
                item_vectors=vectors.reshape(
                    *item_ids.shape,
                    self.hidden_size,
                ),
                metrics=metrics,
            )
        send_counts = torch.bincount(
            owners[remote_mask],
            minlength=self.world_size,
        ).to(device=item_ids.device, dtype=torch.int64)
        receive_counts = torch.empty_like(send_counts)
        counts_seconds = self._collective(
            "embedding_counts",
            collective_phase_guard,
            lambda: dist.all_to_all_single(
                receive_counts,
                send_counts,
                group=self.process_group,
            ),
        )
        order = torch.argsort(owners[remote_mask], stable=True)
        ordered_ids = remote_ids.index_select(0, order)
        ordered_positions = remote_positions.index_select(0, order)
        send_local_ids = modulo_embedding_local_id(
            ordered_ids,
            self.world_size,
        ).contiguous()
        received_local_ids = torch.empty(
            int(receive_counts.sum().item()),
            dtype=torch.int64,
            device=item_ids.device,
        )
        send_splits = tuple(int(value) for value in send_counts.tolist())
        receive_splits = tuple(
            int(value) for value in receive_counts.tolist()
        )
        send_ids_sha256 = _split_item_ids_sha256(
            ordered_ids,
            send_splits,
        )
        id_seconds = self._collective(
            "embedding_ids",
            collective_phase_guard,
            lambda: dist.all_to_all_single(
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
        response_vectors = self.local_weight.index_select(
            0,
            received_local_ids,
        )
        received_vectors = torch.empty(
            (send_local_ids.numel(), self.hidden_size),
            dtype=torch.float32,
            device=item_ids.device,
        )
        vector_seconds = self._collective(
            "embedding_vectors",
            collective_phase_guard,
            lambda: dist.all_to_all_single(
                received_vectors,
                response_vectors,
                output_split_sizes=send_splits,
                input_split_sizes=receive_splits,
                group=self.process_group,
            ),
        )
        if ordered_positions.numel():
            vectors.index_copy_(
                0,
                ordered_positions,
                received_vectors,
            )
        count_element_bytes = send_counts.element_size()
        id_element_bytes = send_local_ids.element_size()
        vector_element_bytes = self.local_weight.element_size()
        count_input_bytes = send_counts.numel() * count_element_bytes
        count_output_bytes = (
            receive_counts.numel() * count_element_bytes
        )
        id_input_bytes = send_local_ids.numel() * id_element_bytes
        id_output_bytes = (
            received_local_ids.numel() * id_element_bytes
        )
        vector_input_bytes = (
            response_vectors.numel() * vector_element_bytes
        )
        vector_output_bytes = (
            received_vectors.numel() * vector_element_bytes
        )
        count_off_diagonal_bytes = (
            self.world_size - 1
        ) * count_element_bytes
        received_global_ids = (
            received_local_ids * self.world_size + self.rank
        )
        receive_ids_sha256 = _split_item_ids_sha256(
            received_global_ids,
            receive_splits,
        )
        metrics = D2ShardedEmbeddingMetrics(
            rank=self.rank,
            world_size=self.world_size,
            requested_tokens=requested,
            unique_tokens=unique,
            local_requested_tokens=local_requested,
            local_unique_tokens=local_unique,
            remote_requested_tokens=remote_requested,
            remote_unique_tokens=remote_unique,
            served_remote_requested_tokens=received_local_ids.numel(),
            served_remote_unique_tokens=torch.unique(
                received_global_ids
            ).numel(),
            requested_ids_sha256=requested_ids_sha256,
            requested_unique_ids_sha256=requested_unique_ids_sha256,
            local_requested_ids_sha256=local_requested_ids_sha256,
            local_unique_ids_sha256=local_unique_ids_sha256,
            remote_requested_ids_sha256=remote_requested_ids_sha256,
            remote_unique_ids_sha256=remote_unique_ids_sha256,
            served_remote_ids_sha256=_item_ids_sha256(
                received_global_ids
            ),
            served_remote_unique_ids_sha256=_unique_item_ids_sha256(
                received_global_ids
            ),
            remote_send_ids_sha256=send_ids_sha256,
            remote_receive_ids_sha256=receive_ids_sha256,
            remote_send_counts=send_splits,
            remote_receive_counts=receive_splits,
            counts_collective_input_bytes=count_input_bytes,
            counts_collective_output_bytes=count_output_bytes,
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
            off_diagonal_collective_calls=3,
            counts_collective_seconds=counts_seconds,
            id_collective_seconds=id_seconds,
            vector_collective_seconds=vector_seconds,
        )
        return D2ShardedEmbeddingLookup(
            item_vectors=vectors.reshape(
                *item_ids.shape,
                self.hidden_size,
            ),
            metrics=metrics,
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
        collective_phase_guard: CollectivePhaseGuard | None = None,
    ) -> D2ShardedEmbeddingLookup:
        return self.lookup(
            item_ids,
            lengths,
            collective_phase_guard=collective_phase_guard,
        )


class ForbiddenFullItemEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, hidden_size: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("full item embedding lookup is forbidden")

    def score(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        raise RuntimeError("full item embedding scoring is forbidden")


@dataclass(frozen=True)
class D2ShardedHSTU:
    dense_model: HSTU
    item_embedding: ModuloRowShardedEmbedding
    rank: int
    world_size: int


def build_modulo_sharded_hstu_from_cpu(
    model: HSTU,
    rank: int,
    world_size: int,
    device: torch.device | str,
    process_group: dist.ProcessGroup | None = None,
) -> D2ShardedHSTU:
    source_devices = {
        value.device
        for value in (*model.parameters(), *model.buffers())
    }
    if source_devices != {torch.device("cpu")}:
        raise ValueError("source HSTU must reside entirely on CPU")
    if not 0 <= rank < world_size:
        raise ValueError("rank differs from world_size")
    target = torch.device(device)
    full_weight = model.item_emb.weight.detach()
    global_rows = torch.arange(
        rank,
        full_weight.shape[0],
        world_size,
        dtype=torch.long,
    )
    local_weight = full_weight.index_select(
        0,
        global_rows,
    ).to(device=target, dtype=torch.float32)
    dense_model = HSTU.__new__(HSTU)
    nn.Module.__init__(dense_model)
    dense_model.cfg = copy.deepcopy(model.cfg)
    dense_model.item_emb = ForbiddenFullItemEmbedding(
        full_weight.shape[0],
        full_weight.shape[1],
    )
    dense_model.behavior_emb = copy.deepcopy(model.behavior_emb)
    dense_model.temporal_enc = copy.deepcopy(model.temporal_enc)
    dense_model.in_proj = copy.deepcopy(model.in_proj)
    dense_model.input_dropout = copy.deepcopy(model.input_dropout)
    dense_model.blocks = copy.deepcopy(model.blocks)
    dense_model.final_norm = copy.deepcopy(model.final_norm)
    dense_model.to(target)
    dense_model.train(model.training)
    item_embedding = ModuloRowShardedEmbedding(
        local_weight=local_weight,
        num_embeddings=full_weight.shape[0],
        rank=rank,
        world_size=world_size,
        process_group=process_group,
    )
    return D2ShardedHSTU(
        dense_model=dense_model,
        item_embedding=item_embedding,
        rank=rank,
        world_size=world_size,
    )


@dataclass(frozen=True)
class D2ShardedExactResult:
    fragment: JaggedMigratedKVBatch | None
    last_hidden: torch.Tensor
    lookup_metrics: D2ShardedEmbeddingMetrics


@dataclass(frozen=True)
class D2ShardedAppendResult:
    updated_cache: HSTUKVCache
    last_hidden: torch.Tensor
    lengths: torch.Tensor
    lookup_metrics: D2ShardedEmbeddingMetrics


@torch.inference_mode()
def sharded_exact_jagged_hidden_and_kv(
    model: D2ShardedHSTU,
    batch: RawHistoryBatch,
    target_version: str,
    dtype: torch.dtype = torch.float16,
    collective_phase_guard: CollectivePhaseGuard | None = None,
) -> D2ShardedExactResult:
    if (
        batch.device != model.item_embedding.local_weight.device
        or model.dense_model.cfg.hidden_size
        != model.item_embedding.hidden_size
    ):
        raise ValueError("sharded exact model and history batch differ")
    lookup = model.item_embedding.lookup(
        batch.item_ids,
        batch.lengths,
        collective_phase_guard=collective_phase_guard,
    )
    if batch.batch_size == 0:
        return D2ShardedExactResult(
            fragment=None,
            last_hidden=torch.empty(
                (0, model.dense_model.cfg.hidden_size),
                dtype=lookup.item_vectors.dtype,
                device=batch.device,
            ),
            lookup_metrics=lookup.metrics,
        )
    if bool(torch.any(batch.lengths < 1)):
        raise ValueError("exact K/V records must contain at least one token")
    hidden, cache = exact_hidden_and_kv_from_item_embeddings(
        model.dense_model,
        lookup.item_vectors,
        batch.behaviors,
        batch.time_deltas,
        lengths=batch.lengths,
    )
    fragment = pack_padded_cache(
        cache,
        batch.lengths,
        batch.record_ids,
        target_version,
        target_version,
        dtype=dtype,
    )
    return D2ShardedExactResult(
        fragment=fragment,
        last_hidden=model.dense_model.last_hidden(
            hidden,
            batch.lengths,
        ),
        lookup_metrics=lookup.metrics,
    )


@torch.inference_mode()
def sharded_append_padded_cache(
    model: D2ShardedHSTU,
    retained_cache: HSTUKVCache,
    suffix_item_ids: torch.Tensor,
    suffix_behaviors: torch.Tensor,
    suffix_time_deltas: torch.Tensor,
    suffix_lengths: torch.Tensor,
    retained_lengths: torch.Tensor | None = None,
    collective_phase_guard: CollectivePhaseGuard | None = None,
) -> D2ShardedAppendResult:
    batch_size = suffix_item_ids.shape[0]
    if (
        suffix_item_ids.ndim != 2
        or suffix_behaviors.shape != suffix_item_ids.shape
        or suffix_time_deltas.shape != suffix_item_ids.shape
        or suffix_lengths.shape != (batch_size,)
        or suffix_item_ids.device
        != model.item_embedding.local_weight.device
        or suffix_behaviors.device != suffix_item_ids.device
        or suffix_time_deltas.device != suffix_item_ids.device
        or suffix_lengths.device != suffix_item_ids.device
        or retained_cache.k.device != suffix_item_ids.device
        or retained_cache.v.device != suffix_item_ids.device
        or retained_cache.k.ndim != 4
        or retained_cache.k.shape != retained_cache.v.shape
        or retained_cache.k.shape[0]
        != len(model.dense_model.blocks)
        or retained_cache.k.shape[1] != batch_size
        or retained_cache.k.shape[2] != retained_cache.seq_len
    ):
        raise ValueError("sharded append inputs differ")
    if retained_lengths is None:
        prepared_retained_lengths = torch.full(
            (batch_size,),
            retained_cache.seq_len,
            dtype=torch.long,
            device=suffix_item_ids.device,
        )
    else:
        if (
            retained_lengths.shape != (batch_size,)
            or retained_lengths.device != suffix_item_ids.device
        ):
            raise ValueError("retained lengths and cache batch differ")
        prepared_retained_lengths = retained_lengths.long()
    prepared_suffix_lengths = suffix_lengths.long()
    if (
        bool(torch.any(prepared_retained_lengths < 0))
        or bool(
            torch.any(
                prepared_retained_lengths > retained_cache.seq_len
            )
        )
        or bool(torch.any(prepared_suffix_lengths < 0))
        or bool(
            torch.any(
                prepared_suffix_lengths > suffix_item_ids.shape[1]
            )
        )
    ):
        raise ValueError("append lengths exceed padded inputs")
    lookup = model.item_embedding.lookup(
        suffix_item_ids,
        prepared_suffix_lengths,
        collective_phase_guard=collective_phase_guard,
    )
    if batch_size == 0:
        return D2ShardedAppendResult(
            updated_cache=retained_cache,
            last_hidden=torch.empty(
                (0, model.dense_model.cfg.hidden_size),
                dtype=lookup.item_vectors.dtype,
                device=suffix_item_ids.device,
            ),
            lengths=prepared_retained_lengths,
            lookup_metrics=lookup.metrics,
        )
    if bool(torch.any(prepared_suffix_lengths < 1)):
        raise ValueError("append records must contain at least one token")
    row_caches = []
    row_hidden = []
    updated_lengths = (
        prepared_retained_lengths + prepared_suffix_lengths
    )
    for row in range(batch_size):
        retained_length = int(prepared_retained_lengths[row])
        suffix_length = int(prepared_suffix_lengths[row])
        row_cache = HSTUKVCache(
            k=retained_cache.k[
                :, row : row + 1, :retained_length
            ].contiguous(),
            v=retained_cache.v[
                :, row : row + 1, :retained_length
            ].contiguous(),
            seq_len=retained_length,
        )
        hidden, updated = (
            model.dense_model.forward_with_cache_from_item_embeddings(
                row_cache,
                lookup.item_vectors[
                    row : row + 1, :suffix_length
                ],
                suffix_behaviors[
                    row : row + 1, :suffix_length
                ],
                suffix_time_deltas[
                    row : row + 1, :suffix_length
                ],
            )
        )
        row_caches.append(updated)
        row_hidden.append(hidden[:, -1])
    width = int(updated_lengths.max())
    shape = (
        retained_cache.k.shape[0],
        batch_size,
        width,
        retained_cache.k.shape[3],
    )
    updated_k = torch.zeros(
        shape,
        dtype=row_caches[0].k.dtype,
        device=suffix_item_ids.device,
    )
    updated_v = torch.zeros_like(updated_k)
    for row, cache in enumerate(row_caches):
        length = int(updated_lengths[row])
        updated_k[:, row, :length].copy_(cache.k[:, 0])
        updated_v[:, row, :length].copy_(cache.v[:, 0])
    return D2ShardedAppendResult(
        updated_cache=HSTUKVCache(
            k=updated_k,
            v=updated_v,
            seq_len=width,
        ),
        last_hidden=torch.cat(row_hidden, dim=0),
        lengths=updated_lengths,
        lookup_metrics=lookup.metrics,
    )
