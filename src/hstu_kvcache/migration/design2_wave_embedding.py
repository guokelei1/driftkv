from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .design2_embedding import (
    ModuloRowShardedEmbedding,
)
from .design2_integrated import (
    IntegratedLookupMetrics,
    fast_modulo_sharded_lookup,
)
from .design2_plan import D2ActionRecord

D2_WAVE_EMBEDDING_BRANCHES = ("mixed", "all_exact")
D2_WAVE_EMBEDDING_LOOKUP_KERNEL = (
    "fast_modulo_sharded_lookup_no_payload_hash_v1"
)
D2_WAVE_EMBEDDING_MODES = (
    "demand_token_microbatch",
    "one_batch_no_dedup",
    "wave_scope_unique_cache",
)


@dataclass(frozen=True)
class D2WaveEmbeddingLogicalRequest:
    branch: str
    rank: int
    world_size: int
    item_ids: torch.Tensor
    phase_token_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (
            self.branch not in D2_WAVE_EMBEDDING_BRANCHES
            or self.world_size < 1
            or not 0 <= self.rank < self.world_size
            or self.item_ids.ndim != 1
            or self.item_ids.dtype != torch.int64
            or not self.item_ids.is_contiguous()
            or len({name for name, _ in self.phase_token_counts})
            != len(self.phase_token_counts)
            or any(
                not name or tokens < 0
                for name, tokens in self.phase_token_counts
            )
            or sum(tokens for _, tokens in self.phase_token_counts)
            != self.item_ids.numel()
        ):
            raise ValueError("D2 wave embedding logical request is invalid")

    @property
    def logical_tokens(self) -> int:
        return self.item_ids.numel()

    @property
    def logical_unique_tokens(self) -> int:
        return torch.unique(self.item_ids).numel()

    @property
    def logical_remote_tokens(self) -> int:
        return int(
            torch.count_nonzero(
                torch.remainder(self.item_ids, self.world_size)
                != self.rank
            ).item()
        )

    @property
    def logical_remote_unique_tokens(self) -> int:
        remote = self.item_ids[
            torch.remainder(self.item_ids, self.world_size) != self.rank
        ]
        return torch.unique(remote).numel()


def _history_tensor(
    target_item_ids: Mapping[int, Sequence[int] | torch.Tensor],
    record: D2ActionRecord,
) -> torch.Tensor:
    if record.record_id not in target_item_ids:
        raise ValueError("D2 wave target history is missing")
    history = torch.as_tensor(
        target_item_ids[record.record_id],
        dtype=torch.int64,
        device="cpu",
    ).flatten()
    if (
        history.numel() != record.final_tokens
        or bool(torch.any(history < 0))
    ):
        raise ValueError("D2 wave target history differs from action plan")
    return history


def _phase_specs(
    branch: str,
) -> tuple[tuple[str, str], ...]:
    if branch == "mixed":
        return (
            ("scheduled_exact_retained", "scheduled"),
            ("natural_exact_target_prefix", "natural"),
            ("delta_append", "delta"),
            ("latest_append", "latest"),
        )
    if branch == "all_exact":
        return (
            ("all_exact_retained", "retained"),
            ("natural_exact_target_prefix", "natural"),
            ("delta_append", "delta"),
            ("latest_append", "latest"),
        )
    raise ValueError("D2 wave embedding branch is unsupported")


def _phase_item_ids(
    record: D2ActionRecord,
    history: torch.Tensor,
    selector: str,
) -> torch.Tensor | None:
    if selector == "scheduled":
        if record.requested_reason != "scheduled_exact":
            return None
        return history[: record.retained_tokens]
    if selector == "retained":
        if record.requested_reason == "natural_exact":
            return None
        return history[: record.retained_tokens]
    if selector == "natural":
        if record.requested_reason != "natural_exact":
            return None
        return history[: record.target_prefix_tokens]
    if selector == "delta":
        if record.requested_reason == "natural_exact":
            return None
        return history[
            record.delta_start : record.target_prefix_tokens
        ]
    if selector == "latest":
        return history[
            record.target_prefix_tokens : record.final_tokens
        ]
    raise ValueError("D2 wave embedding phase selector is unsupported")


def build_d2_wave_embedding_logical_request(
    records: Sequence[D2ActionRecord],
    target_item_ids: Mapping[int, Sequence[int] | torch.Tensor],
    owner_map: Mapping[int, int],
    *,
    branch: str,
    rank: int,
    world_size: int,
) -> D2WaveEmbeddingLogicalRequest:
    if (
        branch not in D2_WAVE_EMBEDDING_BRANCHES
        or world_size < 1
        or not 0 <= rank < world_size
        or not records
    ):
        raise ValueError("D2 wave embedding request inputs are invalid")
    record_ids = tuple(record.record_id for record in records)
    if (
        record_ids != tuple(sorted(record_ids))
        or set(owner_map) != set(record_ids)
        or any(
            not 0 <= int(owner_map[record_id]) < world_size
            for record_id in record_ids
        )
    ):
        raise ValueError("D2 wave embedding owner map is invalid")
    local_records = tuple(
        record
        for record in records
        if int(owner_map[record.record_id]) == rank
    )
    histories = {
        record.record_id: _history_tensor(target_item_ids, record)
        for record in local_records
    }
    parts = []
    phase_counts = []
    for phase, selector in _phase_specs(branch):
        phase_parts = []
        for record in local_records:
            item_ids = _phase_item_ids(
                record,
                histories[record.record_id],
                selector,
            )
            if item_ids is not None and item_ids.numel():
                phase_parts.append(item_ids)
        phase_count = sum(value.numel() for value in phase_parts)
        phase_counts.append((phase, phase_count))
        parts.extend(phase_parts)
    item_ids = (
        torch.cat(parts).contiguous()
        if parts
        else torch.empty(0, dtype=torch.int64)
    )
    return D2WaveEmbeddingLogicalRequest(
        branch=branch,
        rank=rank,
        world_size=world_size,
        item_ids=item_ids,
        phase_token_counts=tuple(phase_counts),
    )


def d2_wave_embedding_demand_calls(
    logical_tokens: int,
    token_microbatch: int,
) -> int:
    if logical_tokens < 0 or token_microbatch < 1:
        raise ValueError("D2 wave embedding microbatch input is invalid")
    return max(1, math.ceil(logical_tokens / token_microbatch))


@dataclass(frozen=True)
class D2WaveEmbeddingLookupPlan:
    request: D2WaveEmbeddingLogicalRequest
    mode: str
    token_microbatch: int
    lookup_batches: tuple[torch.Tensor, ...]
    inverse: torch.Tensor | None

    def __post_init__(self) -> None:
        if (
            self.mode not in D2_WAVE_EMBEDDING_MODES
            or self.token_microbatch < 1
            or not self.lookup_batches
            or any(
                value.ndim != 1
                or value.dtype != torch.int64
                or value.device != self.request.item_ids.device
                or not value.is_contiguous()
                for value in self.lookup_batches
            )
        ):
            raise ValueError("D2 wave embedding lookup plan is invalid")
        requested = sum(value.numel() for value in self.lookup_batches)
        if self.mode == "demand_token_microbatch":
            if (
                self.inverse is not None
                or requested != self.request.logical_tokens
                or any(
                    value.numel() > self.token_microbatch
                    for value in self.lookup_batches
                )
                or not torch.equal(
                    torch.cat(self.lookup_batches),
                    self.request.item_ids,
                )
            ):
                raise ValueError(
                    "D2 demand-order embedding plan is invalid"
                )
        elif self.mode == "one_batch_no_dedup":
            if (
                len(self.lookup_batches) != 1
                or self.inverse is not None
                or not torch.equal(
                    self.lookup_batches[0],
                    self.request.item_ids,
                )
            ):
                raise ValueError(
                    "D2 one-batch embedding plan is invalid"
                )
        else:
            if (
                len(self.lookup_batches) != 1
                or self.inverse is None
                or self.inverse.ndim != 1
                or self.inverse.dtype != torch.int64
                or self.inverse.device != self.request.item_ids.device
                or self.inverse.numel() != self.request.logical_tokens
                or requested != self.request.logical_unique_tokens
                or not torch.equal(
                    self.lookup_batches[0].index_select(
                        0,
                        self.inverse,
                    ),
                    self.request.item_ids,
                )
            ):
                raise ValueError(
                    "D2 wave-scope embedding cache plan is invalid"
                )

    @property
    def lookup_calls(self) -> int:
        return len(self.lookup_batches)

    @property
    def cache_item_id_bytes(self) -> int:
        if self.mode != "wave_scope_unique_cache":
            return 0
        return (
            self.lookup_batches[0].numel()
            * self.lookup_batches[0].element_size()
        )

    @property
    def inverse_bytes(self) -> int:
        if self.inverse is None:
            return 0
        return self.inverse.numel() * self.inverse.element_size()

    def cache_vector_bytes(self, hidden_size: int) -> int:
        if hidden_size < 1:
            raise ValueError("D2 embedding hidden size must be positive")
        if self.mode != "wave_scope_unique_cache":
            return 0
        return (
            self.lookup_batches[0].numel()
            * hidden_size
            * torch.tensor([], dtype=torch.float32).element_size()
        )

    def to(
        self,
        device: torch.device | str,
    ) -> D2WaveEmbeddingLookupPlan:
        target = torch.device(device)
        request = D2WaveEmbeddingLogicalRequest(
            branch=self.request.branch,
            rank=self.request.rank,
            world_size=self.request.world_size,
            item_ids=self.request.item_ids.to(target),
            phase_token_counts=self.request.phase_token_counts,
        )
        return D2WaveEmbeddingLookupPlan(
            request=request,
            mode=self.mode,
            token_microbatch=self.token_microbatch,
            lookup_batches=tuple(
                value.to(target) for value in self.lookup_batches
            ),
            inverse=(
                None
                if self.inverse is None
                else self.inverse.to(target)
            ),
        )


def build_d2_wave_embedding_lookup_plan(
    request: D2WaveEmbeddingLogicalRequest,
    *,
    mode: str,
    token_microbatch: int,
    lookup_calls: int | None = None,
) -> D2WaveEmbeddingLookupPlan:
    if (
        mode not in D2_WAVE_EMBEDDING_MODES
        or token_microbatch < 1
    ):
        raise ValueError("D2 wave embedding mode is invalid")
    if mode == "demand_token_microbatch":
        local_calls = d2_wave_embedding_demand_calls(
            request.logical_tokens,
            token_microbatch,
        )
        total_calls = local_calls if lookup_calls is None else lookup_calls
        if total_calls < local_calls:
            raise ValueError(
                "D2 wave embedding lookup calls cannot drop requests"
            )
        batches = [
            request.item_ids[start : start + token_microbatch].contiguous()
            for start in range(
                0,
                request.logical_tokens,
                token_microbatch,
            )
        ]
        while len(batches) < total_calls:
            batches.append(request.item_ids[:0])
        return D2WaveEmbeddingLookupPlan(
            request=request,
            mode=mode,
            token_microbatch=token_microbatch,
            lookup_batches=tuple(batches),
            inverse=None,
        )
    if lookup_calls not in {None, 1}:
        raise ValueError("D2 one-wave lookup uses exactly one call")
    if mode == "one_batch_no_dedup":
        return D2WaveEmbeddingLookupPlan(
            request=request,
            mode=mode,
            token_microbatch=token_microbatch,
            lookup_batches=(request.item_ids,),
            inverse=None,
        )
    unique, inverse = torch.unique(
        request.item_ids,
        sorted=True,
        return_inverse=True,
    )
    return D2WaveEmbeddingLookupPlan(
        request=request,
        mode=mode,
        token_microbatch=token_microbatch,
        lookup_batches=(unique.contiguous(),),
        inverse=inverse.contiguous(),
    )


@dataclass(frozen=True)
class D2WaveEmbeddingExecution:
    item_vectors: torch.Tensor
    lookup_metrics: tuple[IntegratedLookupMetrics, ...]
    makespan_seconds: float
    reconstruction_seconds: float

    def __post_init__(self) -> None:
        if (
            self.item_vectors.ndim != 2
            or not self.item_vectors.is_floating_point()
            or not self.lookup_metrics
            or not math.isfinite(self.makespan_seconds)
            or self.makespan_seconds < 0
            or not math.isfinite(self.reconstruction_seconds)
            or self.reconstruction_seconds < 0
            or self.reconstruction_seconds > self.makespan_seconds
        ):
            raise ValueError("D2 wave embedding execution is invalid")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def execute_d2_wave_embedding_lookup_plan(
    embedding: ModuloRowShardedEmbedding,
    plan: D2WaveEmbeddingLookupPlan,
) -> D2WaveEmbeddingExecution:
    device = embedding.local_weight.device
    if (
        plan.request.item_ids.device != device
        or plan.request.rank != embedding.rank
        or plan.request.world_size != embedding.world_size
    ):
        raise ValueError("D2 wave embedding plan and shard differ")
    metrics = []
    outputs = []
    _synchronize(device)
    started = time.perf_counter()
    for batch in plan.lookup_batches:
        lookup = fast_modulo_sharded_lookup(
            embedding,
            batch.reshape(1, -1),
            torch.tensor(
                [batch.numel()],
                dtype=torch.int64,
                device=device,
            ),
        )
        outputs.append(
            lookup.item_vectors.reshape(
                batch.numel(),
                embedding.hidden_size,
            )
        )
        metrics.append(lookup.metrics)
    _synchronize(device)
    reconstruction_started = time.perf_counter()
    if plan.mode == "wave_scope_unique_cache":
        if plan.inverse is None:
            raise RuntimeError("D2 wave embedding inverse is absent")
        item_vectors = outputs[0].index_select(0, plan.inverse)
    elif len(outputs) == 1:
        item_vectors = outputs[0]
    else:
        item_vectors = torch.cat(outputs)
    _synchronize(device)
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    makespan_seconds = time.perf_counter() - started
    if item_vectors.shape != (
        plan.request.logical_tokens,
        embedding.hidden_size,
    ):
        raise RuntimeError("D2 wave embedding output shape differs")
    return D2WaveEmbeddingExecution(
        item_vectors=item_vectors,
        lookup_metrics=tuple(metrics),
        makespan_seconds=makespan_seconds,
        reconstruction_seconds=reconstruction_seconds,
    )


def summarize_d2_wave_embedding_execution(
    plan: D2WaveEmbeddingLookupPlan,
    execution: D2WaveEmbeddingExecution,
    *,
    hidden_size: int,
) -> dict[str, object]:
    metrics = execution.lookup_metrics
    if (
        hidden_size < 1
        or len(metrics) != plan.lookup_calls
        or any(
            value.rank != plan.request.rank
            or value.world_size != plan.request.world_size
            for value in metrics
        )
    ):
        raise ValueError("D2 wave embedding execution summary differs")

    def total(name: str) -> int | float:
        return sum(getattr(value, name) for value in metrics)

    lookup_unique_tokens_sum = sum(
        torch.unique(batch).numel() for batch in plan.lookup_batches
    )
    lookup_remote_unique_tokens_sum = sum(
        torch.unique(
            batch[
                torch.remainder(batch, plan.request.world_size)
                != plan.request.rank
            ]
        ).numel()
        for batch in plan.lookup_batches
    )
    cache_vector_bytes = plan.cache_vector_bytes(hidden_size)
    return {
        "branch": plan.request.branch,
        "mode": plan.mode,
        "rank": plan.request.rank,
        "world_size": plan.request.world_size,
        "phase_token_counts": dict(
            plan.request.phase_token_counts
        ),
        "logical_tokens": plan.request.logical_tokens,
        "logical_unique_tokens": (
            plan.request.logical_unique_tokens
        ),
        "logical_remote_tokens": (
            plan.request.logical_remote_tokens
        ),
        "logical_remote_unique_tokens": (
            plan.request.logical_remote_unique_tokens
        ),
        "lookup_calls": plan.lookup_calls,
        "lookup_requested_tokens": total("requested_tokens"),
        "lookup_unique_tokens_sum": lookup_unique_tokens_sum,
        "lookup_remote_requested_tokens": total(
            "remote_requested_tokens"
        ),
        "lookup_remote_unique_tokens_sum": (
            lookup_remote_unique_tokens_sum
        ),
        "served_remote_requested_tokens": total(
            "served_remote_requested_tokens"
        ),
        "counts_collective_payload_bytes": sum(
            value.counts_collective_input_bytes
            + value.counts_collective_output_bytes
            for value in metrics
        ),
        "id_collective_payload_bytes": sum(
            value.id_collective_input_bytes
            + value.id_collective_output_bytes
            for value in metrics
        ),
        "vector_collective_payload_bytes": sum(
            value.vector_collective_input_bytes
            + value.vector_collective_output_bytes
            for value in metrics
        ),
        "collective_tensor_input_bytes": sum(
            value.counts_collective_input_bytes
            + value.id_collective_input_bytes
            + value.vector_collective_input_bytes
            for value in metrics
        ),
        "collective_tensor_output_bytes": sum(
            value.counts_collective_output_bytes
            + value.id_collective_output_bytes
            + value.vector_collective_output_bytes
            for value in metrics
        ),
        "actual_collective_tensor_payload_bytes": sum(
            value.actual_collective_tensor_payload_bytes
            for value in metrics
        ),
        "off_diagonal_send_bytes": sum(
            value.off_diagonal_send_bytes for value in metrics
        ),
        "off_diagonal_receive_bytes": sum(
            value.off_diagonal_receive_bytes for value in metrics
        ),
        "off_diagonal_bytes": sum(
            value.off_diagonal_bytes for value in metrics
        ),
        "collective_calls": total("collective_calls"),
        "off_diagonal_collective_calls": total("collective_calls"),
        "counts_collective_seconds": total(
            "counts_collective_seconds"
        ),
        "id_collective_seconds": total("id_collective_seconds"),
        "vector_collective_seconds": total(
            "vector_collective_seconds"
        ),
        "collective_seconds": sum(
            value.collective_seconds for value in metrics
        ),
        "off_diagonal_collective_seconds": sum(
            value.collective_seconds for value in metrics
        ),
        "makespan_seconds": execution.makespan_seconds,
        "reconstruction_seconds": (
            execution.reconstruction_seconds
        ),
        "cache_item_id_bytes": plan.cache_item_id_bytes,
        "cache_vector_bytes": cache_vector_bytes,
        "inverse_bytes": plan.inverse_bytes,
        "cache_and_inverse_bytes": (
            plan.cache_item_id_bytes
            + cache_vector_bytes
            + plan.inverse_bytes
        ),
        "lookup_kernel": D2_WAVE_EMBEDDING_LOOKUP_KERNEL,
        "timed_payload_hashing": False,
        "timing_cuda_synchronized": True,
    }
