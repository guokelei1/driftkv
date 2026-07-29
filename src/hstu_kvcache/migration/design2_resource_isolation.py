from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from .design2_embedding import modulo_embedding_local_rows
from .design2_embedding_capsule import (
    D2MaterializedEmbeddingCapsuleRankPlan,
)

D2_RESOURCE_ISOLATION_PROTOCOL = (
    "cohortkv_d2_embedding_resource_isolation_development_v1"
)


class D2CollectiveLaunchCoordinator:
    def __init__(
        self,
        order: tuple[tuple[str, int], ...],
    ) -> None:
        if (
            not order
            or any(not name or ordinal < 0 for name, ordinal in order)
            or len(set(order)) != len(order)
        ):
            raise ValueError("D2 collective launch order is invalid")
        self._order = order
        self._position = 0
        self._condition = threading.Condition()

    @property
    def order(self) -> tuple[tuple[str, int], ...]:
        return self._order

    @contextmanager
    def phase(self, name: str, ordinal: int):
        key = (name, ordinal)
        with self._condition:
            while (
                self._position < len(self._order)
                and self._order[self._position] != key
            ):
                if key not in self._order[self._position :]:
                    raise RuntimeError(
                        "D2 collective launch phase is unexpected"
                    )
                self._condition.wait()
            if self._position >= len(self._order):
                raise RuntimeError(
                    "D2 collective launch order is exhausted"
                )
            yield
            self._position += 1
            self._condition.notify_all()

    def assert_complete(self) -> None:
        with self._condition:
            if self._position != len(self._order):
                raise RuntimeError(
                    "D2 collective launch order is incomplete"
                )


@dataclass(frozen=True)
class D2FixedRateSchedule:
    requested_rate_per_second: float
    duration_seconds: float
    release_offsets_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.requested_rate_per_second)
            or self.requested_rate_per_second <= 0
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
            or not self.release_offsets_seconds
            or any(
                not math.isfinite(value)
                or value < 0
                or value >= self.duration_seconds
                for value in self.release_offsets_seconds
            )
            or any(
                right <= left
                for left, right in zip(
                    self.release_offsets_seconds,
                    self.release_offsets_seconds[1:],
                    strict=False,
                )
            )
        ):
            raise ValueError("D2 fixed-rate schedule is invalid")

    @property
    def request_count(self) -> int:
        return len(self.release_offsets_seconds)

    @property
    def actual_offered_rate_per_second(self) -> float:
        return self.request_count / self.duration_seconds

    @property
    def period_seconds(self) -> float:
        return 1.0 / self.requested_rate_per_second


def build_d2_fixed_rate_schedule(
    requested_rate_per_second: float,
    duration_seconds: float,
) -> D2FixedRateSchedule:
    if (
        not math.isfinite(requested_rate_per_second)
        or requested_rate_per_second <= 0
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        raise ValueError("D2 fixed-rate schedule input is invalid")
    count = math.floor(
        requested_rate_per_second * duration_seconds
    )
    if count < 1:
        raise ValueError("D2 fixed-rate schedule has no request")
    return D2FixedRateSchedule(
        requested_rate_per_second=float(requested_rate_per_second),
        duration_seconds=float(duration_seconds),
        release_offsets_seconds=tuple(
            index / requested_rate_per_second
            for index in range(count)
        ),
    )


@dataclass(frozen=True)
class D2ForegroundSample:
    sequence: int
    release_offset_seconds: float
    issue_offset_seconds: float
    completion_offset_seconds: float
    execution_wall_seconds: float
    execution_device_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.release_offset_seconds,
            self.issue_offset_seconds,
            self.completion_offset_seconds,
            self.execution_wall_seconds,
            self.execution_device_seconds,
        )
        if (
            self.sequence < 0
            or any(not math.isfinite(value) or value < 0 for value in values)
            or self.issue_offset_seconds < self.release_offset_seconds
            or self.completion_offset_seconds < self.issue_offset_seconds
            or self.execution_wall_seconds
            > self.completion_offset_seconds
            - self.issue_offset_seconds
            + 1e-6
            or self.execution_device_seconds
            > self.execution_wall_seconds + 1e-3
        ):
            raise ValueError("D2 foreground sample is invalid")

    @property
    def queue_seconds(self) -> float:
        return self.issue_offset_seconds - self.release_offset_seconds

    @property
    def response_seconds(self) -> float:
        return (
            self.completion_offset_seconds
            - self.release_offset_seconds
        )

    def to_dict(self) -> dict[str, int | float]:
        output = asdict(self)
        output["queue_seconds"] = self.queue_seconds
        output["response_seconds"] = self.response_seconds
        return output


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("D2 quantile probability is invalid")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def summarize_d2_foreground_samples(
    samples: tuple[D2ForegroundSample, ...],
    schedule: D2FixedRateSchedule,
    *,
    deadline_seconds: float,
    window_start_seconds: float = 0.0,
    window_end_seconds: float | None = None,
) -> dict[str, object]:
    end = (
        schedule.duration_seconds
        if window_end_seconds is None
        else window_end_seconds
    )
    if (
        not math.isfinite(deadline_seconds)
        or deadline_seconds <= 0
        or not math.isfinite(window_start_seconds)
        or not math.isfinite(end)
        or window_start_seconds < 0
        or end <= window_start_seconds
        or end > schedule.duration_seconds
        or tuple(value.sequence for value in samples)
        != tuple(range(len(samples)))
        or len(samples) != schedule.request_count
        or any(
            not math.isclose(
                value.release_offset_seconds,
                schedule.release_offsets_seconds[value.sequence],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for value in samples
        )
    ):
        raise ValueError("D2 foreground summary input is invalid")
    selected = tuple(
        value
        for value in samples
        if window_start_seconds
        <= value.release_offset_seconds
        < end
    )
    response = [value.response_seconds for value in selected]
    queue = [value.queue_seconds for value in selected]
    wall = [value.execution_wall_seconds for value in selected]
    device = [value.execution_device_seconds for value in selected]
    completion_horizon = max(
        end,
        max(
            (
                value.completion_offset_seconds
                for value in selected
            ),
            default=end,
        ),
    )
    released_before_issue = [
        min(
            schedule.request_count,
            math.floor(
                value.issue_offset_seconds
                * schedule.requested_rate_per_second
                + 1e-12
            )
            + 1,
        )
        for value in selected
    ]
    queue_depth = [
        max(0, released - value.sequence - 1)
        for released, value in zip(
            released_before_issue,
            selected,
            strict=True,
        )
    ]
    has_completed = bool(selected)
    deadline_miss_count = (
        sum(value > deadline_seconds for value in response)
        if has_completed
        else None
    )
    return {
        "observation_status": (
            "complete"
            if has_completed
            else "no_scheduled_requests_in_window"
        ),
        "has_completed_requests": has_completed,
        "no_completed_requests": not has_completed,
        "queue_observation_status": (
            "observed" if has_completed else "no_completed_requests"
        ),
        "deadline_observation_status": (
            "observed" if has_completed else "no_completed_requests"
        ),
        "window_start_seconds": window_start_seconds,
        "window_end_seconds": end,
        "window_duration_seconds": end - window_start_seconds,
        "scheduled_requests": len(selected),
        "completed_requests": len(selected),
        "requested_offered_rate_per_second": (
            schedule.requested_rate_per_second
        ),
        "actual_offered_rate_per_second": (
            len(selected) / (end - window_start_seconds)
        ),
        "achieved_rate_per_second": (
            len(selected)
            / (completion_horizon - window_start_seconds)
            if has_completed
            else None
        ),
        "deadline_seconds": deadline_seconds,
        "deadline_miss_count": deadline_miss_count,
        "deadline_miss_fraction": (
            deadline_miss_count / len(selected)
            if has_completed and deadline_miss_count is not None
            else None
        ),
        "positive_queue_count": (
            sum(value > 0 for value in queue)
            if has_completed
            else None
        ),
        "behind_by_one_period_count": (
            sum(value >= schedule.period_seconds for value in queue)
            if has_completed
            else None
        ),
        "estimated_max_queue_depth_requests": (
            max(queue_depth)
            if has_completed
            else None
        ),
        "queue_p50_seconds": _quantile(queue, 0.50),
        "queue_p99_seconds": _quantile(queue, 0.99),
        "response_p50_seconds": _quantile(response, 0.50),
        "response_p99_seconds": _quantile(response, 0.99),
        "response_max_seconds": max(response, default=None),
        "execution_wall_p50_seconds": _quantile(wall, 0.50),
        "execution_wall_p99_seconds": _quantile(wall, 0.99),
        "execution_device_p50_seconds": _quantile(device, 0.50),
        "execution_device_p99_seconds": _quantile(device, 0.99),
        "last_completion_offset_seconds": max(
            (
                value.completion_offset_seconds
                for value in selected
            ),
            default=None,
        ),
    }


def build_d2_synthetic_foreground_request_ring(
    *,
    num_embeddings: int,
    world_size: int,
    batch_tokens_per_rank: int,
    ring_size: int,
    seed: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if (
        num_embeddings < world_size
        or world_size < 1
        or batch_tokens_per_rank < 1
        or ring_size < 1
        or seed < 0
    ):
        raise ValueError("D2 synthetic foreground ring input is invalid")
    owner_rows = tuple(
        modulo_embedding_local_rows(
            num_embeddings,
            owner,
            world_size,
        )
        for owner in range(world_size)
    )
    ring = []
    for slot in range(ring_size):
        requesters = []
        for requester in range(world_size):
            ids = []
            for index in range(batch_tokens_per_rank):
                owner = (requester + 1 + index) % world_size
                row = (
                    seed
                    + slot * batch_tokens_per_rank
                    + requester * 104729
                    + index // world_size
                ) % owner_rows[owner]
                ids.append(owner + row * world_size)
            requesters.append(tuple(ids))
        ring.append(tuple(requesters))
    return tuple(ring)


@dataclass
class D2VectorExchangeWorkspace:
    materialized: D2MaterializedEmbeddingCapsuleRankPlan
    reconstruct_requested: bool
    unique_vectors: torch.Tensor
    response_vectors: torch.Tensor
    received_vectors: torch.Tensor
    requested_vectors: torch.Tensor | None

    @property
    def device(self) -> torch.device:
        return self.unique_vectors.device


def build_d2_vector_exchange_workspace(
    materialized: D2MaterializedEmbeddingCapsuleRankPlan,
    local_weight: torch.Tensor,
    *,
    reconstruct_requested: bool,
) -> D2VectorExchangeWorkspace:
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
        raise ValueError("D2 isolation local embedding shard differs")
    shape = (materialized.unique_tokens, local_weight.shape[1])
    requested_vectors = (
        torch.empty(
            (materialized.requested_tokens, local_weight.shape[1]),
            dtype=local_weight.dtype,
            device=local_weight.device,
        )
        if reconstruct_requested
        else None
    )
    return D2VectorExchangeWorkspace(
        materialized=materialized,
        reconstruct_requested=reconstruct_requested,
        unique_vectors=torch.empty(
            shape,
            dtype=local_weight.dtype,
            device=local_weight.device,
        ),
        response_vectors=torch.empty(
            (
                materialized.served_remote_unique_tokens,
                local_weight.shape[1],
            ),
            dtype=local_weight.dtype,
            device=local_weight.device,
        ),
        received_vectors=torch.empty(
            (
                materialized.remote_unique_tokens,
                local_weight.shape[1],
            ),
            dtype=local_weight.dtype,
            device=local_weight.device,
        ),
        requested_vectors=requested_vectors,
    )


@dataclass(frozen=True)
class D2VectorExchangeSample:
    rank: int
    world_size: int
    requested_tokens: int
    unique_tokens: int
    local_unique_tokens: int
    remote_unique_tokens: int
    served_remote_unique_tokens: int
    vector_send_bytes: int
    vector_receive_bytes: int
    collective_calls: int
    reconstruct_requested: bool
    wall_seconds: float
    device_seconds: float
    collective_device_seconds: float

    def __post_init__(self) -> None:
        integers = (
            self.rank,
            self.world_size,
            self.requested_tokens,
            self.unique_tokens,
            self.local_unique_tokens,
            self.remote_unique_tokens,
            self.served_remote_unique_tokens,
            self.vector_send_bytes,
            self.vector_receive_bytes,
            self.collective_calls,
        )
        seconds = (
            self.wall_seconds,
            self.device_seconds,
            self.collective_device_seconds,
        )
        if (
            self.world_size < 1
            or not 0 <= self.rank < self.world_size
            or any(value < 0 for value in integers)
            or any(not math.isfinite(value) or value < 0 for value in seconds)
            or self.local_unique_tokens + self.remote_unique_tokens
            != self.unique_tokens
            or self.collective_calls
            != (0 if self.world_size == 1 else 1)
            or self.device_seconds + 1e-3
            < self.collective_device_seconds
            or self.wall_seconds + 1e-3 < self.device_seconds
        ):
            raise ValueError("D2 vector exchange sample is invalid")

    @property
    def vector_endpoint_bytes(self) -> int:
        return self.vector_send_bytes + self.vector_receive_bytes

    def to_dict(self) -> dict[str, int | float | bool]:
        output = asdict(self)
        output["vector_endpoint_bytes"] = self.vector_endpoint_bytes
        return output


def _validate_exchange_group(
    materialized: D2MaterializedEmbeddingCapsuleRankPlan,
    process_group: dist.ProcessGroup | None,
) -> None:
    if materialized.world_size == 1:
        if dist.is_initialized() and (
            dist.get_world_size(group=process_group) != 1
            or dist.get_rank(group=process_group) != materialized.rank
        ):
            raise ValueError("D2 isolation process group differs")
        return
    if (
        process_group is None
        or not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size(group=process_group)
        != materialized.world_size
        or dist.get_rank(group=process_group) != materialized.rank
    ):
        raise RuntimeError("D2 isolation process group is invalid")


@torch.inference_mode()
def execute_d2_vector_exchange(
    workspace: D2VectorExchangeWorkspace,
    local_weight: torch.Tensor,
    *,
    process_group: dist.ProcessGroup | None,
    stream: torch.cuda.Stream | None = None,
    collective_launch_guard: (
        Callable[[], AbstractContextManager[object]] | None
    ) = None,
) -> D2VectorExchangeSample:
    materialized = workspace.materialized
    _validate_exchange_group(materialized, process_group)
    if (
        local_weight.device != workspace.device
        or local_weight.ndim != 2
        or local_weight.shape[1] != workspace.unique_vectors.shape[1]
        or (local_weight.device.type == "cuda") != (stream is not None)
    ):
        raise ValueError("D2 isolation exchange execution differs")
    element_bytes = local_weight.element_size()
    wall_started = time.perf_counter()
    if stream is None:
        if materialized.local_unique_tokens:
            workspace.unique_vectors.index_copy_(
                0,
                materialized.local_capsule_slots,
                local_weight.index_select(
                    0,
                    materialized.local_rows,
                ),
            )
        if materialized.served_remote_unique_tokens:
            torch.index_select(
                local_weight,
                0,
                materialized.send_local_rows,
                out=workspace.response_vectors,
            )
        collective_started = time.perf_counter()
        collective_seconds = (
            0.0
        )
        if materialized.world_size > 1:
            guard = (
                nullcontext()
                if collective_launch_guard is None
                else collective_launch_guard()
            )
            with guard:
                dist.all_to_all_single(
                    workspace.received_vectors,
                    workspace.response_vectors,
                    output_split_sizes=list(
                        materialized.receive_splits
                    ),
                    input_split_sizes=list(
                        materialized.send_splits
                    ),
                    group=process_group,
                )
            collective_seconds = (
                time.perf_counter() - collective_started
            )
        if materialized.remote_unique_tokens:
            workspace.unique_vectors.index_copy_(
                0,
                materialized.receive_capsule_slots,
                workspace.received_vectors,
            )
        if workspace.requested_vectors is not None:
            torch.index_select(
                workspace.unique_vectors,
                0,
                materialized.inverse_slots,
                out=workspace.requested_vectors,
            )
        completed = time.perf_counter()
        device_seconds = completed - wall_started
    else:
        start_event = torch.cuda.Event(enable_timing=True)
        collective_start = torch.cuda.Event(enable_timing=True)
        collective_end = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            if materialized.local_unique_tokens:
                workspace.unique_vectors.index_copy_(
                    0,
                    materialized.local_capsule_slots,
                    local_weight.index_select(
                        0,
                        materialized.local_rows,
                    ),
                )
            if materialized.served_remote_unique_tokens:
                torch.index_select(
                    local_weight,
                    0,
                    materialized.send_local_rows,
                    out=workspace.response_vectors,
                )
            collective_start.record(stream)
            if materialized.world_size > 1:
                guard = (
                    nullcontext()
                    if collective_launch_guard is None
                    else collective_launch_guard()
                )
                with guard:
                    dist.all_to_all_single(
                        workspace.received_vectors,
                        workspace.response_vectors,
                        output_split_sizes=list(
                            materialized.receive_splits
                        ),
                        input_split_sizes=list(
                            materialized.send_splits
                        ),
                        group=process_group,
                    )
            collective_end.record(stream)
            if materialized.remote_unique_tokens:
                workspace.unique_vectors.index_copy_(
                    0,
                    materialized.receive_capsule_slots,
                    workspace.received_vectors,
                )
            if workspace.requested_vectors is not None:
                torch.index_select(
                    workspace.unique_vectors,
                    0,
                    materialized.inverse_slots,
                    out=workspace.requested_vectors,
                )
            end_event.record(stream)
        end_event.synchronize()
        device_seconds = start_event.elapsed_time(end_event) / 1000.0
        collective_seconds = (
            collective_start.elapsed_time(collective_end) / 1000.0
            if materialized.world_size > 1
            else 0.0
        )
    wall_seconds = time.perf_counter() - wall_started
    return D2VectorExchangeSample(
        rank=materialized.rank,
        world_size=materialized.world_size,
        requested_tokens=materialized.requested_tokens,
        unique_tokens=materialized.unique_tokens,
        local_unique_tokens=materialized.local_unique_tokens,
        remote_unique_tokens=materialized.remote_unique_tokens,
        served_remote_unique_tokens=(
            materialized.served_remote_unique_tokens
        ),
        vector_send_bytes=(
            materialized.served_remote_unique_tokens
            * local_weight.shape[1]
            * element_bytes
        ),
        vector_receive_bytes=(
            materialized.remote_unique_tokens
            * local_weight.shape[1]
            * element_bytes
        ),
        collective_calls=(
            0 if materialized.world_size == 1 else 1
        ),
        reconstruct_requested=workspace.reconstruct_requested,
        wall_seconds=wall_seconds,
        device_seconds=device_seconds,
        collective_device_seconds=collective_seconds,
    )
