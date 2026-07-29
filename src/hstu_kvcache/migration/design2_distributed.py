from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist

_T = TypeVar("_T")


@dataclass(frozen=True)
class D2DistributedRuntime:
    rank: int
    world_size: int
    local_rank: int
    backend: str
    init_method: str
    device: torch.device
    timeout_seconds: float
    owns_process_group: bool

    def __post_init__(self) -> None:
        if (
            self.rank < 0
            or self.world_size < 1
            or self.rank >= self.world_size
            or self.local_rank < 0
            or not self.backend
            or not self.init_method
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("D2 distributed runtime is invalid")

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def assert_active(self) -> None:
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() != self.rank
            or dist.get_world_size() != self.world_size
        ):
            raise RuntimeError("D2 process group is not active")


def _environment_int(name: str, fallback: int | None) -> int:
    value = os.environ.get(name)
    if value is not None:
        return int(value)
    if fallback is None:
        raise ValueError(f"D2 distributed runtime requires {name}")
    return fallback


def _resolve_runtime_inputs(
    backend: str | None,
    init_method: str | None,
    rank: int | None,
    world_size: int | None,
    local_rank: int | None,
    device: str | torch.device | None,
) -> tuple[str, str, int, int, int, torch.device]:
    env_launch = "RANK" in os.environ or "WORLD_SIZE" in os.environ
    if env_launch and (
        "RANK" not in os.environ or "WORLD_SIZE" not in os.environ
    ):
        raise ValueError("D2 torchrun environment is incomplete")
    resolved_world_size = _environment_int(
        "WORLD_SIZE",
        1 if world_size is None else world_size,
    )
    resolved_rank = _environment_int(
        "RANK",
        0 if rank is None else rank,
    )
    resolved_local_rank = _environment_int(
        "LOCAL_RANK",
        resolved_rank if local_rank is None else local_rank,
    )
    if world_size is not None and world_size != resolved_world_size:
        raise ValueError("D2 world size differs from the launcher")
    if rank is not None and rank != resolved_rank:
        raise ValueError("D2 rank differs from the launcher")
    if local_rank is not None and local_rank != resolved_local_rank:
        raise ValueError("D2 local rank differs from the launcher")
    resolved_init_method = init_method or ("env://" if env_launch else "")
    if not resolved_init_method:
        raise ValueError(
            "D2 world-size-1 initialization requires file:// or tcp://"
        )
    if not env_launch and resolved_world_size == 1 and not (
        resolved_init_method.startswith("file://")
        or resolved_init_method.startswith("tcp://")
    ):
        raise ValueError(
            "D2 explicit world-size-1 initialization must use file:// or tcp://"
        )
    if device is None:
        resolved_device = torch.device(
            "cuda",
            resolved_local_rank,
        ) if torch.cuda.is_available() else torch.device("cpu")
    else:
        resolved_device = torch.device(device)
    resolved_backend = backend or (
        "nccl" if resolved_device.type == "cuda" else "gloo"
    )
    if resolved_backend == "nccl" and resolved_device.type != "cuda":
        raise ValueError("D2 NCCL runtime requires a CUDA device")
    if not env_launch and resolved_world_size == 1 and resolved_backend != "gloo":
        raise ValueError("D2 explicit world-size-1 runtime must use Gloo")
    return (
        resolved_backend,
        resolved_init_method,
        resolved_rank,
        resolved_world_size,
        resolved_local_rank,
        resolved_device,
    )


def init_d2_distributed_runtime(
    *,
    backend: str | None = None,
    init_method: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    device: str | torch.device | None = None,
    timeout_seconds: float = 120.0,
) -> D2DistributedRuntime:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("D2 process-group timeout must be positive")
    (
        resolved_backend,
        resolved_init_method,
        resolved_rank,
        resolved_world_size,
        resolved_local_rank,
        resolved_device,
    ) = _resolve_runtime_inputs(
        backend,
        init_method,
        rank,
        world_size,
        local_rank,
        device,
    )
    if resolved_device.type == "cuda":
        torch.cuda.set_device(resolved_device)
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        try:
            dist.init_process_group(
                backend=resolved_backend,
                init_method=resolved_init_method,
                rank=resolved_rank,
                world_size=resolved_world_size,
                timeout=timedelta(seconds=timeout_seconds),
            )
        except Exception:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
            raise
    elif (
        dist.get_rank() != resolved_rank
        or dist.get_world_size() != resolved_world_size
        or str(dist.get_backend()) != resolved_backend
    ):
        raise RuntimeError("active process group differs from D2 runtime")
    runtime = D2DistributedRuntime(
        rank=resolved_rank,
        world_size=resolved_world_size,
        local_rank=resolved_local_rank,
        backend=resolved_backend,
        init_method=resolved_init_method,
        device=resolved_device,
        timeout_seconds=float(timeout_seconds),
        owns_process_group=owns_process_group,
    )
    runtime.assert_active()
    return runtime


def close_d2_distributed_runtime(
    runtime: D2DistributedRuntime,
) -> bool:
    if not runtime.owns_process_group or not dist.is_initialized():
        return False
    try:
        dist.destroy_process_group()
    except Exception:
        return False
    return True


@contextmanager
def d2_distributed_runtime(
    **kwargs: object,
):
    runtime = init_d2_distributed_runtime(**kwargs)
    try:
        yield runtime
    finally:
        close_d2_distributed_runtime(runtime)


@dataclass(frozen=True, order=True)
class D2CollectiveStep:
    ordinal: int
    phase: str

    def __post_init__(self) -> None:
        if self.ordinal < 0 or not self.phase:
            raise ValueError("D2 collective step is invalid")

    @property
    def token(self) -> str:
        return f"{self.ordinal:04d}:{self.phase}"


class D2CollectiveGuard:
    def __init__(
        self,
        runtime: D2DistributedRuntime,
        phase_order: Sequence[str],
    ) -> None:
        if not phase_order or any(not value for value in phase_order):
            raise ValueError("D2 collective phase order is invalid")
        runtime.assert_active()
        self._runtime = runtime
        self._phase_order = tuple(phase_order)
        self._trace: list[D2CollectiveStep] = []

    @property
    def trace(self) -> tuple[D2CollectiveStep, ...]:
        return tuple(self._trace)

    @property
    def complete(self) -> bool:
        return len(self._trace) == len(self._phase_order)

    def enter(
        self,
        phase: str,
        ordinal: int | None = None,
    ) -> D2CollectiveStep:
        self._runtime.assert_active()
        expected_ordinal = len(self._trace)
        supplied_ordinal = (
            expected_ordinal if ordinal is None else ordinal
        )
        local_error = None
        if expected_ordinal >= len(self._phase_order):
            local_error = "collective trace exceeds the frozen phase order"
        elif supplied_ordinal != expected_ordinal:
            local_error = "collective ordinal differs from local trace"
        elif phase != self._phase_order[expected_ordinal]:
            local_error = "collective phase differs from local frozen order"
        signature = (
            self._runtime.rank,
            supplied_ordinal,
            phase,
            local_error,
        )
        gathered: list[object] = [None] * self._runtime.world_size
        dist.all_gather_object(gathered, signature)
        expected_signatures = [
            (rank, supplied_ordinal, phase, None)
            for rank in range(self._runtime.world_size)
        ]
        if gathered != expected_signatures:
            raise RuntimeError(
                f"D2 collective order mismatch: {gathered}"
            )
        step = D2CollectiveStep(supplied_ordinal, phase)
        self._trace.append(step)
        return step

    def require_complete(self) -> tuple[D2CollectiveStep, ...]:
        if not self.complete:
            raise RuntimeError("D2 collective trace is incomplete")
        return self.trace


@dataclass(frozen=True)
class D2PreflightVote:
    rank: int
    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.rank < 0
            or self.passed == bool(self.failure_reasons)
            or any(not value for value in self.failure_reasons)
        ):
            raise ValueError("D2 preflight vote is invalid")


@dataclass(frozen=True)
class D2PreflightDecision:
    passed: bool
    votes: tuple[D2PreflightVote, ...]

    def __post_init__(self) -> None:
        if (
            not self.votes
            or tuple(value.rank for value in self.votes)
            != tuple(range(len(self.votes)))
            or self.passed != all(value.passed for value in self.votes)
        ):
            raise ValueError("D2 preflight decision is invalid")

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return tuple(
            f"rank {vote.rank}: {reason}"
            for vote in self.votes
            for reason in vote.failure_reasons
        )


def capture_d2_preflight_failures(
    checks: Mapping[str, Callable[[], bool | None]],
) -> tuple[str, ...]:
    failures = []
    for name, check in checks.items():
        if not name:
            raise ValueError("D2 preflight check name is empty")
        try:
            result = check()
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
        else:
            if result is False:
                failures.append(f"{name}: returned false")
    return tuple(failures)


def vote_d2_preflight(
    runtime: D2DistributedRuntime,
    failure_reasons: Sequence[str],
    *,
    guard: D2CollectiveGuard | None = None,
    phase: str = "preflight_vote",
) -> D2PreflightDecision:
    runtime.assert_active()
    failures = tuple(str(value) for value in failure_reasons)
    if any(not value for value in failures):
        raise ValueError("D2 preflight failure reason is empty")
    if guard is not None:
        guard.enter(phase)
    local_vote = D2PreflightVote(
        rank=runtime.rank,
        passed=not failures,
        failure_reasons=failures,
    )
    gathered: list[object] = [None] * runtime.world_size
    dist.all_gather_object(gathered, local_vote)
    if not all(
        isinstance(value, D2PreflightVote)
        for value in gathered
    ):
        raise RuntimeError("D2 preflight gathered invalid votes")
    votes = tuple(gathered)
    if tuple(value.rank for value in votes) != tuple(
        range(runtime.world_size)
    ):
        raise RuntimeError("D2 preflight gathered duplicate ranks")
    return D2PreflightDecision(
        passed=all(value.passed for value in votes),
        votes=votes,
    )


def gather_d2_rank_metadata(
    runtime: D2DistributedRuntime,
    local_metadata: _T,
    *,
    destination_rank: int = 0,
    guard: D2CollectiveGuard | None = None,
    phase: str = "rank_metadata_gather",
) -> tuple[_T, ...] | None:
    runtime.assert_active()
    if not 0 <= destination_rank < runtime.world_size:
        raise ValueError("D2 metadata destination rank is invalid")
    if guard is not None:
        guard.enter(phase)
    gathered: list[object] = [None] * runtime.world_size
    dist.all_gather_object(gathered, local_metadata)
    if runtime.rank != destination_rank:
        return None
    return tuple(gathered)


def broadcast_d2_metadata(
    runtime: D2DistributedRuntime,
    metadata: _T | None,
    *,
    source_rank: int = 0,
    guard: D2CollectiveGuard | None = None,
    phase: str = "metadata_broadcast",
) -> _T:
    runtime.assert_active()
    if not 0 <= source_rank < runtime.world_size:
        raise ValueError("D2 metadata source rank is invalid")
    if runtime.rank == source_rank and metadata is None:
        raise ValueError("D2 metadata source cannot broadcast None")
    if runtime.rank != source_rank and metadata is not None:
        raise ValueError("D2 metadata non-source must pass None")
    if guard is not None:
        guard.enter(phase)
    values = [metadata]
    kwargs = (
        {"device": runtime.device}
        if runtime.backend == "nccl"
        else {}
    )
    dist.broadcast_object_list(values, src=source_rank, **kwargs)
    if values[0] is None:
        raise RuntimeError("D2 metadata broadcast returned None")
    return values[0]


def d2_file_init_method(path: str | Path) -> str:
    return Path(path).resolve().as_uri()
