from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from ..models import HSTUKVCache
from .capsule import MigratedKVBatch, MigrationCapsuleBatch
from .operator import MigrationOperator, ReferenceMigrationOperator
from .program import MigrationProgram


@dataclass(frozen=True)
class CohortExecutionMetrics:
    batch_count: int
    record_count: int
    token_count: int
    input_bytes: int
    output_bytes: int
    auto_pinned_batches: int
    elapsed_seconds: float

    @property
    def records_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.record_count / self.elapsed_seconds

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.token_count / self.elapsed_seconds

    @property
    def gib_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return (self.input_bytes + self.output_bytes) / (2**30 * self.elapsed_seconds)


@dataclass(frozen=True)
class CohortExecutionReport:
    batches: tuple[MigratedKVBatch, ...]
    metrics: CohortExecutionMetrics


@dataclass
class _PendingCudaBatch:
    done: torch.cuda.Event
    host_capsule: MigrationCapsuleBatch
    device_capsule: MigrationCapsuleBatch
    device_result: MigratedKVBatch
    host_result: MigratedKVBatch


class CohortStreamingExecutor:
    def __init__(
        self,
        program: MigrationProgram,
        device: torch.device | str,
        max_inflight_batches: int = 3,
        pin_inputs: bool = True,
        operator: MigrationOperator | None = None,
    ) -> None:
        device = torch.device(device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("executor device must be CPU or CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA executor requested but CUDA is unavailable")
        if max_inflight_batches < 1:
            raise ValueError("max_inflight_batches must be positive")
        self.device = device
        self.max_inflight_batches = max_inflight_batches
        self.pin_inputs = pin_inputs
        self.operator = operator or ReferenceMigrationOperator()
        self.program = self.operator.prepare_program(program, device)
        self._h2d_stream = None
        self._compute_stream = None
        self._d2h_stream = None
        if device.type == "cuda":
            with torch.cuda.device(device):
                self._h2d_stream = torch.cuda.Stream(device=device)
                self._compute_stream = torch.cuda.Stream(device=device)
                self._d2h_stream = torch.cuda.Stream(device=device)

    @torch.no_grad()
    def run(
        self,
        batches: Iterable[MigrationCapsuleBatch],
    ) -> CohortExecutionReport:
        if self.device.type == "cuda":
            return self._run_cuda(batches)
        return self._run_cpu(batches)

    def _account_capsule(
        self,
        capsule: MigrationCapsuleBatch,
        seen_record_ids: set[int],
    ) -> tuple[int, int]:
        overlap = seen_record_ids.intersection(capsule.record_ids)
        if overlap:
            raise ValueError("record_ids must be unique across a cohort execution")
        seen_record_ids.update(capsule.record_ids)
        return capsule.batch_size, int(capsule.lengths.sum().item())

    @torch.no_grad()
    def _run_cpu(
        self,
        batches: Iterable[MigrationCapsuleBatch],
    ) -> CohortExecutionReport:
        started = time.perf_counter()
        results = []
        seen_record_ids: set[int] = set()
        batch_count = 0
        record_count = 0
        token_count = 0
        input_bytes = 0
        output_bytes = 0
        for capsule in batches:
            records, tokens = self._account_capsule(capsule, seen_record_ids)
            device_capsule = capsule.to(self.device)
            result = self.operator.execute(self.program, device_capsule)
            results.append(result)
            batch_count += 1
            record_count += records
            token_count += tokens
            input_bytes += capsule.nbytes
            output_bytes += result.nbytes
        elapsed = time.perf_counter() - started
        return CohortExecutionReport(
            batches=tuple(results),
            metrics=CohortExecutionMetrics(
                batch_count=batch_count,
                record_count=record_count,
                token_count=token_count,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                auto_pinned_batches=0,
                elapsed_seconds=elapsed,
            ),
        )

    @torch.no_grad()
    def _run_cuda(
        self,
        batches: Iterable[MigrationCapsuleBatch],
    ) -> CohortExecutionReport:
        started = time.perf_counter()
        results = []
        pending: deque[_PendingCudaBatch] = deque()
        seen_record_ids: set[int] = set()
        batch_count = 0
        record_count = 0
        token_count = 0
        input_bytes = 0
        output_bytes = 0
        auto_pinned_batches = 0

        with torch.cuda.device(self.device):
            if (
                self._h2d_stream is None
                or self._compute_stream is None
                or self._d2h_stream is None
            ):
                raise RuntimeError("CUDA streams are not initialized")
            h2d_stream = self._h2d_stream
            compute_stream = self._compute_stream
            d2h_stream = self._d2h_stream

            for capsule in batches:
                if capsule.device.type != "cpu":
                    raise ValueError("CUDA streaming executor requires CPU-resident capsules")
                records, tokens = self._account_capsule(capsule, seen_record_ids)
                host_capsule = capsule
                if self.pin_inputs and not capsule.is_pinned:
                    host_capsule = capsule.pin_memory()
                    auto_pinned_batches += 1

                h2d_done = torch.cuda.Event()
                with torch.cuda.stream(h2d_stream):
                    device_capsule = host_capsule.to(self.device, non_blocking=True)
                    h2d_done.record(h2d_stream)

                compute_done = torch.cuda.Event()
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(h2d_done)
                    device_result = self.operator.execute(
                        self.program,
                        device_capsule,
                    )
                    compute_done.record(compute_stream)

                host_k = torch.empty(
                    device_result.cache.k.shape,
                    dtype=device_result.cache.k.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host_v = torch.empty(
                    device_result.cache.v.shape,
                    dtype=device_result.cache.v.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host_result = MigratedKVBatch(
                    record_ids=host_capsule.record_ids,
                    migration_anchor_version=host_capsule.migration_anchor_version,
                    served_kv_target=self.program.target_version,
                    cache=HSTUKVCache(
                        k=host_k,
                        v=host_v,
                        seq_len=device_result.cache.seq_len,
                    ),
                    lengths=host_capsule.lengths,
                )
                d2h_done = torch.cuda.Event()
                with torch.cuda.stream(d2h_stream):
                    d2h_stream.wait_event(compute_done)
                    host_k.copy_(device_result.cache.k, non_blocking=True)
                    host_v.copy_(device_result.cache.v, non_blocking=True)
                    d2h_done.record(d2h_stream)

                pending.append(
                    _PendingCudaBatch(
                        done=d2h_done,
                        host_capsule=host_capsule,
                        device_capsule=device_capsule,
                        device_result=device_result,
                        host_result=host_result,
                    )
                )
                batch_count += 1
                record_count += records
                token_count += tokens
                input_bytes += capsule.nbytes
                output_bytes += host_result.nbytes

                if len(pending) >= self.max_inflight_batches:
                    completed = pending.popleft()
                    completed.done.synchronize()
                    results.append(completed.host_result)

            while pending:
                completed = pending.popleft()
                completed.done.synchronize()
                results.append(completed.host_result)

        elapsed = time.perf_counter() - started
        return CohortExecutionReport(
            batches=tuple(results),
            metrics=CohortExecutionMetrics(
                batch_count=batch_count,
                record_count=record_count,
                token_count=token_count,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                auto_pinned_batches=auto_pinned_batches,
                elapsed_seconds=elapsed,
            ),
        )
