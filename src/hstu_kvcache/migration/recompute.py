from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from ..models import HSTU, HSTUKVCache
from .capsule import MigratedKVBatch, PinnedKVOutputPool
from .executor import CohortExecutionMetrics, CohortExecutionReport


@dataclass(frozen=True)
class RawHistoryBatch:
    record_ids: tuple[int, ...]
    migration_anchor_version: str
    item_ids: torch.Tensor
    behaviors: torch.Tensor
    time_deltas: torch.Tensor
    lengths: torch.Tensor

    def __post_init__(self) -> None:
        if not self.migration_anchor_version:
            raise ValueError("migration_anchor_version must be nonempty")
        if self.item_ids.ndim != 2:
            raise ValueError("item_ids must have shape [batch, sequence]")
        if self.behaviors.shape != self.item_ids.shape:
            raise ValueError("behaviors and item_ids shapes differ")
        if self.time_deltas.shape != self.item_ids.shape:
            raise ValueError("time_deltas and item_ids shapes differ")
        if self.lengths.shape != (self.item_ids.shape[0],):
            raise ValueError("lengths and history batch dimension differ")
        if len(self.record_ids) != self.item_ids.shape[0]:
            raise ValueError("record_ids and history batch dimension differ")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("record_ids must be unique within a batch")
        devices = {
            self.item_ids.device,
            self.behaviors.device,
            self.time_deltas.device,
            self.lengths.device,
        }
        if len(devices) != 1:
            raise ValueError("raw history tensors must share one device")
        if self.item_ids.device.type == "cpu":
            if bool(torch.any(self.lengths < 0)) or bool(
                torch.any(self.lengths > self.seq_len)
            ):
                raise ValueError("lengths must be within the padded sequence width")

    @property
    def device(self) -> torch.device:
        return self.item_ids.device

    @property
    def batch_size(self) -> int:
        return self.item_ids.shape[0]

    @property
    def seq_len(self) -> int:
        return self.item_ids.shape[1]

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.item_ids,
                self.behaviors,
                self.time_deltas,
                self.lengths,
            )
        )

    @property
    def is_pinned(self) -> bool:
        return self.device.type == "cpu" and all(
            tensor.is_pinned()
            for tensor in (
                self.item_ids,
                self.behaviors,
                self.time_deltas,
                self.lengths,
            )
        )

    def pin_memory(self) -> RawHistoryBatch:
        if self.device.type != "cpu":
            raise ValueError("only CPU histories can be pinned")
        if self.is_pinned:
            return self
        return RawHistoryBatch(
            record_ids=self.record_ids,
            migration_anchor_version=self.migration_anchor_version,
            item_ids=self.item_ids.pin_memory(),
            behaviors=self.behaviors.pin_memory(),
            time_deltas=self.time_deltas.pin_memory(),
            lengths=self.lengths.pin_memory(),
        )

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> RawHistoryBatch:
        return RawHistoryBatch(
            record_ids=self.record_ids,
            migration_anchor_version=self.migration_anchor_version,
            item_ids=self.item_ids.to(device, non_blocking=non_blocking),
            behaviors=self.behaviors.to(device, non_blocking=non_blocking),
            time_deltas=self.time_deltas.to(device, non_blocking=non_blocking),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
        )


@dataclass
class _PendingRecomputeBatch:
    done: torch.cuda.Event
    host_batch: RawHistoryBatch
    device_batch: RawHistoryBatch
    device_result: MigratedKVBatch
    host_result: MigratedKVBatch


class FullRecomputeStreamingExecutor:
    def __init__(
        self,
        model: HSTU,
        source_version: str | Sequence[str],
        target_version: str,
        device: torch.device | str,
        max_inflight_batches: int = 3,
        pin_inputs: bool = True,
        execution_dtype: torch.dtype | None = torch.float16,
        publication_dtype: torch.dtype = torch.float16,
        output_pool: PinnedKVOutputPool | None = None,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("full recompute executor requires a CUDA device")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        source_versions = (
            (source_version,)
            if isinstance(source_version, str)
            else tuple(source_version)
        )
        if (
            not source_versions
            or any(not value for value in source_versions)
            or not target_version
        ):
            raise ValueError("source and target versions must be nonempty")
        if len(set(source_versions)) != len(source_versions):
            raise ValueError("source versions must be unique")
        if max_inflight_batches < 1:
            raise ValueError("max_inflight_batches must be positive")
        if execution_dtype not in {None, torch.float16, torch.bfloat16}:
            raise ValueError("unsupported recompute execution dtype")
        if publication_dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }:
            raise ValueError("unsupported publication dtype")
        model_devices = {
            value.device
            for value in (*model.parameters(), *model.buffers())
        }
        if model_devices != {device}:
            raise ValueError("model must already reside entirely on the executor device")
        if (
            output_pool is not None
            and output_pool.served_kv_target != target_version
        ):
            raise ValueError("output pool and target versions differ")
        self.model = model.eval()
        self.source_versions = frozenset(source_versions)
        self.target_version = target_version
        self.device = device
        self.max_inflight_batches = max_inflight_batches
        self.pin_inputs = pin_inputs
        self.execution_dtype = execution_dtype
        self.publication_dtype = publication_dtype
        self.output_pool = output_pool
        with torch.cuda.device(device):
            self._h2d_stream = torch.cuda.Stream(device=device)
            self._compute_stream = torch.cuda.Stream(device=device)
            self._d2h_stream = torch.cuda.Stream(device=device)

    @property
    def model_nbytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (*self.model.parameters(), *self.model.buffers())
        )

    def _account_batch(
        self,
        batch: RawHistoryBatch,
        seen_record_ids: set[int],
    ) -> tuple[int, int]:
        if batch.migration_anchor_version not in self.source_versions:
            raise ValueError("raw history migration anchor differs from executor")
        overlap = seen_record_ids.intersection(batch.record_ids)
        if overlap:
            raise ValueError("record_ids must be unique across a recompute execution")
        seen_record_ids.update(batch.record_ids)
        return batch.batch_size, int(batch.lengths.sum().item())

    def _compute(self, batch: RawHistoryBatch) -> HSTUKVCache:
        if self.execution_dtype is None:
            return self.model.compute_kv(
                batch.item_ids,
                batch.behaviors,
                batch.time_deltas,
                lengths=batch.lengths,
            )
        with torch.autocast(
            device_type="cuda",
            dtype=self.execution_dtype,
        ):
            return self.model.compute_kv(
                batch.item_ids,
                batch.behaviors,
                batch.time_deltas,
                lengths=batch.lengths,
            )

    @torch.no_grad()
    def run(
        self,
        batches: Iterable[RawHistoryBatch],
    ) -> CohortExecutionReport:
        started = time.perf_counter()
        results = []
        pending: deque[_PendingRecomputeBatch] = deque()
        seen_record_ids: set[int] = set()
        batch_count = 0
        record_count = 0
        token_count = 0
        input_bytes = 0
        output_bytes = 0
        auto_pinned_batches = 0
        preallocated_output_batches = 0

        with torch.cuda.device(self.device):
            for batch in batches:
                if batch.device.type != "cpu":
                    raise ValueError("recompute executor requires CPU-resident histories")
                records, tokens = self._account_batch(batch, seen_record_ids)
                host_batch = batch
                if self.pin_inputs and not batch.is_pinned:
                    host_batch = batch.pin_memory()
                    auto_pinned_batches += 1

                h2d_done = torch.cuda.Event()
                with torch.cuda.stream(self._h2d_stream):
                    device_batch = host_batch.to(self.device, non_blocking=True)
                    h2d_done.record(self._h2d_stream)

                compute_done = torch.cuda.Event()
                with torch.cuda.stream(self._compute_stream):
                    self._compute_stream.wait_event(h2d_done)
                    cache = self._compute(device_batch)
                    device_result = MigratedKVBatch(
                        record_ids=device_batch.record_ids,
                        migration_anchor_version=(
                            device_batch.migration_anchor_version
                        ),
                        served_kv_target=self.target_version,
                        cache=HSTUKVCache(
                            k=cache.k.to(self.publication_dtype),
                            v=cache.v.to(self.publication_dtype),
                            seq_len=cache.seq_len,
                        ),
                        lengths=device_batch.lengths,
                    )
                    compute_done.record(self._compute_stream)

                if self.output_pool is None:
                    host_k = torch.empty(
                        device_result.cache.k.shape,
                        dtype=self.publication_dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    host_v = torch.empty_like(host_k, pin_memory=True)
                    host_result = MigratedKVBatch(
                        record_ids=host_batch.record_ids,
                        migration_anchor_version=(
                            host_batch.migration_anchor_version
                        ),
                        served_kv_target=self.target_version,
                        cache=HSTUKVCache(
                            k=host_k,
                            v=host_v,
                            seq_len=device_result.cache.seq_len,
                        ),
                        lengths=host_batch.lengths,
                    )
                else:
                    host_result = self.output_pool.acquire(host_batch)
                    host_k = host_result.cache.k
                    host_v = host_result.cache.v
                    if host_k.shape != device_result.cache.k.shape:
                        raise ValueError("preallocated K shape differs from recompute result")
                    if host_v.shape != device_result.cache.v.shape:
                        raise ValueError("preallocated V shape differs from recompute result")
                    if host_k.dtype != self.publication_dtype:
                        raise ValueError("preallocated output dtype differs from publication")
                    preallocated_output_batches += 1

                d2h_done = torch.cuda.Event()
                with torch.cuda.stream(self._d2h_stream):
                    self._d2h_stream.wait_event(compute_done)
                    host_k.copy_(device_result.cache.k, non_blocking=True)
                    host_v.copy_(device_result.cache.v, non_blocking=True)
                    d2h_done.record(self._d2h_stream)

                pending.append(
                    _PendingRecomputeBatch(
                        done=d2h_done,
                        host_batch=host_batch,
                        device_batch=device_batch,
                        device_result=device_result,
                        host_result=host_result,
                    )
                )
                batch_count += 1
                record_count += records
                token_count += tokens
                input_bytes += batch.nbytes
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
                preallocated_output_batches=preallocated_output_batches,
                elapsed_seconds=elapsed,
            ),
        )


@dataclass(frozen=True)
class RecomputeDeviceExecutionMetrics:
    device: str
    assigned_work_units: int
    execution: CohortExecutionMetrics


@dataclass(frozen=True)
class MultiGPURecomputeMetrics:
    device_count: int
    batch_count: int
    record_count: int
    token_count: int
    input_bytes: int
    output_bytes: int
    model_replica_bytes: int
    elapsed_seconds: float
    load_imbalance: float
    partition_strategy: str
    devices: tuple[RecomputeDeviceExecutionMetrics, ...]

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
        return (self.input_bytes + self.output_bytes) / (
            2**30 * self.elapsed_seconds
        )


@dataclass(frozen=True)
class MultiGPURecomputeReport:
    batches: tuple[MigratedKVBatch, ...]
    metrics: MultiGPURecomputeMetrics


class MultiGPUFullRecomputeExecutor:
    def __init__(
        self,
        models: Sequence[HSTU],
        source_version: str | Sequence[str],
        target_version: str,
        devices: Sequence[torch.device | str],
        max_inflight_batches: int = 3,
        pin_inputs: bool = True,
        execution_dtype: torch.dtype | None = torch.float16,
        publication_dtype: torch.dtype = torch.float16,
        output_pool: PinnedKVOutputPool | None = None,
        partition_strategy: str = "greedy_lpt",
    ) -> None:
        resolved = tuple(torch.device(device) for device in devices)
        if not resolved:
            raise ValueError("at least one CUDA device is required")
        if len(models) != len(resolved):
            raise ValueError("one model replica is required per CUDA device")
        if len({id(model) for model in models}) != len(models):
            raise ValueError("model replicas must be distinct objects")
        if len(set(resolved)) != len(resolved):
            raise ValueError("multi-GPU recompute devices must be unique")
        if partition_strategy not in {"round_robin", "greedy_lpt"}:
            raise ValueError("unsupported recompute partition strategy")
        self.devices = resolved
        self.partition_strategy = partition_strategy
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=len(resolved))
        self.executors = tuple(
            FullRecomputeStreamingExecutor(
                model=model,
                source_version=source_version,
                target_version=target_version,
                device=device,
                max_inflight_batches=max_inflight_batches,
                pin_inputs=pin_inputs,
                execution_dtype=execution_dtype,
                publication_dtype=publication_dtype,
                output_pool=output_pool,
            )
            for model, device in zip(models, resolved, strict=True)
        )

    def close(self) -> None:
        if not self._closed:
            self._pool.shutdown(wait=True)
            self._closed = True

    def __enter__(self) -> MultiGPUFullRecomputeExecutor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _work_units(self, batch: RawHistoryBatch) -> int:
        return batch.batch_size * batch.seq_len * batch.seq_len

    def _partition(
        self,
        batches: tuple[RawHistoryBatch, ...],
    ) -> tuple[tuple[tuple[int, RawHistoryBatch], ...], ...]:
        assignments: list[list[tuple[int, RawHistoryBatch]]] = [
            [] for _ in self.devices
        ]
        if self.partition_strategy == "round_robin":
            for index, batch in enumerate(batches):
                assignments[index % len(self.devices)].append((index, batch))
            return tuple(tuple(assignment) for assignment in assignments)
        loads = [0 for _ in self.devices]
        indexed = sorted(
            enumerate(batches),
            key=lambda value: (-self._work_units(value[1]), value[0]),
        )
        for index, batch in indexed:
            worker = min(range(len(loads)), key=lambda value: (loads[value], value))
            assignments[worker].append((index, batch))
            loads[worker] += self._work_units(batch)
        return tuple(tuple(assignment) for assignment in assignments)

    @torch.no_grad()
    def run(
        self,
        batches: Iterable[RawHistoryBatch],
    ) -> MultiGPURecomputeReport:
        if self._closed:
            raise RuntimeError("multi-GPU recompute executor is closed")
        batches = tuple(batches)
        seen_record_ids: set[int] = set()
        for batch in batches:
            overlap = seen_record_ids.intersection(batch.record_ids)
            if overlap:
                raise ValueError("record_ids must be unique across recompute batches")
            seen_record_ids.update(batch.record_ids)
        assignments = self._partition(batches)
        started = time.perf_counter()

        def run_worker(worker: int):
            assignment = assignments[worker]
            report = self.executors[worker].run(
                batch for _, batch in assignment
            )
            return worker, assignment, report

        futures = [
            self._pool.submit(run_worker, worker)
            for worker in range(len(self.devices))
        ]
        worker_results = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
        ordered: list[MigratedKVBatch | None] = [None] * len(batches)
        device_metrics = []
        assigned_loads = []
        for worker, assignment, report in sorted(worker_results):
            if len(assignment) != len(report.batches):
                raise RuntimeError("worker result count differs from assignment")
            for (index, _), result in zip(
                assignment,
                report.batches,
                strict=True,
            ):
                ordered[index] = result
            assigned_work_units = sum(
                self._work_units(batch) for _, batch in assignment
            )
            assigned_loads.append(assigned_work_units)
            device_metrics.append(
                RecomputeDeviceExecutionMetrics(
                    device=str(self.devices[worker]),
                    assigned_work_units=assigned_work_units,
                    execution=report.metrics,
                )
            )
        if any(result is None for result in ordered):
            raise RuntimeError("multi-GPU recompute produced incomplete output")
        mean_load = sum(assigned_loads) / len(assigned_loads)
        load_imbalance = (
            max(assigned_loads) / mean_load - 1.0
            if mean_load > 0
            else 0.0
        )
        return MultiGPURecomputeReport(
            batches=tuple(result for result in ordered if result is not None),
            metrics=MultiGPURecomputeMetrics(
                device_count=len(self.devices),
                batch_count=sum(
                    value.execution.batch_count for value in device_metrics
                ),
                record_count=sum(
                    value.execution.record_count for value in device_metrics
                ),
                token_count=sum(
                    value.execution.token_count for value in device_metrics
                ),
                input_bytes=sum(
                    value.execution.input_bytes for value in device_metrics
                ),
                output_bytes=sum(
                    value.execution.output_bytes for value in device_metrics
                ),
                model_replica_bytes=sum(
                    executor.model_nbytes for executor in self.executors
                ),
                elapsed_seconds=elapsed,
                load_imbalance=load_imbalance,
                partition_strategy=self.partition_strategy,
                devices=tuple(device_metrics),
            ),
        )
