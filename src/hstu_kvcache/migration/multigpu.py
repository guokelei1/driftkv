from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from .capsule import (
    MigratedKVBatch,
    MigrationCapsuleBatch,
    PinnedKVOutputPool,
)
from .executor import CohortExecutionMetrics, CohortStreamingExecutor
from .operator import MigrationOperator, ReferenceMigrationOperator
from .program import MigrationProgram


@dataclass(frozen=True)
class DeviceExecutionMetrics:
    device: str
    assigned_work_bytes: int
    execution: CohortExecutionMetrics


@dataclass(frozen=True)
class MultiGPUExecutionMetrics:
    device_count: int
    program_count: int
    batch_count: int
    record_count: int
    token_count: int
    input_bytes: int
    output_bytes: int
    program_replica_bytes: int
    elapsed_seconds: float
    load_imbalance: float
    partition_strategy: str
    cohort_batch_counts: dict[str, int]
    cohort_record_counts: dict[str, int]
    devices: tuple[DeviceExecutionMetrics, ...]

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
class MultiGPUExecutionReport:
    batches: tuple[MigratedKVBatch, ...]
    metrics: MultiGPUExecutionMetrics


class MultiGPUCohortExecutor:
    def __init__(
        self,
        program: MigrationProgram | Sequence[MigrationProgram],
        devices: Sequence[torch.device | str],
        max_inflight_batches: int = 3,
        pin_inputs: bool = True,
        operator: MigrationOperator | None = None,
        output_pool: PinnedKVOutputPool | None = None,
        partition_strategy: str = "greedy_lpt",
    ) -> None:
        resolved = tuple(torch.device(device) for device in devices)
        if not resolved:
            raise ValueError("at least one CUDA device is required")
        if any(device.type != "cuda" for device in resolved):
            raise ValueError("multi-GPU executor requires CUDA devices")
        if len(set(resolved)) != len(resolved):
            raise ValueError("multi-GPU executor devices must be unique")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if any(
            device.index is not None and device.index >= torch.cuda.device_count()
            for device in resolved
        ):
            raise ValueError("requested CUDA device is unavailable")
        if partition_strategy not in {
            "round_robin",
            "greedy_input_order",
            "greedy_lpt",
        }:
            raise ValueError("unsupported partition strategy")
        programs = (
            (program,)
            if isinstance(program, MigrationProgram)
            else tuple(program)
        )
        if not programs:
            raise ValueError("at least one migration program is required")
        sources = [value.source_version for value in programs]
        if len(set(sources)) != len(sources):
            raise ValueError("migration program sources must be unique")
        targets = {value.target_version for value in programs}
        shapes = {
            (value.num_layers, value.input_width, value.kv_width)
            for value in programs
        }
        if len(targets) != 1 or len(shapes) != 1:
            raise ValueError("migration programs must share target and tensor shape")
        shared_operator = operator or ReferenceMigrationOperator()
        self.programs = programs
        self.program = programs[0]
        self.devices = resolved
        self.partition_strategy = partition_strategy
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=len(resolved))
        self.executors = tuple(
            CohortStreamingExecutor(
                programs,
                device=device,
                max_inflight_batches=max_inflight_batches,
                pin_inputs=pin_inputs,
                operator=shared_operator,
                output_pool=output_pool,
            )
            for device in resolved
        )

    def close(self) -> None:
        if not self._closed:
            self._pool.shutdown(wait=True)
            self._closed = True

    def __enter__(self) -> MultiGPUCohortExecutor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _work_bytes(self, capsule: MigrationCapsuleBatch) -> int:
        output_elements = (
            self.program.num_layers
            * capsule.batch_size
            * capsule.seq_len
            * 2
            * self.program.kv_width
        )
        output_bytes = output_elements * capsule.normed.element_size()
        return capsule.nbytes + output_bytes

    def _partition(
        self,
        batches: tuple[MigrationCapsuleBatch, ...],
    ) -> tuple[tuple[tuple[int, MigrationCapsuleBatch], ...], ...]:
        assignments: list[list[tuple[int, MigrationCapsuleBatch]]] = [
            [] for _ in self.devices
        ]
        if self.partition_strategy == "round_robin":
            for index, capsule in enumerate(batches):
                assignments[index % len(self.devices)].append((index, capsule))
            return tuple(tuple(assignment) for assignment in assignments)
        loads = [0 for _ in self.devices]
        indexed = list(enumerate(batches))
        if self.partition_strategy == "greedy_lpt":
            indexed.sort(
                key=lambda value: (
                    -self._work_bytes(value[1]),
                    value[0],
                )
            )
        for index, capsule in indexed:
            worker = min(range(len(loads)), key=lambda value: (loads[value], value))
            assignments[worker].append((index, capsule))
            loads[worker] += self._work_bytes(capsule)
        return tuple(tuple(assignment) for assignment in assignments)

    @torch.no_grad()
    def run(
        self,
        batches: Iterable[MigrationCapsuleBatch],
    ) -> MultiGPUExecutionReport:
        if self._closed:
            raise RuntimeError("multi-GPU executor is closed")
        batches = tuple(batches)
        seen_record_ids: set[int] = set()
        for capsule in batches:
            overlap = seen_record_ids.intersection(capsule.record_ids)
            if overlap:
                raise ValueError("record_ids must be unique across a cohort execution")
            seen_record_ids.update(capsule.record_ids)
        assignments = self._partition(batches)
        started = time.perf_counter()

        def run_worker(worker: int):
            assignment = assignments[worker]
            report = self.executors[worker].run(
                capsule for _, capsule in assignment
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
                raise RuntimeError("worker result count differs from its assignment")
            for (index, _), result in zip(
                assignment,
                report.batches,
                strict=True,
            ):
                ordered[index] = result
            assigned_work_bytes = sum(
                self._work_bytes(capsule) for _, capsule in assignment
            )
            assigned_loads.append(assigned_work_bytes)
            device_metrics.append(
                DeviceExecutionMetrics(
                    device=str(self.devices[worker]),
                    assigned_work_bytes=assigned_work_bytes,
                    execution=report.metrics,
                )
            )
        if any(result is None for result in ordered):
            raise RuntimeError("multi-GPU execution produced incomplete output")
        mean_load = sum(assigned_loads) / len(assigned_loads)
        load_imbalance = (
            max(assigned_loads) / mean_load - 1.0
            if mean_load > 0
            else 0.0
        )
        return MultiGPUExecutionReport(
            batches=tuple(result for result in ordered if result is not None),
            metrics=MultiGPUExecutionMetrics(
                device_count=len(self.devices),
                program_count=len(self.programs),
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
                program_replica_bytes=sum(
                    sum(
                        program.nbytes
                        for program in executor.programs.values()
                    )
                    for executor in self.executors
                ),
                elapsed_seconds=elapsed,
                load_imbalance=load_imbalance,
                partition_strategy=self.partition_strategy,
                cohort_batch_counts={
                    source: sum(
                        batch.migration_anchor_version == source
                        for batch in batches
                    )
                    for source in sorted(
                        {
                            batch.migration_anchor_version
                            for batch in batches
                        }
                    )
                },
                cohort_record_counts={
                    source: sum(
                        batch.batch_size
                        for batch in batches
                        if batch.migration_anchor_version == source
                    )
                    for source in sorted(
                        {
                            batch.migration_anchor_version
                            for batch in batches
                        }
                    )
                },
                devices=tuple(device_metrics),
            ),
        )
