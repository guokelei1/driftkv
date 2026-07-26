from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from .cohort_jagged import (
    FusedJaggedMigrationOperator,
    JaggedCohortStreamingExecutor,
    JaggedMigrationCapsuleBatch,
    JaggedMigrationOperator,
    MultiGPUHBMJaggedCohortExecutor,
    MultiGPUJaggedCohortExecutor,
    PackedJaggedMigrationOperator,
)
from .destination import (
    DestinationKind,
    KVUpdateDestination,
    KVVersionManifest,
    PublicationMode,
)
from .program import MigrationProgram

DESTINATION_OUT_OF_CORE_PROTOCOL = "streamkv_destination_out_of_core_v4"


@dataclass(frozen=True)
class OutOfCoreUpdateMetrics:
    job_id: str
    target_version: str
    destination_kind: DestinationKind
    publication_mode: PublicationMode
    device_count: int
    program_count: int
    publication_queue_depth: int
    wave_count: int
    batch_count: int
    record_count: int
    token_count: int
    input_bytes: int
    output_bytes: int
    program_replica_bytes: int
    peak_wave_input_bytes: int
    peak_wave_output_bytes: int
    execution_seconds: float
    publication_service_seconds: float
    publication_wait_seconds: float
    commit_seconds: float
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
        return (self.input_bytes + self.output_bytes) / (
            2**30 * self.elapsed_seconds
        )


@dataclass(frozen=True)
class OutOfCoreUpdateReport:
    manifest: KVVersionManifest
    metrics: OutOfCoreUpdateMetrics
    protocol: str = DESTINATION_OUT_OF_CORE_PROTOCOL


class OutOfCoreKVUpdateEngine:
    def __init__(
        self,
        program: MigrationProgram | Sequence[MigrationProgram],
        devices: Sequence[torch.device | str],
        destination: KVUpdateDestination,
        wave_batch_limit: int = 8,
        max_inflight_batches: int = 3,
        publication_queue_depth: int = 2,
        operator: JaggedMigrationOperator | None = None,
        partition_strategy: str = "greedy_lpt",
    ) -> None:
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
        resolved = tuple(torch.device(value) for value in devices)
        if not resolved:
            raise ValueError("at least one execution device is required")
        if any(device.type not in {"cpu", "cuda"} for device in resolved):
            raise ValueError("execution devices must be CPU or CUDA")
        if any(device.type == "cpu" for device in resolved) and len(resolved) != 1:
            raise ValueError("CPU execution supports exactly one worker")
        if any(device.type == "cuda" for device in resolved):
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable")
            normalized = []
            for device in resolved:
                if device.type != "cuda":
                    raise ValueError("CPU and CUDA workers cannot be mixed")
                if device.index is None:
                    device = torch.device("cuda", torch.cuda.current_device())
                if device.index >= torch.cuda.device_count():
                    raise ValueError("execution device is unavailable")
                normalized.append(device)
            resolved = tuple(normalized)
        if len(set(resolved)) != len(resolved):
            raise ValueError("execution devices must be unique")
        if wave_batch_limit < 1:
            raise ValueError("wave_batch_limit must be positive")
        if max_inflight_batches < 1:
            raise ValueError("max_inflight_batches must be positive")
        if publication_queue_depth < 1:
            raise ValueError("publication_queue_depth must be positive")
        if partition_strategy not in {
            "round_robin",
            "greedy_input_order",
            "greedy_lpt",
        }:
            raise ValueError("unsupported partition strategy")
        if (
            destination.capabilities.publication_mode
            == PublicationMode.DIRECT_DEVICE
            and any(device.type != "cuda" for device in resolved)
        ):
            raise ValueError("direct-device publication requires CUDA workers")
        self.programs = programs
        self.target_version = next(iter(targets))
        self.devices = resolved
        self.destination = destination
        self.wave_batch_limit = wave_batch_limit
        self.max_inflight_batches = max_inflight_batches
        self.publication_queue_depth = publication_queue_depth
        self.operator = operator or self._default_operator()
        self.partition_strategy = partition_strategy

    def _default_operator(self) -> JaggedMigrationOperator:
        if self.devices[0].type == "cuda":
            return FusedJaggedMigrationOperator()
        return PackedJaggedMigrationOperator(torch.float32)

    def _validate_batches(
        self,
        batches: tuple[JaggedMigrationCapsuleBatch, ...],
    ) -> tuple[int, ...]:
        if not batches:
            raise ValueError("update job must contain at least one batch")
        sources = {program.source_version for program in self.programs}
        record_ids = []
        seen: set[int] = set()
        for batch in batches:
            if batch.device.type != "cpu":
                raise ValueError("out-of-core input capsules must be CPU resident")
            if batch.migration_anchor_version not in sources:
                raise ValueError("no migration program matches a batch anchor")
            overlap = seen.intersection(batch.record_ids)
            if overlap:
                raise ValueError("record IDs must be unique across the update job")
            seen.update(batch.record_ids)
            record_ids.extend(batch.record_ids)
        return tuple(record_ids)

    def _host_runner(self):
        if len(self.devices) == 1:
            return JaggedCohortStreamingExecutor(
                self.programs,
                device=self.devices[0],
                max_inflight_batches=self.max_inflight_batches,
                operator=self.operator,
            )
        return MultiGPUJaggedCohortExecutor(
            self.programs,
            devices=self.devices,
            max_inflight_batches=self.max_inflight_batches,
            operator=self.operator,
            partition_strategy=self.partition_strategy,
        )

    @torch.no_grad()
    def run(
        self,
        job_id: str,
        batches: Sequence[JaggedMigrationCapsuleBatch],
    ) -> OutOfCoreUpdateReport:
        batches = tuple(batches)
        expected_record_ids = self._validate_batches(batches)
        started = time.perf_counter()
        execution_seconds = 0.0
        publication_service_seconds = 0.0
        publication_wait_seconds = 0.0
        commit_seconds = 0.0
        wave_count = 0
        batch_count = 0
        record_count = 0
        token_count = 0
        input_bytes = 0
        output_bytes = 0
        peak_wave_input_bytes = 0
        peak_wave_output_bytes = 0
        active_device_count = len(self.devices)
        transaction = self.destination.begin(
            job_id=job_id,
            target_version=self.target_version,
            expected_record_ids=expected_record_ids,
        )
        try:
            if (
                self.destination.capabilities.publication_mode
                == PublicationMode.DIRECT_DEVICE
            ):
                with MultiGPUHBMJaggedCohortExecutor(
                    self.programs,
                    devices=self.devices,
                    batches=batches,
                    max_inflight_batches=self.max_inflight_batches,
                    operator=self.operator,
                    partition_strategy=self.partition_strategy,
                ) as executor:
                    report = executor.run()
                execution_seconds += report.metrics.elapsed_seconds
                active_device_count = report.metrics.device_count
                publish_started = time.perf_counter()
                for index, batch in enumerate(report.batches):
                    transaction.stage(f"extent-{index:08d}", batch)
                publication_service_seconds += (
                    time.perf_counter() - publish_started
                )
                wave_count = 1
                batch_count = report.metrics.batch_count
                record_count = report.metrics.record_count
                token_count = report.metrics.token_count
                input_bytes = report.metrics.input_bytes
                output_bytes = report.metrics.output_bytes
                peak_wave_input_bytes = input_bytes
                peak_wave_output_bytes = output_bytes
            else:
                runner = self._host_runner()
                publication_pool = ThreadPoolExecutor(max_workers=1)
                pending_publications: deque[Future[float]] = deque()
                try:
                    extent_index = 0
                    for start in range(0, len(batches), self.wave_batch_limit):
                        wave = batches[start : start + self.wave_batch_limit]
                        report = runner.run(wave)
                        execution_seconds += report.metrics.elapsed_seconds
                        wave_input_bytes = report.metrics.input_bytes
                        wave_output_bytes = report.metrics.output_bytes
                        peak_wave_input_bytes = max(
                            peak_wave_input_bytes,
                            wave_input_bytes,
                        )
                        peak_wave_output_bytes = max(
                            peak_wave_output_bytes,
                            wave_output_bytes,
                        )

                        def publish_wave(
                            first_extent: int,
                            outputs=report.batches,
                        ) -> float:
                            publish_started = time.perf_counter()
                            for offset, batch in enumerate(outputs):
                                transaction.stage(
                                    f"extent-{first_extent + offset:08d}",
                                    batch,
                                )
                            return time.perf_counter() - publish_started

                        pending_publications.append(
                            publication_pool.submit(
                                publish_wave,
                                extent_index,
                            )
                        )
                        extent_index += len(report.batches)
                        if (
                            len(pending_publications)
                            >= self.publication_queue_depth
                        ):
                            wait_started = time.perf_counter()
                            publication_service_seconds += (
                                pending_publications.popleft().result()
                            )
                            publication_wait_seconds += (
                                time.perf_counter() - wait_started
                            )
                        wave_count += 1
                        batch_count += report.metrics.batch_count
                        record_count += report.metrics.record_count
                        token_count += report.metrics.token_count
                        input_bytes += wave_input_bytes
                        output_bytes += wave_output_bytes
                    while pending_publications:
                        wait_started = time.perf_counter()
                        publication_service_seconds += (
                            pending_publications.popleft().result()
                        )
                        publication_wait_seconds += (
                            time.perf_counter() - wait_started
                        )
                finally:
                    publication_pool.shutdown(wait=True)
                    close = getattr(runner, "close", None)
                    if close is not None:
                        close()
            commit_started = time.perf_counter()
            manifest = transaction.commit()
            commit_seconds = time.perf_counter() - commit_started
        except BaseException:
            transaction.abort()
            raise
        elapsed = time.perf_counter() - started
        return OutOfCoreUpdateReport(
            manifest=manifest,
            metrics=OutOfCoreUpdateMetrics(
                job_id=job_id,
                target_version=self.target_version,
                destination_kind=self.destination.capabilities.kind,
                publication_mode=self.destination.capabilities.publication_mode,
                device_count=active_device_count,
                program_count=len(self.programs),
                publication_queue_depth=self.publication_queue_depth,
                wave_count=wave_count,
                batch_count=batch_count,
                record_count=record_count,
                token_count=token_count,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                program_replica_bytes=(
                    sum(program.nbytes for program in self.programs)
                    * active_device_count
                ),
                peak_wave_input_bytes=peak_wave_input_bytes,
                peak_wave_output_bytes=peak_wave_output_bytes,
                execution_seconds=execution_seconds,
                publication_service_seconds=publication_service_seconds,
                publication_wait_seconds=publication_wait_seconds,
                commit_seconds=commit_seconds,
                elapsed_seconds=elapsed,
            ),
        )
