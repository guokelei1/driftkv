from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import torch

from .cohort_jagged import JaggedMigratedKVBatch
from .destination import (
    DestinationKind,
    HBMKVUpdateDestination,
    KVUpdateDestination,
    KVVersionManifest,
)

STAGE5_CLOSURE_PROTOCOL = "cohortkv_single_config_stage5_minimal_closure_v1"
STAGE5_PREFLIGHT_PROTOCOL = "cohortkv_stage5_fixed_preflight_v1"
STAGE5_READBACK_PROTOCOL = "cohortkv_stage5_manifest_readback_v1"
STAGE5_GUARD_HOOK = "post_retained_prefix_pre_append"
STAGE5_COMMIT_HOOK = "post_append_full_cache"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _tensor_sha256(digest, value: torch.Tensor) -> None:
    tensor = value.detach().contiguous().view(torch.uint8).cpu()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())


def jagged_kv_sha256(batch: JaggedMigratedKVBatch) -> str:
    digest = hashlib.sha256()
    digest.update(str(batch.record_ids).encode("utf-8"))
    digest.update(batch.migration_anchor_version.encode("utf-8"))
    digest.update(batch.served_kv_target.encode("utf-8"))
    for value in (batch.k, batch.v, batch.lengths, batch.offsets):
        _tensor_sha256(digest, value)
    return digest.hexdigest()


def stage5_lineage_sha256(
    decisions: Sequence[Stage5RecordDecision],
) -> str:
    payload = json.dumps(
        stage5_lineage_metadata(decisions),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stage5_lineage_metadata(
    decisions: Sequence[Stage5RecordDecision],
) -> dict[str, object]:
    return {
        "protocol": STAGE5_CLOSURE_PROTOCOL,
        "commit_hook": STAGE5_COMMIT_HOOK,
        "lineage": [value.to_dict() for value in decisions],
    }


def _record_sha256(
    batch: JaggedMigratedKVBatch,
    record_id: int,
) -> str:
    row = batch.record_index(record_id)
    start = int(batch.offsets[row])
    stop = int(batch.offsets[row + 1])
    digest = hashlib.sha256()
    digest.update(str(record_id).encode("utf-8"))
    digest.update(batch.migration_anchor_version.encode("utf-8"))
    digest.update(batch.served_kv_target.encode("utf-8"))
    for value in (
        batch.k[:, start:stop],
        batch.v[:, start:stop],
        batch.lengths[row : row + 1],
    ):
        _tensor_sha256(digest, value)
    return digest.hexdigest()


def _validate_jagged_payload(
    batch: JaggedMigratedKVBatch,
) -> tuple[int, ...]:
    lengths = tuple(
        int(value) for value in batch.lengths.detach().cpu()
    )
    offsets = tuple(
        int(value) for value in batch.offsets.detach().cpu()
    )
    expected_offsets = [0]
    for length in lengths:
        expected_offsets.append(expected_offsets[-1] + length)
    if (
        any(value < 1 for value in lengths)
        or offsets != tuple(expected_offsets)
        or expected_offsets[-1] != batch.token_count
        or not bool(torch.isfinite(batch.k).all())
        or not bool(torch.isfinite(batch.v).all())
    ):
        raise RuntimeError("Stage 5 jagged K/V payload is invalid")
    return lengths


@dataclass(frozen=True)
class SemanticCanaryObservation:
    cohort_id: str
    record_ids: tuple[int, ...]
    source_version: str
    target_version: str
    program_sha256: str
    metric: str
    observed_relative_l2: float
    maximum_relative_l2: float
    candidate_sha256: str
    reference_sha256: str
    threshold_artifact_sha256: str
    threshold_source: str = "program_selection"
    labels_used: bool = False

    def __post_init__(self) -> None:
        if (
            not self.cohort_id
            or not self.record_ids
            or len(set(self.record_ids)) != len(self.record_ids)
            or not self.source_version
            or not self.target_version
            or not _SHA256.fullmatch(self.program_sha256)
            or self.metric != "kv_relative_l2"
            or not math.isfinite(self.observed_relative_l2)
            or self.observed_relative_l2 < 0
            or not math.isfinite(self.maximum_relative_l2)
            or self.maximum_relative_l2 < 0
            or not _SHA256.fullmatch(self.candidate_sha256)
            or not _SHA256.fullmatch(self.reference_sha256)
            or not _SHA256.fullmatch(self.threshold_artifact_sha256)
            or self.threshold_source != "program_selection"
            or self.labels_used
        ):
            raise ValueError("Stage 5 semantic canary observation is invalid")

    @property
    def passed(self) -> bool:
        return self.observed_relative_l2 <= self.maximum_relative_l2

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "record_ids": list(self.record_ids),
            "passed": self.passed,
        }


def observe_semantic_canary(
    cohort_id: str,
    source_version: str,
    target_version: str,
    candidate: JaggedMigratedKVBatch,
    reference: JaggedMigratedKVBatch,
    maximum_relative_l2: float,
    threshold_artifact_sha256: str,
    program_sha256: str,
) -> SemanticCanaryObservation:
    if (
        candidate.migration_anchor_version != source_version
        or candidate.served_kv_target != target_version
        or reference.migration_anchor_version != target_version
        or reference.served_kv_target != target_version
        or candidate.record_ids != reference.record_ids
        or candidate.k.shape != reference.k.shape
        or candidate.v.shape != reference.v.shape
        or not torch.equal(
            candidate.lengths.detach().cpu(),
            reference.lengths.detach().cpu(),
        )
        or not torch.equal(
            candidate.offsets.detach().cpu(),
            reference.offsets.detach().cpu(),
        )
    ):
        raise ValueError("Stage 5 semantic canary endpoints differ")
    numerator = (
        (candidate.k.float() - reference.k.float()).square().sum()
        + (candidate.v.float() - reference.v.float()).square().sum()
    )
    denominator = (
        reference.k.float().square().sum()
        + reference.v.float().square().sum()
    )
    value = float(
        torch.sqrt(
            numerator / torch.clamp(denominator, min=torch.finfo(torch.float32).eps)
        ).item()
    )
    return SemanticCanaryObservation(
        cohort_id=cohort_id,
        record_ids=candidate.record_ids,
        source_version=source_version,
        target_version=target_version,
        program_sha256=program_sha256,
        metric="kv_relative_l2",
        observed_relative_l2=value,
        maximum_relative_l2=float(maximum_relative_l2),
        candidate_sha256=jagged_kv_sha256(candidate),
        reference_sha256=jagged_kv_sha256(reference),
        threshold_artifact_sha256=threshold_artifact_sha256,
    )


@dataclass(frozen=True)
class Stage5DeviceCapacity:
    device: str
    model_and_program_bytes: int
    old_kv_bytes: int
    complete_new_kv_bytes: int
    transient_bytes: int
    allocator_margin_bytes: int
    capacity_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.model_and_program_bytes,
            self.old_kv_bytes,
            self.complete_new_kv_bytes,
            self.transient_bytes,
            self.allocator_margin_bytes,
        )
        if (
            not self.device
            or any(value < 0 for value in values)
            or self.capacity_bytes < 1
        ):
            raise ValueError("Stage 5 device capacity input is invalid")

    @property
    def required_bytes(self) -> int:
        return (
            self.model_and_program_bytes
            + self.old_kv_bytes
            + self.complete_new_kv_bytes
            + self.transient_bytes
            + self.allocator_margin_bytes
        )

    @property
    def passed(self) -> bool:
        return self.required_bytes <= self.capacity_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "required_bytes": self.required_bytes,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class Stage5PreflightMeasurement:
    artifact_seconds: float
    old_kv_presence_seconds: float
    capacity_seconds: float
    semantic_canary_seconds: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                self.artifact_seconds,
                self.old_kv_presence_seconds,
                self.capacity_seconds,
                self.semantic_canary_seconds,
            )
        ):
            raise ValueError("Stage 5 preflight measurement is invalid")

    @property
    def total_seconds(self) -> float:
        return (
            self.artifact_seconds
            + self.old_kv_presence_seconds
            + self.capacity_seconds
            + self.semantic_canary_seconds
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "total_seconds": self.total_seconds}


@dataclass(frozen=True)
class Stage5RecordRequest:
    record_id: int
    cohort_id: str
    requested_action: str
    source_version: str
    target_version: str
    last_exact_version: str | None
    migration_depth: int
    requested_reason: str
    retained_tokens: int
    final_tokens: int

    def __post_init__(self) -> None:
        if (
            self.record_id < 0
            or not self.cohort_id
            or self.requested_action not in {"migrate", "exact"}
            or not self.source_version
            or not self.target_version
            or self.migration_depth < 0
            or not self.requested_reason
            or self.retained_tokens < 0
            or self.final_tokens < 1
            or self.retained_tokens >= self.final_tokens
            or (
                self.requested_action == "migrate"
                and (
                    self.last_exact_version is None
                    or self.retained_tokens < 1
                )
            )
        ):
            raise ValueError("Stage 5 record request is invalid")


@dataclass(frozen=True)
class Stage5CohortPreflight:
    cohort_id: str
    source_version: str
    target_version: str
    expected_artifact_sha256: str
    observed_artifact_sha256: str
    expected_program_sha256: str | None
    observed_program_sha256: str | None
    expected_program_shape: tuple[int, ...]
    observed_program_shape: tuple[int, ...]
    expected_threshold_artifact_sha256: str | None
    expected_old_record_ids: tuple[int, ...]
    present_old_record_ids: tuple[int, ...]
    device_capacity: tuple[Stage5DeviceCapacity, ...]
    canary: SemanticCanaryObservation | None
    measurement: Stage5PreflightMeasurement
    migration_required: bool = True
    expected_old_records_source: str = "prior_committed_manifest"
    present_old_records_source: str = "destination_readback"

    def __post_init__(self) -> None:
        if (
            not self.cohort_id
            or not self.source_version
            or not self.target_version
            or not _SHA256.fullmatch(self.expected_artifact_sha256)
            or not _SHA256.fullmatch(self.observed_artifact_sha256)
            or len(set(self.expected_old_record_ids))
            != len(self.expected_old_record_ids)
            or len(set(self.present_old_record_ids))
            != len(self.present_old_record_ids)
            or len({value.device for value in self.device_capacity})
            != len(self.device_capacity)
            or self.expected_old_records_source
            != "prior_committed_manifest"
            or self.present_old_records_source != "destination_readback"
            or (
                self.migration_required
                and (
                    not self.expected_old_record_ids
                    or not self.device_capacity
                    or self.expected_program_sha256 is None
                    or not _SHA256.fullmatch(
                        self.expected_program_sha256
                    )
                    or self.observed_program_sha256 is None
                    or not _SHA256.fullmatch(
                        self.observed_program_sha256
                    )
                    or not self.expected_program_shape
                    or any(
                        value < 1
                        for value in self.expected_program_shape
                    )
                    or not self.observed_program_shape
                    or any(
                        value < 1
                        for value in self.observed_program_shape
                    )
                    or self.expected_threshold_artifact_sha256 is None
                    or not _SHA256.fullmatch(
                        self.expected_threshold_artifact_sha256
                    )
                    or self.canary is None
                    or self.canary.cohort_id != self.cohort_id
                    or self.canary.source_version != self.source_version
                    or self.canary.target_version != self.target_version
                    or self.canary.program_sha256
                    != self.observed_program_sha256
                    or self.canary.threshold_artifact_sha256
                    != self.expected_threshold_artifact_sha256
                    or not set(self.canary.record_ids).issubset(
                        self.expected_old_record_ids
                    )
                )
            )
            or (
                not self.migration_required
                and (
                    self.expected_old_record_ids
                    or self.present_old_record_ids
                    or self.canary is not None
                    or self.expected_program_sha256 is not None
                    or self.observed_program_sha256 is not None
                    or self.expected_program_shape
                    or self.observed_program_shape
                    or self.expected_threshold_artifact_sha256 is not None
                )
            )
        ):
            raise ValueError("Stage 5 cohort preflight input is invalid")

    def checks(self) -> dict[str, bool]:
        return {
            "artifact_identity": (
                self.expected_artifact_sha256
                == self.observed_artifact_sha256
            ),
            "program_identity": (
                not self.migration_required
                or self.expected_program_sha256
                == self.observed_program_sha256
            ),
            "program_shape": (
                not self.migration_required
                or self.expected_program_shape
                == self.observed_program_shape
            ),
            "old_kv_presence": (
                not self.migration_required
                or set(self.expected_old_record_ids).issubset(
                    self.present_old_record_ids
                )
            ),
            "capacity": (
                all(value.passed for value in self.device_capacity)
            ),
            "semantic_canary": (
                not self.migration_required
                or (self.canary is not None and self.canary.passed)
            ),
        }


@dataclass(frozen=True)
class Stage5CohortPreflightResult:
    cohort_id: str
    source_version: str
    target_version: str
    checks: dict[str, bool]
    passed: bool
    fallback_reason: str | None
    migration_required: bool
    expected_artifact_sha256: str
    observed_artifact_sha256: str
    expected_program_sha256: str | None
    observed_program_sha256: str | None
    expected_program_shape: tuple[int, ...]
    observed_program_shape: tuple[int, ...]
    expected_threshold_artifact_sha256: str | None
    expected_old_record_ids: tuple[int, ...]
    present_old_record_ids: tuple[int, ...]
    expected_old_records_source: str
    present_old_records_source: str
    canary: SemanticCanaryObservation | None
    device_capacity: tuple[Stage5DeviceCapacity, ...]
    measurement: Stage5PreflightMeasurement

    def to_dict(self) -> dict[str, object]:
        return {
            "cohort_id": self.cohort_id,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "checks": self.checks,
            "passed": self.passed,
            "fallback_reason": self.fallback_reason,
            "migration_required": self.migration_required,
            "expected_artifact_sha256": self.expected_artifact_sha256,
            "observed_artifact_sha256": self.observed_artifact_sha256,
            "expected_program_sha256": self.expected_program_sha256,
            "observed_program_sha256": self.observed_program_sha256,
            "expected_program_shape": list(self.expected_program_shape),
            "observed_program_shape": list(self.observed_program_shape),
            "expected_threshold_artifact_sha256": (
                self.expected_threshold_artifact_sha256
            ),
            "expected_old_record_ids": list(self.expected_old_record_ids),
            "present_old_record_ids": list(self.present_old_record_ids),
            "expected_old_records_source": self.expected_old_records_source,
            "present_old_records_source": self.present_old_records_source,
            "canary": None if self.canary is None else self.canary.to_dict(),
            "device_capacity": [
                value.to_dict() for value in self.device_capacity
            ],
            "measurement": self.measurement.to_dict(),
        }


@dataclass(frozen=True)
class Stage5RecordDecision:
    record_id: int
    cohort_id: str
    requested_action: str
    requested_reason: str
    final_action: str
    fallback_reason: str | None
    source_version: str
    target_version: str
    last_exact_version_before: str | None
    last_exact_version_after: str
    migration_depth_before: int
    migration_depth_after: int
    state_kind_after: str
    retained_tokens: int
    final_tokens: int

    def __post_init__(self) -> None:
        exact = self.final_action == "exact"
        migrated = self.final_action == "migrate"
        if (
            self.record_id < 0
            or not self.cohort_id
            or self.requested_action not in {"migrate", "exact"}
            or not self.requested_reason
            or not (exact or migrated)
            or not self.source_version
            or not self.target_version
            or self.migration_depth_before < 0
            or self.retained_tokens < 0
            or self.retained_tokens >= self.final_tokens
            or (
                self.fallback_reason is not None
                and self.requested_action != "migrate"
            )
            or (
                (self.requested_action == "migrate" and exact)
                != (self.fallback_reason is not None)
            )
            or (
                exact
                and (
                    self.last_exact_version_after != self.target_version
                    or self.migration_depth_after != 0
                    or self.state_kind_after != "exact"
                )
            )
            or (
                migrated
                and (
                    self.last_exact_version_before is None
                    or
                    self.last_exact_version_after
                    != self.last_exact_version_before
                    or self.migration_depth_after
                    != self.migration_depth_before + 1
                    or self.state_kind_after != "migrated"
                )
            )
        ):
            raise ValueError("Stage 5 record decision is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Stage5PreflightReport:
    protocol: str
    elapsed_seconds: float
    input_measurement_seconds: float
    runtime_validation_seconds: float
    decision_seconds: float
    guard_hook: str
    selection_role: str
    labels_used: bool
    cohorts: tuple[Stage5CohortPreflightResult, ...]
    decisions: tuple[Stage5RecordDecision, ...]

    def __post_init__(self) -> None:
        if (
            self.protocol != STAGE5_PREFLIGHT_PROTOCOL
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
            or any(
                not math.isfinite(value) or value < 0
                for value in (
                    self.input_measurement_seconds,
                    self.runtime_validation_seconds,
                    self.decision_seconds,
                )
            )
            or not math.isclose(
                self.elapsed_seconds,
                self.input_measurement_seconds
                + self.runtime_validation_seconds
                + self.decision_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or self.guard_hook != STAGE5_GUARD_HOOK
            or self.selection_role != "program_selection"
            or self.labels_used
        ):
            raise ValueError("Stage 5 preflight report is invalid")

    @property
    def all_cohorts_passed(self) -> bool:
        return all(value.passed for value in self.cohorts)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "elapsed_seconds": self.elapsed_seconds,
            "input_measurement_seconds": self.input_measurement_seconds,
            "runtime_validation_seconds": self.runtime_validation_seconds,
            "decision_seconds": self.decision_seconds,
            "guard_hook": self.guard_hook,
            "selection_role": self.selection_role,
            "labels_used": self.labels_used,
            "all_cohorts_passed": self.all_cohorts_passed,
            "cohorts": [value.to_dict() for value in self.cohorts],
            "decisions": [value.to_dict() for value in self.decisions],
        }


def run_stage5_preflight(
    requests: Sequence[Stage5RecordRequest],
    cohorts: Sequence[Stage5CohortPreflight],
    runtime_validation_seconds: float = 0.0,
) -> Stage5PreflightReport:
    started = time.perf_counter()
    if (
        not math.isfinite(runtime_validation_seconds)
        or runtime_validation_seconds < 0
    ):
        raise ValueError("Stage 5 runtime validation time is invalid")
    if not requests or not cohorts:
        raise ValueError("Stage 5 preflight requires records and cohorts")
    if len({value.record_id for value in requests}) != len(requests):
        raise ValueError("Stage 5 record requests must be unique")
    cohort_by_id = {value.cohort_id: value for value in cohorts}
    if len(cohort_by_id) != len(cohorts):
        raise ValueError("Stage 5 cohort preflights must be unique")
    unknown = {value.cohort_id for value in requests} - set(cohort_by_id)
    if unknown:
        raise ValueError("Stage 5 record request has no cohort preflight")
    results = []
    result_by_id = {}
    for cohort in cohorts:
        cohort_requests = [
            value for value in requests if value.cohort_id == cohort.cohort_id
        ]
        if not cohort_requests:
            raise ValueError("Stage 5 preflight cohort has no records")
        if any(
            value.source_version != cohort.source_version
            or value.target_version != cohort.target_version
            for value in cohort_requests
        ):
            raise ValueError("Stage 5 cohort versions differ from its records")
        checks = cohort.checks()
        failed = tuple(name for name, passed in checks.items() if not passed)
        result = Stage5CohortPreflightResult(
            cohort_id=cohort.cohort_id,
            source_version=cohort.source_version,
            target_version=cohort.target_version,
            checks=checks,
            passed=not failed,
            fallback_reason=None if not failed else "+".join(failed),
            migration_required=cohort.migration_required,
            expected_artifact_sha256=cohort.expected_artifact_sha256,
            observed_artifact_sha256=cohort.observed_artifact_sha256,
            expected_program_sha256=cohort.expected_program_sha256,
            observed_program_sha256=cohort.observed_program_sha256,
            expected_program_shape=cohort.expected_program_shape,
            observed_program_shape=cohort.observed_program_shape,
            expected_threshold_artifact_sha256=(
                cohort.expected_threshold_artifact_sha256
            ),
            expected_old_record_ids=cohort.expected_old_record_ids,
            present_old_record_ids=cohort.present_old_record_ids,
            expected_old_records_source=cohort.expected_old_records_source,
            present_old_records_source=cohort.present_old_records_source,
            canary=cohort.canary,
            device_capacity=cohort.device_capacity,
            measurement=cohort.measurement,
        )
        results.append(result)
        result_by_id[cohort.cohort_id] = result
    fatal_checks = {
        value.cohort_id: tuple(
            name
            for name in ("artifact_identity", "capacity")
            if not value.checks[name]
        )
        for value in results
    }
    fatal_checks = {
        cohort_id: names
        for cohort_id, names in fatal_checks.items()
        if names
    }
    if fatal_checks:
        raise RuntimeError(
            f"Stage 5 preflight has no safe fallback: {fatal_checks}"
        )
    decisions = []
    for request in requests:
        result = result_by_id[request.cohort_id]
        final_action = (
            request.requested_action if result.passed else "exact"
        )
        exact = final_action == "exact"
        decisions.append(
            Stage5RecordDecision(
                record_id=request.record_id,
                cohort_id=request.cohort_id,
                requested_action=request.requested_action,
                requested_reason=request.requested_reason,
                final_action=final_action,
                fallback_reason=(
                    result.fallback_reason
                    if (
                        not result.passed
                        and request.requested_action == "migrate"
                    )
                    else None
                ),
                source_version=request.source_version,
                target_version=request.target_version,
                last_exact_version_before=request.last_exact_version,
                last_exact_version_after=(
                    request.target_version
                    if exact
                    else request.last_exact_version
                ),
                migration_depth_before=request.migration_depth,
                migration_depth_after=(
                    0 if exact else request.migration_depth + 1
                ),
                state_kind_after="exact" if exact else "migrated",
                retained_tokens=request.retained_tokens,
                final_tokens=request.final_tokens,
            )
        )
    decision_seconds = time.perf_counter() - started
    input_measurement_seconds = sum(
        value.measurement.total_seconds for value in cohorts
    )
    return Stage5PreflightReport(
        protocol=STAGE5_PREFLIGHT_PROTOCOL,
        elapsed_seconds=(
            input_measurement_seconds
            + runtime_validation_seconds
            + decision_seconds
        ),
        input_measurement_seconds=input_measurement_seconds,
        runtime_validation_seconds=runtime_validation_seconds,
        decision_seconds=decision_seconds,
        guard_hook=STAGE5_GUARD_HOOK,
        selection_role="program_selection",
        labels_used=False,
        cohorts=tuple(results),
        decisions=tuple(decisions),
    )


@dataclass(frozen=True)
class Stage5RecordFingerprint:
    record_id: int
    extent_id: str
    migration_anchor_version: str
    served_kv_target: str
    num_layers: int
    token_count: int
    kv_width: int
    dtype: str
    finite: bool
    sha256: str

    def __post_init__(self) -> None:
        if (
            self.record_id < 0
            or not self.extent_id
            or not self.migration_anchor_version
            or not self.served_kv_target
            or self.num_layers < 1
            or self.token_count < 1
            or self.kv_width < 1
            or not self.dtype
            or not _SHA256.fullmatch(self.sha256)
        ):
            raise ValueError("Stage 5 record fingerprint is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Stage5ManifestSnapshot:
    target_version: str
    manifest_record_ids: tuple[int, ...]
    records: tuple[Stage5RecordFingerprint, ...]

    def __post_init__(self) -> None:
        record_ids = tuple(value.record_id for value in self.records)
        if (
            not self.target_version
            or not self.manifest_record_ids
            or len(set(self.manifest_record_ids))
            != len(self.manifest_record_ids)
            or set(record_ids) != set(self.manifest_record_ids)
            or len(record_ids) != len(self.manifest_record_ids)
        ):
            raise ValueError("Stage 5 manifest snapshot is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_version": self.target_version,
            "manifest_record_ids": list(self.manifest_record_ids),
            "records": [value.to_dict() for value in self.records],
        }


def _fingerprints_for_batch(
    batch: JaggedMigratedKVBatch,
    extent_id: str,
) -> tuple[Stage5RecordFingerprint, ...]:
    _validate_jagged_payload(batch)
    fingerprints = []
    for record_id in batch.record_ids:
        row = batch.record_index(record_id)
        start = int(batch.offsets[row])
        stop = int(batch.offsets[row + 1])
        k = batch.k[:, start:stop]
        v = batch.v[:, start:stop]
        fingerprints.append(
            Stage5RecordFingerprint(
                record_id=record_id,
                extent_id=extent_id,
                migration_anchor_version=batch.migration_anchor_version,
                served_kv_target=batch.served_kv_target,
                num_layers=batch.k.shape[0],
                token_count=stop - start,
                kv_width=batch.k.shape[2],
                dtype=str(batch.k.dtype).removeprefix("torch."),
                finite=bool(torch.isfinite(k).all())
                and bool(torch.isfinite(v).all()),
                sha256=_record_sha256(batch, record_id),
            )
        )
    return tuple(fingerprints)


def capture_manifest_snapshot(
    destination: KVUpdateDestination,
    manifest: KVVersionManifest,
) -> Stage5ManifestSnapshot:
    if destination.manifest(manifest.target_version) != manifest:
        raise RuntimeError("Stage 5 source manifest is not currently published")
    fingerprints = []
    loaded_ids = []
    for extent in manifest.extents:
        batch = destination.load_extent(
            manifest.target_version,
            extent.extent_id,
        )
        if (
            batch.record_ids != extent.record_ids
            or batch.migration_anchor_version
            != extent.migration_anchor_version
            or batch.served_kv_target != manifest.target_version
            or batch.k.shape[0] != extent.num_layers
            or batch.k.shape[2] != extent.kv_width
            or batch.token_count != extent.token_count
            or str(batch.k.dtype).removeprefix("torch.") != extent.dtype
            or bool(torch.any(batch.lengths <= 0))
            or int(batch.offsets[0]) != 0
            or int(batch.offsets[-1]) != batch.token_count
            or not torch.equal(
                (batch.offsets[1:] - batch.offsets[:-1]).cpu(),
                batch.lengths.cpu(),
            )
        ):
            raise RuntimeError("Stage 5 source extent differs from its manifest")
        fingerprints.extend(
            _fingerprints_for_batch(batch, extent.extent_id)
        )
        loaded_ids.extend(batch.record_ids)
    if tuple(loaded_ids) != manifest.record_ids:
        raise RuntimeError("Stage 5 source readback order differs from manifest")
    if not all(value.finite for value in fingerprints):
        raise RuntimeError("Stage 5 source readback contains nonfinite K/V")
    return Stage5ManifestSnapshot(
        target_version=manifest.target_version,
        manifest_record_ids=manifest.record_ids,
        records=tuple(fingerprints),
    )


@dataclass(frozen=True)
class Stage5ReadbackReport:
    protocol: str
    target_version: str
    expected_records: int
    read_records: int
    manifest_equal: bool
    all_metadata_equal: bool
    all_finite: bool
    all_checksums_equal: bool
    passed: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_manifest_readback(
    destination: KVUpdateDestination,
    manifest: KVVersionManifest,
    snapshot: Stage5ManifestSnapshot,
) -> Stage5ReadbackReport:
    started = time.perf_counter()
    manifest_equal = False
    try:
        manifest_equal = destination.manifest(manifest.target_version) == manifest
        actual = capture_manifest_snapshot(destination, manifest)
    except (KeyError, RuntimeError, ValueError):
        actual = None
    expected_by_id = {value.record_id: value for value in snapshot.records}
    actual_by_id = (
        {}
        if actual is None
        else {value.record_id: value for value in actual.records}
    )
    common = set(expected_by_id).intersection(actual_by_id)
    metadata_equal = len(common) == len(expected_by_id) and all(
        (
            expected_by_id[record_id].extent_id
            == actual_by_id[record_id].extent_id
            and expected_by_id[record_id].migration_anchor_version
            == actual_by_id[record_id].migration_anchor_version
            and expected_by_id[record_id].served_kv_target
            == actual_by_id[record_id].served_kv_target
            and expected_by_id[record_id].num_layers
            == actual_by_id[record_id].num_layers
            and expected_by_id[record_id].token_count
            == actual_by_id[record_id].token_count
            and expected_by_id[record_id].kv_width
            == actual_by_id[record_id].kv_width
            and expected_by_id[record_id].dtype
            == actual_by_id[record_id].dtype
        )
        for record_id in common
    )
    finite = len(common) == len(expected_by_id) and all(
        actual_by_id[record_id].finite for record_id in common
    )
    checksums = len(common) == len(expected_by_id) and all(
        expected_by_id[record_id].sha256
        == actual_by_id[record_id].sha256
        for record_id in common
    )
    passed = (
        manifest_equal
        and actual is not None
        and snapshot.target_version == manifest.target_version
        and actual.manifest_record_ids == snapshot.manifest_record_ids
        and len(actual_by_id) == len(expected_by_id)
        and metadata_equal
        and finite
        and checksums
    )
    return Stage5ReadbackReport(
        protocol=STAGE5_READBACK_PROTOCOL,
        target_version=manifest.target_version,
        expected_records=len(expected_by_id),
        read_records=len(actual_by_id),
        manifest_equal=manifest_equal,
        all_metadata_equal=metadata_equal,
        all_finite=finite,
        all_checksums_equal=checksums,
        passed=passed,
        elapsed_seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class Stage5PreparedExtent:
    record_ids: tuple[int, ...]
    action: str
    cohort_id: str
    source_version: str
    target_version: str
    artifact_sha256: str
    program_sha256: str | None
    program_shape: tuple[int, ...]
    retained_lengths: tuple[int, ...]
    retained_batch: JaggedMigratedKVBatch | None
    num_layers: int
    kv_width: int
    dtype: str
    guard_hook: str = STAGE5_GUARD_HOOK

    def __post_init__(self) -> None:
        expected_anchor = (
            self.source_version
            if self.action == "migrate"
            else self.target_version
        )
        try:
            retained_payload_lengths = (
                None
                if self.retained_batch is None
                else _validate_jagged_payload(self.retained_batch)
            )
        except RuntimeError as exc:
            raise ValueError("Stage 5 retained extent is invalid") from exc
        dtype = getattr(torch, self.dtype, None)
        batch_invalid = self.retained_batch is not None and (
            self.retained_batch.record_ids != self.record_ids
            or self.retained_batch.migration_anchor_version
            != expected_anchor
            or self.retained_batch.served_kv_target
            != self.target_version
            or retained_payload_lengths != self.retained_lengths
            or self.retained_batch.k.shape[0] != self.num_layers
            or self.retained_batch.k.shape[2] != self.kv_width
            or str(self.retained_batch.k.dtype).removeprefix("torch.")
            != self.dtype
        )
        if (
            not self.record_ids
            or len(set(self.record_ids)) != len(self.record_ids)
            or self.action not in {"migrate", "exact"}
            or not self.cohort_id
            or not self.source_version
            or not self.target_version
            or not _SHA256.fullmatch(self.artifact_sha256)
            or self.num_layers < 1
            or self.kv_width < 1
            or not isinstance(dtype, torch.dtype)
            or not dtype.is_floating_point
            or len(self.retained_lengths) != len(self.record_ids)
            or any(value < 0 for value in self.retained_lengths)
            or (
                self.retained_batch is None
                and any(value != 0 for value in self.retained_lengths)
            )
            or (
                self.retained_batch is not None
                and any(value < 1 for value in self.retained_lengths)
            )
            or batch_invalid
            or (
                self.action == "migrate"
                and (
                    self.program_sha256 is None
                    or not _SHA256.fullmatch(self.program_sha256)
                    or not self.program_shape
                    or any(value < 1 for value in self.program_shape)
                    or self.retained_batch is None
                )
            )
            or (
                self.action == "exact"
                and (
                    self.program_sha256 is not None
                    or self.program_shape
                )
            )
            or self.guard_hook != STAGE5_GUARD_HOOK
        ):
            raise ValueError("Stage 5 retained extent is invalid")


@dataclass(frozen=True)
class Stage5ProducedExtent:
    batch: JaggedMigratedKVBatch
    source_guard_hook: str
    publication_state: str = STAGE5_COMMIT_HOOK

    def __post_init__(self) -> None:
        if (
            self.source_guard_hook != STAGE5_GUARD_HOOK
            or self.publication_state != STAGE5_COMMIT_HOOK
        ):
            raise ValueError("Stage 5 may publish only guarded post-append cache")


Stage5RetainedProducer = Callable[
    [tuple[int, ...], str, str],
    Stage5PreparedExtent,
]
Stage5TargetAppender = Callable[[Stage5PreparedExtent], Stage5ProducedExtent]
Stage5Guard = Callable[
    [Stage5PreparedExtent, Stage5CohortPreflightResult],
    None,
]


@dataclass(frozen=True)
class Stage5CommittedManifest:
    protocol: str
    commit_hook: str
    destination_manifest: KVVersionManifest
    lineage: tuple[Stage5RecordDecision, ...]

    def __post_init__(self) -> None:
        lineage_ids = tuple(value.record_id for value in self.lineage)
        lineage_sha256 = stage5_lineage_sha256(self.lineage)
        expected_metadata = stage5_lineage_metadata(self.lineage)
        actual_metadata = (
            None
            if self.destination_manifest.metadata_json is None
            else json.loads(self.destination_manifest.metadata_json)
        )
        if (
            self.protocol != STAGE5_CLOSURE_PROTOCOL
            or self.commit_hook != STAGE5_COMMIT_HOOK
            or not self.lineage
            or lineage_ids != self.destination_manifest.record_ids
            or self.destination_manifest.metadata_sha256 != lineage_sha256
            or actual_metadata != expected_metadata
            or any(
                value.target_version
                != self.destination_manifest.target_version
                for value in self.lineage
            )
        ):
            raise ValueError("Stage 5 committed manifest is invalid")

    @property
    def record_ids(self) -> tuple[int, ...]:
        return self.destination_manifest.record_ids

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "commit_hook": self.commit_hook,
            "lineage_sha256": stage5_lineage_sha256(self.lineage),
            "destination_manifest": self.destination_manifest.to_dict(),
            "lineage": [value.to_dict() for value in self.lineage],
        }


@dataclass(frozen=True)
class Stage5JobReport:
    protocol: str
    job_id: str
    target_version: str
    outcome: str
    fault: str | None
    preflight: Stage5PreflightReport
    target_manifest: Stage5CommittedManifest | None
    target_visible: bool
    partial_target_visible: bool
    staging_reclaimed: bool
    old_readback: Stage5ReadbackReport | None
    target_readback: Stage5ReadbackReport | None
    guard_invocations: int
    staged_extents: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if (
            self.protocol != STAGE5_CLOSURE_PROTOCOL
            or self.outcome not in {"committed", "aborted"}
            or self.fault not in {None, "mid_job", "pre_commit"}
            or self.elapsed_seconds < 0
            or self.guard_invocations < 0
            or self.staged_extents < 0
            or self.guard_invocations != self.staged_extents
            or self.partial_target_visible
            or not self.staging_reclaimed
            or (self.outcome == "committed") != self.target_visible
            or (self.outcome == "committed") != (self.target_manifest is not None)
            or (
                self.outcome == "aborted"
                and (
                    self.old_readback is None
                    or not self.old_readback.passed
                    or self.target_readback is not None
                )
            )
            or (
                self.outcome == "committed"
                and (
                    self.old_readback is not None
                    or self.target_readback is None
                    or not self.target_readback.passed
                )
            )
        ):
            raise ValueError("Stage 5 job report is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "job_id": self.job_id,
            "target_version": self.target_version,
            "outcome": self.outcome,
            "fault": self.fault,
            "preflight": self.preflight.to_dict(),
            "target_manifest": (
                None
                if self.target_manifest is None
                else self.target_manifest.to_dict()
            ),
            "target_visible": self.target_visible,
            "partial_target_visible": self.partial_target_visible,
            "staging_reclaimed": self.staging_reclaimed,
            "old_readback": (
                None
                if self.old_readback is None
                else self.old_readback.to_dict()
            ),
            "target_readback": (
                None
                if self.target_readback is None
                else self.target_readback.to_dict()
            ),
            "guard_invocations": self.guard_invocations,
            "staged_extents": self.staged_extents,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _target_is_visible(
    destination: KVUpdateDestination,
    target_version: str,
) -> bool:
    try:
        destination.manifest(target_version)
    except KeyError:
        return False
    return True


def manifest_present_record_ids(
    destination: KVUpdateDestination,
    manifest: KVVersionManifest,
) -> tuple[int, ...]:
    try:
        if destination.manifest(manifest.target_version) != manifest:
            return ()
    except KeyError:
        return ()
    present = []
    for extent in manifest.extents:
        try:
            batch = destination.load_extent(
                manifest.target_version,
                extent.extent_id,
            )
        except (KeyError, RuntimeError, ValueError):
            continue
        if (
            batch.record_ids != extent.record_ids
            or batch.migration_anchor_version
            != extent.migration_anchor_version
            or batch.served_kv_target != manifest.target_version
            or batch.k.shape[0] != extent.num_layers
            or batch.k.shape[2] != extent.kv_width
            or batch.token_count != extent.token_count
            or str(batch.k.dtype).removeprefix("torch.") != extent.dtype
            or bool(torch.any(batch.lengths <= 0))
            or int(batch.offsets[0]) != 0
            or int(batch.offsets[-1]) != batch.token_count
            or not torch.equal(
                (batch.offsets[1:] - batch.offsets[:-1]).cpu(),
                batch.lengths.cpu(),
            )
        ):
            continue
        present.extend(batch.record_ids)
    return tuple(present)


def _execution_groups(
    decisions: Sequence[Stage5RecordDecision],
    maximum_records_per_extent: int,
    planned_extents: Sequence[tuple[int, ...]] | None,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    if maximum_records_per_extent < 1:
        raise ValueError("Stage 5 maximum records per extent must be positive")
    decision_by_id = {value.record_id: value for value in decisions}
    if planned_extents is None:
        source_extents = (tuple(decision_by_id),)
    else:
        source_extents = tuple(tuple(value) for value in planned_extents)
        flat = tuple(record_id for extent in source_extents for record_id in extent)
        if (
            not source_extents
            or any(not extent for extent in source_extents)
            or len(set(flat)) != len(flat)
            or set(flat) != set(decision_by_id)
        ):
            raise ValueError("Stage 5 planned extents differ from job records")
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    order = []
    groups = []
    for source_extent in source_extents:
        grouped.clear()
        order.clear()
        for record_id in source_extent:
            decision = decision_by_id[record_id]
            key = (decision.cohort_id, decision.final_action)
            if key not in grouped:
                order.append(key)
            grouped[key].append(record_id)
        for cohort_id, action in order:
            record_ids = grouped[(cohort_id, action)]
            for start in range(0, len(record_ids), maximum_records_per_extent):
                groups.append(
                    (
                        cohort_id,
                        action,
                        tuple(
                            record_ids[
                                start : start + maximum_records_per_extent
                            ]
                        ),
                    )
                )
    return tuple(groups)


def run_stage5_job(
    job_id: str,
    requests: Sequence[Stage5RecordRequest],
    cohorts: Sequence[Stage5CohortPreflight],
    destination: KVUpdateDestination,
    retained_producer: Stage5RetainedProducer,
    target_appender: Stage5TargetAppender,
    guard: Stage5Guard,
    old_manifest: KVVersionManifest | None = None,
    old_snapshot: Stage5ManifestSnapshot | None = None,
    fault: str | None = None,
    maximum_records_per_extent: int = 4,
    planned_extents: Sequence[tuple[int, ...]] | None = None,
) -> Stage5JobReport:
    if fault not in {None, "mid_job", "pre_commit"}:
        raise ValueError("Stage 5 fault is unsupported")
    if not requests:
        raise ValueError("Stage 5 job has no records")
    if not callable(guard):
        raise ValueError("Stage 5 guard must be callable")
    target_versions = {value.target_version for value in requests}
    if len(target_versions) != 1:
        raise ValueError("Stage 5 job must have one target version")
    started = time.perf_counter()
    if old_snapshot is not None and old_manifest is None:
        raise ValueError("Stage 5 old snapshot requires its manifest")
    if fault is not None and old_manifest is None:
        raise ValueError("Stage 5 fault jobs require old readback state")
    if fault is not None and old_snapshot is None:
        raise ValueError("Stage 5 fault jobs require an old snapshot")
    requested_migrants = {
        value.record_id for value in requests if value.requested_action == "migrate"
    }
    if requested_migrants and old_manifest is None:
        raise ValueError("Stage 5 migration requires a committed old manifest")
    runtime_validation_started = time.perf_counter()
    cohort_by_id = {value.cohort_id: value for value in cohorts}
    if len(cohort_by_id) != len(cohorts):
        raise ValueError("Stage 5 cohort preflights must be unique")
    expected_by_cohort = {
        cohort_id: {
            value.record_id
            for value in requests
            if value.cohort_id == cohort_id
            and value.requested_action == "migrate"
        }
        for cohort_id in cohort_by_id
    }
    if any(
        cohort.migration_required != bool(expected_by_cohort[cohort_id])
        for cohort_id, cohort in cohort_by_id.items()
    ):
        raise ValueError("Stage 5 cohort migration requirement differs")
    if old_manifest is not None:
        actual_present = set(
            manifest_present_record_ids(destination, old_manifest)
        )
        manifest_expected = set(old_manifest.record_ids)
        for cohort_id, cohort in cohort_by_id.items():
            expected = expected_by_cohort[cohort_id]
            if (
                set(cohort.expected_old_record_ids) != expected
                or not expected.issubset(manifest_expected)
                or set(cohort.present_old_record_ids)
                != expected.intersection(actual_present)
                or (
                    expected
                    and old_manifest.target_version
                    != cohort.source_version
                )
            ):
                raise ValueError(
                    "Stage 5 old-cache preflight provenance is invalid"
                )
        migrant_sources = {
            value.source_version
            for value in requests
            if value.requested_action == "migrate"
        }
        if migrant_sources and (
            migrant_sources != {old_manifest.target_version}
            or any(
                extent.migration_anchor_version
                != old_manifest.target_version
                for extent in old_manifest.extents
            )
        ):
            raise ValueError("Stage 5 old-cache version lineage is invalid")
    if destination.capabilities.kind == DestinationKind.HBM:
        if not isinstance(destination, HBMKVUpdateDestination):
            raise TypeError("Stage 5 HBM destination type is unsupported")
        actual_devices = {str(value) for value in destination.devices}
        actual_old_bytes = {value: 0 for value in actual_devices}
        if old_manifest is not None:
            for extent in old_manifest.extents:
                if extent.device is None:
                    raise ValueError("Stage 5 HBM extent has no device")
                actual_old_bytes[extent.device] += extent.payload_bytes
        capacity_sets = [
            tuple(cohort.device_capacity)
            for cohort in cohorts
            if cohort.device_capacity
        ]
        if not capacity_sets or any(
            value != capacity_sets[0] for value in capacity_sets[1:]
        ):
            raise ValueError("Stage 5 HBM capacity declarations differ")
        capacity_by_device = {
            value.device: value for value in capacity_sets[0]
        }
        if (
            set(capacity_by_device) != actual_devices
            or {
                device: value.old_kv_bytes
                for device, value in capacity_by_device.items()
            }
            != actual_old_bytes
            or any(
                value.capacity_bytes
                != torch.cuda.get_device_properties(
                    torch.device(value.device)
                ).total_memory
                for value in capacity_sets[0]
            )
        ):
            raise ValueError("Stage 5 HBM capacity evidence is unbound")
    runtime_validation_seconds = (
        time.perf_counter() - runtime_validation_started
    )
    preflight = run_stage5_preflight(
        requests,
        cohorts,
        runtime_validation_seconds,
    )
    target_version = target_versions.pop()
    expected = tuple(value.record_id for value in requests)
    request_by_id = {value.record_id: value for value in requests}
    decision_by_id = {
        value.record_id: value for value in preflight.decisions
    }
    groups = _execution_groups(
        preflight.decisions,
        maximum_records_per_extent,
        planned_extents,
    )
    publication_record_ids = tuple(
        record_id
        for _, _, record_ids in groups
        for record_id in record_ids
    )
    lineage_in_publication_order = tuple(
        decision_by_id[record_id] for record_id in publication_record_ids
    )
    transaction = destination.begin(
        job_id,
        target_version,
        expected,
        metadata=stage5_lineage_metadata(lineage_in_publication_order),
    )
    result_by_cohort = {
        value.cohort_id: value for value in preflight.cohorts
    }
    hbm_complete_new_bytes: dict[str, int] = {}
    if destination.capabilities.kind == DestinationKind.HBM:
        for cohort in cohorts:
            for value in cohort.device_capacity:
                previous = hbm_complete_new_bytes.setdefault(
                    value.device,
                    value.complete_new_kv_bytes,
                )
                if previous != value.complete_new_kv_bytes:
                    raise ValueError(
                        "Stage 5 HBM capacity declarations differ"
                    )
    staged_hbm_bytes: dict[str, int] = defaultdict(int)
    guard_invocations = 0
    staged_extents = 0
    expected_target_fingerprints = []
    try:
        for index, (cohort_id, action, record_ids) in enumerate(groups):
            prepared = retained_producer(record_ids, action, cohort_id)
            cohort = cohort_by_id[cohort_id]
            requested_retained_lengths = tuple(
                request_by_id[record_id].retained_tokens
                for record_id in record_ids
            )
            expected_program_sha256 = (
                cohort.observed_program_sha256
                if action == "migrate"
                else None
            )
            expected_program_shape = (
                cohort.observed_program_shape
                if action == "migrate"
                else ()
            )
            if (
                prepared.record_ids != record_ids
                or prepared.action != action
                or prepared.cohort_id != cohort_id
                or prepared.source_version != cohort.source_version
                or prepared.target_version != cohort.target_version
                or prepared.artifact_sha256
                != cohort.observed_artifact_sha256
                or prepared.program_sha256 != expected_program_sha256
                or prepared.program_shape != expected_program_shape
                or prepared.retained_lengths != requested_retained_lengths
            ):
                raise RuntimeError(
                    "Stage 5 retained producer differs from its group"
                )
            guard(prepared, result_by_cohort[cohort_id])
            guard_invocations += 1
            produced = target_appender(prepared)
            produced_lengths = _validate_jagged_payload(produced.batch)
            expected_final_lengths = tuple(
                request_by_id[record_id].final_tokens
                for record_id in record_ids
            )
            if (
                produced.batch.record_ids != record_ids
                or produced.batch.migration_anchor_version
                != cohort.target_version
                or produced.batch.served_kv_target != target_version
                or produced_lengths != expected_final_lengths
                or produced.batch.k.shape[0] != prepared.num_layers
                or produced.batch.k.shape[2] != prepared.kv_width
                or str(produced.batch.k.dtype).removeprefix("torch.")
                != prepared.dtype
                or produced.source_guard_hook != prepared.guard_hook
            ):
                raise RuntimeError(
                    "Stage 5 target append differs from its retained group"
                )
            if (
                destination.capabilities.kind == DestinationKind.HBM
                and hbm_complete_new_bytes
            ):
                device = str(produced.batch.k.device)
                staged_hbm_bytes[device] += produced.batch.nbytes
                if (
                    device not in hbm_complete_new_bytes
                    or staged_hbm_bytes[device]
                    > hbm_complete_new_bytes[device]
                ):
                    raise RuntimeError(
                        "Stage 5 produced cache exceeds HBM capacity evidence"
                    )
            extent_id = f"extent-{index:08d}"
            expected_target_fingerprints.extend(
                _fingerprints_for_batch(produced.batch, extent_id)
            )
            transaction.stage(extent_id, produced.batch)
            staged_extents += 1
            if fault == "mid_job" and index == 0:
                raise _Stage5InjectedFault("mid_job")
        if fault == "pre_commit":
            raise _Stage5InjectedFault("pre_commit")
        manifest = transaction.commit()
    except _Stage5InjectedFault as exc:
        transaction.abort()
        target_visible = _target_is_visible(destination, target_version)
        staging_reclaimed = not destination.staging_exists(
            transaction.transaction_id
        )
        readback = verify_manifest_readback(
            destination,
            old_manifest,
            old_snapshot,
        )
        return Stage5JobReport(
            protocol=STAGE5_CLOSURE_PROTOCOL,
            job_id=job_id,
            target_version=target_version,
            outcome="aborted",
            fault=str(exc),
            preflight=preflight,
            target_manifest=None,
            target_visible=target_visible,
            partial_target_visible=target_visible,
            staging_reclaimed=staging_reclaimed,
            guard_invocations=guard_invocations,
            old_readback=readback,
            target_readback=None,
            staged_extents=staged_extents,
            elapsed_seconds=time.perf_counter() - started,
        )
    except BaseException:
        transaction.abort()
        raise
    manifest_decisions = tuple(
        decision_by_id[record_id] for record_id in manifest.record_ids
    )
    committed = Stage5CommittedManifest(
        protocol=STAGE5_CLOSURE_PROTOCOL,
        commit_hook=STAGE5_COMMIT_HOOK,
        destination_manifest=manifest,
        lineage=manifest_decisions,
    )
    target_readback = verify_manifest_readback(
        destination,
        manifest,
        Stage5ManifestSnapshot(
            target_version=target_version,
            manifest_record_ids=manifest.record_ids,
            records=tuple(expected_target_fingerprints),
        ),
    )
    if not target_readback.passed:
        raise RuntimeError("Stage 5 committed target readback failed")
    staging_reclaimed = not destination.staging_exists(
        transaction.transaction_id
    )
    return Stage5JobReport(
        protocol=STAGE5_CLOSURE_PROTOCOL,
        job_id=job_id,
        target_version=target_version,
        outcome="committed",
        fault=None,
        preflight=preflight,
        target_manifest=committed,
        target_visible=_target_is_visible(destination, target_version),
        partial_target_visible=False,
        staging_reclaimed=staging_reclaimed,
        old_readback=None,
        target_readback=target_readback,
        guard_invocations=guard_invocations,
        staged_extents=staged_extents,
        elapsed_seconds=time.perf_counter() - started,
    )


class _Stage5InjectedFault(RuntimeError):
    pass
