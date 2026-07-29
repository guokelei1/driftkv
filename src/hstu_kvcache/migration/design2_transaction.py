from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .design2_distributed import D2CollectiveStep
from .design2_plan import canonical_sha256

D2_PRIVATE_FRAGMENT_PROTOCOL = "cohortkv_d2_private_fragment_v1"
D2_STAGE_B_DECISION_PROTOCOL = "cohortkv_d2_stage_b_decision_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class D2RankCapacity:
    required_bytes: int
    capacity_bytes: int
    measured_peak_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            self.required_bytes < 0
            or self.capacity_bytes < 1
            or (
                self.measured_peak_bytes is not None
                and self.measured_peak_bytes < 0
            )
        ):
            raise ValueError("D2 rank capacity is invalid")

    @property
    def admitted(self) -> bool:
        observed = (
            self.required_bytes
            if self.measured_peak_bytes is None
            else max(self.required_bytes, self.measured_peak_bytes)
        )
        return observed <= self.capacity_bytes

    @property
    def margin_bytes(self) -> int:
        observed = (
            self.required_bytes
            if self.measured_peak_bytes is None
            else max(self.required_bytes, self.measured_peak_bytes)
        )
        return self.capacity_bytes - observed

    def to_dict(self) -> dict[str, object]:
        return {
            "required_bytes": self.required_bytes,
            "capacity_bytes": self.capacity_bytes,
            "measured_peak_bytes": self.measured_peak_bytes,
            "admitted": self.admitted,
            "margin_bytes": self.margin_bytes,
        }


def _validate_phase_trace(
    phase_trace: Sequence[D2CollectiveStep],
) -> None:
    if (
        not phase_trace
        or tuple(value.ordinal for value in phase_trace)
        != tuple(range(len(phase_trace)))
    ):
        raise ValueError("D2 fragment phase trace is invalid")


def _empty_fragment_sha256(
    action_plan_sha256: str,
    target_version: str,
    rank: int,
) -> str:
    return canonical_sha256(
        {
            "action_plan_sha256": action_plan_sha256,
            "target_version": target_version,
            "rank": rank,
            "owner_record_ids": [],
            "payload_bytes": 0,
        }
    )


@dataclass(frozen=True)
class D2RankFragmentMetadata:
    action_plan_sha256: str
    target_version: str
    rank: int
    world_size: int
    owner_record_ids: tuple[int, ...]
    fragment_sha256: str
    payload_bytes: int
    phase_trace: tuple[D2CollectiveStep, ...]
    capacity: D2RankCapacity
    failure_reason: str | None = None
    protocol: str = D2_PRIVATE_FRAGMENT_PROTOCOL

    def __post_init__(self) -> None:
        _validate_phase_trace(self.phase_trace)
        if (
            self.protocol != D2_PRIVATE_FRAGMENT_PROTOCOL
            or not _SHA256.fullmatch(self.action_plan_sha256)
            or not self.target_version
            or self.rank < 0
            or self.world_size < 1
            or self.rank >= self.world_size
            or self.owner_record_ids
            != tuple(sorted(self.owner_record_ids))
            or len(set(self.owner_record_ids))
            != len(self.owner_record_ids)
            or any(value < 0 for value in self.owner_record_ids)
            or not _SHA256.fullmatch(self.fragment_sha256)
            or self.payload_bytes < 0
            or (
                not self.owner_record_ids
                and self.payload_bytes != 0
            )
            or (
                self.failure_reason is None
                and self.owner_record_ids
                and self.payload_bytes < 1
            )
            or self.failure_reason == ""
        ):
            raise ValueError("D2 rank fragment metadata is invalid")

    @property
    def ready(self) -> bool:
        return self.failure_reason is None and self.capacity.admitted

    @classmethod
    def empty(
        cls,
        *,
        action_plan_sha256: str,
        target_version: str,
        rank: int,
        world_size: int,
        phase_trace: Sequence[D2CollectiveStep],
        capacity: D2RankCapacity,
    ) -> D2RankFragmentMetadata:
        return cls(
            action_plan_sha256=action_plan_sha256,
            target_version=target_version,
            rank=rank,
            world_size=world_size,
            owner_record_ids=(),
            fragment_sha256=_empty_fragment_sha256(
                action_plan_sha256,
                target_version,
                rank,
            ),
            payload_bytes=0,
            phase_trace=tuple(phase_trace),
            capacity=capacity,
        )

    def with_synthetic_failure(
        self,
        reason: str,
    ) -> D2RankFragmentMetadata:
        if not reason:
            raise ValueError("D2 synthetic failure reason is empty")
        return replace(self, failure_reason=f"synthetic: {reason}")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "action_plan_sha256": self.action_plan_sha256,
            "target_version": self.target_version,
            "rank": self.rank,
            "world_size": self.world_size,
            "owner_record_ids": list(self.owner_record_ids),
            "fragment_sha256": self.fragment_sha256,
            "payload_bytes": self.payload_bytes,
            "phase_trace": [
                {
                    "ordinal": value.ordinal,
                    "phase": value.phase,
                    "token": value.token,
                }
                for value in self.phase_trace
            ],
            "capacity": self.capacity.to_dict(),
            "failure_reason": self.failure_reason,
            "ready": self.ready,
        }


@dataclass(frozen=True)
class D2StageBTransactionDecision:
    status: str
    action_plan_sha256: str
    target_version: str
    world_size: int
    expected_record_ids: tuple[int, ...]
    covered_record_ids: tuple[int, ...]
    ready_ranks: tuple[int, ...]
    total_fragment_bytes: int
    fragment_set_sha256: str
    phase_trace: tuple[D2CollectiveStep, ...]
    failure_reasons: tuple[str, ...]
    publishes_target_epoch: bool = False
    protocol: str = D2_STAGE_B_DECISION_PROTOCOL

    def __post_init__(self) -> None:
        if self.phase_trace:
            _validate_phase_trace(self.phase_trace)
        if (
            self.protocol != D2_STAGE_B_DECISION_PROTOCOL
            or self.status not in {"ready", "abort"}
            or not _SHA256.fullmatch(self.action_plan_sha256)
            or not self.target_version
            or self.world_size < 1
            or self.expected_record_ids
            != tuple(sorted(self.expected_record_ids))
            or self.covered_record_ids
            != tuple(sorted(self.covered_record_ids))
            or self.ready_ranks != tuple(sorted(self.ready_ranks))
            or len(set(self.ready_ranks)) != len(self.ready_ranks)
            or any(
                not 0 <= value < self.world_size
                for value in self.ready_ranks
            )
            or self.total_fragment_bytes < 0
            or not _SHA256.fullmatch(self.fragment_set_sha256)
            or self.publishes_target_epoch
            or (
                self.status == "ready"
                and (
                    self.failure_reasons
                    or self.covered_record_ids
                    != self.expected_record_ids
                    or self.ready_ranks
                    != tuple(range(self.world_size))
                    or not self.phase_trace
                )
            )
            or (
                self.status == "abort"
                and not self.failure_reasons
            )
        ):
            raise ValueError("D2 Stage B transaction decision is invalid")

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "status": self.status,
            "action_plan_sha256": self.action_plan_sha256,
            "target_version": self.target_version,
            "world_size": self.world_size,
            "expected_record_ids": list(self.expected_record_ids),
            "covered_record_ids": list(self.covered_record_ids),
            "ready_ranks": list(self.ready_ranks),
            "total_fragment_bytes": self.total_fragment_bytes,
            "fragment_set_sha256": self.fragment_set_sha256,
            "phase_trace": [
                {
                    "ordinal": value.ordinal,
                    "phase": value.phase,
                    "token": value.token,
                }
                for value in self.phase_trace
            ],
            "failure_reasons": list(self.failure_reasons),
            "publishes_target_epoch": self.publishes_target_epoch,
        }


def _fragment_set_sha256(
    rank_metadata: Sequence[D2RankFragmentMetadata],
) -> str:
    return canonical_sha256(
        {
            "private_fragments": [
                {
                    "rank": value.rank,
                    "fragment_sha256": value.fragment_sha256,
                    "payload_bytes": value.payload_bytes,
                    "owner_record_ids": list(value.owner_record_ids),
                }
                for value in sorted(
                    rank_metadata,
                    key=lambda item: item.rank,
                )
            ]
        }
    )


def validate_d2_private_fragments(
    *,
    action_plan_sha256: str,
    target_version: str,
    world_size: int,
    record_owner_map: Mapping[int, int],
    rank_metadata: Sequence[D2RankFragmentMetadata],
    expected_phase_trace: Sequence[D2CollectiveStep] | None = None,
) -> D2StageBTransactionDecision:
    if (
        not _SHA256.fullmatch(action_plan_sha256)
        or not target_version
        or world_size < 1
        or not record_owner_map
        or any(
            record_id < 0 or not 0 <= owner < world_size
            for record_id, owner in record_owner_map.items()
        )
    ):
        raise ValueError("D2 transaction expectation is invalid")
    failures = []
    by_rank = Counter(value.rank for value in rank_metadata)
    expected_ranks = tuple(range(world_size))
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != expected_ranks or any(
        count != 1 for count in by_rank.values()
    ):
        failures.append(
            f"rank metadata coverage mismatch: expected {expected_ranks}, "
            f"observed {observed_ranks}"
        )
    for value in rank_metadata:
        if value.world_size != world_size:
            failures.append(
                f"rank {value.rank} world size mismatch"
            )
        if value.action_plan_sha256 != action_plan_sha256:
            failures.append(
                f"rank {value.rank} action plan hash mismatch"
            )
        if value.target_version != target_version:
            failures.append(
                f"rank {value.rank} target version mismatch"
            )
        if value.failure_reason is not None:
            failures.append(
                f"rank {value.rank} failure: {value.failure_reason}"
            )
        if not value.capacity.admitted:
            failures.append(
                f"rank {value.rank} capacity rejected by "
                f"{-value.capacity.margin_bytes} bytes"
            )
    covered = [
        record_id
        for value in rank_metadata
        for record_id in value.owner_record_ids
    ]
    coverage_counts = Counter(covered)
    duplicates = tuple(
        sorted(
            record_id
            for record_id, count in coverage_counts.items()
            if count > 1
        )
    )
    expected_records = tuple(sorted(record_owner_map))
    observed_records = tuple(sorted(coverage_counts))
    missing = tuple(
        sorted(set(expected_records) - set(observed_records))
    )
    unexpected = tuple(
        sorted(set(observed_records) - set(expected_records))
    )
    if duplicates:
        failures.append(f"duplicate record coverage: {duplicates}")
    if missing:
        failures.append(f"missing record coverage: {missing}")
    if unexpected:
        failures.append(f"unexpected record coverage: {unexpected}")
    for value in rank_metadata:
        misplaced = tuple(
            record_id
            for record_id in value.owner_record_ids
            if record_owner_map.get(record_id) != value.rank
        )
        if misplaced:
            failures.append(
                f"rank {value.rank} owner mismatch: {misplaced}"
            )
    traces = {
        tuple(value.phase_trace)
        for value in rank_metadata
    }
    if expected_phase_trace is not None:
        frozen_trace = tuple(expected_phase_trace)
        _validate_phase_trace(frozen_trace)
        if any(trace != frozen_trace for trace in traces):
            failures.append("rank phase trace differs from expected trace")
        decision_trace = frozen_trace
    elif len(traces) == 1:
        decision_trace = next(iter(traces))
    else:
        failures.append("rank phase traces differ")
        decision_trace = ()
    ready_ranks = tuple(
        sorted(value.rank for value in rank_metadata if value.ready)
    )
    status = "ready" if not failures else "abort"
    return D2StageBTransactionDecision(
        status=status,
        action_plan_sha256=action_plan_sha256,
        target_version=target_version,
        world_size=world_size,
        expected_record_ids=expected_records,
        covered_record_ids=observed_records,
        ready_ranks=ready_ranks,
        total_fragment_bytes=sum(
            value.payload_bytes for value in rank_metadata
        ),
        fragment_set_sha256=_fragment_set_sha256(rank_metadata),
        phase_trace=decision_trace,
        failure_reasons=tuple(failures),
    )
