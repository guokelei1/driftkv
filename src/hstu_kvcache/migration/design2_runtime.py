from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from .design2_plan import D2ActionPlan, D2ActionRecord

D2_WAVE_PLAN_PROTOCOL = "cohortkv_d2_wave_plan_v1"
D2_WAVE_REPORT_PROTOCOL = "cohortkv_d2_wave_report_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^theta([0-9]+)$")


@dataclass(frozen=True)
class D2WaveRecordAction:
    record: D2ActionRecord
    cohort_id: str
    source_version: str
    target_version: str
    old_owner_rank: int
    program_id: str | None
    old_extent_id: str | None
    raw_history_ref: str

    def __post_init__(self) -> None:
        compiled = self.record.requested_action == "compiled"
        if (
            not self.cohort_id
            or _VERSION.fullmatch(self.source_version) is None
            or _VERSION.fullmatch(self.target_version) is None
            or self.old_owner_rank < 0
            or not self.raw_history_ref
            or (
                compiled
                and (not self.program_id or not self.old_extent_id)
            )
            or (
                not compiled
                and (
                    self.program_id is not None
                    or self.old_extent_id is not None
                )
            )
        ):
            raise ValueError("D2 wave record action is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.record.to_dict(),
            "cohort_id": self.cohort_id,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "old_owner_rank": self.old_owner_rank,
            "program_id": self.program_id,
            "old_extent_id": self.old_extent_id,
            "raw_history_ref": self.raw_history_ref,
        }


@dataclass(frozen=True)
class D2WavePlan:
    job_id: str
    action_plan_sha256: str
    target_version: str
    serving_layout: str
    publication_mode: str
    world_size: int
    records: tuple[D2WaveRecordAction, ...]
    protocol: str = D2_WAVE_PLAN_PROTOCOL

    def __post_init__(self) -> None:
        record_ids = tuple(value.record.record_id for value in self.records)
        if (
            self.protocol != D2_WAVE_PLAN_PROTOCOL
            or not self.job_id
            or not _SHA256.fullmatch(self.action_plan_sha256)
            or _VERSION.fullmatch(self.target_version) is None
            or not self.serving_layout
            or not self.publication_mode
            or self.world_size < 1
            or not self.records
            or record_ids != tuple(sorted(record_ids))
            or len(set(record_ids)) != len(record_ids)
            or any(
                value.target_version != self.target_version
                or value.old_owner_rank >= self.world_size
                for value in self.records
            )
        ):
            raise ValueError("D2 wave plan is invalid")

    @classmethod
    def single_rank(
        cls,
        action_plan: D2ActionPlan,
        job_id: str,
    ) -> D2WavePlan:
        program_id = (
            f"{action_plan.source_version}_to_"
            f"{action_plan.target_version}_direct_oldkv"
        )
        records = tuple(
            D2WaveRecordAction(
                record=record,
                cohort_id=(
                    f"{action_plan.source_version}_to_"
                    f"{action_plan.target_version}:"
                    f"{record.requested_reason}"
                ),
                source_version=action_plan.source_version,
                target_version=action_plan.target_version,
                old_owner_rank=0,
                program_id=(
                    program_id
                    if record.requested_action == "compiled"
                    else None
                ),
                old_extent_id=(
                    f"record:{record.record_id}:old_kv"
                    if record.requested_action == "compiled"
                    else None
                ),
                raw_history_ref=(
                    f"prepared_user:{record.prepared_user_id}:"
                    f"{record.target_history_sha256}"
                ),
            )
            for record in action_plan.records
        )
        wave = cls(
            job_id=job_id,
            action_plan_sha256=action_plan.content_sha256,
            target_version=action_plan.target_version,
            serving_layout="single_rank_stage_a_adapter",
            publication_mode="stage5_compatible_cow",
            world_size=1,
            records=records,
        )
        wave.validate_against_action_plan(action_plan)
        return wave

    def validate_against_action_plan(
        self,
        action_plan: D2ActionPlan,
    ) -> None:
        if (
            self.action_plan_sha256 != action_plan.content_sha256
            or self.target_version != action_plan.target_version
            or len(self.records) != len(action_plan.records)
            or any(
                wave.record != action
                or wave.source_version != action_plan.source_version
                or wave.target_version != action_plan.target_version
                for wave, action in zip(
                    self.records,
                    action_plan.records,
                    strict=True,
                )
            )
        ):
            raise ValueError("D2 wave plan differs from its action plan")

    def to_stage5_requests(self) -> tuple[object, ...]:
        from .stage5_closure import Stage5RecordRequest

        return tuple(
            Stage5RecordRequest(
                record_id=value.record.record_id,
                cohort_id=value.cohort_id,
                requested_action=(
                    "migrate"
                    if value.record.requested_action == "compiled"
                    else "exact"
                ),
                source_version=value.source_version,
                target_version=value.target_version,
                last_exact_version=value.record.last_exact_version,
                migration_depth=value.record.migration_depth,
                requested_reason=value.record.requested_reason,
                retained_tokens=value.record.retained_tokens,
                final_tokens=value.record.final_tokens,
            )
            for value in self.records
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "job_id": self.job_id,
            "action_plan_sha256": self.action_plan_sha256,
            "target_version": self.target_version,
            "serving_layout": self.serving_layout,
            "publication_mode": self.publication_mode,
            "world_size": self.world_size,
            "records": [value.to_dict() for value in self.records],
        }


@dataclass(frozen=True)
class D2WaveReport:
    job_id: str
    action_plan_sha256: str
    world_size: int
    status: str
    scientific_result: bool
    coverage_record_ids: tuple[int, ...]
    requested_counts: Mapping[str, int]
    phase_ledger: Mapping[str, object]
    protocol: str = D2_WAVE_REPORT_PROTOCOL

    def __post_init__(self) -> None:
        if (
            self.protocol != D2_WAVE_REPORT_PROTOCOL
            or not self.job_id
            or not _SHA256.fullmatch(self.action_plan_sha256)
            or self.world_size < 1
            or self.status != "stage_a_adapter_validated"
            or self.scientific_result
            or not self.coverage_record_ids
            or self.coverage_record_ids
            != tuple(sorted(self.coverage_record_ids))
            or len(set(self.coverage_record_ids))
            != len(self.coverage_record_ids)
            or sum(self.requested_counts.values())
            != len(self.coverage_record_ids)
            or not self.phase_ledger
        ):
            raise ValueError("D2 wave report is invalid")

    @classmethod
    def from_single_rank_adapter(
        cls,
        wave_plan: D2WavePlan,
        phase_ledger: Mapping[str, object],
    ) -> D2WaveReport:
        counts = Counter(
            value.record.requested_reason
            for value in wave_plan.records
        )
        return cls(
            job_id=wave_plan.job_id,
            action_plan_sha256=wave_plan.action_plan_sha256,
            world_size=wave_plan.world_size,
            status="stage_a_adapter_validated",
            scientific_result=False,
            coverage_record_ids=tuple(
                value.record.record_id for value in wave_plan.records
            ),
            requested_counts={
                "migrate": counts["migrate"],
                "scheduled_exact": counts["scheduled_exact"],
                "natural_exact": counts["natural_exact"],
            },
            phase_ledger=phase_ledger,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "job_id": self.job_id,
            "action_plan_sha256": self.action_plan_sha256,
            "world_size": self.world_size,
            "status": self.status,
            "scientific_result": self.scientific_result,
            "coverage_record_ids": list(self.coverage_record_ids),
            "requested_counts": dict(self.requested_counts),
            "phase_ledger": dict(self.phase_ledger),
        }
