from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from hstu_kvcache.migration import (
    D2ActionPlan,
    D2WavePlan,
    DRAMKVUpdateDestination,
    JaggedMigratedKVBatch,
    SemanticCanaryObservation,
    Stage5CohortPreflight,
    Stage5DeviceCapacity,
    Stage5PreflightMeasurement,
    Stage5PreparedExtent,
    Stage5ProducedExtent,
    Stage5RecordRequest,
    canonical_sha256,
    capture_manifest_snapshot,
    run_stage5_job,
)
from hstu_kvcache.migration.design2_plan import file_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_OUTPUT = (
    "configs/cohortkv_d2/stage_a_stage5_adapter_validation.json"
)
PROTOCOL = "cohortkv_d2_stage_a_stage5_adapter_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _batch(
    record_ids: tuple[int, ...],
    lengths: tuple[int, ...],
    anchor: str,
    target: str,
    value: float,
) -> JaggedMigratedKVBatch:
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            length_tensor.cumsum(0),
        )
    )
    shape = (1, int(offsets[-1]), 1)
    return JaggedMigratedKVBatch(
        record_ids=record_ids,
        migration_anchor_version=anchor,
        served_kv_target=target,
        k=torch.full(shape, value, dtype=torch.float16),
        v=torch.full(shape, -value, dtype=torch.float16),
        lengths=length_tensor,
        offsets=offsets,
    )


def _direct_requests(
    plan: D2ActionPlan,
    wave: D2WavePlan,
) -> tuple[Stage5RecordRequest, ...]:
    cohort_by_id = {
        value.record.record_id: value.cohort_id
        for value in wave.records
    }
    return tuple(
        Stage5RecordRequest(
            record_id=value.record_id,
            cohort_id=cohort_by_id[value.record_id],
            requested_action=(
                "migrate"
                if value.requested_action == "compiled"
                else "exact"
            ),
            source_version=plan.source_version,
            target_version=plan.target_version,
            last_exact_version=value.last_exact_version,
            migration_depth=value.migration_depth,
            requested_reason=value.requested_reason,
            retained_tokens=value.retained_tokens,
            final_tokens=value.final_tokens,
        )
        for value in plan.records
    )


def _canary(
    cohort_id: str,
    record_id: int,
    source_version: str,
    target_version: str,
    passed: bool,
) -> SemanticCanaryObservation:
    return SemanticCanaryObservation(
        cohort_id=cohort_id,
        record_ids=(record_id,),
        source_version=source_version,
        target_version=target_version,
        program_sha256=_sha("d2-stage-a-program"),
        metric="kv_relative_l2",
        observed_relative_l2=0.01 if passed else 0.2,
        maximum_relative_l2=0.05,
        candidate_sha256=_sha("d2-stage-a-candidate"),
        reference_sha256=_sha("d2-stage-a-reference"),
        threshold_artifact_sha256=_sha("d2-stage-a-threshold"),
    )


def _cohorts(
    wave: D2WavePlan,
    canary_passed: bool,
) -> tuple[Stage5CohortPreflight, ...]:
    grouped = {}
    for value in wave.records:
        grouped.setdefault(value.cohort_id, []).append(value)
    output = []
    for cohort_id, records in grouped.items():
        migrants = tuple(
            value.record.record_id
            for value in records
            if value.record.requested_action == "compiled"
        )
        migration_required = bool(migrants)
        output.append(
            Stage5CohortPreflight(
                cohort_id=cohort_id,
                source_version=wave.records[0].source_version,
                target_version=wave.target_version,
                expected_artifact_sha256=_sha(
                    f"{cohort_id}-artifact"
                ),
                observed_artifact_sha256=_sha(
                    f"{cohort_id}-artifact"
                ),
                expected_program_sha256=(
                    _sha("d2-stage-a-program")
                    if migration_required
                    else None
                ),
                observed_program_sha256=(
                    _sha("d2-stage-a-program")
                    if migration_required
                    else None
                ),
                expected_program_shape=(
                    (1, 2, 2) if migration_required else ()
                ),
                observed_program_shape=(
                    (1, 2, 2) if migration_required else ()
                ),
                expected_threshold_artifact_sha256=(
                    _sha("d2-stage-a-threshold")
                    if migration_required
                    else None
                ),
                expected_old_record_ids=migrants,
                present_old_record_ids=migrants,
                device_capacity=(
                    (
                        Stage5DeviceCapacity(
                            device="cpu",
                            model_and_program_bytes=0,
                            old_kv_bytes=0,
                            complete_new_kv_bytes=0,
                            transient_bytes=0,
                            allocator_margin_bytes=0,
                            capacity_bytes=1,
                        ),
                    )
                    if migration_required
                    else ()
                ),
                canary=(
                    _canary(
                        cohort_id,
                        migrants[0],
                        wave.records[0].source_version,
                        wave.target_version,
                        canary_passed,
                    )
                    if migration_required
                    else None
                ),
                measurement=Stage5PreflightMeasurement(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                migration_required=migration_required,
            )
        )
    return tuple(output)


def _old_destination(
    requests: tuple[Stage5RecordRequest, ...],
    source_version: str,
):
    migrant_ids = tuple(
        value.record_id
        for value in requests
        if value.requested_action == "migrate"
    )
    destination = DRAMKVUpdateDestination()
    transaction = destination.begin(
        "d2-stage-a-old",
        source_version,
        migrant_ids,
    )
    transaction.stage(
        "old-00000000",
        _batch(
            migrant_ids,
            (1,) * len(migrant_ids),
            source_version,
            source_version,
            1.0,
        ),
    )
    manifest = transaction.commit()
    return (
        destination,
        manifest,
        capture_manifest_snapshot(destination, manifest),
    )


def _signature(report) -> dict[str, object]:
    manifest = report.target_manifest
    destination_manifest = (
        None if manifest is None else manifest.destination_manifest
    )
    return {
        "outcome": report.outcome,
        "fault": report.fault,
        "target_visible": report.target_visible,
        "partial_target_visible": report.partial_target_visible,
        "staging_reclaimed": report.staging_reclaimed,
        "guard_invocations": report.guard_invocations,
        "staged_extents": report.staged_extents,
        "decisions": [
            value.to_dict() for value in report.preflight.decisions
        ],
        "target_manifest": (
            None
            if destination_manifest is None
            else {
                "record_ids": list(manifest.record_ids),
                "lineage_sha256": manifest.to_dict()[
                    "lineage_sha256"
                ],
                "record_count": destination_manifest.record_count,
                "token_count": destination_manifest.token_count,
                "payload_bytes": destination_manifest.payload_bytes,
                "extents": [
                    {
                        "record_ids": list(value.record_ids),
                        "migration_anchor_version": (
                            value.migration_anchor_version
                        ),
                        "served_kv_target": value.served_kv_target,
                        "token_count": value.token_count,
                        "payload_bytes": value.payload_bytes,
                        "checksum_sha256": value.checksum_sha256,
                    }
                    for value in destination_manifest.extents
                ],
            }
        ),
        "old_readback": (
            None
            if report.old_readback is None
            else {
                "expected_records": report.old_readback.expected_records,
                "read_records": report.old_readback.read_records,
                "passed": report.old_readback.passed,
            }
        ),
        "target_readback": (
            None
            if report.target_readback is None
            else {
                "expected_records": (
                    report.target_readback.expected_records
                ),
                "read_records": report.target_readback.read_records,
                "passed": report.target_readback.passed,
            }
        ),
    }


def _run_scenario(
    plan: D2ActionPlan,
    wave: D2WavePlan,
    requests: tuple[Stage5RecordRequest, ...],
    canary_passed: bool,
    fault: str | None,
):
    destination, old_manifest, old_snapshot = _old_destination(
        requests,
        plan.source_version,
    )
    request_by_id = {
        value.record_id: value for value in requests
    }
    cohort_by_id = {
        value.cohort_id: value
        for value in _cohorts(wave, canary_passed)
    }

    def retained_producer(
        record_ids: tuple[int, ...],
        action: str,
        cohort_id: str,
    ) -> Stage5PreparedExtent:
        lengths = tuple(
            request_by_id[value].retained_tokens
            for value in record_ids
        )
        retained_batch = (
            None
            if not any(lengths)
            else _batch(
                record_ids,
                lengths,
                (
                    plan.source_version
                    if action == "migrate"
                    else plan.target_version
                ),
                plan.target_version,
                2.0 if action == "migrate" else 3.0,
            )
        )
        cohort = cohort_by_id[cohort_id]
        return Stage5PreparedExtent(
            record_ids=record_ids,
            action=action,
            cohort_id=cohort_id,
            source_version=plan.source_version,
            target_version=plan.target_version,
            artifact_sha256=cohort.observed_artifact_sha256,
            program_sha256=(
                cohort.observed_program_sha256
                if action == "migrate"
                else None
            ),
            program_shape=(
                cohort.observed_program_shape
                if action == "migrate"
                else ()
            ),
            retained_lengths=lengths,
            retained_batch=retained_batch,
            num_layers=1,
            kv_width=1,
            dtype="float16",
        )

    def target_appender(
        prepared: Stage5PreparedExtent,
    ) -> Stage5ProducedExtent:
        return Stage5ProducedExtent(
            batch=_batch(
                prepared.record_ids,
                tuple(
                    request_by_id[value].final_tokens
                    for value in prepared.record_ids
                ),
                plan.target_version,
                plan.target_version,
                4.0 if prepared.action == "migrate" else 5.0,
            ),
            source_guard_hook=prepared.guard_hook,
        )

    def guard(prepared, result) -> None:
        if prepared.cohort_id != result.cohort_id:
            raise RuntimeError("D2 Stage 5 guard cohort differs")

    return run_stage5_job(
        job_id=(
            f"d2-stage-a-{fault or 'normal'}-"
            f"{'pass' if canary_passed else 'fallback'}"
        ),
        requests=requests,
        cohorts=tuple(cohort_by_id.values()),
        destination=destination,
        retained_producer=retained_producer,
        target_appender=target_appender,
        guard=guard,
        old_manifest=old_manifest,
        old_snapshot=old_snapshot,
        fault=fault,
        maximum_records_per_extent=128,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    output_path = _path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "D2 Stage A Stage 5 output exists; pass --force"
        )
    action_plan_path = _path(args.action_plan)
    plan = D2ActionPlan.load(action_plan_path)
    wave = D2WavePlan.single_rank(
        plan,
        "d2-stage-a-stage5-adapter",
    )
    wave.validate_against_action_plan(plan)
    adapted = wave.to_stage5_requests()
    direct = _direct_requests(plan, wave)
    scenarios = (
        ("normal_commit", True, None),
        ("semantic_fallback", False, None),
        ("mid_job_abort", True, "mid_job"),
        ("pre_commit_abort", True, "pre_commit"),
    )
    results = {}
    checks = {
        "request_fields_equal": adapted == direct,
        "record_coverage": (
            tuple(value.record_id for value in adapted)
            == tuple(value.record_id for value in plan.records)
        ),
    }
    for name, canary_passed, fault in scenarios:
        adapted_report = _run_scenario(
            plan,
            wave,
            adapted,
            canary_passed,
            fault,
        )
        direct_report = _run_scenario(
            plan,
            wave,
            direct,
            canary_passed,
            fault,
        )
        adapted_signature = _signature(adapted_report)
        direct_signature = _signature(direct_report)
        equal = adapted_signature == direct_signature
        checks[f"{name}_behavior_equal"] = equal
        results[name] = {
            "behavior_equal": equal,
            "signature_sha256": canonical_sha256(
                adapted_signature
            ),
            "outcome": adapted_report.outcome,
            "requested_counts": {
                "migrate": sum(
                    value.requested_action == "migrate"
                    for value in adapted
                ),
                "exact": sum(
                    value.requested_action == "exact"
                    for value in adapted
                ),
            },
            "final_counts": {
                "migrate": sum(
                    value.final_action == "migrate"
                    for value in adapted_report.preflight.decisions
                ),
                "exact": sum(
                    value.final_action == "exact"
                    for value in adapted_report.preflight.decisions
                ),
            },
            "target_payload_bytes": (
                None
                if adapted_report.target_manifest is None
                else adapted_report.target_manifest.destination_manifest.payload_bytes
            ),
            "target_records": (
                0
                if adapted_report.target_manifest is None
                else len(adapted_report.target_manifest.record_ids)
            ),
            "old_readback_passed": (
                None
                if adapted_report.old_readback is None
                else adapted_report.old_readback.passed
            ),
            "target_readback_passed": (
                None
                if adapted_report.target_readback is None
                else adapted_report.target_readback.passed
            ),
        }
    if not all(checks.values()):
        raise RuntimeError(
            f"D2 Stage A Stage 5 checks failed: {checks}"
        )
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "action_plan": {
            "path": str(action_plan_path.relative_to(ROOT)),
            "content_sha256": plan.content_sha256,
            "file_sha256": file_sha256(action_plan_path),
        },
        "scope": {
            "full_frozen_h12_action_plan": True,
            "actual_stage5_transaction_engine": True,
            "synthetic_kv_payload": True,
            "real_edge_performance_measured": False,
            "behavior_fields": [
                "per_record_decisions",
                "fallback",
                "commit_abort",
                "manifest",
                "payload_bytes",
                "readback",
            ],
        },
        "scenarios": results,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
