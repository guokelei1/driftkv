from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from hstu_kvcache.migration import (
    HBMKVUpdateDestination,
    JaggedMigratedKVBatch,
    Stage5CohortPreflight,
    Stage5DeviceCapacity,
    Stage5PreflightMeasurement,
    Stage5PreparedExtent,
    Stage5ProducedExtent,
    Stage5RecordRequest,
    capture_manifest_snapshot,
    manifest_present_record_ids,
    observe_semantic_canary,
    run_stage5_job,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cohortkv_single_config_stage5_gpu_smoke_v1"
OUTPUT = (
    ROOT
    / "results/system/cohortkv_single_config_full_chain_v1"
    / "stage5_gpu_smoke_seed0.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        nargs=2,
        default=("cuda:0", "cuda:1"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[torch.device, ...]:
    if not args.smoke_test:
        raise ValueError("Stage 5 GPU smoke requires --smoke-test")
    devices = tuple(torch.device(value) for value in args.devices)
    if (
        len(set(devices)) != 2
        or any(
            value.type != "cuda"
            or value.index is None
            or value.index >= torch.cuda.device_count()
            for value in devices
        )
    ):
        raise ValueError("Stage 5 GPU smoke requires two distinct CUDA devices")
    return devices


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _batch(
    record_ids: tuple[int, ...],
    anchor: str,
    target: str,
    device: torch.device,
    value: float,
) -> JaggedMigratedKVBatch:
    lengths = torch.tensor(
        [2 for _ in record_ids],
        dtype=torch.long,
        device=device,
    )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            lengths.cumsum(0),
        )
    )
    shape = (2, int(offsets[-1]), 8)
    k = torch.full(shape, value, dtype=torch.float16, device=device)
    v = torch.full(shape, -value, dtype=torch.float16, device=device)
    return JaggedMigratedKVBatch(
        record_ids=record_ids,
        migration_anchor_version=anchor,
        served_kv_target=target,
        k=k,
        v=v,
        lengths=lengths,
        offsets=offsets,
    )


def _append_one_token(
    source: JaggedMigratedKVBatch,
) -> JaggedMigratedKVBatch:
    k_rows = []
    v_rows = []
    for row, record_id in enumerate(source.record_ids):
        start = int(source.offsets[row])
        stop = int(source.offsets[row + 1])
        suffix_value = float(record_id) / 100
        suffix_k = torch.full(
            (source.k.shape[0], 1, source.k.shape[2]),
            suffix_value,
            dtype=source.k.dtype,
            device=source.k.device,
        )
        suffix_v = -suffix_k
        k_rows.append(torch.cat((source.k[:, start:stop], suffix_k), dim=1))
        v_rows.append(torch.cat((source.v[:, start:stop], suffix_v), dim=1))
    lengths = source.lengths + 1
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=source.k.device),
            lengths.cumsum(0),
        )
    )
    return JaggedMigratedKVBatch(
        record_ids=source.record_ids,
        migration_anchor_version=source.served_kv_target,
        served_kv_target=source.served_kv_target,
        k=torch.cat(k_rows, dim=1).contiguous(),
        v=torch.cat(v_rows, dim=1).contiguous(),
        lengths=lengths,
        offsets=offsets,
    )


def _requests() -> tuple[Stage5RecordRequest, ...]:
    return (
        Stage5RecordRequest(
            10,
            "theta0-cohort",
            "migrate",
            "theta0",
            "theta1",
            "theta0",
            1,
            "scheduler_migrate",
            2,
            3,
        ),
        Stage5RecordRequest(
            11,
            "theta0-cohort",
            "exact",
            "theta0",
            "theta1",
            "theta0",
            2,
            "scheduled_exact",
            2,
            3,
        ),
        Stage5RecordRequest(
            12,
            "theta0-cohort",
            "migrate",
            "theta0",
            "theta1",
            "theta0",
            1,
            "scheduler_migrate",
            2,
            3,
        ),
        Stage5RecordRequest(
            13,
            "cold",
            "exact",
            "theta0",
            "theta1",
            None,
            0,
            "cold_exact",
            2,
            3,
        ),
    )


def _publish_old(
    destination: HBMKVUpdateDestination,
    devices: tuple[torch.device, ...],
):
    transaction = destination.begin("stage5-smoke-old", "theta0", (10, 11, 12))
    transaction.stage(
        "old-00000000",
        _batch((10, 11), "theta0", "theta0", devices[0], 1.0),
    )
    transaction.stage(
        "old-00000001",
        _batch((12,), "theta0", "theta0", devices[1], 2.0),
    )
    manifest = transaction.commit()
    snapshot = capture_manifest_snapshot(destination, manifest)
    return manifest, snapshot


def _capacity(
    manifest,
    devices: tuple[torch.device, ...],
) -> tuple[Stage5DeviceCapacity, ...]:
    old_by_device = {
        str(device): sum(
            extent.payload_bytes
            for extent in manifest.extents
            if extent.device == str(device)
        )
        for device in devices
    }
    return tuple(
        Stage5DeviceCapacity(
            device=str(device),
            model_and_program_bytes=0,
            old_kv_bytes=old_by_device[str(device)],
            complete_new_kv_bytes=old_by_device[str(device)] + 4096,
            transient_bytes=4096,
            allocator_margin_bytes=1024 * 1024,
            capacity_bytes=torch.cuda.get_device_properties(
                device
            ).total_memory,
        )
        for device in devices
    )


def _cohorts(
    destination: HBMKVUpdateDestination,
    old_manifest,
    devices: tuple[torch.device, ...],
    perturb: bool,
) -> tuple[Stage5CohortPreflight, ...]:
    artifact_started = time.perf_counter()
    artifact_sha = _sha(Path(__file__).read_text())
    artifact_seconds = time.perf_counter() - artifact_started
    presence_started = time.perf_counter()
    present = set(manifest_present_record_ids(destination, old_manifest))
    presence_seconds = time.perf_counter() - presence_started
    capacity_started = time.perf_counter()
    capacity = _capacity(old_manifest, devices)
    capacity_seconds = time.perf_counter() - capacity_started
    reference = _batch((10,), "theta1", "theta1", devices[0], 1.0)
    candidate = _batch(
        (10,),
        "theta0",
        "theta1",
        devices[0],
        1.5 if perturb else 1.01,
    )
    canary_started = time.perf_counter()
    canary = observe_semantic_canary(
        "theta0-cohort",
        "theta0",
        "theta1",
        candidate,
        reference,
        0.05,
        _sha("stage5-smoke-threshold-v1"),
        _sha("stage5-smoke-program"),
    )
    torch.cuda.synchronize(devices[0])
    canary_seconds = time.perf_counter() - canary_started
    measurement = Stage5PreflightMeasurement(
        artifact_seconds,
        presence_seconds,
        capacity_seconds,
        canary_seconds,
    )
    migration = Stage5CohortPreflight(
        cohort_id="theta0-cohort",
        source_version="theta0",
        target_version="theta1",
        expected_artifact_sha256=artifact_sha,
        observed_artifact_sha256=artifact_sha,
        expected_program_sha256=_sha("stage5-smoke-program"),
        observed_program_sha256=_sha("stage5-smoke-program"),
        expected_program_shape=(2, 8, 8),
        observed_program_shape=(2, 8, 8),
        expected_threshold_artifact_sha256=_sha(
            "stage5-smoke-threshold-v1"
        ),
        expected_old_record_ids=(10, 12),
        present_old_record_ids=tuple(
            value for value in (10, 12) if value in present
        ),
        device_capacity=capacity,
        canary=canary,
        measurement=measurement,
    )
    cold = Stage5CohortPreflight(
        cohort_id="cold",
        source_version="theta0",
        target_version="theta1",
        expected_artifact_sha256=artifact_sha,
        observed_artifact_sha256=artifact_sha,
        expected_program_sha256=None,
        observed_program_sha256=None,
        expected_program_shape=(),
        observed_program_shape=(),
        expected_threshold_artifact_sha256=None,
        expected_old_record_ids=(),
        present_old_record_ids=(),
        device_capacity=(),
        canary=None,
        measurement=Stage5PreflightMeasurement(0.0, 0.0, 0.0, 0.0),
        migration_required=False,
    )
    return migration, cold


def _run_case(
    devices: tuple[torch.device, ...],
    name: str,
    perturb: bool,
    fault: str | None,
) -> dict[str, object]:
    destination = HBMKVUpdateDestination(
        devices,
        destination_id=f"stage5-{name}",
    )
    old_manifest, old_snapshot = _publish_old(destination, devices)
    cohorts = _cohorts(destination, old_manifest, devices, perturb)
    events = []

    def retained_producer(record_ids, action, cohort_id):
        events.append(["retained", list(record_ids), action, cohort_id])
        device = devices[record_ids[0] % len(devices)]
        anchor = "theta0" if action == "migrate" else "theta1"
        value = 2.5 if action == "migrate" else 3.5
        return Stage5PreparedExtent(
            record_ids=record_ids,
            action=action,
            cohort_id=cohort_id,
            source_version="theta0",
            target_version="theta1",
            artifact_sha256=cohorts[
                0 if cohort_id == "theta0-cohort" else 1
            ].observed_artifact_sha256,
            program_sha256=(
                _sha("stage5-smoke-program")
                if action == "migrate"
                else None
            ),
            program_shape=(2, 8, 8) if action == "migrate" else (),
            retained_lengths=tuple(2 for _ in record_ids),
            retained_batch=_batch(
                record_ids,
                anchor,
                "theta1",
                device,
                value,
            ),
            num_layers=2,
            kv_width=8,
            dtype="float16",
        )

    def guard(prepared, result):
        events.append(
            [
                "guard",
                list(prepared.record_ids),
                prepared.action,
                result.passed,
            ]
        )

    def target_appender(prepared):
        events.append(
            ["append", list(prepared.record_ids), prepared.action]
        )
        return Stage5ProducedExtent(
            _append_one_token(prepared.retained_batch),
            source_guard_hook=prepared.guard_hook,
        )

    report = run_stage5_job(
        f"stage5-{name}",
        _requests(),
        cohorts,
        destination,
        retained_producer,
        target_appender,
        guard,
        old_manifest,
        old_snapshot,
        fault=fault,
        maximum_records_per_extent=4,
        planned_extents=((10, 11), (12, 13)),
    )
    return {
        **report.to_dict(),
        "events": events,
        "copy_on_write_capacity": [
            value.to_dict() for value in cohorts[0].device_capacity
        ],
    }


def main() -> None:
    args = parse_args()
    devices = validate_args(args)
    for device in devices:
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    cases = {
        "normal": _run_case(devices, "normal", False, None),
        "semantic_fallback": _run_case(
            devices,
            "semantic-fallback",
            True,
            None,
        ),
        "mid_job_abort": _run_case(
            devices,
            "mid-job",
            False,
            "mid_job",
        ),
        "pre_commit_abort": _run_case(
            devices,
            "pre-commit",
            False,
            "pre_commit",
        ),
    }
    checks = {
        "normal_committed": cases["normal"]["outcome"] == "committed",
        "semantic_committed": (
            cases["semantic_fallback"]["outcome"] == "committed"
        ),
        "semantic_fallback_exact": all(
            value["final_action"] == "exact"
            for value in cases["semantic_fallback"]["preflight"]["decisions"]
            if value["cohort_id"] == "theta0-cohort"
        ),
        "mid_job_aborted": cases["mid_job_abort"]["outcome"] == "aborted",
        "pre_commit_aborted": (
            cases["pre_commit_abort"]["outcome"] == "aborted"
        ),
        "abort_readback_passed": all(
            cases[name]["old_readback"]["passed"]
            for name in ("mid_job_abort", "pre_commit_abort")
        ),
        "all_capacity_passed": all(
            point["passed"]
            for case in cases.values()
            for point in case["copy_on_write_capacity"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 5 GPU smoke failed: {checks}")
    payload = {
        "protocol": PROTOCOL,
        "status": "gpu_smoke_passed",
        "scientific_result": False,
        "devices": [str(value) for value in devices],
        "device_names": [
            torch.cuda.get_device_name(value) for value in devices
        ],
        "cases": cases,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
