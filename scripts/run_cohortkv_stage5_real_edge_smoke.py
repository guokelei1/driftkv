from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import cohortkv_stage4_8_sweep_common as stage48
import run_cohortkv_stage4_7_organic_chain as base
import run_cohortkv_stage4_9_rollout_boundary as stage49
import torch
from cohortkv_stage4_7_common import (
    CHECKPOINT_DIR,
    COMPILER_OUTPUT,
    PREPARED_PATH,
    RUNTIME_DIR,
    TRAINING_PATH,
    direct_program_path,
    load_inputs,
    sha256,
)
from evaluate_cohortkv_stage4_6_lifecycle import LAUNCH, execute_direct
from motivation_validity import seed_everything

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
    jagged_kv_sha256,
    manifest_present_record_ids,
    observe_semantic_canary,
    run_stage5_job,
    tail_slice_jagged_cache,
    verify_manifest_readback,
)
from hstu_kvcache.migration.stage45_oldkv import DirectOldKVFusedOperator
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_organic_windows,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cohortkv_single_config_stage5_real_edge_gpu_smoke_v1"
CANARY_PROTOCOL = "cohortkv_stage5_real_edge_smoke_canary_v1"
OUTPUT = (
    ROOT
    / "results/system/cohortkv_single_config_full_chain_v1"
    / "stage5_real_edge_gpu_smoke_seed0.json"
)
CANARY_CONFIG = (
    ROOT
    / "configs/cohortkv_single_config_v1"
    / "stage5_real_edge_smoke_canary.json"
)
BATCH_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepared-data", default=PREPARED_PATH)
    parser.add_argument("--training-result", default=TRAINING_PATH)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--compiler-result", default=COMPILER_OUTPUT)
    parser.add_argument("--runtime-dir", default=RUNTIME_DIR)
    parser.add_argument("--baseline", default=stage48.BASELINE_PATH)
    parser.add_argument("--canary-config", default=str(CANARY_CONFIG))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--candidate",
        choices=tuple(stage49.CANDIDATES),
        default="token_debt_total10",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> torch.device:
    if not args.smoke_test:
        raise ValueError("Stage 5 real-edge runner requires --smoke-test")
    if args.seed != 0 or args.batch_size != BATCH_SIZE:
        raise ValueError("Stage 5 real-edge smoke freezes seed 0 and batch size 4")
    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
        raise ValueError(
            "Stage 5 real-edge smoke requires an available explicit CUDA index"
        )
    return device


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_canary_config(path: str | Path) -> tuple[dict[str, object], str]:
    source = _repo_path(path)
    value = json.loads(source.read_text())
    expected = {
        "labels_used": False,
        "metric": "kv_relative_l2",
        "protocol": CANARY_PROTOCOL,
        "scientific_result": False,
        "scope": "integration_smoke_only",
        "selection_role": "program_selection",
        "source_version": "theta0",
        "target_version": "theta1",
    }
    checks = {
        name: value.get(name) == expected_value
        for name, expected_value in expected.items()
    }
    threshold = value.get("maximum_relative_l2")
    checks["maximum_relative_l2"] = (
        isinstance(threshold, int | float)
        and math.isfinite(float(threshold))
        and float(threshold) > 0
    )
    if not all(checks.values()):
        raise ValueError(f"Stage 5 real-edge canary config differs: {checks}")
    return value, sha256(source)


def select_real_edge_records(
    plans,
    selection,
    manifest_records: list[dict],
) -> tuple[int, int]:
    roles = {
        int(value["record_id"]): str(value["evaluation_role"])
        for value in manifest_records
    }
    migrate = sorted(
        record_id
        for record_id in selection.migrate_ids
        if roles.get(record_id) == "program_selection"
        and plans[record_id].migration_eligible
        and plans[record_id].retained_tokens > 0
        and plans[record_id].delta_tokens > 0
        and plans[record_id].latest_tokens == 1
    )
    exact = sorted(
        record_id
        for record_id in selection.scheduled_exact_ids
        if plans[record_id].migration_eligible
        and plans[record_id].retained_tokens > 0
        and plans[record_id].delta_tokens > 0
        and plans[record_id].latest_tokens == 1
        and record_id not in set(migrate[:1])
    )
    if not migrate or not exact:
        raise RuntimeError(
            "Stage 5 real-edge smoke lacks a program-selection migrant "
            "or scheduled exact record"
        )
    return migrate[0], exact[0]


def build_requests(
    migrate_id: int,
    exact_id: int,
    plans,
) -> tuple[Stage5RecordRequest, ...]:
    if migrate_id == exact_id:
        raise ValueError("Stage 5 real-edge actions must use distinct records")
    return (
        Stage5RecordRequest(
            record_id=migrate_id,
            cohort_id="theta0-cohort",
            requested_action="migrate",
            source_version="theta0",
            target_version="theta1",
            last_exact_version="theta0",
            migration_depth=0,
            requested_reason="stage4_9_scheduler_migrate",
            retained_tokens=plans[migrate_id].retained_tokens,
            final_tokens=plans[migrate_id].final_tokens,
        ),
        Stage5RecordRequest(
            record_id=exact_id,
            cohort_id="scheduled-exact",
            requested_action="exact",
            source_version="theta0",
            target_version="theta1",
            last_exact_version="theta0",
            migration_depth=0,
            requested_reason="stage4_9_scheduled_exact",
            retained_tokens=plans[exact_id].retained_tokens,
            final_tokens=plans[exact_id].final_tokens,
        ),
    )


def validate_event_sequence(events: list[dict[str, object]]) -> bool:
    if not events or len(events) % 3:
        return False
    for start in range(0, len(events), 3):
        retained, guard, append = events[start : start + 3]
        identity = (
            retained.get("record_ids"),
            retained.get("action"),
            retained.get("cohort_id"),
        )
        if (
            tuple(value.get("kind") for value in (retained, guard, append))
            != ("retained", "guard", "append")
            or identity
            != (
                guard.get("record_ids"),
                guard.get("action"),
                guard.get("cohort_id"),
            )
            or identity
            != (
                append.get("record_ids"),
                append.get("action"),
                append.get("cohort_id"),
            )
        ):
            return False
    return True


def _edge_identity(
    expected_checkpoints: list[dict],
    observed_checkpoint_hashes: dict[str, str],
    compiler_descriptor: dict,
    observed_compiler_sha256: str,
) -> tuple[str, str]:
    expected = _canonical_sha256(
        {
            "source_checkpoint_sha256": expected_checkpoints[0]["sha256"],
            "target_checkpoint_sha256": expected_checkpoints[1]["sha256"],
            "compiler_sha256": compiler_descriptor["sha256"],
        }
    )
    observed = _canonical_sha256(
        {
            "source_checkpoint_sha256": observed_checkpoint_hashes["theta0"],
            "target_checkpoint_sha256": observed_checkpoint_hashes["theta1"],
            "compiler_sha256": observed_compiler_sha256,
        }
    )
    return expected, observed


def _capacity(
    device: torch.device,
    target_model,
    program,
    old_manifest,
    final_tokens: int,
    records: int,
    num_layers: int,
    kv_width: int,
) -> tuple[Stage5DeviceCapacity, ...]:
    model_bytes = sum(
        value.numel() * value.element_size()
        for value in (
            *target_model.parameters(),
            *target_model.buffers(),
        )
    )
    complete_new_bytes = (
        2 * num_layers * final_tokens * kv_width * torch.float16.itemsize
        + 3 * records * torch.int64.itemsize
    )
    old_bytes = sum(value.payload_bytes for value in old_manifest.extents)
    transient_bytes = max(old_bytes, complete_new_bytes)
    return (
        Stage5DeviceCapacity(
            device=str(device),
            model_and_program_bytes=model_bytes + program.nbytes,
            old_kv_bytes=old_bytes,
            complete_new_kv_bytes=complete_new_bytes,
            transient_bytes=transient_bytes,
            allocator_margin_bytes=1024**3,
            capacity_bytes=torch.cuda.get_device_properties(device).total_memory,
        ),
    )


def _perturb_canary(
    candidate: JaggedMigratedKVBatch,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=candidate.record_ids,
        migration_anchor_version=candidate.migration_anchor_version,
        served_kv_target=candidate.served_kv_target,
        k=(candidate.k + 1024.0).contiguous(),
        v=(candidate.v - 1024.0).contiguous(),
        lengths=candidate.lengths,
        offsets=candidate.offsets,
    )


def _assert_retained_state(
    prepared: Stage5PreparedExtent,
    result,
    expected_edge_sha256: str,
    expected_program_sha256: str,
    expected_program_shape: tuple[int, ...],
) -> None:
    cache = prepared.retained_batch
    if (
        cache is None
        or prepared.artifact_sha256 != expected_edge_sha256
        or cache.served_kv_target != "theta1"
        or not bool(torch.isfinite(cache.k).all())
        or not bool(torch.isfinite(cache.v).all())
        or (
            prepared.action == "migrate"
            and (
                not result.passed
                or prepared.program_sha256 != expected_program_sha256
                or prepared.program_shape != expected_program_shape
                or cache.migration_anchor_version != "theta0"
            )
        )
        or (
            prepared.action == "exact"
            and (
                prepared.program_sha256 is not None
                or prepared.program_shape
                or cache.migration_anchor_version != "theta1"
            )
        )
    ):
        raise RuntimeError("Stage 5 real-edge retained guard failed")


def _relabel_target(
    cache: JaggedMigratedKVBatch,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=cache.record_ids,
        migration_anchor_version="theta1",
        served_kv_target="theta1",
        k=cache.k,
        v=cache.v,
        lengths=cache.lengths,
        offsets=cache.offsets,
    )


def _published_batches(destination, target_manifest) -> dict[int, JaggedMigratedKVBatch]:
    output = {}
    manifest = target_manifest.destination_manifest
    for extent in manifest.extents:
        batch = destination.load_extent(manifest.target_version, extent.extent_id)
        for record_id, record_batch in base._split_cache(batch).items():
            if record_id in output:
                raise RuntimeError("Stage 5 target record was published twice")
            output[record_id] = record_batch
    return output


@torch.inference_mode()
def _run_case(
    name: str,
    perturb: bool,
    device: torch.device,
    requests: tuple[Stage5RecordRequest, ...],
    plans,
    record_by_id: dict[int, dict],
    old_window,
    target_window,
    cfg,
    target_model,
    operator,
    program,
    old_cache: JaggedMigratedKVBatch,
    edge_expected_sha256: str,
    edge_observed_sha256: str,
    program_expected_sha256: str,
    program_observed_sha256: str,
    program_expected_shape: tuple[int, ...],
    program_observed_shape: tuple[int, ...],
    canary_config: dict[str, object],
    canary_config_sha256: str,
    artifact_seconds: float,
) -> dict[str, object]:
    destination = HBMKVUpdateDestination(
        (device,),
        destination_id=f"stage5-real-edge-{name}",
    )
    old_transaction = destination.begin(
        f"stage5-real-edge-{name}-old",
        "theta0",
        old_cache.record_ids,
    )
    old_transaction.stage("old-00000000", old_cache)
    old_manifest = old_transaction.commit()
    old_snapshot = capture_manifest_snapshot(destination, old_manifest)
    presence_started = time.perf_counter()
    present_ids = manifest_present_record_ids(destination, old_manifest)
    presence_seconds = time.perf_counter() - presence_started
    capacity_started = time.perf_counter()
    capacity = _capacity(
        device,
        target_model,
        program,
        old_manifest,
        sum(plans[value.record_id].final_tokens for value in requests),
        len(requests),
        cfg.num_layers,
        cfg.num_heads * cfg.head_dim,
    )
    capacity_seconds = time.perf_counter() - capacity_started
    migrate_ids = tuple(
        value.record_id
        for value in requests
        if value.requested_action == "migrate"
    )
    canary_started = time.perf_counter()
    sliced = tail_slice_jagged_cache(
        old_cache,
        tuple(plans[value].retained_tokens for value in migrate_ids),
    )
    if sliced.cache is None:
        raise RuntimeError("Stage 5 real-edge canary retained cache is empty")
    canary_candidate = execute_direct(operator, program, sliced.cache, 1)
    canary_batch = stage49._retained_batch(
        migrate_ids,
        plans,
        record_by_id,
        target_window,
        device,
    )
    canary_reference = stage49._exact_cache(
        target_model,
        canary_batch,
        migrate_ids,
        1,
        torch.float16,
    )
    if perturb:
        canary_candidate = _perturb_canary(canary_candidate)
    canary = observe_semantic_canary(
        "theta0-cohort",
        "theta0",
        "theta1",
        canary_candidate,
        canary_reference,
        float(canary_config["maximum_relative_l2"]),
        canary_config_sha256,
        program_observed_sha256,
    )
    torch.cuda.synchronize(device)
    canary_seconds = time.perf_counter() - canary_started
    measurement = Stage5PreflightMeasurement(
        artifact_seconds=artifact_seconds,
        old_kv_presence_seconds=presence_seconds,
        capacity_seconds=capacity_seconds,
        semantic_canary_seconds=canary_seconds,
    )
    cohorts = (
        Stage5CohortPreflight(
            cohort_id="theta0-cohort",
            source_version="theta0",
            target_version="theta1",
            expected_artifact_sha256=edge_expected_sha256,
            observed_artifact_sha256=edge_observed_sha256,
            expected_program_sha256=program_expected_sha256,
            observed_program_sha256=program_observed_sha256,
            expected_program_shape=program_expected_shape,
            observed_program_shape=program_observed_shape,
            expected_threshold_artifact_sha256=canary_config_sha256,
            expected_old_record_ids=migrate_ids,
            present_old_record_ids=tuple(
                value for value in migrate_ids if value in set(present_ids)
            ),
            device_capacity=capacity,
            canary=canary,
            measurement=measurement,
        ),
        Stage5CohortPreflight(
            cohort_id="scheduled-exact",
            source_version="theta0",
            target_version="theta1",
            expected_artifact_sha256=edge_expected_sha256,
            observed_artifact_sha256=edge_observed_sha256,
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
        ),
    )
    events: list[dict[str, object]] = []
    produced_hashes: dict[int, str] = {}
    migration_operator_calls = 0

    def retained_producer(record_ids, action, cohort_id):
        nonlocal migration_operator_calls
        if action == "migrate":
            if tuple(record_ids) != old_cache.record_ids:
                raise RuntimeError(
                    "Stage 5 real-edge migrant differs from old manifest"
                )
            retained = tail_slice_jagged_cache(
                destination.load_extent("theta0", "old-00000000"),
                tuple(plans[value].retained_tokens for value in record_ids),
            )
            if retained.cache is None:
                raise RuntimeError(
                    "Stage 5 real-edge migrated retained cache is empty"
                )
            cache = execute_direct(operator, program, retained.cache, 1)
            migration_operator_calls += 1
            program_sha256 = program_observed_sha256
            program_shape = program_observed_shape
        else:
            batch = stage49._retained_batch(
                record_ids,
                plans,
                record_by_id,
                target_window,
                device,
            )
            cache = stage49._exact_cache(
                target_model,
                batch,
                record_ids,
                1,
                torch.float16,
            )
            program_sha256 = None
            program_shape = ()
        retained_lengths = tuple(
            plans[value].retained_tokens for value in record_ids
        )
        events.append(
            {
                "kind": "retained",
                "record_ids": list(record_ids),
                "action": action,
                "cohort_id": cohort_id,
                "lengths": list(retained_lengths),
                "migration_anchor_version": cache.migration_anchor_version,
                "served_kv_target": cache.served_kv_target,
            }
        )
        return Stage5PreparedExtent(
            record_ids=record_ids,
            action=action,
            cohort_id=cohort_id,
            source_version="theta0",
            target_version="theta1",
            artifact_sha256=edge_observed_sha256,
            program_sha256=program_sha256,
            program_shape=program_shape,
            retained_lengths=retained_lengths,
            retained_batch=cache,
            num_layers=cfg.num_layers,
            kv_width=cfg.num_heads * cfg.head_dim,
            dtype="float16",
        )

    def guard(prepared, result):
        _assert_retained_state(
            prepared,
            result,
            edge_expected_sha256,
            program_expected_sha256,
            program_expected_shape,
        )
        events.append(
            {
                "kind": "guard",
                "record_ids": list(prepared.record_ids),
                "action": prepared.action,
                "cohort_id": prepared.cohort_id,
                "preflight_passed": result.passed,
            }
        )

    def target_appender(prepared):
        if prepared.retained_batch is None:
            raise RuntimeError("Stage 5 real-edge retained batch is missing")
        after_delta, _ = stage49._append_delta(
            target_model,
            prepared.retained_batch,
            plans,
            record_by_id,
            target_window,
            device,
            torch.float16,
        )
        final, hidden, _ = stage49._append_latest(
            target_model,
            after_delta,
            record_by_id,
            target_window,
            device,
            torch.float16,
        )
        final = _relabel_target(final)
        actual_lengths = tuple(int(value) for value in final.lengths)
        expected_lengths = tuple(
            plans[value].final_tokens for value in prepared.record_ids
        )
        if (
            final.record_ids != prepared.record_ids
            or final.migration_anchor_version != "theta1"
            or final.served_kv_target != "theta1"
            or actual_lengths != expected_lengths
            or hidden.shape[0] != len(prepared.record_ids)
            or not bool(torch.isfinite(final.k).all())
            or not bool(torch.isfinite(final.v).all())
            or not bool(torch.isfinite(hidden).all())
        ):
            raise RuntimeError(
                "Stage 5 real-edge target append endpoint differs"
            )
        split = base._split_cache(final)
        produced_hashes.update(
            {
                record_id: jagged_kv_sha256(value)
                for record_id, value in split.items()
            }
        )
        events.append(
            {
                "kind": "append",
                "record_ids": list(prepared.record_ids),
                "action": prepared.action,
                "cohort_id": prepared.cohort_id,
                "after_delta_lengths": [
                    int(value) for value in after_delta.lengths
                ],
                "final_lengths": list(actual_lengths),
                "migration_anchor_version": final.migration_anchor_version,
                "served_kv_target": final.served_kv_target,
            }
        )
        return Stage5ProducedExtent(
            final,
            source_guard_hook=prepared.guard_hook,
        )

    report = run_stage5_job(
        job_id=f"stage5-real-edge-{name}",
        requests=requests,
        cohorts=cohorts,
        destination=destination,
        retained_producer=retained_producer,
        target_appender=target_appender,
        old_manifest=old_manifest,
        old_snapshot=old_snapshot,
        guard=guard,
        maximum_records_per_extent=BATCH_SIZE,
        planned_extents=(
            tuple(value.record_id for value in requests),
        ),
    )
    if report.target_manifest is None:
        raise RuntimeError("Stage 5 real-edge target was not committed")
    published = _published_batches(destination, report.target_manifest)
    readback_hashes = {
        record_id: jagged_kv_sha256(value)
        for record_id, value in published.items()
    }
    old_readback = verify_manifest_readback(
        destination,
        old_manifest,
        old_snapshot,
    )
    decisions = report.preflight.decisions
    checks = {
        "committed": report.outcome == "committed" and report.target_visible,
        "same_hbm_destination": (
            old_manifest.destination_id
            == report.target_manifest.destination_manifest.destination_id
            == destination.destination_id
            and old_manifest.destination_kind.value == "hbm"
            and report.target_manifest.destination_manifest.destination_kind.value
            == "hbm"
        ),
        "copy_on_write_old_readback": old_readback.passed,
        "guard_before_real_append": validate_event_sequence(events),
        "published_records_complete": set(published)
        == {value.record_id for value in requests},
        "published_hashes_match_producer": readback_hashes == produced_hashes,
        "published_lengths_match_target": all(
            tuple(int(value) for value in published[record_id].lengths)
            == (plans[record_id].final_tokens,)
            for record_id in published
        ),
        "published_finite": all(
            bool(torch.isfinite(value.k).all())
            and bool(torch.isfinite(value.v).all())
            for value in published.values()
        ),
        "program_and_edge_identity_bound": (
            edge_expected_sha256 == edge_observed_sha256
            and program_expected_sha256 == program_observed_sha256
            and program_expected_shape == program_observed_shape
        ),
        "semantic_expectation": canary.passed is (not perturb),
        "normal_migration_executed": perturb or migration_operator_calls == 1,
        "fallback_job_skipped_migration": (
            not perturb or migration_operator_calls == 0
        ),
        "fallback_actions_exact": (
            not perturb
            or all(value.final_action == "exact" for value in decisions)
        ),
        "normal_actions_mixed": (
            perturb
            or {value.final_action for value in decisions}
            == {"migrate", "exact"}
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Stage 5 real-edge case failed: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    return {
        "name": name,
        "semantic_perturbation": perturb,
        "scientific_result": False,
        "report": report.to_dict(),
        "canary": canary.to_dict(),
        "migration_operator_calls_inside_job": migration_operator_calls,
        "events": events,
        "old_readback_after_commit": old_readback.to_dict(),
        "produced_record_sha256": {
            str(key): value for key, value in produced_hashes.items()
        },
        "published_record_sha256": {
            str(key): value for key, value in readback_hashes.items()
        },
        "checks": checks,
    }


@torch.inference_mode()
def run_smoke(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    baseline = stage48.load_exact_baseline(args.baseline)
    plan, metadata, training, cfg, manifest, checkpoints = load_inputs(
        args.prepared_data,
        args.training_result,
        args.checkpoint_dir,
    )
    user_ids = tuple(int(value["user_id"]) for value in manifest["records"])
    windows = reconstruct_organic_windows(plan, user_ids)
    compiler_path = _repo_path(args.compiler_result)
    compiler = json.loads(compiler_path.read_text())
    window_checks = base.validate_windows(windows, manifest)
    compiler_checks = base.validate_compiler_payload(
        compiler,
        manifest,
        windows,
        checkpoints,
    )
    provenance_checks = stage48.validate_runtime_provenance(
        args,
        baseline,
        metadata,
        training,
        manifest,
        checkpoints,
        windows,
        compiler,
    )
    if torch.cuda.get_device_name(device) != baseline["configuration"][
        "device_class"
    ]:
        raise ValueError("Stage 5 real-edge smoke device class differs")
    old_window, target_window = windows[:2]
    record_by_id = {
        int(value["record_id"]): value for value in manifest["records"]
    }
    previous_ids = {
        record_id
        for record_id, descriptor in record_by_id.items()
        if old_window.records[int(descriptor["user_id"])].history is not None
    }
    plans, plan_checks = stage49._plan_edge(
        old_window,
        target_window,
        manifest["records"],
        previous_ids,
        previous_ids,
    )
    last_exact = {record_id: 0 for record_id in previous_ids}
    spec = stage49._candidate_spec(args.candidate)
    selection, scheduler_checks = stage49._select_actions(
        plans,
        last_exact,
        0,
        1,
        spec,
        None,
    )
    migrate_id, exact_id = select_real_edge_records(
        plans,
        selection,
        manifest["records"],
    )
    requests = build_requests(migrate_id, exact_id, plans)
    source_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        0,
        device,
    )
    old_records = stage49._records_for_ids(
        (migrate_id,),
        record_by_id,
        old_window,
    )
    old_batch = base._history_batch(
        old_records,
        cfg.max_seq_len,
        device,
        prefix=False,
    )
    old_cache, _ = base._exact_full_batch(
        source_model,
        old_batch,
        (migrate_id,),
        0,
    )
    if (
        old_cache.migration_anchor_version != "theta0"
        or old_cache.served_kv_target != "theta0"
    ):
        raise RuntimeError("Stage 5 real-edge source cache version differs")
    del source_model, old_batch
    gc.collect()
    torch.cuda.empty_cache()
    operator = DirectOldKVFusedOperator(**LAUNCH)
    program, loaded_program, program_cpu = base._load_program(
        args,
        cfg,
        compiler,
        0,
        device,
        operator,
    )
    del program_cpu
    target_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        1,
        device,
    )
    compiler_descriptor = compiler["pairs"][0]["direct_program"]
    program_path = direct_program_path(args.runtime_dir, 0, 1)
    expected_program_sha256 = str(compiler_descriptor["sha256"])
    expected_program_shape = tuple(
        int(value) for value in compiler_descriptor["weights_shape"]
    )
    observed_program_shape = tuple(int(value) for value in program.weights.shape)
    artifact_started = time.perf_counter()
    observed_program_sha256 = sha256(program_path)
    observed_checkpoint_hashes = {
        value["version"]: sha256(_repo_path(value["path"]))
        for value in checkpoints[:2]
    }
    observed_compiler_sha256 = sha256(compiler_path)
    artifact_seconds = time.perf_counter() - artifact_started
    edge_expected_sha256, edge_observed_sha256 = _edge_identity(
        checkpoints,
        observed_checkpoint_hashes,
        baseline["source_artifacts"]["stage4_7_compiler"],
        observed_compiler_sha256,
    )
    canary_config, canary_config_sha256 = load_canary_config(
        args.canary_config
    )
    cases = {
        "normal": _run_case(
            "normal",
            False,
            device,
            requests,
            plans,
            record_by_id,
            old_window,
            target_window,
            cfg,
            target_model,
            operator,
            program,
            old_cache,
            edge_expected_sha256,
            edge_observed_sha256,
            expected_program_sha256,
            observed_program_sha256,
            expected_program_shape,
            observed_program_shape,
            canary_config,
            canary_config_sha256,
            artifact_seconds,
        ),
        "semantic_fallback": _run_case(
            "semantic-fallback",
            True,
            device,
            requests,
            plans,
            record_by_id,
            old_window,
            target_window,
            cfg,
            target_model,
            operator,
            program,
            old_cache,
            edge_expected_sha256,
            edge_observed_sha256,
            expected_program_sha256,
            observed_program_sha256,
            expected_program_shape,
            observed_program_shape,
            canary_config,
            canary_config_sha256,
            artifact_seconds,
        ),
    }
    checks = {
        "causality": all(window_checks.values()),
        "compiler": all(compiler_checks.values()),
        "provenance": all(provenance_checks.values()),
        "retained_plan": all(plan_checks.values()),
        "scheduler": all(scheduler_checks.values()),
        "real_source_cache": old_cache.k.device == device,
        "real_target_model": next(target_model.parameters()).device == device,
        "real_program": program.device == device,
        "normal_passed": all(cases["normal"]["checks"].values()),
        "semantic_fallback_passed": all(
            cases["semantic_fallback"]["checks"].values()
        ),
        "formal_result_not_written": True,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Stage 5 real-edge smoke failed: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    payload = {
        "protocol": PROTOCOL,
        "status": "real_edge_gpu_smoke_passed",
        "scientific_result": False,
        "formal_result_written": False,
        "configuration": {
            "dataset": "KuaiRand-1K",
            "split": "4+12",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "source_version": 0,
            "target_version": 1,
            "candidate": spec.to_dict(),
        },
        "selected_records": {
            "migrate_id": migrate_id,
            "migrate_role": record_by_id[migrate_id]["evaluation_role"],
            "scheduled_exact_id": exact_id,
            "scheduled_exact_role": record_by_id[exact_id][
                "evaluation_role"
            ],
        },
        "input_provenance": {
            "prepared_data_sha256": sha256(_repo_path(args.prepared_data)),
            "training_result_sha256": sha256(
                _repo_path(args.training_result)
            ),
            "compiler_result_sha256": sha256(compiler_path),
            "source_checkpoint_sha256": checkpoints[0]["sha256"],
            "target_checkpoint_sha256": checkpoints[1]["sha256"],
            "edge_artifact_sha256": edge_observed_sha256,
            "program_sha256": observed_program_sha256,
            "program_shape": list(observed_program_shape),
            "program_protocol": loaded_program["protocol"],
            "canary_config_path": str(_repo_path(args.canary_config)),
            "canary_config_sha256": canary_config_sha256,
        },
        "cases": cases,
        "checks": checks,
    }
    del old_cache, program, target_model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def main() -> None:
    args = parse_args()
    device = validate_args(args)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    seed_everything(args.seed)
    payload = run_smoke(args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
