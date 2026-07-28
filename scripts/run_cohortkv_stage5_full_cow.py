from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType

import cohortkv_stage4_8_sweep_common as stage48
import freeze_cohortkv_single_config_v1 as freeze
import run_cohortkv_stage4_7_organic_chain as base
import run_cohortkv_stage4_9_formal_confirmation as stage49_formal
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
    STAGE5_CLOSURE_PROTOCOL,
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
    select_jagged_rows,
    tail_slice_jagged_cache,
)
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    DirectOldKVProgram,
)
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_organic_windows,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cohortkv_single_config_stage5_full_cow_integration_v1"
STATIC_PROTOCOL = "cohortkv_single_config_stage5_full_cow_static_v1"
OUTPUT = (
    ROOT
    / "results/system/cohortkv_single_config_full_chain_v1"
    / "stage5_full_cow_theta0_theta1_seed0.json"
)
STAGE49_SUMMARY = (
    ROOT
    / "results/system/cohortkv_single_config_full_chain_v1"
    / "stage4_9_same_device_confirmation_seed0.json"
)
RESULT_SCHEMA = (
    ROOT / "configs/cohortkv_single_config_v1/result.schema.json"
)
CANARY_CONFIG = (
    ROOT / "configs/cohortkv_single_config_v1/stage5_formal_canary.json"
)
BATCH_SIZE = 4
GPU_COUNT = 2
EXPECTED_RECORDS = 682
CANARY_RECORDS = 4
CANDIDATES = tuple(stage49.CANDIDATES)
IMPLEMENTATION_PATHS = {
    "runner": Path(__file__).resolve(),
    "stage5_closure": ROOT
    / "src/hstu_kvcache/migration/stage5_closure.py",
    "destination": ROOT
    / "src/hstu_kvcache/migration/destination.py",
    "rollout_abi": ROOT / "src/hstu_kvcache/migration/rollout.py",
    "stage4_9_runner": ROOT
    / "scripts/run_cohortkv_stage4_9_formal_confirmation.py",
    "direct_oldkv": ROOT
    / "src/hstu_kvcache/migration/stage45_oldkv.py",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs=GPU_COUNT)
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--stage4-9-summary", type=Path, default=STAGE49_SUMMARY)
    parser.add_argument("--prepared-data", default=PREPARED_PATH)
    parser.add_argument("--training-result", default=TRAINING_PATH)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--compiler-result", default=COMPILER_OUTPUT)
    parser.add_argument("--runtime-dir", default=RUNTIME_DIR)
    parser.add_argument("--baseline", default=stage48.BASELINE_PATH)
    parser.add_argument("--result-schema", type=Path, default=RESULT_SCHEMA)
    parser.add_argument("--canary-config", type=Path, default=CANARY_CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[torch.device, ...]:
    if args.seed != 0 or args.batch_size != BATCH_SIZE:
        raise ValueError("Stage 5 freezes seed 0 and batch size 4")
    if args.smoke_test:
        if args.devices is not None:
            raise ValueError("Stage 5 static smoke does not accept --devices")
        if args.candidate is None:
            raise ValueError("Stage 5 static smoke requires --candidate")
        return ()
    if args.devices is None:
        raise ValueError("Stage 5 formal COW requires two explicit CUDA devices")
    devices = tuple(torch.device(value) for value in args.devices)
    if (
        len(set(devices)) != GPU_COUNT
        or any(
            value.type != "cuda"
            or value.index is None
            or value.index >= torch.cuda.device_count()
            for value in devices
        )
    ):
        raise ValueError(
            "Stage 5 formal COW requires two distinct available CUDA indices"
        )
    return devices


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


def _repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def implementation_snapshot() -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for name, path in IMPLEMENTATION_PATHS.items()
    }


def _atomic_save_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_formal_candidate(
    summary: dict[str, object],
    requested: str | None,
) -> str:
    selected = summary.get("selected_candidate")
    if selected is not None and selected not in CANDIDATES:
        raise ValueError("Stage 4.9 selected candidate is unsupported")
    if requested is not None and selected is not None and requested != selected:
        raise ValueError("CLI candidate differs from Stage 4.9 selection")
    candidate = requested if requested is not None else selected
    if candidate is None:
        raise ValueError(
            "Stage 4.9 summary has no selected_candidate; pass --candidate"
        )
    return str(candidate)


def load_formal_confirmation(
    path: str | Path,
    requested_candidate: str | None,
) -> tuple[str, dict[str, object], dict[str, object]]:
    source = _repo_path(path)
    summary = json.loads(source.read_text())
    candidate = resolve_formal_candidate(summary, requested_candidate)
    entries = {
        str(value["candidate_name"]): value
        for value in summary.get("results", [])
    }
    checks = {
        "protocol": summary.get("protocol") == stage49_formal.PROTOCOL,
        "status": summary.get("status") == "complete",
        "scientific_result": summary.get("scientific_result") is True,
        "summary_checks": summary.get("checks", {}).get("all_passed") is True,
        "candidate_present": candidate in entries,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage 4.9 formal summary is invalid: {checks}")
    descriptor = entries[candidate]
    result_path = _repo_path(str(descriptor["path"]))
    if sha256(result_path) != descriptor.get("sha256"):
        raise ValueError("Stage 4.9 candidate artifact hash differs")
    result = json.loads(result_path.read_text())
    current_implementation = stage49_formal.implementation_snapshot()
    result_checks = {
        "protocol": result.get("protocol") == stage49_formal.PROTOCOL,
        "status": result.get("status") == "complete",
        "scientific_result": result.get("scientific_result") is True,
        "candidate": result.get("candidate_name") == candidate,
        "all_checks": result.get("checks", {}).get("all_passed") is True,
        "implementation": result.get("implementation")
        == current_implementation,
        "eleven_edges": len(result.get("steps", [])) == 11,
        "first_edge": bool(result.get("steps"))
        and result["steps"][0].get("source_version") == 0
        and result["steps"][0].get("target_version") == 1,
        "post_migration_append": result.get("measurement_boundary", {}).get(
            "recursive_state"
        )
        == "previous_actual_post_append_mixed_cache",
        "no_old_denominator": result.get("measurement_boundary", {}).get(
            "old_exact_denominator_reused"
        )
        is False,
    }
    if not all(result_checks.values()):
        raise ValueError(
            f"Stage 4.9 candidate confirmation is invalid: {result_checks}"
        )
    return candidate, summary, result


def place_groups_two_gpu(
    groups,
    old_tokens: dict[int, int],
    final_tokens: dict[int, int],
) -> tuple[int, ...]:
    loads = [0, 0]
    placement = []
    for group in groups:
        record_ids = tuple(int(value["record_id"]) for value in group)
        weight = sum(
            old_tokens.get(record_id, 0)
            + final_tokens.get(record_id, 0)
            for record_id in record_ids
        )
        device_index = min(range(GPU_COUNT), key=lambda value: (loads[value], value))
        loads[device_index] += weight
        placement.append(device_index)
    if not placement or set(placement) != {0, 1}:
        raise ValueError("Stage 5 placement does not use both GPUs")
    return tuple(placement)


def _category(record_id: int, plan, selection) -> str:
    if record_id in set(selection.migrate_ids):
        return "migration"
    if record_id in set(selection.scheduled_exact_ids):
        return "scheduled-exact"
    if record_id not in set(selection.natural_exact_ids):
        raise ValueError("Stage 5 record is outside scheduler actions")
    if plan.timed_retained_rebuild:
        return "natural-missing-exact"
    if plan.target_prefix_tokens > 0:
        return "natural-prefix-exact"
    return "natural-short-exact"


def build_requests(
    plans,
    selection,
    previous_expected_ids: set[int],
) -> tuple[Stage5RecordRequest, ...]:
    requests = []
    for record_id in sorted(plans):
        plan = plans[record_id]
        if plan.final_tokens < 1:
            continue
        cohort_id = _category(record_id, plan, selection)
        requested_action = "migrate" if cohort_id == "migration" else "exact"
        if cohort_id in {
            "migration",
            "scheduled-exact",
            "natural-missing-exact",
        }:
            retained_tokens = int(plan.retained_tokens)
        elif cohort_id == "natural-prefix-exact":
            retained_tokens = int(plan.target_prefix_tokens)
        else:
            retained_tokens = 0
        requests.append(
            Stage5RecordRequest(
                record_id=record_id,
                cohort_id=cohort_id,
                requested_action=requested_action,
                source_version="theta0",
                target_version="theta1",
                last_exact_version=(
                    "theta0" if record_id in previous_expected_ids else None
                ),
                migration_depth=0,
                requested_reason={
                    "migration": "stage4_9_scheduler_migrate",
                    "scheduled-exact": "stage4_9_scheduled_exact",
                    "natural-missing-exact": "missing_expected_cache_exact",
                    "natural-prefix-exact": "natural_prefix_exact",
                    "natural-short-exact": "natural_latest_only_exact",
                }[cohort_id],
                retained_tokens=retained_tokens,
                final_tokens=int(plan.final_tokens),
            )
        )
    return tuple(requests)


def validate_formal_edge_binding(
    result: dict[str, object],
    selection,
    requests: tuple[Stage5RecordRequest, ...],
) -> dict[str, bool]:
    first = result["steps"][0]
    actions = Counter(
        {
            "migrate": sum(
                value.requested_action == "migrate" for value in requests
            ),
            "scheduled_exact": sum(
                value.cohort_id == "scheduled-exact" for value in requests
            ),
            "natural_exact": sum(
                value.cohort_id.startswith("natural-") for value in requests
            ),
        }
    )
    checks = {
        "scheduled_ids": tuple(
            int(value)
            for value in first["scheduler"]["scheduled_exact_ids"]
        )
        == tuple(int(value) for value in selection.scheduled_exact_ids),
        "migrate_count": actions["migrate"] == first["actions"]["migrate"],
        "scheduled_count": actions["scheduled_exact"]
        == first["actions"]["scheduled_exact"],
        "natural_count": actions["natural_exact"]
        == first["actions"]["natural_exact"],
        "resident_count": len(requests) == first["actions"]["resident_records"],
    }
    if not all(checks.values()):
        raise ValueError(f"Stage 5 edge actions differ from Stage 4.9: {checks}")
    return checks


def validate_closure_artifact(
    closure: dict[str, object],
    schema_path: str | Path,
    jsonschema_module: ModuleType | None = None,
) -> dict[str, object]:
    module = (
        importlib.import_module("jsonschema")
        if jsonschema_module is None
        else jsonschema_module
    )
    source = _repo_path(schema_path)
    schema = json.loads(source.read_text())
    closure_schema = schema["properties"]["stage5_closure"]
    module.validate(instance=closure, schema=closure_schema)
    freeze.validate_stage5_closure_semantics(closure)
    try:
        recorded_path = str(source.relative_to(ROOT))
    except ValueError:
        recorded_path = str(source)
    return {
        "schema_path": recorded_path,
        "schema_sha256": sha256(source),
        "validation_scope": "stage5_closure_subschema",
        "jsonschema_validated": True,
        "cross_field_validator": (
            "freeze_cohortkv_single_config_v1."
            "validate_stage5_closure_semantics"
        ),
        "cross_field_validated": True,
    }


def smoke_payload(args: argparse.Namespace) -> dict[str, object]:
    groups = tuple(
        tuple({"record_id": start + offset} for offset in range(4))
        for start in range(0, 12, 4)
    )
    placement = place_groups_two_gpu(
        groups,
        {value: 10 + value for value in range(12)},
        {value: 11 + value for value in range(12)},
    )
    canary = json.loads(CANARY_CONFIG.read_text())
    canary_ids = canary["record_selection"]["candidate_record_ids"][
        args.candidate
    ]
    threshold = canary["threshold"]
    derived_threshold = (
        math.floor(
            float(threshold["calibration_observation"])
            * float(threshold["calibration_multiple"])
            * 10.0
        )
        + 1
    ) / 10.0
    checks = {
        "candidate_known": args.candidate in CANDIDATES,
        "two_gpu_placement": set(placement) == {0, 1},
        "formal_summary_required_for_scientific_mode": True,
        "schema_exists": _repo_path(args.result_schema).is_file(),
        "formal_canary_config_exists": CANARY_CONFIG.is_file(),
        "formal_canary_config": (
            canary["protocol"]
            == "cohortkv_single_config_stage5_formal_canary_v1"
            and canary["selection_role"] == "program_selection"
            and canary["labels_used"] is False
            and len(canary_ids) == CANARY_RECORDS
            and _canonical_sha256(canary_ids)
            == canary["record_selection"]["record_ids_sha256"]
            and float(threshold["maximum_relative_l2"])
            == derived_threshold
            and all(
                sha256(_repo_path(value["path"])) == value["sha256"]
                for value in canary["calibration_artifacts"].values()
            )
        ),
        "full_population_contract": EXPECTED_RECORDS == freeze.EXPECTED_RECORDS,
        "four_cases": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 5 static smoke failed: {checks}")
    return {
        "protocol": STATIC_PROTOCOL,
        "status": "smoke_passed",
        "scientific_result": False,
        "formal_result_written": False,
        "candidate": stage49._candidate_spec(args.candidate).to_dict(),
        "synthetic_group_placement": list(placement),
        "implementation": implementation_snapshot(),
        "checks": checks,
    }


def _edge_identity(
    checkpoints: list[dict],
    observed_checkpoint_hashes: dict[str, str],
    compiler_descriptor: dict,
    observed_compiler_sha256: str,
) -> tuple[str, str]:
    expected = _canonical_sha256(
        {
            "source_checkpoint_sha256": checkpoints[0]["sha256"],
            "target_checkpoint_sha256": checkpoints[1]["sha256"],
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


def _formal_input_binding(
    args: argparse.Namespace,
    result: dict[str, object],
    compiler_path: Path,
) -> dict[str, bool]:
    provenance = result["input_provenance"]
    return {
        "prepared_data": provenance["prepared_data"]["sha256"]
        == sha256(_repo_path(args.prepared_data)),
        "training_result": provenance["training_result"]["sha256"]
        == sha256(_repo_path(args.training_result)),
        "compiler": provenance["compiler"]["sha256"] == sha256(compiler_path),
        "seed": result["configuration"]["seed"] == args.seed,
        "batch_size": result["configuration"]["batch_size"] == args.batch_size,
        "records": result["configuration"]["records"] == EXPECTED_RECORDS,
    }


def load_edge_inputs(
    args: argparse.Namespace,
    devices: tuple[torch.device, ...],
    candidate_result: dict[str, object],
) -> dict[str, object]:
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
    checks = {
        "windows": all(base.validate_windows(windows, manifest).values()),
        "compiler": all(
            base.validate_compiler_payload(
                compiler,
                manifest,
                windows,
                checkpoints,
            ).values()
        ),
        "runtime_provenance": all(
            stage48.validate_runtime_provenance(
                args,
                baseline,
                metadata,
                training,
                manifest,
                checkpoints,
                windows,
                compiler,
            ).values()
        ),
        "formal_input_binding": all(
            _formal_input_binding(
                args,
                candidate_result,
                compiler_path,
            ).values()
        ),
        "record_count": len(manifest["records"]) == EXPECTED_RECORDS,
        "window_count": len(windows) == 12,
        "device_class": all(
            torch.cuda.get_device_name(value)
            == baseline["configuration"]["device_class"]
            for value in devices
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Stage 5 input validation failed: {checks}")
    record_by_id = {
        int(value["record_id"]): value for value in manifest["records"]
    }
    old_window, target_window = windows[:2]
    previous_expected_ids = {
        record_id
        for record_id, descriptor in record_by_id.items()
        if old_window.records[int(descriptor["user_id"])].history is not None
    }
    plans, plan_checks = stage49._plan_edge(
        old_window,
        target_window,
        manifest["records"],
        previous_expected_ids,
        set(previous_expected_ids),
    )
    spec = stage49._candidate_spec(candidate_result["candidate_name"])
    selection, scheduler_checks = stage49._select_actions(
        plans,
        {record_id: 0 for record_id in previous_expected_ids},
        0,
        1,
        spec,
        None,
    )
    requests = build_requests(plans, selection, previous_expected_ids)
    action_checks = validate_formal_edge_binding(
        candidate_result,
        selection,
        requests,
    )
    category_counts = Counter(value.cohort_id for value in requests)
    edge_checks = {
        "plan": all(plan_checks.values()),
        "scheduler": all(scheduler_checks.values()),
        "request_coverage": len(requests) == EXPECTED_RECORDS,
        "old_coverage": len(previous_expected_ids) == EXPECTED_RECORDS,
        "migrate_present": category_counts["migration"] > 0,
        "scheduled_exact_present": category_counts["scheduled-exact"] > 0,
        "natural_exact_present": sum(
            count
            for name, count in category_counts.items()
            if name.startswith("natural-")
        )
        > 0,
        "formal_action_binding": all(action_checks.values()),
    }
    if not all(edge_checks.values()):
        raise ValueError(f"Stage 5 edge population differs: {edge_checks}")
    groups = base.fixed_record_groups(manifest, args.batch_size)
    placement = place_groups_two_gpu(
        groups,
        {
            record_id: int(value.old_tokens)
            for record_id, value in plans.items()
        },
        {
            record_id: int(value.final_tokens)
            for record_id, value in plans.items()
        },
    )
    record_device_index = {
        int(descriptor["record_id"]): device_index
        for group, device_index in zip(groups, placement, strict=True)
        for descriptor in group
    }
    planned_extents = tuple(
        tuple(
            int(value["record_id"])
            for value in group
            if plans[int(value["record_id"])].final_tokens > 0
        )
        for group in groups
    )
    planned_extents = tuple(value for value in planned_extents if value)
    return {
        "baseline": baseline,
        "metadata": metadata,
        "training": training,
        "cfg": cfg,
        "manifest": manifest,
        "checkpoints": checkpoints,
        "compiler": compiler,
        "compiler_path": compiler_path,
        "prepared_data_path": _repo_path(args.prepared_data),
        "training_result_path": _repo_path(args.training_result),
        "windows": windows,
        "old_window": old_window,
        "target_window": target_window,
        "record_by_id": record_by_id,
        "previous_expected_ids": previous_expected_ids,
        "plans": plans,
        "selection": selection,
        "requests": requests,
        "groups": groups,
        "placement": placement,
        "record_device_index": record_device_index,
        "planned_extents": planned_extents,
        "checks": {**checks, **edge_checks},
        "action_binding": action_checks,
    }


def load_formal_canary_config(
    path: str | Path,
    candidate: str,
    shared: dict[str, object],
) -> tuple[dict[str, object], str]:
    source = _repo_path(path)
    value = json.loads(source.read_text())
    record_ids = tuple(
        int(item)
        for item in value["record_selection"]["candidate_record_ids"][
            candidate
        ]
    )
    role_by_id = {
        int(item["record_id"]): str(item["evaluation_role"])
        for item in shared["manifest"]["records"]
    }
    calibration = value["calibration_artifacts"]
    canonical = value["canonical_artifacts"]
    threshold = value["threshold"]
    observed = float(threshold["calibration_observation"])
    multiple = float(threshold["calibration_multiple"])
    expected_threshold = (
        math.floor(observed * multiple * 10.0) + 1
    ) / 10.0
    program_selection_migrants: dict[int, list[int]] = defaultdict(list)
    for record_id in shared["selection"].migrate_ids:
        if role_by_id.get(record_id) == "program_selection":
            program_selection_migrants[
                shared["record_device_index"][record_id]
            ].append(record_id)
    canary_device_index = min(
        program_selection_migrants,
        key=lambda value: (
            -len(program_selection_migrants[value]),
            value,
        ),
    )
    expected_record_ids = tuple(
        sorted(program_selection_migrants[canary_device_index])[
            :CANARY_RECORDS
        ]
    )
    checks = {
        "protocol": value.get("protocol")
        == "cohortkv_single_config_stage5_formal_canary_v1",
        "status": value.get("status") == "frozen_before_formal_stage5_run",
        "versions": value.get("source_version") == "theta0"
        and value.get("target_version") == "theta1",
        "role": value.get("selection_role") == "program_selection",
        "labels": value.get("labels_used") is False
        and threshold.get("recommendation_labels_used") is False,
        "metric": value.get("metric") == "kv_relative_l2",
        "records": len(record_ids) == CANARY_RECORDS
        and len(set(record_ids)) == CANARY_RECORDS
        and all(role_by_id.get(item) == "program_selection" for item in record_ids)
        and set(record_ids).issubset(shared["selection"].migrate_ids),
        "record_rule": record_ids == expected_record_ids,
        "record_hash": _canonical_sha256(list(record_ids))
        == value["record_selection"]["record_ids_sha256"],
        "single_device": len(
            {
                shared["record_device_index"][record_id]
                for record_id in record_ids
            }
        )
        == 1,
        "threshold_rule": math.isclose(
            float(threshold["maximum_relative_l2"]),
            expected_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "calibration_config": sha256(
            _repo_path(calibration["config"]["path"])
        )
        == calibration["config"]["sha256"],
        "calibration_result": sha256(
            _repo_path(calibration["result"]["path"])
        )
        == calibration["result"]["sha256"],
        "prepared_data": canonical["prepared_data_sha256"]
        == sha256(shared["prepared_data_path"]),
        "training": canonical["training_result_sha256"]
        == sha256(shared["training_result_path"]),
        "compiler": canonical["compiler_sha256"]
        == sha256(shared["compiler_path"]),
        "source_checkpoint": canonical["source_checkpoint_sha256"]
        == shared["checkpoints"][0]["sha256"],
        "target_checkpoint": canonical["target_checkpoint_sha256"]
        == shared["checkpoints"][1]["sha256"],
        "program_file": canonical["program_file_sha256"]
        == shared["compiler"]["pairs"][0]["direct_program"]["sha256"],
        "perturbation": value["semantic_injection"]["shape_preserved"]
        is True
        and value["semantic_injection"]["dtype_preserved"] is True
        and value["semantic_injection"]["finite_required"] is True
        and value["semantic_injection"]["job_expected_hash_is_perturbed_hash"]
        is True,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage 5 formal canary config differs: {checks}")
    return value, sha256(source)


@torch.inference_mode()
def publish_old_exact(
    args: argparse.Namespace,
    shared: dict[str, object],
    devices: tuple[torch.device, ...],
    destination: HBMKVUpdateDestination,
    job_id: str,
):
    expected = tuple(sorted(shared["previous_expected_ids"]))
    transaction = destination.begin(
        job_id,
        "theta0",
        expected,
    )
    models = {
        device: load_checkpoint_model(
            shared["cfg"],
            args.checkpoint_dir,
            0,
            device,
        )
        for device in devices
    }
    started = time.perf_counter()
    initialization_ms = {str(value): 0.0 for value in devices}
    for group_index, (group, device_index) in enumerate(
        zip(shared["groups"], shared["placement"], strict=True)
    ):
        selected = [
            value
            for value in group
            if int(value["record_id"]) in shared["previous_expected_ids"]
        ]
        if not selected:
            continue
        device = devices[device_index]
        record_ids = tuple(int(value["record_id"]) for value in selected)
        records = [
            shared["old_window"].records[int(value["user_id"])]
            for value in selected
        ]
        with torch.cuda.device(device):
            batch = base._history_batch(
                records,
                shared["cfg"].max_seq_len,
                device,
                prefix=False,
            )
            torch.cuda.synchronize(device)
            gpu_started = time.perf_counter()
            full, hidden = base._exact_full_batch(
                models[device],
                batch,
                record_ids,
                0,
            )
            torch.cuda.synchronize(device)
            initialization_ms[str(device)] += (
                time.perf_counter() - gpu_started
            ) * 1000.0
        transaction.stage(f"old-{group_index:08d}", full)
        del batch, hidden, full
    old_manifest = transaction.commit()
    del models
    gc.collect()
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    checks = {
        "record_coverage": old_manifest.record_count == EXPECTED_RECORDS,
        "record_identity": set(old_manifest.record_ids) == set(expected),
        "two_gpu_distribution": {
            value.device for value in old_manifest.extents
        }
        == {str(value) for value in devices},
        "fp16": all(value.dtype == "float16" for value in old_manifest.extents),
        "theta0_anchor": all(
            value.migration_anchor_version == "theta0"
            for value in old_manifest.extents
        ),
        "source_inputs_released_after_stage": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 5 old source differs: {checks}")
    evidence = {
        "elapsed_seconds": time.perf_counter() - started,
        "initialization_gpu_ms": initialization_ms,
        "manifest": old_manifest.to_dict(),
        "checks": checks,
    }
    return old_manifest, evidence


def _load_old_records(
    destination: HBMKVUpdateDestination,
    old_manifest,
    record_ids: tuple[int, ...],
) -> JaggedMigratedKVBatch:
    extent_by_record = {
        record_id: extent
        for extent in old_manifest.extents
        for record_id in extent.record_ids
    }
    requested_by_extent: dict[str, list[int]] = defaultdict(list)
    for record_id in record_ids:
        requested_by_extent[extent_by_record[record_id].extent_id].append(
            record_id
        )
    record_batches = {}
    for extent_id, requested in requested_by_extent.items():
        source = destination.load_extent("theta0", extent_id)
        selected = select_jagged_rows(
            source,
            tuple(source.record_index(value) for value in requested),
        )
        record_batches.update(base._split_cache(selected))
        del source, selected
    return base._assemble_record_caches(record_ids, record_batches)


def _relabel_target(
    batch: JaggedMigratedKVBatch,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=batch.record_ids,
        migration_anchor_version="theta1",
        served_kv_target="theta1",
        k=batch.k,
        v=batch.v,
        lengths=batch.lengths,
        offsets=batch.offsets,
    )


def _direct_program_sha256(program: DirectOldKVProgram) -> str:
    digest = hashlib.sha256()
    digest.update(program.source_version.encode("utf-8"))
    digest.update(program.target_version.encode("utf-8"))
    for value in (program.weights, program.biases):
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _batch_bytes(
    records: int,
    tokens: int,
    extents: int,
    num_layers: int,
    kv_width: int,
) -> int:
    return (
        2
        * num_layers
        * tokens
        * kv_width
        * torch.float16.itemsize
        + records * torch.int64.itemsize
        + (records + extents) * torch.int64.itemsize
    )


def build_capacity(
    shared: dict[str, object],
    devices: tuple[torch.device, ...],
    target_models: dict[torch.device, object],
    programs: dict[torch.device, object],
    old_manifest=None,
) -> tuple[
    tuple[Stage5DeviceCapacity, ...],
    dict[str, dict[str, int]],
]:
    request_by_id = {
        value.record_id: value for value in shared["requests"]
    }
    old_bytes = {str(value): 0 for value in devices}
    maximum_old_extent_bytes = {str(value): 0 for value in devices}
    if old_manifest is not None:
        for extent in old_manifest.extents:
            old_bytes[str(extent.device)] += int(extent.payload_bytes)
            maximum_old_extent_bytes[str(extent.device)] = max(
                maximum_old_extent_bytes[str(extent.device)],
                int(extent.payload_bytes),
            )
    else:
        for group, device_index in zip(
            shared["groups"],
            shared["placement"],
            strict=True,
        ):
            record_ids = tuple(
                int(value["record_id"])
                for value in group
                if int(value["record_id"])
                in shared["previous_expected_ids"]
            )
            if not record_ids:
                continue
            device = devices[device_index]
            extent_bytes = _batch_bytes(
                len(record_ids),
                sum(
                    shared["plans"][value].old_tokens
                    for value in record_ids
                ),
                1,
                shared["cfg"].num_layers,
                shared["cfg"].num_heads * shared["cfg"].head_dim,
            )
            old_bytes[str(device)] += extent_bytes
            maximum_old_extent_bytes[str(device)] = max(
                maximum_old_extent_bytes[str(device)],
                extent_bytes,
            )
    new_records = {str(value): 0 for value in devices}
    new_tokens = {str(value): 0 for value in devices}
    new_extents = {str(value): 0 for value in devices}
    maximum_extent_bytes = {str(value): 0 for value in devices}
    for extent_ids in shared["planned_extents"]:
        grouped: dict[tuple[str, str], list[int]] = {}
        order = []
        for record_id in extent_ids:
            request = request_by_id[record_id]
            key = (request.cohort_id, request.requested_action)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(record_id)
        for key in order:
            record_ids = tuple(grouped[key])
            device = devices[shared["record_device_index"][record_ids[0]]]
            if any(
                devices[shared["record_device_index"][record_id]] != device
                for record_id in record_ids
            ):
                raise RuntimeError("Stage 5 target extent crosses devices")
            name = str(device)
            tokens = sum(
                request_by_id[record_id].final_tokens
                for record_id in record_ids
            )
            new_records[name] += len(record_ids)
            new_tokens[name] += tokens
            new_extents[name] += 1
            maximum_extent_bytes[name] = max(
                maximum_extent_bytes[name],
                _batch_bytes(
                    len(record_ids),
                    tokens,
                    1,
                    shared["cfg"].num_layers,
                    shared["cfg"].num_heads * shared["cfg"].head_dim,
                ),
            )
    capacities = []
    observations = {}
    for device in devices:
        name = str(device)
        total = torch.cuda.get_device_properties(device).total_memory
        model_bytes = sum(
            value.numel() * value.element_size()
            for value in (
                *target_models[device].parameters(),
                *target_models[device].buffers(),
            )
        )
        complete_new = _batch_bytes(
            new_records[name],
            new_tokens[name],
            new_extents[name],
            shared["cfg"].num_layers,
            shared["cfg"].num_heads * shared["cfg"].head_dim,
        )
        transient_extent = max(
            maximum_extent_bytes[name],
            maximum_old_extent_bytes[name],
        )
        source_setup_model_bytes = model_bytes
        transient = max(
            1,
            source_setup_model_bytes + transient_extent * 8,
        )
        margin = max(1, total // 20)
        capacity = Stage5DeviceCapacity(
            device=name,
            model_and_program_bytes=model_bytes + programs[device].nbytes,
            old_kv_bytes=old_bytes[name],
            complete_new_kv_bytes=complete_new,
            transient_bytes=transient,
            allocator_margin_bytes=margin,
            capacity_bytes=total,
        )
        if not capacity.passed:
            raise MemoryError(f"Stage 5 COW capacity fails on {name}")
        free_bytes, observed_total = torch.cuda.mem_get_info(device)
        if observed_total != total:
            raise RuntimeError("Stage 5 CUDA capacity observations differ")
        required_free = (
            old_bytes[name]
            + complete_new
            + transient
            + margin
        )
        free_capacity_passed = free_bytes >= required_free
        if not free_capacity_passed:
            raise MemoryError(
                f"Stage 5 observed free HBM is insufficient on {name}"
            )
        observations[name] = {
            "free_bytes_before_jobs": int(free_bytes),
            "required_free_bytes_before_jobs": int(required_free),
            "observed_free_capacity_passed": free_capacity_passed,
            "allocated_bytes_before_jobs": int(
                torch.cuda.memory_allocated(device)
            ),
            "reserved_bytes_before_jobs": int(
                torch.cuda.memory_reserved(device)
            ),
            "old_manifest_bytes": old_bytes[name],
            "derived_complete_new_bytes": complete_new,
            "derived_maximum_extent_bytes": maximum_extent_bytes[name],
            "derived_maximum_old_extent_bytes": (
                maximum_old_extent_bytes[name]
            ),
            "stage_and_load_clone_peak_in_transient": True,
            "transient_extent_multiplier": 8,
            "source_setup_model_bytes_in_transient": (
                source_setup_model_bytes
            ),
            "source_setup_model_released_before_preflight": True,
            "target_records": new_records[name],
            "target_tokens": new_tokens[name],
            "target_extents": new_extents[name],
        }
        capacities.append(capacity)
    return tuple(capacities), observations


@torch.inference_mode()
def observe_case_canary(
    perturb: bool,
    shared: dict[str, object],
    devices: tuple[torch.device, ...],
    destination: HBMKVUpdateDestination,
    old_manifest,
    target_models: dict[torch.device, object],
    operators: dict[torch.device, DirectOldKVFusedOperator],
    programs: dict[torch.device, object],
    program_sha256: str,
):
    record_ids = tuple(
        int(value)
        for value in shared["canary_config"]["record_selection"][
            "candidate_record_ids"
        ][shared["candidate_name"]]
    )
    device_index = shared["record_device_index"][record_ids[0]]
    device = devices[device_index]
    started = time.perf_counter()
    with torch.cuda.device(device):
        canary_program = programs[device]
        canary_program_sha256 = program_sha256
        if perturb:
            perturbed_biases = canary_program.biases.clone()
            perturbed_biases.add_(1024.0)
            canary_program = DirectOldKVProgram(
                source_version=canary_program.source_version,
                target_version=canary_program.target_version,
                weights=canary_program.weights.clone(),
                biases=perturbed_biases,
            )
            canary_program_sha256 = _direct_program_sha256(
                canary_program
            )
            if (
                canary_program_sha256
                != shared["canary_config"]["semantic_injection"][
                    "perturbed_program_memory_sha256"
                ]
            ):
                raise RuntimeError(
                    "Stage 5 perturbed program hash differs from config"
                )
        old = _load_old_records(destination, old_manifest, record_ids)
        sliced = tail_slice_jagged_cache(
            old,
            tuple(
                shared["plans"][value].retained_tokens
                for value in record_ids
            ),
        )
        if sliced.cache is None:
            raise RuntimeError("Stage 5 canary retained source is empty")
        candidate = execute_direct(
            operators[device],
            canary_program,
            sliced.cache,
            1,
        )
        retained_batch = stage49._retained_batch(
            record_ids,
            shared["plans"],
            shared["record_by_id"],
            shared["target_window"],
            device,
        )
        reference = stage49._exact_cache(
            target_models[device],
            retained_batch,
            record_ids,
            1,
            torch.float16,
        )
        observation = observe_semantic_canary(
            "migration",
            "theta0",
            "theta1",
            candidate,
            reference,
            float(
                shared["canary_config"]["threshold"][
                    "maximum_relative_l2"
                ]
            ),
            shared["canary_config_sha256"],
            canary_program_sha256,
        )
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    del old, sliced, candidate, retained_batch, reference, canary_program
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    return observation, elapsed, device, canary_program_sha256


def build_cohorts(
    shared: dict[str, object],
    old_manifest,
    present_ids: tuple[int, ...],
    capacities: tuple[Stage5DeviceCapacity, ...],
    canary,
    edge_expected_sha256: str,
    edge_observed_sha256: str,
    program_expected_sha256: str,
    program_observed_sha256: str,
    program_expected_shape: tuple[int, ...],
    program_observed_shape: tuple[int, ...],
    artifact_seconds: float,
    presence_seconds: float,
    capacity_seconds: float,
    canary_seconds: float,
) -> tuple[Stage5CohortPreflight, ...]:
    requests_by_cohort: dict[str, list[Stage5RecordRequest]] = defaultdict(list)
    for request in shared["requests"]:
        requests_by_cohort[request.cohort_id].append(request)
    migrate_ids = tuple(
        value.record_id for value in requests_by_cohort["migration"]
    )
    present = set(present_ids)
    cohorts = [
        Stage5CohortPreflight(
            cohort_id="migration",
            source_version="theta0",
            target_version="theta1",
            expected_artifact_sha256=edge_expected_sha256,
            observed_artifact_sha256=edge_observed_sha256,
            expected_program_sha256=program_expected_sha256,
            observed_program_sha256=program_observed_sha256,
            expected_program_shape=program_expected_shape,
            observed_program_shape=program_observed_shape,
            expected_threshold_artifact_sha256=shared[
                "canary_config_sha256"
            ],
            expected_old_record_ids=migrate_ids,
            present_old_record_ids=tuple(
                value for value in migrate_ids if value in present
            ),
            device_capacity=capacities,
            canary=canary,
            measurement=Stage5PreflightMeasurement(
                artifact_seconds=artifact_seconds,
                old_kv_presence_seconds=presence_seconds,
                capacity_seconds=capacity_seconds,
                semantic_canary_seconds=canary_seconds,
            ),
        )
    ]
    for cohort_id in sorted(set(requests_by_cohort) - {"migration"}):
        cohorts.append(
            Stage5CohortPreflight(
                cohort_id=cohort_id,
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
                measurement=Stage5PreflightMeasurement(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                migration_required=False,
            )
        )
    if set(requests_by_cohort) != {value.cohort_id for value in cohorts}:
        raise RuntimeError("Stage 5 cohort coverage differs")
    if old_manifest.record_count != EXPECTED_RECORDS:
        raise RuntimeError("Stage 5 old manifest is incomplete")
    return tuple(cohorts)


def _validate_event_sequence(events: list[dict[str, object]]) -> bool:
    if not events or len(events) % 3:
        return False
    for start in range(0, len(events), 3):
        retained, guard, append = events[start : start + 3]
        identity = (
            retained["record_ids"],
            retained["action"],
            retained["cohort_id"],
        )
        if (
            tuple(value["kind"] for value in (retained, guard, append))
            != ("retained", "guard", "append")
            or identity
            != (
                guard["record_ids"],
                guard["action"],
                guard["cohort_id"],
            )
            or identity
            != (
                append["record_ids"],
                append["action"],
                append["cohort_id"],
            )
        ):
            return False
    return True


def validate_target_manifest(
    destination: HBMKVUpdateDestination,
    committed,
    shared: dict[str, object],
    devices: tuple[torch.device, ...],
) -> dict[str, bool]:
    manifest = committed.destination_manifest
    observed_ids = []
    length_checks = []
    finite_checks = []
    device_checks = []
    for extent in manifest.extents:
        batch = destination.load_extent(
            manifest.target_version,
            extent.extent_id,
        )
        observed_ids.extend(batch.record_ids)
        length_checks.extend(
            int(length) == shared["plans"][record_id].final_tokens
            for record_id, length in zip(
                batch.record_ids,
                batch.lengths.detach().cpu().tolist(),
                strict=True,
            )
        )
        finite_checks.append(
            bool(torch.isfinite(batch.k).all())
            and bool(torch.isfinite(batch.v).all())
        )
        device_checks.append(
            batch.k.device
            == devices[shared["record_device_index"][batch.record_ids[0]]]
            and all(
                shared["record_device_index"][record_id]
                == shared["record_device_index"][batch.record_ids[0]]
                for record_id in batch.record_ids
            )
        )
    return {
        "record_count": manifest.record_count == EXPECTED_RECORDS,
        "record_coverage": set(observed_ids)
        == {value.record_id for value in shared["requests"]},
        "record_unique": len(observed_ids) == len(set(observed_ids)),
        "lengths": all(length_checks),
        "finite": all(finite_checks),
        "two_gpu_distribution": {
            value.device for value in manifest.extents
        }
        == {str(value) for value in devices},
        "placement": all(device_checks),
        "target_version": all(
            value.migration_anchor_version == "theta1"
            and value.served_kv_target == "theta1"
            for value in manifest.extents
        ),
    }


@torch.inference_mode()
def run_case(
    name: str,
    perturb: bool,
    fault: str | None,
    args: argparse.Namespace,
    shared: dict[str, object],
    devices: tuple[torch.device, ...],
    target_models: dict[torch.device, object],
    operators: dict[torch.device, DirectOldKVFusedOperator],
    programs: dict[torch.device, object],
    capacities: tuple[Stage5DeviceCapacity, ...],
    edge_expected_sha256: str,
    edge_observed_sha256: str,
    program_expected_sha256: str,
    program_observed_sha256: str,
    program_expected_shape: tuple[int, ...],
    program_observed_shape: tuple[int, ...],
    artifact_seconds: float,
    capacity_seconds: float,
) -> tuple[dict[str, object], dict[str, object]]:
    destination = HBMKVUpdateDestination(
        devices,
        destination_id=f"stage5-cow-{name}",
    )
    old_manifest, source_evidence = publish_old_exact(
        args,
        shared,
        devices,
        destination,
        f"stage5-cow-{name}-old",
    )
    old_snapshot = (
        capture_manifest_snapshot(
            destination,
            old_manifest,
        )
        if fault is not None
        else None
    )
    presence_started = time.perf_counter()
    present_ids = manifest_present_record_ids(destination, old_manifest)
    presence_seconds = time.perf_counter() - presence_started
    (
        canary,
        canary_seconds,
        canary_device,
        case_program_sha256,
    ) = observe_case_canary(
        perturb,
        shared,
        devices,
        destination,
        old_manifest,
        target_models,
        operators,
        programs,
        program_observed_sha256,
    )
    cohorts = build_cohorts(
        shared,
        old_manifest,
        present_ids,
        capacities,
        canary,
        edge_expected_sha256,
        edge_observed_sha256,
        (
            case_program_sha256
            if perturb
            else program_expected_sha256
        ),
        case_program_sha256,
        program_expected_shape,
        program_observed_shape,
        artifact_seconds,
        presence_seconds,
        capacity_seconds,
        canary_seconds,
    )
    events: list[dict[str, object]] = []
    migration_calls = 0

    def retained_producer(record_ids, action, cohort_id):
        nonlocal migration_calls
        device = devices[shared["record_device_index"][record_ids[0]]]
        if any(
            devices[shared["record_device_index"][value]] != device
            for value in record_ids
        ):
            raise RuntimeError("Stage 5 execution group crosses devices")
        with torch.cuda.device(device):
            if action == "migrate":
                old = _load_old_records(
                    destination,
                    old_manifest,
                    record_ids,
                )
                retained = tail_slice_jagged_cache(
                    old,
                    tuple(
                        shared["plans"][value].retained_tokens
                        for value in record_ids
                    ),
                )
                if retained.cache is None:
                    raise RuntimeError("Stage 5 migrated retained cache is empty")
                cache = execute_direct(
                    operators[device],
                    programs[device],
                    retained.cache,
                    1,
                )
                migration_calls += 1
                program_sha256 = program_observed_sha256
                program_shape = program_observed_shape
            elif cohort_id == "natural-short-exact":
                cache = None
                program_sha256 = None
                program_shape = ()
            elif cohort_id == "natural-prefix-exact":
                cache = stage49_formal._build_natural_prefix_once(
                    target_models[device],
                    record_ids,
                    shared["plans"],
                    shared["record_by_id"],
                    shared["target_window"],
                    1,
                    shared["cfg"],
                    device,
                    torch.float16,
                )
                program_sha256 = None
                program_shape = ()
            else:
                batch = stage49._retained_batch(
                    record_ids,
                    shared["plans"],
                    shared["record_by_id"],
                    shared["target_window"],
                    device,
                )
                cache = stage49._exact_cache(
                    target_models[device],
                    batch,
                    record_ids,
                    1,
                    torch.float16,
                )
                program_sha256 = None
                program_shape = ()
        retained_lengths = tuple(
            next(
                value.retained_tokens
                for value in shared["requests"]
                if value.record_id == record_id
            )
            for record_id in record_ids
        )
        events.append(
            {
                "kind": "retained",
                "record_ids": list(record_ids),
                "action": action,
                "cohort_id": cohort_id,
                "device": str(device),
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
            num_layers=shared["cfg"].num_layers,
            kv_width=shared["cfg"].num_heads * shared["cfg"].head_dim,
            dtype="float16",
        )

    def guard(prepared, result):
        cache = prepared.retained_batch
        if (
            prepared.artifact_sha256 != edge_observed_sha256
            or (
                cache is not None
                and (
                    not bool(torch.isfinite(cache.k).all())
                    or not bool(torch.isfinite(cache.v).all())
                    or cache.served_kv_target != "theta1"
                )
            )
            or (
                prepared.action == "migrate"
                and (
                    not result.passed
                    or prepared.program_sha256 != case_program_sha256
                    or prepared.program_shape != program_observed_shape
                    or cache is None
                    or cache.migration_anchor_version != "theta0"
                )
            )
            or (
                prepared.action == "exact"
                and (
                    prepared.program_sha256 is not None
                    or prepared.program_shape
                    or (
                        cache is not None
                        and cache.migration_anchor_version != "theta1"
                    )
                )
            )
        ):
            raise RuntimeError("Stage 5 retained guard rejected an extent")
        events.append(
            {
                "kind": "guard",
                "record_ids": list(prepared.record_ids),
                "action": prepared.action,
                "cohort_id": prepared.cohort_id,
            }
        )

    def target_appender(prepared):
        device = devices[
            shared["record_device_index"][prepared.record_ids[0]]
        ]
        with torch.cuda.device(device):
            if prepared.cohort_id == "natural-short-exact":
                final, hidden = stage49_formal._append_fresh_latest_once(
                    target_models[device],
                    prepared.record_ids,
                    shared["record_by_id"],
                    shared["target_window"],
                    1,
                    shared["cfg"],
                    device,
                    torch.float16,
                )
            elif prepared.cohort_id == "natural-prefix-exact":
                if prepared.retained_batch is None:
                    raise RuntimeError("Stage 5 natural prefix is empty")
                final, hidden = stage49_formal._append_latest_once(
                    target_models[device],
                    prepared.retained_batch,
                    shared["record_by_id"],
                    shared["target_window"],
                    device,
                    torch.float16,
                )
            else:
                if prepared.retained_batch is None:
                    raise RuntimeError("Stage 5 retained cache is empty")
                after_delta = stage49_formal._append_delta_once(
                    target_models[device],
                    prepared.retained_batch,
                    shared["plans"],
                    shared["record_by_id"],
                    shared["target_window"],
                    device,
                    torch.float16,
                )
                final, hidden = stage49_formal._append_latest_once(
                    target_models[device],
                    after_delta,
                    shared["record_by_id"],
                    shared["target_window"],
                    device,
                    torch.float16,
                )
            final = _relabel_target(final)
            if (
                tuple(int(value) for value in final.lengths.detach().cpu())
                != tuple(
                    shared["plans"][value].final_tokens
                    for value in prepared.record_ids
                )
                or not bool(torch.isfinite(hidden).all())
            ):
                raise RuntimeError("Stage 5 post-append endpoint differs")
        events.append(
            {
                "kind": "append",
                "record_ids": list(prepared.record_ids),
                "action": prepared.action,
                "cohort_id": prepared.cohort_id,
                "device": str(device),
            }
        )
        return Stage5ProducedExtent(
            final,
            source_guard_hook=prepared.guard_hook,
        )

    report = run_stage5_job(
        job_id=f"stage5-cow-{name}",
        requests=shared["requests"],
        cohorts=cohorts,
        destination=destination,
        retained_producer=retained_producer,
        target_appender=target_appender,
        guard=guard,
        old_manifest=old_manifest,
        old_snapshot=old_snapshot,
        fault=fault,
        maximum_records_per_extent=BATCH_SIZE,
        planned_extents=shared["planned_extents"],
    )
    report_value = report.to_dict()
    checks = {
        "event_sequence": _validate_event_sequence(events),
        "old_manifest_complete": old_manifest.record_count == EXPECTED_RECORDS,
        "old_manifest_two_gpu": {
            value.device for value in old_manifest.extents
        }
        == {str(value) for value in devices},
        "semantic_expectation": canary.passed is (not perturb),
        "migration_execution": (
            migration_calls > 0
            if fault is None and not perturb
            else migration_calls == 0
            if perturb
            else migration_calls >= 0
        ),
    }
    if fault is None:
        if report.target_manifest is None:
            raise RuntimeError("Stage 5 committed case has no target manifest")
        target_checks = validate_target_manifest(
            destination,
            report.target_manifest,
            shared,
            devices,
        )
        checks.update(
            {
                "committed": report.outcome == "committed"
                and report.target_visible,
                "target_complete": all(target_checks.values()),
                "fallback_actions": (
                    not perturb
                    or all(
                        value.final_action == "exact"
                        for value in report.preflight.decisions
                    )
                ),
                "normal_mixed_actions": (
                    perturb
                    or {value.final_action for value in report.preflight.decisions}
                    == {"migrate", "exact"}
                ),
            }
        )
    else:
        target_checks = {}
        checks.update(
            {
                "aborted": report.outcome == "aborted",
                "target_invisible": not report.target_visible,
                "old_readback_complete": report.old_readback is not None
                and report.old_readback.passed
                and report.old_readback.expected_records == EXPECTED_RECORDS
                and report.old_readback.read_records == EXPECTED_RECORDS,
            }
        )
    if not all(checks.values()):
        raise RuntimeError(f"Stage 5 {name} case failed: {checks}")
    evidence = {
        "name": name,
        "semantic_perturbation": perturb,
        "fault": fault,
        "canary_device": str(canary_device),
        "canary": canary.to_dict(),
        "integrity_accepted_program": {
            "expected_sha256": case_program_sha256,
            "observed_sha256": case_program_sha256,
            "base_program_sha256": program_observed_sha256,
            "hash_kind": "canonical_loaded_program_tensor_content_v1",
            "serialized_program_file_sha256": shared["compiler"]["pairs"][0][
                "direct_program"
            ]["sha256"],
            "actual_perturbed_program_executed": perturb,
        },
        "migration_operator_calls_inside_job": migration_calls,
        "old_source_initialization": source_evidence,
        "events": events,
        "target_checks": target_checks,
        "checks": checks,
    }
    if perturb:
        report_value["semantic_perturbation_detected"] = not canary.passed
        report_value["affected_cohort_final_action"] = "exact"
    return report_value, evidence


@torch.inference_mode()
def run_formal(
    args: argparse.Namespace,
    devices: tuple[torch.device, ...],
) -> dict[str, object]:
    candidate, formal_summary, candidate_result = load_formal_confirmation(
        args.stage4_9_summary,
        args.candidate,
    )
    shared = load_edge_inputs(args, devices, candidate_result)
    canary_config, canary_config_sha256 = load_formal_canary_config(
        args.canary_config,
        candidate,
        shared,
    )
    shared["candidate_name"] = candidate
    shared["canary_config"] = canary_config
    shared["canary_config_sha256"] = canary_config_sha256
    compiler_descriptor = shared["compiler"]["pairs"][0]["direct_program"]
    program_path = direct_program_path(args.runtime_dir, 0, 1)
    artifact_started = time.perf_counter()
    observed_program_file_sha256 = sha256(program_path)
    observed_compiler_sha256 = sha256(shared["compiler_path"])
    observed_checkpoint_hashes = {
        value["version"]: sha256(_repo_path(value["path"]))
        for value in shared["checkpoints"][:2]
    }
    edge_expected_sha256, edge_observed_sha256 = _edge_identity(
        shared["checkpoints"],
        observed_checkpoint_hashes,
        shared["baseline"]["source_artifacts"]["stage4_7_compiler"],
        observed_compiler_sha256,
    )
    artifact_seconds = time.perf_counter() - artifact_started
    expected_program_file_sha256 = str(compiler_descriptor["sha256"])
    expected_program_shape = tuple(
        int(value) for value in compiler_descriptor["weights_shape"]
    )
    if (
        edge_expected_sha256 != edge_observed_sha256
        or expected_program_file_sha256 != observed_program_file_sha256
    ):
        raise ValueError("Stage 5 deployed artifacts differ from descriptors")
    operators = {
        device: DirectOldKVFusedOperator(**LAUNCH) for device in devices
    }
    programs = {}
    loaded_programs = {}
    for device in devices:
        with torch.cuda.device(device):
            program, loaded, program_cpu = base._load_program(
                args,
                shared["cfg"],
                shared["compiler"],
                0,
                device,
                operators[device],
            )
        programs[device] = program
        loaded_programs[str(device)] = loaded
        del program_cpu
    observed_program_shapes = {
        tuple(int(value) for value in program.weights.shape)
        for program in programs.values()
    }
    if (
        observed_program_shapes != {expected_program_shape}
        or any(
            program.source_version != "theta0"
            or program.target_version != "theta1"
            or program.device != device
            for device, program in programs.items()
        )
    ):
        raise ValueError("Stage 5 deployed programs differ across devices")
    program_memory_sha256_by_device = {
        str(device): _direct_program_sha256(program)
        for device, program in programs.items()
    }
    canonical_program_memory_sha256 = next(
        iter(program_memory_sha256_by_device.values())
    )
    if (
        set(program_memory_sha256_by_device.values())
        != {canonical_program_memory_sha256}
        or canonical_program_memory_sha256
        != canary_config["canonical_artifacts"][
            "program_memory_sha256"
        ]
    ):
        raise ValueError("Stage 5 canonical program memory hash differs")
    target_models = {
        device: load_checkpoint_model(
            shared["cfg"],
            args.checkpoint_dir,
            1,
            device,
        )
        for device in devices
    }
    capacity_started = time.perf_counter()
    capacities, capacity_observations = build_capacity(
        shared,
        devices,
        target_models,
        programs,
    )
    capacity_seconds = time.perf_counter() - capacity_started
    cases = {}
    case_evidence = {}
    case_specs = (
        ("normal", False, None),
        ("semantic-fallback", True, None),
        ("mid-job", False, "mid_job"),
        ("pre-commit", False, "pre_commit"),
    )
    for name, perturb, fault in case_specs:
        report, evidence = run_case(
            name,
            perturb,
            fault,
            args,
            shared,
            devices,
            target_models,
            operators,
            programs,
            capacities,
            edge_expected_sha256,
            edge_observed_sha256,
            canonical_program_memory_sha256,
            canonical_program_memory_sha256,
            expected_program_shape,
            next(iter(observed_program_shapes)),
            artifact_seconds,
            capacity_seconds,
        )
        cases[name] = report
        case_evidence[name] = evidence
        gc.collect()
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    closure = {
        "protocol": STAGE5_CLOSURE_PROTOCOL,
        "canary_artifact": {
            "path": str(_repo_path(args.canary_config).relative_to(ROOT)),
            "sha256": canary_config_sha256,
            "protocol": canary_config["protocol"],
            "source_version": canary_config["source_version"],
            "target_version": canary_config["target_version"],
            "selection_role": canary_config["selection_role"],
            "labels_used": canary_config["labels_used"],
            "metric": canary_config["metric"],
            "maximum_relative_l2": canary_config["threshold"][
                "maximum_relative_l2"
            ],
        },
        "copy_on_write_gpu_count": GPU_COUNT,
        "copy_on_write_capacity": {
            "mode": "copy_on_write",
            "old_extents_retained_until_commit": True,
            "all_devices_passed": all(value.passed for value in capacities),
            "devices": [value.to_dict() for value in capacities],
            "observed_free_capacity": {
                "measurement_boundary": (
                    "target models and direct programs resident; before "
                    "per-case old-cache publication"
                ),
                "all_devices_passed": all(
                    value["observed_free_capacity_passed"]
                    for value in capacity_observations.values()
                ),
                "devices": [
                    {
                        "device": device,
                        "free_bytes": value[
                            "free_bytes_before_jobs"
                        ],
                        "required_free_bytes": value[
                            "required_free_bytes_before_jobs"
                        ],
                        "passed": value[
                            "observed_free_capacity_passed"
                        ],
                    }
                    for device, value in capacity_observations.items()
                ],
            },
        },
        "normal_job": cases["normal"],
        "semantic_fallback_job": cases["semantic-fallback"],
        "abort_jobs": [
            cases["mid-job"],
            cases["pre-commit"],
        ],
    }
    validation = validate_closure_artifact(
        closure,
        args.result_schema,
    )
    checks = {
        "formal_stage4_9_gate": candidate_result["scientific_result"] is True,
        "formal_candidate": candidate_result["candidate_name"] == candidate,
        "all_inputs": all(shared["checks"].values()),
        "old_source": all(
            all(value["old_source_initialization"]["checks"].values())
            for value in case_evidence.values()
        ),
        "capacity": all(value.passed for value in capacities)
        and all(
            bool(value["observed_free_capacity_passed"])
            for value in capacity_observations.values()
        ),
        "normal": all(case_evidence["normal"]["checks"].values()),
        "semantic_fallback": all(
            case_evidence["semantic-fallback"]["checks"].values()
        ),
        "mid_job": all(case_evidence["mid-job"]["checks"].values()),
        "pre_commit": all(case_evidence["pre-commit"]["checks"].values()),
        "jsonschema": validation["jsonschema_validated"] is True,
        "cross_field": validation["cross_field_validated"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 5 formal closure failed: {checks}")
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": True,
        "study_stage": "single_configuration_seed0_development",
        "repository_commit": _repository_commit(),
        "configuration": {
            "dataset": "KuaiRand-1K",
            "split": "4+12",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "source_version": 0,
            "target_version": 1,
            "devices": [str(value) for value in devices],
            "device_names": [
                torch.cuda.get_device_name(value) for value in devices
            ],
            "records": EXPECTED_RECORDS,
            "candidate_name": candidate,
            "candidate": stage49._candidate_spec(candidate).to_dict(),
        },
        "formal_stage4_9_gate": {
            "summary_path": str(_repo_path(args.stage4_9_summary)),
            "summary_sha256": sha256(_repo_path(args.stage4_9_summary)),
            "summary_protocol": formal_summary["protocol"],
            "candidate_result_path": next(
                value["path"]
                for value in formal_summary["results"]
                if value["candidate_name"] == candidate
            ),
            "candidate_result_sha256": next(
                value["sha256"]
                for value in formal_summary["results"]
                if value["candidate_name"] == candidate
            ),
            "scientific_result": True,
            "edge_action_binding": shared["action_binding"],
        },
        "input_provenance": {
            "prepared_data_sha256": sha256(
                _repo_path(args.prepared_data)
            ),
            "training_result_sha256": sha256(
                _repo_path(args.training_result)
            ),
            "compiler_result_sha256": observed_compiler_sha256,
            "source_checkpoint_sha256": observed_checkpoint_hashes["theta0"],
            "target_checkpoint_sha256": observed_checkpoint_hashes["theta1"],
            "edge_artifact_sha256": edge_observed_sha256,
            "program_file_sha256": observed_program_file_sha256,
            "program_memory_sha256": canonical_program_memory_sha256,
            "program_memory_sha256_by_device": (
                program_memory_sha256_by_device
            ),
            "program_identity_hash_kind": (
                "canonical_loaded_program_tensor_content_v1"
            ),
            "program_shape": list(expected_program_shape),
            "program_loads": loaded_programs,
        },
        "implementation": implementation_snapshot(),
        "source_initialization": {
            name: value["old_source_initialization"]
            for name, value in case_evidence.items()
        },
        "capacity_observations": capacity_observations,
        "canary_contract": {
            "selection_role": "program_selection",
            "labels_used": False,
            "metric": "kv_relative_l2",
            "maximum_relative_l2": canary_config["threshold"][
                "maximum_relative_l2"
            ],
            "records": CANARY_RECORDS,
            "threshold_source": str(_repo_path(args.canary_config)),
            "threshold_artifact_sha256": canary_config_sha256,
        },
        "stage5_closure": closure,
        "case_evidence": case_evidence,
        "closure_validation": validation,
        "checks": {**checks, "all_passed": True},
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    devices = validate_args(args)
    if args.smoke_test:
        print(json.dumps(smoke_payload(args), indent=2))
        return
    output = _repo_path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(
            "Stage 5 formal output exists; use --force only after "
            f"resolving provenance: {output}"
        )
    seed_everything(args.seed)
    payload = run_formal(args, devices)
    _atomic_save_json(payload, output)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "status": payload["status"],
                "scientific_result": payload["scientific_result"],
                "output": str(output),
                "output_sha256": sha256(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
