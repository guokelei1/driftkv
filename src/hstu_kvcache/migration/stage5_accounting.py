from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

STAGE5_ACCOUNTING_PROTOCOL = "cohortkv_stage5_source_state_accounting_v1"
_INPUT_PROTOCOLS = {
    "stage2": "cohortkv_single_config_stage2_frozen_v1",
    "stage4": "cohortkv_single_config_stage4_frozen_v1",
    "stage4_5": "cohortkv_single_config_stage4_5_frozen_v1",
}
_INPUT_STATUSES = {
    "stage2": "stage2_frozen",
    "stage4": "stage4_frozen",
    "stage4_5": "stage4_5_source_plan_frozen",
}
_INPUT_SHA256 = {
    "stage2": "09461c81ad7d9a061a6aae2358e478c151befa07614f73afff972bd4b90a8126",
    "stage4": "2c891e2fb085708bdb83c6c39410f9f7509f25697ff0e8b0a4085906ef7219b6",
    "stage4_5": "31cb442563152a250705d8ad3405461238810c5a5e81cac09c2ab20ae294e2f8",
}
_SHA256_LENGTH = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Stage 5 accounting input must be an object")
    return value


def _input_record(
    name: str,
    path: Path,
    value: dict[str, object],
) -> dict[str, object]:
    if value.get("protocol") != _INPUT_PROTOCOLS[name]:
        raise ValueError(f"Stage 5 accounting {name} protocol differs")
    if value.get("status") != _INPUT_STATUSES[name]:
        raise ValueError(f"Stage 5 accounting {name} status differs")
    observed_sha256 = _sha256(path)
    if observed_sha256 != _INPUT_SHA256[name]:
        raise ValueError(f"Stage 5 accounting {name} frozen SHA-256 differs")
    return {
        "path": str(path),
        "protocol": value["protocol"],
        "sha256": observed_sha256,
        "status": value["status"],
    }


def _matched_runs(
    runs: list[dict[str, object]],
    method: str,
) -> dict[tuple[str, int], dict[str, object]]:
    values = [value for value in runs if value["method"] == method]
    selected = {
        (str(value["destination"]), int(value["gpu_count"])): value
        for value in values
    }
    expected = {
        (destination, gpu_count)
        for destination in ("hbm", "dram")
        for gpu_count in (1, 2, 4)
    }
    if len(values) != len(expected) or set(selected) != expected:
        raise ValueError(f"Stage 5 accounting {method} endpoints differ")
    return selected


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_nonnegative(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def _validate_upstream_chain(
    stage2: dict[str, object],
    stage4: dict[str, object],
    stage4_5: dict[str, object],
) -> None:
    stage4_stage2 = stage4.get("stage2_summary", {})
    stage4_5_stage4 = stage4_5.get("upstream", {}).get("stage4_summary", {})
    stage4_source = stage4.get("source_manifest", {})
    stage4_5_source = stage4_5.get("upstream", {}).get("source_manifest", {})
    workload_file_hashes = {
        stage2.get("workload", {}).get("file_sha256"),
        stage4.get("workload", {}).get("file_sha256"),
        stage4_5.get("upstream", {})
        .get("workload_manifest", {})
        .get("sha256"),
    }
    if (
        stage4_stage2.get("protocol") != _INPUT_PROTOCOLS["stage2"]
        or stage4_stage2.get("sha256") != _INPUT_SHA256["stage2"]
        or stage4_5_stage4.get("protocol") != _INPUT_PROTOCOLS["stage4"]
        or stage4_5_stage4.get("sha256") != _INPUT_SHA256["stage4"]
        or stage4_source.get("sha256") != stage4_5_source.get("sha256")
        or len(workload_file_hashes) != 1
        or not _is_sha256(next(iter(workload_file_hashes)))
    ):
        raise ValueError("Stage 5 accounting frozen upstream chain differs")


def _validate_active_route(
    stage4_5: dict[str, object],
    programs: list[dict[str, object]],
    direct_points: list[dict[str, object]],
) -> None:
    compiler = stage4_5["compiler"]
    source_plan = stage4_5["source_plan"]
    expected_program_pairs = {
        ("theta0", "theta11"),
        ("theta4", "theta11"),
        ("theta10", "theta11"),
    }
    observed_program_pairs = {
        (value.get("source_version"), value.get("target_version"))
        for value in programs
    }
    if (
        compiler.get("representation") != "existing_old_kv_fp16"
        or int(compiler.get("additional_per_record_source_state_bytes", -1))
        != 0
        or source_plan.get("source_representation")
        != "existing_old_kv_fp16"
        or source_plan.get("placement") != "existing serving cache in HBM"
        or source_plan.get("supply") != "direct device read"
        or source_plan.get("normal_action") != "compiled_old_kv"
        or source_plan.get("fallback_action") != "exact"
        or int(source_plan.get("additional_per_record_source_state_bytes", -1))
        != 0
        or int(source_plan.get("additional_normx_bytes", -1)) != 0
        or int(source_plan.get("program_bytes", -1))
        != int(compiler["program_bytes"])
        or observed_program_pairs != expected_program_pairs
        or any(
            not _is_sha256(value.get("sha256"))
            or int(value.get("bytes", 0)) < 1
            or not _finite_nonnegative(value.get("compile_seconds"))
            for value in programs
        )
    ):
        raise ValueError("Stage 5 direct old-K/V route differs")
    if (
        len(direct_points) != 3
        or {int(value["gpu_count"]) for value in direct_points} != {1, 2, 4}
    ):
        raise ValueError("Stage 5 direct old-K/V points differ")
    for value in direct_points:
        correctness = value.get("correctness", {})
        capacity = value.get("capacity_preflight", {})
        source_lifecycle = value.get("source_lifecycle", {})
        manifest = value.get("manifest", {})
        reclamation = value.get("reclamation", {})
        if (
            value.get("source_tier") != "existing_old_kv_hbm"
            or capacity.get("passed") is not True
            or capacity.get("source_tier") != "existing_old_kv_hbm"
            or any(
                correctness.get(name) is not True
                for name in (
                    "allclose",
                    "finite",
                    "lengths_offsets_valid",
                    "record_order_valid",
                )
            )
            or source_lifecycle.get("additional_source_state_bytes") != 0
            or source_lifecycle.get("h2d_bytes") != 0
            or float(source_lifecycle.get("preload_seconds", -1)) != 0.0
            or source_lifecycle.get("standing_source")
            != "existing serving old K/V in HBM"
            or manifest.get("record_count") != 682
            or manifest.get("prefix_tokens") != 1_087_785
            or reclamation.get("initial_old_kv_bytes")
            != reclamation.get("retired_old_kv_bytes")
            or not _finite_nonnegative(value.get("median_seconds"))
            or int(value.get("maximum_peak_hbm_bytes", 0)) < 1
        ):
            raise ValueError("Stage 5 direct old-K/V evidence differs")


def _validate_capsule_runs(
    compiled: dict[tuple[str, int], dict[str, object]],
    exact: dict[tuple[str, int], dict[str, object]],
) -> None:
    for key in compiled:
        for value in (compiled[key], exact[key]):
            correctness = value.get("correctness", {})
            if (
                any(
                    correctness.get(name) is not True
                    for name in (
                        "allclose",
                        "finite",
                        "lengths_offsets_valid",
                        "record_order_valid",
                    )
                )
                or not _finite_nonnegative(
                    value.get("timing", {}).get("median_seconds")
                )
            ):
                raise ValueError("Stage 5 capsule endpoint evidence differs")
        elapsed = float(compiled[key]["timing"]["median_seconds"])
        source_read = float(
            compiled[key]["timing"]["breakdown_seconds"]["source_read"]
        )
        if elapsed <= 0 or not 0 <= source_read <= elapsed:
            raise ValueError("Stage 5 capsule source-read timing differs")


def build_stage5_source_state_accounting(
    stage2_path: Path | str,
    stage4_path: Path | str,
    stage4_5_path: Path | str,
) -> dict[str, object]:
    paths = {
        "stage2": Path(stage2_path),
        "stage4": Path(stage4_path),
        "stage4_5": Path(stage4_5_path),
    }
    values = {name: _load(path) for name, path in paths.items()}
    inputs = {
        name: _input_record(name, paths[name], values[name])
        for name in paths
    }
    stage2 = values["stage2"]
    stage4 = values["stage4"]
    stage4_5 = values["stage4_5"]
    _validate_upstream_chain(stage2, stage4, stage4_5)
    workload_hashes = {
        stage2["workload"]["content_sha256"],
        stage4["workload"]["content_sha256"],
        stage4_5["upstream"]["workload_manifest"]["content_sha256"],
    }
    if len(workload_hashes) != 1:
        raise ValueError("Stage 5 accounting workloads differ")
    materialization = stage4["source_manifest"]["materialization"]
    logical = materialization["logical_bytes"]
    physical = materialization["physical_bytes"]
    compiler = stage4_5["compiler"]
    programs = compiler["programs"]
    program_file_bytes = sum(int(value["bytes"]) for value in programs)
    composition_seconds = sum(
        float(value["compile_seconds"]) for value in programs
    )
    if (
        program_file_bytes != int(compiler["program_bytes"])
        or len(programs) != 3
        or {value["source_version"] for value in programs}
        != {"theta0", "theta4", "theta10"}
    ):
        raise ValueError("Stage 5 direct program set is inconsistent")
    direct_points = [
        value
        for value in stage4_5["system"]["points"]
        if value["method"] == "compiled_old_kv"
    ]
    _validate_active_route(stage4_5, programs, direct_points)
    resident_tensor_bytes = {
        int(device["transform_resident_bytes"])
        for value in direct_points
        for device in value["capacity_preflight"]["per_gpu"]
    }
    if len(resident_tensor_bytes) != 1:
        raise ValueError("Stage 5 direct program residency differs by device")
    direct_normal_path = [
        {
            "gpu_count": int(value["gpu_count"]),
            "initial_old_kv_bytes_all_devices": int(
                value["reclamation"]["initial_old_kv_bytes"]
            ),
            "peak_old_plus_new_kv_bytes_all_devices": int(
                value["reclamation"]["peak_old_plus_new_kv_bytes"]
            ),
            "maximum_peak_hbm_bytes_single_device": int(
                value["maximum_peak_hbm_bytes"]
            ),
            "median_seconds": float(value["median_seconds"]),
            "measurement_mode": "extent_reclaim_normal_path",
            "abort_safe": False,
        }
        for value in sorted(
            direct_points,
            key=lambda item: int(item["gpu_count"]),
        )
    ]
    aggregate = stage2["aggregate"]
    fit_seconds = float(aggregate["historical_fit_seconds"])
    runtime_prepare_seconds = float(aggregate["compile_seconds"])
    certificate_seconds = float(aggregate["certificate_seconds"])
    stage2_one_time_seconds = (
        fit_seconds + runtime_prepare_seconds + certificate_seconds
    )
    if not math.isclose(
        stage2_one_time_seconds,
        float(aggregate["amortized_seconds_per_record_at_682"]) * 682,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError("Stage 5 Stage-2 amortization arithmetic differs")
    runs = stage4["runs"]
    compiled = _matched_runs(runs, "compiled")
    exact = _matched_runs(runs, "exact")
    _validate_capsule_runs(compiled, exact)
    capsule_points = []
    for key in sorted(compiled, key=lambda item: (item[0], item[1])):
        compiled_run = compiled[key]
        exact_run = exact[key]
        elapsed = float(compiled_run["timing"]["median_seconds"])
        source_read = float(
            compiled_run["timing"]["breakdown_seconds"]["source_read"]
        )
        exact_elapsed = float(exact_run["timing"]["median_seconds"])
        capsule_points.append(
            {
                "destination": key[0],
                "gpu_count": key[1],
                "compiled_median_seconds": elapsed,
                "source_read_seconds": source_read,
                "source_read_fraction": source_read / elapsed,
                "paired_exact_median_seconds": exact_elapsed,
                "beats_paired_exact": elapsed < exact_elapsed,
            }
        )
    fractions = [value["source_read_fraction"] for value in capsule_points]
    beats_exact = sum(value["beats_paired_exact"] for value in capsule_points)
    if beats_exact != 0:
        raise ValueError("Stage 5 normalized-capsule negative result changed")
    backup = stage4_5["candidate_history"][
        "normalized_capsule_dram_candidate"
    ]
    backup_points = [
        {
            "gpu_count": int(value["gpu_count"]),
            "preload_seconds": float(value["compiled_preload_seconds"]),
            "standing_host_source_bytes": int(
                value["standing_host_source_bytes"]
            ),
            "compiled_median_seconds": float(
                value["compiled_median_seconds"]
            ),
            "paired_exact_median_seconds": float(
                value["exact_median_seconds"]
            ),
            "beats_paired_exact_after_preload": (
                float(value["compiled_median_seconds"])
                < float(value["exact_median_seconds"])
            ),
        }
        for value in backup["comparisons"]
    ]
    return {
        "protocol": STAGE5_ACCOUNTING_PROTOCOL,
        "status": "artifact_derived",
        "scientific_result": True,
        "inputs": inputs,
        "workload": {
            "content_sha256": workload_hashes.pop(),
            "records": int(stage4["workload"]["records"]),
            "prefix_tokens": int(stage4["workload"]["prefix_tokens"]),
        },
        "active_direct_oldkv": {
            "representation": compiler["representation"],
            "placement": stage4_5["source_plan"]["placement"],
            "additional_per_record_source_state_bytes": int(
                compiler["additional_per_record_source_state_bytes"]
            ),
            "independent_capture_required": False,
            "independent_encode_required": False,
            "independent_preload_required": False,
            "existing_old_kv_logical_bytes": int(logical["old_kv_fp16"]),
            "program_set": {
                "serialized_file_bytes": program_file_bytes,
                "resident_tensor_bytes_per_worker": resident_tensor_bytes.pop(),
                "composition_seconds": composition_seconds,
                "serialization_timing_available": False,
                "programs": [
                    {
                        "source_version": value["source_version"],
                        "target_version": value["target_version"],
                        "serialized_file_bytes": int(value["bytes"]),
                        "composition_seconds": float(
                            value["compile_seconds"]
                        ),
                        "sha256": value["sha256"],
                    }
                    for value in programs
                ],
            },
            "normal_path_points": direct_normal_path,
            "copy_on_write_abort_safe_peak_measured": False,
        },
        "offline_setup": {
            "historical_fit_seconds": fit_seconds,
            "runtime_prepare_seconds": runtime_prepare_seconds,
            "certificate_seconds": certificate_seconds,
            "stage2_one_time_seconds": stage2_one_time_seconds,
            "full_catalog_score_seconds_included_in_certificate": float(
                aggregate["full_catalog_score_seconds"]
            ),
            "stage2_parent_program_file_bytes": int(
                aggregate["program_bytes"]
            ),
            "direct_program_composition_seconds": composition_seconds,
            "seconds_per_record_at_682_stage2_floor": float(
                aggregate["amortized_seconds_per_record_at_682"]
            ),
            "stage2_amortization_curve": aggregate["amortization_curve"],
        },
        "rejected_fp16_normalized_capsule": {
            "logical_bytes": int(logical["normalized_capsule_fp16"]),
            "physical_bytes": int(physical["normalized_capsule_fp16"]),
            "matched_points": len(capsule_points),
            "beats_paired_exact_points": beats_exact,
            "source_read_fraction_min": min(fractions),
            "source_read_fraction_max": max(fractions),
            "points": capsule_points,
            "joint_materialization_seconds_not_attributed": float(
                materialization["elapsed_seconds"]
            ),
        },
        "dram_resident_backup": {
            "active_route": False,
            "status": backup["status"],
            "points": backup_points,
        },
        "claim_boundary": {
            "primary_claim": "prepublished-program hot-HBM data-plane",
            "page_cache_condition": stage4["measurement_boundary"][
                "os_page_cache_condition"
            ],
            "posix_backend": "correctness interface only",
            "physical_ssd_performance_claim": False,
            "cold_filesystem_speedup_claim": False,
            "capsule_capture_claim": False,
            "int8_claim": False,
            "time_break_even_claim": False,
        },
    }


def validate_stage5_source_state_accounting(
    value: dict[str, object],
    stage2_path: Path | str,
    stage4_path: Path | str,
    stage4_5_path: Path | str,
) -> None:
    expected = build_stage5_source_state_accounting(
        stage2_path,
        stage4_path,
        stage4_5_path,
    )
    observed_bytes = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    expected_bytes = json.dumps(
        expected,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if observed_bytes != expected_bytes:
        raise ValueError(
            "Stage 5 accounting is not the exact frozen-input derivation"
        )
