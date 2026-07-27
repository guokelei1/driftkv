from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from hstu_kvcache.migration import sha256_file

PROTOCOL = "cohortkv_single_config_stage4_5_frozen_v1"
STAGE4_PROTOCOL = "cohortkv_single_config_stage4_frozen_v1"
COMPILER_PROTOCOL = "cohortkv_single_config_stage4_5_oldkv_compiler_v1"
CERTIFICATE_PROTOCOL = (
    "cohortkv_single_config_stage4_5_oldkv_certificate_v1"
)
SYSTEM_PROTOCOL = "cohortkv_single_config_stage4_5_oldkv_system_v1"
TRANSPORT_PROTOCOL = (
    "cohortkv_single_config_stage4_5_oldkv_full_transport_v1"
)
DIRECT_PROGRAM_PROTOCOL = "cohortkv_stage4_5_direct_oldkv_program_v1"
DIRECT_ENGINE_PROTOCOL = "cohortkv_stage4_5_direct_oldkv_engine_v1"
STAGE4 = Path(
    "configs/cohortkv_single_config_v1/stage4_system_summary.json"
)
WORKLOAD = Path(
    "configs/cohortkv_single_config_v1/workload_manifest.json"
)
SOURCE_MANIFEST = Path(
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0/"
    "single_config_v1/source_shards/source_manifest.json"
)
COMPILER = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_compiler_seed0.json"
)
CERTIFICATE = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_certificate_seed0.json"
)
SYSTEM = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_system_seed0.json"
)
SYSTEM_EXPANSION = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_system_expansion_seed0.json"
)
FULL_TRANSPORT = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_full_transport_seed0.json"
)
RESIDENT_CEILING = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_resident_ceiling_seed0.json"
)
NORMALIZED_RECLAIM = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_reclaim_candidate_seed0.json"
)
OUTPUT = Path(
    "configs/cohortkv_single_config_v1/"
    "stage4_5_source_plan_summary.json"
)
RECORDS = 682
PREFIX_TOKENS = 1_087_785
VALID_ELEMENTS = 17_822_269_440
OUTPUT_BYTES = 35_644_538_880
WORKLOAD_CONTENT_SHA256 = (
    "41b7ad10a8dc3a05ce99342a0d73a09e09847ddf42b9111d318b3ddd3c62a910"
)
SOURCE_MANIFEST_SHA256 = (
    "ad1abb3317335f4e3b0eca84961fc8fa02b5a7e8f3d6fcf7003b81be7e212c7a"
)
PROGRAM_SET_SHA256 = (
    "208ed037630475cb54f491949a7e7b7d623caed39aaa75764cd1aee69a0ff6e3"
)
PROGRAM_BYTES = 100_777_103
GPU_COUNTS = (1, 2, 4)
SOURCE_VERSIONS = ("theta0", "theta4", "theta10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def descriptor(root: Path, path: Path, protocol: str | None = None) -> dict:
    value = {
        "path": str(path),
        "bytes": (root / path).stat().st_size,
        "sha256": sha256_file(root / path),
    }
    if protocol is not None:
        value["protocol"] = protocol
    return value


def load_artifact(
    root: Path,
    path: Path,
    protocol: str,
    status: str,
) -> dict:
    value = json.loads((root / path).read_text())
    require(value.get("protocol") == protocol, f"{path} protocol differs")
    require(value.get("status") == status, f"{path} status differs")
    require(value.get("seed") == 0, f"{path} seed differs")
    require(value.get("labels_used") is False, f"{path} used labels")
    return value


def validate_current_snapshot(root: Path, value: dict, name: str) -> None:
    implementation = value.get("implementation", {})
    files = implementation.get("files", [])
    require(bool(files), f"{name} implementation snapshot is empty")
    require(
        len({item.get("path") for item in files}) == len(files),
        f"{name} implementation paths repeat",
    )
    for item in files:
        path = root / item["path"]
        require(path.is_file(), f"{name} implementation file is missing")
        require(path.stat().st_size == item["bytes"], f"{name} bytes differ")
        require(
            sha256_file(path) == item["sha256"],
            f"{name} implementation hash differs",
        )
    require(
        sha256_bytes(canonical_json_bytes(files))
        == implementation["code_snapshot_sha256"],
        f"{name} implementation aggregate differs",
    )


def validate_upstream(root: Path) -> tuple[dict, dict]:
    stage4 = json.loads((root / STAGE4).read_text())
    workload = json.loads((root / WORKLOAD).read_text())
    require(stage4.get("protocol") == STAGE4_PROTOCOL, "Stage 4 protocol differs")
    require(stage4.get("status") == "stage4_frozen", "Stage 4 is not frozen")
    require(
        stage4.get("derived", {}).get("compiled_beats_exact_points") == 0,
        "Stage 4 negative gate differs",
    )
    require(
        workload.get("content_sha256") == WORKLOAD_CONTENT_SHA256,
        "workload content differs",
    )
    require(
        workload.get("summary", {}).get("records") == RECORDS,
        "workload records differ",
    )
    require(
        workload.get("summary", {}).get("prefix_tokens") == PREFIX_TOKENS,
        "workload tokens differ",
    )
    require(
        sha256_file(root / SOURCE_MANIFEST) == SOURCE_MANIFEST_SHA256,
        "source manifest differs",
    )
    return stage4, workload


def program_set_hash(programs: list[dict]) -> str:
    values = [
        {
            "source_version": value["source_version"],
            "target_version": value["target_version"],
            "sha256": value["sha256"],
            "bytes": value["bytes"],
        }
        for value in programs
    ]
    return hashlib.sha256(
        json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def validate_compiler(root: Path, value: dict) -> dict:
    representation = value.get("representation", {})
    programs = representation.get("programs", [])
    require(
        value.get("parent_protocol") == DIRECT_PROGRAM_PROTOCOL,
        "compiler parent protocol differs",
    )
    require(
        value.get("study_stage") == "stage4_5_b_direct_oldkv_seed0",
        "compiler study stage differs",
    )
    require(
        value.get("inputs", {}).get("stage4_summary_sha256")
        == sha256_file(root / STAGE4),
        "compiler Stage 4 input differs",
    )
    require(
        value.get("inputs", {}).get("source_manifest_sha256")
        == SOURCE_MANIFEST_SHA256,
        "compiler source manifest differs",
    )
    require(
        representation.get("input") == "existing_old_kv_fp16",
        "compiler source representation differs",
    )
    require(
        representation.get("additional_per_record_source_state_bytes") == 0,
        "compiler retains extra record state",
    )
    require(
        representation.get("program_bytes") == PROGRAM_BYTES,
        "compiler program bytes differ",
    )
    require(
        representation.get("program_set_sha256") == PROGRAM_SET_SHA256,
        "compiler program set differs",
    )
    require(len(programs) == 3, "compiler program coverage differs")
    require(
        {item.get("source_version") for item in programs}
        == set(SOURCE_VERSIONS),
        "compiler source versions differ",
    )
    require(
        all(item.get("target_version") == "theta11" for item in programs),
        "compiler target version differs",
    )
    require(
        sum(item.get("bytes", 0) for item in programs) == PROGRAM_BYTES,
        "compiler program-byte sum differs",
    )
    require(
        program_set_hash(programs) == PROGRAM_SET_SHA256,
        "compiler derived program set differs",
    )
    compact_programs = []
    for item in programs:
        path = Path(item["path"])
        require(path.is_absolute(), "compiler program path is not absolute")
        require(path.is_relative_to(root), "compiler program is outside repository")
        require(path.is_file(), "compiler program is missing")
        require(path.stat().st_size == item["bytes"], "compiler program bytes differ")
        require(sha256_file(path) == item["sha256"], "compiler program hash differs")
        metrics = item["compile_metrics"]
        require(
            metrics["condition_number_min"] > 0
            and metrics["condition_number_max"] < 11,
            "compiler projection conditioning differs",
        )
        require(
            item.get("load_validation", {}).get("passed") is True,
            "compiler program reload failed",
        )
        compact_programs.append(
            {
                "source_version": item["source_version"],
                "target_version": item["target_version"],
                "path": str(path.relative_to(root)),
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "condition_number_range": [
                    metrics["condition_number_min"],
                    metrics["condition_number_max"],
                ],
                "compile_seconds": metrics["elapsed_seconds"],
            }
        )
    transport = value.get("transport_certificate", {})
    require(transport.get("passed") is True, "selection transport failed")
    require(
        transport.get("atol") == 0.02 and transport.get("rtol") == 0.02,
        "selection transport tolerance differs",
    )
    for role, expected_records in (
        ("program_selection", 60),
        ("certificate", 60),
    ):
        aggregate = transport["roles"][role]["aggregate"]
        require(
            aggregate.get("records") == expected_records
            and aggregate.get("allclose") is True
            and aggregate.get("finite") is True
            and aggregate.get("mismatched_elements") == 0,
            f"{role} compiler transport differs",
        )
    winner = value.get("operator_selection", {}).get("winner", {})
    require(
        winner
        == {
            "block_k": 64,
            "block_m": 64,
            "block_n": 128,
            "num_stages": 3,
            "num_warps": 8,
        },
        "direct old-K/V operator winner differs",
    )
    return {
        "representation": "existing_old_kv_fp16",
        "derivation": representation["derivation"],
        "additional_per_record_source_state_bytes": 0,
        "program_bytes": PROGRAM_BYTES,
        "program_set_sha256": PROGRAM_SET_SHA256,
        "programs": compact_programs,
        "operator": {
            "kind": "direct_oldkv_fused_fp16",
            "launch": winner,
            "selection_role": "program_selection",
        },
        "selection_transport": {
            "roles": ["program_selection", "certificate"],
            "all_outputs_finite_and_allclose": True,
            "mismatched_elements": 0,
            "maximum_absolute_error": max(
                transport["roles"][role]["aggregate"]["max_abs_error"]
                for role in ("program_selection", "certificate")
            ),
            "atol": 0.02,
            "rtol": 0.02,
        },
    }


def validate_certificate(value: dict) -> dict:
    aggregate = value.get("aggregate", {})
    require(
        value.get("study_stage") == "stage4_5_b_direct_oldkv_seed0",
        "certificate study stage differs",
    )
    require(
        aggregate.get("all_selected_direct_oldkv") is True,
        "direct old-K/V was not selected for every pair",
    )
    require(
        aggregate.get("all_exact_fallback") is True,
        "exact fallback is not complete",
    )
    require(
        aggregate.get("minimum_worst_view_recovery", 0) >= 0.7,
        "certificate recovery gate failed",
    )
    require(
        aggregate.get("maximum_cost_ratio_to_exact", 1) <= 0.3,
        "certificate cost gate failed",
    )
    pairs = value.get("pairs", [])
    require(len(pairs) == 3, "certificate pair count differs")
    compact = []
    for pair in pairs:
        certificate = pair["certificate"]
        direct = next(
            item
            for item in certificate["certificates"]
            if item["action_name"] == "compiled_old_kv"
        )
        require(
            certificate.get("selected_action") == "compiled_old_kv",
            "certificate selected action differs",
        )
        require(
            certificate.get("fallback_actions") == ["recompute"],
            "certificate fallback differs",
        )
        require(
            certificate.get("labels_used") is False,
            "certificate used labels",
        )
        require(
            direct.get("fidelity_passed") is True
            and direct.get("budget_passed") is True,
            "direct old-K/V certificate failed",
        )
        require(
            direct.get("worst_recovery_lower_bound", 0) >= 0.7
            and direct.get("worst_coverage_lower_bound", 0) >= 0.8,
            "direct old-K/V certificate lower bound failed",
        )
        compact.append(
            {
                "source_version": pair["source_version"],
                "target_version": pair["target_version"],
                "selected_action": "compiled_old_kv",
                "fallback_actions": ["recompute"],
                "cost_ratio_to_exact": pair["summary"]["cost_ratio_to_exact"],
                "cache_recovery": pair["summary"]["cache_recovery"],
                "score_recovery": pair["summary"]["score_recovery"],
                "top100_recovery": pair["summary"]["top100_recovery"],
                "worst_view_recovery": min(
                    pair["summary"]["cache_recovery"],
                    pair["summary"]["score_recovery"],
                    pair["summary"]["top100_recovery"],
                ),
                "worst_recovery_lower_bound": direct[
                    "worst_recovery_lower_bound"
                ],
                "worst_coverage_lower_bound": direct[
                    "worst_coverage_lower_bound"
                ],
            }
        )
    require(
        {item["source_version"] for item in compact} == set(SOURCE_VERSIONS),
        "certificate source versions differ",
    )
    return {
        "contract": value["contract"],
        "records_per_source_pair": 60,
        "recommendation_labels_used": False,
        "pairs": compact,
        "minimum_worst_view_recovery": aggregate[
            "minimum_worst_view_recovery"
        ],
        "maximum_cost_ratio_to_exact": aggregate[
            "maximum_cost_ratio_to_exact"
        ],
        "all_selected_direct_oldkv": True,
        "all_exact_fallback": True,
    }


def validate_full_transport(root: Path, value: dict) -> dict:
    validate_current_snapshot(root, value, "full transport")
    aggregate = value.get("aggregate", {})
    require(
        aggregate.get("records") == RECORDS
        and aggregate.get("prefix_tokens") == PREFIX_TOKENS
        and aggregate.get("valid_elements") == VALID_ELEMENTS,
        "full transport coverage differs",
    )
    require(
        aggregate.get("finite") is True
        and aggregate.get("allclose") is True
        and aggregate.get("mismatched_elements") == 0,
        "full transport failed",
    )
    require(
        value.get("atol") == 0.02 and value.get("rtol") == 0.02,
        "full transport tolerance differs",
    )
    expected_roles = {
        "fit": 40,
        "program_selection": 60,
        "certificate": 60,
        "final_test": 522,
    }
    require(set(value.get("by_role", {})) == set(expected_roles), "roles differ")
    for role, records in expected_roles.items():
        item = value["by_role"][role]
        require(
            item.get("records") == records
            and item.get("allclose") is True
            and item.get("mismatched_elements") == 0,
            f"{role} full transport differs",
        )
    require(
        set(value.get("by_source", {})) == set(SOURCE_VERSIONS),
        "full transport source versions differ",
    )
    return {
        "measurement": value["measurement_boundary"],
        "records": RECORDS,
        "prefix_tokens": PREFIX_TOKENS,
        "valid_elements": VALID_ELEMENTS,
        "roles": expected_roles,
        "recommendation_labels_used": False,
        "finite": True,
        "allclose": True,
        "mismatched_elements": 0,
        "maximum_absolute_error": aggregate["max_abs_error"],
        "mean_absolute_error": aggregate["mean_abs_error"],
        "atol": 0.02,
        "rtol": 0.02,
    }


def validate_reclamation(value: dict, expected_records: int) -> None:
    require(
        value.get("initial_old_kv_bytes") == value.get("retired_old_kv_bytes"),
        "old K/V byte retirement differs",
    )
    require(value.get("final_old_kv_bytes") == 0, "old K/V remains resident")
    require(
        value.get("final_new_kv_bytes", 0) >= OUTPUT_BYTES,
        "new K/V publication is incomplete",
    )
    require(
        value.get("retired_extent_count") == expected_records,
        "old K/V extent retirement differs",
    )


def compact_point(point: dict) -> dict:
    summary = point["summary"]
    correctness = point["correctness_job"]["correctness"]
    reclamation = point["correctness_job"]["reclamation"]
    require(point.get("status") == "complete", "system point is incomplete")
    require(
        point.get("capacity_preflight", {}).get("passed") is True,
        "system capacity preflight failed",
    )
    require(
        len(summary.get("samples_seconds", [])) == 5,
        "system repetition count differs",
    )
    require(
        math.isclose(
            summary["median_seconds"],
            statistics.median(summary["samples_seconds"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "system median differs",
    )
    require(summary.get("stable_hbm_baseline") is True, "HBM baseline moved")
    require(
        correctness.get("finite") is True
        and correctness.get("allclose") is True
        and correctness.get("record_order_valid") is True
        and correctness.get("lengths_offsets_valid") is True
        and correctness.get("valid_element_count") == VALID_ELEMENTS,
        "system correctness differs",
    )
    require(
        point["correctness_job"]["manifest"].get("record_count") == RECORDS
        and point["correctness_job"]["manifest"].get("prefix_tokens")
        == PREFIX_TOKENS,
        "system manifest coverage differs",
    )
    validate_reclamation(reclamation, point["plan"]["extent_count"])
    return {
        "key": point["key"],
        "method": point["method"],
        "source_tier": point["source_tier"],
        "gpu_count": point["gpu_count"],
        "median_seconds": summary["median_seconds"],
        "samples_seconds": summary["samples_seconds"],
        "median_timing_breakdown": summary["median_timing_breakdown"],
        "maximum_peak_hbm_bytes": summary["maximum_peak_hbm_bytes"],
        "maximum_peak_old_plus_new_kv_bytes": summary[
            "maximum_peak_old_plus_new_kv_bytes"
        ],
        "capacity_preflight": point["capacity_preflight"],
        "correctness": correctness,
        "manifest": point["correctness_job"]["manifest"],
        "reclamation": reclamation,
        "source_lifecycle": point.get("source_lifecycle"),
    }


def validate_system(
    root: Path,
    representative: dict,
    expansion: dict,
) -> dict:
    validate_current_snapshot(root, representative, "representative system")
    validate_current_snapshot(root, expansion, "system expansion")
    require(
        representative.get("parent_protocol") == DIRECT_ENGINE_PROTOCOL
        and expansion.get("parent_protocol") == DIRECT_ENGINE_PROTOCOL,
        "system parent protocol differs",
    )
    require(
        representative.get("record_count") == RECORDS
        and expansion.get("record_count") == RECORDS
        and representative.get("prefix_tokens") == PREFIX_TOKENS
        and expansion.get("prefix_tokens") == PREFIX_TOKENS,
        "system workload coverage differs",
    )
    require(
        representative.get("measurement_boundary")
        == expansion.get("measurement_boundary"),
        "system measurement boundaries differ",
    )
    points = representative.get("points", []) + expansion.get("points", [])
    require(len(points) == 6, "system point count differs")
    compact_points = [compact_point(item) for item in points]
    require(
        {
            (item["method"], item["gpu_count"])
            for item in compact_points
        }
        == {
            (method, gpu_count)
            for method in ("compiled_old_kv", "exact")
            for gpu_count in GPU_COUNTS
        },
        "system matrix differs",
    )
    comparisons = (
        representative.get("comparisons", [])
        + expansion.get("comparisons", [])
    )
    require(len(comparisons) == 3, "system comparison count differs")
    compact_comparisons = []
    for item in comparisons:
        require(
            item.get("gpu_count") in GPU_COUNTS
            and item.get("completion_gate_passed") is True
            and item.get("all_compiled_below_all_exact") is True
            and item.get("difference_exceeds_variation_band") is True
            and item.get("additional_compiled_source_state_bytes") == 0
            and item.get("speedup", 0) > 1,
            "system completion gate failed",
        )
        require(
            item.get("compiled_source_tier") == "existing_old_kv_hbm"
            and item.get("exact_source_tier") == "raw_history_hbm",
            "system source-tier comparison differs",
        )
        validate_reclamation(
            item["compiled_reclamation"],
            item["compiled_reclamation"]["retired_extent_count"],
        )
        validate_reclamation(
            item["exact_reclamation"],
            item["exact_reclamation"]["retired_extent_count"],
        )
        compact_comparisons.append(
            {
                "gpu_count": item["gpu_count"],
                "compiled_median_seconds": item[
                    "compiled_median_seconds"
                ],
                "exact_median_seconds": item["exact_median_seconds"],
                "speedup": item["speedup"],
                "all_compiled_below_all_exact": True,
                "difference_exceeds_variation_band": True,
                "completion_gate_passed": True,
                "additional_compiled_source_state_bytes": 0,
                "compiled_source_tier": item["compiled_source_tier"],
                "exact_source_tier": item["exact_source_tier"],
                "peak_old_plus_new_kv_bytes": item[
                    "compiled_reclamation"
                ]["peak_old_plus_new_kv_bytes"],
                "final_old_kv_bytes": item["compiled_reclamation"][
                    "final_old_kv_bytes"
                ],
            }
        )
    compact_points.sort(key=lambda item: (item["gpu_count"], item["method"]))
    compact_comparisons.sort(key=lambda item: item["gpu_count"])
    return {
        "measurement_boundary": representative["measurement_boundary"],
        "destination": "hbm",
        "records": RECORDS,
        "prefix_tokens": PREFIX_TOKENS,
        "target_layout": "complete unpadded FP16 K/V extents and atomic manifest",
        "warmup_runs_per_point": 1,
        "measured_repetitions_per_point": 5,
        "recommendation_labels_used": False,
        "points": compact_points,
        "comparisons": compact_comparisons,
        "all_capacity_preflights_passed": True,
        "all_outputs_finite_and_allclose": True,
        "all_manifests_complete": True,
        "representative_gate_passed": all(
            item["completion_gate_passed"]
            for item in compact_comparisons
            if item["gpu_count"] in {1, 4}
        ),
        "two_gpu_expansion_passed": next(
            item["completion_gate_passed"]
            for item in compact_comparisons
            if item["gpu_count"] == 2
        ),
    }


def validate_candidate_history(
    root: Path,
    resident: dict,
    normalized: dict,
) -> dict:
    require(
        resident.get("protocol") == "cohortkv_stage4_5_resident_ceiling_v1"
        and resident.get("status") == "stage4_5_a_complete",
        "resident ceiling differs",
    )
    require(resident.get("labels_used") is False, "resident ceiling used labels")
    require(
        len(resident.get("comparisons", [])) == 4
        and all(
            item.get("resident_completion_gate_passed") is True
            for item in resident["comparisons"]
        ),
        "resident ceiling did not close",
    )
    require(
        normalized.get("protocol")
        == "cohortkv_single_config_stage4_5_reclaim_candidate_v1"
        and normalized.get("status") == "reclaim_candidate_full_complete",
        "normalized reclaim candidate differs",
    )
    require(normalized.get("labels_used") is False, "normalized candidate used labels")
    comparisons = normalized.get("comparisons", [])
    require(
        {item.get("gpu_count") for item in comparisons} == {1, 4}
        and all(item.get("completion_gate_passed") is True for item in comparisons),
        "normalized candidate full-cohort gate differs",
    )
    compact_normalized = []
    for item in comparisons:
        compact_normalized.append(
            {
                "gpu_count": item["gpu_count"],
                "compiled_median_seconds": item[
                    "compiled_median_seconds"
                ],
                "exact_median_seconds": item["exact_median_seconds"],
                "speedup": item["speedup"],
                "standing_host_source_bytes": item["compiled_preload"][
                    "standing_host_bytes"
                ],
                "compiled_preload_seconds": item["compiled_preload"][
                    "elapsed_seconds"
                ],
                "break_even_updates": item["break_even_updates"],
            }
        )
    compact_normalized.sort(key=lambda item: item["gpu_count"])
    return {
        "resident_ceiling": {
            "artifact": descriptor(root, RESIDENT_CEILING),
            "selection_role_points": 4,
            "all_matched_resident_completion_gates_passed": True,
        },
        "normalized_capsule_dram_candidate": {
            "artifact": descriptor(root, NORMALIZED_RECLAIM),
            "status": (
                "valid full-cohort backup, retired from primary source plan "
                "because it retains and preloads 17.86 GB of extra host state"
            ),
            "comparisons": compact_normalized,
        },
        "selection": (
            "direct existing-old-K/V reparameterization strictly removes "
            "the extra per-record source state and is the simplest "
            "frontier-changing source plan"
        ),
    }


def build_summary(root: Path) -> dict:
    validate_upstream(root)
    compiler = load_artifact(
        root,
        COMPILER,
        COMPILER_PROTOCOL,
        "oldkv_program_transport_frozen",
    )
    certificate = load_artifact(
        root,
        CERTIFICATE,
        CERTIFICATE_PROTOCOL,
        "oldkv_semantic_certificate_frozen",
    )
    representative = load_artifact(
        root,
        SYSTEM,
        SYSTEM_PROTOCOL,
        "oldkv_system_representative_complete",
    )
    expansion = load_artifact(
        root,
        SYSTEM_EXPANSION,
        SYSTEM_PROTOCOL,
        "oldkv_system_representative_complete",
    )
    transport = load_artifact(
        root,
        FULL_TRANSPORT,
        TRANSPORT_PROTOCOL,
        "oldkv_full_transport_frozen",
    )
    resident = json.loads((root / RESIDENT_CEILING).read_text())
    normalized = json.loads((root / NORMALIZED_RECLAIM).read_text())
    compiler_summary = validate_compiler(root, compiler)
    certificate_summary = validate_certificate(certificate)
    transport_summary = validate_full_transport(root, transport)
    system_summary = validate_system(root, representative, expansion)
    candidate_history = validate_candidate_history(
        root,
        resident,
        normalized,
    )
    return {
        "protocol": PROTOCOL,
        "status": "stage4_5_source_plan_frozen",
        "study_stage": "single_configuration_seed0_development",
        "frozen_date": "2026-07-27",
        "upstream": {
            "stage4_summary": descriptor(root, STAGE4, STAGE4_PROTOCOL),
            "workload_manifest": {
                **descriptor(root, WORKLOAD),
                "content_sha256": WORKLOAD_CONTENT_SHA256,
            },
            "source_manifest": descriptor(root, SOURCE_MANIFEST),
        },
        "evidence_artifacts": {
            "compiler": descriptor(root, COMPILER, COMPILER_PROTOCOL),
            "certificate": descriptor(
                root,
                CERTIFICATE,
                CERTIFICATE_PROTOCOL,
            ),
            "full_transport": descriptor(
                root,
                FULL_TRANSPORT,
                TRANSPORT_PROTOCOL,
            ),
            "representative_system": descriptor(
                root,
                SYSTEM,
                SYSTEM_PROTOCOL,
            ),
            "two_gpu_expansion": descriptor(
                root,
                SYSTEM_EXPANSION,
                SYSTEM_PROTOCOL,
            ),
        },
        "source_plan": {
            "normal_action": "compiled_old_kv",
            "source_representation": "existing_old_kv_fp16",
            "placement": "existing serving cache in HBM",
            "supply": "direct device read",
            "additional_normx_bytes": 0,
            "additional_per_record_source_state_bytes": 0,
            "program_bytes": PROGRAM_BYTES,
            "reclamation": (
                "retire each old K/V extent only after its replacement "
                "extent is accepted by the destination transaction"
            ),
            "capacity_preflight": (
                "account model, replicated direct programs, complete "
                "existing old K/V, maximum replacement wave, and 2 GiB "
                "allocator margin per GPU"
            ),
            "fallback_action": "exact",
            "fallback_conditions": [
                "capacity preflight failure",
                "existing old K/V unavailable",
                "program verification failure",
            ],
            "policy_dispatch_interface_implemented": True,
            "automatic_transactional_fallback_execution_implemented": False,
        },
        "compiler": compiler_summary,
        "semantic_certificate": certificate_summary,
        "full_real_transport": transport_summary,
        "system": system_summary,
        "candidate_history": candidate_history,
        "gate": {
            "extra_normx_eliminated": True,
            "representative_1gpu_passed": True,
            "representative_4gpu_passed": True,
            "two_gpu_expansion_passed": True,
            "all_compiled_samples_below_all_paired_exact_samples": True,
            "stable_end_to_end_pareto_point": True,
            "stage5_admitted": True,
        },
        "declared_operating_regime": {
            "cache_state": "complete source-version old K/V already in HBM",
            "hardware": "1, 2, or 4 NVIDIA A40 GPUs",
            "target": "theta11 FP16 K/V in HBM",
            "cohort": "frozen 682-record controlled theta0/theta4/theta10 mix",
            "exact_control": "complete raw history already in HBM",
            "excluded_claims": [
                "cold filesystem or durable SSD speedup",
                "automatic online cache-tier selection",
                "organic mixed-version scheduling",
                "failure-safe automatic fallback execution",
                "new-seed or cross-dataset replication",
            ],
        },
        "measurement_disclosures": {
            "direct_performance_source_values": (
                "shape-, dtype-, layout-, and occupancy-equivalent old K/V; "
                "the complete real old-K/V transport is validated separately"
            ),
            "full_real_transport_includes_final_test": (
                "yes, for label-free transport correctness only; it did not "
                "affect candidate, operator, policy, or threshold selection"
            ),
            "compiled_preload_seconds": 0.0,
            "compiled_preload_reason": (
                "the old K/V is the pre-existing serving cache, not newly "
                "created Stage-4.5 state"
            ),
            "program_compilation": "reported once per source/target pair",
            "benchmark_old_cache_reset": (
                "reported outside each timed repetition because it only "
                "reconstructs the same starting state for benchmark replay"
            ),
        },
        "next_stage": {
            "stage": 5,
            "objective": (
                "connect the frozen source-plan decision to verified plan "
                "fallback, semantic guard/preflight, transactional rework, "
                "and failure visibility"
            ),
            "source_plan_may_change": False,
            "open_work": [
                "automatic exact dispatch",
                "semantic degradation detection",
                "mid-job rework",
                "failure injection and atomic visibility",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    payload = canonical_json_bytes(build_summary(root))
    resolved = root / output
    if args.check:
        if not resolved.is_file() or resolved.read_bytes() != payload:
            raise RuntimeError(
                "Stage 4.5 frozen summary differs from evidence artifacts"
            )
        status = "verified"
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(payload)
        status = "frozen"
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": status,
                "output": str(output),
                "sha256": sha256_bytes(payload),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
