from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from hstu_kvcache.migration import sha256_file

PROTOCOL = "cohortkv_single_config_stage3_frozen_v1"
SOURCE_PROTOCOL = "cohortkv_single_config_stage3_operator_v1"
PARENT_PROTOCOL = "cohortkv_single_config_full_chain_development_v1"
STAGE2_PROTOCOL = "cohortkv_single_config_stage2_frozen_v1"
SOURCE_RESULT = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage3_operator_seed0.json"
)
BLUEPRINT = Path("configs/cohortkv_single_config_v1/blueprint.json")
WORKLOAD = Path("configs/cohortkv_single_config_v1/workload_manifest.json")
STAGE2 = Path("configs/cohortkv_single_config_v1/stage2_compiler_summary.json")
OUTPUT = Path(
    "configs/cohortkv_single_config_v1/stage3_operator_summary.json"
)
BATCH_SIZES = (1, 2, 4)
BUCKET_WIDTHS = (16, 32, 64)
OPERATORS = ("packed_fp16", "fused_fp16")
LAYOUTS = {
    f"b{batch}_w{bucket}"
    for batch in BATCH_SIZES
    for bucket in BUCKET_WIDTHS
}
VALID_TOKENS = 88_085
VALID_ELEMENTS = 1_443_184_640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", default=str(SOURCE_RESULT))
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


def validate_identity(
    root: Path,
    source: dict,
    workload: dict,
    stage2: dict,
) -> None:
    if (
        source.get("protocol") != SOURCE_PROTOCOL
        or source.get("parent_protocol") != PARENT_PROTOCOL
        or source.get("status") != "stage3_complete"
        or source.get("study_stage")
        != "single_configuration_seed0_development"
        or source.get("seed") != 0
        or source.get("labels_used") is not False
        or source.get("final_test_evaluated") is not False
        or source.get("role") != "program_selection"
    ):
        raise ValueError("Stage 3 source identity is invalid")
    if source.get("role_counts") != {
        "certificate": 60,
        "final_test": 522,
        "fit": 40,
        "program_selection": 60,
    }:
        raise ValueError("Stage 3 role counts differ from the frozen split")
    if (
        source["workload_manifest"]["path"] != str(WORKLOAD)
        or source["workload_manifest"]["sha256"]
        != sha256_file(root / WORKLOAD)
        or source["workload_manifest"]["content_sha256"]
        != workload["content_sha256"]
        or source["stage2_summary"]["path"] != str(STAGE2)
        or source["stage2_summary"]["sha256"]
        != sha256_file(root / STAGE2)
        or source["stage2_summary"]["protocol"] != STAGE2_PROTOCOL
        or stage2.get("protocol") != STAGE2_PROTOCOL
        or stage2.get("status") != "stage2_frozen"
    ):
        raise ValueError("Stage 3 frozen-input descriptors are invalid")
    if source["blueprint"]["path"] != str(BLUEPRINT):
        raise ValueError("Stage 3 blueprint path is invalid")
    if (
        source["blueprint"].get("protocol") != PARENT_PROTOCOL
        or not isinstance(source["blueprint"].get("sha256"), str)
        or len(source["blueprint"]["sha256"]) != 64
    ):
        raise ValueError("Stage 3 blueprint descriptor is invalid")
    expected_programs = [
        {
            "source_version": pair["source_version"],
            "target_version": pair["target_version"],
            "path": pair["runtime_program"]["path"],
            "sha256": pair["runtime_program"]["sha256"],
            "bytes": pair["runtime_program"]["bytes"],
            "dtype": pair["runtime_program"]["dtype"],
        }
        for pair in stage2["pairs"]
    ]
    if source.get("runtime_programs") != expected_programs or any(
        not (root / value["path"]).is_file()
        or sha256_file(root / value["path"]) != value["sha256"]
        or (root / value["path"]).stat().st_size != value["bytes"]
        for value in expected_programs
    ):
        raise ValueError("Stage 3 runtime program descriptors are invalid")
    environment = source.get("environment", {})
    if (
        environment.get("gpu_name") != "NVIDIA A40"
        or environment.get("device") != "cuda:0"
        or environment.get("gpu_total_bytes", 0) < 40 * 1024**3
    ):
        raise ValueError("Stage 3 execution environment is invalid")


def validate_contracts(source: dict) -> None:
    if source.get("contracts") != {
        "capsule": {
            "migration_anchor_preserved": True,
            "served_kv_target": "theta11",
            "storage_layout": (
                "layer-major unpadded FP16 [L,T,H] plus offsets"
            ),
            "execution_layout": (
                "dense length-bucketed FP16 [L,B,S,H]"
            ),
            "length_scope": "history[:-1]",
        },
        "output_extent": {
            "layout": (
                "separate contiguous unpadded FP16 [L,T,Dkv] "
                "K/V plus lengths and offsets"
            ),
            "allocation": "preallocated outside resident operator timing",
            "write": (
                "all operators use execute_into on the same destination ABI"
            ),
            "padding_published": False,
        },
        "transport_allclose": {
            "atol": 0.02,
            "rtol": 0.02,
            "finite_required": True,
        },
    }:
        raise ValueError("Stage 3 operator contracts are invalid")


def validate_materialization(source: dict, workload: dict) -> None:
    materialization = source.get("materialization", {})
    identities = [
        {
            "record_id": value["record_id"],
            "user_id": value["user_id"],
            "source_version": value["source_version"],
            "prefix_tokens": value["prefix_tokens"],
        }
        for value in sorted(
            (
                record
                for record in workload["records"]
                if record["evaluation_role"] == "program_selection"
            ),
            key=lambda record: record["record_id"],
        )
    ]
    identity_sha256 = hashlib.sha256(
        json.dumps(
            identities,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    expected_cohorts = [
        {
            "source_version": source_version,
            "records": sum(
                value["source_version"] == source_version
                for value in identities
            ),
            "valid_tokens": sum(
                value["prefix_tokens"]
                for value in identities
                if value["source_version"] == source_version
            ),
        }
        for source_version in ("theta0", "theta4", "theta10")
    ]
    if (
        materialization.get("records") != 60
        or materialization.get("valid_tokens") != VALID_TOKENS
        or materialization.get("logical_capsule_bytes") != 1_443_184_640
        or materialization.get("dtype") != "float16"
        or materialization.get("labels_used") is not False
        or materialization.get("final_test_evaluated") is not False
        or materialization.get("record_identity_sha256")
        != identity_sha256
        or [
            {
                "source_version": value.get("source_version"),
                "records": value.get("records"),
                "valid_tokens": value.get("valid_tokens"),
            }
            for value in materialization.get("cohorts", [])
        ]
        != expected_cohorts
    ):
        raise ValueError("Stage 3 materialized selection distribution is invalid")


def validate_correctness(source: dict) -> dict:
    layouts = source.get("correctness_by_layout", {})
    if set(layouts) != LAYOUTS:
        raise ValueError("Stage 3 correctness layouts are incomplete")
    maximum = {
        "packed_from_reference": 0.0,
        "fused_from_reference": 0.0,
        "fused_from_packed": 0.0,
    }
    for layout, value in layouts.items():
        operator_keys = {
            "reference_fp32",
            "packed_fp16",
            "fused_fp16",
        }
        if (
            value.get("records") != 60
            or value.get("valid_tokens") != VALID_TOKENS
            or value.get("valid_fp16_kv_elements") != VALID_ELEMENTS
            or value.get("source_padding_nonzero") != 0
            or set(value.get("dense_output_padding_nonzero", {}))
            != operator_keys
            or set(value.get("finite", {})) != operator_keys
            or set(value.get("destination_pointer_preserved", {}))
            != operator_keys
            or any(value.get("dense_output_padding_nonzero", {}).values())
            or not all(value.get("finite", {}).values())
            or not all(value.get("destination_pointer_preserved", {}).values())
            or value.get("output_contract")
            != (
                "separate contiguous unpadded FP16 [L,T,Dkv] K/V "
                "with lengths and offsets"
            )
        ):
            raise ValueError(f"Stage 3 layout contract failed: {layout}")
        differences = value.get("differences", {})
        expected = {
            "packed_from_reference",
            "fused_from_reference",
            "fused_from_packed",
            "reference_dense_extent_identity",
            "packed_dense_extent_identity",
            "fused_dense_extent_identity",
        }
        if (
            set(differences) != expected
            or any(
                difference.get("elements") != VALID_ELEMENTS
                or difference.get("mismatched_elements") != 0
                or not all(
                    math.isfinite(difference.get(metric, math.nan))
                    and difference.get(metric, -1) >= 0
                    for metric in ("max_abs", "rms", "fro_relative")
                )
                for difference in differences.values()
            )
            or any(
                differences[name]["max_abs"] != 0.0
                for name in (
                    "reference_dense_extent_identity",
                    "packed_dense_extent_identity",
                    "fused_dense_extent_identity",
                )
            )
        ):
            raise ValueError(f"Stage 3 numerical correctness failed: {layout}")
        for name in maximum:
            maximum[name] = max(
                maximum[name],
                differences[name]["max_abs"],
            )
    if maximum["packed_from_reference"] > 0.02 or maximum[
        "fused_from_reference"
    ] > 0.02:
        raise ValueError("Stage 3 transport tolerance was exceeded")
    return {
        "layouts": len(layouts),
        "records_per_layout": 60,
        "valid_tokens_per_layout": VALID_TOKENS,
        "valid_fp16_kv_elements_per_layout": VALID_ELEMENTS,
        "total_valid_element_comparisons_per_path": (
            len(layouts) * VALID_ELEMENTS
        ),
        "all_source_padding_zero": True,
        "all_dense_output_padding_zero": True,
        "all_outputs_finite": True,
        "all_destination_pointers_preserved": True,
        "all_dense_extent_outputs_identical": True,
        "transport_mismatched_elements": 0,
        "maximum_absolute_difference": maximum,
    }


def validate_timing(value: dict, repeats: int) -> None:
    samples = value.get("values_ms", [])
    peaks = value.get("temporary_peak_bytes", [])
    if (
        len(samples) != repeats
        or len(peaks) != repeats
        or any(
            not isinstance(sample, (int, float))
            or not math.isfinite(sample)
            or sample <= 0
            for sample in samples
        )
        or any(
            not isinstance(peak, int) or isinstance(peak, bool) or peak < 0
            for peak in peaks
        )
    ):
        raise ValueError("Stage 3 timing samples are invalid")
    mean = statistics.fmean(samples)
    cv = statistics.pstdev(samples) / mean if repeats > 1 else 0.0
    expected = {
        "median_ms": statistics.median(samples),
        "mean_ms": mean,
        "cv": cv,
        "maximum_temporary_peak_bytes": max(peaks),
    }
    if any(
        not math.isclose(
            value.get(name, math.nan),
            result,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for name, result in expected.items()
    ):
        raise ValueError("Stage 3 timing aggregates do not match samples")


def validate_candidates(source: dict) -> tuple[list[dict], dict]:
    grid = source.get("candidate_grid", {})
    if grid != {
        "batch_sizes": list(BATCH_SIZES),
        "bucket_widths": list(BUCKET_WIDTHS),
        "operators": list(OPERATORS),
        "candidate_count": 18,
        "candidate_order_seed": 73421,
    }:
        raise ValueError("Stage 3 candidate grid is invalid")
    candidates = source.get("candidate_screen", [])
    expected_ids = {
        f"{operator}_b{batch}_w{bucket}"
        for operator in OPERATORS
        for batch in BATCH_SIZES
        for bucket in BUCKET_WIDTHS
    }
    if (
        len(candidates) != 18
        or {value.get("candidate_id") for value in candidates}
        != expected_ids
        or sorted(value.get("screen_position") for value in candidates)
        != list(range(18))
    ):
        raise ValueError("Stage 3 candidate screen is incomplete")
    ordered_candidates = [
        f"{operator}_b{batch}_w{bucket}"
        for batch in BATCH_SIZES
        for bucket in BUCKET_WIDTHS
        for operator in OPERATORS
    ]
    order = np.random.default_rng(73421).permutation(
        len(ordered_candidates)
    )
    expected_order = [
        ordered_candidates[int(index)]
        for index in order
    ]
    if [
        value["candidate_id"]
        for value in sorted(
            candidates,
            key=lambda value: value["screen_position"],
        )
    ] != expected_order:
        raise ValueError("Stage 3 candidate order differs from the frozen seed")
    compact = []
    for value in candidates:
        packing = value.get("packing", {})
        timing = value.get("screen_timing", {})
        validate_timing(timing, 1)
        if (
            value.get("candidate_id")
            != (
                f"{value.get('operator')}_b{value.get('batch_size')}_"
                f"w{value.get('bucket_width')}"
            )
            or value.get("operator") not in OPERATORS
            or value.get("batch_size") not in BATCH_SIZES
            or value.get("bucket_width") not in BUCKET_WIDTHS
            or packing.get("records") != 60
            or packing.get("logical_tokens") != VALID_TOKENS
            or packing.get("batch_size") != value.get("batch_size")
            or packing.get("bucket_width") != value.get("bucket_width")
            or value.get("correctness_layout")
            != f"b{value['batch_size']}_w{value['bucket_width']}"
            or not isinstance(packing.get("batches"), int)
            or isinstance(packing.get("batches"), bool)
            or packing["batches"] < 1
            or not isinstance(packing.get("padding_tokens"), int)
            or isinstance(packing.get("padding_tokens"), bool)
            or packing["padding_tokens"] < 0
            or not isinstance(packing.get("allocated_tokens"), int)
            or isinstance(packing.get("allocated_tokens"), bool)
            or packing["allocated_tokens"] < 1
            or packing.get("allocated_tokens", 0)
            != VALID_TOKENS + packing.get("padding_tokens", -1)
            or not math.isclose(
                packing.get("padding_fraction", math.nan),
                packing["padding_tokens"] / packing["allocated_tokens"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Stage 3 candidate record is invalid")
        compact.append(
            {
                "candidate_id": value["candidate_id"],
                "operator": value["operator"],
                "batch_size": value["batch_size"],
                "bucket_width": value["bucket_width"],
                "screen_position": value["screen_position"],
                "screen_ms": timing["median_ms"],
                "batches": packing["batches"],
                "allocated_tokens": packing["allocated_tokens"],
                "padding_tokens": packing["padding_tokens"],
                "padding_fraction": packing["padding_fraction"],
                "maximum_temporary_peak_bytes": timing[
                    "maximum_temporary_peak_bytes"
                ],
            }
        )
    finalists = source.get("finalist_timings", {})
    procedure = source.get("selection_procedure", {})
    top_three = procedure.get("top_three_screen_candidates", [])
    ranked = sorted(
        candidates,
        key=lambda value: (
            value["screen_timing"]["median_ms"],
            value["screen_timing"]["maximum_temporary_peak_bytes"],
            value["packing"]["padding_tokens"],
        ),
    )
    expected_top_three = [
        value["candidate_id"]
        for value in ranked[:3]
    ]
    fastest_per_operator = {
        operator: min(
            (
                value
                for value in candidates
                if value["operator"] == operator
            ),
            key=lambda value: value["screen_timing"]["median_ms"],
        )["candidate_id"]
        for operator in OPERATORS
    }
    expected_finalists = set(expected_top_three) | set(
        fastest_per_operator.values()
    )
    for value in finalists.values():
        validate_timing(value, 3)
    if (
        top_three != expected_top_three
        or set(finalists) != expected_finalists
        or set(procedure.get("fully_measured_candidates", []))
        != set(finalists)
    ):
        raise ValueError("Stage 3 finalist measurements are invalid")
    return compact, finalists


def validate_selection(
    source: dict,
    finalists: dict,
    candidates: list[dict],
) -> dict:
    selection = source.get("selection", {})
    stability = selection.get("fused_stability_gate", {})
    candidates_by_id = {
        value["candidate_id"]: value
        for value in source["candidate_screen"]
    }
    top_three = source["selection_procedure"][
        "top_three_screen_candidates"
    ]
    selected_id = min(
        top_three,
        key=lambda value: (
            finalists[value]["median_ms"],
            finalists[value]["maximum_temporary_peak_bytes"],
            candidates_by_id[value]["packing"]["padding_tokens"],
        ),
    )
    selected_candidate = candidates_by_id[selected_id]
    fastest_packed_id = min(
        (
            value
            for value in finalists
            if candidates_by_id[value]["operator"] == "packed_fp16"
        ),
        key=lambda value: finalists[value]["median_ms"],
    )
    fastest_fused_id = min(
        (
            value
            for value in finalists
            if candidates_by_id[value]["operator"] == "fused_fp16"
        ),
        key=lambda value: finalists[value]["median_ms"],
    )
    tested_fused_id = (
        selected_id
        if selected_candidate["operator"] == "fused_fp16"
        else fastest_fused_id
    )
    packed_samples = finalists[fastest_packed_id]["values_ms"]
    fused_samples = finalists[tested_fused_id]["values_ms"]
    stable = (
        finalists[tested_fused_id]["median_ms"]
        < finalists[fastest_packed_id]["median_ms"]
        and max(fused_samples) < min(packed_samples)
    )
    fallback_applied = selected_candidate["operator"] == "fused_fp16" and not stable
    if fallback_applied:
        selected_id = fastest_packed_id
        selected_candidate = candidates_by_id[selected_id]
    expected_compact = next(
        value
        for value in candidates
        if value["candidate_id"] == selected_id
    )
    if (
        selected_id != "fused_fp16_b4_w32"
        or selection.get("candidate_id") != selected_id
        or selection.get("operator") != selected_candidate["operator"]
        or selection.get("operator_implementation")
        != "FusedMigrationOperator"
        or selection.get("batch_size") != expected_compact["batch_size"]
        or selection.get("bucket_width") != expected_compact["bucket_width"]
        or selection.get("timing") != finalists[selected_id]
        or stability.get("fastest_packed_candidate")
        != fastest_packed_id
        or stability.get("fastest_fused_candidate") != fastest_fused_id
        or stability.get("tested_fused_candidate") != tested_fused_id
        or stability.get("all_fused_samples_below_all_packed_samples")
        is not stable
        or stability.get("packed_fallback_applied") is not fallback_applied
        or not math.isclose(
            stability.get("fused_speedup_over_packed", math.nan),
            finalists[fastest_packed_id]["median_ms"]
            / finalists[tested_fused_id]["median_ms"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or stability.get("packed_median_ms")
        != finalists[fastest_packed_id]["median_ms"]
        or stability.get("fused_median_ms")
        != finalists[tested_fused_id]["median_ms"]
    ):
        raise ValueError("Stage 3 frozen operator selection is invalid")
    return selection


def validate_profile(source: dict) -> dict:
    profile = source.get("representative_profile", {})
    profiles = profile.get("profiles", {})
    fused_name = next(
        (
            name
            for name in profiles
            if name.startswith("fused_triton_")
        ),
        None,
    )
    if (
        profile.get("source_version") != "theta0"
        or profile.get("records") != 4
        or profile.get("sequence_width") != 2047
        or profile.get("lengths") != [2047] * 4
        or profile.get("valid_tokens") != 8188
        or profile.get("capsule_bytes") != 134_152_224
        or profile.get("output_extent_bytes") != 268_304_456
        or profile.get("output_layout")
        != "separate contiguous unpadded FP16 [L,T,Dkv] K/V"
        or set(profiles)
        != {"reference_fp32", "packed_float16", fused_name}
        or fused_name is None
        or any(
            len(value.get("latency_ms", [])) != 20
            or value.get("median_ms", 0) <= 0
            or len(value.get("temporary_peak_bytes", [])) != 20
            for value in profiles.values()
        )
        or profiles[fused_name]["maximum_temporary_peak_bytes"] != 0
        or profiles["packed_float16"]["maximum_temporary_peak_bytes"] <= 0
        or profiles[fused_name]["median_ms"]
        >= profiles["packed_float16"]["median_ms"]
    ):
        raise ValueError("Stage 3 representative operator profile is invalid")
    for value in profiles.values():
        samples = value["latency_ms"]
        peaks = value["temporary_peak_bytes"]
        if (
            any(
                not isinstance(sample, (int, float))
                or not math.isfinite(sample)
                or sample <= 0
                for sample in samples
            )
            or any(
                not isinstance(peak, int)
                or isinstance(peak, bool)
                or peak < 0
                for peak in peaks
            )
            or not math.isclose(
                value["median_ms"],
                statistics.median(samples),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or value["maximum_temporary_peak_bytes"] != max(peaks)
        ):
            raise ValueError("Stage 3 profile aggregates do not match samples")
    breakdown = profile.get("epilogue_breakdown", {})
    if (
        "reference_fp32_extent" not in breakdown
        or "packed_float16_extent" not in breakdown
        or f"{fused_name}_extent" not in breakdown
        or "split_compact_extent_write"
        not in breakdown["packed_float16_extent"]["stages"]
        or "affine_bias_length_split_direct_extent_write"
        not in breakdown[f"{fused_name}_extent"]["stages"]
    ):
        raise ValueError("Stage 3 epilogue profile is incomplete")
    for value in breakdown.values():
        total = value.get("total_ms", [])
        if (
            len(total) != 20
            or any(
                not isinstance(sample, (int, float))
                or not math.isfinite(sample)
                or sample <= 0
                for sample in total
            )
            or not math.isclose(
                value.get("total_median_ms", math.nan),
                statistics.median(total),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Stage 3 epilogue total is invalid")
        for stage in value.get("stages", {}).values():
            samples = stage.get("values_ms", [])
            if (
                len(samples) != 20
                or any(
                    not isinstance(sample, (int, float))
                    or not math.isfinite(sample)
                    or sample <= 0
                    for sample in samples
                )
                or not math.isclose(
                    stage.get("median_ms", math.nan),
                    statistics.median(samples),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("Stage 3 epilogue stage is invalid")
    temporary = profile.get("logical_temporary_inventory", {})
    if (
        temporary.get("fused_global_temporary_bytes") != 0
        or temporary.get("packed_projected_concat_fp16", 0) <= 0
        or temporary.get("fused_components_separately_timed") is not False
    ):
        raise ValueError("Stage 3 temporary inventory is invalid")
    return {
        "source_version": profile["source_version"],
        "records": profile["records"],
        "sequence_width": profile["sequence_width"],
        "lengths": profile["lengths"],
        "valid_tokens": profile["valid_tokens"],
        "capsule_bytes": profile["capsule_bytes"],
        "output_extent_bytes": profile["output_extent_bytes"],
        "latency": {
            name: {
                "values_ms": value["latency_ms"],
                "median_ms": value["median_ms"],
                "maximum_temporary_peak_bytes": value[
                    "maximum_temporary_peak_bytes"
                ],
            }
            for name, value in profiles.items()
        },
        "epilogue_stage_medians_ms": {
            name: {
                stage: values["median_ms"]
                for stage, values in profile_value["stages"].items()
            }
            for name, profile_value in breakdown.items()
        },
        "logical_temporary_inventory": temporary,
    }


def build_summary(
    root: Path,
    source_path: Path,
    source: dict,
    workload: dict,
    stage2: dict,
) -> dict:
    validate_identity(root, source, workload, stage2)
    validate_contracts(source)
    validate_materialization(source, workload)
    correctness = validate_correctness(source)
    candidates, finalists = validate_candidates(source)
    selection = validate_selection(source, finalists, candidates)
    profile = validate_profile(source)
    negative = source.get("retained_negative_layout", {})
    if (
        negative.get("protocol")
        != "kuairand_long_context_4plus12_cohort_jagged_system_v3"
        or sha256_file(root / negative["path"]) != negative.get("sha256")
    ):
        raise ValueError("Stage 3 retained negative-layout evidence is invalid")
    return {
        "protocol": PROTOCOL,
        "status": "stage3_frozen",
        "study_stage": "single_configuration_seed0_development",
        "source_result": {
            "path": str(source_path),
            "sha256": sha256_file(root / source_path),
            "protocol": SOURCE_PROTOCOL,
        },
        "parent_blueprint": {
            **source["blueprint"],
            "hash_scope": (
                "blueprint bytes used by Stage 3 before the downstream "
                "Stage-3 completion amendment"
            ),
        },
        "workload": {
            "path": str(WORKLOAD),
            "file_sha256": sha256_file(root / WORKLOAD),
            "content_sha256": workload["content_sha256"],
            "program_selection_records": 60,
            "program_selection_prefix_tokens": VALID_TOKENS,
        },
        "stage2_summary": {
            "path": str(STAGE2),
            "sha256": sha256_file(root / STAGE2),
            "protocol": STAGE2_PROTOCOL,
        },
        "measurement_boundary": {
            "execution": (
                "GPU-resident FP16 capsule and program through complete "
                "preallocated unpadded FP16 K/V extent write"
            ),
            "source_role": "program_selection",
            "source_records": 60,
            "source_valid_tokens": VALID_TOKENS,
            "source_versions": ["theta0", "theta4", "theta10"],
            "operators": [
                "reference_fp32",
                "packed_fp16",
                "fused_fp16",
            ],
            "target_allocation_timed": False,
            "source_io_timed": False,
            "destination_transport_timed": False,
            "final_test_evaluated": False,
            "recommendation_labels_used": False,
            "gpu_name": source["environment"]["gpu_name"],
        },
        "contracts": source["contracts"],
        "materialization": source["materialization"],
        "candidate_grid": source["candidate_grid"],
        "correctness": correctness,
        "candidate_screen": candidates,
        "finalist_timings": finalists,
        "selection": selection,
        "selection_procedure": source["selection_procedure"],
        "representative_profile": profile,
        "retained_negative_layout": negative,
        "downstream_rule": {
            "stage4_default_operator": selection["operator"],
            "stage4_default_batch_size": selection["batch_size"],
            "stage4_default_bucket_width": selection["bucket_width"],
            "retune_requirement": (
                "Stage 4 still searches the complete frozen grid "
                "independently per method, destination, and GPU count"
            ),
            "common_api": (
                "all compiled operator paths consume dense length-bucketed "
                "capsules and execute_into the same contiguous unpadded "
                "K/V extent ABI"
            ),
            "claim_boundary": (
                "Stage 3 is resident operator evidence; it is not a "
                "full-cohort, HBM, DRAM, or end-to-end speedup"
            ),
            "layout_boundary": (
                "retain prior jagged/page exactness and negative performance "
                "result; do not reopen layout search without a new measured "
                "bottleneck"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    source_path = Path(args.source_result)
    output_path = Path(args.output)
    source = json.loads((root / source_path).read_text())
    workload = json.loads((root / WORKLOAD).read_text())
    stage2 = json.loads((root / STAGE2).read_text())
    payload = canonical_json_bytes(
        build_summary(
            root,
            source_path,
            source,
            workload,
            stage2,
        )
    )
    resolved = root / output_path
    if args.check:
        if not resolved.is_file() or resolved.read_bytes() != payload:
            raise RuntimeError("Stage 3 frozen summary differs from source result")
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
                "output": str(output_path),
                "sha256": sha256_bytes(payload),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
