from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import numpy as np

from hstu_kvcache.migration import Stage4SourceManifest, sha256_file

PROTOCOL = "cohortkv_single_config_stage4_frozen_v1"
SOURCE_PROTOCOL = "cohortkv_single_config_stage4_system_v1"
SOURCE_MANIFEST_PROTOCOL = "cohortkv_stage4_source_manifest_v1"
PARENT_PROTOCOL = "cohortkv_single_config_full_chain_development_v1"
STAGE2_PROTOCOL = "cohortkv_single_config_stage2_frozen_v1"
STAGE3_PROTOCOL = "cohortkv_single_config_stage3_frozen_v1"
SOURCE_RESULT = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_system_seed0.json"
)
SOURCE_MANIFEST = Path(
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0/"
    "single_config_v1/source_shards/source_manifest.json"
)
BLUEPRINT = Path("configs/cohortkv_single_config_v1/blueprint.json")
WORKLOAD = Path("configs/cohortkv_single_config_v1/workload_manifest.json")
STAGE2 = Path("configs/cohortkv_single_config_v1/stage2_compiler_summary.json")
STAGE3 = Path("configs/cohortkv_single_config_v1/stage3_operator_summary.json")
OUTPUT = Path("configs/cohortkv_single_config_v1/stage4_system_summary.json")
METHODS = (
    "compiled",
    "selective_contiguous",
    "exact",
    "residual_p",
    "no_transform",
)
PRIMARY_METHODS = ("compiled", "selective_contiguous", "exact")
CONTROL_METHODS = ("residual_p", "no_transform")
DESTINATIONS = ("hbm", "dram")
GPU_COUNTS = (1, 2, 4)
BATCH_SIZES = (1, 2, 4)
BUCKET_WIDTHS = (16, 32, 64)
INFLIGHT_DEPTHS = (2, 3, 4)
COMPILED_OPERATORS = ("packed_fp16", "fused_fp16")
EXACT_COMPUTE = ("bfloat16", "float32")
SELECTION_SEED = 73421
WARMUP_RUNS = 1
MEASURED_REPEATS = 3
RECORDS = 682
PREFIX_TOKENS = 1_087_785
SELECTION_RECORDS = 60
SELECTION_TOKENS = 88_085
NUM_LAYERS = 16
KV_WIDTH = 512
OUTPUT_BYTES = 35_644_538_880
FULL_VALID_ELEMENTS = 17_822_269_440
SELECTION_VALID_ELEMENTS = 1_443_184_640
WORKLOAD_CONTENT_SHA256 = (
    "41b7ad10a8dc3a05ce99342a0d73a09e09847ddf42b9111d318b3ddd3c62a910"
)
GPU_NAME = "NVIDIA A40"
GPU_TOTAL_BYTES = 47_699_722_240
SOURCE_REPRESENTATIONS = {
    "compiled": ("normalized_capsule_fp16",),
    "selective_contiguous": ("old_kv_fp16", "raw_history"),
    "exact": ("raw_history",),
    "residual_p": ("raw_history", "residual_hidden_suffix_bf16"),
    "no_transform": ("old_kv_fp16",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", default=str(SOURCE_RESULT))
    parser.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
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


def point_key(method: str, destination: str, gpu_count: int) -> str:
    return f"{method}:{destination}:{gpu_count}"


def required_keys() -> set[str]:
    return {
        point_key(method, destination, gpu_count)
        for method in METHODS
        for destination in DESTINATIONS
        for gpu_count in GPU_COUNTS
    }


def candidate_id(value: dict[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(value))[:16]


def candidate_grid(method: str) -> list[dict[str, object]]:
    candidates = []
    for batch, bucket, inflight in itertools.product(
        BATCH_SIZES,
        BUCKET_WIDTHS,
        INFLIGHT_DEPTHS,
    ):
        common = {
            "batch_size": batch,
            "length_bucket_width": bucket,
            "max_inflight": inflight,
        }
        if method == "compiled":
            candidates.extend(
                {**common, "compiled_operator": operator}
                for operator in COMPILED_OPERATORS
            )
        elif method == "exact":
            candidates.extend(
                {**common, "exact_compute": compute}
                for compute in EXACT_COMPUTE
            )
        else:
            candidates.append(common)
    order = np.random.default_rng(SELECTION_SEED).permutation(
        len(candidates)
    )
    return [candidates[int(index)] for index in order]


def finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def finite_positive(value: object) -> bool:
    return finite_nonnegative(value) and value > 0


def validate_correctness(value: dict, expected_elements: int) -> None:
    if (
        value.get("finite") is not True
        or value.get("allclose") is not True
        or value.get("record_order_valid") is not True
        or value.get("lengths_offsets_valid") is not True
        or value.get("reference_kind")
        != (
            "same selected method and numeric path resident on the same "
            "serialized source representation"
        )
        or value.get("atol") != 0.02
        or value.get("rtol") != 0.02
        or value.get("valid_element_count") != expected_elements
        or not finite_nonnegative(value.get("max_abs_error"))
    ):
        raise ValueError("Stage 4 numerical or layout correctness is invalid")


def validate_job(value: dict, gpu_count: int, expected_elements: int | None) -> None:
    if (
        not finite_positive(value.get("elapsed_seconds"))
        or value.get("logical_input_bytes", 0) < 1
        or value.get("physical_input_bytes", 0) < 1
        or value.get("logical_output_bytes", 0) < 1
        or value.get("physical_output_bytes", 0) < 1
        or len(value.get("per_gpu", [])) != gpu_count
        or [device.get("index") for device in value["per_gpu"]]
        != list(range(gpu_count))
        or value.get("manifest", {}).get("protocol")
        != "streamkv_destination_manifest_v1"
        or value.get("manifest", {}).get("record_count", 0) < 1
        or value.get("manifest", {}).get("prefix_tokens", 0) < 1
    ):
        raise ValueError("Stage 4 job record is invalid")
    required_breakdown = {
        "source_read",
        "h2d",
        "compute",
        "d2h",
        "stage",
        "commit",
        "elapsed",
    }
    breakdown = value.get("timing_breakdown", {})
    if (
        not required_breakdown.issubset(breakdown)
        or any(
            not finite_nonnegative(breakdown[name])
            for name in required_breakdown
        )
        or not math.isclose(
            breakdown["elapsed"],
            value["elapsed_seconds"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Stage 4 job timing breakdown is invalid")
    if expected_elements is None:
        if value.get("correctness") is not None:
            raise ValueError("Stage 4 timed job unexpectedly carries correctness")
    else:
        validate_correctness(value.get("correctness", {}), expected_elements)


def validate_identity(
    root: Path,
    source_path: Path,
    source: dict,
    source_manifest_path: Path,
    manifest: Stage4SourceManifest,
    workload: dict,
    stage2: dict,
    stage3: dict,
) -> None:
    if (
        source.get("protocol") != SOURCE_PROTOCOL
        or source.get("parent_protocol") != PARENT_PROTOCOL
        or source.get("status") != "stage4_complete"
        or source.get("study_stage")
        != "single_configuration_seed0_development"
        or source.get("seed") != 0
        or source.get("labels_used") is not False
        or source.get("coverage")
        != {
            "required_points": 30,
            "tuned_points": 30,
            "full_points": 30,
            "primary_full_points": 18,
            "control_full_points": 12,
        }
    ):
        raise ValueError("Stage 4 source result identity is invalid")
    descriptors = (
        (source["workload_manifest"], WORKLOAD, None),
        (source["stage2_summary"], STAGE2, STAGE2_PROTOCOL),
        (source["stage3_summary"], STAGE3, STAGE3_PROTOCOL),
        (
            source["source_manifest"],
            source_manifest_path,
            SOURCE_MANIFEST_PROTOCOL,
        ),
    )
    for descriptor, expected_path, protocol in descriptors:
        if (
            descriptor.get("path") != str(expected_path)
            or descriptor.get("sha256")
            != sha256_file(root / expected_path)
            or (
                protocol is not None
                and descriptor.get("protocol") != protocol
            )
        ):
            raise ValueError("Stage 4 frozen input descriptor is invalid")
    blueprint = source.get("blueprint", {})
    if (
        blueprint.get("path") != str(BLUEPRINT)
        or not isinstance(blueprint.get("sha256"), str)
        or len(blueprint["sha256"]) != 64
    ):
        raise ValueError("Stage 4 parent blueprint descriptor is invalid")
    if (
        workload.get("content_sha256") != WORKLOAD_CONTENT_SHA256
        or workload.get("summary", {}).get("records") != RECORDS
        or workload.get("summary", {}).get("prefix_tokens") != PREFIX_TOKENS
        or source["workload_manifest"].get("content_sha256")
        != WORKLOAD_CONTENT_SHA256
        or stage2.get("protocol") != STAGE2_PROTOCOL
        or stage2.get("status") != "stage2_frozen"
        or stage3.get("protocol") != STAGE3_PROTOCOL
        or stage3.get("status") != "stage3_frozen"
    ):
        raise ValueError("Stage 4 upstream protocol identity is invalid")
    if (
        manifest.protocol != SOURCE_MANIFEST_PROTOCOL
        or manifest.record_count != RECORDS
        or manifest.prefix_tokens != PREFIX_TOKENS
        or manifest.num_layers != NUM_LAYERS
        or manifest.kv_width != KV_WIDTH
        or manifest.workload_content_sha256 != WORKLOAD_CONTENT_SHA256
        or manifest.workload_file_sha256 != sha256_file(root / WORKLOAD)
    ):
        raise ValueError("Stage 4 source manifest identity is invalid")
    expected_records = [
        (
            record["record_id"],
            record["user_id"],
            record["evaluation_role"],
            record["source_version"],
            record["target_version"],
            record["prefix_tokens"],
        )
        for record in workload["records"]
    ]
    actual_records = [
        (
            record.record_id,
            record.user_id,
            record.evaluation_role,
            record.source_version,
            record.target_version,
            record.prefix_tokens,
        )
        for record in manifest.records
    ]
    if actual_records != expected_records:
        raise ValueError("Stage 4 source record identities changed")
    environment = source.get("environment", {})
    gpus = environment.get("gpus", [])
    if (
        environment.get("gpu_count") != 4
        or len(gpus) != 4
        or [gpu.get("index") for gpu in gpus] != list(range(4))
        or any(
            gpu.get("name") != GPU_NAME
            or gpu.get("total_bytes") != GPU_TOTAL_BYTES
            for gpu in gpus
        )
        or not isinstance(environment.get("python"), str)
        or not isinstance(environment.get("torch"), str)
        or not isinstance(environment.get("cuda_runtime"), str)
        or not isinstance(environment.get("cuda_driver"), str)
        or environment.get("source_storage")
        != {
            "mount": "/data",
            "device": "/dev/nvme2n1p1",
            "device_model": "INTEL SSDPF2KX038XZ",
            "filesystem": "ext4",
            "free_bytes_before_materialization": 433_665_679_360,
        }
        or "source shards reopen and decode every repetition"
        not in environment.get("page_cache_condition", "")
    ):
        raise ValueError("Stage 4 GPU environment is invalid")
    implementation = source.get("implementation", {})
    full_phase = implementation.get("full_phase", {})
    files = full_phase.get("files", [])
    if (
        not isinstance(full_phase.get("repository_commit"), str)
        or len(full_phase["repository_commit"]) != 40
        or not files
        or len({value.get("path") for value in files}) != len(files)
        or any(
            not (root / value["path"]).is_file()
            or (root / value["path"]).stat().st_size != value.get("bytes")
            or sha256_file(root / value["path"]) != value.get("sha256")
            for value in files
        )
        or full_phase.get("code_snapshot_sha256")
        != sha256_bytes(canonical_json_bytes(files))
        or "source-wave peak accounting"
        not in implementation.get("tuning_to_full_amendment", "")
    ):
        raise ValueError("Stage 4 full-phase implementation snapshot is invalid")
    if not (root / source_path).is_file():
        raise ValueError("Stage 4 source result is missing")


def validate_source_materialization(
    root: Path,
    source_manifest_path: Path,
    manifest: Stage4SourceManifest,
) -> dict:
    creation = manifest.creation
    if (
        creation.get("protocol")
        != "cohortkv_stage4_source_materialization_v1"
        or creation.get("capture_batch_size") != 4
        or creation.get("residual_start_layer") != 8
        or creation.get("residual_storage_dtype") != "bfloat16"
        or creation.get("residual_source_versions")
        != ["theta0", "theta10"]
        or not finite_positive(creation.get("elapsed_seconds"))
    ):
        raise ValueError("Stage 4 source materialization contract is invalid")
    preflight = creation.get("source_preflight", {})
    if (
        preflight.get("mount") != "/data"
        or preflight.get("source") != "/dev/nvme2n1p1"
        or preflight.get("device_model") != "INTEL SSDPF2KX038XZ"
        or preflight.get("filesystem") != "ext4"
        or preflight.get("passed") is not True
        or preflight.get("observed_free_bytes", 0)
        < preflight.get("minimum_free_bytes", 1)
    ):
        raise ValueError("Stage 4 source storage preflight is invalid")
    expected_logical = {
        "normalized_capsule_fp16": 17_822_269_440,
        "old_kv_fp16": 35_644_538_880,
        "raw_history": 21_755_700,
        "residual_hidden_suffix_bf16": 6_255_345_664,
    }
    if creation.get("logical_bytes") != expected_logical:
        raise ValueError("Stage 4 source logical bytes changed")
    manifest_dir = root / source_manifest_path.parent
    checked_bytes = 0
    checked_shards = 0
    for record in manifest.records:
        for shard in record.shards:
            path = manifest_dir / shard.path
            if (
                not path.is_file()
                or path.stat().st_size != shard.physical_bytes
                or sha256_file(path) != shard.sha256
            ):
                raise ValueError(f"Stage 4 source shard is invalid: {path}")
            checked_bytes += shard.physical_bytes
            checked_shards += 1
    physical = creation.get("physical_bytes", {})
    if checked_bytes != sum(physical.values()):
        raise ValueError("Stage 4 source physical-byte inventory is invalid")
    return {
        "record_count": manifest.record_count,
        "prefix_tokens": manifest.prefix_tokens,
        "shard_count": checked_shards,
        "logical_bytes": creation["logical_bytes"],
        "physical_bytes": physical,
        "physical_bytes_verified": checked_bytes,
        "elapsed_seconds": creation["elapsed_seconds"],
        "source_preflight": preflight,
        "residual_start_layer": creation["residual_start_layer"],
        "residual_storage_dtype": creation["residual_storage_dtype"],
        "residual_source_versions": creation["residual_source_versions"],
    }


def validate_tuning(source: dict) -> list[dict]:
    tuning = source.get("runtime_tuning", {})
    if (
        tuning.get("candidate_order_seed") != SELECTION_SEED
        or tuning.get("selection_role") != "program_selection"
        or tuning.get("warmup_runs") != WARMUP_RUNS
        or tuning.get("measured_repetitions") != MEASURED_REPEATS
    ):
        raise ValueError("Stage 4 tuning protocol is invalid")
    points = tuning.get("points", [])
    if (
        len(points) != 30
        or {value.get("key") for value in points} != required_keys()
    ):
        raise ValueError("Stage 4 tuning point coverage is incomplete")
    compact = []
    for point in points:
        method = point.get("method")
        destination = point.get("destination")
        gpu_count = point.get("gpu_count")
        key = point_key(method, destination, gpu_count)
        expected_grid = candidate_grid(method)
        candidates = point.get("candidates", [])
        if (
            point.get("key") != key
            or point.get("status") != "tuned"
            or len(candidates) != len(expected_grid)
            or [value.get("runtime_config") for value in candidates]
            != expected_grid
            or [value.get("candidate_id") for value in candidates]
            != [candidate_id(value) for value in expected_grid]
        ):
            raise ValueError(f"Stage 4 candidate grid is invalid for {key}")
        warmup = point.get("source_warmup", {})
        if (
            warmup.get("record_count") != SELECTION_RECORDS
            or warmup.get("complete") is not True
            or warmup.get("logical_bytes", 0) < 1
            or warmup.get("physical_bytes", 0) < 1
            or not finite_positive(warmup.get("elapsed_seconds"))
        ):
            raise ValueError(f"Stage 4 source warmup is invalid for {key}")
        probe = point.get("pinned_extent_probe")
        if destination == "dram":
            if (
                not isinstance(probe, dict)
                or probe.get("passed") is not True
                or probe.get("maximum_extent_bytes", 0) < 1
                or not finite_positive(probe.get("elapsed_seconds"))
            ):
                raise ValueError(f"Stage 4 pinned probe is invalid for {key}")
        elif probe is not None:
            raise ValueError(f"Stage 4 HBM point has a pinned probe: {key}")
        for candidate in candidates:
            validate_correctness(
                candidate.get("correctness", {}),
                SELECTION_VALID_ELEMENTS,
            )
            validate_job(
                candidate.get("correctness_job", {}),
                gpu_count,
                SELECTION_VALID_ELEMENTS,
            )
            validate_job(
                candidate.get("screen_job", {}),
                gpu_count,
                None,
            )
            if (
                not math.isclose(
                    candidate.get("screen_elapsed_seconds", math.nan),
                    candidate["screen_job"]["elapsed_seconds"],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or candidate.get("padding_tokens", -1) < 0
                or len(candidate.get("maximum_transient_hbm_bytes", []))
                != gpu_count
                or any(
                    not finite_nonnegative(value)
                    for value in candidate["maximum_transient_hbm_bytes"]
                )
            ):
                raise ValueError(f"Stage 4 candidate record is invalid for {key}")
        ranked = sorted(
            candidates,
            key=lambda value: (
                value["screen_elapsed_seconds"],
                value["screen_job"]["peak_hbm_bytes"],
                value["padding_tokens"],
                value["candidate_id"],
            ),
        )
        finalists = ranked[:3]
        finalist_ids = [value["candidate_id"] for value in finalists]
        if point.get("finalist_ids") != finalist_ids:
            raise ValueError(f"Stage 4 finalists are invalid for {key}")
        for candidate in candidates:
            finalist = candidate.get("finalist")
            if candidate["candidate_id"] not in finalist_ids:
                if finalist is not None:
                    raise ValueError(f"Stage 4 non-finalist was measured for {key}")
                continue
            if (
                not isinstance(finalist, dict)
                or len(finalist.get("warmup_jobs", [])) != WARMUP_RUNS
                or len(finalist.get("measured_jobs", []))
                != MEASURED_REPEATS
                or len(finalist.get("samples_seconds", []))
                != MEASURED_REPEATS
            ):
                raise ValueError(f"Stage 4 finalist repeats are invalid for {key}")
            for job in finalist["warmup_jobs"] + finalist["measured_jobs"]:
                validate_job(job, gpu_count, None)
            samples = [
                value["elapsed_seconds"]
                for value in finalist["measured_jobs"]
            ]
            if (
                finalist["samples_seconds"] != samples
                or not math.isclose(
                    finalist.get("median_seconds", math.nan),
                    statistics.median(samples),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or finalist.get("peak_hbm_bytes")
                != max(
                    value["peak_hbm_bytes"]
                    for value in finalist["measured_jobs"]
                )
            ):
                raise ValueError(f"Stage 4 finalist aggregate is invalid for {key}")
        winner = min(
            finalists,
            key=lambda value: (
                value["finalist"]["median_seconds"],
                value["finalist"]["peak_hbm_bytes"],
                value["padding_tokens"],
                value["candidate_id"],
            ),
        )
        if (
            point.get("winner_id") != winner["candidate_id"]
            or point.get("winner_runtime_config")
            != winner["runtime_config"]
        ):
            raise ValueError(f"Stage 4 winner is not derived for {key}")
        compact.append(
            {
                "key": key,
                "method": method,
                "destination": destination,
                "gpu_count": gpu_count,
                "candidate_count": len(candidates),
                "candidate_order_sha256": sha256_bytes(
                    canonical_json_bytes(
                        [value["candidate_id"] for value in candidates]
                    )
                ),
                "source_warmup": warmup,
                "pinned_extent_probe": probe,
                "finalists": [
                    {
                        "candidate_id": value["candidate_id"],
                        "runtime_config": value["runtime_config"],
                        "screen_elapsed_seconds": value[
                            "screen_elapsed_seconds"
                        ],
                        "samples_seconds": value["finalist"][
                            "samples_seconds"
                        ],
                        "median_seconds": value["finalist"][
                            "median_seconds"
                        ],
                        "peak_hbm_bytes": value["finalist"][
                            "peak_hbm_bytes"
                        ],
                    }
                    for value in finalists
                ],
                "winner_id": winner["candidate_id"],
                "winner_runtime_config": winner["runtime_config"],
                "winner_selection_median_seconds": winner["finalist"][
                    "median_seconds"
                ],
            }
        )
    return compact


def expected_components(
    manifest: Stage4SourceManifest,
    method: str,
) -> list[dict[str, object]]:
    components = []
    for representation in SOURCE_REPRESENTATIONS[method]:
        shards = [
            record.shard_map[representation]
            for record in manifest.records
            if representation in record.shard_map
            and (
                method != "residual_p"
                or representation != "residual_hidden_suffix_bf16"
                or record.source_version in {"theta0", "theta10"}
            )
        ]
        components.append(
            {
                "representation": representation,
                "logical_bytes": sum(value.logical_bytes for value in shards),
                "physical_bytes": sum(value.physical_bytes for value in shards),
            }
        )
    return components


def validate_preflight(value: dict, gpu_count: int) -> None:
    jobs = value.get("per_job", [])
    if (
        value.get("passed") is not True
        or len(jobs) != 1 + WARMUP_RUNS + MEASURED_REPEATS
        or not finite_nonnegative(
            value.get("minimum_observed_free_hbm_bytes")
        )
        or not finite_nonnegative(value.get("required_peak_hbm_bytes"))
        or not finite_nonnegative(
            value.get("minimum_observed_available_host_bytes")
        )
        or not finite_nonnegative(value.get("required_peak_host_bytes"))
    ):
        raise ValueError("Stage 4 aggregate capacity preflight is invalid")
    for job in jobs:
        if (
            job.get("passed") is not True
            or len(job.get("per_gpu", [])) != gpu_count
            or any(device.get("passed") is not True for device in job["per_gpu"])
        ):
            raise ValueError("Stage 4 per-job capacity preflight is invalid")


def validate_runs(
    source: dict,
    manifest: Stage4SourceManifest,
    tuning: list[dict],
) -> tuple[list[dict], dict]:
    runs = source.get("runs", [])
    keys = [
        point_key(
            value.get("method"),
            value.get("destination"),
            value.get("gpu_count"),
        )
        for value in runs
    ]
    if len(runs) != 30 or set(keys) != required_keys() or len(set(keys)) != 30:
        raise ValueError("Stage 4 full-run coverage is incomplete")
    tuning_by_key = {value["key"]: value for value in tuning}
    implementation_snapshot_sha256 = source["implementation"]["full_phase"][
        "code_snapshot_sha256"
    ]
    compact_runs = []
    for run, key in zip(runs, keys, strict=True):
        method = run["method"]
        destination = run["destination"]
        gpu_count = run["gpu_count"]
        components = expected_components(manifest, method)
        logical_input = sum(value["logical_bytes"] for value in components)
        physical_input = sum(value["physical_bytes"] for value in components)
        per_gpu = run.get("per_gpu", [])
        timing = run.get("timing", {})
        samples = timing.get("samples_seconds", [])
        breakdown = timing.get("breakdown_seconds", {})
        expected_metadata = {
            "compiled": (True, True),
            "selective_contiguous": (False, False),
            "exact": (True, True),
            "residual_p": (True, True),
            "no_transform": (False, False),
        }[method]
        if (
            run.get("source_representations")
            != list(SOURCE_REPRESENTATIONS[method])
            or run.get("input_components") != components
            or run.get("record_count") != RECORDS
            or run.get("prefix_tokens") != PREFIX_TOKENS
            or run.get("placement_policy") != "byte_weighted_lpt"
            or run.get("logical_input_bytes") != logical_input
            or run.get("physical_input_bytes") != physical_input
            or run.get("logical_output_bytes") != OUTPUT_BYTES
            or run.get("physical_output_bytes", 0) < OUTPUT_BYTES
            or run.get("source_manifest_sha256")
            != source["source_manifest"]["sha256"]
            or run.get("implementation_snapshot_sha256")
            != implementation_snapshot_sha256
            or run.get("selected_runtime_config")
            != tuning_by_key[key]["winner_runtime_config"]
            or run.get("certificate_passed") is not expected_metadata[0]
            or run.get("publishable_sync_action") is not expected_metadata[1]
            or len(per_gpu) != gpu_count
            or [value.get("index") for value in per_gpu]
            != list(range(gpu_count))
            or sum(value.get("record_count", 0) for value in per_gpu)
            != RECORDS
            or sum(value.get("prefix_tokens", 0) for value in per_gpu)
            != PREFIX_TOKENS
            or sum(value.get("logical_input_bytes", 0) for value in per_gpu)
            != logical_input
            or sum(value.get("logical_output_bytes", 0) for value in per_gpu)
            != OUTPUT_BYTES
            or sum(value.get("physical_input_bytes", 0) for value in per_gpu)
            != physical_input
            or sum(value.get("physical_output_bytes", 0) for value in per_gpu)
            != run["physical_output_bytes"]
            or len(samples) != MEASURED_REPEATS
            or any(not finite_positive(value) for value in samples)
            or not math.isclose(
                timing.get("median_seconds", math.nan),
                statistics.median(samples),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                run.get("records_per_second", math.nan),
                RECORDS / timing["median_seconds"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                run.get("tokens_per_second", math.nan),
                PREFIX_TOKENS / timing["median_seconds"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"Stage 4 aggregate run is invalid for {key}")
        required_breakdown = {
            "source_read",
            "h2d",
            "compute",
            "d2h",
            "stage",
            "commit",
            "elapsed",
        }
        if (
            not required_breakdown.issubset(breakdown)
            or any(
                not finite_nonnegative(breakdown[name])
                for name in required_breakdown
            )
            or not math.isclose(
                breakdown["elapsed"],
                timing["median_seconds"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or any(
                not finite_nonnegative(run.get(name))
                for name in (
                    "peak_hbm_bytes",
                    "peak_host_bytes",
                    "peak_source_resident_bytes",
                    "peak_staging_bytes",
                    "peak_publication_queue_bytes",
                    "load_imbalance_ratio",
                )
            )
        ):
            raise ValueError(f"Stage 4 run timing or peaks are invalid for {key}")
        run_manifest = run.get("manifest", {})
        if run_manifest != {
            "protocol": "streamkv_destination_manifest_v1",
            "record_count": RECORDS,
            "prefix_tokens": PREFIX_TOKENS,
            "workload_content_sha256": WORKLOAD_CONTENT_SHA256,
            "complete": True,
            "duplicate_free": True,
        }:
            raise ValueError(f"Stage 4 publication manifest is invalid for {key}")
        validate_correctness(run.get("correctness", {}), FULL_VALID_ELEMENTS)
        validate_preflight(run.get("capacity_preflight", {}), gpu_count)
        compact_runs.append(run)
    by_key = {
        point_key(run["method"], run["destination"], run["gpu_count"]): run
        for run in compact_runs
    }
    comparisons = []
    for destination in DESTINATIONS:
        for gpu_count in GPU_COUNTS:
            compiled = by_key[point_key("compiled", destination, gpu_count)]
            exact = by_key[point_key("exact", destination, gpu_count)]
            selective = by_key[
                point_key("selective_contiguous", destination, gpu_count)
            ]
            movement = by_key[
                point_key("no_transform", destination, gpu_count)
            ]
            comparisons.append(
                {
                    "destination": destination,
                    "gpu_count": gpu_count,
                    "compiled_seconds": compiled["timing"]["median_seconds"],
                    "exact_seconds": exact["timing"]["median_seconds"],
                    "selective_seconds": selective["timing"][
                        "median_seconds"
                    ],
                    "no_transform_seconds": movement["timing"][
                        "median_seconds"
                    ],
                    "compiled_speedup_over_exact": (
                        exact["timing"]["median_seconds"]
                        / compiled["timing"]["median_seconds"]
                    ),
                    "compiled_speedup_over_selective": (
                        selective["timing"]["median_seconds"]
                        / compiled["timing"]["median_seconds"]
                    ),
                    "compiled_over_movement_floor": (
                        compiled["timing"]["median_seconds"]
                        / movement["timing"]["median_seconds"]
                    ),
                    "compiled_source_read_fraction": (
                        compiled["timing"]["breakdown_seconds"]["source_read"]
                        / compiled["timing"]["median_seconds"]
                    ),
                }
            )
    scaling = []
    for method in METHODS:
        for destination in DESTINATIONS:
            one = by_key[point_key(method, destination, 1)][
                "timing"
            ]["median_seconds"]
            scaling.append(
                {
                    "method": method,
                    "destination": destination,
                    "speedup_2gpu_over_1gpu": (
                        one
                        / by_key[point_key(method, destination, 2)][
                            "timing"
                        ]["median_seconds"]
                    ),
                    "speedup_4gpu_over_1gpu": (
                        one
                        / by_key[point_key(method, destination, 4)][
                            "timing"
                        ]["median_seconds"]
                    ),
                }
            )
    derived = {
        "comparisons": comparisons,
        "scaling": scaling,
        "compiled_beats_exact_points": sum(
            value["compiled_speedup_over_exact"] > 1.0
            for value in comparisons
        ),
        "compiled_beats_selective_points": sum(
            value["compiled_speedup_over_selective"] > 1.0
            for value in comparisons
        ),
        "compiled_source_read_fraction_range": [
            min(
                value["compiled_source_read_fraction"]
                for value in comparisons
            ),
            max(
                value["compiled_source_read_fraction"]
                for value in comparisons
            ),
        ],
        "maximum_peak_source_resident_bytes": max(
            run["peak_source_resident_bytes"] for run in compact_runs
        ),
        "maximum_peak_staging_bytes": max(
            run["peak_staging_bytes"] for run in compact_runs
        ),
        "maximum_peak_publication_queue_bytes": max(
            run["peak_publication_queue_bytes"] for run in compact_runs
        ),
        "all_capacity_preflights_passed": True,
        "all_outputs_finite_and_allclose": True,
        "all_manifests_complete_and_duplicate_free": True,
    }
    return compact_runs, derived


def build_summary(
    root: Path,
    source_path: Path,
    source: dict,
    source_manifest_path: Path,
    manifest: Stage4SourceManifest,
    workload: dict,
    stage2: dict,
    stage3: dict,
) -> dict:
    validate_identity(
        root,
        source_path,
        source,
        source_manifest_path,
        manifest,
        workload,
        stage2,
        stage3,
    )
    materialization = validate_source_materialization(
        root,
        source_manifest_path,
        manifest,
    )
    tuning = validate_tuning(source)
    runs, derived = validate_runs(source, manifest, tuning)
    return {
        "protocol": PROTOCOL,
        "status": "stage4_frozen",
        "study_stage": "single_configuration_seed0_development",
        "source_result": {
            "path": str(source_path),
            "sha256": sha256_file(root / source_path),
            "protocol": SOURCE_PROTOCOL,
        },
        "parent_blueprint": {
            **source["blueprint"],
            "protocol": PARENT_PROTOCOL,
            "hash_scope": (
                "blueprint bytes used by Stage 4 before the downstream "
                "Stage-4 completion amendment"
            ),
        },
        "workload": {
            "path": str(WORKLOAD),
            "file_sha256": sha256_file(root / WORKLOAD),
            "content_sha256": WORKLOAD_CONTENT_SHA256,
            "records": RECORDS,
            "prefix_tokens": PREFIX_TOKENS,
            "program_selection_records": SELECTION_RECORDS,
            "program_selection_prefix_tokens": SELECTION_TOKENS,
        },
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": sha256_file(root / source_manifest_path),
            "protocol": SOURCE_MANIFEST_PROTOCOL,
            "materialization": materialization,
        },
        "stage2_summary": {
            "path": str(STAGE2),
            "sha256": sha256_file(root / STAGE2),
            "protocol": STAGE2_PROTOCOL,
        },
        "stage3_summary": {
            "path": str(STAGE3),
            "sha256": sha256_file(root / STAGE3),
            "protocol": STAGE3_PROTOCOL,
        },
        "implementation": source["implementation"],
        "measurement_boundary": {
            "execution": (
                "complete 682-record lazy source read through fresh "
                "destination allocation, transfer, transform, staging, "
                "coverage validation, and atomic manifest commit"
            ),
            "destinations": ["hbm", "dram"],
            "gpu_counts": [1, 2, 4],
            "primary_methods": list(PRIMARY_METHODS),
            "control_methods": list(CONTROL_METHODS),
            "full_points": 30,
            "primary_points": 18,
            "control_points": 12,
            "timed_repetitions_per_point": MEASURED_REPEATS,
            "recommendation_labels_used": False,
            "final_test_quality_evaluated": False,
            "os_page_cache_condition": (
                "one complete untimed source warmup before tuning; full "
                "repetitions reopen and decode shards without explicit "
                "page-cache eviction"
            ),
        },
        "runtime_tuning": {
            "candidate_order_seed": SELECTION_SEED,
            "selection_role": "program_selection",
            "labels_used": False,
            "separate_per_method_destination_gpu": True,
            "points": tuning,
        },
        "runs": runs,
        "derived": derived,
        "downstream_rule": {
            "normal_path_closed": True,
            "end_to_end_pareto_gate_passed": False,
            "selective_status": (
                "certificate-failed diagnostic baseline; exact remains its "
                "publishable fallback"
            ),
            "bounded_source_rule": (
                "source and staging residency are extent-bounded rather "
                "than cohort-sized"
            ),
            "next_stage": "stage4_5_source_state_footprint_optimization",
            "stage4_5_objective": (
                "make complete-cohort compiled completion stably faster "
                "than paired exact under an equally favorable source tier "
                "while disclosing standing state and lifecycle cost"
            ),
            "representative_iteration_points": [
                "compiled:hbm:1",
                "compiled:hbm:4",
            ],
            "expansion_rule": (
                "establish matched resident ceilings and screen source-state "
                "candidates on program-selection records, then run the "
                "complete cohort only at the two representative HBM points "
                "with paired exact; expand the matrix only after a candidate "
                "changes the Pareto frontier"
            ),
            "stage5_status": (
                "paused until a capacity-accounted source policy produces "
                "a stable end-to-end compiled Pareto point in its declared "
                "regime; automatic guard dispatch and failure recovery "
                "remain unimplemented"
            ),
            "claim_boundary": (
                "Stage 4 supports full-cohort normal-path HBM/DRAM system "
                "results and falsifies an end-to-end compiled speedup for "
                "the current FP16 capsule source path; it does not support "
                "automatic fallback or failure-recovery claims"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    source_path = Path(args.source_result)
    source_manifest_path = Path(args.source_manifest)
    output_path = Path(args.output)
    source = json.loads((root / source_path).read_text())
    workload = json.loads((root / WORKLOAD).read_text())
    stage2 = json.loads((root / STAGE2).read_text())
    stage3 = json.loads((root / STAGE3).read_text())
    manifest = Stage4SourceManifest.load(root / source_manifest_path)
    payload = canonical_json_bytes(
        build_summary(
            root,
            source_path,
            source,
            source_manifest_path,
            manifest,
            workload,
            stage2,
            stage3,
        )
    )
    resolved = root / output_path
    if args.check:
        if not resolved.is_file() or resolved.read_bytes() != payload:
            raise RuntimeError("Stage 4 frozen summary differs from source result")
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
