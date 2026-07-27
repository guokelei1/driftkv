from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import torch

from hstu_kvcache.migration import (
    CompiledStage4Transform,
    ExactStage4Transform,
    FusedMigrationOperator,
    LazyStage4SourceReader,
    NoTransformStage4Transform,
    PackedMigrationOperator,
    Stage4RuntimeConfig,
    load_runtime_program,
    sha256_file,
)
from hstu_kvcache.migration.stage45_resident import (
    STAGE45_RESIDENT_PROTOCOL,
    Stage45JobReport,
    Stage45ResidentEngine,
    build_stage45_resident_plan,
    materialize_stage45_resident_source,
    stage45_resident_preflight,
)
from hstu_kvcache.models import HSTUConfig
from hstu_kvcache.streaming import load_checkpoint_model

PARENT_PROTOCOL = "cohortkv_single_config_full_chain_development_v1"
STAGE2_PROTOCOL = "cohortkv_single_config_stage2_frozen_v1"
STAGE3_PROTOCOL = "cohortkv_single_config_stage3_frozen_v1"
STAGE4_PROTOCOL = "cohortkv_single_config_stage4_frozen_v1"
DEFAULT_BLUEPRINT = "configs/cohortkv_single_config_v1/blueprint.json"
DEFAULT_WORKLOAD = "configs/cohortkv_single_config_v1/workload_manifest.json"
DEFAULT_STAGE2 = "configs/cohortkv_single_config_v1/stage2_compiler_summary.json"
DEFAULT_STAGE3 = "configs/cohortkv_single_config_v1/stage3_operator_summary.json"
DEFAULT_STAGE4 = "configs/cohortkv_single_config_v1/stage4_system_summary.json"
DEFAULT_TRAINING = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
DEFAULT_CHECKPOINTS = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)
DEFAULT_SOURCE = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0/"
    "single_config_v1/source_shards/source_manifest.json"
)
DEFAULT_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_resident_ceiling_seed0.json"
)
SOURCE_VERSIONS = ("theta0", "theta4", "theta10")
TARGET_VERSION = "theta11"
METHODS = ("compiled", "exact", "no_transform")
SOURCE_TIERS = ("hbm_resident", "dram_resident")
GPU_COUNTS = (1, 4)
WARMUP_RUNS = 1
MEASURED_REPEATS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", default=DEFAULT_BLUEPRINT)
    parser.add_argument("--workload-manifest", default=DEFAULT_WORKLOAD)
    parser.add_argument("--stage2-summary", default=DEFAULT_STAGE2)
    parser.add_argument("--stage3-summary", default=DEFAULT_STAGE3)
    parser.add_argument("--stage4-summary", default=DEFAULT_STAGE4)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument(
        "--source-tiers",
        nargs="+",
        choices=SOURCE_TIERS,
        default=list(SOURCE_TIERS),
    )
    parser.add_argument(
        "--gpu-counts",
        nargs="+",
        type=int,
        choices=GPU_COUNTS,
        default=list(GPU_COUNTS),
    )
    parser.add_argument(
        "--measured-repeats",
        type=int,
        default=MEASURED_REPEATS,
    )
    return parser.parse_args()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = canonical_json(value)
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def implementation_snapshot(root: Path) -> dict[str, object]:
    paths = (
        Path("src/hstu_kvcache/migration/stage45_resident.py"),
        Path("scripts/benchmark_cohortkv_stage4_5_resident_ceiling.py"),
    )
    files = [
        {
            "path": str(path),
            "bytes": (root / path).stat().st_size,
            "sha256": sha256_file(root / path),
        }
        for path in paths
    ]
    return {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip(),
        "files": files,
        "code_snapshot_sha256": hashlib.sha256(
            canonical_json(files)
        ).hexdigest(),
    }


def validate_inputs(
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict, dict, dict, dict, dict, HSTUConfig, LazyStage4SourceReader]:
    if args.measured_repeats < 1:
        raise ValueError("measured repeats must be positive")
    if torch.cuda.device_count() < max(args.gpu_counts):
        raise ValueError("requested Stage 4.5 GPU count is unavailable")
    blueprint = json.loads((root / args.blueprint).read_text())
    workload = json.loads((root / args.workload_manifest).read_text())
    stage2 = json.loads((root / args.stage2_summary).read_text())
    stage3 = json.loads((root / args.stage3_summary).read_text())
    stage4 = json.loads((root / args.stage4_summary).read_text())
    training = json.loads((root / args.training_result).read_text())
    if (
        blueprint.get("protocol") != PARENT_PROTOCOL
        or blueprint.get("status") != "stage4_core_frozen"
        or tuple(blueprint.get("scope", {}).get("completed_stages", ()))
        != (0, 1, 2, 3, 4)
    ):
        raise ValueError("Stage 4.5 parent blueprint is invalid")
    frozen = blueprint["frozen_inputs"]
    expected = (
        (args.workload_manifest, frozen["workload_manifest"]["file_sha256"]),
        (args.stage2_summary, frozen["stage2_compiler_summary"]["sha256"]),
        (args.stage3_summary, frozen["stage3_operator_summary"]["sha256"]),
        (args.stage4_summary, frozen["stage4_system_summary"]["sha256"]),
        (args.training_result, frozen["training_result"]["sha256"]),
        (args.source_manifest, frozen["stage4_source_manifest"]["sha256"]),
    )
    if any(sha256_file(root / path) != digest for path, digest in expected):
        raise ValueError("Stage 4.5 frozen input hash mismatch")
    if (
        workload.get("protocol") != "cohortkv_single_config_workload_v1"
        or workload.get("status") != "frozen"
        or stage2.get("protocol") != STAGE2_PROTOCOL
        or stage2.get("status") != "stage2_frozen"
        or stage3.get("protocol") != STAGE3_PROTOCOL
        or stage3.get("status") != "stage3_frozen"
        or stage4.get("protocol") != STAGE4_PROTOCOL
        or stage4.get("status") != "stage4_frozen"
    ):
        raise ValueError("Stage 4.5 frozen input protocol mismatch")
    config = HSTUConfig(**training["model"])
    if training["model"] != blueprint["data_and_model"]["model"]:
        raise ValueError("Stage 4.5 model configuration differs")
    target = next(
        value
        for value in frozen["checkpoints"]
        if value["version"] == TARGET_VERSION
    )
    target_path = root / target["path"]
    if (
        sha256_file(target_path) != target["sha256"]
        or target_path.stat().st_size != target["bytes"]
    ):
        raise ValueError("Stage 4.5 target checkpoint differs")
    reader = LazyStage4SourceReader(
        root / args.source_manifest,
        workload["content_sha256"],
    )
    if (
        reader.manifest_file_sha256
        != frozen["stage4_source_manifest"]["sha256"]
        or reader.manifest.record_count != workload["summary"]["records"]
        or reader.manifest.prefix_tokens
        != workload["summary"]["prefix_tokens"]
    ):
        raise ValueError("Stage 4.5 source manifest differs")
    expected_records = [
        (
            value["record_id"],
            value["user_id"],
            value["evaluation_role"],
            value["source_version"],
            value["target_version"],
            value["prefix_tokens"],
        )
        for value in workload["records"]
    ]
    actual_records = [
        (
            value.record_id,
            value.user_id,
            value.evaluation_role,
            value.source_version,
            value.target_version,
            value.prefix_tokens,
        )
        for value in reader.manifest.records
    ]
    if actual_records != expected_records:
        raise ValueError("Stage 4.5 source record identity differs")
    return blueprint, workload, stage2, stage3, stage4, config, reader


def load_programs(
    root: Path,
    stage2: dict,
    model: dict,
) -> dict[str, object]:
    programs = {}
    for pair in stage2["pairs"]:
        descriptor = pair["runtime_program"]
        program, loaded = load_runtime_program(
            root / descriptor["path"],
            expected_sha256=descriptor["sha256"],
            expected_source_version=pair["source_version"],
            expected_target_version=TARGET_VERSION,
            expected_model=model,
        )
        if (
            loaded["dtype"] != "float16"
            or program.adapter.weights.dtype != torch.float16
            or program.adapter.biases.dtype != torch.float16
        ):
            raise ValueError("Stage 4.5 runtime program is not deployed FP16")
        programs[pair["source_version"]] = program
    if set(programs) != set(SOURCE_VERSIONS):
        raise ValueError("Stage 4.5 runtime program coverage is incomplete")
    return programs


def stage4_point(stage4: dict, key: str) -> dict:
    try:
        return next(
            value
            for value in stage4["runtime_tuning"]["points"]
            if value["key"] == key
        )
    except StopIteration as exc:
        raise ValueError(f"Stage 4.5 has no frozen runtime point {key}") from exc


def runtime_config(
    stage4: dict,
    method: str,
    gpu_count: int,
) -> Stage4RuntimeConfig:
    compiled = stage4_point(
        stage4,
        f"compiled:hbm:{gpu_count}",
    )["winner_runtime_config"]
    common = {
        "batch_size": int(compiled["batch_size"]),
        "length_bucket_width": int(compiled["length_bucket_width"]),
        "max_inflight": int(compiled["max_inflight"]),
    }
    if method == "compiled":
        return Stage4RuntimeConfig(
            **common,
            compiled_operator=str(compiled["compiled_operator"]),
        )
    if method == "exact":
        return Stage4RuntimeConfig(
            **common,
            exact_compute="bfloat16",
        )
    return Stage4RuntimeConfig(**common)


def build_transforms(
    root: Path,
    args: argparse.Namespace,
    config: HSTUConfig,
    programs: dict[str, object],
    method: str,
    gpu_count: int,
    runtime: Stage4RuntimeConfig,
) -> tuple[object, ...]:
    transforms = []
    for index in range(gpu_count):
        device = torch.device("cuda", index)
        if method == "compiled":
            operator = (
                FusedMigrationOperator()
                if runtime.compiled_operator == "fused_fp16"
                else PackedMigrationOperator(torch.float16)
            )
            transform = CompiledStage4Transform(
                programs,
                operator,
                device,
            )
        elif method == "exact":
            model = load_checkpoint_model(
                config,
                str(root / args.checkpoint_dir),
                11,
                device,
            )
            transform = ExactStage4Transform(
                model,
                TARGET_VERSION,
                torch.bfloat16,
            )
        else:
            transform = NoTransformStage4Transform(
                device,
                TARGET_VERSION,
            )
        transforms.append(transform)
    return tuple(transforms)


def point_key(method: str, source_tier: str, gpu_count: int) -> str:
    return f"{method}:{source_tier}:{gpu_count}"


def cleanup(gpu_count: int) -> None:
    gc.collect()
    for index in range(gpu_count):
        with torch.cuda.device(index):
            torch.cuda.synchronize(index)
            torch.cuda.empty_cache()


def target_layout(plan) -> dict[str, object]:
    layout = [
        {
            "device_index": index,
            "extent_id": extent.extent_id,
            "record_ids": list(extent.record_ids),
            "prefix_tokens": extent.token_count,
        }
        for index, assignment in enumerate(plan.assignments)
        for extent in assignment
    ]
    return {
        "extent_count": len(layout),
        "sha256": hashlib.sha256(canonical_json(layout)).hexdigest(),
    }


def compact_report(report: Stage45JobReport) -> dict[str, object]:
    return {
        "elapsed_seconds": report.elapsed_seconds,
        "timing_breakdown": report.timing_breakdown(),
        "logical_source_bytes": report.logical_source_bytes,
        "physical_source_bytes": report.physical_source_bytes,
        "resident_source_bytes": report.resident_source_bytes,
        "logical_output_bytes": report.logical_output_bytes,
        "physical_output_bytes": report.physical_output_bytes,
        "load_imbalance_ratio": report.load_imbalance_ratio,
        "per_gpu": [value.to_dict() for value in report.devices],
        "manifest": {
            "protocol": report.manifest.protocol,
            "record_count": report.manifest.record_count,
            "prefix_tokens": report.manifest.token_count,
            "payload_bytes": report.manifest.payload_bytes,
            "extent_count": len(report.manifest.extents),
            "coverage_sha256": hashlib.sha256(
                canonical_json(list(report.manifest.record_ids))
            ).hexdigest(),
        },
        "correctness": (
            None
            if report.correctness is None
            else report.correctness.to_dict()
        ),
    }


def run_once(
    engine: Stage45ResidentEngine,
    validate: bool,
    job_id: str,
) -> dict[str, object]:
    result = engine.run(validate=validate, job_id=job_id)
    payload = compact_report(result.report)
    del result
    gc.collect()
    for transform in engine.transforms:
        with torch.cuda.device(transform.device):
            torch.cuda.synchronize(transform.device)
            torch.cuda.empty_cache()
    return payload


def summarize_samples(jobs: list[dict[str, object]]) -> dict[str, object]:
    samples = [float(value["elapsed_seconds"]) for value in jobs]
    timing_keys = jobs[0]["timing_breakdown"]
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "range_seconds": max(samples) - min(samples),
        "population_std_seconds": statistics.pstdev(samples),
        "median_timing_breakdown": {
            key: statistics.median(
                float(value["timing_breakdown"][key])
                for value in jobs
            )
            for key in timing_keys
        },
        "maximum_peak_hbm_bytes": max(
            max(
                int(device["peak_hbm_bytes"])
                for device in value["per_gpu"]
            )
            for value in jobs
        ),
    }


def source_components(reader: LazyStage4SourceReader) -> dict[str, object]:
    values = {}
    for representation in (
        "normalized_capsule_fp16",
        "old_kv_fp16",
        "raw_history",
    ):
        shards = [
            record.shard_map[representation]
            for record in reader.manifest.records
            if record.evaluation_role == "program_selection"
        ]
        values[representation] = {
            "logical_bytes": sum(value.logical_bytes for value in shards),
            "physical_bytes": sum(value.physical_bytes for value in shards),
        }
    return values


def new_result(
    root: Path,
    args: argparse.Namespace,
    workload: dict,
    reader: LazyStage4SourceReader,
) -> dict[str, object]:
    return {
        "protocol": STAGE45_RESIDENT_PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "status": "in_progress",
        "study_stage": "stage4_5_a_resident_source_ceiling_seed0",
        "seed": 0,
        "labels_used": False,
        "selection_role": "program_selection",
        "measurement_boundary": (
            "resident source ready through complete FP16 HBM target and "
            "atomically committed manifest"
        ),
        "requested_matrix": {
            "methods": list(args.methods),
            "source_tiers": list(args.source_tiers),
            "gpu_counts": list(args.gpu_counts),
            "warmup_runs": WARMUP_RUNS,
            "measured_repeats": args.measured_repeats,
        },
        "inputs": {
            "blueprint": {
                "path": args.blueprint,
                "sha256": sha256_file(root / args.blueprint),
            },
            "workload": {
                "path": args.workload_manifest,
                "sha256": sha256_file(root / args.workload_manifest),
                "content_sha256": workload["content_sha256"],
            },
            "stage2": {
                "path": args.stage2_summary,
                "sha256": sha256_file(root / args.stage2_summary),
            },
            "stage3": {
                "path": args.stage3_summary,
                "sha256": sha256_file(root / args.stage3_summary),
            },
            "stage4": {
                "path": args.stage4_summary,
                "sha256": sha256_file(root / args.stage4_summary),
            },
            "source_manifest": {
                "path": args.source_manifest,
                "sha256": reader.manifest_file_sha256,
            },
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_bytes": torch.cuda.get_device_properties(
                        index
                    ).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "implementation": implementation_snapshot(root),
        "source_components": source_components(reader),
        "points": [],
        "comparisons": [],
    }


def load_or_create_result(
    root: Path,
    args: argparse.Namespace,
    workload: dict,
    reader: LazyStage4SourceReader,
) -> dict[str, object]:
    path = root / args.output
    expected = new_result(root, args, workload, reader)
    if not path.exists():
        return expected
    result = json.loads(path.read_text())
    if (
        result.get("protocol") != STAGE45_RESIDENT_PROTOCOL
        or result.get("requested_matrix")
        != expected["requested_matrix"]
        or result.get("inputs") != expected["inputs"]
        or result.get("implementation") != expected["implementation"]
    ):
        raise ValueError("existing Stage 4.5 result belongs to different inputs")
    return result


def save_result(root: Path, args: argparse.Namespace, result: dict) -> None:
    write_json_atomic(root / args.output, result)


def run_point(
    root: Path,
    args: argparse.Namespace,
    result: dict,
    workload: dict,
    stage4: dict,
    model_config: HSTUConfig,
    programs: dict[str, object],
    reader: LazyStage4SourceReader,
    method: str,
    source_tier: str,
    gpu_count: int,
) -> None:
    key = point_key(method, source_tier, gpu_count)
    if any(value["key"] == key for value in result["points"]):
        print(json.dumps({"point": key, "status": "already_complete"}), flush=True)
        return
    runtime = runtime_config(stage4, method, gpu_count)
    transforms = build_transforms(
        root,
        args,
        model_config,
        programs,
        method,
        gpu_count,
        runtime,
    )
    selection_ids = tuple(
        value["record_id"]
        for value in workload["records"]
        if value["evaluation_role"] == "program_selection"
    )
    plan = build_stage45_resident_plan(
        root / args.source_manifest,
        transforms,
        source_tier,
        runtime,
        selection_ids,
        expected_source_manifest_sha256=reader.manifest_file_sha256,
        expected_workload_content_sha256=workload["content_sha256"],
    )
    preflight = stage45_resident_preflight(plan, transforms)
    if not preflight["passed"]:
        raise MemoryError(f"Stage 4.5 point {key} failed capacity preflight")
    print(
        json.dumps(
            {
                "point": key,
                "status": "materializing",
                "resident_source_bytes": plan.resident_source_bytes,
            }
        ),
        flush=True,
    )
    source = materialize_stage45_resident_source(plan, transforms)
    engine = Stage45ResidentEngine(source, transforms)
    correctness_job = run_once(
        engine,
        True,
        f"s45-{method}-{source_tier}-{gpu_count}-check",
    )
    correctness = correctness_job["correctness"]
    if not (
        correctness["finite"]
        and correctness["allclose"]
        and correctness["record_order_valid"]
        and correctness["lengths_offsets_valid"]
    ):
        raise RuntimeError(f"Stage 4.5 point {key} failed correctness")
    warmup_jobs = [
        run_once(
            engine,
            False,
            f"s45-{method}-{source_tier}-{gpu_count}-warm{repeat}",
        )
        for repeat in range(WARMUP_RUNS)
    ]
    measured_jobs = []
    for repeat in range(args.measured_repeats):
        job = run_once(
            engine,
            False,
            f"s45-{method}-{source_tier}-{gpu_count}-measure{repeat}",
        )
        measured_jobs.append(job)
        print(
            json.dumps(
                {
                    "point": key,
                    "repeat": repeat + 1,
                    "total": args.measured_repeats,
                    "elapsed_seconds": job["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    point = {
        "key": key,
        "method": method,
        "source_tier": source_tier,
        "gpu_count": gpu_count,
        "status": "complete",
        "runtime_config": runtime.to_dict(),
        "frozen_compiled_layout_config": stage4_point(
            stage4,
            f"compiled:hbm:{gpu_count}",
        )["winner_runtime_config"],
        "target_layout": target_layout(plan),
        "plan": plan.to_dict(),
        "capacity_preflight": preflight,
        "preload": source.preload.to_dict(),
        "correctness_job": correctness_job,
        "warmup_jobs": warmup_jobs,
        "measured_jobs": measured_jobs,
        "summary": summarize_samples(measured_jobs),
    }
    result["points"].append(point)
    save_result(root, args, result)
    print(
        json.dumps(
            {
                "point": key,
                "status": "complete",
                "median_seconds": point["summary"]["median_seconds"],
            }
        ),
        flush=True,
    )
    engine.close()
    del engine, source, plan, transforms
    cleanup(gpu_count)


def build_comparisons(result: dict) -> list[dict[str, object]]:
    points = {value["key"]: value for value in result["points"]}
    comparisons = []
    matrix = result["requested_matrix"]
    if not {"compiled", "exact"}.issubset(matrix["methods"]):
        return comparisons
    for source_tier in matrix["source_tiers"]:
        for gpu_count in matrix["gpu_counts"]:
            compiled = points[
                point_key("compiled", source_tier, gpu_count)
            ]
            exact = points[point_key("exact", source_tier, gpu_count)]
            if compiled["target_layout"] != exact["target_layout"]:
                raise RuntimeError(
                    "Stage 4.5 paired target layouts are not identical"
                )
            compiled_summary = compiled["summary"]
            exact_summary = exact["summary"]
            compiled_samples = compiled_summary["samples_seconds"]
            exact_samples = exact_summary["samples_seconds"]
            savings = (
                exact_summary["median_seconds"]
                - compiled_summary["median_seconds"]
            )
            variation = max(
                compiled_summary["range_seconds"],
                exact_summary["range_seconds"],
            )
            preload_delta = max(
                compiled["preload"]["elapsed_seconds"]
                - exact["preload"]["elapsed_seconds"],
                0.0,
            )
            comparisons.append(
                {
                    "source_tier": source_tier,
                    "gpu_count": gpu_count,
                    "target_layout": compiled["target_layout"],
                    "compiled_median_seconds": (
                        compiled_summary["median_seconds"]
                    ),
                    "exact_median_seconds": exact_summary["median_seconds"],
                    "compiled_over_exact": (
                        compiled_summary["median_seconds"]
                        / exact_summary["median_seconds"]
                    ),
                    "speedup": (
                        exact_summary["median_seconds"]
                        / compiled_summary["median_seconds"]
                    ),
                    "median_savings_seconds": savings,
                    "maximum_measured_variation_band_seconds": variation,
                    "all_compiled_below_all_exact": (
                        max(compiled_samples) < min(exact_samples)
                    ),
                    "difference_exceeds_variation_band": savings > variation,
                    "resident_completion_gate_passed": (
                        savings > 0
                        and max(compiled_samples) < min(exact_samples)
                        and savings > variation
                    ),
                    "compiled_preload_seconds": compiled["preload"][
                        "elapsed_seconds"
                    ],
                    "exact_preload_seconds": exact["preload"][
                        "elapsed_seconds"
                    ],
                    "incremental_preload_seconds": preload_delta,
                    "break_even_updates": (
                        None
                        if savings <= 0
                        else max(1, int(preload_delta / savings) + 1)
                    ),
                    "compiled_resident_source_bytes": compiled["plan"][
                        "resident_source_bytes"
                    ],
                    "exact_resident_source_bytes": exact["plan"][
                        "resident_source_bytes"
                    ],
                }
            )
    return comparisons


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    (
        _,
        workload,
        stage2,
        _,
        stage4,
        model_config,
        reader,
    ) = validate_inputs(root, args)
    result = load_or_create_result(root, args, workload, reader)
    programs = load_programs(root, stage2, model_config.__dict__)
    started = time.perf_counter()
    for source_tier in args.source_tiers:
        for gpu_count in args.gpu_counts:
            for method in args.methods:
                run_point(
                    root,
                    args,
                    result,
                    workload,
                    stage4,
                    model_config,
                    programs,
                    reader,
                    method,
                    source_tier,
                    gpu_count,
                )
    result["comparisons"] = build_comparisons(result)
    expected_points = (
        len(args.methods)
        * len(args.source_tiers)
        * len(args.gpu_counts)
    )
    result["status"] = (
        "stage4_5_a_complete"
        if len(result["points"]) == expected_points
        else "in_progress"
    )
    result["last_invocation_seconds"] = time.perf_counter() - started
    save_result(root, args, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "points": len(result["points"]),
                "comparisons": result["comparisons"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
