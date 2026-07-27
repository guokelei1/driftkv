from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import platform
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from hstu_kvcache.migration import (
    CompiledStage4Transform,
    ExactStage4Transform,
    FusedMigrationOperator,
    LazyStage4SourceReader,
    NoTransformStage4Transform,
    PackedMigrationOperator,
    ResidualPStage4Transform,
    SelectiveStage4Transform,
    Stage4CoreEngine,
    Stage4JobReport,
    Stage4RuntimeConfig,
    build_stage4_extents,
    load_runtime_program,
    place_stage4_extents_lpt,
    sha256_file,
    stage4_capacity_preflight,
)
from hstu_kvcache.models import HSTUConfig
from hstu_kvcache.streaming import load_checkpoint_model

PROTOCOL = "cohortkv_single_config_stage4_system_v1"
PARENT_PROTOCOL = "cohortkv_single_config_full_chain_development_v1"
STAGE2_PROTOCOL = "cohortkv_single_config_stage2_frozen_v1"
STAGE3_PROTOCOL = "cohortkv_single_config_stage3_frozen_v1"
DEFAULT_BLUEPRINT = "configs/cohortkv_single_config_v1/blueprint.json"
DEFAULT_WORKLOAD = "configs/cohortkv_single_config_v1/workload_manifest.json"
DEFAULT_STAGE2 = "configs/cohortkv_single_config_v1/stage2_compiler_summary.json"
DEFAULT_STAGE3 = "configs/cohortkv_single_config_v1/stage3_operator_summary.json"
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
    "stage4_system_seed0.json"
)
SOURCE_VERSIONS = ("theta0", "theta4", "theta10")
TARGET_VERSION = "theta11"
PRIMARY_METHODS = ("compiled", "selective_contiguous", "exact")
CONTROL_METHODS = ("residual_p", "no_transform")
METHODS = (*PRIMARY_METHODS, *CONTROL_METHODS)
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
ATOL = 0.02
RTOL = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", default=DEFAULT_BLUEPRINT)
    parser.add_argument("--workload-manifest", default=DEFAULT_WORKLOAD)
    parser.add_argument("--stage2-summary", default=DEFAULT_STAGE2)
    parser.add_argument("--stage3-summary", default=DEFAULT_STAGE3)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase",
        choices=("tune", "full", "all"),
        default="all",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument(
        "--destinations",
        nargs="+",
        choices=DESTINATIONS,
        default=list(DESTINATIONS),
    )
    parser.add_argument(
        "--gpu-counts",
        nargs="+",
        type=int,
        choices=GPU_COUNTS,
        default=list(GPU_COUNTS),
    )
    parser.add_argument(
        "--rerun-full-points",
        nargs="*",
        default=[],
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
    paths = sorted(
        path.relative_to(root)
        for path in (root / "src" / "hstu_kvcache").rglob("*.py")
    )
    paths.extend(
        (
            Path("scripts/benchmark_cohortkv_stage4_system.py"),
            Path("scripts/materialize_cohortkv_stage4_sources.py"),
        )
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
) -> tuple[dict, dict, dict, dict, HSTUConfig, LazyStage4SourceReader]:
    if torch.cuda.device_count() < max(args.gpu_counts):
        raise ValueError("requested Stage 4 GPU count is unavailable")
    blueprint = json.loads((root / args.blueprint).read_text())
    workload = json.loads((root / args.workload_manifest).read_text())
    stage2 = json.loads((root / args.stage2_summary).read_text())
    stage3 = json.loads((root / args.stage3_summary).read_text())
    training = json.loads((root / args.training_result).read_text())
    blueprint_state = (
        blueprint.get("status"),
        tuple(blueprint.get("scope", {}).get("completed_stages", ())),
    )
    if (
        blueprint.get("protocol") != PARENT_PROTOCOL
        or blueprint_state
        not in {
            ("stage3_operator_frozen", (0, 1, 2, 3)),
            ("stage4_core_frozen", (0, 1, 2, 3, 4)),
        }
    ):
        raise ValueError("Stage 4 parent blueprint is invalid")
    frozen = blueprint["frozen_inputs"]
    expected = (
        (args.workload_manifest, frozen["workload_manifest"]["file_sha256"]),
        (args.stage2_summary, frozen["stage2_compiler_summary"]["sha256"]),
        (args.stage3_summary, frozen["stage3_operator_summary"]["sha256"]),
        (args.training_result, frozen["training_result"]["sha256"]),
    )
    if any(sha256_file(root / path) != digest for path, digest in expected):
        raise ValueError("Stage 4 frozen input hash mismatch")
    if (
        workload.get("protocol") != "cohortkv_single_config_workload_v1"
        or workload.get("status") != "frozen"
        or workload.get("content_sha256")
        != frozen["workload_manifest"]["content_sha256"]
        or stage2.get("protocol") != STAGE2_PROTOCOL
        or stage2.get("status") != "stage2_frozen"
        or stage3.get("protocol") != STAGE3_PROTOCOL
        or stage3.get("status") != "stage3_frozen"
    ):
        raise ValueError("Stage 4 frozen input protocol mismatch")
    tuning = blueprint["runtime_tuning_contract"]
    if (
        tuning["candidate_order_seed"] != SELECTION_SEED
        or tuning["warmup_runs"] != WARMUP_RUNS
        or tuning["measured_repetitions"] != MEASURED_REPEATS
        or tuning["grid"]["batch_size"] != list(BATCH_SIZES)
        or tuning["grid"]["length_bucket_width"] != list(BUCKET_WIDTHS)
        or tuning["grid"]["max_inflight"] != list(INFLIGHT_DEPTHS)
        or tuning["grid"]["compiled_operator"]
        != list(COMPILED_OPERATORS)
        or tuning["grid"]["exact_compute"] != list(EXACT_COMPUTE)
    ):
        raise ValueError("Stage 4 runtime grid differs from Stage 0")
    cfg = HSTUConfig(**training["model"])
    if training["model"] != blueprint["data_and_model"]["model"]:
        raise ValueError("Stage 4 model configuration differs from Stage 0")
    target_descriptor = next(
        value
        for value in frozen["checkpoints"]
        if value["version"] == TARGET_VERSION
    )
    target_path = (
        root
        / args.checkpoint_dir
        / f"theta_{TARGET_VERSION.removeprefix('theta')}.pt"
    )
    if (
        sha256_file(target_path) != target_descriptor["sha256"]
        or target_path.stat().st_size != target_descriptor["bytes"]
    ):
        raise ValueError("Stage 4 target checkpoint differs from Stage 0")
    reader = LazyStage4SourceReader(
        root / args.source_manifest,
        workload["content_sha256"],
    )
    if (
        reader.manifest.record_count != workload["summary"]["records"]
        or reader.manifest.prefix_tokens
        != workload["summary"]["prefix_tokens"]
        or reader.manifest.workload_file_sha256
        != sha256_file(root / args.workload_manifest)
    ):
        raise ValueError("Stage 4 source manifest differs from the workload")
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
        raise ValueError("Stage 4 source record identity differs")
    return blueprint, workload, stage2, stage3, cfg, reader


def new_result(
    root: Path,
    args: argparse.Namespace,
    workload: dict,
    reader: LazyStage4SourceReader,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "status": "in_progress",
        "study_stage": "single_configuration_seed0_development",
        "seed": 0,
        "labels_used": False,
        "source_manifest": {
            "path": args.source_manifest,
            "sha256": reader.manifest_file_sha256,
            "protocol": reader.manifest.protocol,
            "record_count": reader.manifest.record_count,
            "prefix_tokens": reader.manifest.prefix_tokens,
        },
        "workload_manifest": {
            "path": args.workload_manifest,
            "sha256": sha256_file(root / args.workload_manifest),
            "content_sha256": workload["content_sha256"],
        },
        "blueprint": {
            "path": args.blueprint,
            "sha256": sha256_file(root / args.blueprint),
        },
        "stage2_summary": {
            "path": args.stage2_summary,
            "sha256": sha256_file(root / args.stage2_summary),
            "protocol": STAGE2_PROTOCOL,
        },
        "stage3_summary": {
            "path": args.stage3_summary,
            "sha256": sha256_file(root / args.stage3_summary),
            "protocol": STAGE3_PROTOCOL,
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
        "runtime_tuning": {
            "candidate_order_seed": SELECTION_SEED,
            "selection_role": "program_selection",
            "warmup_runs": WARMUP_RUNS,
            "measured_repetitions": MEASURED_REPEATS,
            "points": [],
        },
        "runs": [],
    }


def load_or_create_result(
    root: Path,
    args: argparse.Namespace,
    workload: dict,
    reader: LazyStage4SourceReader,
) -> dict:
    path = root / args.output
    if not path.exists():
        return new_result(root, args, workload, reader)
    result = json.loads(path.read_text())
    if (
        result.get("protocol") != PROTOCOL
        or result.get("source_manifest", {}).get("sha256")
        != reader.manifest_file_sha256
        or result.get("workload_manifest", {}).get("content_sha256")
        != workload["content_sha256"]
        or result.get("blueprint", {}).get("sha256")
        != sha256_file(root / args.blueprint)
        or result.get("stage2_summary", {}).get("sha256")
        != sha256_file(root / args.stage2_summary)
        or result.get("stage3_summary", {}).get("sha256")
        != sha256_file(root / args.stage3_summary)
    ):
        raise ValueError("existing Stage 4 result belongs to different inputs")
    return result


def point_key(method: str, destination: str, gpu_count: int) -> str:
    return f"{method}:{destination}:{gpu_count}"


def candidate_id(value: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()[:16]


def candidate_grid(method: str) -> list[dict[str, object]]:
    base = itertools.product(BATCH_SIZES, BUCKET_WIDTHS, INFLIGHT_DEPTHS)
    candidates = []
    for batch, bucket, inflight in base:
        common = {
            "batch_size": batch,
            "length_bucket_width": bucket,
            "max_inflight": inflight,
        }
        if method == "compiled":
            for operator in COMPILED_OPERATORS:
                candidates.append(
                    {**common, "compiled_operator": operator}
                )
        elif method == "exact":
            for compute in EXACT_COMPUTE:
                candidates.append({**common, "exact_compute": compute})
        else:
            candidates.append(common)
    order = np.random.default_rng(SELECTION_SEED).permutation(
        len(candidates)
    )
    return [candidates[index] for index in order]


def runtime_config(value: dict[str, object]) -> Stage4RuntimeConfig:
    return Stage4RuntimeConfig(
        batch_size=int(value["batch_size"]),
        length_bucket_width=int(value["length_bucket_width"]),
        max_inflight=int(value["max_inflight"]),
        compiled_operator=(
            None
            if "compiled_operator" not in value
            else str(value["compiled_operator"])
        ),
        exact_compute=(
            None
            if "exact_compute" not in value
            else str(value["exact_compute"])
        ),
    )


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
            raise ValueError("Stage 4 runtime program is not deployed FP16")
        programs[pair["source_version"]] = program
    if set(programs) != set(SOURCE_VERSIONS):
        raise ValueError("Stage 4 runtime program coverage is incomplete")
    return programs


def build_transforms(
    root: Path,
    args: argparse.Namespace,
    cfg: HSTUConfig,
    programs: dict[str, object],
    method: str,
    gpu_count: int,
    config: Stage4RuntimeConfig,
) -> tuple[object, ...]:
    transforms = []
    for index in range(gpu_count):
        device = torch.device("cuda", index)
        if method == "compiled":
            operator = (
                FusedMigrationOperator()
                if config.compiled_operator == "fused_fp16"
                else PackedMigrationOperator(torch.float16)
            )
            transform = CompiledStage4Transform(
                programs,
                operator,
                device,
            )
        elif method == "no_transform":
            transform = NoTransformStage4Transform(
                device,
                TARGET_VERSION,
            )
        else:
            model = load_checkpoint_model(
                cfg,
                str(root / args.checkpoint_dir),
                11,
                device,
            )
            if method == "exact":
                dtype = (
                    torch.bfloat16
                    if config.exact_compute == "bfloat16"
                    else None
                )
                transform = ExactStage4Transform(
                    model,
                    TARGET_VERSION,
                    dtype,
                )
            elif method == "selective_contiguous":
                transform = SelectiveStage4Transform(
                    model,
                    TARGET_VERSION,
                    0,
                    11,
                )
            elif method == "residual_p":
                transform = ResidualPStage4Transform(
                    model,
                    TARGET_VERSION,
                    8,
                    ("theta0", "theta10"),
                )
            else:
                raise ValueError("unsupported Stage 4 method")
        transforms.append(transform)
    return tuple(transforms)


def representations(
    method: str,
    source: str,
) -> tuple[str, ...]:
    if method == "compiled":
        return ("normalized_capsule_fp16",)
    if method == "selective_contiguous":
        return ("old_kv_fp16", "raw_history")
    if method == "exact":
        return ("raw_history",)
    if method == "no_transform":
        return ("old_kv_fp16",)
    if method == "residual_p":
        if source in {"theta0", "theta10"}:
            return ("raw_history", "residual_hidden_suffix_bf16")
        return ("raw_history",)
    raise ValueError("unsupported Stage 4 method")


def plan(
    reader: LazyStage4SourceReader,
    record_ids: tuple[int, ...],
    method: str,
    config: Stage4RuntimeConfig,
    gpu_count: int,
):
    source_versions = {
        reader.manifest.record_map[value].source_version
        for value in record_ids
    }
    extents = build_stage4_extents(
        reader.manifest,
        record_ids,
        {
            source: representations(method, source)
            for source in source_versions
        },
        config.batch_size,
        config.length_bucket_width,
    )
    return extents, place_stage4_extents_lpt(extents, gpu_count)


def warm_source(
    reader: LazyStage4SourceReader,
    record_ids: tuple[int, ...],
    method: str,
) -> dict[str, object]:
    config = Stage4RuntimeConfig(
        batch_size=4,
        length_bucket_width=64,
        max_inflight=2,
        compiled_operator=(
            "fused_fp16" if method == "compiled" else None
        ),
        exact_compute="bfloat16" if method == "exact" else None,
    )
    extents, _ = plan(reader, record_ids, method, config, 1)
    started = time.perf_counter()
    physical = 0
    logical = 0
    peak = 0
    for extent in extents:
        batch, metrics = reader.read_extent(extent, pin_memory=False)
        physical += metrics.physical_bytes
        logical += metrics.logical_bytes
        peak = max(peak, metrics.peak_source_resident_bytes)
        del batch
    return {
        "record_count": len(record_ids),
        "physical_bytes": physical,
        "logical_bytes": logical,
        "peak_source_resident_bytes": peak,
        "elapsed_seconds": time.perf_counter() - started,
        "complete": True,
    }


def pinned_extent_probe(
    reader: LazyStage4SourceReader,
    record_ids: tuple[int, ...],
    method: str,
) -> dict[str, object]:
    maximum = 0
    for batch in BATCH_SIZES:
        for bucket in BUCKET_WIDTHS:
            config = Stage4RuntimeConfig(
                batch_size=batch,
                length_bucket_width=bucket,
                max_inflight=2,
                compiled_operator=(
                    "fused_fp16" if method == "compiled" else None
                ),
                exact_compute=(
                    "bfloat16" if method == "exact" else None
                ),
            )
            extents, _ = plan(reader, record_ids, method, config, 1)
            maximum = max(
                maximum,
                max(value.logical_output_bytes for value in extents),
            )
    started = time.perf_counter()
    first = torch.empty(maximum // 2, dtype=torch.uint8, pin_memory=True)
    second = torch.empty(maximum // 2, dtype=torch.uint8, pin_memory=True)
    passed = first.is_pinned() and second.is_pinned()
    del first, second
    gc.collect()
    if not passed:
        raise RuntimeError("Stage 4 maximum pinned extent probe failed")
    return {
        "maximum_extent_bytes": maximum,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": True,
    }


def compact_job(report: Stage4JobReport) -> dict[str, object]:
    return {
        "elapsed_seconds": report.elapsed_seconds,
        "timing_breakdown": report.timing_breakdown(),
        "logical_input_bytes": report.logical_input_bytes,
        "physical_input_bytes": report.physical_input_bytes,
        "logical_output_bytes": report.logical_output_bytes,
        "physical_output_bytes": report.physical_output_bytes,
        "peak_hbm_bytes": report.peak_hbm_bytes,
        "peak_host_bytes": report.peak_host_bytes,
        "peak_source_resident_bytes": report.peak_source_resident_bytes,
        "peak_staging_bytes": report.peak_staging_bytes,
        "peak_publication_queue_bytes": (
            report.peak_publication_queue_bytes
        ),
        "load_imbalance_ratio": report.load_imbalance_ratio,
        "extent_count": len(report.manifest.extents),
        "manifest": {
            "protocol": report.manifest.protocol,
            "record_count": report.manifest.record_count,
            "prefix_tokens": report.manifest.token_count,
            "payload_bytes": report.manifest.payload_bytes,
        },
        "per_gpu": [value.to_dict() for value in report.devices],
        "correctness": (
            None
            if report.correctness is None
            else report.correctness.to_dict()
        ),
    }


def cleanup(gpu_count: int) -> None:
    gc.collect()
    for index in range(gpu_count):
        with torch.cuda.device(index):
            torch.cuda.synchronize(index)
            torch.cuda.empty_cache()


def run_engine(
    source_path: Path,
    source_hash: str,
    workload_hash: str,
    transforms: tuple[object, ...],
    destination: str,
    config: Stage4RuntimeConfig,
    record_ids: tuple[int, ...],
    validate: bool,
    job_id: str,
) -> tuple[dict[str, object], tuple[int, ...]]:
    baseline = tuple(
        torch.cuda.memory_allocated(index)
        for index in range(len(transforms))
    )
    engine = Stage4CoreEngine(
        source_path,
        transforms,
        destination,
        config,
        expected_source_manifest_sha256=source_hash,
        expected_workload_content_sha256=workload_hash,
    )
    result = engine.run(
        record_ids=record_ids,
        validate=validate,
        job_id=job_id,
    )
    payload = compact_job(result.report)
    del result, engine
    cleanup(len(transforms))
    return payload, baseline


def transient_bytes(
    job: dict[str, object],
    baseline: tuple[int, ...],
    assignments,
    destination: str,
) -> tuple[int, ...]:
    values = []
    for index, (device, assigned) in enumerate(
        zip(job["per_gpu"], assignments, strict=True)
    ):
        target = (
            sum(value.logical_output_bytes for value in assigned)
            if destination == "hbm"
            else 0
        )
        values.append(
            max(
                0,
                int(device["peak_hbm_bytes"])
                - baseline[index]
                - target,
            )
        )
    return tuple(values)


def save_result(root: Path, args: argparse.Namespace, result: dict) -> None:
    write_json_atomic(root / args.output, result)


def get_point(result: dict, key: str) -> dict | None:
    return next(
        (
            value
            for value in result["runtime_tuning"]["points"]
            if value["key"] == key
        ),
        None,
    )


def tune_point(
    root: Path,
    args: argparse.Namespace,
    result: dict,
    reader: LazyStage4SourceReader,
    workload: dict,
    cfg: HSTUConfig,
    programs: dict[str, object],
    method: str,
    destination: str,
    gpu_count: int,
) -> None:
    key = point_key(method, destination, gpu_count)
    point = get_point(result, key)
    if point is not None and point.get("status") == "tuned":
        print(json.dumps({"point": key, "status": "already_tuned"}), flush=True)
        return
    if point is None:
        point = {
            "key": key,
            "method": method,
            "destination": destination,
            "gpu_count": gpu_count,
            "status": "in_progress",
            "source_warmup": None,
            "pinned_extent_probe": None,
            "candidates": [],
            "finalist_ids": [],
            "winner_id": None,
        }
        result["runtime_tuning"]["points"].append(point)
        save_result(root, args, result)
    selection_ids = tuple(
        value["record_id"]
        for value in workload["records"]
        if value["evaluation_role"] == "program_selection"
    )
    if point["source_warmup"] is None:
        point["source_warmup"] = warm_source(
            reader,
            selection_ids,
            method,
        )
        save_result(root, args, result)
    if destination == "dram" and point["pinned_extent_probe"] is None:
        all_ids = tuple(value.record_id for value in reader.manifest.records)
        point["pinned_extent_probe"] = pinned_extent_probe(
            reader,
            all_ids,
            method,
        )
        save_result(root, args, result)
    completed = {
        value["candidate_id"]: value
        for value in point["candidates"]
    }
    for value in candidate_grid(method):
        identifier = candidate_id(value)
        if identifier in completed:
            continue
        config = runtime_config(value)
        transforms = build_transforms(
            root,
            args,
            cfg,
            programs,
            method,
            gpu_count,
            config,
        )
        _, assignments = plan(
            reader,
            selection_ids,
            method,
            config,
            gpu_count,
        )
        short = f"s4-{identifier}"
        correctness_job, correctness_baseline = run_engine(
            root / args.source_manifest,
            reader.manifest_file_sha256,
            workload["content_sha256"],
            transforms,
            destination,
            config,
            selection_ids,
            True,
            f"{short}-check",
        )
        correctness = correctness_job["correctness"]
        if not (
            correctness["finite"]
            and correctness["allclose"]
            and correctness["record_order_valid"]
            and correctness["lengths_offsets_valid"]
        ):
            raise RuntimeError(f"Stage 4 candidate {identifier} is incorrect")
        screen_job, screen_baseline = run_engine(
            root / args.source_manifest,
            reader.manifest_file_sha256,
            workload["content_sha256"],
            transforms,
            destination,
            config,
            selection_ids,
            False,
            f"{short}-screen",
        )
        correctness_transient = transient_bytes(
            correctness_job,
            correctness_baseline,
            assignments,
            destination,
        )
        screen_transient = transient_bytes(
            screen_job,
            screen_baseline,
            assignments,
            destination,
        )
        candidate = {
            "candidate_id": identifier,
            "runtime_config": value,
            "padding_tokens": sum(
                extent.padding_tokens
                for assignment in assignments
                for extent in assignment
            ),
            "correctness": correctness,
            "correctness_job": correctness_job,
            "screen_job": screen_job,
            "screen_elapsed_seconds": screen_job["elapsed_seconds"],
            "maximum_transient_hbm_bytes": [
                max(first, second)
                for first, second in zip(
                    correctness_transient,
                    screen_transient,
                    strict=True,
                )
            ],
            "finalist": None,
        }
        point["candidates"].append(candidate)
        save_result(root, args, result)
        print(
            json.dumps(
                {
                    "point": key,
                    "candidate": identifier,
                    "completed": len(point["candidates"]),
                    "total": len(candidate_grid(method)),
                    "screen_seconds": screen_job["elapsed_seconds"],
                }
            ),
            flush=True,
        )
        del transforms
        cleanup(gpu_count)
    ranked = sorted(
        point["candidates"],
        key=lambda value: (
            value["screen_elapsed_seconds"],
            max(value["screen_job"]["peak_hbm_bytes"], 0),
            value["padding_tokens"],
            value["candidate_id"],
        ),
    )
    finalists = ranked[:3]
    point["finalist_ids"] = [
        value["candidate_id"] for value in finalists
    ]
    for candidate in finalists:
        if candidate["finalist"] is not None:
            continue
        config = runtime_config(candidate["runtime_config"])
        transforms = build_transforms(
            root,
            args,
            cfg,
            programs,
            method,
            gpu_count,
            config,
        )
        _, assignments = plan(
            reader,
            selection_ids,
            method,
            config,
            gpu_count,
        )
        warmup_jobs = []
        for repeat in range(WARMUP_RUNS):
            job, baseline = run_engine(
                root / args.source_manifest,
                reader.manifest_file_sha256,
                workload["content_sha256"],
                transforms,
                destination,
                config,
                selection_ids,
                False,
                f"s4-{candidate['candidate_id']}-warm{repeat}",
            )
            warmup_jobs.append(job)
            current = transient_bytes(
                job,
                baseline,
                assignments,
                destination,
            )
            candidate["maximum_transient_hbm_bytes"] = [
                max(first, second)
                for first, second in zip(
                    candidate["maximum_transient_hbm_bytes"],
                    current,
                    strict=True,
                )
            ]
        measured_jobs = []
        for repeat in range(MEASURED_REPEATS):
            job, baseline = run_engine(
                root / args.source_manifest,
                reader.manifest_file_sha256,
                workload["content_sha256"],
                transforms,
                destination,
                config,
                selection_ids,
                False,
                f"s4-{candidate['candidate_id']}-measure{repeat}",
            )
            measured_jobs.append(job)
            current = transient_bytes(
                job,
                baseline,
                assignments,
                destination,
            )
            candidate["maximum_transient_hbm_bytes"] = [
                max(first, second)
                for first, second in zip(
                    candidate["maximum_transient_hbm_bytes"],
                    current,
                    strict=True,
                )
            ]
        samples = [value["elapsed_seconds"] for value in measured_jobs]
        candidate["finalist"] = {
            "warmup_jobs": warmup_jobs,
            "measured_jobs": measured_jobs,
            "samples_seconds": samples,
            "median_seconds": statistics.median(samples),
            "peak_hbm_bytes": max(
                value["peak_hbm_bytes"] for value in measured_jobs
            ),
        }
        save_result(root, args, result)
        print(
            json.dumps(
                {
                    "point": key,
                    "finalist": candidate["candidate_id"],
                    "median_seconds": candidate["finalist"][
                        "median_seconds"
                    ],
                }
            ),
            flush=True,
        )
        del transforms
        cleanup(gpu_count)
    winner = min(
        finalists,
        key=lambda value: (
            value["finalist"]["median_seconds"],
            value["finalist"]["peak_hbm_bytes"],
            value["padding_tokens"],
            value["candidate_id"],
        ),
    )
    point["winner_id"] = winner["candidate_id"]
    point["winner_runtime_config"] = winner["runtime_config"]
    point["status"] = "tuned"
    save_result(root, args, result)
    print(
        json.dumps(
            {
                "point": key,
                "status": "tuned",
                "winner": point["winner_id"],
                "runtime_config": point["winner_runtime_config"],
                "median_seconds": winner["finalist"]["median_seconds"],
            }
        ),
        flush=True,
    )


def input_components(
    reader: LazyStage4SourceReader,
    method: str,
) -> list[dict[str, object]]:
    order = {
        "compiled": ("normalized_capsule_fp16",),
        "selective_contiguous": ("old_kv_fp16", "raw_history"),
        "residual_p": ("raw_history", "residual_hidden_suffix_bf16"),
        "exact": ("raw_history",),
        "no_transform": ("old_kv_fp16",),
    }[method]
    components = []
    for representation in order:
        selected = [
            record.shard_map[representation]
            for record in reader.manifest.records
            if representation in representations(
                method,
                record.source_version,
            )
        ]
        components.append(
            {
                "representation": representation,
                "logical_bytes": sum(
                    value.logical_bytes for value in selected
                ),
                "physical_bytes": sum(
                    value.physical_bytes for value in selected
                ),
            }
        )
    return components


def aggregate_preflights(values: list[dict[str, object]]) -> dict[str, object]:
    return {
        "minimum_observed_free_hbm_bytes": min(
            value["minimum_observed_free_hbm_bytes"]
            for value in values
        ),
        "required_peak_hbm_bytes": max(
            value["required_peak_hbm_bytes"] for value in values
        ),
        "minimum_observed_available_host_bytes": min(
            value["minimum_observed_available_host_bytes"]
            for value in values
        ),
        "required_peak_host_bytes": max(
            value["required_peak_host_bytes"] for value in values
        ),
        "per_job": values,
        "passed": all(value["passed"] for value in values),
    }


def validate_capacity_observation(
    job: dict[str, object],
    preflight: dict[str, object],
) -> None:
    if job["peak_host_bytes"] > preflight["required_peak_host_bytes"]:
        raise RuntimeError("Stage 4 observed host peak exceeded its preflight")
    for observed, required in zip(
        job["per_gpu"],
        preflight["per_gpu"],
        strict=True,
    ):
        if observed["peak_hbm_bytes"] > required["required_peak_hbm_bytes"]:
            raise RuntimeError("Stage 4 observed HBM peak exceeded its preflight")


def aggregate_run(
    reader: LazyStage4SourceReader,
    workload: dict,
    method: str,
    destination: str,
    gpu_count: int,
    config: Stage4RuntimeConfig,
    preflights: list[dict[str, object]],
    correctness_job: dict[str, object],
    measured_jobs: list[dict[str, object]],
) -> dict[str, object]:
    samples = [value["elapsed_seconds"] for value in measured_jobs]
    median_seconds = statistics.median(samples)
    breakdown_keys = measured_jobs[0]["timing_breakdown"]
    breakdown = {
        key: statistics.median(
            value["timing_breakdown"][key] for value in measured_jobs
        )
        for key in breakdown_keys
    }
    per_gpu = []
    for index in range(gpu_count):
        values = [job["per_gpu"][index] for job in measured_jobs]
        first = values[0]
        per_gpu.append(
            {
                "index": index,
                "record_count": first["record_count"],
                "prefix_tokens": first["prefix_tokens"],
                "logical_input_bytes": first["logical_input_bytes"],
                "logical_output_bytes": first["logical_output_bytes"],
                "elapsed_seconds": statistics.median(
                    value["elapsed_seconds"] for value in values
                ),
                "peak_hbm_bytes": max(
                    value["peak_hbm_bytes"] for value in values
                ),
                "physical_input_bytes": first["physical_input_bytes"],
                "physical_output_bytes": first["physical_output_bytes"],
            }
        )
    first = measured_jobs[0]
    components = input_components(reader, method)
    run = {
        "method": method,
        "destination": destination,
        "gpu_count": gpu_count,
        "source_representations": [
            value["representation"] for value in components
        ],
        "input_components": components,
        "record_count": reader.manifest.record_count,
        "prefix_tokens": reader.manifest.prefix_tokens,
        "placement_policy": "byte_weighted_lpt",
        "per_gpu": per_gpu,
        "load_imbalance_ratio": first["load_imbalance_ratio"],
        "records_per_second": reader.manifest.record_count / median_seconds,
        "tokens_per_second": reader.manifest.prefix_tokens / median_seconds,
        "selected_runtime_config": config.to_dict(),
        "capacity_preflight": aggregate_preflights(preflights),
        "timing": {
            "samples_seconds": samples,
            "median_seconds": median_seconds,
            "breakdown_seconds": breakdown,
        },
        "logical_input_bytes": first["logical_input_bytes"],
        "physical_input_bytes": first["physical_input_bytes"],
        "logical_output_bytes": first["logical_output_bytes"],
        "physical_output_bytes": first["physical_output_bytes"],
        "peak_hbm_bytes": max(
            value["peak_hbm_bytes"] for value in measured_jobs
        ),
        "peak_host_bytes": max(
            value["peak_host_bytes"] for value in measured_jobs
        ),
        "peak_source_resident_bytes": max(
            value["peak_source_resident_bytes"]
            for value in measured_jobs
        ),
        "peak_staging_bytes": max(
            value["peak_staging_bytes"] for value in measured_jobs
        ),
        "peak_publication_queue_bytes": max(
            value["peak_publication_queue_bytes"]
            for value in measured_jobs
        ),
        "manifest": {
            "protocol": first["manifest"]["protocol"],
            "record_count": first["manifest"]["record_count"],
            "prefix_tokens": first["manifest"]["prefix_tokens"],
            "workload_content_sha256": workload["content_sha256"],
            "complete": True,
            "duplicate_free": True,
        },
        "correctness": correctness_job["correctness"],
        "source_manifest_sha256": reader.manifest_file_sha256,
    }
    if method == "selective_contiguous":
        run.update(
            {
                "action_configuration": {
                    "m": 12,
                    "start_layer": 0,
                    "end_layer": 11,
                },
                "certificate_passed": False,
                "publishable_sync_action": False,
            }
        )
    elif method == "compiled":
        run.update(
            {
                "action_configuration": {
                    "kind": "compiled_full_affine",
                },
                "certificate_passed": True,
                "publishable_sync_action": True,
            }
        )
    elif method == "exact":
        run.update(
            {
                "action_configuration": {"kind": "full_recompute"},
                "certificate_passed": True,
                "publishable_sync_action": True,
            }
        )
    elif method == "residual_p":
        run.update(
            {
                "action_configuration": {
                    "p": 8,
                    "residual_sources": ["theta0", "theta10"],
                    "exact_fallback_sources": ["theta4"],
                },
                "certificate_passed": True,
                "publishable_sync_action": True,
            }
        )
    else:
        run.update(
            {
                "action_configuration": {"kind": "no_transform"},
                "certificate_passed": False,
                "publishable_sync_action": False,
            }
        )
    return run


def full_point(
    root: Path,
    args: argparse.Namespace,
    result: dict,
    reader: LazyStage4SourceReader,
    workload: dict,
    cfg: HSTUConfig,
    programs: dict[str, object],
    method: str,
    destination: str,
    gpu_count: int,
) -> None:
    key = point_key(method, destination, gpu_count)
    existing = any(
        value["method"] == method
        and value["destination"] == destination
        and value["gpu_count"] == gpu_count
        for value in result["runs"]
    )
    if existing and key not in args.rerun_full_points:
        print(json.dumps({"point": key, "status": "already_full"}), flush=True)
        return
    if existing:
        result["runs"] = [
            value
            for value in result["runs"]
            if not (
                value["method"] == method
                and value["destination"] == destination
                and value["gpu_count"] == gpu_count
            )
        ]
        save_result(root, args, result)
        print(json.dumps({"point": key, "status": "rerun_full"}), flush=True)
    point = get_point(result, key)
    if point is None or point.get("status") != "tuned":
        raise ValueError(f"Stage 4 point {key} is not tuned")
    winner = next(
        value
        for value in point["candidates"]
        if value["candidate_id"] == point["winner_id"]
    )
    config = runtime_config(winner["runtime_config"])
    cleanup(torch.cuda.device_count())
    transforms = build_transforms(
        root,
        args,
        cfg,
        programs,
        method,
        gpu_count,
        config,
    )
    record_ids = tuple(value.record_id for value in reader.manifest.records)
    _, assignments = plan(
        reader,
        record_ids,
        method,
        config,
        gpu_count,
    )
    selection_ids = tuple(
        value["record_id"]
        for value in workload["records"]
        if value["evaluation_role"] == "program_selection"
    )
    _, calibration_assignments = plan(
        reader,
        selection_ids,
        method,
        config,
        gpu_count,
    )
    transient = tuple(winner["maximum_transient_hbm_bytes"])
    preflights = []

    def preflight() -> dict[str, object]:
        value = stage4_capacity_preflight(
            assignments,
            transforms,
            destination,
            transient,
            config.max_inflight,
            calibration_assignments=calibration_assignments,
        )
        preflights.append(value)
        if not value["passed"]:
            raise MemoryError(f"Stage 4 capacity preflight failed for {key}")
        return value

    capacity = preflight()
    correctness_job, _ = run_engine(
        root / args.source_manifest,
        reader.manifest_file_sha256,
        workload["content_sha256"],
        transforms,
        destination,
        config,
        record_ids,
        True,
        f"s4-{hashlib.sha256(key.encode()).hexdigest()[:12]}-check",
    )
    validate_capacity_observation(correctness_job, capacity)
    correctness = correctness_job["correctness"]
    expected_elements = (
        reader.manifest.prefix_tokens
        * reader.manifest.num_layers
        * reader.manifest.kv_width
        * 2
    )
    if not (
        correctness["finite"]
        and correctness["allclose"]
        and correctness["record_order_valid"]
        and correctness["lengths_offsets_valid"]
        and correctness["valid_element_count"] == expected_elements
    ):
        raise RuntimeError(f"Stage 4 full correctness failed for {key}")
    for repeat in range(WARMUP_RUNS):
        capacity = preflight()
        warmup_job, _ = run_engine(
            root / args.source_manifest,
            reader.manifest_file_sha256,
            workload["content_sha256"],
            transforms,
            destination,
            config,
            record_ids,
            False,
            (
                f"s4-{hashlib.sha256(key.encode()).hexdigest()[:12]}"
                f"-warm{repeat}"
            ),
        )
        validate_capacity_observation(warmup_job, capacity)
    measured_jobs = []
    for repeat in range(MEASURED_REPEATS):
        capacity = preflight()
        job, _ = run_engine(
            root / args.source_manifest,
            reader.manifest_file_sha256,
            workload["content_sha256"],
            transforms,
            destination,
            config,
            record_ids,
            False,
            (
                f"s4-{hashlib.sha256(key.encode()).hexdigest()[:12]}"
                f"-measure{repeat}"
            ),
        )
        validate_capacity_observation(job, capacity)
        measured_jobs.append(job)
        print(
            json.dumps(
                {
                    "point": key,
                    "full_repeat": repeat,
                    "elapsed_seconds": job["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    run = aggregate_run(
        reader,
        workload,
        method,
        destination,
        gpu_count,
        config,
        preflights,
        correctness_job,
        measured_jobs,
    )
    run["implementation_snapshot_sha256"] = result["implementation"][
        "full_phase"
    ]["code_snapshot_sha256"]
    result["runs"].append(run)
    save_result(root, args, result)
    print(
        json.dumps(
            {
                "point": key,
                "status": "full_complete",
                "median_seconds": run["timing"]["median_seconds"],
                "tokens_per_second": run["tokens_per_second"],
            }
        ),
        flush=True,
    )
    transforms = ()
    cleanup(gpu_count)


def update_status(result: dict) -> None:
    required = {
        point_key(method, destination, gpu_count)
        for method in METHODS
        for destination in DESTINATIONS
        for gpu_count in GPU_COUNTS
    }
    tuned = {
        value["key"]
        for value in result["runtime_tuning"]["points"]
        if value.get("status") == "tuned"
    }
    full = {
        point_key(
            value["method"],
            value["destination"],
            value["gpu_count"],
        )
        for value in result["runs"]
    }
    if required.issubset(full):
        result["status"] = "stage4_complete"
    elif required.issubset(tuned):
        result["status"] = "tuning_complete"
    else:
        result["status"] = "in_progress"
    result["coverage"] = {
        "required_points": len(required),
        "tuned_points": len(required.intersection(tuned)),
        "full_points": len(required.intersection(full)),
        "primary_full_points": sum(
            point_key(method, destination, gpu_count) in full
            for method in PRIMARY_METHODS
            for destination in DESTINATIONS
            for gpu_count in GPU_COUNTS
        ),
        "control_full_points": sum(
            point_key(method, destination, gpu_count) in full
            for method in CONTROL_METHODS
            for destination in DESTINATIONS
            for gpu_count in GPU_COUNTS
        ),
    }


def main() -> None:
    args = parse_args()
    invalid_reruns = set(args.rerun_full_points) - {
        point_key(method, destination, gpu_count)
        for method in METHODS
        for destination in DESTINATIONS
        for gpu_count in GPU_COUNTS
    }
    if invalid_reruns:
        raise ValueError(f"invalid Stage 4 rerun points: {sorted(invalid_reruns)}")
    root = Path(__file__).resolve().parents[1]
    (
        blueprint,
        workload,
        stage2,
        _,
        cfg,
        reader,
    ) = validate_inputs(root, args)
    result = load_or_create_result(root, args, workload, reader)
    if args.phase in {"full", "all"}:
        source_storage = reader.manifest.creation["source_preflight"]
        full_snapshot = implementation_snapshot(root)
        stale_runs = {
            point_key(
                value["method"],
                value["destination"],
                value["gpu_count"],
            )
            for value in result["runs"]
            if value.get("implementation_snapshot_sha256")
            != full_snapshot["code_snapshot_sha256"]
        }
        uncovered_stale_runs = stale_runs - set(args.rerun_full_points)
        if uncovered_stale_runs:
            raise ValueError(
                "Stage 4 implementation changed; rerun points are required: "
                f"{sorted(uncovered_stale_runs)}"
            )
        if stale_runs:
            result["runs"] = [
                value
                for value in result["runs"]
                if point_key(
                    value["method"],
                    value["destination"],
                    value["gpu_count"],
                )
                not in stale_runs
            ]
            print(
                json.dumps(
                    {
                        "status": "removed_stale_full_runs",
                        "points": sorted(stale_runs),
                    }
                ),
                flush=True,
            )
        result["environment"].update(
            {
                "python": platform.python_version(),
                "cuda_runtime": torch.version.cuda,
                "cuda_driver": subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                        "-i",
                        "0",
                    ],
                    text=True,
                ).strip(),
                "source_storage": {
                    "mount": source_storage["mount"],
                    "device": source_storage["source"],
                    "device_model": source_storage["device_model"],
                    "filesystem": source_storage["filesystem"],
                    "free_bytes_before_materialization": source_storage[
                        "observed_free_bytes"
                    ],
                },
                "page_cache_condition": (
                    "correctness and one complete untimed warmup precede "
                    "three measured repetitions without explicit page-cache "
                    "eviction; source shards reopen and decode every "
                    "repetition"
                ),
            }
        )
        result["implementation"] = {
            "full_phase": full_snapshot,
            "tuning_to_full_amendment": (
                "after tuning, source-wave peak accounting and capacity "
                "preflight were corrected from one extent to the actual "
                "max-inflight rolling window, and full-cohort device waves "
                "were combined with calibration-subset compute slack rather "
                "than retaining device-specific calibration placement; "
                "transformation, extent planning, transfer, and publication "
                "semantics are unchanged, and every full point uses the "
                "corrected code"
            ),
        }
        save_result(root, args, result)
    programs = load_programs(
        root,
        stage2,
        blueprint["data_and_model"]["model"],
    )
    points = [
        (method, destination, gpu_count)
        for method in args.methods
        for destination in args.destinations
        for gpu_count in args.gpu_counts
    ]
    if args.phase in {"tune", "all"}:
        for method, destination, gpu_count in points:
            tune_point(
                root,
                args,
                result,
                reader,
                workload,
                cfg,
                programs,
                method,
                destination,
                gpu_count,
            )
            update_status(result)
            save_result(root, args, result)
    if args.phase in {"full", "all"}:
        for method, destination, gpu_count in points:
            full_point(
                root,
                args,
                result,
                reader,
                workload,
                cfg,
                programs,
                method,
                destination,
                gpu_count,
            )
            update_status(result)
            save_result(root, args, result)
    update_status(result)
    save_result(root, args, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "coverage": result["coverage"],
                "output": args.output,
                "sha256": sha256_file(root / args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
