from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import benchmark_cohortkv_stage4_5_resident_ceiling as ceiling
import torch

from hstu_kvcache.migration.stage45_oldkv import (
    DIRECT_OLDKV_ENGINE_PROTOCOL,
    DirectOldKVFusedOperator,
    DirectOldKVTransform,
    Stage45OldKVEngine,
    build_stage45_oldkv_plan,
    load_direct_oldkv_program,
    stage45_oldkv_preflight,
)
from hstu_kvcache.migration.stage45_reclaim import (
    Stage45ReclaimingEngine,
    allocate_reclaimable_old_kv,
    stage45_reclaim_preflight,
)
from hstu_kvcache.migration.stage45_resident import (
    Stage45ResidentPlan,
    build_stage45_resident_plan,
    materialize_stage45_resident_source,
)

PROTOCOL = "cohortkv_single_config_stage4_5_oldkv_system_v1"
DEFAULT_COMPILER = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_compiler_seed0.json"
)
DEFAULT_CERTIFICATE = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_certificate_seed0.json"
)
DEFAULT_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_system_seed0.json"
)
GPU_COUNTS = (1, 2, 4)
WARMUP_RUNS = 1
MEASURED_REPEATS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", default=ceiling.DEFAULT_BLUEPRINT)
    parser.add_argument(
        "--workload-manifest",
        default=ceiling.DEFAULT_WORKLOAD,
    )
    parser.add_argument("--stage2-summary", default=ceiling.DEFAULT_STAGE2)
    parser.add_argument("--stage3-summary", default=ceiling.DEFAULT_STAGE3)
    parser.add_argument("--stage4-summary", default=ceiling.DEFAULT_STAGE4)
    parser.add_argument("--training-result", default=ceiling.DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=ceiling.DEFAULT_CHECKPOINTS)
    parser.add_argument("--source-manifest", default=ceiling.DEFAULT_SOURCE)
    parser.add_argument("--compiler-result", default=DEFAULT_COMPILER)
    parser.add_argument("--certificate-result", default=DEFAULT_CERTIFICATE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gpu-counts",
        nargs="+",
        type=int,
        choices=GPU_COUNTS,
        default=[1, 4],
    )
    parser.add_argument(
        "--measured-repeats",
        type=int,
        default=MEASURED_REPEATS,
    )
    return parser.parse_args()


def implementation_snapshot(root: Path) -> dict[str, object]:
    paths = (
        Path("src/hstu_kvcache/migration/stage45_oldkv.py"),
        Path("src/hstu_kvcache/migration/stage45_reclaim.py"),
        Path("src/hstu_kvcache/migration/stage45_resident.py"),
        Path("scripts/benchmark_cohortkv_stage4_5_oldkv.py"),
    )
    files = [
        {
            "path": str(path),
            "bytes": (root / path).stat().st_size,
            "sha256": ceiling.sha256_file(root / path),
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
            ceiling.canonical_json(files)
        ).hexdigest(),
    }


def load_inputs(
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict, dict, dict, dict, object, object, dict, dict]:
    (
        blueprint,
        workload,
        stage2,
        _,
        stage4,
        config,
        reader,
    ) = ceiling.validate_inputs(root, args)
    compiler = json.loads((root / args.compiler_result).read_text())
    certificate = json.loads((root / args.certificate_result).read_text())
    if (
        compiler.get("protocol")
        != "cohortkv_single_config_stage4_5_oldkv_compiler_v1"
        or compiler.get("status") != "oldkv_program_transport_frozen"
        or compiler.get("labels_used") is not False
        or certificate.get("protocol")
        != "cohortkv_single_config_stage4_5_oldkv_certificate_v1"
        or certificate.get("status")
        != "oldkv_semantic_certificate_frozen"
        or certificate.get("labels_used") is not False
        or certificate["inputs"]["compiler_result"]["sha256"]
        != ceiling.sha256_file(root / args.compiler_result)
        or not certificate["aggregate"]["all_selected_direct_oldkv"]
        or not certificate["aggregate"]["all_exact_fallback"]
        or compiler["inputs"]["source_manifest_sha256"]
        != reader.manifest_file_sha256
        or compiler["inputs"]["workload_content_sha256"]
        != workload["content_sha256"]
    ):
        raise ValueError("Stage 4.5 old-K/V frozen inputs differ")
    return (
        blueprint,
        workload,
        stage2,
        stage4,
        config,
        reader,
        compiler,
        certificate,
    )


def load_direct_programs(
    compiler: dict,
    config,
) -> dict[str, object]:
    programs = {}
    for descriptor in compiler["representation"]["programs"]:
        source = descriptor["source_version"]
        program, _ = load_direct_oldkv_program(
            descriptor["path"],
            expected_sha256=descriptor["sha256"],
            expected_source_version=source,
            expected_target_version="theta11",
            expected_num_layers=config.num_layers,
            expected_kv_width=config.num_heads * config.head_dim,
        )
        programs[source] = program
    if set(programs) != set(ceiling.SOURCE_VERSIONS):
        raise ValueError("Stage 4.5 direct old-K/V programs are incomplete")
    return programs


def build_direct_transforms(
    programs: dict[str, object],
    launch: dict[str, int],
    gpu_count: int,
) -> tuple[DirectOldKVTransform, ...]:
    return tuple(
        DirectOldKVTransform(
            programs,
            DirectOldKVFusedOperator(**launch),
            torch.device("cuda", index),
        )
        for index in range(gpu_count)
    )


def pair_exact_layout(
    exact: Stage45ResidentPlan,
    direct: Stage45ResidentPlan,
) -> Stage45ResidentPlan:
    exact_map = {value.extent_id: value for value in exact.extents}
    if set(exact_map) != {value.extent_id for value in direct.extents}:
        raise ValueError("Stage 4.5 paired extent IDs differ")
    assignments = tuple(
        tuple(exact_map[value.extent_id] for value in assignment)
        for assignment in direct.assignments
    )
    paired = dataclasses.replace(exact, assignments=assignments)
    direct_layout = ceiling.target_layout(direct)
    exact_layout = ceiling.target_layout(paired)
    if direct_layout != exact_layout:
        raise ValueError("Stage 4.5 paired target layouts differ")
    return paired


def compact_report(report) -> dict[str, object]:
    payload = ceiling.compact_report(report)
    ordered = sorted(report.manifest.record_ids)
    payload["manifest"]["coverage_sha256"] = hashlib.sha256(
        ceiling.canonical_json(ordered)
    ).hexdigest()
    return payload


def cleanup(transforms) -> None:
    gc.collect()
    for transform in transforms:
        with torch.cuda.device(transform.device):
            torch.cuda.synchronize(transform.device)
            torch.cuda.empty_cache()


def run_direct_once(
    engine: Stage45OldKVEngine,
    validate: bool,
    job_id: str,
) -> dict[str, object]:
    allocation_started = time.perf_counter()
    old_cache = allocate_reclaimable_old_kv(
        engine.plan,
        engine.transforms,
        zero=True,
    )
    allocation_seconds = time.perf_counter() - allocation_started
    engine.install_old_cache(old_cache)
    result = engine.run(
        validate_zero_source=validate,
        job_id=job_id,
    )
    payload = compact_report(result.report)
    payload["reclamation"] = result.reclamation.to_dict()
    payload["benchmark_old_cache_reset_seconds"] = allocation_seconds
    del result, old_cache
    cleanup(engine.transforms)
    return payload


def run_exact_once(
    engine: Stage45ReclaimingEngine,
    validate: bool,
    job_id: str,
) -> dict[str, object]:
    allocation_started = time.perf_counter()
    old_cache = allocate_reclaimable_old_kv(
        engine.source.plan,
        engine.transforms,
        zero=True,
    )
    allocation_seconds = time.perf_counter() - allocation_started
    engine.install_old_cache(old_cache)
    result = engine.run(validate=validate, job_id=job_id)
    payload = compact_report(result.report)
    payload["reclamation"] = result.reclamation.to_dict()
    payload["benchmark_old_cache_reset_seconds"] = allocation_seconds
    del result, old_cache
    cleanup(engine.transforms)
    return payload


def summarize_jobs(
    jobs: list[dict[str, object]],
) -> dict[str, object]:
    output = ceiling.summarize_samples(jobs)
    output["median_old_cache_reset_seconds"] = statistics.median(
        float(value["benchmark_old_cache_reset_seconds"])
        for value in jobs
    )
    output["maximum_peak_old_plus_new_kv_bytes"] = max(
        int(value["reclamation"]["peak_old_plus_new_kv_bytes"])
        for value in jobs
    )
    baselines: dict[int, set[int]] = {}
    for job in jobs:
        for device in job["per_gpu"]:
            baselines.setdefault(int(device["index"]), set()).add(
                int(device["baseline_hbm_bytes"])
            )
    output["stable_hbm_baseline"] = all(
        len(values) == 1 for values in baselines.values()
    )
    output["baseline_hbm_bytes"] = {
        str(index): next(iter(values))
        for index, values in baselines.items()
        if len(values) == 1
    }
    return output


def run_direct_point(
    root: Path,
    args: argparse.Namespace,
    result: dict,
    stage4: dict,
    reader,
    programs: dict[str, object],
    launch: dict[str, int],
    gpu_count: int,
) -> Stage45ResidentPlan:
    key = f"compiled_old_kv:existing_old_kv_hbm:{gpu_count}"
    existing = next(
        (value for value in result["points"] if value["key"] == key),
        None,
    )
    if existing is not None:
        print(
            json.dumps({"point": key, "status": "already_complete"}),
            flush=True,
        )
        transforms = build_direct_transforms(
            programs,
            launch,
            gpu_count,
        )
        runtime = ceiling.runtime_config(stage4, "compiled", gpu_count)
        plan = build_stage45_oldkv_plan(
            root / args.source_manifest,
            transforms,
            runtime,
            expected_source_manifest_sha256=reader.manifest_file_sha256,
            expected_workload_content_sha256=(
                reader.manifest.workload_content_sha256
            ),
        )
        del transforms
        cleanup(())
        return plan
    runtime = ceiling.runtime_config(stage4, "compiled", gpu_count)
    transforms = build_direct_transforms(programs, launch, gpu_count)
    plan = build_stage45_oldkv_plan(
        root / args.source_manifest,
        transforms,
        runtime,
        expected_source_manifest_sha256=reader.manifest_file_sha256,
        expected_workload_content_sha256=(
            reader.manifest.workload_content_sha256
        ),
    )
    preflight = stage45_oldkv_preflight(plan, transforms)
    print(
        json.dumps(
            {
                "point": key,
                "status": "preflight",
                "passed": preflight["passed"],
                "per_gpu": preflight["per_gpu"],
            }
        ),
        flush=True,
    )
    if not preflight["passed"]:
        raise MemoryError(f"Stage 4.5 point {key} does not fit")
    engine = Stage45OldKVEngine(plan, transforms)
    correctness = run_direct_once(
        engine,
        True,
        f"s45o-direct-{gpu_count}-check",
    )
    valid = correctness["correctness"]
    if not (
        valid["finite"]
        and valid["allclose"]
        and valid["record_order_valid"]
        and valid["lengths_offsets_valid"]
        and correctness["reclamation"]["final_old_kv_bytes"] == 0
    ):
        raise RuntimeError(f"Stage 4.5 point {key} is incorrect")
    warmups = [
        run_direct_once(
            engine,
            False,
            f"s45o-direct-{gpu_count}-warm{repeat}",
        )
        for repeat in range(WARMUP_RUNS)
    ]
    measured = []
    for repeat in range(args.measured_repeats):
        job = run_direct_once(
            engine,
            False,
            f"s45o-direct-{gpu_count}-measure{repeat}",
        )
        measured.append(job)
        print(
            json.dumps(
                {
                    "point": key,
                    "repeat": repeat + 1,
                    "total": args.measured_repeats,
                    "elapsed_seconds": job["elapsed_seconds"],
                    "peak_hbm_bytes": max(
                        value["peak_hbm_bytes"]
                        for value in job["per_gpu"]
                    ),
                }
            ),
            flush=True,
        )
    point = {
        "key": key,
        "method": "compiled_old_kv",
        "source_tier": "existing_old_kv_hbm",
        "gpu_count": gpu_count,
        "status": "complete",
        "runtime_config": runtime.to_dict(),
        "operator": {
            "kind": "direct_oldkv_fused_fp16",
            "launch": launch,
        },
        "target_layout": ceiling.target_layout(plan),
        "plan": plan.to_dict(),
        "capacity_preflight": preflight,
        "source_lifecycle": {
            "additional_source_state_bytes": 0,
            "standing_source": "existing serving old K/V in HBM",
            "preload_seconds": 0.0,
            "h2d_bytes": 0,
        },
        "correctness_job": correctness,
        "warmup_jobs": warmups,
        "measured_jobs": measured,
        "summary": summarize_jobs(measured),
    }
    if not point["summary"]["stable_hbm_baseline"]:
        raise RuntimeError(f"Stage 4.5 point {key} grows HBM")
    result["points"].append(point)
    ceiling.write_json_atomic(root / args.output, result)
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
    del engine, transforms
    cleanup(())
    return plan


def run_exact_point(
    root: Path,
    args: argparse.Namespace,
    result: dict,
    stage4: dict,
    config,
    reader,
    direct_plan: Stage45ResidentPlan,
    gpu_count: int,
) -> None:
    key = f"exact:raw_history_hbm:{gpu_count}"
    if any(value["key"] == key for value in result["points"]):
        print(
            json.dumps({"point": key, "status": "already_complete"}),
            flush=True,
        )
        return
    runtime = ceiling.runtime_config(stage4, "exact", gpu_count)
    transforms = ceiling.build_transforms(
        root,
        args,
        config,
        {},
        "exact",
        gpu_count,
        runtime,
    )
    plan = build_stage45_resident_plan(
        root / args.source_manifest,
        transforms,
        "hbm_resident",
        runtime,
        expected_source_manifest_sha256=reader.manifest_file_sha256,
        expected_workload_content_sha256=(
            reader.manifest.workload_content_sha256
        ),
    )
    plan = pair_exact_layout(plan, direct_plan)
    preflight = stage45_reclaim_preflight(plan, transforms)
    print(
        json.dumps(
            {
                "point": key,
                "status": "preflight",
                "passed": preflight["passed"],
                "per_gpu": preflight["per_gpu"],
            }
        ),
        flush=True,
    )
    if not preflight["passed"]:
        raise MemoryError(f"Stage 4.5 point {key} does not fit")
    source = materialize_stage45_resident_source(
        plan,
        transforms,
        require_capacity=False,
    )
    print(
        json.dumps(
            {
                "point": key,
                "status": "source_ready",
                "preload_seconds": source.preload.elapsed_seconds,
                "standing_hbm_bytes": source.preload.standing_hbm_bytes,
            }
        ),
        flush=True,
    )
    engine = Stage45ReclaimingEngine(source, transforms)
    correctness = run_exact_once(
        engine,
        True,
        f"s45o-exact-{gpu_count}-check",
    )
    valid = correctness["correctness"]
    if not (
        valid["finite"]
        and valid["allclose"]
        and valid["record_order_valid"]
        and valid["lengths_offsets_valid"]
        and correctness["reclamation"]["final_old_kv_bytes"] == 0
    ):
        raise RuntimeError(f"Stage 4.5 point {key} is incorrect")
    warmups = [
        run_exact_once(
            engine,
            False,
            f"s45o-exact-{gpu_count}-warm{repeat}",
        )
        for repeat in range(WARMUP_RUNS)
    ]
    measured = []
    for repeat in range(args.measured_repeats):
        job = run_exact_once(
            engine,
            False,
            f"s45o-exact-{gpu_count}-measure{repeat}",
        )
        measured.append(job)
        print(
            json.dumps(
                {
                    "point": key,
                    "repeat": repeat + 1,
                    "total": args.measured_repeats,
                    "elapsed_seconds": job["elapsed_seconds"],
                    "peak_hbm_bytes": max(
                        value["peak_hbm_bytes"]
                        for value in job["per_gpu"]
                    ),
                }
            ),
            flush=True,
        )
    point = {
        "key": key,
        "method": "exact",
        "source_tier": "raw_history_hbm",
        "gpu_count": gpu_count,
        "status": "complete",
        "runtime_config": runtime.to_dict(),
        "target_layout": ceiling.target_layout(plan),
        "plan": plan.to_dict(),
        "capacity_preflight": preflight,
        "preload": source.preload.to_dict(),
        "correctness_job": correctness,
        "warmup_jobs": warmups,
        "measured_jobs": measured,
        "summary": summarize_jobs(measured),
    }
    if not point["summary"]["stable_hbm_baseline"]:
        raise RuntimeError(f"Stage 4.5 point {key} grows HBM")
    result["points"].append(point)
    ceiling.write_json_atomic(root / args.output, result)
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
    del engine, source, transforms
    cleanup(())


def stage4_baseline(
    stage4: dict,
    method: str,
    gpu_count: int,
) -> dict[str, object]:
    run = next(
        value
        for value in stage4["runs"]
        if value["method"] == method
        and value["destination"] == "hbm"
        and value["gpu_count"] == gpu_count
    )
    return {
        "protocol": stage4["protocol"],
        "source_kind": "filesystem_per_job",
        "median_seconds": run["timing"]["median_seconds"],
    }


def comparisons(
    result: dict,
    stage4: dict,
) -> list[dict[str, object]]:
    points = {value["key"]: value for value in result["points"]}
    output = []
    for gpu_count in result["requested_matrix"]["gpu_counts"]:
        direct = points[
            f"compiled_old_kv:existing_old_kv_hbm:{gpu_count}"
        ]
        exact = points[f"exact:raw_history_hbm:{gpu_count}"]
        if direct["target_layout"] != exact["target_layout"]:
            raise RuntimeError("Stage 4.5 paired target layouts differ")
        direct_samples = direct["summary"]["samples_seconds"]
        exact_samples = exact["summary"]["samples_seconds"]
        direct_median = direct["summary"]["median_seconds"]
        exact_median = exact["summary"]["median_seconds"]
        savings = exact_median - direct_median
        variation = max(
            direct["summary"]["range_seconds"],
            exact["summary"]["range_seconds"],
        )
        output.append(
            {
                "gpu_count": gpu_count,
                "compiled_method": "compiled_old_kv",
                "compiled_source_tier": "existing_old_kv_hbm",
                "exact_source_tier": "raw_history_hbm",
                "additional_compiled_source_state_bytes": 0,
                "target_layout": direct["target_layout"],
                "compiled_median_seconds": direct_median,
                "exact_median_seconds": exact_median,
                "speedup": exact_median / direct_median,
                "median_savings_seconds": savings,
                "maximum_measured_variation_band_seconds": variation,
                "all_compiled_below_all_exact": (
                    max(direct_samples) < min(exact_samples)
                ),
                "difference_exceeds_variation_band": savings > variation,
                "completion_gate_passed": (
                    savings > 0
                    and max(direct_samples) < min(exact_samples)
                    and savings > variation
                    and direct["summary"]["stable_hbm_baseline"]
                    and exact["summary"]["stable_hbm_baseline"]
                ),
                "compiled_reclamation": direct["correctness_job"][
                    "reclamation"
                ],
                "exact_reclamation": exact["correctness_job"][
                    "reclamation"
                ],
                "exact_raw_history_preload": exact["preload"],
                "stage4_compiled_baseline": stage4_baseline(
                    stage4,
                    "compiled",
                    gpu_count,
                ),
                "stage4_exact_baseline": stage4_baseline(
                    stage4,
                    "exact",
                    gpu_count,
                ),
            }
        )
    return output


def new_result(
    root: Path,
    args: argparse.Namespace,
    workload: dict,
    reader,
    compiler: dict,
    certificate: dict,
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "parent_protocol": DIRECT_OLDKV_ENGINE_PROTOCOL,
        "status": "in_progress",
        "study_stage": "stage4_5_cd_direct_oldkv_seed0",
        "seed": 0,
        "labels_used": False,
        "record_count": workload["summary"]["records"],
        "prefix_tokens": workload["summary"]["prefix_tokens"],
        "measurement_boundary": (
            "existing old FP16 K/V in HBM plus source-plan program/raw "
            "history through complete replacement HBM K/V and atomic commit"
        ),
        "requested_matrix": {
            "methods": ["compiled_old_kv", "exact"],
            "gpu_counts": list(args.gpu_counts),
            "warmup_runs": WARMUP_RUNS,
            "measured_repeats": args.measured_repeats,
        },
        "source_policy": {
            "normal_action": "compiled_old_kv",
            "source_representation": "existing_old_kv_fp16",
            "additional_normx_bytes": 0,
            "placement": "existing serving HBM cache",
            "supply": "direct device read",
            "reclamation": "extent-wise after replacement stage",
            "capacity_failure_action": "exact",
            "missing_program_action": "exact",
            "missing_old_kv_action": "exact",
        },
        "inputs": {
            "workload_content_sha256": workload["content_sha256"],
            "source_manifest_sha256": reader.manifest_file_sha256,
            "compiler_result": {
                "path": args.compiler_result,
                "sha256": ceiling.sha256_file(
                    root / args.compiler_result
                ),
            },
            "certificate_result": {
                "path": args.certificate_result,
                "sha256": ceiling.sha256_file(
                    root / args.certificate_result
                ),
            },
            "program_set_sha256": compiler["representation"][
                "program_set_sha256"
            ],
            "semantic_certificate_minimum_worst_view_recovery": (
                certificate["aggregate"][
                    "minimum_worst_view_recovery"
                ]
            ),
        },
        "operator": {
            "kind": "direct_oldkv_fused_fp16",
            "launch": compiler["operator_selection"]["winner"],
            "program_bytes": compiler["representation"]["program_bytes"],
        },
        "implementation": implementation_snapshot(root),
        "points": [],
        "comparisons": [],
    }


def load_or_create(
    root: Path,
    args: argparse.Namespace,
    workload: dict,
    reader,
    compiler: dict,
    certificate: dict,
) -> dict:
    expected = new_result(
        root,
        args,
        workload,
        reader,
        compiler,
        certificate,
    )
    path = root / args.output
    if not path.exists():
        return expected
    result = json.loads(path.read_text())
    for key in (
        "protocol",
        "record_count",
        "prefix_tokens",
        "requested_matrix",
        "inputs",
        "operator",
        "implementation",
    ):
        if result.get(key) != expected[key]:
            raise ValueError("existing Stage 4.5 system result differs")
    return result


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    (
        _,
        workload,
        _,
        stage4,
        config,
        reader,
        compiler,
        certificate,
    ) = load_inputs(root, args)
    programs = load_direct_programs(compiler, config)
    result = load_or_create(
        root,
        args,
        workload,
        reader,
        compiler,
        certificate,
    )
    launch = compiler["operator_selection"]["winner"]
    for gpu_count in args.gpu_counts:
        direct_plan = run_direct_point(
            root,
            args,
            result,
            stage4,
            reader,
            programs,
            launch,
            gpu_count,
        )
        run_exact_point(
            root,
            args,
            result,
            stage4,
            config,
            reader,
            direct_plan,
            gpu_count,
        )
    result["comparisons"] = comparisons(result, stage4)
    if any(
        not value["completion_gate_passed"]
        for value in result["comparisons"]
    ):
        raise RuntimeError("Stage 4.5 direct old-K/V gate failed")
    result["status"] = "oldkv_system_representative_complete"
    result["last_invocation_seconds"] = time.perf_counter() - started
    ceiling.write_json_atomic(root / args.output, result)
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
