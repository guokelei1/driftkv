from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import benchmark_cohortkv_stage4_5_resident_ceiling as ceiling
import torch

from hstu_kvcache.migration import sha256_file
from hstu_kvcache.migration.stage45_reclaim import (
    STAGE45_RECLAIM_PROTOCOL,
    Stage45ReclaimingEngine,
    allocate_reclaimable_old_kv,
    stage45_reclaim_preflight,
)
from hstu_kvcache.migration.stage45_resident import (
    build_stage45_resident_plan,
    materialize_stage45_resident_source,
)

PROTOCOL = "cohortkv_single_config_stage4_5_reclaim_candidate_v1"
DEFAULT_CEILING = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_resident_ceiling_seed0.json"
)
DEFAULT_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_reclaim_candidate_seed0.json"
)
METHODS = ("compiled", "exact")
GPU_COUNTS = (1, 4)
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
    parser.add_argument(
        "--training-result",
        default=ceiling.DEFAULT_TRAINING,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=ceiling.DEFAULT_CHECKPOINTS,
    )
    parser.add_argument(
        "--source-manifest",
        default=ceiling.DEFAULT_SOURCE,
    )
    parser.add_argument("--ceiling-result", default=DEFAULT_CEILING)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gpu-counts",
        nargs="+",
        type=int,
        choices=GPU_COUNTS,
        default=list(GPU_COUNTS),
    )
    parser.add_argument(
        "--record-scope",
        choices=("program_selection", "full"),
        default="full",
    )
    parser.add_argument(
        "--measured-repeats",
        type=int,
        default=MEASURED_REPEATS,
    )
    return parser.parse_args()


def implementation_snapshot(root: Path) -> dict[str, object]:
    paths = (
        Path("src/hstu_kvcache/migration/stage45_resident.py"),
        Path("src/hstu_kvcache/migration/stage45_reclaim.py"),
        Path("scripts/benchmark_cohortkv_stage4_5_resident_ceiling.py"),
        Path("scripts/benchmark_cohortkv_stage4_5_reclaim_candidate.py"),
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
            ceiling.canonical_json(files)
        ).hexdigest(),
    }


def validate_ceiling(
    root: Path,
    args: argparse.Namespace,
    workload: dict,
    stage4: dict,
    reader,
) -> dict:
    result = json.loads((root / args.ceiling_result).read_text())
    if (
        result.get("protocol") != ceiling.STAGE45_RESIDENT_PROTOCOL
        or result.get("status") != "stage4_5_a_complete"
        or result.get("labels_used") is not False
        or result["inputs"]["workload"]["content_sha256"]
        != workload["content_sha256"]
        or result["inputs"]["stage4"]["sha256"]
        != sha256_file(root / args.stage4_summary)
        or result["inputs"]["source_manifest"]["sha256"]
        != reader.manifest_file_sha256
    ):
        raise ValueError("Stage 4.5 ceiling result is invalid")
    comparisons = {
        (value["source_tier"], value["gpu_count"]): value
        for value in result["comparisons"]
    }
    if any(
        not comparisons[("dram_resident", gpu_count)][
            "resident_completion_gate_passed"
        ]
        for gpu_count in args.gpu_counts
    ):
        raise ValueError("Stage 4.5 DRAM resident ceiling did not pass")
    frozen_stage4 = stage4["source_manifest"]["sha256"]
    if frozen_stage4 != reader.manifest_file_sha256:
        raise ValueError("Stage 4.5 ceiling and Stage 4 source differ")
    return result


def selected_record_ids(
    workload: dict,
    scope: str,
) -> tuple[int, ...]:
    if scope == "full":
        return tuple(value["record_id"] for value in workload["records"])
    return tuple(
        value["record_id"]
        for value in workload["records"]
        if value["evaluation_role"] == "program_selection"
    )


def run_reclaim_once(
    engine: Stage45ReclaimingEngine,
    validate: bool,
    job_id: str,
) -> dict[str, object]:
    allocation_started = time.perf_counter()
    old_cache = allocate_reclaimable_old_kv(
        engine.source.plan,
        engine.transforms,
    )
    old_allocation_seconds = time.perf_counter() - allocation_started
    engine.install_old_cache(old_cache)
    result = engine.run(validate=validate, job_id=job_id)
    payload = ceiling.compact_report(result.report)
    payload["reclamation"] = result.reclamation.to_dict()
    payload["benchmark_old_footprint_allocation_seconds"] = (
        old_allocation_seconds
    )
    del result, old_cache
    gc.collect()
    for transform in engine.transforms:
        with torch.cuda.device(transform.device):
            torch.cuda.synchronize(transform.device)
            torch.cuda.empty_cache()
    return payload


def summarize_jobs(jobs: list[dict[str, object]]) -> dict[str, object]:
    summary = ceiling.summarize_samples(jobs)
    summary["median_old_footprint_allocation_seconds"] = statistics.median(
        value["benchmark_old_footprint_allocation_seconds"]
        for value in jobs
    )
    summary["maximum_peak_old_plus_new_kv_bytes"] = max(
        value["reclamation"]["peak_old_plus_new_kv_bytes"]
        for value in jobs
    )
    return summary


def new_result(
    root: Path,
    args: argparse.Namespace,
    workload: dict,
    stage4: dict,
    ceiling_result: dict,
    reader,
) -> dict[str, object]:
    record_ids = selected_record_ids(workload, args.record_scope)
    return {
        "protocol": PROTOCOL,
        "parent_protocol": STAGE45_RECLAIM_PROTOCOL,
        "status": "in_progress",
        "study_stage": "stage4_5_bc_pinned_dram_extent_reclaim_seed0",
        "seed": 0,
        "labels_used": False,
        "record_scope": args.record_scope,
        "record_count": len(record_ids),
        "prefix_tokens": sum(
            reader.manifest.record_map[value].prefix_tokens
            for value in record_ids
        ),
        "candidate": {
            "source_representation": "normalized_capsule_fp16",
            "source_placement": "standing_pinned_dram",
            "source_supply": "asynchronous_extent_h2d",
            "target_placement": "hbm",
            "old_kv_policy": "retire_after_replacement_stage",
            "cold_policy": "exact_fallback_not_yet_integrated",
            "failure_atomicity": "Stage 5 open",
        },
        "measurement_boundary": (
            "pinned source and old HBM K/V ready through complete "
            "replacement HBM target and atomically committed manifest"
        ),
        "requested_matrix": {
            "methods": list(METHODS),
            "source_tier": "dram_resident",
            "gpu_counts": list(args.gpu_counts),
            "warmup_runs": WARMUP_RUNS,
            "measured_repeats": args.measured_repeats,
        },
        "inputs": {
            "workload_content_sha256": workload["content_sha256"],
            "stage4_summary_sha256": sha256_file(
                root / args.stage4_summary
            ),
            "ceiling_result": {
                "path": args.ceiling_result,
                "sha256": sha256_file(root / args.ceiling_result),
                "protocol": ceiling_result["protocol"],
            },
            "source_manifest_sha256": reader.manifest_file_sha256,
        },
        "source_capture_disclosure": {
            "shared_multi_representation_materialization_seconds": stage4[
                "source_manifest"
            ]["materialization"]["elapsed_seconds"],
            "capsule_logical_bytes": stage4["source_manifest"][
                "materialization"
            ]["logical_bytes"]["normalized_capsule_fp16"],
            "capsule_physical_bytes": stage4["source_manifest"][
                "materialization"
            ]["physical_bytes"]["normalized_capsule_fp16"],
            "limitation": (
                "the frozen Stage 4 source build did not isolate capsule-only "
                "capture time"
            ),
        },
        "implementation": implementation_snapshot(root),
        "points": [],
        "comparisons": [],
    }


def load_or_create_result(
    root: Path,
    args: argparse.Namespace,
    workload: dict,
    stage4: dict,
    ceiling_result: dict,
    reader,
) -> dict:
    expected = new_result(
        root,
        args,
        workload,
        stage4,
        ceiling_result,
        reader,
    )
    path = root / args.output
    if not path.exists():
        return expected
    result = json.loads(path.read_text())
    for key in (
        "protocol",
        "record_scope",
        "record_count",
        "prefix_tokens",
        "requested_matrix",
        "inputs",
        "implementation",
    ):
        if result.get(key) != expected[key]:
            raise ValueError(
                "existing Stage 4.5 reclaim result belongs to different inputs"
            )
    return result


def run_point(
    root: Path,
    args: argparse.Namespace,
    result: dict,
    workload: dict,
    stage4: dict,
    config,
    programs,
    reader,
    method: str,
    gpu_count: int,
) -> None:
    key = f"{method}:dram_resident:reclaim:{gpu_count}"
    if any(value["key"] == key for value in result["points"]):
        print(json.dumps({"point": key, "status": "already_complete"}), flush=True)
        return
    runtime = ceiling.runtime_config(stage4, method, gpu_count)
    transforms = ceiling.build_transforms(
        root,
        args,
        config,
        programs,
        method,
        gpu_count,
        runtime,
    )
    record_ids = selected_record_ids(workload, args.record_scope)
    plan = build_stage45_resident_plan(
        root / args.source_manifest,
        transforms,
        "dram_resident",
        runtime,
        record_ids,
        expected_source_manifest_sha256=reader.manifest_file_sha256,
        expected_workload_content_sha256=workload["content_sha256"],
    )
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
        raise MemoryError(f"Stage 4.5 reclaim point {key} does not fit")
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
                "standing_host_bytes": source.preload.standing_host_bytes,
            }
        ),
        flush=True,
    )
    engine = Stage45ReclaimingEngine(source, transforms)
    correctness_job = run_reclaim_once(
        engine,
        True,
        f"s45r-{method}-{gpu_count}-check",
    )
    correctness = correctness_job["correctness"]
    if not (
        correctness["finite"]
        and correctness["allclose"]
        and correctness["record_order_valid"]
        and correctness["lengths_offsets_valid"]
        and correctness_job["reclamation"]["final_old_kv_bytes"] == 0
    ):
        raise RuntimeError(f"Stage 4.5 reclaim point {key} is incorrect")
    warmup_jobs = [
        run_reclaim_once(
            engine,
            False,
            f"s45r-{method}-{gpu_count}-warm{repeat}",
        )
        for repeat in range(WARMUP_RUNS)
    ]
    measured_jobs = []
    for repeat in range(args.measured_repeats):
        job = run_reclaim_once(
            engine,
            False,
            f"s45r-{method}-{gpu_count}-measure{repeat}",
        )
        measured_jobs.append(job)
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
        "method": method,
        "gpu_count": gpu_count,
        "status": "complete",
        "runtime_config": runtime.to_dict(),
        "target_layout": ceiling.target_layout(plan),
        "plan": plan.to_dict(),
        "capacity_preflight": preflight,
        "preload": source.preload.to_dict(),
        "correctness_job": correctness_job,
        "warmup_jobs": warmup_jobs,
        "measured_jobs": measured_jobs,
        "summary": summarize_jobs(measured_jobs),
    }
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
    del engine, source, plan, transforms
    ceiling.cleanup(gpu_count)


def stage4_baseline(stage4: dict, method: str, gpu_count: int) -> dict:
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
        "samples_seconds": run["timing"]["samples_seconds"],
    }


def build_comparisons(result: dict, stage4: dict) -> list[dict[str, object]]:
    points = {value["key"]: value for value in result["points"]}
    comparisons = []
    for gpu_count in result["requested_matrix"]["gpu_counts"]:
        compiled = points[f"compiled:dram_resident:reclaim:{gpu_count}"]
        exact = points[f"exact:dram_resident:reclaim:{gpu_count}"]
        if compiled["target_layout"] != exact["target_layout"]:
            raise RuntimeError(
                "Stage 4.5 reclaim paired target layouts differ"
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
                "gpu_count": gpu_count,
                "source_tier": "dram_resident",
                "old_kv_policy": "extent_reclaim",
                "target_layout": compiled["target_layout"],
                "compiled_median_seconds": (
                    compiled_summary["median_seconds"]
                ),
                "exact_median_seconds": exact_summary["median_seconds"],
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
                "completion_gate_passed": (
                    savings > 0
                    and max(compiled_samples) < min(exact_samples)
                    and savings > variation
                ),
                "incremental_preload_seconds": preload_delta,
                "break_even_updates": (
                    None
                    if savings <= 0
                    else max(1, int(preload_delta / savings) + 1)
                ),
                "compiled_preload": compiled["preload"],
                "exact_preload": exact["preload"],
                "compiled_reclamation": compiled["correctness_job"][
                    "reclamation"
                ],
                "exact_reclamation": exact["correctness_job"][
                    "reclamation"
                ],
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
        config,
        reader,
    ) = ceiling.validate_inputs(root, args)
    ceiling_result = validate_ceiling(
        root,
        args,
        workload,
        stage4,
        reader,
    )
    result = load_or_create_result(
        root,
        args,
        workload,
        stage4,
        ceiling_result,
        reader,
    )
    programs = ceiling.load_programs(root, stage2, config.__dict__)
    started = time.perf_counter()
    for gpu_count in args.gpu_counts:
        for method in METHODS:
            run_point(
                root,
                args,
                result,
                workload,
                stage4,
                config,
                programs,
                reader,
                method,
                gpu_count,
            )
    result["comparisons"] = build_comparisons(result, stage4)
    expected = len(METHODS) * len(args.gpu_counts)
    result["status"] = (
        "reclaim_candidate_full_complete"
        if args.record_scope == "full" and len(result["points"]) == expected
        else (
            "reclaim_candidate_selection_complete"
            if len(result["points"]) == expected
            else "in_progress"
        )
    )
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
