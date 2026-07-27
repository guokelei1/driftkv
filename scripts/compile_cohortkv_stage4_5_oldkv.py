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

from hstu_kvcache.migration.cohort_jagged import JaggedMigratedKVBatch
from hstu_kvcache.migration.stage45_oldkv import (
    DIRECT_OLDKV_PROGRAM_PROTOCOL,
    DirectOldKVFusedOperator,
    DirectOldKVProgram,
    compile_direct_oldkv_program,
    direct_oldkv_program_set_sha256,
    load_direct_oldkv_program,
    write_direct_oldkv_program,
)
from hstu_kvcache.streaming import load_checkpoint_model

PROTOCOL = "cohortkv_single_config_stage4_5_oldkv_compiler_v1"
DEFAULT_RUNTIME_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0/"
    "single_config_v1/stage4_5_oldkv_runtime"
)
DEFAULT_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_compiler_seed0.json"
)
SOURCE_INDICES = {"theta0": 0, "theta4": 4, "theta10": 10}
TRANSPORT_ROLES = ("program_selection", "certificate")
ATOL = 0.02
RTOL = 0.02
LAUNCH_CANDIDATES = (
    {
        "block_m": 32,
        "block_n": 128,
        "block_k": 64,
        "num_warps": 8,
        "num_stages": 3,
    },
    {
        "block_m": 64,
        "block_n": 128,
        "block_k": 64,
        "num_warps": 8,
        "num_stages": 3,
    },
    {
        "block_m": 32,
        "block_n": 64,
        "block_k": 32,
        "num_warps": 4,
        "num_stages": 3,
    },
    {
        "block_m": 64,
        "block_n": 64,
        "block_k": 32,
        "num_warps": 4,
        "num_stages": 3,
    },
)


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
    parser.add_argument("--runtime-dir", default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def implementation_snapshot(root: Path) -> dict[str, object]:
    paths = (
        Path("src/hstu_kvcache/migration/stage45_oldkv.py"),
        Path("src/hstu_kvcache/migration/stage45_reclaim.py"),
        Path("scripts/compile_cohortkv_stage4_5_oldkv.py"),
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


def program_path(
    root: Path,
    runtime_dir: str,
    source_version: str,
) -> Path:
    return root / runtime_dir / (
        f"{source_version}_to_theta11_direct_oldkv_fp16.pt"
    )


def compile_programs(
    root: Path,
    args: argparse.Namespace,
    blueprint: dict,
    stage2: dict,
    config,
    compiled_programs: dict[str, object],
) -> tuple[
    dict[str, DirectOldKVProgram],
    list[dict[str, object]],
]:
    checkpoint_hashes = {
        value["version"]: value["sha256"]
        for value in blueprint["frozen_inputs"]["checkpoints"]
    }
    descriptors = []
    programs = {}
    for source_version, index in SOURCE_INDICES.items():
        model = load_checkpoint_model(
            config,
            str(root / args.checkpoint_dir),
            index,
            args.device,
        ).eval()
        program, metrics = compile_direct_oldkv_program(
            model,
            compiled_programs[source_version],
        )
        frozen_parent = next(
            value
            for value in blueprint["frozen_inputs"]["verified_programs"]
            if value["source_version"] == source_version
        )
        deployed_parent = next(
            value
            for value in stage2["pairs"]
            if value["source_version"] == source_version
        )
        descriptor = write_direct_oldkv_program(
            program,
            program_path(
                root,
                args.runtime_dir,
                source_version,
            ),
            {
                "labels_used": False,
                "derivation": (
                    "minimum-norm right inverse of the stacked source "
                    "K/V projection followed by the frozen FP16 affine"
                ),
                "source_representation": "existing_old_kv_fp16",
                "source_checkpoint_sha256": checkpoint_hashes[
                    source_version
                ],
                "parent_runtime_program_sha256": deployed_parent[
                    "runtime_program"
                ]["sha256"],
                "parent_verified_plan_sha256": frozen_parent[
                    "verified_plan"
                ]["sha256"],
                "workload_manifest_sha256": blueprint["frozen_inputs"][
                    "workload_manifest"
                ]["file_sha256"],
            },
            metrics,
        )
        loaded, loaded_descriptor = load_direct_oldkv_program(
            descriptor["path"],
            expected_sha256=descriptor["sha256"],
            expected_source_version=source_version,
            expected_target_version="theta11",
            expected_num_layers=config.num_layers,
            expected_kv_width=config.num_heads * config.head_dim,
        )
        descriptor["load_validation"] = {
            "passed": True,
            "provenance": loaded_descriptor["provenance"],
        }
        programs[source_version] = loaded
        descriptors.append(descriptor)
        del model, program, loaded
        gc.collect()
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()
    return programs, descriptors


def output_batch(
    source: JaggedMigratedKVBatch,
    target_version: str,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=source.record_ids,
        migration_anchor_version=source.migration_anchor_version,
        served_kv_target=target_version,
        k=torch.empty_like(source.k),
        v=torch.empty_like(source.v),
        lengths=source.lengths.clone(),
        offsets=source.offsets.clone(),
    )


def load_record_sources(
    reader,
    record,
    device: torch.device,
) -> tuple[torch.Tensor, JaggedMigratedKVBatch]:
    normalized, _, _ = reader._read_shard(
        record,
        "normalized_capsule_fp16",
    )
    old, _, _ = reader._read_shard(record, "old_kv_fp16")
    length = record.prefix_tokens
    lengths = torch.tensor([length], dtype=torch.long, device=device)
    offsets = torch.tensor([0, length], dtype=torch.long, device=device)
    source = JaggedMigratedKVBatch(
        record_ids=(record.record_id,),
        migration_anchor_version=record.source_version,
        served_kv_target=record.source_version,
        k=old["k"].to(device),
        v=old["v"].to(device),
        lengths=lengths,
        offsets=offsets,
    )
    return normalized["normed"].to(device), source


def tune_launch(
    reader,
    programs: dict[str, DirectOldKVProgram],
    device: torch.device,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    record = max(
        (
            value
            for value in reader.manifest.records
            if value.evaluation_role == "program_selection"
        ),
        key=lambda value: (value.prefix_tokens, -value.record_id),
    )
    _, source = load_record_sources(reader, record, device)
    candidates = []
    for launch in LAUNCH_CANDIDATES:
        operator = DirectOldKVFusedOperator(**launch)
        program = operator.prepare_program(
            programs[record.source_version],
            device,
        )
        destination = output_batch(source, program.target_version)
        for _ in range(2):
            operator.execute_into(program, source, destination)
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operator.execute_into(program, source, destination)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        joined = torch.cat((source.k, source.v), dim=-1)
        expected = torch.baddbmm(
            program.biases[:, None, :].expand(
                program.num_layers,
                source.token_count,
                2 * program.kv_width,
            ),
            joined,
            program.weights,
        )
        actual = torch.cat((destination.k, destination.v), dim=-1)
        maximum = float((actual.float() - expected.float()).abs().max())
        allclose = bool(
            torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)
        )
        candidates.append(
            {
                "launch": dict(launch),
                "record_id": record.record_id,
                "prefix_tokens": record.prefix_tokens,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "range_ms": max(samples) - min(samples),
                "allclose": allclose,
                "max_abs_error": maximum,
            }
        )
    qualified = [value for value in candidates if value["allclose"]]
    if len(qualified) != len(candidates):
        raise RuntimeError("direct old-K/V launch candidate is incorrect")
    winner = min(
        qualified,
        key=lambda value: (
            value["median_ms"],
            value["range_ms"],
            tuple(value["launch"].values()),
        ),
    )
    return dict(winner["launch"]), candidates


def new_transport_stats() -> dict[str, object]:
    return {
        "records": 0,
        "prefix_tokens": 0,
        "valid_elements": 0,
        "mismatched_elements": 0,
        "max_abs_error": 0.0,
        "sum_abs_error": 0.0,
        "finite": True,
    }


def update_transport(
    stats: dict[str, object],
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    for start in range(0, actual.shape[1], 256):
        actual_chunk = actual[:, start : start + 256]
        expected_chunk = expected[:, start : start + 256]
        delta = (actual_chunk.float() - expected_chunk.float()).abs()
        stats["finite"] = bool(stats["finite"]) and bool(
            torch.isfinite(actual_chunk).all()
            and torch.isfinite(expected_chunk).all()
        )
        stats["mismatched_elements"] = int(
            stats["mismatched_elements"]
        ) + int(
            torch.count_nonzero(
                ~torch.isclose(
                    actual_chunk,
                    expected_chunk,
                    atol=ATOL,
                    rtol=RTOL,
                )
            )
        )
        stats["valid_elements"] = int(stats["valid_elements"]) + int(
            actual_chunk.numel()
        )
        stats["sum_abs_error"] = float(stats["sum_abs_error"]) + float(
            delta.sum()
        )
        if delta.numel():
            stats["max_abs_error"] = max(
                float(stats["max_abs_error"]),
                float(delta.max()),
            )


def validate_transport(
    reader,
    compiled_programs: dict[str, object],
    direct_programs: dict[str, DirectOldKVProgram],
    launch: dict[str, int],
    device: torch.device,
) -> dict[str, object]:
    operator = DirectOldKVFusedOperator(**launch)
    compiled = {
        source: value.to(device, dtype=torch.float16)
        for source, value in compiled_programs.items()
    }
    direct = {
        source: operator.prepare_program(value, device)
        for source, value in direct_programs.items()
    }
    roles = {
        role: {
            "aggregate": new_transport_stats(),
            "by_source": {
                source: new_transport_stats() for source in SOURCE_INDICES
            },
        }
        for role in TRANSPORT_ROLES
    }
    started = time.perf_counter()
    selected = [
        value
        for value in reader.manifest.records
        if value.evaluation_role in TRANSPORT_ROLES
    ]
    for position, record in enumerate(selected, 1):
        normalized, source = load_record_sources(
            reader,
            record,
            device,
        )
        direct_program = direct[record.source_version]
        destination = output_batch(
            source,
            direct_program.target_version,
        )
        operator.execute_into(direct_program, source, destination)
        parent = compiled[record.source_version]
        expected = torch.baddbmm(
            parent.adapter.biases[:, None, :].expand(
                parent.num_layers,
                record.prefix_tokens,
                2 * parent.kv_width,
            ),
            normalized,
            parent.adapter.weights,
        )
        actual = torch.cat(
            (destination.k, destination.v),
            dim=-1,
        )
        role_stats = roles[record.evaluation_role]
        for stats in (
            role_stats["aggregate"],
            role_stats["by_source"][record.source_version],
        ):
            update_transport(stats, actual, expected)
            stats["records"] = int(stats["records"]) + 1
            stats["prefix_tokens"] = int(stats["prefix_tokens"]) + (
                record.prefix_tokens
            )
        if position % 20 == 0:
            print(
                json.dumps(
                    {
                        "status": "transport_progress",
                        "complete": position,
                        "total": len(selected),
                    }
                ),
                flush=True,
            )
    for role in roles.values():
        for stats in (
            role["aggregate"],
            *role["by_source"].values(),
        ):
            elements = int(stats["valid_elements"])
            stats["mean_abs_error"] = (
                float(stats.pop("sum_abs_error")) / elements
                if elements
                else 0.0
            )
            stats["allclose"] = (
                bool(stats["finite"])
                and int(stats["mismatched_elements"]) == 0
            )
    return {
        "protocol": DIRECT_OLDKV_RUNTIME_PROTOCOL,
        "roles": roles,
        "elapsed_seconds": time.perf_counter() - started,
        "atol": ATOL,
        "rtol": RTOL,
        "reference": (
            "frozen deployed FP16 normalized-capsule affine output"
        ),
        "candidate": "deployed FP16 direct old-K/V fused output",
        "passed": all(
            value["aggregate"]["allclose"]
            for value in roles.values()
        ),
    }


DIRECT_OLDKV_RUNTIME_PROTOCOL = (
    "cohortkv_stage4_5_direct_oldkv_transport_certificate_v1"
)


def main() -> None:
    args = parse_args()
    args.measured_repeats = 1
    args.gpu_counts = [1]
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    (
        blueprint,
        workload,
        stage2,
        _,
        stage4,
        config,
        reader,
    ) = ceiling.validate_inputs(root, args)
    compiled_programs = ceiling.load_programs(
        root,
        stage2,
        blueprint["data_and_model"]["model"],
    )
    direct_programs, descriptors = compile_programs(
        root,
        args,
        blueprint,
        stage2,
        config,
        compiled_programs,
    )
    device = torch.device(args.device)
    launch, launch_screen = tune_launch(
        reader,
        direct_programs,
        device,
    )
    transport = validate_transport(
        reader,
        compiled_programs,
        direct_programs,
        launch,
        device,
    )
    if not transport["passed"]:
        raise RuntimeError("direct old-K/V transport certificate failed")
    result = {
        "protocol": PROTOCOL,
        "parent_protocol": DIRECT_OLDKV_PROGRAM_PROTOCOL,
        "status": "oldkv_program_transport_frozen",
        "study_stage": "stage4_5_b_direct_oldkv_seed0",
        "seed": 0,
        "labels_used": False,
        "inputs": {
            "workload_content_sha256": workload["content_sha256"],
            "stage2_summary_sha256": ceiling.sha256_file(
                root / args.stage2_summary
            ),
            "stage4_summary_sha256": ceiling.sha256_file(
                root / args.stage4_summary
            ),
            "source_manifest_sha256": reader.manifest_file_sha256,
        },
        "representation": {
            "input": "existing_old_kv_fp16",
            "additional_per_record_source_state_bytes": 0,
            "derivation": (
                "stack source K/V projections, apply their minimum-norm "
                "right inverse, then compose the frozen compiled affine"
            ),
            "program_set_sha256": direct_oldkv_program_set_sha256(
                descriptors
            ),
            "program_bytes": sum(
                int(value["bytes"]) for value in descriptors
            ),
            "programs": descriptors,
        },
        "operator_selection": {
            "role": "program_selection",
            "winner": launch,
            "candidates": launch_screen,
        },
        "transport_certificate": transport,
        "source_policy": {
            "normal_source": "existing serving old K/V in HBM",
            "extra_normx_state": "not retained",
            "target": "FP16 HBM K/V extents",
            "lifecycle": "retire each old extent after replacement stage",
            "cold_fallback": "exact",
        },
        "implementation": implementation_snapshot(root),
        "last_invocation_seconds": time.perf_counter() - started,
    }
    ceiling.write_json_atomic(root / args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "program_bytes": result["representation"][
                    "program_bytes"
                ],
                "launch": launch,
                "selection_max_abs_error": transport["roles"][
                    "program_selection"
                ]["aggregate"]["max_abs_error"],
                "certificate_max_abs_error": transport["roles"][
                    "certificate"
                ]["aggregate"]["max_abs_error"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
