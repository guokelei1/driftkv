from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import benchmark_cohortkv_stage4_5_resident_ceiling as ceiling
import compile_cohortkv_stage4_5_oldkv as compiler_tools
import torch

from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    load_direct_oldkv_program,
)

PROTOCOL = "cohortkv_single_config_stage4_5_oldkv_full_transport_v1"
DEFAULT_COMPILER = compiler_tools.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_full_transport_seed0.json"
)
ATOL = 0.02
RTOL = 0.02


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
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def implementation_snapshot(root: Path) -> dict[str, object]:
    paths = (
        Path("src/hstu_kvcache/migration/stage45_oldkv.py"),
        Path("scripts/validate_cohortkv_stage4_5_oldkv_full_transport.py"),
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


def new_stats() -> dict[str, object]:
    return {
        "records": 0,
        "prefix_tokens": 0,
        "valid_elements": 0,
        "mismatched_elements": 0,
        "max_abs_error": 0.0,
        "sum_abs_error": 0.0,
        "finite": True,
    }


def update(
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


def finish(stats: dict[str, object]) -> dict[str, object]:
    elements = int(stats["valid_elements"])
    stats["mean_abs_error"] = (
        float(stats.pop("sum_abs_error")) / elements if elements else 0.0
    )
    stats["allclose"] = (
        bool(stats["finite"])
        and int(stats["mismatched_elements"]) == 0
    )
    return stats


def main() -> None:
    args = parse_args()
    args.measured_repeats = 1
    args.gpu_counts = [1]
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    (
        _,
        workload,
        stage2,
        _,
        _,
        config,
        reader,
    ) = ceiling.validate_inputs(root, args)
    compiler = json.loads((root / args.compiler_result).read_text())
    if (
        compiler.get("status") != "oldkv_program_transport_frozen"
        or compiler.get("labels_used") is not False
        or compiler["inputs"]["source_manifest_sha256"]
        != reader.manifest_file_sha256
    ):
        raise ValueError("full old-K/V transport inputs differ")
    device = torch.device(args.device)
    parent = ceiling.load_programs(
        root,
        stage2,
        config.__dict__,
    )
    parent = {
        source: value.to(device, dtype=torch.float16)
        for source, value in parent.items()
    }
    operator = DirectOldKVFusedOperator(
        **compiler["operator_selection"]["winner"]
    )
    direct = {}
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
        direct[source] = operator.prepare_program(program, device)
    aggregate = new_stats()
    by_source = {
        source: new_stats() for source in ceiling.SOURCE_VERSIONS
    }
    by_role = {
        role: new_stats()
        for role in (
            "fit",
            "program_selection",
            "certificate",
            "final_test",
        )
    }
    records = reader.manifest.records
    for position, record in enumerate(records, 1):
        normalized, source = compiler_tools.load_record_sources(
            reader,
            record,
            device,
        )
        direct_program = direct[record.source_version]
        destination = compiler_tools.output_batch(
            source,
            direct_program.target_version,
        )
        operator.execute_into(direct_program, source, destination)
        parent_program = parent[record.source_version]
        expected = torch.baddbmm(
            parent_program.adapter.biases[:, None, :].expand(
                parent_program.num_layers,
                record.prefix_tokens,
                2 * parent_program.kv_width,
            ),
            normalized,
            parent_program.adapter.weights,
        )
        actual = torch.cat((destination.k, destination.v), dim=-1)
        for stats in (
            aggregate,
            by_source[record.source_version],
            by_role[record.evaluation_role],
        ):
            update(stats, actual, expected)
            stats["records"] = int(stats["records"]) + 1
            stats["prefix_tokens"] = int(stats["prefix_tokens"]) + (
                record.prefix_tokens
            )
        if position % 50 == 0 or position == len(records):
            print(
                json.dumps(
                    {
                        "status": "full_transport_progress",
                        "complete": position,
                        "total": len(records),
                    }
                ),
                flush=True,
            )
    aggregate = finish(aggregate)
    by_source = {
        key: finish(value) for key, value in by_source.items()
    }
    by_role = {key: finish(value) for key, value in by_role.items()}
    expected_elements = (
        workload["summary"]["logical_target_kv_bytes_fp16"]
        // torch.tensor([], dtype=torch.float16).element_size()
    )
    if (
        not aggregate["allclose"]
        or aggregate["records"] != workload["summary"]["records"]
        or aggregate["prefix_tokens"]
        != workload["summary"]["prefix_tokens"]
        or aggregate["valid_elements"] != expected_elements
        or any(not value["allclose"] for value in by_source.values())
        or any(not value["allclose"] for value in by_role.values())
    ):
        raise RuntimeError("full direct old-K/V transport failed")
    result = {
        "protocol": PROTOCOL,
        "status": "oldkv_full_transport_frozen",
        "study_stage": "stage4_5_c_direct_oldkv_seed0",
        "seed": 0,
        "labels_used": False,
        "measurement_boundary": (
            "all real serialized FP16 old K/V through deployed direct fused "
            "operator against the frozen normalized-capsule FP16 output"
        ),
        "inputs": {
            "compiler_result": {
                "path": args.compiler_result,
                "sha256": ceiling.sha256_file(
                    root / args.compiler_result
                ),
            },
            "source_manifest_sha256": reader.manifest_file_sha256,
            "workload_content_sha256": workload["content_sha256"],
        },
        "atol": ATOL,
        "rtol": RTOL,
        "aggregate": aggregate,
        "by_source": by_source,
        "by_role": by_role,
        "implementation": implementation_snapshot(root),
        "last_invocation_seconds": time.perf_counter() - started,
    }
    ceiling.write_json_atomic(root / args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "records": aggregate["records"],
                "valid_elements": aggregate["valid_elements"],
                "max_abs_error": aggregate["max_abs_error"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
