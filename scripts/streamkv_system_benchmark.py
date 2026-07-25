from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from hstu_kvcache.migration import (
    CohortStreamingExecutor,
    CompiledCacheAdapter,
    MigrationCapsuleBatch,
    MigrationProgram,
    MultiGPUCohortExecutor,
    PackedMigrationOperator,
    ReferenceMigrationOperator,
    benchmark_cuda_operator,
    build_contiguous_cohort_plan,
    build_length_bucketed_cohort_plan,
    profile_packed_operator_stages,
    profile_reference_operator_stages,
)
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="streamkv_system_prototype_v1")
    parser.add_argument(
        "--algorithm-summary",
        default="results/motivation_scale/cohort_tiered_migration_v1_summary.json",
    )
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--operator-batch-sizes", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--operator-profile-batch-size", type=int, default=32)
    parser.add_argument("--system-cohort-sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--system-batch-size", type=int, default=32)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--stage-profile-repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        default="results/system/streamkv_system_prototype_v1.json",
    )
    return parser.parse_args()


def load_algorithm_evidence(path: str) -> dict:
    source = json.loads(Path(path).read_text())
    validation = source["cross_cell_primary_validation"]
    return {
        "source": path,
        "protocol": source["protocol"],
        "statistical_unit": source["statistical_unit"],
        "method_selection": source["method_selection"],
        "primary_validation": {
            "num_validation_seeds": validation["num_validation_seeds"],
            "selected_family_counts": validation["selected_family_counts"],
            "cost_ratio_to_full": validation["cost_ratio_to_full"],
            "cache_fidelity_recovery": validation["cache_fidelity_recovery"],
            "test_fidelity_target_met": validation["test_fidelity_target_met"],
            "cells_passing_all_frozen_mean_gates": validation[
                "cells_passing_all_frozen_mean_gates"
            ],
            "gate_by_cell": validation["gate_by_cell"],
            "full_endpoint_tracking": validation["full_endpoint_tracking"],
        },
    }


def make_program(args: argparse.Namespace) -> MigrationProgram:
    generator = torch.Generator().manual_seed(args.seed)
    weights = torch.randn(
        args.num_layers,
        args.hidden_size,
        2 * args.hidden_size,
        generator=generator,
    ) / args.hidden_size**0.5
    biases = torch.randn(
        args.num_layers,
        2 * args.hidden_size,
        generator=generator,
    )
    return MigrationProgram(
        source_version="theta-old",
        target_version="theta-current",
        adapter=CompiledCacheAdapter(
            weights=weights,
            biases=biases,
            source_rank=8,
            ridge=1e-3,
        ),
    )


def make_capsule(
    args: argparse.Namespace,
    records: int,
    seed_offset: int,
) -> MigrationCapsuleBatch:
    generator = torch.Generator().manual_seed(args.seed + seed_offset)
    normed = torch.randn(
        args.num_layers,
        records,
        args.seq_len,
        args.hidden_size,
        generator=generator,
        dtype=torch.float16,
    )
    lengths = torch.randint(
        max(1, args.seq_len // 2),
        args.seq_len + 1,
        (records,),
        generator=generator,
    )
    return MigrationCapsuleBatch(
        record_ids=tuple(range(records)),
        migration_anchor_version="theta-old",
        normed=normed,
        lengths=lengths,
    )


def latency_record(samples) -> dict:
    return {
        "values_ms": list(samples.values_ms),
        "median_ms": samples.median_ms,
        "mean_ms": samples.mean_ms,
    }


def cache_difference(actual, expected) -> dict:
    delta_k = (actual.cache.k - expected.cache.k).float()
    delta_v = (actual.cache.v - expected.cache.v).float()
    reference_k = expected.cache.k.float()
    reference_v = expected.cache.v.float()
    return {
        "max_abs": max(
            float(delta_k.abs().max()),
            float(delta_v.abs().max()),
        ),
        "rms": float(
            torch.cat((delta_k.flatten(), delta_v.flatten()))
            .pow(2)
            .mean()
            .sqrt()
        ),
        "fro_relative": float(
            torch.sqrt(delta_k.pow(2).sum() + delta_v.pow(2).sum())
            / (
                torch.sqrt(reference_k.pow(2).sum() + reference_v.pow(2).sum())
                + 1e-12
            )
        ),
    }


def estimated_materialized_bytes(
    capsule: MigrationCapsuleBatch,
    execution_dtype: torch.dtype,
) -> int:
    output_elements = (
        capsule.num_layers
        * capsule.batch_size
        * capsule.seq_len
        * 2
        * capsule.hidden_size
    )
    input_cast_bytes = (
        capsule.normed.numel() * torch.empty((), dtype=execution_dtype).element_size()
        if capsule.normed.dtype != execution_dtype
        else 0
    )
    projected_bytes = (
        output_elements * torch.empty((), dtype=execution_dtype).element_size()
    )
    output_cast_bytes = (
        output_elements * capsule.normed.element_size()
        if capsule.normed.dtype != execution_dtype
        else 0
    )
    return input_cast_bytes + projected_bytes + output_cast_bytes


def run_operator_experiment(
    args: argparse.Namespace,
    program: MigrationProgram,
    device: torch.device,
) -> dict:
    operators = (
        ReferenceMigrationOperator(),
        PackedMigrationOperator(torch.float32),
        PackedMigrationOperator(torch.float16),
    )
    points = []
    for index, batch_size in enumerate(args.operator_batch_sizes):
        capsule = make_capsule(args, batch_size, 100 + index).pin_memory()
        capsule = capsule.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        outputs = {}
        latencies = {}
        for operator in operators:
            output, latency = benchmark_cuda_operator(
                operator,
                program,
                capsule,
                args.warmup_repeats,
                args.timing_repeats,
            )
            outputs[operator.name] = output
            latencies[operator.name] = latency
        reference = outputs["reference_fp32"]
        records = {}
        for operator in operators:
            latency = latencies[operator.name]
            execution_dtype = (
                torch.float32
                if operator.name != "packed_float16"
                else torch.float16
            )
            records[operator.name] = {
                "latency": latency_record(latency),
                "records_per_second": batch_size * 1000.0 / latency.median_ms,
                "estimated_materialized_bytes": estimated_materialized_bytes(
                    capsule,
                    execution_dtype,
                ),
                "difference_from_reference": (
                    {
                        "max_abs": 0.0,
                        "rms": 0.0,
                        "fro_relative": 0.0,
                    }
                    if operator.name == "reference_fp32"
                    else cache_difference(outputs[operator.name], reference)
                ),
            }
        reference_latency = latencies["reference_fp32"].median_ms
        for record in records.values():
            record["speedup_over_reference"] = (
                reference_latency / record["latency"]["median_ms"]
            )
        points.append(
            {
                "batch_size": batch_size,
                "seq_len": args.seq_len,
                "operators": records,
            }
        )

    profile_capsule = make_capsule(
        args,
        args.operator_profile_batch_size,
        500,
    ).pin_memory()
    profile_capsule = profile_capsule.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    profiles = (
        profile_reference_operator_stages(
            program,
            profile_capsule,
            args.stage_profile_repeats,
        ),
        profile_packed_operator_stages(
            program,
            profile_capsule,
            torch.float16,
            args.stage_profile_repeats,
        ),
    )
    return {
        "input": "synthetic contiguous old-Norm capsule; resident GPU operator timing",
        "program_weights": "synthetic values with capacity-v2 large shape",
        "points": points,
        "stage_profile": {
            profile.operator: {
                "total": latency_record(profile.total),
                "stages": {
                    name: latency_record(samples)
                    for name, samples in profile.stages.items()
                },
            }
            for profile in profiles
        },
    }


def result_padding_max_abs(result) -> float:
    if not result.batches:
        return 0.0
    batch = result.batches[0]
    positions = torch.arange(batch.cache.seq_len).unsqueeze(0)
    invalid = positions >= batch.lengths.unsqueeze(1)
    if not bool(torch.any(invalid)):
        return 0.0
    mask = invalid.unsqueeze(0).unsqueeze(-1)
    return max(
        float(batch.cache.k.masked_select(mask).abs().max()),
        float(batch.cache.v.masked_select(mask).abs().max()),
    )


def collect_runtime_samples(executor, batches, warmup_repeats: int, timing_repeats: int):
    for _ in range(warmup_repeats):
        executor.run(batches)
    elapsed = []
    records_per_second = []
    tokens_per_second = []
    gib_per_second = []
    report = None
    for index in range(timing_repeats):
        current = executor.run(batches)
        elapsed.append(current.metrics.elapsed_seconds)
        records_per_second.append(current.metrics.records_per_second)
        tokens_per_second.append(current.metrics.tokens_per_second)
        gib_per_second.append(current.metrics.gib_per_second)
        if index == timing_repeats - 1:
            report = current
        else:
            del current
    if report is None:
        raise RuntimeError("runtime benchmark produced no report")
    record_ids = tuple(
        record_id
        for batch in report.batches
        for record_id in batch.record_ids
    )
    record = {
        "elapsed_seconds": elapsed,
        "median_elapsed_seconds": statistics.median(elapsed),
        "median_records_per_second": statistics.median(records_per_second),
        "median_tokens_per_second": statistics.median(tokens_per_second),
        "median_gib_per_second": statistics.median(gib_per_second),
        "batch_count": report.metrics.batch_count,
        "record_count": report.metrics.record_count,
        "token_count": report.metrics.token_count,
        "input_bytes": report.metrics.input_bytes,
        "output_bytes": report.metrics.output_bytes,
        "record_coverage_valid": (
            len(record_ids) == len(set(record_ids))
            and set(record_ids) == set(range(len(record_ids)))
        ),
        "physical_order_is_logical": record_ids == tuple(range(len(record_ids))),
        "max_padding_abs": result_padding_max_abs(report),
    }
    if hasattr(report.metrics, "devices"):
        record["load_imbalance"] = report.metrics.load_imbalance
        record["program_replica_bytes"] = report.metrics.program_replica_bytes
        record["devices"] = [
            {
                "device": value.device,
                "assigned_work_bytes": value.assigned_work_bytes,
                "batch_count": value.execution.batch_count,
                "record_count": value.execution.record_count,
                "elapsed_seconds": value.execution.elapsed_seconds,
                "records_per_second": value.execution.records_per_second,
            }
            for value in report.metrics.devices
        ]
    return record


def run_system_experiment(
    args: argparse.Namespace,
    program: MigrationProgram,
    devices: tuple[torch.device, ...],
) -> dict:
    points = []
    for index, cohort_size in enumerate(args.system_cohort_sizes):
        capsule = make_capsule(args, cohort_size, 1000 + index)
        contiguous_plan = build_contiguous_cohort_plan(
            capsule,
            args.system_batch_size,
        )
        bucketed_plan = build_length_bucketed_cohort_plan(
            capsule,
            max_records=args.system_batch_size,
            bucket_width=16,
        )
        contiguous_batches = tuple(
            batch.pin_memory()
            for batch in contiguous_plan.batches
        )
        bucketed_batches = tuple(
            batch.pin_memory()
            for batch in bucketed_plan.batches
        )
        configurations = {}

        reference = CohortStreamingExecutor(
            program,
            device=devices[0],
            max_inflight_batches=1,
            operator=ReferenceMigrationOperator(),
        )
        configurations["reference_fp32_sync_1gpu"] = collect_runtime_samples(
            reference,
            contiguous_batches,
            args.warmup_repeats,
            args.timing_repeats,
        )

        for inflight in (1, 3):
            executor = CohortStreamingExecutor(
                program,
                device=devices[0],
                max_inflight_batches=inflight,
                operator=PackedMigrationOperator(torch.float16),
            )
            name = f"packed_fp16_inflight{inflight}_1gpu"
            configurations[name] = collect_runtime_samples(
                executor,
                contiguous_batches,
                args.warmup_repeats,
                args.timing_repeats,
            )

        bucketed_executor = CohortStreamingExecutor(
            program,
            device=devices[0],
            max_inflight_batches=3,
            operator=PackedMigrationOperator(torch.float16),
        )
        configurations["packed_fp16_inflight3_bucketed_1gpu"] = collect_runtime_samples(
            bucketed_executor,
            bucketed_batches,
            args.warmup_repeats,
            args.timing_repeats,
        )

        for device_count in (2, 4):
            if len(devices) < device_count:
                continue
            with MultiGPUCohortExecutor(
                program,
                devices=devices[:device_count],
                max_inflight_batches=3,
                operator=PackedMigrationOperator(torch.float16),
            ) as executor:
                name = f"packed_fp16_inflight3_bucketed_{device_count}gpu"
                configurations[name] = collect_runtime_samples(
                    executor,
                    bucketed_batches,
                    args.warmup_repeats,
                    args.timing_repeats,
                )

        baseline = configurations["reference_fp32_sync_1gpu"][
            "median_elapsed_seconds"
        ]
        packed_unbucketed = configurations["packed_fp16_inflight3_1gpu"][
            "median_elapsed_seconds"
        ]
        packed_bucketed = configurations["packed_fp16_inflight3_bucketed_1gpu"][
            "median_elapsed_seconds"
        ]
        for record in configurations.values():
            record["speedup_over_reference_sync_1gpu"] = (
                baseline / record["median_elapsed_seconds"]
            )
            record["speedup_over_packed_unbucketed_1gpu"] = (
                packed_unbucketed / record["median_elapsed_seconds"]
            )
            record["speedup_over_packed_bucketed_1gpu"] = (
                packed_bucketed / record["median_elapsed_seconds"]
            )
        points.append(
            {
                "cohort_size": cohort_size,
                "batch_size": args.system_batch_size,
                "plans": {
                    "contiguous": {
                        "batch_count": len(contiguous_plan.batches),
                        "payload_bytes": contiguous_plan.payload_nbytes,
                        "padded_tokens": contiguous_plan.padded_tokens,
                    },
                    "length_bucketed_16": {
                        "batch_count": len(bucketed_plan.batches),
                        "payload_bytes": bucketed_plan.payload_nbytes,
                        "padded_tokens": bucketed_plan.padded_tokens,
                        "payload_ratio_to_contiguous": (
                            bucketed_plan.payload_nbytes
                            / contiguous_plan.payload_nbytes
                        ),
                    },
                },
                "configurations": configurations,
            }
        )
    return {
        "input": "pre-pinned host capsule to host K/V; program distribution and input generation excluded",
        "partition": "greedy extent assignment by estimated input plus output bytes",
        "points": points,
    }


def validate_args(args: argparse.Namespace) -> tuple[torch.device, ...]:
    if args.num_layers < 1 or args.hidden_size < 1 or args.seq_len < 1:
        raise ValueError("model dimensions must be positive")
    if args.system_batch_size < 1:
        raise ValueError("system batch size must be positive")
    if any(value < 1 for value in args.operator_batch_sizes):
        raise ValueError("operator batch sizes must be positive")
    if any(value < 1 for value in args.system_cohort_sizes):
        raise ValueError("system cohort sizes must be positive")
    if args.warmup_repeats < 0 or args.timing_repeats < 1:
        raise ValueError("benchmark repeat counts are invalid")
    if args.stage_profile_repeats < 1:
        raise ValueError("stage profile repeats must be positive")
    devices = tuple(torch.device(value) for value in args.devices)
    if not devices or any(device.type != "cuda" for device in devices):
        raise ValueError("system benchmark requires CUDA devices")
    if len(set(devices)) != len(devices):
        raise ValueError("benchmark devices must be unique")
    if len(devices) > torch.cuda.device_count():
        raise ValueError("requested more CUDA devices than are available")
    return devices


def main() -> None:
    args = parse_args()
    devices = validate_args(args)
    torch.manual_seed(args.seed)
    algorithm = load_algorithm_evidence(args.algorithm_summary)
    program = make_program(args)
    operator = run_operator_experiment(args, program, devices[0])
    system = run_system_experiment(args, program, devices)
    result = {
        "protocol": args.protocol,
        "evidence_status": (
            "algorithm section reuses frozen replicated evidence; operator and system "
            "sections are preliminary synthetic prototype results"
        ),
        "hardware": {
            "devices": [
                {
                    "device": str(device),
                    "name": torch.cuda.get_device_name(device),
                }
                for device in devices
            ],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "peer_access": {
                str(source): {
                    str(target): (
                        source == target
                        or torch.cuda.can_device_access_peer(source, target)
                    )
                    for target in range(len(devices))
                }
                for source in range(len(devices))
            },
        },
        "shape": {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "seq_len": args.seq_len,
            "capsule_dtype": "float16",
            "program_bytes_fp32": program.nbytes,
        },
        "repeats": {
            "warmup": args.warmup_repeats,
            "timing": args.timing_repeats,
            "stage_profile": args.stage_profile_repeats,
        },
        "algorithm": algorithm,
        "operator": operator,
        "system": system,
    }
    save_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
