from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from evaluate_kuairand_long_context_sync_design import sha256
from layerwise_validity import timed_call
from motivation_validity import eval_batches, move_batch, seed_everything
from streamkv_system_benchmark import cache_difference, collect_runtime_samples

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    CompiledCacheAdapter,
    MigrationCapsuleBatch,
    MigrationProgram,
    MultiGPUCohortExecutor,
    PackedMigrationOperator,
    ReferenceMigrationOperator,
    benchmark_cuda_operator,
    capture_layerwise_state,
)
from hstu_kvcache.models import HSTUConfig
from hstu_kvcache.streaming import (
    SYNC_DESIGN_PROTOCOL,
    SYNC_SYSTEM_PROTOCOL,
    load_checkpoint_model,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

DEFAULT_PREPARED = (
    "data/processed/kuairand_long_context_4plus12_exploration_v1.npz"
)
DEFAULT_TRAINING = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
DEFAULT_CHECKPOINTS = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)
DEFAULT_PROGRAM = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0/"
    "sync_programs/theta0_to_theta11_rank32.pt"
)
DEFAULT_OUTPUT = (
    "results/system/"
    "kuairand_long_context_4plus12_progressive_sync_system_seed0.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--program", default=DEFAULT_PROGRAM)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--base-days", type=int, default=4)
    parser.add_argument("--source-version", type=int, default=0)
    parser.add_argument("--target-version", type=int, default=11)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--max-users", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--max-inflight", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-diagnostic-protocol",
        "--allow-diagnostic-device-count",
        dest="allow_diagnostic_protocol",
        action="store_true",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_program(path: str | Path) -> tuple[MigrationProgram, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != SYNC_DESIGN_PROTOCOL:
        raise ValueError("migration program protocol mismatch")
    program = MigrationProgram(
        source_version=payload["source_version"],
        target_version=payload["target_version"],
        adapter=CompiledCacheAdapter(
            weights=payload["weights"],
            biases=payload["biases"],
            source_rank=int(payload["source_rank"]),
            ridge=float(payload["ridge"]),
        ),
    )
    return program, payload


def latency(samples) -> dict:
    return {
        "values_ms": list(samples.values_ms),
        "median_ms": samples.median_ms,
        "mean_ms": samples.mean_ms,
    }


@torch.inference_mode()
def materialize_real_capsules(
    old,
    current,
    samples: list[dict],
    seq_len: int,
    batch_size: int,
    source_version: int,
    device: torch.device,
    timing_repeats: int,
) -> tuple[tuple[MigrationCapsuleBatch, ...], tuple[dict, ...], dict]:
    ordered = sorted(samples, key=lambda sample: len(sample["history"]["item_ids"]))
    capsules = []
    raw_batches = []
    exact_ms = []
    record_id = 0
    logical_tokens = 0
    padded_tokens = 0
    for selected, _, prefix_cpu, _ in eval_batches(ordered, seq_len, batch_size):
        prefix = move_batch(prefix_cpu, device)
        raw_batches.append(
            {
                name: value.pin_memory()
                for name, value in prefix_cpu.items()
            }
        )
        state = capture_layerwise_state(
            old,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            prefix["lengths"],
        )
        _, elapsed = timed_call(
            lambda prefix=prefix: current.compute_kv(
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                lengths=prefix["lengths"],
            ),
            device,
            timing_repeats,
        )
        exact_ms.append(elapsed)
        normed = torch.stack(state.normed_states).to(torch.float16).cpu()
        lengths = state.lengths.cpu()
        ids = tuple(range(record_id, record_id + len(selected)))
        record_id += len(selected)
        capsule = MigrationCapsuleBatch(
            record_ids=ids,
            migration_anchor_version=f"theta{source_version}",
            normed=normed,
            lengths=lengths,
        ).pin_memory()
        capsules.append(capsule)
        logical_tokens += int(lengths.sum().item())
        padded_tokens += capsule.batch_size * capsule.seq_len
        del state, prefix, normed
    return tuple(capsules), tuple(raw_batches), {
        "records": record_id,
        "batches": len(capsules),
        "logical_tokens": logical_tokens,
        "padded_tokens": padded_tokens,
        "padding_fraction": 1.0 - logical_tokens / max(padded_tokens, 1),
        "capsule_bytes": sum(capsule.nbytes for capsule in capsules),
        "exact_resident_compute_ms_per_batch": exact_ms,
        "exact_resident_compute_median_ms_per_batch": statistics.median(exact_ms),
        "exact_resident_compute_ms_per_user": sum(exact_ms) / max(record_id, 1),
    }


@torch.inference_mode()
def benchmark_full_host_boundary(
    current,
    raw_batches: tuple[dict, ...],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    elapsed = []
    input_bytes = sum(
        value.numel() * value.element_size()
        for batch in raw_batches
        for value in batch.values()
    )
    output_bytes = 0
    records = sum(int(batch["lengths"].shape[0]) for batch in raw_batches)
    tokens = sum(int(batch["lengths"].sum().item()) for batch in raw_batches)
    for iteration in range(warmup + repeats):
        started = time.perf_counter()
        current_output_bytes = 0
        for host in raw_batches:
            inputs = {
                name: value.to(device, non_blocking=True)
                for name, value in host.items()
            }
            cache = current.compute_kv(
                inputs["item_ids"],
                inputs["behaviors"],
                inputs["time_deltas"],
                lengths=inputs["lengths"],
            )
            output_k = cache.k.to(torch.float16)
            output_v = cache.v.to(torch.float16)
            host_k = torch.empty(
                output_k.shape,
                dtype=torch.float16,
                device="cpu",
                pin_memory=True,
            )
            host_v = torch.empty_like(host_k, pin_memory=True)
            host_k.copy_(output_k, non_blocking=True)
            host_v.copy_(output_v, non_blocking=True)
            torch.cuda.synchronize(device)
            current_output_bytes += (
                host_k.numel() * host_k.element_size()
                + host_v.numel() * host_v.element_size()
            )
        duration = time.perf_counter() - started
        output_bytes = current_output_bytes
        if iteration >= warmup:
            elapsed.append(duration)
    median = statistics.median(elapsed)
    return {
        "boundary": (
            "pinned raw history through current-model recompute and FP16 "
            "pinned-host K/V publication"
        ),
        "execution": "single-GPU synchronous batch baseline",
        "elapsed_seconds": elapsed,
        "median_elapsed_seconds": median,
        "records": records,
        "logical_tokens": tokens,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "median_records_per_second": records / median,
        "median_tokens_per_second": tokens / median,
        "median_gib_per_second": (
            (input_bytes + output_bytes) / (2**30 * median)
        ),
    }


def resident_operator_benchmark(
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    resident = capsule.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    reference, reference_latency = benchmark_cuda_operator(
        ReferenceMigrationOperator(),
        program,
        resident,
        warmup,
        repeats,
    )
    packed, packed_latency = benchmark_cuda_operator(
        PackedMigrationOperator(torch.float16),
        program,
        resident,
        warmup,
        repeats,
    )
    return {
        "records": capsule.batch_size,
        "sequence_width": capsule.seq_len,
        "logical_tokens": int(capsule.lengths.sum().item()),
        "reference_fp32": latency(reference_latency),
        "packed_fp16": latency(packed_latency),
        "packed_speedup": reference_latency.median_ms / packed_latency.median_ms,
        "packed_difference_from_reference": cache_difference(packed, reference),
    }


def run_runtime_points(
    program: MigrationProgram,
    capsules: tuple[MigrationCapsuleBatch, ...],
    devices: list[str],
    args: argparse.Namespace,
) -> list[dict]:
    points = []
    operator = PackedMigrationOperator(torch.float16)
    for count in range(1, len(devices) + 1):
        selected = devices[:count]
        with MultiGPUCohortExecutor(
            program,
            selected,
            max_inflight_batches=args.max_inflight,
            pin_inputs=False,
            operator=operator,
        ) as executor:
            result = collect_runtime_samples(
                executor,
                capsules,
                args.warmup_repeats,
                args.timing_repeats,
            )
        result["device_count"] = count
        result["devices_requested"] = selected
        points.append(result)
        torch.cuda.empty_cache()
    baseline = points[0]["median_elapsed_seconds"]
    for point in points:
        point["speedup_over_one_gpu"] = (
            baseline / point["median_elapsed_seconds"]
        )
        point["parallel_efficiency"] = (
            point["speedup_over_one_gpu"] / point["device_count"]
        )
    return points


def smoke(args: argparse.Namespace) -> None:
    generator = torch.Generator().manual_seed(0)
    program = MigrationProgram(
        source_version="theta0",
        target_version="theta1",
        adapter=CompiledCacheAdapter(
            weights=torch.randn(2, 8, 16, generator=generator),
            biases=torch.randn(2, 16, generator=generator),
            source_rank=2,
            ridge=1e-3,
        ),
    )
    capsule = MigrationCapsuleBatch(
        record_ids=(0, 1),
        migration_anchor_version="theta0",
        normed=torch.randn(2, 2, 4, 8, generator=generator),
        lengths=torch.tensor([4, 3]),
    )
    device = torch.device(args.devices[0])
    point = resident_operator_benchmark(program, capsule, device, 0, 1)
    if point["packed_difference_from_reference"]["fro_relative"] > 0.01:
        raise RuntimeError("packed operator smoke error exceeds tolerance")
    print(json.dumps({"status": "ok", "device": str(device)}, indent=2))


def validate_args(args: argparse.Namespace) -> None:
    if args.max_users < 1 or args.batch_size < 1:
        raise ValueError("max_users and batch_size must be positive")
    if args.warmup_repeats < 0 or args.timing_repeats < 1:
        raise ValueError("repeat counts are invalid")
    if args.max_inflight < 1:
        raise ValueError("max_inflight must be positive")
    devices = [torch.device(value) for value in args.devices]
    if any(device.type != "cuda" for device in devices):
        raise ValueError("system benchmark requires CUDA devices")
    if len(set(devices)) != len(devices):
        raise ValueError("devices must be unique")


def formal_configuration(args: argparse.Namespace) -> bool:
    return (
        args.base_days == 4
        and args.source_version == 0
        and args.target_version == 11
        and args.program == DEFAULT_PROGRAM
        and len(args.devices) == 4
        and args.max_users == 64
        and args.batch_size == 2
        and args.warmup_repeats == 1
        and args.timing_repeats == 3
        and args.max_inflight == 3
        and args.seed == 0
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.smoke_test:
        smoke(args)
        return
    formal = formal_configuration(args)
    if not formal and not args.allow_diagnostic_protocol:
        raise ValueError(
            "formal system evaluation requires every frozen default; use "
            "--allow-diagnostic-protocol only for diagnostics"
        )
    seed_everything(args.seed)
    training = json.loads(Path(args.training_result).read_text())
    if training.get("protocol") != training_protocol_for_base_days(args.base_days):
        raise ValueError("training protocol mismatch")
    if training.get("status") != "complete":
        raise ValueError("training result is incomplete")
    prepared_hash = sha256(args.prepared_data)
    if prepared_hash != training["prepared_data"]["sha256"]:
        raise ValueError("prepared data differs from the training artifact")
    plan, metadata = load_prepared_kuairand_plan(args.prepared_data)
    validate_long_context_plan(plan, metadata, args.base_days)
    date, samples = reconstruct_online_eval_samples(
        plan,
        (args.target_version,),
        args.max_users,
    )[args.target_version]
    rng = np.random.default_rng(1907 + args.seed)
    samples = [samples[index] for index in rng.permutation(len(samples))]
    samples = samples[: args.max_users]
    program, program_payload = load_program(args.program)
    if program.source_version != f"theta{args.source_version}":
        raise ValueError("program source version differs from requested source")
    if program.target_version != f"theta{args.target_version}":
        raise ValueError("program target version differs from requested target")
    materialize_device = torch.device(args.devices[0])
    torch.cuda.set_device(materialize_device)
    cfg = HSTUConfig(**training["model"])
    old = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        args.source_version,
        materialize_device,
    )
    current = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        args.target_version,
        materialize_device,
    )
    capsules, raw_batches, materialization = materialize_real_capsules(
        old,
        current,
        samples,
        cfg.max_seq_len,
        args.batch_size,
        args.source_version,
        materialize_device,
        args.timing_repeats,
    )
    full_host = benchmark_full_host_boundary(
        current,
        raw_batches,
        materialize_device,
        args.warmup_repeats,
        args.timing_repeats,
    )
    del old, current, raw_batches
    torch.cuda.empty_cache()
    representative = max(
        capsules,
        key=lambda capsule: int(capsule.lengths.sum().item()),
    )
    resident = resident_operator_benchmark(
        program,
        representative,
        materialize_device,
        args.warmup_repeats,
        args.timing_repeats,
    )
    runtime = run_runtime_points(
        program,
        capsules,
        args.devices,
        args,
    )
    for point in runtime:
        point["speedup_over_single_gpu_synchronous_full"] = (
            full_host["median_elapsed_seconds"]
            / point["median_elapsed_seconds"]
        )
    result = {
        "protocol": SYNC_SYSTEM_PROTOCOL,
        "status": "complete" if formal else "diagnostic_complete",
        "formal_protocol": formal,
        "source_training_result": args.training_result,
        "program": {
            "path": args.program,
            "protocol": program_payload["protocol"],
            "source_version": program.source_version,
            "target_version": program.target_version,
            "rank": program.adapter.source_rank,
            "bytes_fp32": program.nbytes,
        },
        "prepared_data": {
            "path": args.prepared_data,
            "sha256": prepared_hash,
        },
        "eval_date": date,
        "source_version": args.source_version,
        "target_version": args.target_version,
        "cache_age_updates": args.target_version - args.source_version,
        "devices": args.devices,
        "device_names": [
            torch.cuda.get_device_name(torch.device(device))
            for device in args.devices
        ],
        "tier_semantics": {
            "hot": "resident HBM capsule and packed migration operator",
            "warm": (
                "pinned-host capsule with overlapped H2D, packed compute, "
                "and D2H cache publication"
            ),
            "placement_unit": (
                "version-cohort cache extent; model is replicated because it "
                "is small relative to aggregate cache state"
            ),
            "admission": "none; every stale extent receives the same sync ladder",
        },
        "materialization": materialization,
        "resident_hbm_operator": resident,
        "full_recompute_host_boundary": full_host,
        "host_warm_runtime": runtime,
    }
    save_json(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
