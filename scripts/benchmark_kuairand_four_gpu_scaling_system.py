from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from benchmark_kuairand_cohort_jagged_system import (
    collect_hbm_runtime,
    collect_jagged_runtime,
    pack_jagged_records,
    summarize_record_group,
)
from benchmark_kuairand_two_gpu_migration_system import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_PREPARED,
    DEFAULT_PROGRAM_DIR,
    DEFAULT_TRAINING,
    collect_recompute_runtime,
    fixed_count_assignment,
    load_verified_programs,
    materialize_records,
    pack_raw_records,
    sha256,
    split_verified_test,
    system_topology,
)
from motivation_validity import seed_everything

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    FusedJaggedMigrationOperator,
    PinnedJaggedKVOutputPool,
    PinnedKVOutputPool,
)
from hstu_kvcache.models import HSTUConfig
from hstu_kvcache.streaming import (
    COHORT_JAGGED_SYSTEM_PROTOCOL,
    load_checkpoint_model,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

FOUR_GPU_SCALING_PROTOCOL = (
    "kuairand_long_context_4plus12_mixed_version_four_gpu_scaling_v1"
)
DEFAULT_V3_RESULT = (
    "results/system/"
    "kuairand_long_context_4plus12_cohort_jagged_system_seed0.json"
)
DEFAULT_OUTPUT = (
    "results/system/"
    "kuairand_long_context_4plus12_four_gpu_scaling_seed0.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--program-dir", default=DEFAULT_PROGRAM_DIR)
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--v3-result", default=DEFAULT_V3_RESULT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    )
    parser.add_argument(
        "--device-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )
    parser.add_argument("--base-days", type=int, default=4)
    parser.add_argument(
        "--source-versions",
        type=int,
        nargs="+",
        default=[0, 4, 10],
    )
    parser.add_argument(
        "--source-weights",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.5],
    )
    parser.add_argument("--target-version", type=int, default=11)
    parser.add_argument("--layout-search-users", type=int, default=32)
    parser.add_argument("--max-users", type=int, default=64)
    parser.add_argument("--materialize-batch-size", type=int, default=2)
    parser.add_argument("--exact-batch-size", type=int, default=2)
    parser.add_argument("--exact-bucket-width", type=int, default=32)
    parser.add_argument("--max-inflight", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-diagnostic-protocol", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    devices = [torch.device(value) for value in args.devices]
    if any(device.type != "cuda" for device in devices):
        raise ValueError("scaling benchmark requires CUDA devices")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA devices must be unique")
    if max(args.device_counts) > len(devices):
        raise ValueError("device count exceeds the declared CUDA device set")
    if sorted(set(args.device_counts)) != args.device_counts:
        raise ValueError("device counts must be sorted and unique")
    if args.device_counts[0] != 1:
        raise ValueError("device counts must include the one-GPU baseline")
    if args.source_versions != [0, 4, 10]:
        raise ValueError("protocol freezes theta0, theta4, and theta10")
    if args.target_version != 11 or args.base_days != 4:
        raise ValueError("protocol freezes theta11 under 4+12")
    if min(
        args.layout_search_users,
        args.max_users,
        args.materialize_batch_size,
        args.exact_batch_size,
        args.exact_bucket_width,
        args.max_inflight,
        args.timing_repeats,
        *args.device_counts,
    ) < 1:
        raise ValueError("counts and batching settings must be positive")
    if args.warmup_repeats < 0:
        raise ValueError("warmup count must be nonnegative")


def formal_configuration(args: argparse.Namespace) -> bool:
    return (
        args.prepared_data == DEFAULT_PREPARED
        and args.training_result == DEFAULT_TRAINING
        and args.checkpoint_dir == DEFAULT_CHECKPOINTS
        and args.program_dir == DEFAULT_PROGRAM_DIR
        and args.manifest_dir == DEFAULT_MANIFEST_DIR
        and args.v3_result == DEFAULT_V3_RESULT
        and args.output == DEFAULT_OUTPUT
        and args.devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        and args.device_counts == [1, 2, 4]
        and args.source_weights == [0.2, 0.3, 0.5]
        and args.layout_search_users == 32
        and args.max_users == 64
        and args.materialize_batch_size == 2
        and args.exact_batch_size == 2
        and args.exact_bucket_width == 32
        and args.max_inflight == 3
        and args.warmup_repeats == 1
        and args.timing_repeats == 5
        and args.seed == 0
    )


def reset_memory_peaks(devices: list[str]) -> None:
    for value in devices:
        device = torch.device(value)
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)


def memory_snapshot(devices: list[str]) -> dict:
    per_device = []
    for value in devices:
        device = torch.device(value)
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            per_device.append(
                {
                    "device": str(device),
                    "current_allocated_bytes": torch.cuda.memory_allocated(
                        device
                    ),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                        device
                    ),
                    "current_reserved_bytes": torch.cuda.memory_reserved(
                        device
                    ),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                        device
                    ),
                }
            )
    return {
        "per_device": per_device,
        "aggregate_peak_allocated_bytes": sum(
            value["peak_allocated_bytes"] for value in per_device
        ),
        "maximum_device_peak_allocated_bytes": max(
            value["peak_allocated_bytes"] for value in per_device
        ),
        "aggregate_peak_reserved_bytes": sum(
            value["peak_reserved_bytes"] for value in per_device
        ),
        "maximum_device_peak_reserved_bytes": max(
            value["peak_reserved_bytes"] for value in per_device
        ),
    }


def add_stability(point: dict) -> None:
    samples = point["elapsed_seconds"]
    mean = statistics.mean(samples)
    point["timing_mean_seconds"] = mean
    point["timing_stdev_seconds"] = (
        statistics.stdev(samples) if len(samples) > 1 else 0.0
    )
    point["timing_cv"] = (
        point["timing_stdev_seconds"] / mean if mean > 0 else 0.0
    )


def add_scaling(points: list[dict]) -> None:
    one = next(value for value in points if value["device_count"] == 1)
    baseline = one["median_elapsed_seconds"]
    for point in points:
        point["speedup_over_one_gpu"] = (
            baseline / point["median_elapsed_seconds"]
        )
        point["parallel_efficiency"] = (
            point["speedup_over_one_gpu"] / point["device_count"]
        )
        add_stability(point)


def main() -> None:
    args = parse_args()
    validate_args(args)
    formal = formal_configuration(args)
    if not formal and not args.allow_diagnostic_protocol:
        raise ValueError(
            "nondefault settings require --allow-diagnostic-protocol"
        )
    seed_everything(args.seed)
    training = json.loads(Path(args.training_result).read_text())
    if training.get("protocol") != training_protocol_for_base_days(
        args.base_days
    ):
        raise ValueError("training protocol mismatch")
    if training.get("status") != "complete":
        raise ValueError("training result is incomplete")
    if int(training["args"]["seed"]) != args.seed:
        raise ValueError("training seed differs from scaling protocol")
    prepared_hash = sha256(args.prepared_data)
    if prepared_hash != training["prepared_data"]["sha256"]:
        raise ValueError("prepared data differs from training")
    cfg = HSTUConfig(**training["model"])
    plan, metadata = load_prepared_kuairand_plan(args.prepared_data)
    validate_long_context_plan(plan, metadata, args.base_days)
    eval_date, all_samples = reconstruct_online_eval_samples(
        plan,
        (args.target_version,),
        1000,
    )[args.target_version]
    verified_test, split = split_verified_test(all_samples, args.seed)
    ordering = np.random.default_rng(43091 + args.seed).permutation(
        len(verified_test)
    )
    final_samples = [
        verified_test[index]
        for index in ordering[
            args.layout_search_users : (
                args.layout_search_users + args.max_users
            )
        ]
    ]
    if len(final_samples) != args.max_users:
        raise ValueError("verified final split is too small")
    assignments, source_counts = fixed_count_assignment(
        len(final_samples),
        args.source_versions,
        args.source_weights,
        58211 + args.seed,
    )
    entries = [
        (
            args.layout_search_users + record_id,
            sample,
            source_version,
        )
        for record_id, (sample, source_version) in enumerate(
            zip(final_samples, assignments, strict=True)
        )
    ]
    programs, program_evidence = load_verified_programs(args, cfg)
    materialize_device = torch.device(args.devices[0])
    torch.cuda.set_device(materialize_device)
    records, materialization = materialize_records(
        entries,
        cfg,
        args.checkpoint_dir,
        materialize_device,
        args.materialize_batch_size,
    )
    v3 = json.loads(Path(args.v3_result).read_text())
    if v3.get("protocol") != COHORT_JAGGED_SYSTEM_PROTOCOL:
        raise ValueError("v3 result protocol mismatch")
    frozen_token_budget = int(
        v3["layout_search"]["selected_host_token_budget"]
    )
    if (
        int(v3["layout_search"]["selected_hbm_token_budget"])
        != frozen_token_budget
    ):
        raise ValueError("v3 host and HBM layouts are not identical")
    record_summary = summarize_record_group(records)
    expected_counts = {
        f"theta{version}": count for version, count in source_counts.items()
    }
    if record_summary["source_counts"] != expected_counts:
        raise ValueError("materialized source counts differ")
    if v3["final_trace"] != record_summary:
        raise ValueError("four-GPU trace differs from the frozen v3 trace")
    capsules, packing = pack_jagged_records(records, frozen_token_budget)
    histories, exact_packing = pack_raw_records(
        records,
        args.exact_batch_size,
        args.exact_bucket_width,
        cfg.max_seq_len - 1,
    )
    del records
    gc.collect()
    host_pool = PinnedJaggedKVOutputPool.allocate(
        capsules,
        served_kv_target=f"theta{args.target_version}",
        num_layers=cfg.num_layers,
        kv_width=cfg.num_heads * cfg.head_dim,
        dtype=torch.float16,
    )
    host_points = []
    for count in args.device_counts:
        selected = args.devices[:count]
        reset_memory_peaks(selected)
        point = collect_jagged_runtime(
            programs,
            capsules,
            selected,
            FusedJaggedMigrationOperator(),
            host_pool,
            args.max_inflight,
            args.warmup_repeats,
            args.timing_repeats,
        )
        point["gpu_memory"] = memory_snapshot(selected)
        host_points.append(point)
    add_scaling(host_points)
    host_output_bytes = host_pool.nbytes
    del host_pool
    gc.collect()
    hbm_points = []
    for count in args.device_counts:
        selected = args.devices[:count]
        reset_memory_peaks(selected)
        point = collect_hbm_runtime(
            programs,
            capsules,
            selected,
            FusedJaggedMigrationOperator(),
            args.max_inflight,
            args.warmup_repeats,
            args.timing_repeats,
        )
        point["gpu_memory"] = memory_snapshot(selected)
        hbm_points.append(point)
    add_scaling(hbm_points)
    exact_pool = PinnedKVOutputPool.allocate(
        histories,
        served_kv_target=f"theta{args.target_version}",
        num_layers=cfg.num_layers,
        kv_width=cfg.num_heads * cfg.head_dim,
        dtype=torch.float16,
    )
    current_models = [
        load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            args.target_version,
            torch.device(value),
        )
        for value in args.devices
    ]
    exact_points = []
    for count in args.device_counts:
        selected = args.devices[:count]
        reset_memory_peaks(selected)
        point = collect_recompute_runtime(
            current_models,
            ["raw_history"],
            f"theta{args.target_version}",
            histories,
            selected,
            exact_pool,
            torch.bfloat16,
            args.max_inflight,
            args.warmup_repeats,
            args.timing_repeats,
        )
        point["gpu_memory"] = memory_snapshot(selected)
        exact_points.append(point)
    add_scaling(exact_points)
    exact_output_bytes = exact_pool.nbytes
    model_replica_bytes = sum(
        value.numel() * value.element_size()
        for model in current_models
        for value in (*model.parameters(), *model.buffers())
    )
    logical_old_kv_bytes = (
        2
        * cfg.num_layers
        * record_summary["logical_tokens"]
        * cfg.num_heads
        * cfg.head_dim
        * torch.empty((), dtype=torch.float16).element_size()
    )
    host_by_count = {
        value["device_count"]: value for value in host_points
    }
    hbm_by_count = {
        value["device_count"]: value for value in hbm_points
    }
    exact_by_count = {
        value["device_count"]: value for value in exact_points
    }
    derived = {}
    for count in args.device_counts:
        derived[str(count)] = {
            "host_migration_speedup_over_bf16_exact": (
                exact_by_count[count]["median_elapsed_seconds"]
                / host_by_count[count]["median_elapsed_seconds"]
            ),
            "hbm_completion_ratio_over_host_publication": (
                host_by_count[count]["median_elapsed_seconds"]
                / hbm_by_count[count]["median_elapsed_seconds"]
            ),
        }
    result = {
        "protocol": FOUR_GPU_SCALING_PROTOCOL,
        "status": (
            "adaptive_system_complete"
            if formal
            else "diagnostic_complete"
        ),
        "formal_protocol": formal,
        "study_stage": "adaptive_seed0_four_gpu_system_scaling",
        "source_training_result": args.training_result,
        "prepared_data": {
            "path": args.prepared_data,
            "sha256": prepared_hash,
        },
        "checkpoint_dir": args.checkpoint_dir,
        "source_v3_result": args.v3_result,
        "eval_date": eval_date,
        "seed": args.seed,
        "split": split,
        "workload": {
            "kind": (
                "frozen mixed-version held-out KuaiRand update cohort"
            ),
            "users": len(final_samples),
            "source_versions": args.source_versions,
            "source_weights": args.source_weights,
            "source_counts": expected_counts,
            "target_version": args.target_version,
            "labels_used": False,
            "selection_offset": args.layout_search_users,
        },
        "frozen_layout": {
            "source": args.v3_result,
            "selection_users_reused_for_timing": False,
            "valid_token_budget": frozen_token_budget,
            "packing": packing,
        },
        "programs": program_evidence,
        "topology": system_topology(args.devices),
        "materialization": materialization,
        "final_trace": record_summary,
        "state_capacity": {
            "cached_old_norm_capsule_bytes": (
                record_summary["capsule_bytes_unpadded"]
            ),
            "logical_old_kv_bytes_at_fp16": logical_old_kv_bytes,
            "extra_capsule_ratio_to_logical_old_kv": (
                record_summary["capsule_bytes_unpadded"]
                / logical_old_kv_bytes
            ),
            "host_persistent_target_bytes": host_output_bytes,
            "exact_persistent_target_bytes": exact_output_bytes,
            "current_model_replica_bytes_fp32": model_replica_bytes,
        },
        "exact_packing": exact_packing,
        "host_staged_dram_scaling": host_points,
        "direct_hbm_scaling": hbm_points,
        "bf16_full_recompute_host_scaling": exact_points,
        "derived": derived,
        "scope_boundary": {
            "included": (
                "frozen real mixed-version capsules, published programs, "
                "fused valid-token operator, pinned-host movement, direct "
                "HBM publication, persistent host publication, LPT extent "
                "assignment, per-device work, GPU memory, and 1/2/4 scaling"
            ),
            "excluded": (
                "new layout search, recommendation-quality evaluation, "
                "organic cohort generation, physical SSD or network, "
                "same-boundary HBM full recomputation, and foreground serving"
            ),
            "comparison_rule": (
                "compiled migration and full recomputation are compared only "
                "at the common pinned-host-to-pinned-host boundary; HBM versus "
                "DRAM times are endpoint completion ratios, not operator speedups"
            ),
        },
    }
    save_json(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
