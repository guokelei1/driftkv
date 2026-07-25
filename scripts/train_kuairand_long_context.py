"""Train all checkpoints for a supported KuaiRand long-context split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from motivation_validity import seed_everything
from torch.nn.parallel import DistributedDataParallel

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import (
    SUPPORTED_LONG_CONTEXT_BASE_DAYS,
    DistributedRuntime,
    batch_coverage,
    close_distributed_runtime,
    init_distributed_runtime,
    long_context_split_name,
    make_long_context_config,
    model_shape_summary,
    parameter_distance,
    prefix_state_footprint,
    primary_log,
    shard_train_batches,
    train_distributed_batches,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

DEFAULT_CHECKPOINT_DIR = "checkpoints/kuairand_long_context_8plus8/seed0"
DEFAULT_OUTPUT = "results/motivation_scale/long_context_8plus8_training_seed0.json"
DEFAULT_PREPARED_DATA = "data/processed/kuairand_long_context_8plus8_v2.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-days",
        type=int,
        choices=SUPPORTED_LONG_CONTEXT_BASE_DAYS,
        default=8,
    )
    parser.add_argument("--prepared-data")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--base-epochs", type=int, default=6)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--stream-epochs", type=int, default=2)
    parser.add_argument("--stream-lr", type=float, default=1e-4)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=64)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--data-dry-run", action="store_true")
    parser.add_argument("--distributed-smoke-test", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    split = long_context_split_name(args.base_days)
    if args.prepared_data is None:
        args.prepared_data = (
            DEFAULT_PREPARED_DATA
            if args.base_days == 8
            else (
                f"data/processed/kuairand_long_context_{split}"
                "_exploration_v1.npz"
            )
        )
    if args.checkpoint_dir is None:
        args.checkpoint_dir = (
            DEFAULT_CHECKPOINT_DIR
            if args.base_days == 8 and args.seed == 0
            else (
                f"checkpoints/kuairand_long_context_{split}"
                f"{'_exploration' if args.base_days != 8 else ''}/seed{args.seed}"
            )
        )
    if args.output is None:
        args.output = (
            DEFAULT_OUTPUT
            if args.base_days == 8 and args.seed == 0
            else (
                f"results/motivation_scale/long_context_{split}_training"
                f"{'_exploration' if args.base_days != 8 else ''}"
                f"_seed{args.seed}.json"
            )
        )


def validate_args(args: argparse.Namespace) -> None:
    values = {
        "batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "base_epochs": args.base_epochs,
        "stream_epochs": args.stream_epochs,
        "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb,
    }
    invalid = [name for name, value in values.items() if value < 1]
    if invalid:
        raise ValueError(f"positive values required for: {', '.join(invalid)}")
    if args.base_lr <= 0 or args.stream_lr <= 0:
        raise ValueError("learning rates must be positive")
    frozen = {
        "batch_size": (args.batch_size, 4),
        "micro_batch_size": (args.micro_batch_size, 1),
        "base_epochs": (args.base_epochs, 6),
        "base_lr": (args.base_lr, 3e-4),
        "stream_epochs": (args.stream_epochs, 2),
        "stream_lr": (args.stream_lr, 1e-4),
    }
    changed = {
        name: {"expected": expected, "actual": actual}
        for name, (actual, expected) in frozen.items()
        if actual != expected
    }
    if changed and not args.distributed_smoke_test:
        raise ValueError(f"frozen training protocol arguments changed: {changed}")


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    model: HSTU,
    checkpoint_dir: str,
    version: int,
    runtime: DistributedRuntime,
) -> None:
    if not runtime.is_primary:
        return
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / f"theta_{version}.pt")


def make_smoke_batch(offset: int) -> dict:
    return {
        "item_ids": torch.tensor(
            [
                [1 + offset, 2 + offset, 3 + offset, 4 + offset],
                [5 + offset, 6 + offset, 7 + offset, 0],
            ],
            dtype=torch.long,
        ),
        "behaviors": torch.tensor([[1, 2, 3, 2], [1, 2, 3, 0]]),
        "time_deltas": torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 0.0]]
        ),
        "lengths": torch.tensor([4, 3]),
        "labels": torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]]),
        "train_mask": torch.tensor(
            [[False, True, True, True], [False, True, True, False]]
        ),
    }


def run_distributed_smoke_test(
    args: argparse.Namespace,
    runtime: DistributedRuntime,
) -> None:
    seed_everything(args.seed)
    cfg = HSTUConfig(
        num_items=128,
        num_prediction_items=96,
        num_behaviors=9,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        max_seq_len=8,
        activation="relu",
        input_dropout=0.0,
    )
    model = HSTU(cfg).to(runtime.device)
    if runtime.initialized:
        training_model: torch.nn.Module = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    else:
        training_model = model
    global_batch_count = max(1, runtime.world_size - 1)
    global_batches = [
        make_smoke_batch(index * 8)
        for index in range(global_batch_count)
    ]
    local_batches, coverage = shard_train_batches(global_batches, runtime)
    optimizer = torch.optim.AdamW(training_model.parameters(), lr=1e-3)
    loss = train_distributed_batches(
        training_model,
        optimizer,
        local_batches,
        runtime,
        micro_batch_size=args.micro_batch_size,
    )
    flat = torch.cat(
        [parameter.detach().flatten() for parameter in model.parameters()]
    )
    max_parameter_difference = 0.0
    if runtime.initialized:
        replicas = [torch.empty_like(flat) for _ in range(runtime.world_size)]
        dist.all_gather(replicas, flat)
        max_parameter_difference = max(
            float((replica - replicas[0]).abs().max().item())
            for replica in replicas[1:]
        )
    if not np.isfinite(loss):
        raise RuntimeError("distributed smoke test produced a non-finite loss")
    if max_parameter_difference > 1e-6:
        raise RuntimeError("distributed replicas diverged after one optimizer step")
    primary_log(
        runtime,
        json.dumps(
            {
                "world_size": runtime.world_size,
                "loss": loss,
                "coverage": coverage,
                "max_parameter_difference": max_parameter_difference,
                "status": "ok",
            },
            indent=2,
        ),
    )


def run_data_dry_run(
    plan,
    prepared_metadata: dict,
    args: argparse.Namespace,
    runtime: DistributedRuntime,
    protocol: str,
) -> None:
    cfg = make_long_context_config(
        plan.num_items,
        plan.num_prediction_items,
        plan.num_behaviors,
    )
    plan.init_base()
    base_batches = list(
        plan.iter_base_train_batches(
            args.batch_size,
            all_chunks=True,
            bucket_by_length=True,
            pad_to_max_seq_len=False,
        )
    )
    base = batch_coverage(base_batches)
    base["four_gpu_steps_per_epoch"] = (
        base["batches"] + 3
    ) // 4
    base["four_gpu_padding_batches"] = (
        base["four_gpu_steps_per_epoch"] * 4 - base["batches"]
    )
    del base_batches
    online = []
    evaluation = []
    last_evaluable_version = len(plan.stream_dates) - 1
    for online_index, date in enumerate(plan.stream_dates):
        samples = plan.get_eval_set(date)
        current_version = online_index
        evaluation.append(
            {
                "date": date,
                "current_version": current_version,
                "eligible_users": len(samples),
                "role": (
                    "final_primary"
                    if current_version == last_evaluable_version
                    else ("age_curve" if current_version > 0 else "base_diagnostic")
                ),
                "state": prefix_state_footprint(samples, cfg),
            }
        )
        plan.ingest_day(date)
        batches = list(
            plan.iter_train_batches(
                date,
                args.batch_size,
                all_chunks=True,
                bucket_by_length=True,
                pad_to_max_seq_len=False,
            )
        )
        coverage = batch_coverage(batches)
        coverage["four_gpu_steps_per_epoch"] = (
            coverage["batches"] + 3
        ) // 4
        coverage["four_gpu_padding_batches"] = (
            coverage["four_gpu_steps_per_epoch"] * 4 - coverage["batches"]
        )
        online.append(
            {
                "date": date,
                "new_version": online_index + 1,
                "coverage": coverage,
            }
        )
        del batches
    primary_log(
        runtime,
        json.dumps(
            {
                "protocol": protocol,
                "prepared_data": prepared_metadata,
                "model": model_shape_summary(cfg),
                "batching": {
                    "per_gpu_logical_batch": args.batch_size,
                    "per_gpu_micro_batch": args.micro_batch_size,
                    "gradient_accumulation_steps": (
                        args.batch_size + args.micro_batch_size - 1
                    ) // args.micro_batch_size,
                    "planned_world_size": 4,
                    "planned_global_batch": args.batch_size * 4,
                    "training_sequences": "all_chunks",
                    "padding": "dynamic length-bucketed",
                },
                "base_training_per_epoch": base,
                "online_training_per_epoch": online,
                "evaluation": evaluation,
                "formal_training_started": False,
            },
            indent=2,
        ),
    )


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    validate_args(args)
    training_protocol = training_protocol_for_base_days(args.base_days)
    expected_cfg = make_long_context_config(312144, 50000, 9)
    if args.validate_only:
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(model_shape_summary(expected_cfg), indent=2), flush=True)
        return

    runtime = init_distributed_runtime(
        args.device,
        args.distributed_backend,
    )
    args.device = str(runtime.device)
    started = time.perf_counter()
    try:
        if args.distributed_smoke_test:
            run_distributed_smoke_test(args, runtime)
            return
        seed_everything(args.seed)
        torch.set_float32_matmul_precision("high")
        plan, prepared_metadata = load_prepared_kuairand_plan(args.prepared_data)
        validate_long_context_plan(plan, prepared_metadata, args.base_days)
        if args.data_dry_run:
            run_data_dry_run(
                plan,
                prepared_metadata,
                args,
                runtime,
                training_protocol,
            )
            return
        if runtime.world_size != 4:
            raise ValueError("formal long-context training requires exactly four DDP workers")
        cfg = make_long_context_config(
            plan.num_items,
            plan.num_prediction_items,
            plan.num_behaviors,
        )
        plan.init_base()
        model = HSTU(cfg).to(runtime.device)
        if runtime.initialized:
            training_model: torch.nn.Module = DistributedDataParallel(
                model,
                device_ids=[runtime.local_rank],
                output_device=runtime.local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                bucket_cap_mb=args.ddp_bucket_cap_mb,
            )
        else:
            training_model = model
        torch.manual_seed(args.seed + runtime.rank * 100003)
        if runtime.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + runtime.rank * 100003)
        shape = model_shape_summary(cfg)
        primary_log(
            runtime,
            f"world_size={runtime.world_size} parameters={shape['num_parameters']:,} "
            f"per_device_batch={args.batch_size} "
            f"global_batch={args.batch_size * runtime.world_size}",
        )
        result = {
            "protocol": training_protocol,
            "args": vars(args),
            "prepared_data": {
                "path": args.prepared_data,
                "sha256": artifact_sha256(args.prepared_data),
                "metadata": prepared_metadata,
            },
            "schedule": {
                "base_dates": plan.base_dates,
                "update_dates": plan.stream_dates,
                "last_evaluable_date": plan.stream_dates[-1],
                "last_evaluable_version": len(plan.stream_dates) - 1,
                "post_horizon_version": len(plan.stream_dates),
                "versions": {
                    "theta0": "trained on base dates",
                    **{
                        f"theta{index + 1}": (
                            f"theta{index} updated on {date}"
                        )
                        for index, date in enumerate(plan.stream_dates)
                    },
                },
            },
            "execution": {
                "world_size": runtime.world_size,
                "per_device_batch_size": args.batch_size,
                "per_device_micro_batch_size": args.micro_batch_size,
                "gradient_accumulation_steps": (
                    args.batch_size + args.micro_batch_size - 1
                ) // args.micro_batch_size,
                "global_batch_size": args.batch_size * runtime.world_size,
                "precision": "float32",
                "training_sequences": "all_chunks",
                "attention_execution": "dense_length_bucketed",
                "distributed_backend": (
                    args.distributed_backend if runtime.initialized else None
                ),
            },
            "model": asdict(cfg),
            "num_parameters": shape["num_parameters"],
            "base_training": [],
            "online_updates": [],
        }
        base_optimizer = torch.optim.AdamW(
            training_model.parameters(),
            lr=args.base_lr,
            weight_decay=1e-4,
        )
        for epoch in range(args.base_epochs):
            global_batches = list(
                plan.iter_base_train_batches(
                    args.batch_size,
                    all_chunks=True,
                    bucket_by_length=True,
                    pad_to_max_seq_len=False,
                )
            )
            local_batches, coverage = shard_train_batches(
                global_batches,
                runtime,
            )
            loss = train_distributed_batches(
                training_model,
                base_optimizer,
                local_batches,
                runtime,
                micro_batch_size=args.micro_batch_size,
            )
            if runtime.is_primary:
                result["base_training"].append(
                    {
                        "epoch": epoch + 1,
                        "loss": loss,
                        "coverage": coverage,
                    }
                )
                save_json(result, args.output)
            primary_log(runtime, f"base_epoch={epoch + 1} loss={loss:.6f}")
        theta0_model = HSTU(cfg).to(runtime.device)
        theta0_model.load_state_dict(model.state_dict())
        theta0_model.eval()
        for parameter in theta0_model.parameters():
            parameter.requires_grad_(False)
        save_checkpoint(model, args.checkpoint_dir, 0, runtime)

        stream_optimizer = torch.optim.AdamW(
            training_model.parameters(),
            lr=args.stream_lr,
            weight_decay=1e-4,
        )
        for online_index, date in enumerate(plan.stream_dates):
            plan.ingest_day(date)
            global_batches = list(
                plan.iter_train_batches(
                    date,
                    args.batch_size,
                    all_chunks=True,
                    bucket_by_length=True,
                    pad_to_max_seq_len=False,
                )
            )
            epoch_losses = []
            coverage = None
            for _ in range(args.stream_epochs):
                random.shuffle(global_batches)
                local_batches, coverage = shard_train_batches(
                    global_batches,
                    runtime,
                )
                epoch_losses.append(
                    train_distributed_batches(
                        training_model,
                        stream_optimizer,
                        local_batches,
                        runtime,
                        micro_batch_size=args.micro_batch_size,
                    )
                )
            version = online_index + 1
            update = {
                "train_date": date,
                "new_version": version,
                "epoch_losses": epoch_losses,
                "coverage": coverage,
                "cumulative_dtheta_rel": parameter_distance(
                    model,
                    theta0_model,
                ),
            }
            if runtime.is_primary:
                result["online_updates"].append(update)
                save_json(result, args.output)
            save_checkpoint(model, args.checkpoint_dir, version, runtime)
            primary_log(
                runtime,
                f"trained_theta={version} date={date} "
                f"loss={float(np.mean(epoch_losses)):.6f} "
                f"cumulative_dtheta={update['cumulative_dtheta_rel']:.6f}",
            )
        if runtime.is_primary:
            result["runtime_seconds"] = time.perf_counter() - started
            result["status"] = "complete"
            save_json(result, args.output)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "checkpoint_dir": args.checkpoint_dir,
                        "versions": list(range(len(plan.stream_dates) + 1)),
                        "runtime_seconds": result["runtime_seconds"],
                    },
                    indent=2,
                ),
                flush=True,
            )
    finally:
        close_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
