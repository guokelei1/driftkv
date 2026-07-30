from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from hstu_kvcache.data import load_prepared_exposure_plan
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import (
    DistributedRuntime,
    batch_coverage,
    close_distributed_runtime,
    init_distributed_runtime,
    model_shape_summary,
    primary_log,
    shard_train_batches,
    train_distributed_batches,
)
from hstu_kvcache.utils import save_json

PROTOCOL = "evokv_design3_m1_qk_two_version_training_dev_v0"
DEFAULT_PREPARED_DATA = (
    "data/processed/evokv_d3_m1_qk_ctx512144_8704.npz"
)
DEFAULT_CHECKPOINT_DIR = (
    "checkpoints/evokv_design3_m1_qk_ctx512144/seed0"
)
DEFAULT_OUTPUT = (
    "results/system/evokv_design3_m1/"
    "qk_ctx512144_two_version_training_seed0.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED_DATA)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--base-epochs", type=int, default=1)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--stream-epochs", type=int, default=1)
    parser.add_argument("--stream-lr", type=float, default=1e-4)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=64)
    parser.add_argument("--data-dry-run", action="store_true")
    parser.add_argument("--tiny-cpu-smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qk_model_config(
    num_items: int,
    num_behaviors: int,
    num_prediction_items: int | None = None,
) -> HSTUConfig:
    return HSTUConfig(
        num_items=num_items,
        num_prediction_items=num_prediction_items,
        num_behaviors=num_behaviors,
        hidden_size=512,
        num_layers=16,
        num_heads=8,
        head_dim=64,
        max_seq_len=512,
        activation="relu",
        input_dropout=0.0,
    )


def validate_args(args: argparse.Namespace) -> None:
    positive_integers = {
        "batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "base_epochs": args.base_epochs,
        "stream_epochs": args.stream_epochs,
        "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb,
    }
    invalid = [
        name for name, value in positive_integers.items() if value < 1
    ]
    if invalid:
        raise ValueError(
            f"positive values required for: {', '.join(invalid)}"
        )
    if args.base_lr <= 0 or args.stream_lr <= 0:
        raise ValueError("learning rates must be positive")


def validate_prepared_metadata(metadata: dict[str, object]) -> None:
    dataset = str(metadata.get("dataset", "")).lower().replace("_", "-")
    if dataset != "tenrec-qk":
        raise ValueError("M1 edge training requires a Tenrec QK prepared stream")
    if int(metadata.get("window_count", 0)) < 1:
        raise ValueError("M1 edge training requires window_0")
    if int(metadata.get("selected_users", 0)) < 1:
        raise ValueError("M1 edge training requires at least one user")
    if int(metadata.get("fitted_items", 0)) < 1:
        raise ValueError("M1 edge training requires a fitted item vocabulary")


def boundary_metadata(metadata: dict[str, object]) -> dict[str, object]:
    keys = (
        "protocol",
        "dataset",
        "catalog_fit",
        "cohort_selection",
        "cohort_selection_mode",
        "base_prefix",
        "base_filtered_events",
        "base_event_count",
        "window_size",
        "window_filtered_events",
        "window_count",
        "ordinal_unit",
        "ordering",
        "selected_users",
        "fitted_items",
        "num_prediction_items",
        "context_hash_buckets",
        "context_hash_function",
        "context_rule",
        "prediction_catalog_rows",
        "context_rows",
        "unique_context_buckets_touched",
        "rows",
        "positive_rows",
        "split_rows",
        "split_positive_rows",
    )
    return {
        key: metadata[key]
        for key in keys
        if key in metadata
    }


def history_length_summary(plan) -> dict[str, float | int]:
    lengths = np.asarray(
        [
            len(value["item_ids"])
            for value in plan.user_histories.values()
        ],
        dtype=np.int64,
    )
    if len(lengths) == 0:
        return {"count": 0}
    return {
        "count": int(len(lengths)),
        "minimum": int(lengths.min()),
        "p50": float(np.quantile(lengths, 0.50)),
        "p90": float(np.quantile(lengths, 0.90)),
        "p99": float(np.quantile(lengths, 0.99)),
        "maximum": int(lengths.max()),
        "at_least_512": int(np.count_nonzero(lengths >= 512)),
    }


def load_qk_plan(path: str | Path):
    plan, metadata = load_prepared_exposure_plan(
        path,
        max_seq_len=512,
    )
    validate_prepared_metadata(metadata)
    if plan.base_dates != ["base"] or not plan.stream_dates:
        raise ValueError("M1 edge prepared stream has no base/window_0 boundary")
    if plan.stream_dates[0] != "window_0":
        raise ValueError("M1 edge first update must be window_0")
    return plan, metadata


def inspect_prepared_data(
    path: str | Path,
    batch_size: int,
) -> dict[str, object]:
    plan, metadata = load_qk_plan(path)
    plan.init_base()
    base_histories = history_length_summary(plan)
    base_batches = list(
        plan.iter_base_train_batches(
            batch_size,
            all_chunks=True,
            bucket_by_length=True,
            pad_to_max_seq_len=False,
        )
    )
    base_coverage = batch_coverage(base_batches)
    del base_batches
    plan.ingest_day("window_0")
    target_histories = history_length_summary(plan)
    update_batches = list(
        plan.iter_train_batches(
            "window_0",
            batch_size,
            all_chunks=True,
            bucket_by_length=True,
            pad_to_max_seq_len=False,
        )
    )
    update_coverage = batch_coverage(update_batches)
    del update_batches
    cfg = qk_model_config(
        plan.num_items,
        plan.num_behaviors,
        plan.num_prediction_items,
    )
    return {
        "protocol": PROTOCOL,
        "mode": "data_dry_run",
        "scientific_result": False,
        "prepared_data": {
            "path": str(path),
            "sha256": artifact_sha256(path),
            "boundary": boundary_metadata(metadata),
        },
        "data": {
            "users": plan.num_users,
            "items": plan.num_items,
            "behaviors": plan.num_behaviors,
            "base_histories": base_histories,
            "theta1_boundary_histories": target_histories,
        },
        "model": model_shape_summary(cfg),
        "training": {
            "training_sequences": "all_chunks",
            "base": base_coverage,
            "window_0": update_coverage,
        },
    }


def make_tiny_batch(update_only: bool) -> dict[str, torch.Tensor]:
    item_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
        ],
        dtype=torch.long,
    )
    train_mask = torch.ones_like(item_ids, dtype=torch.bool)
    if update_only:
        train_mask[:, :-2] = False
    return {
        "item_ids": item_ids,
        "behaviors": torch.tensor(
            [
                [1, 2, 1, 2, 1, 2, 1, 2],
                [2, 1, 2, 1, 2, 1, 2, 1],
            ],
            dtype=torch.long,
        ),
        "time_deltas": torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        "labels": torch.ones_like(item_ids),
        "lengths": torch.tensor([8, 8], dtype=torch.long),
        "train_mask": train_mask,
    }


def capture_parameters(model: HSTU) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
    }


def relative_parameter_distance(
    model: HSTU,
    reference: dict[str, torch.Tensor],
) -> float:
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    for name, value in model.named_parameters():
        base = reference[name]
        current = value.detach().cpu()
        numerator += (current.double() - base.double()).square().sum()
        denominator += base.double().square().sum()
    return float(
        (numerator.sqrt() / denominator.sqrt().clamp_min(1e-12)).item()
    )


def run_tiny_cpu_smoke(seed: int = 0) -> dict[str, object]:
    seed_everything(seed)
    cfg = HSTUConfig(
        num_items=32,
        num_prediction_items=32,
        num_behaviors=3,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        max_seq_len=8,
        activation="relu",
        input_dropout=0.0,
    )
    model = HSTU(cfg)
    runtime = DistributedRuntime(
        rank=0,
        world_size=1,
        local_rank=0,
        device=torch.device("cpu"),
        initialized=False,
    )
    base_batches, base_coverage = shard_train_batches(
        [make_tiny_batch(False)],
        runtime,
    )
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    base_loss = train_distributed_batches(
        model,
        base_optimizer,
        base_batches,
        runtime,
        micro_batch_size=1,
    )
    theta0 = capture_parameters(model)
    update_batches, update_coverage = shard_train_batches(
        [make_tiny_batch(True)],
        runtime,
    )
    update_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    update_loss = train_distributed_batches(
        model,
        update_optimizer,
        update_batches,
        runtime,
        micro_batch_size=1,
    )
    distance = relative_parameter_distance(model, theta0)
    passed = (
        math.isfinite(base_loss)
        and math.isfinite(update_loss)
        and distance > 0
        and base_coverage["eligible_targets"] > 0
        and update_coverage["eligible_targets"] > 0
    )
    if not passed:
        raise RuntimeError("M1 tiny CPU smoke test failed")
    return {
        "protocol": PROTOCOL,
        "mode": "tiny_cpu_smoke_test",
        "status": "complete",
        "base_loss": base_loss,
        "update_loss": update_loss,
        "theta0_to_theta1_dtheta_rel": distance,
        "base_coverage": base_coverage,
        "update_coverage": update_coverage,
    }


def checkpoint_path(directory: str | Path, version: int) -> Path:
    return Path(directory) / f"theta_{version}.pt"


def save_checkpoint(
    model: HSTU,
    directory: str | Path,
    version: int,
    runtime: DistributedRuntime,
) -> None:
    if not runtime.is_primary:
        return
    path = checkpoint_path(directory, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def checkpoint_descriptor(
    directory: str | Path,
    version: int,
) -> dict[str, object]:
    path = checkpoint_path(directory, version)
    return {
        "version": f"theta{version}",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": artifact_sha256(path),
    }


def validate_output_targets(args: argparse.Namespace) -> None:
    targets = (
        Path(args.output),
        checkpoint_path(args.checkpoint_dir, 0),
        checkpoint_path(args.checkpoint_dir, 1),
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            f"M1 output exists; pass --force to replace: {existing}"
        )


def run_training(args: argparse.Namespace) -> dict[str, object] | None:
    validate_output_targets(args)
    runtime = init_distributed_runtime(
        args.device,
        args.distributed_backend,
    )
    started = time.perf_counter()
    try:
        if (
            runtime.world_size != 2
            or not runtime.initialized
            or runtime.device.type != "cuda"
        ):
            raise RuntimeError(
                "M1 formal edge training requires two torchrun CUDA ranks"
            )
        seed_everything(args.seed)
        torch.set_float32_matmul_precision("high")
        plan, metadata = load_qk_plan(args.prepared_data)
        cfg = qk_model_config(
            plan.num_items,
            plan.num_behaviors,
            plan.num_prediction_items,
        )
        plan.init_base()
        model = HSTU(cfg).to(runtime.device)
        training_model: torch.nn.Module = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
        )
        torch.manual_seed(args.seed + runtime.rank * 100003)
        torch.cuda.manual_seed_all(args.seed + runtime.rank * 100003)
        primary_log(
            runtime,
            (
                f"world_size=2 parameters="
                f"{sum(value.numel() for value in model.parameters()):,} "
                f"users={plan.num_users} items={plan.num_items}"
            ),
        )
        base_training = []
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
            if not math.isfinite(loss):
                raise RuntimeError("M1 base training produced a non-finite loss")
            base_training.append(
                {
                    "epoch": epoch + 1,
                    "loss": loss,
                    "coverage": coverage,
                }
            )
            primary_log(
                runtime,
                f"base_epoch={epoch + 1} loss={loss:.6f}",
            )
            del global_batches, local_batches
        save_checkpoint(model, args.checkpoint_dir, 0, runtime)
        theta0 = capture_parameters(model) if runtime.is_primary else None
        dist.barrier()
        plan.ingest_day("window_0")
        update_training = []
        stream_optimizer = torch.optim.AdamW(
            training_model.parameters(),
            lr=args.stream_lr,
            weight_decay=1e-4,
        )
        for epoch in range(args.stream_epochs):
            global_batches = list(
                plan.iter_train_batches(
                    "window_0",
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
                stream_optimizer,
                local_batches,
                runtime,
                micro_batch_size=args.micro_batch_size,
            )
            if not math.isfinite(loss):
                raise RuntimeError(
                    "M1 window_0 training produced a non-finite loss"
                )
            update_training.append(
                {
                    "epoch": epoch + 1,
                    "loss": loss,
                    "coverage": coverage,
                }
            )
            primary_log(
                runtime,
                f"window_0_epoch={epoch + 1} loss={loss:.6f}",
            )
            del global_batches, local_batches
        save_checkpoint(model, args.checkpoint_dir, 1, runtime)
        dist.barrier()
        if not runtime.is_primary:
            return None
        if theta0 is None:
            raise RuntimeError("M1 primary theta0 state is missing")
        result = {
            "protocol": PROTOCOL,
            "status": "complete",
            "scientific_result": False,
            "formal_design3": False,
            "artifact_role": "m1_two_version_model_edge",
            "prepared_data": {
                "path": args.prepared_data,
                "sha256": artifact_sha256(args.prepared_data),
                "boundary": boundary_metadata(metadata),
            },
            "data": {
                "users": plan.num_users,
                "items": plan.num_items,
                "behaviors": plan.num_behaviors,
            },
            "model": asdict(cfg),
            "num_parameters": sum(
                value.numel() for value in model.parameters()
            ),
            "execution": {
                "world_size": runtime.world_size,
                "distributed_backend": args.distributed_backend,
                "visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES",
                    "",
                ),
                "per_rank_logical_batch_size": args.batch_size,
                "per_rank_micro_batch_size": args.micro_batch_size,
                "training_sequences": "all_chunks",
                "padding": "dynamic_length_bucketed",
                "precision": "float32",
                "seed": args.seed,
            },
            "schedule": {
                "theta0": "base",
                "theta1": "theta0 updated on window_0",
                "updates_consumed": ["window_0"],
            },
            "optimization": {
                "base_epochs": args.base_epochs,
                "base_lr": args.base_lr,
                "stream_epochs": args.stream_epochs,
                "stream_lr": args.stream_lr,
                "weight_decay": 1e-4,
            },
            "base_training": base_training,
            "update_training": {
                "window": "window_0",
                "epochs": update_training,
            },
            "theta0_to_theta1_dtheta_rel": (
                relative_parameter_distance(model, theta0)
            ),
            "checkpoints": [
                checkpoint_descriptor(args.checkpoint_dir, 0),
                checkpoint_descriptor(args.checkpoint_dir, 1),
            ],
            "runtime_seconds": time.perf_counter() - started,
        }
        save_json(result, args.output)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": args.output,
                    "checkpoints": [
                        value["path"] for value in result["checkpoints"]
                    ],
                    "runtime_seconds": result["runtime_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )
        return result
    finally:
        close_distributed_runtime(runtime)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.tiny_cpu_smoke_test:
        print(json.dumps(run_tiny_cpu_smoke(args.seed), indent=2))
        return
    if args.data_dry_run:
        print(
            json.dumps(
                inspect_prepared_data(
                    args.prepared_data,
                    args.batch_size,
                ),
                indent=2,
            )
        )
        return
    run_training(args)


if __name__ == "__main__":
    main()
