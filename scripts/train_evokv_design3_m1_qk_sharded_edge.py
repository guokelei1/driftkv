from __future__ import annotations

import argparse
import hashlib
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
from hstu_kvcache.streaming import (
    DistributedRuntime,
    batch_coverage,
    close_distributed_runtime,
    init_distributed_runtime,
    primary_log,
    shard_train_batches,
)
from hstu_kvcache.streaming.sharded_edge import (
    ExternalEmbeddingHSTU,
    FixedHeldoutBatch,
    ShardedEdgeModelSpec,
    TrainableModuloRowShardedEmbedding,
    evaluate_fixed_heldout,
    make_fixed_heldout_batch,
    model_memory_estimate,
    save_sharded_edge_checkpoint,
    sharded_edge_train_step,
    sparse_sgd,
)
from hstu_kvcache.utils import save_json

PROTOCOL = "evokv_design3_m1_qk_sharded_two_version_training_dev_v0"
DEFAULT_PREPARED_DATA = "data/processed/evokv_d3_m1_qk_entity_2560.npz"
DEFAULT_CHECKPOINT_DIR = "checkpoints/evokv_design3_m1_qk_entity_h1536/seed0"
DEFAULT_OUTPUT = (
    "results/system/evokv_design3_m1/qk_entity_h1536_sharded_two_version_training_seed0.json"
)
PHYSICAL_EMBEDDING_ROWS = 2_859_836
HIDDEN_SIZE = 1536
NUM_LAYERS = 24
NUM_HEADS = 24
HEAD_DIM = 64
MAX_SEQ_LEN = 512
KV_RECORDS = 2048


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED_DATA)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-epochs", type=int, default=1)
    parser.add_argument("--base-dense-lr", type=float, default=3e-4)
    parser.add_argument("--base-embedding-lr", type=float, default=3e-2)
    parser.add_argument("--stream-epochs", type=int, default=1)
    parser.add_argument("--stream-dense-lr", type=float, default=1e-4)
    parser.add_argument("--stream-embedding-lr", type=float, default=1e-2)
    parser.add_argument("--train-negatives", type=int, default=8)
    parser.add_argument("--heldout-negatives", type=int, default=99)
    parser.add_argument("--heldout-seed", type=int, default=20260730)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=64)
    parser.add_argument("--data-dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    positive_integers = {
        "batch_size": args.batch_size,
        "base_epochs": args.base_epochs,
        "stream_epochs": args.stream_epochs,
        "train_negatives": args.train_negatives,
        "heldout_negatives": args.heldout_negatives,
        "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb,
    }
    invalid = [name for name, value in positive_integers.items() if value < 1]
    if invalid:
        raise ValueError(f"positive values required for: {', '.join(invalid)}")
    learning_rates = (
        args.base_dense_lr,
        args.base_embedding_lr,
        args.stream_dense_lr,
        args.stream_embedding_lr,
    )
    if any(value <= 0 for value in learning_rates):
        raise ValueError("all learning rates must be positive")


def model_spec(
    num_prediction_items: int,
    num_behaviors: int,
) -> ShardedEdgeModelSpec:
    return ShardedEdgeModelSpec(
        num_embeddings=PHYSICAL_EMBEDDING_ROWS,
        num_prediction_items=num_prediction_items,
        num_behaviors=num_behaviors,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
    )


def validate_prepared_metadata(
    metadata: dict[str, object],
    spec: ShardedEdgeModelSpec,
) -> None:
    dataset = str(metadata.get("dataset", "")).lower().replace("_", "-")
    if dataset != "tenrec-qk":
        raise ValueError("large M1 training requires Tenrec QK")
    if int(metadata.get("fitted_items", 0)) + 1 != spec.num_embeddings:
        raise ValueError("prepared entity rows differ from large M1")
    if int(metadata.get("base_entity_items", 0)) not in {
        0,
        spec.num_items,
    }:
        raise ValueError("prepared base entity catalog differs")
    if int(metadata.get("num_prediction_items", 0)) != (spec.num_prediction_items):
        raise ValueError("prepared prediction catalog differs")
    if int(metadata.get("window_count", 0)) < 2:
        raise ValueError("large M1 requires window_0 training and window_1 heldout")
    if int(metadata.get("history_length", MAX_SEQ_LEN)) != MAX_SEQ_LEN:
        raise ValueError("prepared history length differs from model")
    if int(metadata.get("primary_benchmark_users", KV_RECORDS)) != (KV_RECORDS):
        raise ValueError("prepared benchmark cohort differs from 2048")


def load_plan(path: str | Path):
    plan, metadata = load_prepared_exposure_plan(
        path,
        max_seq_len=MAX_SEQ_LEN,
    )
    spec = model_spec(
        plan.num_prediction_items,
        plan.num_behaviors,
    )
    validate_prepared_metadata(metadata, spec)
    if plan.num_items != spec.num_items:
        raise ValueError("prepared plan item count differs from entity table")
    if plan.base_dates != ["base"] or plan.stream_dates[:2] != [
        "window_0",
        "window_1",
    ]:
        raise ValueError("prepared model edge boundary differs")
    return plan, metadata, spec


def inspect_prepared_data(
    path: str | Path,
    batch_size: int,
) -> dict[str, object]:
    plan, metadata, spec = load_plan(path)
    plan.init_base()
    base_batches = list(
        plan.iter_base_train_batches(
            batch_size,
            all_chunks=True,
            bucket_by_length=True,
            pad_to_max_seq_len=False,
        )
    )
    base = batch_coverage(base_batches)
    plan.ingest_day("window_0")
    update_batches = list(
        plan.iter_train_batches(
            "window_0",
            batch_size,
            all_chunks=True,
            bucket_by_length=True,
            pad_to_max_seq_len=False,
        )
    )
    update = batch_coverage(update_batches)
    plan.ingest_day("window_1")
    heldout_batches = list(
        plan.iter_train_batches(
            "window_1",
            batch_size,
            all_chunks=True,
            bucket_by_length=True,
            pad_to_max_seq_len=False,
        )
    )
    heldout = batch_coverage(heldout_batches)
    return {
        "protocol": PROTOCOL,
        "mode": "data_dry_run",
        "scientific_result": False,
        "prepared_data": {
            "path": str(path),
            "sha256": artifact_sha256(path),
            "metadata": metadata,
        },
        "spec": asdict(spec),
        "memory": model_memory_estimate(spec, 2, KV_RECORDS),
        "training": {
            "theta0_base": base,
            "theta1_update": update,
            "fixed_heldout": heldout,
        },
    }


def reduce_epoch(
    local_loss_sum: float,
    local_targets: int,
    runtime: DistributedRuntime,
) -> tuple[float, list[int] | None]:
    totals = torch.tensor(
        [local_loss_sum, local_targets],
        dtype=torch.float64,
        device=runtime.device,
    )
    if runtime.initialized:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        per_rank = [None] * runtime.world_size
        dist.all_gather_object(per_rank, local_targets)
    else:
        per_rank = [local_targets]
    if totals[1].item() <= 0:
        raise RuntimeError("training epoch has no positive targets")
    return float((totals[0] / totals[1]).item()), per_rank


def train_epoch(
    dense_model: torch.nn.Module,
    embedding: TrainableModuloRowShardedEmbedding,
    dense_optimizer: torch.optim.Optimizer,
    embedding_optimizer: torch.optim.Optimizer,
    local_batches: list[tuple[dict[str, torch.Tensor], float]],
    runtime: DistributedRuntime,
    num_prediction_items: int,
    negative_count: int,
    seed_base: int,
) -> dict[str, object]:
    local_loss_sum = 0.0
    local_targets = 0
    global_targets_per_step = []
    for step, (batch, _) in enumerate(local_batches):
        loss, targets, global_targets = sharded_edge_train_step(
            dense_model,
            embedding,
            batch,
            dense_optimizer,
            embedding_optimizer,
            runtime.device,
            num_prediction_items,
            negative_count,
            seed_base + step * runtime.world_size + runtime.rank,
        )
        local_loss_sum += loss * targets
        local_targets += targets
        global_targets_per_step.append(global_targets)
    loss, per_rank_targets = reduce_epoch(
        local_loss_sum,
        local_targets,
        runtime,
    )
    return {
        "loss": loss,
        "steps_per_rank": len(local_batches),
        "targets_per_rank": per_rank_targets,
        "global_targets": sum(per_rank_targets or []),
        "minimum_step_global_targets": min(global_targets_per_step),
        "maximum_step_global_targets": max(global_targets_per_step),
        "normalization": (
            "global target mean; dense DDP scale includes world size and "
            "owner-local sparse embedding gradient divides world size"
        ),
    }


def prepare_fixed_heldout(
    local_batches: list[tuple[dict[str, torch.Tensor], float]],
    spec: ShardedEdgeModelSpec,
    runtime: DistributedRuntime,
    negative_count: int,
    heldout_seed: int,
) -> list[FixedHeldoutBatch]:
    return [
        make_fixed_heldout_batch(
            batch,
            spec.num_prediction_items,
            negative_count,
            (heldout_seed + index * runtime.world_size + runtime.rank),
        )
        for index, (batch, _) in enumerate(local_batches)
    ]


def materialize_fixed_heldout_batches(
    plan,
    batch_size: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return list(
            plan.iter_train_batches(
                "window_1",
                batch_size,
                all_chunks=True,
                bucket_by_length=True,
                pad_to_max_seq_len=False,
            )
        )
    finally:
        np.random.set_state(state)


def heldout_candidate_hashes(
    heldout: list[FixedHeldoutBatch],
    runtime: DistributedRuntime,
) -> list[str]:
    digest = hashlib.sha256()
    for value in heldout:
        tensor = value.candidates.contiguous()
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    local = digest.hexdigest()
    if runtime.initialized:
        gathered: list[str | None] = [None] * runtime.world_size
        dist.all_gather_object(gathered, local)
        return [str(value) for value in gathered]
    return [local]


def validate_output_targets(args: argparse.Namespace) -> None:
    targets = (
        Path(args.output),
        Path(args.checkpoint_dir) / "theta_0" / "manifest.json",
        Path(args.checkpoint_dir) / "theta_1" / "manifest.json",
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"large M1 output exists; pass --force to replace: {existing}")


def run_training(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    validate_output_targets(args)
    runtime = init_distributed_runtime(
        args.device,
        args.distributed_backend,
    )
    started = time.perf_counter()
    try:
        if runtime.world_size != 2 or not runtime.initialized or runtime.device.type != "cuda":
            raise RuntimeError("large M1 formal training requires two torchrun CUDA ranks")
        seed_everything(args.seed)
        torch.set_float32_matmul_precision("high")
        plan, metadata, spec = load_plan(args.prepared_data)
        plan.init_base()
        embedding_group = dist.new_group(ranks=list(range(runtime.world_size)))
        dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(runtime.device)
        dense_model: torch.nn.Module = DistributedDataParallel(
            dense,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
        )
        embedding = TrainableModuloRowShardedEmbedding.initialize(
            spec.num_embeddings,
            spec.hidden_size,
            runtime.rank,
            runtime.world_size,
            runtime.device,
            args.seed,
            process_group=embedding_group,
        )
        primary_log(
            runtime,
            (
                f"rows={spec.num_embeddings:,} hidden={spec.hidden_size} "
                f"layers={spec.num_layers} local_rows="
                f"{embedding.local_rows:,}"
            ),
        )
        base_batches = list(
            plan.iter_base_train_batches(
                args.batch_size,
                all_chunks=True,
                bucket_by_length=True,
                pad_to_max_seq_len=False,
            )
        )
        local_base, base_coverage = shard_train_batches(
            base_batches,
            runtime,
        )
        del base_batches
        base_dense_optimizer = torch.optim.AdamW(
            dense_model.parameters(),
            lr=args.base_dense_lr,
            weight_decay=1e-4,
            foreach=False,
        )
        base_embedding_optimizer = sparse_sgd(
            embedding,
            args.base_embedding_lr,
        )
        base_epochs = []
        for epoch in range(args.base_epochs):
            value = train_epoch(
                dense_model,
                embedding,
                base_dense_optimizer,
                base_embedding_optimizer,
                local_base,
                runtime,
                spec.num_prediction_items,
                args.train_negatives,
                args.seed + epoch * 10_000_019,
            )
            value["epoch"] = epoch + 1
            base_epochs.append(value)
            primary_log(
                runtime,
                f"base_epoch={epoch + 1} loss={value['loss']:.6f}",
            )
        del base_dense_optimizer, base_embedding_optimizer
        torch.cuda.empty_cache()
        theta0_checkpoint = save_sharded_edge_checkpoint(
            args.checkpoint_dir,
            0,
            spec,
            dense_model,
            embedding,
        )
        plan.ingest_day("window_0")
        update_batches = list(
            plan.iter_train_batches(
                "window_0",
                args.batch_size,
                all_chunks=True,
                bucket_by_length=True,
                pad_to_max_seq_len=False,
            )
        )
        local_update, update_coverage = shard_train_batches(
            update_batches,
            runtime,
        )
        del update_batches
        plan.ingest_day("window_1")
        heldout_batches = materialize_fixed_heldout_batches(
            plan,
            args.batch_size,
            args.heldout_seed,
        )
        local_heldout, heldout_coverage = shard_train_batches(
            heldout_batches,
            runtime,
        )
        del heldout_batches
        fixed_heldout = prepare_fixed_heldout(
            local_heldout,
            spec,
            runtime,
            args.heldout_negatives,
            args.heldout_seed,
        )
        candidate_hashes = heldout_candidate_hashes(
            fixed_heldout,
            runtime,
        )
        theta0_heldout = evaluate_fixed_heldout(
            dense_model,
            embedding,
            fixed_heldout,
            runtime.device,
            process_group=embedding_group,
        )
        stream_dense_optimizer = torch.optim.AdamW(
            dense_model.parameters(),
            lr=args.stream_dense_lr,
            weight_decay=1e-4,
            foreach=False,
        )
        stream_embedding_optimizer = sparse_sgd(
            embedding,
            args.stream_embedding_lr,
        )
        stream_epochs = []
        for epoch in range(args.stream_epochs):
            value = train_epoch(
                dense_model,
                embedding,
                stream_dense_optimizer,
                stream_embedding_optimizer,
                local_update,
                runtime,
                spec.num_prediction_items,
                args.train_negatives,
                args.seed + 100_000_007 + epoch * 10_000_019,
            )
            value["epoch"] = epoch + 1
            stream_epochs.append(value)
            primary_log(
                runtime,
                f"window_0_epoch={epoch + 1} loss={value['loss']:.6f}",
            )
        theta1_heldout = evaluate_fixed_heldout(
            dense_model,
            embedding,
            fixed_heldout,
            runtime.device,
            process_group=embedding_group,
        )
        theta1_checkpoint = save_sharded_edge_checkpoint(
            args.checkpoint_dir,
            1,
            spec,
            dense_model,
            embedding,
        )
        ndcg_delta = float(theta1_heldout["ndcg_at_10"]) - float(theta0_heldout["ndcg_at_10"])
        if not runtime.is_primary:
            return None
        result = {
            "protocol": PROTOCOL,
            "status": "complete",
            "scientific_result": False,
            "formal_design3": False,
            "artifact_role": ("large_qk_two_version_row_sharded_training_edge"),
            "prepared_data": {
                "path": args.prepared_data,
                "sha256": artifact_sha256(args.prepared_data),
                "metadata": metadata,
            },
            "model": {
                "spec": asdict(spec),
                "dense_config": asdict(spec.hstu_config()),
                "embedding_layout": "modulo_row_sharded",
                "embedding_optimizer": ("owner-local sparse SGD without momentum"),
                "dense_optimizer": "replicated AdamW under dense-only DDP",
            },
            "memory": model_memory_estimate(spec, 2, KV_RECORDS),
            "execution": {
                "world_size": runtime.world_size,
                "backend": args.distributed_backend,
                "visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES",
                    "",
                ),
                "batch_size_per_rank": args.batch_size,
                "precision": "float32 model and embedding",
                "seed": args.seed,
            },
            "coverage": {
                "theta0_base": base_coverage,
                "theta1_update_window_0": update_coverage,
                "fixed_heldout_window_1": heldout_coverage,
            },
            "training": {
                "theta0_base": base_epochs,
                "theta1_update_window_0": stream_epochs,
            },
            "fixed_heldout": {
                "window": "window_1",
                "negative_count": args.heldout_negatives,
                "candidate_count": args.heldout_negatives + 1,
                "seed": args.heldout_seed,
                "candidate_sha256_per_rank": candidate_hashes,
                "primary_metric": "ndcg_at_10",
                "theta0": theta0_heldout,
                "theta1": theta1_heldout,
                "theta1_minus_theta0_ndcg_at_10": ndcg_delta,
                "positive_recommendation_signal": ndcg_delta > 0,
            },
            "checkpoints": {
                "theta0": theta0_checkpoint,
                "theta1": theta1_checkpoint,
            },
            "runtime_seconds": time.perf_counter() - started,
        }
        save_json(result, args.output)
        return result
    finally:
        close_distributed_runtime(runtime)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.data_dry_run:
        result = inspect_prepared_data(
            args.prepared_data,
            args.batch_size,
        )
        save_json(result, args.output)
        print(args.output)
        return
    result = run_training(args)
    if result is not None:
        print(args.output)


if __name__ == "__main__":
    main()
