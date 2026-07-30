from __future__ import annotations

import argparse
import hashlib
import math
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
    close_distributed_runtime,
    init_distributed_runtime,
)
from hstu_kvcache.streaming.distributed import batch_target_count
from hstu_kvcache.streaming.sharded_edge import (
    ExternalEmbeddingHSTU,
    ShardedEdgeModelSpec,
    TrainableModuloRowShardedEmbedding,
    model_memory_estimate,
    sharded_edge_train_step,
    sparse_sgd,
)
from hstu_kvcache.utils import save_json

PROTOCOL = "evokv_design3_m1_qk_sharded_calibration_dev_v0"
DEFAULT_PREPARED_DATA = "data/processed/evokv_d3_m1_qk_entity_2560.npz"
DEFAULT_OUTPUT = "results/system/evokv_design3_m1/qk_entity_h1536_sharded_calibration_seed0.json"
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
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dense-lr", type=float, default=3e-4)
    parser.add_argument("--embedding-lr", type=float, default=3e-2)
    parser.add_argument("--negative-count", type=int, default=8)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=64)
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


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.batch_size < 1
        or args.negative_count < 1
        or args.ddp_bucket_cap_mb < 1
        or args.dense_lr <= 0
        or args.embedding_lr <= 0
    ):
        raise ValueError("invalid calibration argument")
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"calibration output exists; pass --force to replace: {output}")


def validate_prepared(
    plan,
    metadata: dict[str, object],
    spec: ShardedEdgeModelSpec,
) -> None:
    dataset = str(metadata.get("dataset", "")).lower().replace("_", "-")
    if (
        dataset != "tenrec-qk"
        or plan.num_items + 1 != spec.num_embeddings
        or int(metadata.get("fitted_items", 0)) + 1 != spec.num_embeddings
        or int(metadata.get("base_entity_items", 0)) != spec.num_items
        or plan.num_prediction_items != spec.num_prediction_items
        or int(metadata.get("primary_benchmark_users", 0)) != KV_RECORDS
    ):
        raise ValueError("prepared entity data differs from large M1")


def select_real_base_batches(
    plan,
    batch_size: int,
    world_size: int,
) -> list[dict[str, torch.Tensor]]:
    selected = []
    for batch in plan.iter_base_train_batches(
        batch_size,
        all_chunks=True,
        bucket_by_length=True,
        pad_to_max_seq_len=False,
    ):
        if batch_target_count(batch) > 0:
            selected.append(batch)
        if len(selected) == world_size:
            break
    if len(selected) != world_size:
        raise RuntimeError("calibration cannot find one positive base batch per rank")
    return selected


def dense_checksum(
    dense_model: DistributedDataParallel,
) -> torch.Tensor:
    values = torch.zeros(
        2,
        dtype=torch.float64,
        device=next(dense_model.parameters()).device,
    )
    for parameter in dense_model.module.parameters():
        value = parameter.detach()
        values[0] += value.double().sum()
        values[1] += value.double().square().sum()
    return values


def run(args: argparse.Namespace) -> dict[str, object] | None:
    runtime = init_distributed_runtime(
        args.device,
        args.distributed_backend,
    )
    try:
        if (
            runtime.world_size != 2
            or not runtime.initialized
            or runtime.device.type != "cuda"
            or args.distributed_backend != "nccl"
        ):
            raise RuntimeError("large M1 calibration requires two torchrun NCCL CUDA ranks")
        seed_everything(args.seed)
        plan, metadata = load_prepared_exposure_plan(
            args.prepared_data,
            max_seq_len=MAX_SEQ_LEN,
        )
        spec = model_spec(
            plan.num_prediction_items,
            plan.num_behaviors,
        )
        validate_prepared(plan, metadata, spec)
        plan.init_base()
        base_batches = select_real_base_batches(
            plan,
            args.batch_size,
            runtime.world_size,
        )
        batch = base_batches[runtime.rank]
        embedding_group = dist.new_group(ranks=list(range(runtime.world_size)))
        torch.cuda.synchronize(runtime.device)
        initialization_started = time.perf_counter()
        dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(runtime.device)
        dense_model = DistributedDataParallel(
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
        dense_optimizer = torch.optim.AdamW(
            dense_model.parameters(),
            lr=args.dense_lr,
            weight_decay=1e-4,
            foreach=False,
        )
        embedding_optimizer = sparse_sgd(
            embedding,
            args.embedding_lr,
        )
        torch.cuda.synchronize(runtime.device)
        initialization_seconds = time.perf_counter() - initialization_started
        initialized_allocated = torch.cuda.memory_allocated(runtime.device)
        initialized_reserved = torch.cuda.memory_reserved(runtime.device)
        torch.cuda.synchronize(runtime.device)
        warmup_started = time.perf_counter()
        warmup_loss, warmup_targets, warmup_global_targets = sharded_edge_train_step(
            dense_model,
            embedding,
            batch,
            dense_optimizer,
            embedding_optimizer,
            runtime.device,
            spec.num_prediction_items,
            args.negative_count,
            args.seed + 71,
        )
        torch.cuda.synchronize(runtime.device)
        warmup_seconds = time.perf_counter() - warmup_started
        torch.cuda.reset_peak_memory_stats(runtime.device)
        torch.cuda.synchronize(runtime.device)
        step_started = time.perf_counter()
        loss, local_targets, global_targets = sharded_edge_train_step(
            dense_model,
            embedding,
            batch,
            dense_optimizer,
            embedding_optimizer,
            runtime.device,
            spec.num_prediction_items,
            args.negative_count,
            args.seed + 73,
        )
        torch.cuda.synchronize(runtime.device)
        step_seconds = time.perf_counter() - step_started
        peak_allocated = torch.cuda.max_memory_allocated(runtime.device)
        peak_reserved = torch.cuda.max_memory_reserved(runtime.device)
        final_allocated = torch.cuda.memory_allocated(runtime.device)
        final_reserved = torch.cuda.memory_reserved(runtime.device)
        gradient = embedding.local_weight.grad
        sparse_gradient = gradient is not None and gradient.is_sparse
        sparse_gradient_rows = int(gradient.coalesce()._nnz()) if sparse_gradient else 0
        item_ids = batch["item_ids"]
        lengths = batch["lengths"].long()
        valid = torch.arange(item_ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        requested = item_ids[valid]
        offrank_input_tokens = int(
            ((requested > 0) & (requested.remainder(runtime.world_size) != runtime.rank)).sum()
        )
        checksum = dense_checksum(dense_model)
        gathered_checksums = [torch.empty_like(checksum) for _ in range(runtime.world_size)]
        dist.all_gather(gathered_checksums, checksum)
        checksum_equal = all(
            torch.equal(gathered_checksums[0], value) for value in gathered_checksums[1:]
        )
        local_record = {
            "rank": runtime.rank,
            "device": str(runtime.device),
            "initialization_seconds": initialization_seconds,
            "initialized_allocated_bytes": initialized_allocated,
            "initialized_reserved_bytes": initialized_reserved,
            "warmup_seconds": warmup_seconds,
            "warmup_loss": warmup_loss,
            "warmup_local_targets": warmup_targets,
            "warmup_global_targets": warmup_global_targets,
            "step_seconds": step_seconds,
            "loss": loss,
            "local_targets": local_targets,
            "global_targets": global_targets,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "final_allocated_bytes": final_allocated,
            "final_reserved_bytes": final_reserved,
            "embedding_gradient_is_sparse": sparse_gradient,
            "embedding_sparse_gradient_rows": (sparse_gradient_rows),
            "input_tokens": int(lengths.sum()),
            "offrank_input_tokens": offrank_input_tokens,
            "embedding_local_rows": embedding.local_rows,
            "embedding_local_bytes_fp32": (
                embedding.local_weight.numel() * embedding.local_weight.element_size()
            ),
            "dense_checksum": checksum.tolist(),
        }
        gathered: list[dict[str, object] | None] = [None] * runtime.world_size
        dist.all_gather_object(gathered, local_record)
        records = [value for value in gathered if value is not None]
        target_sum = sum(int(value["local_targets"]) for value in records)
        nccl_correct = (
            len(records) == runtime.world_size
            and all(int(value["global_targets"]) == target_sum for value in records)
            and all(bool(value["embedding_gradient_is_sparse"]) for value in records)
            and all(int(value["offrank_input_tokens"]) > 0 for value in records)
            and checksum_equal
            and dist.get_backend() == "nccl"
            and dist.get_backend(embedding_group) == "nccl"
        )
        if not runtime.is_primary:
            return None
        result = {
            "protocol": PROTOCOL,
            "status": "complete",
            "scientific_result": False,
            "formal_training": False,
            "artifact_role": ("two_gpu_large_model_capacity_and_step_calibration"),
            "scope": (
                "one real base batch per rank, one repeated-batch warmup, "
                "and one measured optimizer step; no checkpoint"
            ),
            "prepared_data": {
                "path": args.prepared_data,
                "bytes": Path(args.prepared_data).stat().st_size,
                "sha256": artifact_sha256(args.prepared_data),
            },
            "model": asdict(spec),
            "memory_estimate": model_memory_estimate(
                spec,
                runtime.world_size,
                KV_RECORDS,
            ),
            "calibration": {
                "world_size": runtime.world_size,
                "backend": args.distributed_backend,
                "batch_size_per_rank": args.batch_size,
                "negative_count": args.negative_count,
                "seed": args.seed,
                "records": records,
            },
            "correctness": {
                "target_sum": target_sum,
                "dense_replica_checksums_equal": checksum_equal,
                "embedding_gradients_sparse": all(
                    bool(value["embedding_gradient_is_sparse"]) for value in records
                ),
                "both_ranks_issue_offrank_input_requests": all(
                    int(value["offrank_input_tokens"]) > 0 for value in records
                ),
                "default_process_group_backend": dist.get_backend(),
                "embedding_process_group_backend": dist.get_backend(embedding_group),
                "nccl_path_correct": nccl_correct,
            },
        }
        save_json(result, args.output)
        return result
    finally:
        close_distributed_runtime(runtime)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    result = run(args)
    if result is not None:
        if not math.isfinite(
            max(float(value["step_seconds"]) for value in result["calibration"]["records"])
        ):
            raise RuntimeError("calibration produced non-finite timing")
        print(args.output)


if __name__ == "__main__":
    main()
