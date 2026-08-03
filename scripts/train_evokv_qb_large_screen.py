from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.streaming.qb_multifield_training import (
    PROTOCOL,
    QBEvaluationBatch,
    build_base_batches,
    build_role_batches,
    cooccurrence_negatives,
    load_qb_large_corpus,
    prefix_cache,
    prepare_evaluation_batches,
    score_sums,
    scores_with_cache,
    summarize_score_sums,
    train_cooccurrence_step,
    train_multifield_step,
)
from hstu_kvcache.streaming.sharded_edge import (
    ExternalEmbeddingHSTU,
    modulo_local_rows,
)
from hstu_kvcache.streaming.xp_projected_edge import (
    OptimizerActiveRowTracker,
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
    load_xp_projected_checkpoint,
    save_xp_projected_checkpoint,
    sparse_embedding_sgd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--target-checkpoint-root", type=Path)
    parser.add_argument("--resume-base-root", type=Path)
    parser.add_argument("--resume-chain-result", type=Path)
    parser.add_argument("--resume-version", type=int, choices=(2, 3))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=1536)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--embedding-width", type=int)
    parser.add_argument("--base-epochs", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--continuation-update-epochs", type=int)
    parser.add_argument("--continuation-train-start", type=int)
    parser.add_argument("--continuation-train-end", type=int)
    parser.add_argument("--continuation-evaluation-end", type=int)
    parser.add_argument("--continuation-policy-name")
    parser.add_argument("--continuation-dense-lr", type=float)
    parser.add_argument("--continuation-projection-lr", type=float)
    parser.add_argument("--continuation-embedding-lr", type=float)
    parser.add_argument("--continuation-train-negatives", type=int)
    parser.add_argument("--continuation-current-extra-passes", type=int, default=0)
    parser.add_argument("--continuation-extra-start", type=int)
    parser.add_argument("--continuation-reset-adamw", action="store_true")
    parser.add_argument("--stop-version", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--batch-size-per-rank", type=int, default=1)
    parser.add_argument("--contrastive-batch-size", type=int, default=4096)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--base-dense-lr", type=float, default=1e-4)
    parser.add_argument("--base-projection-lr", type=float, default=1e-4)
    parser.add_argument("--base-embedding-lr", type=float, default=1e-3)
    parser.add_argument("--update-dense-lr", type=float, default=1.5e-5)
    parser.add_argument("--update-projection-lr", type=float, default=1.5e-5)
    parser.add_argument("--update-embedding-lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-negatives", type=int, default=8)
    parser.add_argument("--evaluation-negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--maximum-contrastive-pairs", type=int, default=0)
    parser.add_argument("--maximum-base-steps", type=int, default=0)
    parser.add_argument("--maximum-role-steps", type=int, default=0)
    parser.add_argument("--development-canary", action="store_true")
    parser.add_argument("--hbm-canary-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.hidden_size,
        args.layers,
        args.heads,
        args.head_dim,
        args.max_seq_len,
        args.base_epochs,
        args.update_epochs,
        args.batch_size_per_rank,
        args.contrastive_batch_size,
        args.train_negatives,
        args.evaluation_negatives,
        args.progress_every,
    )
    rates = (
        args.base_dense_lr,
        args.base_projection_lr,
        args.base_embedding_lr,
        args.update_dense_lr,
        args.update_projection_lr,
        args.update_embedding_lr,
        args.contrastive_temperature,
    )
    if (
        min(positive) < 1
        or min(rates) <= 0
        or args.heads * args.head_dim != args.hidden_size
        or args.weight_decay < 0
        or min(
            args.maximum_contrastive_pairs,
            args.maximum_base_steps,
            args.maximum_role_steps,
        )
        < 0
        or (args.hbm_canary_only and args.resume_base_root is not None)
        or (args.hbm_canary_only and args.resume_chain_result is not None)
        or (args.resume_base_root is not None and args.resume_chain_result is not None)
        or ((args.resume_chain_result is None) != (args.resume_version is None))
        or (args.target_checkpoint_root is not None and args.resume_chain_result is None)
        or (
            any(
                value is not None
                for value in (
                    args.continuation_update_epochs,
                    args.continuation_train_start,
                    args.continuation_train_end,
                    args.continuation_evaluation_end,
                    args.continuation_policy_name,
                    args.continuation_dense_lr,
                    args.continuation_projection_lr,
                    args.continuation_embedding_lr,
                    args.continuation_train_negatives,
                    args.continuation_extra_start,
                )
            )
            or args.continuation_current_extra_passes != 0
            or args.continuation_reset_adamw
        )
        and (
            args.resume_version != 3
            or args.stop_version != 4
            or args.target_checkpoint_root is None
            or args.continuation_update_epochs is None
            or args.continuation_train_start is None
            or not args.continuation_policy_name
        )
        or (
            any(
                value is not None
                for value in (
                    args.continuation_dense_lr,
                    args.continuation_projection_lr,
                    args.continuation_embedding_lr,
                )
            )
            and (
                args.continuation_dense_lr is None
                or args.continuation_projection_lr is None
                or args.continuation_embedding_lr is None
            )
        )
        or (
            args.continuation_update_epochs is not None
            and args.continuation_update_epochs < 1
        )
        or (
            args.continuation_train_start is not None
            and not 0
            <= args.continuation_train_start
            < continuation_extents(args)[0]
        )
        or (
            (args.continuation_train_end is None)
            != (args.continuation_evaluation_end is None)
        )
        or (
            args.continuation_policy_name is not None
            and not 88
            < continuation_extents(args)[0]
            < continuation_extents(args)[1]
            <= args.max_seq_len
        )
        or args.continuation_current_extra_passes < 0
        or (
            args.continuation_extra_start is not None
            and (
                args.continuation_current_extra_passes < 1
                or args.continuation_train_start is None
                or not args.continuation_train_start
                <= args.continuation_extra_start
                < continuation_extents(args)[0]
            )
        )
        or (
            args.continuation_train_negatives is not None
            and args.continuation_train_negatives < 1
        )
        or any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in (
                args.continuation_dense_lr,
                args.continuation_projection_lr,
                args.continuation_embedding_lr,
            )
        )
        or (
            args.continuation_policy_name is not None
            and sum(continuation_learning_rates(args).values()) <= 0
        )
        or (
            args.resume_version is not None
            and args.stop_version <= args.resume_version
        )
    ):
        raise ValueError("QB large screen arguments are invalid")
    checked_outputs = (args.output,) if args.resume_chain_result is not None else (
        args.output,
        args.checkpoint_root,
    )
    existing = [str(path) for path in checked_outputs if path.exists()]
    if args.target_checkpoint_root is not None and args.target_checkpoint_root.exists():
        existing.append(str(args.target_checkpoint_root))
    if existing:
        raise FileExistsError(f"QB large screen outputs exist: {existing}")
    if args.resume_base_root is not None:
        required = (
            args.resume_base_root / "theta_0" / "manifest.json",
            args.resume_base_root / "optimizer_after_theta_0.pt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"QB base-resume artifacts are missing: {missing}")
    if args.resume_chain_result is not None:
        if not args.resume_chain_result.is_file() or not args.checkpoint_root.is_dir():
            raise FileNotFoundError("QB chain-resume source is missing")
        target_root = args.target_checkpoint_root or args.checkpoint_root
        targets = tuple(
            target_root / f"theta_{version}"
            for version in range(args.resume_version + 1, args.stop_version + 1)
        )
        conflicts = [str(path) for path in targets if path.exists()]
        if conflicts:
            raise FileExistsError(f"QB continuation targets exist: {conflicts}")


def init_distributed() -> tuple[int, int, torch.device, str]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2 or not torch.cuda.is_available():
        raise RuntimeError("QB large screen requires two CUDA ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    if rank != dist.get_rank() or world_size != dist.get_world_size():
        raise RuntimeError("QB distributed environment differs")
    return rank, world_size, device, "nccl"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tensor_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def synchronize_seconds(device: torch.device, started: float) -> float:
    torch.cuda.synchronize(device)
    local = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return float(local.item())


def global_active_rows(
    tracker: OptimizerActiveRowTracker,
    device: torch.device,
) -> int:
    value = torch.tensor(tracker.local_active_count, dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return int(value.item())


def hbm_report(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
    }


def set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def training_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "base_epochs": args.base_epochs,
        "update_epochs": args.update_epochs,
        "stop_version": args.stop_version,
        "batch_size_per_rank": args.batch_size_per_rank,
        "contrastive_batch_size_per_rank": args.contrastive_batch_size,
        "contrastive_temperature": args.contrastive_temperature,
        "base_learning_rates": {
            "dense": args.base_dense_lr,
            "projection": args.base_projection_lr,
            "embedding": args.base_embedding_lr,
        },
        "update_learning_rates": {
            "dense": args.update_dense_lr,
            "projection": args.update_projection_lr,
            "embedding": args.update_embedding_lr,
        },
        "weight_decay": args.weight_decay,
        "train_negatives": args.train_negatives,
        "evaluation_negatives": args.evaluation_negatives,
        "maximum_contrastive_pairs": args.maximum_contrastive_pairs,
        "maximum_base_steps": args.maximum_base_steps,
        "maximum_role_steps": args.maximum_role_steps,
    }


def base_training_configuration(args: argparse.Namespace) -> dict[str, object]:
    value = training_configuration(args)
    return {
        key: value[key]
        for key in (
            "seed",
            "base_epochs",
            "batch_size_per_rank",
            "contrastive_batch_size_per_rank",
            "contrastive_temperature",
            "base_learning_rates",
            "weight_decay",
            "train_negatives",
            "maximum_contrastive_pairs",
            "maximum_base_steps",
        )
    }


def continuation_training_configuration(args: argparse.Namespace) -> dict[str, object]:
    value = training_configuration(args)
    value.pop("stop_version")
    return value


def continuation_learning_rates(args: argparse.Namespace) -> dict[str, float]:
    return {
        "dense": (
            args.update_dense_lr
            if args.continuation_dense_lr is None
            else args.continuation_dense_lr
        ),
        "projection": (
            args.update_projection_lr
            if args.continuation_projection_lr is None
            else args.continuation_projection_lr
        ),
        "embedding": (
            args.update_embedding_lr
            if args.continuation_embedding_lr is None
            else args.continuation_embedding_lr
        ),
    }


def continuation_extents(args: argparse.Namespace) -> tuple[int, int]:
    return (
        96 if args.continuation_train_end is None else args.continuation_train_end,
        (
            104
            if args.continuation_evaluation_end is None
            else args.continuation_evaluation_end
        ),
    )


def continuation_policy(args: argparse.Namespace) -> dict[str, object] | None:
    if args.continuation_policy_name is None:
        return None
    train_end, evaluation_end = continuation_extents(args)
    return {
        "name": args.continuation_policy_name,
        "source_version": args.resume_version,
        "target_version": args.stop_version,
        "train_start": args.continuation_train_start,
        "train_end": train_end,
        "evaluation_end": evaluation_end,
        "epochs": args.continuation_update_epochs,
        "current_window_extra_passes": args.continuation_current_extra_passes,
        "extra_pass_start": (
            args.continuation_extra_start
            if args.continuation_current_extra_passes
            else None
        ),
        "learning_rates": continuation_learning_rates(args),
        "train_negatives": (
            args.train_negatives
            if args.continuation_train_negatives is None
            else args.continuation_train_negatives
        ),
        "reset_adamw": args.continuation_reset_adamw,
    }


def continuation_corpus_compatible(
    source_result: dict[str, object],
    corpus,
    active_policy: dict[str, object] | None,
) -> bool:
    source = source_result.get("corpus", {})
    if source.get("content_sha256") == corpus.content_sha256:
        return True
    metadata = corpus.metadata
    return bool(
        active_policy is not None
        and metadata.get("parent_corpus_content_sha256") == source.get("content_sha256")
        and metadata.get("parent_corpus_file_sha256") == source.get("file_sha256")
        and metadata.get("roles") == source.get("roles")
        and int(metadata.get("parent_required_horizon", -1))
        < int(metadata.get("required_horizon", -1))
    )


def run_contrastive(
    corpus,
    embedding,
    tracker,
    projection_optimizer,
    embedding_optimizer,
    args,
    rank,
    world_size,
    device,
) -> dict[str, object]:
    anchors = corpus.arrays["cooccurrence_anchor_row"]
    positives = corpus.arrays["cooccurrence_positive_row"]
    negatives = cooccurrence_negatives(corpus)
    if args.maximum_contrastive_pairs:
        limit = min(args.maximum_contrastive_pairs, len(anchors))
        anchors = anchors[:limit]
        positives = positives[:limit]
        negatives = negatives[:limit]
    negative_hash = hashlib.sha256(np.ascontiguousarray(negatives).view(np.uint8)).hexdigest()
    anchors = anchors[rank::world_size]
    positives = positives[rank::world_size]
    negatives = negatives[rank::world_size]
    local_steps = math.ceil(len(anchors) / args.contrastive_batch_size)
    step_tensor = torch.tensor(local_steps, dtype=torch.int64, device=device)
    dist.all_reduce(step_tensor, op=dist.ReduceOp.MAX)
    steps = int(step_tensor.item())
    if steps != local_steps:
        raise ValueError("QB contrastive shards have unequal step counts")
    local_loss = 0.0
    local_pairs = 0
    started = time.perf_counter()
    for step, start in enumerate(range(0, len(anchors), args.contrastive_batch_size), start=1):
        stop = min(start + args.contrastive_batch_size, len(anchors))
        loss_sum, _ = train_cooccurrence_step(
            embedding,
            tracker,
            anchors[start:stop],
            positives[start:stop],
            negatives[start:stop],
            projection_optimizer,
            embedding_optimizer,
            device=device,
            temperature=args.contrastive_temperature,
        )
        local_loss += loss_sum
        local_pairs += stop - start
        if rank == 0 and (step == 1 or step == steps or step % args.progress_every == 0):
            print(
                f"phase=qb_contrastive step={step}/{steps} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    totals = torch.tensor([local_loss, local_pairs], dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    return {
        "global_pairs": int(totals[1].item()),
        "steps_per_rank": steps,
        "batch_size_per_rank": args.contrastive_batch_size,
        "mean_loss": float((totals[0] / totals[1]).item()),
        "temperature": args.contrastive_temperature,
        "wall_seconds": synchronize_seconds(device, started),
        "negative_policy": "cross-user deterministic cyclic assignment",
        "negative_rows_sha256": negative_hash,
    }


def run_training(
    phase,
    batches,
    epochs,
    dense,
    embedding,
    tracker,
    optimizers,
    args,
    rank,
    world_size,
    device,
    seed_offset,
    negative_count=None,
) -> dict[str, object]:
    dense_optimizer, projection_optimizer, embedding_optimizer = optimizers
    local_loss = 0.0
    local_targets = 0
    global_targets_seen = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        for step, batch in enumerate(batches, start=1):
            loss, targets, global_targets = train_multifield_step(
                dense,
                embedding,
                tracker,
                batch,
                dense_optimizer,
                projection_optimizer,
                embedding_optimizer,
                device=device,
                num_prediction_items=corpus_num_prediction_items(batch, args),
                negative_count=(args.train_negatives if negative_count is None else negative_count),
                negative_seed=(
                    args.seed + seed_offset + epoch * 10_000_019 + step * world_size + rank
                ),
            )
            local_loss += loss * targets
            local_targets += targets
            global_targets_seen += global_targets
            if rank == 0 and (step == 1 or step == len(batches) or step % args.progress_every == 0):
                print(
                    f"phase={phase} epoch={epoch + 1}/{epochs} "
                    f"step={step}/{len(batches)} loss={loss:.6f} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    totals = torch.tensor([local_loss, local_targets], dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    return {
        "phase": phase,
        "epochs": epochs,
        "steps_per_rank_per_epoch": len(batches),
        "global_positive_targets": int(totals[1].item()),
        "global_target_events_seen_across_steps": global_targets_seen,
        "global_target_mean_loss": float((totals[0] / totals[1]).item()),
        "negative_count": args.train_negatives if negative_count is None else negative_count,
        "wall_seconds": synchronize_seconds(device, started),
        "numerical_stability_passed": bool(torch.isfinite(totals).all().item()),
    }


def corpus_num_prediction_items(batch: dict[str, torch.Tensor], args: argparse.Namespace) -> int:
    value = getattr(args, "_num_prediction_items", None)
    if value is None or batch["target_item_ids"].ndim != 2:
        raise ValueError("QB prediction catalog binding is absent")
    return int(value)


def make_evaluations(
    corpus,
    role,
    history_end,
    evaluation_end,
    args,
    rank,
    world_size,
) -> tuple[list[QBEvaluationBatch], dict[str, object]]:
    batches, coverage = build_role_batches(
        corpus,
        role,
        width=evaluation_end,
        train_start=history_end,
        train_end=evaluation_end,
        batch_size_per_rank=args.batch_size_per_rank,
        rank=rank,
        world_size=world_size,
        maximum_steps=args.maximum_role_steps,
    )
    heldout, candidate_hash = prepare_evaluation_batches(
        batches,
        num_prediction_items=corpus.catalog.num_prediction_items,
        negative_count=args.evaluation_negatives,
        seed=args.seed + history_end * 1_000_003 + ROLE_SEED[role],
        rank=rank,
        world_size=world_size,
    )
    return heldout, {**coverage, "candidate_sha256": candidate_hash}


ROLE_SEED = {"tuning": 17, "qualification": 31}


@torch.no_grad()
def source_evaluation(
    dense,
    embedding,
    heldout,
    history_end,
    role,
    rank,
    device,
) -> tuple[list, torch.Tensor]:
    caches = []
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    started = time.perf_counter()
    for index, value in enumerate(heldout, start=1):
        cache = prefix_cache(dense, embedding, value.batch, history_end, device)
        scores = scores_with_cache(dense, embedding, value, cache, history_end, device)
        totals += score_sums(scores)
        caches.append(cache)
        if rank == 0 and (index == 1 or index == len(heldout) or index % 100 == 0):
            print(
                f"phase=qb_source_quality role={role} "
                f"batch={index}/{len(heldout)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return caches, totals


@torch.no_grad()
def target_evaluation(
    dense,
    embedding,
    heldout,
    old_caches,
    history_end,
    role,
    rank,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(heldout) != len(old_caches):
        raise ValueError("QB source cache count differs")
    exact_totals = torch.zeros(8, dtype=torch.float64, device=device)
    reuse_totals = torch.zeros(8, dtype=torch.float64, device=device)
    cache_totals = torch.zeros(4, dtype=torch.float64, device=device)
    started = time.perf_counter()
    for index, (value, old_cache) in enumerate(zip(heldout, old_caches, strict=True), start=1):
        exact_cache = prefix_cache(dense, embedding, value.batch, history_end, device)
        exact_scores = scores_with_cache(dense, embedding, value, exact_cache, history_end, device)
        reuse_scores = scores_with_cache(dense, embedding, value, old_cache, history_end, device)
        exact_totals += score_sums(exact_scores, exact_scores)
        reuse_totals += score_sums(reuse_scores, exact_scores)
        old_k = old_cache.k.float()
        old_v = old_cache.v.float()
        exact_k = exact_cache.k.float()
        exact_v = exact_cache.v.float()
        cache_totals += torch.tensor(
            [
                float(torch.sum((old_k - exact_k) ** 2).item()),
                float(torch.sum(exact_k**2).item()),
                float(torch.sum((old_v - exact_v) ** 2).item()),
                float(torch.sum(exact_v**2).item()),
            ],
            dtype=torch.float64,
            device=device,
        )
        if rank == 0 and (index == 1 or index == len(heldout) or index % 100 == 0):
            print(
                f"phase=qb_target_quality role={role} "
                f"batch={index}/{len(heldout)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return exact_totals, reuse_totals, cache_totals


def reduce_observation(
    frozen,
    exact,
    reuse,
    cache,
) -> dict[str, object]:
    for value in (frozen, exact, reuse, cache):
        dist.all_reduce(value)
    frozen_metrics = summarize_score_sums(frozen, relative=False)
    exact_metrics = summarize_score_sums(exact, relative=True)
    reuse_metrics = summarize_score_sums(reuse, relative=True)
    return {
        "frozen": frozen_metrics,
        "reuse": reuse_metrics,
        "exact": exact_metrics,
        "reuse_exact_opportunity": {
            "sampled_cross_entropy_gap": (
                reuse_metrics["sampled_cross_entropy"] - exact_metrics["sampled_cross_entropy"]
            ),
            "ndcg_at_10_gap": (exact_metrics["ndcg_at_10"] - reuse_metrics["ndcg_at_10"]),
            "mean_reciprocal_rank_gap": (
                exact_metrics["mean_reciprocal_rank"] - reuse_metrics["mean_reciprocal_rank"]
            ),
        },
        "reuse_cache_relative_l2": {
            "k": float(torch.sqrt(cache[0] / cache[1].clamp_min(1e-30)).item()),
            "v": float(torch.sqrt(cache[2] / cache[3].clamp_min(1e-30)).item()),
        },
    }


def save_optimizer_state(
    root: Path,
    version: int,
    dense_optimizer,
    projection_optimizer,
    rank: int,
) -> dict[str, object] | None:
    path = root / f"optimizer_after_theta_{version}.pt"
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(
            {
                "protocol": PROTOCOL,
                "version": version,
                "dense_adamw": dense_optimizer.state_dict(),
                "projection_adamw": projection_optimizer.state_dict(),
                "embedding_sgd": "stateless",
            },
            temporary,
        )
        temporary.replace(path)
    dist.barrier()
    if rank != 0:
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": tensor_file_sha256(path),
        "version": version,
    }


def load_optimizer_state(
    root: Path,
    version: int,
    dense_optimizer,
    projection_optimizer,
) -> dict[str, object]:
    path = root / f"optimizer_after_theta_{version}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("version") != version
        or payload.get("embedding_sgd") != "stateless"
        or not isinstance(payload.get("dense_adamw"), dict)
        or not isinstance(payload.get("projection_adamw"), dict)
    ):
        raise ValueError("QB base optimizer state differs")
    dense_optimizer.load_state_dict(payload["dense_adamw"])
    projection_optimizer.load_state_dict(payload["projection_adamw"])
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": tensor_file_sha256(path),
        "version": version,
    }


def checkpoint_binding(root: Path, version: int) -> dict[str, object]:
    path = root / f"theta_{version}" / "manifest.json"
    return {
        "root": str(root),
        "version": version,
        "manifest_path": str(path),
        "manifest_sha256": tensor_file_sha256(path),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    target_checkpoint_root = args.target_checkpoint_root or args.checkpoint_root
    active_continuation_policy = continuation_policy(args)
    rank, world_size, device, backend = init_distributed()
    started = time.perf_counter()
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats(device)
    try:
        corpus = load_qb_large_corpus(args.corpus, args.catalog)
        if corpus.catalog.profile.name != args.profile:
            raise ValueError("QB requested profile differs from corpus")
        embedding_width = args.embedding_width or corpus.catalog.profile.embedding_width
        if (
            not args.development_canary
            and embedding_width != corpus.catalog.profile.embedding_width
        ):
            raise ValueError("QB production embedding width differs from profile")
        spec = XPProjectedModelSpec(
            num_embeddings=corpus.catalog.num_embeddings,
            embedding_width=embedding_width,
            hidden_size=args.hidden_size,
            num_prediction_items=corpus.catalog.num_prediction_items,
            num_behaviors=5,
            num_layers=args.layers,
            num_heads=args.heads,
            head_dim=args.head_dim,
            max_seq_len=args.max_seq_len,
        )
        args._num_prediction_items = spec.num_prediction_items
        torch.manual_seed(args.seed + 101)
        dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
        embedding = TrainableProjectedModuloEmbedding.initialize(
            num_embeddings=spec.num_embeddings,
            embedding_width=spec.embedding_width,
            hidden_size=spec.hidden_size,
            rank=rank,
            world_size=world_size,
            device=device,
            embedding_seed=args.seed + 211,
            projection_seed=args.seed + 307,
        )
        tracker = OptimizerActiveRowTracker(
            num_embeddings=spec.num_embeddings,
            rank=rank,
            world_size=world_size,
        )
        dense_optimizer = torch.optim.AdamW(
            dense.parameters(),
            lr=args.base_dense_lr,
            weight_decay=args.weight_decay,
        )
        projection_optimizer = torch.optim.AdamW(
            [embedding.projection_weight],
            lr=args.base_projection_lr,
            weight_decay=args.weight_decay,
        )
        embedding_optimizer = sparse_embedding_sgd(embedding, args.base_embedding_lr)
        if args.hbm_canary_only:
            if args.maximum_contrastive_pairs < 1:
                raise ValueError("QB HBM canary requires a bounded contrastive sample")
            contrastive = run_contrastive(
                corpus,
                embedding,
                tracker,
                projection_optimizer,
                embedding_optimizer,
                args,
                rank,
                world_size,
                device,
            )
            base_batches, base_coverage = build_base_batches(
                corpus,
                batch_size_per_rank=args.batch_size_per_rank,
                rank=rank,
                world_size=world_size,
                maximum_steps=1,
            )
            base_training = run_training(
                "qb_hbm_canary_next_item",
                base_batches,
                1,
                dense,
                embedding,
                tracker,
                (dense_optimizer, projection_optimizer, embedding_optimizer),
                args,
                rank,
                world_size,
                device,
                1_000_003,
            )
            local_report = {
                "rank": rank,
                "local_embedding_rows": embedding.local_rows,
                "hbm": hbm_report(device),
            }
            reports: list[object] = [None] * world_size
            dist.all_gather_object(reports, local_report)
            if rank == 0:
                dense_bytes = sum(
                    value.numel() * value.element_size() for value in dense.parameters()
                )
                atomic_json(
                    args.output,
                    {
                        "protocol": "evokv_qb_large_hbm_admission_canary_v0",
                        "status": "pass",
                        "scientific_result": False,
                        "formal_design2": False,
                        "formal_design3": False,
                        "profile": args.profile,
                        "spec": asdict(spec),
                        "capacity": {
                            "semantic_rows": corpus.catalog.semantic_rows,
                            "global_fixed_bytes_fp32": (
                                corpus.catalog.semantic_rows * spec.embedding_width * 4
                                + dense_bytes
                                + spec.projection_bytes_fp32
                            ),
                            "single_a40_allocatable_reference_bytes": 47_699_722_240,
                        },
                        "contrastive_sample": contrastive,
                        "base_coverage": base_coverage,
                        "base_step": base_training,
                        "training_configuration": training_configuration(args),
                        "ranks": reports,
                    },
                )
                print(json.dumps({"status": "pass", "output": str(args.output)}))
            dist.barrier()
            return
        source_code = {
            name: {
                "path": str(path),
                "sha256": tensor_file_sha256(path),
            }
            for name, path in {
                "runner": Path(__file__),
                "training": Path("src/hstu_kvcache/streaming/qb_multifield_training.py"),
                "data": Path("src/hstu_kvcache/data/qb_large_multifield.py"),
                "lookup": Path("src/hstu_kvcache/streaming/multifield_projected.py"),
            }.items()
        }
        resume_record = None
        if args.resume_chain_result is not None:
            source_result = json.loads(args.resume_chain_result.read_text())
            source_configuration = dict(source_result.get("training_configuration", {}))
            source_configuration.pop("stop_version", None)
            source_binding = source_result.get("checkpoints", [])[args.resume_version]
            if (
                source_result.get("protocol") != PROTOCOL
                or source_result.get("status") != "complete"
                or source_result.get("scientific_result") is not False
                or source_result.get("formal_design2") is not False
                or source_result.get("formal_design3") is not False
                or source_result.get("profile") != args.profile
                or source_result.get("spec") != asdict(spec)
                or source_result.get("catalog", {}).get("content_sha256")
                != corpus.catalog.metadata["content_sha256"]
                or not continuation_corpus_compatible(
                    source_result,
                    corpus,
                    active_continuation_policy,
                )
                or source_result.get("execution", {}).get("development_canary")
                is not args.development_canary
                or (
                    not args.development_canary
                    and source_result.get("capacity", {}).get(
                        "forced_sharding_gate_passed"
                    )
                    is not True
                )
                or source_configuration != continuation_training_configuration(args)
                or source_binding.get("version") != args.resume_version
                or Path(source_binding.get("root", "")).resolve()
                != args.checkpoint_root.resolve()
            ):
                raise ValueError("QB continuation source result differs")
            resumed_manifest = load_xp_projected_checkpoint(
                args.checkpoint_root,
                args.resume_version,
                spec,
                dense,
                embedding,
                tracker,
            )
            observed_binding = checkpoint_binding(
                args.checkpoint_root,
                args.resume_version,
            )
            if observed_binding != source_binding:
                raise ValueError("QB continuation checkpoint binding differs")
            optimizer_binding = load_optimizer_state(
                args.checkpoint_root,
                args.resume_version,
                dense_optimizer,
                projection_optimizer,
            )
            if optimizer_binding != source_result.get("continuation_optimizer"):
                raise ValueError("QB continuation optimizer binding differs")
            if active_continuation_policy is not None and args.continuation_reset_adamw:
                dense_optimizer.state.clear()
                projection_optimizer.state.clear()
            contrastive = source_result["contrastive"]
            base_coverage = source_result["base_coverage"]
            base_training = source_result["base_training"]
            active_after_contrastive = int(source_result["active_rows_after_contrastive"])
            base_optimizer = source_result["base_optimizer"]
            checkpoints = source_result["checkpoints"][: args.resume_version + 1]
            base_source = f"resumed_chain_theta{args.resume_version}"
            start_version = args.resume_version
            resume_record = {
                "source_result": str(args.resume_chain_result),
                "source_result_sha256": tensor_file_sha256(args.resume_chain_result),
                "source_version": args.resume_version,
                "source_checkpoint": observed_binding,
                "source_optimizer": optimizer_binding,
                "source_manifest_active_rows": resumed_manifest["optimizer_active_rows"][
                    "global_active_rows"
                ],
            }
        elif args.resume_base_root is None:
            contrastive = run_contrastive(
                corpus,
                embedding,
                tracker,
                projection_optimizer,
                embedding_optimizer,
                args,
                rank,
                world_size,
                device,
            )
            active_after_contrastive = global_active_rows(tracker, device)
            if (
                not args.development_canary
                and active_after_contrastive != corpus.catalog.semantic_rows
            ):
                raise RuntimeError("QB active semantic-row coverage gate failed")
            base_batches, base_coverage = build_base_batches(
                corpus,
                batch_size_per_rank=args.batch_size_per_rank,
                rank=rank,
                world_size=world_size,
                maximum_steps=args.maximum_base_steps,
            )
            base_training = run_training(
                "qb_base_next_item",
                base_batches,
                args.base_epochs,
                dense,
                embedding,
                tracker,
                (dense_optimizer, projection_optimizer, embedding_optimizer),
                args,
                rank,
                world_size,
                device,
                1_000_003,
            )
            base_provenance = {
                "protocol": PROTOCOL,
                "profile": args.profile,
                "scientific_result": False,
                "formal_design2": False,
                "formal_design3": False,
                "corpus": {
                    "path": str(corpus.path),
                    "file_sha256": corpus.file_sha256,
                    "content_sha256": corpus.content_sha256,
                    "catalog_path": str(corpus.catalog_path),
                    "catalog_content_sha256": corpus.catalog.metadata["content_sha256"],
                },
                "contrastive": contrastive,
                "base_coverage": base_coverage,
                "base_training": base_training,
                "base_training_configuration": base_training_configuration(args),
                "source_code": source_code,
            }
            save_xp_projected_checkpoint(
                args.checkpoint_root,
                0,
                spec,
                dense,
                embedding,
                tracker,
                provenance=base_provenance,
            )
            base_optimizer = save_optimizer_state(
                args.checkpoint_root,
                0,
                dense_optimizer,
                projection_optimizer,
                rank,
            )
            checkpoints = [checkpoint_binding(args.checkpoint_root, 0)]
            base_source = "constructed_in_this_candidate"
            start_version = 0
        else:
            base_manifest = load_xp_projected_checkpoint(
                args.resume_base_root,
                0,
                spec,
                dense,
                embedding,
                tracker,
            )
            provenance = base_manifest.get("provenance")
            if (
                not isinstance(provenance, dict)
                or provenance.get("protocol") != PROTOCOL
                or provenance.get("profile") != args.profile
                or provenance.get("corpus", {}).get("content_sha256")
                != corpus.content_sha256
                or provenance.get("corpus", {}).get("catalog_content_sha256")
                != corpus.catalog.metadata["content_sha256"]
                or provenance.get("base_training_configuration")
                != base_training_configuration(args)
            ):
                raise ValueError("QB resumed base provenance differs")
            contrastive = provenance["contrastive"]
            base_coverage = provenance["base_coverage"]
            base_training = provenance["base_training"]
            active_after_contrastive = int(
                base_manifest["optimizer_active_rows"]["global_active_rows"]
            )
            base_optimizer = load_optimizer_state(
                args.resume_base_root,
                0,
                dense_optimizer,
                projection_optimizer,
            )
            checkpoints = [checkpoint_binding(args.resume_base_root, 0)]
            base_source = "resumed_common_theta0"
            start_version = 0
        active_learning_rates = (
            continuation_learning_rates(args)
            if active_continuation_policy is not None
            else {
                "dense": args.update_dense_lr,
                "projection": args.update_projection_lr,
                "embedding": args.update_embedding_lr,
            }
        )
        set_optimizer_lr(dense_optimizer, active_learning_rates["dense"])
        set_optimizer_lr(projection_optimizer, active_learning_rates["projection"])
        set_optimizer_lr(embedding_optimizer, active_learning_rates["embedding"])
        if active_continuation_policy is not None:
            train_end, evaluation_end = continuation_extents(args)
            updates = ((3, 4, 88, train_end, evaluation_end),)
        else:
            updates = (
                (0, 1, 64, 72, 80),
                (1, 2, 72, 80, 88),
                (2, 3, 80, 88, 96),
                (3, 4, 88, 96, 104),
            )[start_version : args.stop_version]
        edges = []
        for (
            source_version,
            target_version,
            update_start,
            update_end,
            evaluation_end,
        ) in updates:
            effective_train_start = (
                args.continuation_train_start
                if active_continuation_policy is not None
                else update_start
            )
            effective_update_epochs = (
                args.continuation_update_epochs
                if active_continuation_policy is not None
                else args.update_epochs
            )
            evaluations = {}
            source_caches = {}
            frozen_sums = {}
            for role in ("tuning", "qualification"):
                heldout, coverage = make_evaluations(
                    corpus,
                    role,
                    update_end,
                    evaluation_end,
                    args,
                    rank,
                    world_size,
                )
                caches, sums = source_evaluation(
                    dense,
                    embedding,
                    heldout,
                    update_end,
                    role,
                    rank,
                    device,
                )
                evaluations[role] = (heldout, coverage)
                source_caches[role] = caches
                frozen_sums[role] = sums
            update_batches, update_coverage = build_role_batches(
                corpus,
                "train",
                width=update_end,
                train_start=effective_train_start,
                train_end=update_end,
                batch_size_per_rank=args.batch_size_per_rank,
                rank=rank,
                world_size=world_size,
                maximum_steps=args.maximum_role_steps,
            )
            if (
                active_continuation_policy is not None
                and args.continuation_current_extra_passes
            ):
                current_batches, current_coverage = build_role_batches(
                    corpus,
                    "train",
                    width=update_end,
                    train_start=(
                        update_start
                        if args.continuation_extra_start is None
                        else args.continuation_extra_start
                    ),
                    train_end=update_end,
                    batch_size_per_rank=args.batch_size_per_rank,
                    rank=rank,
                    world_size=world_size,
                    maximum_steps=args.maximum_role_steps,
                )
                update_batches = [
                    *update_batches,
                    *(current_batches * args.continuation_current_extra_passes),
                ]
                update_coverage = {
                    "schedule": "replay_then_current",
                    "replay": update_coverage,
                    "current": current_coverage,
                    "current_extra_passes": args.continuation_current_extra_passes,
                    "combined_steps_per_rank": len(update_batches),
                }
            active_before = global_active_rows(tracker, device)
            training = run_training(
                f"qb_stream_theta{source_version}_theta{target_version}",
                update_batches,
                effective_update_epochs,
                dense,
                embedding,
                tracker,
                (dense_optimizer, projection_optimizer, embedding_optimizer),
                args,
                rank,
                world_size,
                device,
                (source_version + 2) * 1_000_003,
                negative_count=(
                    args.continuation_train_negatives
                    if active_continuation_policy is not None
                    and args.continuation_train_negatives is not None
                    else args.train_negatives
                ),
            )
            observations = {}
            for role in ("tuning", "qualification"):
                heldout, coverage = evaluations[role]
                exact, reuse, cache = target_evaluation(
                    dense,
                    embedding,
                    heldout,
                    source_caches[role],
                    update_end,
                    role,
                    rank,
                    device,
                )
                observations[role] = {
                    **reduce_observation(frozen_sums[role], exact, reuse, cache),
                    "coverage": coverage,
                    "used_for_configuration_selection": role == "tuning",
                }
                del source_caches[role]
            active_after = global_active_rows(tracker, device)
            edge = {
                "source_version": source_version,
                "target_version": target_version,
                "training_window": [effective_train_start, update_end],
                "next_unseen_evaluation_window": [update_end, evaluation_end],
                "continuation_training_policy": active_continuation_policy,
                "update_coverage": update_coverage,
                "training": training,
                "active_rows_before": active_before,
                "active_rows_after": active_after,
                "observations": observations,
            }
            manifest = save_xp_projected_checkpoint(
                target_checkpoint_root,
                target_version,
                spec,
                dense,
                embedding,
                tracker,
                provenance={
                    "protocol": PROTOCOL,
                    "profile": args.profile,
                    "scientific_result": False,
                    "formal_design2": False,
                    "formal_design3": False,
                    "parent_checkpoint": checkpoints[-1],
                    "edge": edge,
                    "training_configuration": training_configuration(args),
                    "continuation_training_policy": active_continuation_policy,
                    "source_code": source_code,
                },
            )
            binding = checkpoint_binding(target_checkpoint_root, target_version)
            checkpoints.append(binding)
            edge["checkpoint"] = binding
            edges.append(edge)
            if rank == 0:
                atomic_json(
                    target_checkpoint_root
                    / f"edge_theta{source_version}_theta{target_version}.json",
                    edge,
                )
            dist.barrier()
            del evaluations, source_caches, frozen_sums, update_batches, manifest
            gc.collect()
            torch.cuda.empty_cache()
        optimizer = save_optimizer_state(
            target_checkpoint_root,
            args.stop_version,
            dense_optimizer,
            projection_optimizer,
            rank,
        )
        local_report = {
            "rank": rank,
            "local_embedding_rows": modulo_local_rows(spec.num_embeddings, rank, world_size),
            "local_embedding_bytes": (
                embedding.local_weight.numel() * embedding.local_weight.element_size()
            ),
            "active_rows": tracker.local_active_count,
            "hbm": hbm_report(device),
        }
        rank_reports: list[object] = [None] * world_size
        dist.all_gather_object(rank_reports, local_report)
        final_active_rows = global_active_rows(tracker, device)
        if rank == 0:
            dense_bytes = sum(
                parameter.numel() * parameter.element_size() for parameter in dense.parameters()
            )
            result = {
                "protocol": PROTOCOL,
                "status": "complete",
                "scientific_result": False,
                "formal_design2": False,
                "formal_design3": False,
                "profile": args.profile,
                "execution": {
                    "backend": backend,
                    "world_size": world_size,
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "wall_seconds": time.perf_counter() - started,
                    "development_canary": args.development_canary,
                },
                "spec": asdict(spec),
                "feature_fields": list(corpus.catalog.profile.fields),
                "feature_rows_per_token": corpus.catalog.profile.feature_count,
                "catalog": {
                    "path": str(corpus.catalog_path),
                    "content_sha256": corpus.catalog.metadata["content_sha256"],
                    "semantic_rows": corpus.catalog.semantic_rows,
                    "field_rows": corpus.catalog.metadata["field_rows"],
                },
                "corpus": {
                    "path": str(corpus.path),
                    "file_sha256": corpus.file_sha256,
                    "content_sha256": corpus.content_sha256,
                    "roles": corpus.metadata["roles"],
                },
                "capacity": {
                    "global_active_embedding_bytes_fp32": (
                        final_active_rows * spec.embedding_width * 4
                    ),
                    "dense_bytes_fp32": dense_bytes,
                    "projection_bytes_fp32": spec.projection_bytes_fp32,
                    "global_active_fixed_bytes_fp32": (
                        final_active_rows * spec.embedding_width * 4
                        + dense_bytes
                        + spec.projection_bytes_fp32
                    ),
                    "single_a40_allocatable_reference_bytes": 47_699_722_240,
                    "forced_sharding_gate_passed": (
                        final_active_rows == corpus.catalog.semantic_rows
                        and final_active_rows * spec.embedding_width * 4
                        + dense_bytes
                        + spec.projection_bytes_fp32
                        > 47_699_722_240
                    ),
                },
                "training_configuration": training_configuration(args),
                "continuation_training_policy": active_continuation_policy,
                "continuation": resume_record,
                "base_source": base_source,
                "base_optimizer": base_optimizer,
                "contrastive": contrastive,
                "active_rows_after_contrastive": active_after_contrastive,
                "base_coverage": base_coverage,
                "base_training": base_training,
                "edges": edges,
                "checkpoints": checkpoints,
                "continuation_optimizer": optimizer,
                "source_code": source_code,
                "ranks": rank_reports,
            }
            atomic_json(args.output, result)
            print(json.dumps({"status": "complete", "output": str(args.output)}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
