from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.streaming.sharded_edge import (
    ExternalEmbeddingHSTU,
    evaluate_fixed_heldout,
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
from hstu_kvcache.streaming.xp_version_training import (
    XP_VERSION_TRAINING_PROTOCOL,
    build_role_batches,
    file_sha256,
    global_tracker_delta,
    load_xp_fixed_edge_corpus,
    prepare_fixed_qualification,
    tracker_count_snapshot,
    xp_projected_next_item_train_step,
)

DEFAULT_EDGE_INPUTS = Path(
    "data/processed/evokv_foundation/"
    "qk_xp_fixed_edge_inputs.npz"
)
DEFAULT_EDGE_SUMMARY = Path(
    "configs/evokv_foundation/"
    "qk_xp_fixed_edge_inputs_summary.json"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "checkpoints/evokv_xp_qk_e4096_h1536/seed0"
)
DEFAULT_OUTPUT = Path(
    "results/system/evokv_design3_foundation/"
    "xp_theta0_theta1_theta2_training_development.json"
)
CANONICAL_NUM_EMBEDDINGS = 2_859_836
CANONICAL_EMBEDDING_WIDTH = 4096
CANONICAL_HIDDEN_SIZE = 1536
CANONICAL_LAYERS = 24
CANONICAL_HEADS = 24
CANONICAL_HEAD_DIM = 64
CANONICAL_MAX_SEQ_LEN = 512
CANONICAL_ROLE_COUNTS = {
    "theta01": 2_560,
    "theta12": 2_048,
    "qualification": 512,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge-inputs",
        type=Path,
        default=DEFAULT_EDGE_INPUTS,
    )
    parser.add_argument(
        "--edge-summary",
        type=Path,
        default=DEFAULT_EDGE_SUMMARY,
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--backend")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--batch-size-per-rank",
        type=int,
        default=1,
    )
    parser.add_argument("--epochs-per-edge", type=int, default=1)
    parser.add_argument(
        "--dense-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--projection-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--embedding-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-negatives", type=int, default=8)
    parser.add_argument(
        "--qualification-negatives",
        type=int,
        default=99,
    )
    parser.add_argument(
        "--qualification-seed",
        type=int,
        default=20260801,
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--development-canary", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size_per_rank": args.batch_size_per_rank,
        "epochs_per_edge": args.epochs_per_edge,
        "train_negatives": args.train_negatives,
        "qualification_negatives": (
            args.qualification_negatives
        ),
        "progress_every": args.progress_every,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if (
        invalid
        or args.dense_learning_rate <= 0
        or args.projection_learning_rate <= 0
        or args.embedding_learning_rate <= 0
        or args.weight_decay < 0
    ):
        raise ValueError(
            "XP two-edge training arguments are invalid: "
            f"{', '.join(invalid)}"
        )
    existing = [
        path
        for path in (
            args.output,
            args.checkpoint_root / "theta_1" / "manifest.json",
            args.checkpoint_root / "theta_2" / "manifest.json",
        )
        if path.exists()
    ]
    if existing and not args.force:
        raise FileExistsError(
            "XP two-edge outputs exist; pass --force to replace: "
            f"{[str(path) for path in existing]}"
        )


def init_runtime(
    args: argparse.Namespace,
) -> tuple[int, int, torch.device, str]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2:
        raise RuntimeError(
            "XP successor edge training requires exactly two ranks"
        )
    backend = args.backend or (
        "nccl" if args.device == "cuda" else "gloo"
    )
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("XP CUDA execution is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if not args.development_canary:
            raise RuntimeError(
                "CPU execution is restricted to development canaries"
            )
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("XP process-group identity differs")
    return rank, world_size, device, backend


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_theta0(
    checkpoint_root: Path,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[
    XPProjectedModelSpec,
    ExternalEmbeddingHSTU,
    TrainableProjectedModuloEmbedding,
    OptimizerActiveRowTracker,
    dict[str, object],
]:
    manifest_path = checkpoint_root / "theta_0" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        not isinstance(manifest.get("spec"), dict)
        or manifest.get("version") != 0
        or manifest.get("world_size") != world_size
    ):
        raise ValueError("XP theta0 manifest differs")
    spec = XPProjectedModelSpec(**manifest["spec"])
    dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
    local_weight = torch.empty(
        (
            modulo_local_rows(
                spec.num_embeddings,
                rank,
                world_size,
            ),
            spec.embedding_width,
        ),
        dtype=torch.float32,
        device=device,
    )
    projection = torch.empty(
        (spec.hidden_size, spec.embedding_width),
        dtype=torch.float32,
        device=device,
    )
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=local_weight,
        projection_weight=projection,
        num_embeddings=spec.num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    tracker = OptimizerActiveRowTracker(
        num_embeddings=spec.num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    loaded = load_xp_projected_checkpoint(
        checkpoint_root,
        0,
        spec,
        dense,
        embedding,
        tracker,
    )
    if loaded != manifest:
        raise ValueError("XP theta0 manifest changed during load")
    return spec, dense, embedding, tracker, {
        "root": str(checkpoint_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest": manifest,
    }


def validate_geometry(
    spec: XPProjectedModelSpec,
    corpus,
    development_canary: bool,
) -> None:
    if development_canary:
        return
    if (
        spec.num_embeddings != CANONICAL_NUM_EMBEDDINGS
        or spec.embedding_width
        != CANONICAL_EMBEDDING_WIDTH
        or spec.hidden_size != CANONICAL_HIDDEN_SIZE
        or spec.num_layers != CANONICAL_LAYERS
        or spec.num_heads != CANONICAL_HEADS
        or spec.head_dim != CANONICAL_HEAD_DIM
        or spec.max_seq_len != CANONICAL_MAX_SEQ_LEN
        or {
            role: len(corpus.role_records(role))
            for role in CANONICAL_ROLE_COUNTS
        }
        != CANONICAL_ROLE_COUNTS
    ):
        raise ValueError("canonical XP edge geometry differs")


def gather_coverage(
    local: dict[str, object],
    world_size: int,
) -> dict[str, object]:
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return {
        "role": local["role"],
        "global_records": local["global_records"],
        "global_eligible_records": local[
            "global_eligible_records"
        ],
        "global_zero_target_records_removed": local[
            "global_zero_target_records_removed"
        ],
        "steps_per_rank": local["steps_per_rank"],
        "batch_size_per_rank": local[
            "batch_size_per_rank"
        ],
        "real_records_per_rank": [
            value["local_real_records"] for value in gathered
        ],
        "padding_records_per_rank": [
            value["local_padding_records"] for value in gathered
        ],
        "tokens_per_rank": [
            value["local_tokens"] for value in gathered
        ],
        "targets_per_rank": [
            value["local_targets"] for value in gathered
        ],
        "global_targets": sum(
            value["local_targets"] for value in gathered
        ),
        "maximum_model_context": local[
            "maximum_model_context"
        ],
        "physical_sequence_width": local[
            "physical_sequence_width"
        ],
        "causal_window_start": local[
            "causal_window_start"
        ],
        "time_delta_semantics": local[
            "time_delta_semantics"
        ],
    }


def train_edge(
    role: str,
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    batches: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    seed_base: int,
) -> tuple[dict[str, object], dict[str, object]]:
    before = tracker_count_snapshot(tracker)
    dense_optimizer = torch.optim.AdamW(
        dense.parameters(),
        lr=args.dense_learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    projection_optimizer = torch.optim.AdamW(
        [embedding.projection_weight],
        lr=args.projection_learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    embedding_optimizer = sparse_embedding_sgd(
        embedding,
        args.embedding_learning_rate,
    )
    epoch_reports = []
    started = time.perf_counter()
    for epoch in range(args.epochs_per_edge):
        local_loss_sum = 0.0
        local_targets = 0
        minimum_global_targets = None
        maximum_global_targets = 0
        for step, batch in enumerate(batches):
            loss, targets, global_targets, _ = (
                xp_projected_next_item_train_step(
                    dense,
                    embedding,
                    tracker,
                    batch,
                    dense_optimizer,
                    projection_optimizer,
                    embedding_optimizer,
                    device=device,
                    num_prediction_items=(
                        dense.cfg.num_prediction_items
                    ),
                    negative_count=args.train_negatives,
                    negative_seed=(
                        seed_base
                        + epoch * 10_000_019
                        + step * world_size
                        + rank
                    ),
                )
            )
            local_loss_sum += loss * targets
            local_targets += targets
            minimum_global_targets = (
                global_targets
                if minimum_global_targets is None
                else min(minimum_global_targets, global_targets)
            )
            maximum_global_targets = max(
                maximum_global_targets,
                global_targets,
            )
            if rank == 0 and (
                step == 0
                or step + 1 == len(batches)
                or (step + 1) % args.progress_every == 0
            ):
                print(
                    f"phase={role} epoch={epoch + 1}/"
                    f"{args.epochs_per_edge} step={step + 1}/"
                    f"{len(batches)} loss={loss:.6f} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        totals = torch.tensor(
            [local_loss_sum, local_targets],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[1].item() <= 0:
            raise RuntimeError(f"XP edge {role} has no targets")
        epoch_reports.append(
            {
                "epoch": epoch + 1,
                "global_target_mean_loss": float(
                    (totals[0] / totals[1]).item()
                ),
                "global_targets": int(totals[1].item()),
                "minimum_step_global_targets": (
                    minimum_global_targets
                ),
                "maximum_step_global_targets": (
                    maximum_global_targets
                ),
            }
        )
    delta = global_tracker_delta(tracker, before)
    return {
        "role": role,
        "objective": (
            "leak-free sampled next-item cross entropy on update "
            "targets only, conditioned on causal history and prior "
            "update events"
        ),
        "epochs": epoch_reports,
        "dense_optimizer": "AdamW with manual summed replica gradients",
        "projection_optimizer": (
            "replicated AdamW after projected-lookup gradient all-reduce"
        ),
        "embedding_optimizer": (
            "owner-local sparse SGD without momentum"
        ),
        "dense_learning_rate": args.dense_learning_rate,
        "projection_learning_rate": (
            args.projection_learning_rate
        ),
        "embedding_learning_rate": (
            args.embedding_learning_rate
        ),
        "weight_decay": args.weight_decay,
        "negative_count": args.train_negatives,
        "negative_seed_base": seed_base,
        "wall_seconds": time.perf_counter() - started,
    }, delta


def qualification_signal(
    before: dict[str, float | int],
    after: dict[str, float | int],
) -> dict[str, object]:
    cross_entropy_reduction = float(
        before["sampled_cross_entropy"]
    ) - float(after["sampled_cross_entropy"])
    return {
        "predeclared_primary_metric": (
            "sampled_cross_entropy_reduction"
        ),
        "before": before,
        "after": after,
        "sampled_cross_entropy_reduction": (
            cross_entropy_reduction
        ),
        "ndcg_at_10_delta": (
            float(after["ndcg_at_10"])
            - float(before["ndcg_at_10"])
        ),
        "hit_rate_at_10_delta": (
            float(after["hit_rate_at_10"])
            - float(before["hit_rate_at_10"])
        ),
        "mean_reciprocal_rank_delta": (
            float(after["mean_reciprocal_rank"])
            - float(before["mean_reciprocal_rank"])
        ),
        "positive_recommendation_signal": (
            cross_entropy_reduction > 0
        ),
    }


def source_code_binding() -> dict[str, object]:
    paths = (
        Path(__file__),
        Path(
            "src/hstu_kvcache/streaming/"
            "xp_version_training.py"
        ),
        Path(
            "src/hstu_kvcache/streaming/"
            "xp_projected_edge.py"
        ),
    )
    return {
        path.name: {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for path in paths
    }


def save_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def hbm_report(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "device_total_bytes": 0,
        }
    return {
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device)
        ),
        "device_total_bytes": (
            torch.cuda.get_device_properties(device).total_memory
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object] | None:
    validate_args(args)
    rank, world_size, device, backend = init_runtime(args)
    started = time.perf_counter()
    try:
        seed_everything(args.seed)
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.cuda.reset_peak_memory_stats(device)
        spec, dense, embedding, tracker, theta0_binding = (
            load_theta0(
                args.checkpoint_root,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        )
        corpus = load_xp_fixed_edge_corpus(
            args.edge_inputs,
            args.edge_summary,
            num_embeddings=spec.num_embeddings,
            num_prediction_items=spec.num_prediction_items,
            num_behaviors=spec.num_behaviors,
        )
        validate_geometry(
            spec,
            corpus,
            args.development_canary,
        )
        theta01_batches, theta01_local_coverage = (
            build_role_batches(
                corpus,
                "theta01",
                max_seq_len=spec.max_seq_len,
                batch_size_per_rank=args.batch_size_per_rank,
                rank=rank,
                world_size=world_size,
            )
        )
        theta12_batches, theta12_local_coverage = (
            build_role_batches(
                corpus,
                "theta12",
                max_seq_len=spec.max_seq_len,
                batch_size_per_rank=args.batch_size_per_rank,
                rank=rank,
                world_size=world_size,
            )
        )
        qualification_batches, qualification_local_coverage = (
            build_role_batches(
                corpus,
                "qualification",
                max_seq_len=spec.max_seq_len,
                batch_size_per_rank=args.batch_size_per_rank,
                rank=rank,
                world_size=world_size,
            )
        )
        coverage = {
            "theta01": gather_coverage(
                theta01_local_coverage,
                world_size,
            ),
            "theta12": gather_coverage(
                theta12_local_coverage,
                world_size,
            ),
            "qualification": gather_coverage(
                qualification_local_coverage,
                world_size,
            ),
        }
        fixed_qualification, local_candidate_hash = (
            prepare_fixed_qualification(
                qualification_batches,
                num_prediction_items=(
                    spec.num_prediction_items
                ),
                negative_count=(
                    args.qualification_negatives
                ),
                seed=args.qualification_seed,
                rank=rank,
                world_size=world_size,
            )
        )
        candidate_hashes: list[object] = [None] * world_size
        dist.all_gather_object(
            candidate_hashes,
            local_candidate_hash,
        )
        theta0_qualification = evaluate_fixed_heldout(
            dense,
            embedding,
            fixed_qualification,
            device,
        )
        theta01_training, theta01_delta = train_edge(
            "theta01",
            dense,
            embedding,
            tracker,
            theta01_batches,
            args,
            rank=rank,
            world_size=world_size,
            device=device,
            seed_base=args.seed + 100_000_007,
        )
        theta1_qualification = evaluate_fixed_heldout(
            dense,
            embedding,
            fixed_qualification,
            device,
        )
        theta01_signal = qualification_signal(
            theta0_qualification,
            theta1_qualification,
        )
        common_provenance = {
            "protocol": XP_VERSION_TRAINING_PROTOCOL,
            "scientific_result": False,
            "formal_design2": False,
            "formal_design3": False,
            "fixed_edge_inputs": {
                "path": str(corpus.path),
                "file_sha256": corpus.file_sha256,
                "content_sha256": corpus.content_sha256,
                "summary_path": str(corpus.summary_path),
                "summary_sha256": corpus.summary_sha256,
                "boundaries": corpus.metadata["boundaries"],
                "frozen_roles": corpus.metadata[
                    "frozen_roles"
                ],
            },
            "leakage_contract": {
                "theta01_optimizer_role": "theta01",
                "theta12_optimizer_role": "theta12",
                "qualification_used_by_optimizer": False,
                "qualification_used_for_hyperparameter_selection": False,
                "excluded_upstream_roles": [
                    "fit",
                    "profile",
                    "final",
                ],
                "forbidden_role_values_present": False,
            },
            "qualification_protocol": {
                "role": "qualification",
                "negative_count": (
                    args.qualification_negatives
                ),
                "candidate_count": (
                    args.qualification_negatives + 1
                ),
                "seed": args.qualification_seed,
                "candidate_sha256_per_rank": candidate_hashes,
                "primary_metric_frozen_before_training": (
                    "sampled_cross_entropy_reduction"
                ),
            },
            "source_code": source_code_binding(),
        }
        theta1_checkpoint = save_xp_projected_checkpoint(
            args.checkpoint_root,
            1,
            spec,
            dense,
            embedding,
            tracker,
            provenance={
                **common_provenance,
                "parent_checkpoint": {
                    "version": 0,
                    "manifest_sha256": theta0_binding[
                        "manifest_sha256"
                    ],
                },
                "edge": {
                    "source_version": 0,
                    "target_version": 1,
                    "training": theta01_training,
                    "optimizer_active_delta": theta01_delta,
                    "qualification": theta01_signal,
                },
            },
        )
        theta1_manifest_path = (
            args.checkpoint_root / "theta_1" / "manifest.json"
        )
        theta12_training, theta12_delta = train_edge(
            "theta12",
            dense,
            embedding,
            tracker,
            theta12_batches,
            args,
            rank=rank,
            world_size=world_size,
            device=device,
            seed_base=args.seed + 200_000_033,
        )
        theta2_qualification = evaluate_fixed_heldout(
            dense,
            embedding,
            fixed_qualification,
            device,
        )
        theta12_signal = qualification_signal(
            theta1_qualification,
            theta2_qualification,
        )
        theta2_checkpoint = save_xp_projected_checkpoint(
            args.checkpoint_root,
            2,
            spec,
            dense,
            embedding,
            tracker,
            provenance={
                **common_provenance,
                "parent_checkpoint": {
                    "version": 1,
                    "manifest_sha256": file_sha256(
                        theta1_manifest_path
                    ),
                },
                "edge": {
                    "source_version": 1,
                    "target_version": 2,
                    "training": theta12_training,
                    "optimizer_active_delta": theta12_delta,
                    "qualification": theta12_signal,
                },
            },
        )
        local_report = {
            "rank": rank,
            "local_embedding_rows": embedding.local_rows,
            "local_embedding_bytes": (
                embedding.local_weight.numel()
                * embedding.local_weight.element_size()
            ),
            "local_cumulative_active_rows": (
                tracker.local_active_count
            ),
            "memory": hbm_report(device),
        }
        rank_reports: list[object] = [None] * world_size
        dist.all_gather_object(rank_reports, local_report)
        both_signals = bool(
            theta01_signal["positive_recommendation_signal"]
            and theta12_signal[
                "positive_recommendation_signal"
            ]
        )
        result = None
        if rank == 0:
            result = {
                "protocol": XP_VERSION_TRAINING_PROTOCOL,
                "status": (
                    "complete"
                    if both_signals
                    else "complete_positive_signal_gate_failed"
                ),
                "scientific_result": False,
                "formal_design2": False,
                "formal_design3": False,
                "artifact_role": (
                    "successor_xp_real_theta0_theta1_theta2_edges"
                ),
                "execution": {
                    "device": args.device,
                    "backend": backend,
                    "world_size": world_size,
                    "visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES",
                        "",
                    ),
                    "seed": args.seed,
                    "development_canary": (
                        args.development_canary
                    ),
                    "total_wall_seconds": (
                        time.perf_counter() - started
                    ),
                },
                "model": {
                    "spec": asdict(spec),
                    "embedding_layout": (
                        "modulo row-sharded FP32 E-width"
                    ),
                    "projection_layout": (
                        "replicated bias-free owner-side E-to-H"
                    ),
                    "dense_layout": (
                        "replicated with manually reduced gradients"
                    ),
                },
                "fixed_edge_inputs": (
                    common_provenance["fixed_edge_inputs"]
                ),
                "leakage_contract": (
                    common_provenance["leakage_contract"]
                ),
                "coverage": coverage,
                "training": {
                    "theta0_to_theta1": theta01_training,
                    "theta1_to_theta2": theta12_training,
                },
                "optimizer_active_delta": {
                    "theta0_to_theta1": theta01_delta,
                    "theta1_to_theta2": theta12_delta,
                },
                "independent_qualification": {
                    **common_provenance[
                        "qualification_protocol"
                    ],
                    "theta0": theta0_qualification,
                    "theta1": theta1_qualification,
                    "theta2": theta2_qualification,
                    "theta0_to_theta1": theta01_signal,
                    "theta1_to_theta2": theta12_signal,
                    "both_edges_positive": both_signals,
                },
                "checkpoints": {
                    "theta0": theta0_binding,
                    "theta1": theta1_checkpoint,
                    "theta2": theta2_checkpoint,
                },
                "ranks": rank_reports,
                "source_code": (
                    common_provenance["source_code"]
                ),
            }
            save_json_atomic(args.output, result)
        dist.barrier()
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    result = run(args)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
