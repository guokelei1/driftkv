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

from hstu_kvcache.streaming.sharded_edge import (
    ExternalEmbeddingHSTU,
    evaluate_fixed_heldout,
    modulo_local_rows,
)
from hstu_kvcache.streaming.xp_multiversion import (
    XP_MULTIVERSION_PROTOCOL,
    XP_PREQUENTIAL_MULTIVERSION_PROTOCOL,
    XPLearningRateCandidate,
    XPMultiversionSchedule,
    XPUpdateWindow,
    build_window_batches,
    load_xp_multiversion_schedule,
    qualification_signal,
    select_learning_rate_candidate,
    validate_xp_multiversion_corpus,
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
    file_sha256,
    global_tracker_delta,
    prepare_fixed_qualification,
    tracker_count_snapshot,
    xp_projected_next_item_train_step,
)

DEFAULT_SCHEDULE = Path(
    "configs/evokv_foundation/"
    "xp_qk_multiversion_prequential3_development_v1.json"
)
DEFAULT_BASE_CHECKPOINT_ROOT = Path(
    "checkpoints/evokv_xp_qk_e4096_h1536/seed0"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "checkpoints/evokv_xp_qk_e4096_h1536/"
    "seed0_multiversion_prequential3_development_v1"
)
DEFAULT_OUTPUT = Path(
    "results/system/evokv_design3_foundation/"
    "xp_multiversion_prequential3_training_development_v1.json"
)
DEFAULT_LEDGER_DIR = Path(
    "results/system/evokv_design3_foundation/"
    "xp_multiversion_prequential3_ledgers_development_v1"
)
CANONICAL_SPEC = XPProjectedModelSpec(
    num_embeddings=2_859_836,
    embedding_width=4_096,
    hidden_size=1_536,
    num_prediction_items=250_000,
    num_behaviors=5,
    num_layers=24,
    num_heads=24,
    head_dim=64,
    max_seq_len=512,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument(
        "--base-checkpoint-root",
        type=Path,
        default=DEFAULT_BASE_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=DEFAULT_LEDGER_DIR,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backend")
    parser.add_argument("--batch-size-per-rank", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--development-canary", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_outputs(args: argparse.Namespace, schedule: XPMultiversionSchedule) -> None:
    if args.batch_size_per_rank < 1 or args.progress_every < 1:
        raise ValueError("XP multiversion runtime arguments are invalid")
    if args.checkpoint_root.resolve() == args.base_checkpoint_root.resolve():
        raise ValueError(
            "XP multiversion target root must differ from the base root"
        )
    expected = [args.output]
    expected.extend(
        args.checkpoint_root
        / f"theta_{update.target_version}"
        for update in schedule.updates
    )
    expected.extend(
        args.ledger_dir / f"version_{update.target_version:05d}.json"
        for update in schedule.updates
    )
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise FileExistsError(
            "XP multiversion outputs already exist; use a new stack root: "
            f"{existing}"
        )


def _init_runtime(
    args: argparse.Namespace,
) -> tuple[int, int, torch.device, str]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2:
        raise RuntimeError("XP multiversion training requires two ranks")
    backend = args.backend or (
        "nccl" if args.device == "cuda" else "gloo"
    )
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("XP multiversion CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if not args.development_canary:
            raise RuntimeError(
                "CPU multiversion execution is restricted to canaries"
            )
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("XP multiversion process identity differs")
    return rank, world_size, device, backend


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_base(
    root: Path,
    version: int,
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
    manifest_path = root / f"theta_{version}" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("version") != version
        or manifest.get("world_size") != world_size
        or not isinstance(manifest.get("spec"), dict)
    ):
        raise ValueError("XP multiversion base manifest differs")
    spec = XPProjectedModelSpec(**manifest["spec"])
    dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
    local_weight = torch.empty(
        (
            modulo_local_rows(spec.num_embeddings, rank, world_size),
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
        root,
        version,
        spec,
        dense,
        embedding,
        tracker,
    )
    if loaded != manifest:
        raise ValueError("XP multiversion base changed during load")
    return spec, dense, embedding, tracker, {
        "version": version,
        "root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "global_active_rows": manifest["optimizer_active_rows"][
            "global_active_rows"
        ],
    }


def _capture_state(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
) -> dict[str, object]:
    return {
        "dense": {
            name: value.detach().cpu().clone()
            for name, value in dense.state_dict().items()
        },
        "embedding": embedding.local_weight.detach().cpu().clone(),
        "projection": (
            embedding.projection_weight.detach().cpu().clone()
        ),
        "active_bitmap": tracker.local_bitmap.clone(),
        "active_counts": tracker.local_update_counts.clone(),
    }


def _restore_state(
    state: dict[str, object],
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
) -> None:
    dense.load_state_dict(state["dense"])
    with torch.no_grad():
        embedding.local_weight.copy_(state["embedding"])
        embedding.projection_weight.copy_(state["projection"])
    tracker.load_activity(state["active_bitmap"], state["active_counts"])


def _gather_coverage(
    local: dict[str, object],
    world_size: int,
) -> dict[str, object]:
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local)
    invariant_names = (
        "role",
        "source_version",
        "target_version",
        "history_end",
        "update_end",
        "update_width",
        "global_records",
        "global_eligible_records",
        "global_zero_target_records_removed",
        "steps_per_rank",
        "batch_size_per_rank",
        "maximum_model_context",
        "physical_sequence_width",
        "causal_window_start",
        "window_targets_sha256",
    )
    if any(
        any(value[name] != local[name] for name in invariant_names)
        for value in gathered
    ):
        raise ValueError("XP multiversion coverage replicas differ")
    return {
        **{name: local[name] for name in invariant_names},
        "real_records_per_rank": [
            value["local_real_records"] for value in gathered
        ],
        "padding_records_per_rank": [
            value["local_padding_records"] for value in gathered
        ],
        "tokens_per_rank": [value["local_tokens"] for value in gathered],
        "targets_per_rank": [value["local_targets"] for value in gathered],
        "global_targets": sum(
            value["local_targets"] for value in gathered
        ),
    }


def _candidate_hashes(local: str, world_size: int) -> list[object]:
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return gathered


def _prepare_edges(
    validated,
    spec: XPProjectedModelSpec,
    schedule: XPMultiversionSchedule,
    *,
    rank: int,
    world_size: int,
    batch_size_per_rank: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    prepared = []
    coverage = {}
    for index, update in enumerate(schedule.updates):
        edge_batches = {}
        edge_coverage = {}
        for split, role in schedule.split_roles.items():
            batches, local = build_window_batches(
                validated.corpus,
                role,
                update,
                max_seq_len=spec.max_seq_len,
                batch_size_per_rank=batch_size_per_rank,
                rank=rank,
                world_size=world_size,
            )
            edge_batches[split] = batches
            edge_coverage[split] = _gather_coverage(local, world_size)
        tuning_fixed, tuning_hash = prepare_fixed_qualification(
            edge_batches["tuning"],
            num_prediction_items=spec.num_prediction_items,
            negative_count=schedule.tuning_negatives,
            seed=schedule.tuning_seed + index * 1_000_003,
            rank=rank,
            world_size=world_size,
        )
        quality_fixed, quality_hash = prepare_fixed_qualification(
            edge_batches["quality"],
            num_prediction_items=spec.num_prediction_items,
            negative_count=schedule.quality_negatives,
            seed=schedule.quality_seed + index * 1_000_033,
            rank=rank,
            world_size=world_size,
        )
        prepared.append(
            {
                "update": update,
                "train": edge_batches["train"],
                "tuning": tuning_fixed,
                "quality": quality_fixed,
                "tuning_candidate_sha256_per_rank": _candidate_hashes(
                    tuning_hash,
                    world_size,
                ),
                "quality_candidate_sha256_per_rank": _candidate_hashes(
                    quality_hash,
                    world_size,
                ),
            }
        )
        coverage[f"theta{update.source_version}_to_theta{update.target_version}"] = (
            edge_coverage
        )
    return prepared, coverage


def _prepare_prequential(
    validated,
    spec: XPProjectedModelSpec,
    schedule: XPMultiversionSchedule,
    *,
    rank: int,
    world_size: int,
    batch_size_per_rank: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    prepared_updates = []
    training_coverage = {}
    for update in schedule.updates:
        batches, local = build_window_batches(
            validated.corpus,
            schedule.split_roles["train"],
            update,
            max_seq_len=spec.max_seq_len,
            batch_size_per_rank=batch_size_per_rank,
            rank=rank,
            world_size=world_size,
        )
        prepared_updates.append({"update": update, "train": batches})
        training_coverage[
            f"theta{update.source_version}_to_theta{update.target_version}"
        ] = _gather_coverage(local, world_size)
    prepared_evaluations = []
    evaluation_coverage = {}
    for index, evaluation in enumerate(schedule.evaluation_windows):
        window = XPUpdateWindow(
            source_version=evaluation.model_version,
            target_version=evaluation.model_version,
            history_end=evaluation.history_end,
            update_end=evaluation.evaluation_end,
        )
        fixed = {}
        role_coverage = {}
        for split in ("tuning", "quality"):
            batches, local = build_window_batches(
                validated.corpus,
                schedule.split_roles[split],
                window,
                max_seq_len=spec.max_seq_len,
                batch_size_per_rank=batch_size_per_rank,
                rank=rank,
                world_size=world_size,
            )
            negative_count = (
                schedule.tuning_negatives
                if split == "tuning"
                else schedule.quality_negatives
            )
            seed = (
                schedule.tuning_seed + index * 1_000_003
                if split == "tuning"
                else schedule.quality_seed + index * 1_000_033
            )
            heldout, candidate_hash = prepare_fixed_qualification(
                batches,
                num_prediction_items=spec.num_prediction_items,
                negative_count=negative_count,
                seed=seed,
                rank=rank,
                world_size=world_size,
            )
            fixed[split] = heldout
            fixed[f"{split}_candidate_sha256_per_rank"] = (
                _candidate_hashes(candidate_hash, world_size)
            )
            role_coverage[split] = _gather_coverage(local, world_size)
        prepared_evaluations.append(
            {"evaluation": evaluation, **fixed}
        )
        evaluation_coverage[
            f"theta{evaluation.model_version}"
        ] = role_coverage
    return (
        prepared_updates,
        prepared_evaluations,
        {
            "training_edges": training_coverage,
            "prequential_models": evaluation_coverage,
        },
    )


def _build_optimizers(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    candidate: XPLearningRateCandidate,
    schedule: XPMultiversionSchedule,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    return (
        torch.optim.AdamW(
            dense.parameters(),
            lr=candidate.dense,
            weight_decay=schedule.weight_decay,
            foreach=False,
        ),
        torch.optim.AdamW(
            [embedding.projection_weight],
            lr=candidate.projection,
            weight_decay=schedule.weight_decay,
            foreach=False,
        ),
        sparse_embedding_sgd(
            embedding,
            candidate.embedding,
        ),
    )


def _numeric_stability(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    before: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    dense_projection_finite = all(
        bool(torch.all(torch.isfinite(parameter)))
        for parameter in (
            *dense.parameters(),
            embedding.projection_weight,
        )
    )
    changed = torch.nonzero(
        tracker.local_update_counts > before,
        as_tuple=False,
    ).flatten()
    updated_embedding_finite = True
    for start in range(0, len(changed), 4096):
        rows = changed[start : start + 4096].to(device=device)
        if rows.numel() and not bool(
            torch.all(
                torch.isfinite(
                    embedding.local_weight.index_select(0, rows)
                )
            )
        ):
            updated_embedding_finite = False
            break
    finite = torch.tensor(
        [
            int(dense_projection_finite),
            int(updated_embedding_finite),
        ],
        dtype=torch.int64,
        device=device,
    )
    ranks_with_updates = torch.tensor(
        int(len(changed) > 0),
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    dist.all_reduce(ranks_with_updates, op=dist.ReduceOp.SUM)
    return {
        "dense_and_projection_finite_all_ranks": bool(finite[0].item()),
        "updated_embedding_rows_finite_all_ranks": bool(finite[1].item()),
        "ranks_with_updated_embedding_rows": int(
            ranks_with_updates.item()
        ),
        "local_updated_embedding_rows": len(changed),
        "passed": bool(
            torch.all(finite > 0).item()
            and ranks_with_updates.item() > 0
        ),
    }


def _train_update(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    batches: list[dict[str, torch.Tensor]],
    candidate: XPLearningRateCandidate,
    schedule: XPMultiversionSchedule,
    update: XPUpdateWindow,
    *,
    edge_index: int,
    rank: int,
    world_size: int,
    device: torch.device,
    progress_every: int,
    phase: str,
    optimizers: tuple[
        torch.optim.Optimizer,
        torch.optim.Optimizer,
        torch.optim.Optimizer,
    ]
    | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    before = tracker_count_snapshot(tracker)
    if optimizers is None:
        optimizers = _build_optimizers(
            dense,
            embedding,
            candidate,
            schedule,
        )
    dense_optimizer, projection_optimizer, embedding_optimizer = optimizers
    epochs = []
    started = time.perf_counter()
    seed_base = schedule.training_seed + edge_index * 100_000_007
    for epoch in range(schedule.epochs_per_update):
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
                    num_prediction_items=spec_num_prediction_items(dense),
                    negative_count=schedule.train_negatives,
                    negative_seed=(
                        seed_base
                        + epoch * 10_000_019
                        + step * world_size
                        + rank
                    ),
                )
            )
            if not np.isfinite(loss):
                raise RuntimeError("XP multiversion loss is not finite")
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
                or (step + 1) % progress_every == 0
            ):
                print(
                    f"phase={phase} edge={update.source_version}->"
                    f"{update.target_version} epoch={epoch + 1}/"
                    f"{schedule.epochs_per_update} step={step + 1}/"
                    f"{len(batches)} loss={loss:.6f} elapsed="
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        totals = torch.tensor(
            [local_loss_sum, local_targets],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[1].item() <= 0:
            raise RuntimeError("XP multiversion update has no targets")
        epochs.append(
            {
                "epoch": epoch + 1,
                "global_target_mean_loss": float(
                    (totals[0] / totals[1]).item()
                ),
                "global_targets": int(totals[1].item()),
                "minimum_step_global_targets": minimum_global_targets,
                "maximum_step_global_targets": maximum_global_targets,
            }
        )
    delta = global_tracker_delta(tracker, before)
    stability = _numeric_stability(
        dense,
        embedding,
        tracker,
        before,
        device,
    )
    return {
        "phase": phase,
        "source_version": update.source_version,
        "target_version": update.target_version,
        "epochs": epochs,
        "learning_rates": asdict(candidate),
        "weight_decay": schedule.weight_decay,
        "negative_count": schedule.train_negatives,
        "negative_seed_base": seed_base,
        "wall_seconds": time.perf_counter() - started,
        "numerical_stability": stability,
    }, delta


def spec_num_prediction_items(dense: ExternalEmbeddingHSTU) -> int:
    return dense.cfg.num_prediction_items


def _checkpoint_binding(
    root: Path,
    version: int,
    manifest: dict[str, object],
) -> dict[str, object]:
    path = root / f"theta_{version}" / "manifest.json"
    return {
        "version": version,
        "root": str(root),
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "global_active_rows": manifest["optimizer_active_rows"][
            "global_active_rows"
        ],
    }


def _source_binding() -> dict[str, object]:
    paths = (
        Path(__file__),
        Path("src/hstu_kvcache/streaming/xp_multiversion.py"),
        Path("src/hstu_kvcache/streaming/xp_version_training.py"),
        Path("src/hstu_kvcache/streaming/xp_projected_edge.py"),
    )
    return {
        path.name: {"path": str(path), "sha256": file_sha256(path)}
        for path in paths
    }


def _hbm_report(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "device_total_bytes": 0,
        }
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_total_bytes": torch.cuda.get_device_properties(
            device
        ).total_memory,
    }


def _base_result(
    args: argparse.Namespace,
    schedule: XPMultiversionSchedule,
    spec: XPProjectedModelSpec,
    validated,
    coverage: dict[str, object],
    backend: str,
    world_size: int,
    started: float,
    base_binding: dict[str, object],
    screen: dict[str, object],
    edges: list[dict[str, object]],
    status: str,
) -> dict[str, object]:
    quality_edges = {
        f"theta{edge['source_version']}_to_theta{edge['target_version']}": (
            edge["quality_signal"]
        )
        for edge in edges
    }
    return {
        "protocol": XP_MULTIVERSION_PROTOCOL,
        "status": status,
        "downstream_d1_d2_gate_passed": status == "complete",
        "scientific_result": False,
        "formal_design2": False,
        "formal_design3": False,
        "stack_identity": schedule.stack_identity,
        "legacy_two_edge_result_compatible": False,
        "execution": {
            "device": args.device,
            "backend": backend,
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size_per_rank,
            "total_wall_seconds": time.perf_counter() - started,
            "development_canary": args.development_canary,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "schedule": {
            "path": str(schedule.path),
            "file_sha256": schedule.file_sha256,
            "semantic_sha256": schedule.semantic_sha256,
            "document": schedule.document,
        },
        "model": asdict(spec),
        "corpus_audit": validated.audit,
        "coverage": coverage,
        "leakage_contract": {
            "train_role": schedule.split_roles["train"],
            "tuning_role": schedule.split_roles["tuning"],
            "quality_role": schedule.split_roles["quality"],
            "split_users_pairwise_disjoint": True,
            "update_windows_contiguous_and_nonoverlapping": True,
            "tuning_used_only_for_lr_selection_and_update_gate": True,
            "quality_used_for_selection": False,
            "quality_controls_training": False,
        },
        "learning_rate_screen": screen,
        "updates": edges,
        "all_update_qualification": {
            "role": "quality",
            "selection_or_gate_input": False,
            "per_edge": quality_edges,
            "reported_edges": len(quality_edges),
        },
        "base_checkpoint": base_binding,
        "checkpoint_root": str(args.checkpoint_root),
        "ledger_dir": str(args.ledger_dir),
        "source_code": _source_binding(),
    }


def _prequential_observation(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    prepared: dict[str, object],
    schedule: XPMultiversionSchedule,
    device: torch.device,
) -> dict[str, object]:
    evaluation = prepared["evaluation"]
    return {
        "model_version": evaluation.model_version,
        "history_end": evaluation.history_end,
        "evaluation_end": evaluation.evaluation_end,
        "evaluation_width": evaluation.width,
        "semantics": "next_unseen_window",
        "tuning_observation": {
            "role": schedule.split_roles["tuning"],
            "used_for_selection_or_gate": False,
            "candidate_sha256_per_rank": prepared[
                "tuning_candidate_sha256_per_rank"
            ],
            "metrics": evaluate_fixed_heldout(
                dense,
                embedding,
                prepared["tuning"],
                device,
            ),
        },
        "quality_observation": {
            "role": schedule.split_roles["quality"],
            "used_for_selection_or_gate": False,
            "candidate_sha256_per_rank": prepared[
                "quality_candidate_sha256_per_rank"
            ],
            "metrics": evaluate_fixed_heldout(
                dense,
                embedding,
                prepared["quality"],
                device,
            ),
        },
    }


def _run_prequential_chain(
    args: argparse.Namespace,
    schedule: XPMultiversionSchedule,
    spec: XPProjectedModelSpec,
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    validated,
    base_binding: dict[str, object],
    backend: str,
    rank: int,
    world_size: int,
    device: torch.device,
    started: float,
) -> dict[str, object] | None:
    prepared_updates, prepared_evaluations, coverage = (
        _prepare_prequential(
            validated,
            spec,
            schedule,
            rank=rank,
            world_size=world_size,
            batch_size_per_rank=args.batch_size_per_rank,
        )
    )
    candidate = next(
        value
        for value in schedule.learning_rate_candidates
        if value.name == schedule.fixed_learning_rate_name
    )
    optimizers = _build_optimizers(
        dense,
        embedding,
        candidate,
        schedule,
    )
    observations = [
        _prequential_observation(
            dense,
            embedding,
            prepared_evaluations[0],
            schedule,
            device,
        )
    ]
    edges = []
    current_binding = base_binding
    for index, prepared in enumerate(prepared_updates):
        update = prepared["update"]
        training, active_delta = _train_update(
            dense,
            embedding,
            tracker,
            prepared["train"],
            candidate,
            schedule,
            update,
            edge_index=index,
            rank=rank,
            world_size=world_size,
            device=device,
            progress_every=args.progress_every,
            phase="prequential_stream_update",
            optimizers=optimizers,
        )
        observation = _prequential_observation(
            dense,
            embedding,
            prepared_evaluations[index + 1],
            schedule,
            device,
        )
        stability_passed = bool(
            training["numerical_stability"]["passed"]
            and active_delta["global_updated_rows"] > 0
        )
        edge = {
            "source_version": update.source_version,
            "target_version": update.target_version,
            "training_window": {
                "history_end": update.history_end,
                "update_end": update.update_end,
                "update_width": update.width,
            },
            "training": training,
            "optimizer_active_delta": active_delta,
            "optimizer_state_continuity": {
                "dense_adamw": "continuous_within_round",
                "projection_adamw": "continuous_within_round",
                "embedding_sgd": "stateless_continuous_within_round",
                "optimizer_consumed_through": update.update_end,
            },
            "checkpoint_admission": {
                "policy": schedule.admission_policy,
                "ranking_metrics_used": False,
                "numerical_stability_passed": training[
                    "numerical_stability"
                ]["passed"],
                "global_updated_rows_nonzero": (
                    active_delta["global_updated_rows"] > 0
                ),
                "passed": stability_passed,
            },
            "target_prequential_evaluation": observation,
            "target_checkpoint_committed": False,
        }
        if not stability_passed:
            edges.append(edge)
            result = None
            if rank == 0:
                result = {
                    "protocol": schedule.protocol,
                    "status": "numerical_stability_admission_failed",
                    "downstream_d1_d2_gate_passed": False,
                    "scientific_result": False,
                    "formal_design2": False,
                    "formal_design3": False,
                    "stack_identity": schedule.stack_identity,
                    "execution": {
                        "device": args.device,
                        "backend": backend,
                        "world_size": world_size,
                        "batch_size_per_rank": args.batch_size_per_rank,
                        "total_wall_seconds": time.perf_counter() - started,
                        "visible_devices": os.environ.get(
                            "CUDA_VISIBLE_DEVICES", ""
                        ),
                    },
                    "schedule": {
                        "path": str(schedule.path),
                        "file_sha256": schedule.file_sha256,
                        "semantic_sha256": schedule.semantic_sha256,
                        "document": schedule.document,
                    },
                    "model": asdict(spec),
                    "corpus_audit": validated.audit,
                    "coverage": coverage,
                    "prequential_evaluations": observations,
                    "updates": edges,
                    "base_checkpoint": base_binding,
                    "checkpoint_root": str(args.checkpoint_root),
                    "ledger_dir": str(args.ledger_dir),
                    "source_code": _source_binding(),
                }
                _atomic_json(args.output, result)
            dist.barrier()
            return result
        provenance = {
            "protocol": schedule.protocol,
            "stack_identity": schedule.stack_identity,
            "scientific_result": False,
            "formal_design2": False,
            "formal_design3": False,
            "schedule": {
                "path": str(schedule.path),
                "file_sha256": schedule.file_sha256,
                "semantic_sha256": schedule.semantic_sha256,
            },
            "corpus_audit": validated.audit,
            "parent_checkpoint": current_binding,
            "edge": {**edge, "target_checkpoint_committed": True},
            "learning_rate_policy": {
                "mode": "predeclared_fixed",
                "candidate": asdict(candidate),
                "ranking_metrics_used_for_selection": False,
            },
            "source_code": _source_binding(),
        }
        manifest = save_xp_projected_checkpoint(
            args.checkpoint_root,
            update.target_version,
            spec,
            dense,
            embedding,
            tracker,
            provenance=provenance,
        )
        current_binding = _checkpoint_binding(
            args.checkpoint_root,
            update.target_version,
            manifest,
        )
        edge["target_checkpoint_committed"] = True
        edge["checkpoint"] = current_binding
        edges.append(edge)
        observations.append(observation)
        if rank == 0:
            _atomic_json(
                args.ledger_dir
                / f"version_{update.target_version:05d}.json",
                {
                    "protocol": schedule.protocol,
                    "status": "committed",
                    "stack_identity": schedule.stack_identity,
                    "schedule_semantic_sha256": schedule.semantic_sha256,
                    "edge": edge,
                    "checkpoint": current_binding,
                    "checkpoint_admission_is_quality_independent": True,
                },
            )
        dist.barrier()
    local_report = {
        "rank": rank,
        "local_embedding_rows": embedding.local_rows,
        "local_cumulative_active_rows": tracker.local_active_count,
        "memory": _hbm_report(device),
    }
    rank_reports: list[object] = [None] * world_size
    dist.all_gather_object(rank_reports, local_report)
    result = None
    if rank == 0:
        result = {
            "protocol": schedule.protocol,
            "status": "complete",
            "downstream_d1_d2_gate_passed": True,
            "downstream_gate_semantics": (
                "numerical stability, nonzero model update, and committed "
                "checkpoint only; ranking quality is reported but never gates"
            ),
            "scientific_result": False,
            "formal_design2": False,
            "formal_design3": False,
            "stack_identity": schedule.stack_identity,
            "execution": {
                "device": args.device,
                "backend": backend,
                "world_size": world_size,
                "batch_size_per_rank": args.batch_size_per_rank,
                "total_wall_seconds": time.perf_counter() - started,
                "development_canary": args.development_canary,
                "visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", ""
                ),
            },
            "schedule": {
                "path": str(schedule.path),
                "file_sha256": schedule.file_sha256,
                "semantic_sha256": schedule.semantic_sha256,
                "document": schedule.document,
            },
            "model": asdict(spec),
            "corpus_audit": validated.audit,
            "coverage": coverage,
            "learning_rate_policy": {
                "mode": "predeclared_fixed",
                "candidate": asdict(candidate),
                "selection_role": "none",
                "ranking_metrics_used_for_selection": False,
            },
            "checkpoint_admission": {
                "policy": schedule.admission_policy,
                "ranking_metrics_used": False,
            },
            "prequential_evaluations": observations,
            "updates": edges,
            "base_checkpoint": base_binding,
            "final_checkpoint": current_binding,
            "checkpoint_root": str(args.checkpoint_root),
            "ledger_dir": str(args.ledger_dir),
            "ranks": rank_reports,
            "source_code": _source_binding(),
        }
        _atomic_json(args.output, result)
    dist.barrier()
    return result


def run(args: argparse.Namespace) -> dict[str, object] | None:
    schedule = load_xp_multiversion_schedule(args.schedule)
    _validate_outputs(args, schedule)
    rank, world_size, device, backend = _init_runtime(args)
    started = time.perf_counter()
    try:
        _seed_everything(schedule.training_seed)
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.cuda.reset_peak_memory_stats(device)
        spec, dense, embedding, tracker, base_binding = _load_base(
            args.base_checkpoint_root,
            schedule.base_version,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        if not args.development_canary and spec != CANONICAL_SPEC:
            raise ValueError("canonical XP multiversion geometry differs")
        validated = validate_xp_multiversion_corpus(schedule, spec)
        if schedule.protocol == XP_PREQUENTIAL_MULTIVERSION_PROTOCOL:
            return _run_prequential_chain(
                args,
                schedule,
                spec,
                dense,
                embedding,
                tracker,
                validated,
                base_binding,
                backend,
                rank,
                world_size,
                device,
                started,
            )
        prepared, coverage = _prepare_edges(
            validated,
            spec,
            schedule,
            rank=rank,
            world_size=world_size,
            batch_size_per_rank=args.batch_size_per_rank,
        )
        first = prepared[0]
        source_state = _capture_state(dense, embedding, tracker)
        tuning_before = evaluate_fixed_heldout(
            dense,
            embedding,
            first["tuning"],
            device,
        )
        candidate_reports = []
        for index, candidate in enumerate(
            schedule.learning_rate_candidates
        ):
            if index:
                _restore_state(source_state, dense, embedding, tracker)
            training, active_delta = _train_update(
                dense,
                embedding,
                tracker,
                first["train"],
                candidate,
                schedule,
                first["update"],
                edge_index=0,
                rank=rank,
                world_size=world_size,
                device=device,
                progress_every=args.progress_every,
                phase=f"lr_screen_{candidate.name}",
            )
            tuning_after = evaluate_fixed_heldout(
                dense,
                embedding,
                first["tuning"],
                device,
            )
            candidate_reports.append(
                {
                    "candidate": candidate.name,
                    "learning_rates": asdict(candidate),
                    "training": training,
                    "optimizer_active_delta": active_delta,
                    "tuning_signal": qualification_signal(
                        tuning_before,
                        tuning_after,
                        schedule.minimum_tuning_cross_entropy_reduction,
                    ),
                }
            )
        winner = select_learning_rate_candidate(
            schedule.learning_rate_candidates,
            candidate_reports,
            schedule.minimum_tuning_cross_entropy_reduction,
        )
        screen = {
            "edge_index": 0,
            "selection_role": "tuning",
            "quality_observed_during_screen": False,
            "primary_metric": "sampled_cross_entropy_reduction",
            "minimum_cross_entropy_reduction": (
                schedule.minimum_tuning_cross_entropy_reduction
            ),
            "candidate_order": [
                value.name for value in schedule.learning_rate_candidates
            ],
            "candidate_order_sha256": hashlib.sha256(
                json.dumps(
                    [asdict(value) for value in schedule.learning_rate_candidates],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "candidates": candidate_reports,
            "winner": None if winner is None else asdict(winner),
            "gate_passed": winner is not None,
        }
        if winner is None:
            result = None
            if rank == 0:
                result = _base_result(
                    args,
                    schedule,
                    spec,
                    validated,
                    coverage,
                    backend,
                    world_size,
                    started,
                    base_binding,
                    screen,
                    [],
                    "learning_rate_screen_positive_signal_gate_failed",
                )
                _atomic_json(args.output, result)
            dist.barrier()
            return result
        edges = []
        current_binding = base_binding
        for index, value in enumerate(prepared):
            update = value["update"]
            if index == 0:
                _restore_state(source_state, dense, embedding, tracker)
                del source_state
            tuning_source = evaluate_fixed_heldout(
                dense,
                embedding,
                value["tuning"],
                device,
            )
            quality_source = evaluate_fixed_heldout(
                dense,
                embedding,
                value["quality"],
                device,
            )
            training, active_delta = _train_update(
                dense,
                embedding,
                tracker,
                value["train"],
                winner,
                schedule,
                update,
                edge_index=index,
                rank=rank,
                world_size=world_size,
                device=device,
                progress_every=args.progress_every,
                phase="committed_candidate",
            )
            tuning_target = evaluate_fixed_heldout(
                dense,
                embedding,
                value["tuning"],
                device,
            )
            quality_target = evaluate_fixed_heldout(
                dense,
                embedding,
                value["quality"],
                device,
            )
            tuning = qualification_signal(
                tuning_source,
                tuning_target,
                schedule.minimum_tuning_cross_entropy_reduction,
            )
            quality = qualification_signal(
                quality_source,
                quality_target,
                0.0,
            )
            edge = {
                "source_version": update.source_version,
                "target_version": update.target_version,
                "history_end": update.history_end,
                "update_end": update.update_end,
                "training": training,
                "optimizer_active_delta": active_delta,
                "tuning_candidate_sha256_per_rank": value[
                    "tuning_candidate_sha256_per_rank"
                ],
                "quality_candidate_sha256_per_rank": value[
                    "quality_candidate_sha256_per_rank"
                ],
                "tuning_signal": tuning,
                "quality_signal": quality,
                "quality_used_for_selection_or_gate": False,
                "target_checkpoint_committed": False,
            }
            if not tuning["positive_signal_gate_passed"]:
                edges.append(edge)
                result = None
                if rank == 0:
                    result = _base_result(
                        args,
                        schedule,
                        spec,
                        validated,
                        coverage,
                        backend,
                        world_size,
                        started,
                        base_binding,
                        screen,
                        edges,
                        "update_tuning_positive_signal_gate_failed",
                    )
                    result["failed_update"] = {
                        "source_version": update.source_version,
                        "target_version": update.target_version,
                    }
                    _atomic_json(args.output, result)
                dist.barrier()
                return result
            provenance = {
                "protocol": XP_MULTIVERSION_PROTOCOL,
                "stack_identity": schedule.stack_identity,
                "scientific_result": False,
                "formal_design2": False,
                "formal_design3": False,
                "schedule": {
                    "path": str(schedule.path),
                    "file_sha256": schedule.file_sha256,
                    "semantic_sha256": schedule.semantic_sha256,
                },
                "corpus_audit": validated.audit,
                "split_contract": {
                    "train": schedule.split_roles["train"],
                    "tuning": schedule.split_roles["tuning"],
                    "quality": schedule.split_roles["quality"],
                    "quality_used_for_selection_or_gate": False,
                },
                "parent_checkpoint": current_binding,
                "edge": {
                    **edge,
                    "target_checkpoint_committed": True,
                },
                "learning_rate_screen": {
                    "winner": asdict(winner),
                    "candidate_order_sha256": screen[
                        "candidate_order_sha256"
                    ],
                },
                "source_code": _source_binding(),
            }
            manifest = save_xp_projected_checkpoint(
                args.checkpoint_root,
                update.target_version,
                spec,
                dense,
                embedding,
                tracker,
                provenance=provenance,
            )
            current_binding = _checkpoint_binding(
                args.checkpoint_root,
                update.target_version,
                manifest,
            )
            edge["target_checkpoint_committed"] = True
            edge["checkpoint"] = current_binding
            edges.append(edge)
            if rank == 0:
                _atomic_json(
                    args.ledger_dir
                    / f"version_{update.target_version:05d}.json",
                    {
                        "protocol": XP_MULTIVERSION_PROTOCOL,
                        "status": "committed",
                        "stack_identity": schedule.stack_identity,
                        "schedule_semantic_sha256": schedule.semantic_sha256,
                        "edge": edge,
                        "checkpoint": current_binding,
                        "downstream_gate_passed_through_version": True,
                    },
                )
            dist.barrier()
        local_report = {
            "rank": rank,
            "local_embedding_rows": embedding.local_rows,
            "local_cumulative_active_rows": tracker.local_active_count,
            "memory": _hbm_report(device),
        }
        rank_reports: list[object] = [None] * world_size
        dist.all_gather_object(rank_reports, local_report)
        result = None
        if rank == 0:
            result = _base_result(
                args,
                schedule,
                spec,
                validated,
                coverage,
                backend,
                world_size,
                started,
                base_binding,
                screen,
                edges,
                "complete",
            )
            result["ranks"] = rank_reports
            result["final_checkpoint"] = current_binding
            _atomic_json(args.output, result)
        dist.barrier()
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    result = run(parse_args())
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
