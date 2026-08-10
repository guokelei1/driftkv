from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.data.qk_theta0 import (
    build_rank_batch,
    epoch_record_order,
    file_sha256,
    load_qk_theta0_corpus,
)
from hstu_kvcache.streaming.sharded_edge import ExternalEmbeddingHSTU
from hstu_kvcache.streaming.xp_projected_edge import (
    OptimizerActiveRowTracker,
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
    load_xp_projected_checkpoint,
    save_xp_projected_checkpoint,
    sparse_embedding_sgd,
)
from hstu_kvcache.streaming.xp_version_training import (
    xp_projected_next_item_train_step,
)

PROTOCOL = "evokv_qk_theta0_next_item_training_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _runtime() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2 or not torch.cuda.is_available():
        raise RuntimeError("QK theta0 training requires exactly two CUDA ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("QK theta0 distributed identity differs")
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _model_spec(document: dict[str, object]) -> XPProjectedModelSpec:
    model = document["model"]
    return XPProjectedModelSpec(
        num_embeddings=int(model["num_embeddings"]),
        embedding_width=int(model["embedding_width"]),
        hidden_size=int(model["hidden_size"]),
        num_prediction_items=int(model["num_prediction_items"]),
        num_behaviors=int(model["num_behaviors"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        head_dim=int(model["head_dim"]),
        max_seq_len=int(model["max_seq_len"]),
    )


def _parameter_probe(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
) -> dict[str, list[float]]:
    dense_values = []
    for parameter in dense.parameters():
        if parameter.numel():
            dense_values.extend(
                float(value)
                for value in parameter.detach().reshape(-1)[:4].cpu().tolist()
            )
        if len(dense_values) >= 16:
            break
    projection = embedding.projection_weight.detach().reshape(-1)[:16].cpu()
    return {
        "dense": dense_values[:16],
        "projection": [float(value) for value in projection.tolist()],
    }


def _probe_delta(
    before: dict[str, list[float]],
    after: dict[str, list[float]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("dense", "projection"):
        left = np.asarray(before[name], dtype=np.float64)
        right = np.asarray(after[name], dtype=np.float64)
        result[name] = {
            "sample_values": len(left),
            "maximum_absolute_delta": float(np.max(np.abs(right - left))),
            "changed": bool(np.any(right != left)),
        }
    return result


def _memory_preflight(
    document: dict[str, object],
    spec: XPProjectedModelSpec,
    device: torch.device,
) -> dict[str, object]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    required = int(document["execution"]["minimum_free_hbm_bytes_per_rank"])
    if free_bytes < required:
        raise RuntimeError(
            f"QK theta0 rank has insufficient free HBM: {free_bytes} < {required}"
        )
    return {
        "device": str(device),
        "name": torch.cuda.get_device_name(device),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "required_free_bytes": required,
        "local_embedding_parameter_bytes": (
            ((spec.num_embeddings + 1) // 2) * spec.embedding_width * 4
        ),
    }


def _optimizers(
    document: dict[str, object],
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
    training = document["training"]
    dense_optimizer = torch.optim.AdamW(
        dense.parameters(),
        lr=float(training["dense_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        foreach=False,
    )
    projection_optimizer = torch.optim.AdamW(
        [embedding.projection_weight],
        lr=float(training["projection_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        foreach=False,
    )
    embedding_optimizer = sparse_embedding_sgd(
        embedding, float(training["embedding_learning_rate"])
    )
    return dense_optimizer, projection_optimizer, embedding_optimizer


def _resume_paths(
    checkpoint_root: Path,
    slot: int,
) -> tuple[Path, Path, Path, Path]:
    if slot not in (0, 1):
        raise ValueError("QK theta0 resume slot differs")
    work_root = checkpoint_root / ".theta_0_work"
    slot_root = work_root / f"slot_{slot}"
    directory = slot_root / "theta_0"
    return (
        work_root,
        slot_root,
        directory / "optimizer_resume.pt",
        directory / "training_state.json",
    )


def _resume_pointer(checkpoint_root: Path) -> Path:
    return checkpoint_root / ".theta_0_work" / "current.json"


def _save_resume(
    document: dict[str, object],
    spec: XPProjectedModelSpec,
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer],
    state: dict[str, object],
    *,
    rank: int,
    config_path: Path,
    corpus_binding: dict[str, object],
) -> dict[str, object]:
    protocol = str(document["protocol"])
    training = document["training"]
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    pointer_path = _resume_pointer(checkpoint_root)
    current = json.loads(pointer_path.read_text()) if pointer_path.is_file() else None
    slot = 0 if current is None else 1 - int(current["slot"])
    generation = 1 if current is None else int(current["generation"]) + 1
    work_root, slot_root, optimizer_path, state_path = _resume_paths(
        checkpoint_root, slot
    )
    manifest = save_xp_projected_checkpoint(
        slot_root,
        0,
        spec,
        dense,
        embedding,
        tracker,
        provenance={
            "protocol": protocol,
            "objective": "sampled next-item cross entropy",
            "target_policy": training.get(
                "target_policy", "all_effective"
            ),
            "targets_per_record": training.get("targets_per_record"),
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "corpus": corpus_binding,
        },
    )
    dense_optimizer, projection_optimizer, _ = optimizers
    if rank == 0:
        _atomic_torch_save(
            optimizer_path,
            {
                "protocol": protocol,
                "dense_optimizer": dense_optimizer.state_dict(),
                "projection_optimizer": projection_optimizer.state_dict(),
            },
        )
        payload = {
            **state,
            "protocol": protocol,
            "config_sha256": file_sha256(config_path),
            "corpus_file_sha256": corpus_binding["file_sha256"],
            "optimizer_resume": _artifact(optimizer_path),
            "checkpoint_manifest_sha256": file_sha256(
                slot_root / "theta_0" / "manifest.json"
            ),
            "resume_generation": generation,
            "resume_slot": slot,
        }
        _atomic_json(state_path, payload)
    dist.barrier()
    if rank == 0:
        _atomic_json(
            pointer_path,
            {
                "protocol": protocol,
                "generation": generation,
                "slot": slot,
                "state_sha256": file_sha256(state_path),
            },
        )
    dist.barrier()
    return manifest


def _load_resume(
    document: dict[str, object],
    spec: XPProjectedModelSpec,
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer],
    *,
    config_path: Path,
    corpus_file_sha256: str,
) -> dict[str, object] | None:
    protocol = str(document["protocol"])
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    pointer_path = _resume_pointer(checkpoint_root)
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text())
    slot = int(pointer.get("slot", -1))
    _, slot_root, optimizer_path, state_path = _resume_paths(
        checkpoint_root, slot
    )
    if (
        pointer.get("protocol") != protocol
        or not state_path.is_file()
        or file_sha256(state_path) != pointer.get("state_sha256")
    ):
        raise ValueError("QK theta0 resume pointer differs")
    state = json.loads(state_path.read_text())
    if (
        state.get("protocol") != protocol
        or state.get("config_sha256") != file_sha256(config_path)
        or state.get("corpus_file_sha256") != corpus_file_sha256
    ):
        raise ValueError("QK theta0 resume binding differs")
    descriptor = state.get("optimizer_resume")
    if (
        not isinstance(descriptor, dict)
        or optimizer_path.stat().st_size != int(descriptor.get("bytes", -1))
        or file_sha256(optimizer_path) != descriptor.get("sha256")
    ):
        raise ValueError("QK theta0 optimizer resume artifact differs")
    load_xp_projected_checkpoint(
        slot_root, 0, spec, dense, embedding, tracker
    )
    optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    dense_optimizer, projection_optimizer, _ = optimizers
    dense_optimizer.load_state_dict(optimizer["dense_optimizer"])
    projection_optimizer.load_state_dict(optimizer["projection_optimizer"])
    return state


def _global_active_rows(
    tracker: OptimizerActiveRowTracker,
    device: torch.device,
) -> int:
    value = torch.tensor(tracker.local_active_count, dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return int(value.item())


def _global_step_stat(
    loss: float,
    targets: int,
    device: torch.device,
) -> tuple[float, int]:
    value = torch.tensor(
        [loss * targets, targets], dtype=torch.float64, device=device
    )
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return float(value[0].item()), int(value[1].item())


def _write_result(path: Path, value: dict[str, object]) -> None:
    _atomic_json(path, value)


def run(config_path: Path) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    if (
        document.get("protocol") != PROTOCOL
        or document.get("training", {}).get(
            "target_policy", "all_effective"
        )
        != "all_effective"
    ):
        raise ValueError("QK theta0 training config protocol differs")
    protocol = str(document["protocol"])
    spec = _model_spec(document)
    corpus = load_qk_theta0_corpus(
        document["data"]["corpus"],
        num_embeddings=spec.num_embeddings,
        num_prediction_items=spec.num_prediction_items,
    )
    output = Path(document["outputs"]["result"])
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    final_checkpoint = checkpoint_root / "theta_0"
    if output.exists() or final_checkpoint.exists():
        raise FileExistsError("QK theta0 valid target already exists")
    rank, world_size, local_rank, device = _runtime()
    started = time.perf_counter()
    try:
        seed = int(document["training"]["seed"])
        _seed_everything(seed)
        torch.set_float32_matmul_precision("high")
        preflight = _memory_preflight(document, spec, device)
        dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
        embedding = TrainableProjectedModuloEmbedding.initialize(
            num_embeddings=spec.num_embeddings,
            embedding_width=spec.embedding_width,
            hidden_size=spec.hidden_size,
            rank=rank,
            world_size=world_size,
            device=device,
            embedding_seed=seed + 17,
            projection_seed=seed + 31,
        )
        tracker = OptimizerActiveRowTracker(
            num_embeddings=spec.num_embeddings,
            rank=rank,
            world_size=world_size,
        )
        optimizers = _optimizers(document, dense, embedding)
        resume = _load_resume(
            document,
            spec,
            dense,
            embedding,
            tracker,
            optimizers,
            config_path=config_path,
            corpus_file_sha256=corpus.file_sha256,
        )
        before = _parameter_probe(dense, embedding)
        execution = document["execution"]
        training = document["training"]
        batch_size = int(execution["batch_size_per_rank"])
        epochs = int(training["epochs"])
        bucket_size = int(execution["length_bucket_records"])
        global_batch = batch_size * world_size
        resume_every = int(execution["resume_every_steps"])
        maximum_steps = int(execution.get("maximum_steps", 0))
        next_epoch = int(resume.get("next_epoch", 0)) if resume else 0
        next_step = int(resume.get("next_step", 0)) if resume else 0
        epoch_stats = list(resume.get("epoch_stats", [])) if resume else []
        total_steps = int(resume.get("total_steps", 0)) if resume else 0
        stop_for_canary = False
        corpus_binding = {
            "path": str(corpus.path),
            "file_sha256": corpus.file_sha256,
            "content_sha256": corpus.content_sha256,
            "records": corpus.records,
            "tokens": corpus.tokens,
        }
        for epoch in range(next_epoch, epochs):
            order = epoch_record_order(
                corpus,
                seed=seed,
                epoch=epoch,
                bucket_size=bucket_size,
            )
            steps = (len(order) + global_batch - 1) // global_batch
            accumulator = (
                dict(epoch_stats[epoch])
                if epoch < len(epoch_stats)
                else {
                    "epoch": epoch + 1,
                    "global_loss_sum": 0.0,
                    "global_targets": 0,
                    "completed_steps": 0,
                    "steps": steps,
                }
            )
            start_step = next_step if epoch == next_epoch else 0
            for step in range(start_step, steps):
                left = step * global_batch + rank * batch_size
                right = min(left + batch_size, len(order))
                records = order[left:right] if left < len(order) else order[:0]
                batch = build_rank_batch(corpus, records, batch_size=batch_size)
                loss, local_targets, global_targets, _ = xp_projected_next_item_train_step(
                    dense,
                    embedding,
                    tracker,
                    batch,
                    optimizers[0],
                    optimizers[1],
                    optimizers[2],
                    device=device,
                    num_prediction_items=spec.num_prediction_items,
                    negative_count=int(training["negative_count"]),
                    negative_seed=(
                        int(training["negative_seed"])
                        + epoch * 100_000_007
                        + step * world_size
                        + rank
                    ),
                )
                loss_sum, observed_targets = _global_step_stat(
                    loss, local_targets, device
                )
                if observed_targets != global_targets:
                    raise RuntimeError("QK theta0 target reduction differs")
                accumulator["global_loss_sum"] += loss_sum
                accumulator["global_targets"] += observed_targets
                accumulator["completed_steps"] = step + 1
                total_steps += 1
                should_report = (
                    step == start_step
                    or step + 1 == steps
                    or (step + 1) % int(execution["progress_every_steps"]) == 0
                )
                active_at_report = (
                    _global_active_rows(tracker, device) if should_report else 0
                )
                if rank == 0 and should_report:
                    mean = accumulator["global_loss_sum"] / max(
                        1, accumulator["global_targets"]
                    )
                    print(
                        f"epoch={epoch + 1}/{epochs} step={step + 1}/{steps} "
                        f"loss={mean:.6f} active={active_at_report:,}",
                        flush=True,
                    )
                if epoch < len(epoch_stats):
                    epoch_stats[epoch] = accumulator
                else:
                    epoch_stats.append(accumulator)
                is_epoch_end = step + 1 == steps
                state = {
                    "complete": False,
                    "next_epoch": epoch + 1 if is_epoch_end else epoch,
                    "next_step": 0 if is_epoch_end else step + 1,
                    "total_steps": total_steps,
                    "epoch_stats": epoch_stats,
                }
                should_save_resume = (
                    (step + 1) % resume_every == 0
                    and not (is_epoch_end and epoch + 1 == epochs)
                ) or (is_epoch_end and epoch + 1 < epochs)
                if bool(execution["commit_checkpoint"]) and should_save_resume:
                    observed = _probe_delta(
                        before, _parameter_probe(dense, embedding)
                    )
                    state["dense_update_observed"] = bool(
                        (resume or {}).get("dense_update_observed", False)
                        or observed["dense"]["changed"]
                    )
                    state["projection_update_observed"] = bool(
                        (resume or {}).get("projection_update_observed", False)
                        or observed["projection"]["changed"]
                    )
                    _save_resume(
                        document,
                        spec,
                        dense,
                        embedding,
                        tracker,
                        optimizers,
                        state,
                        rank=rank,
                        config_path=config_path,
                        corpus_binding=corpus_binding,
                    )
                if maximum_steps and total_steps >= maximum_steps:
                    stop_for_canary = True
                    break
            next_step = 0
            if stop_for_canary:
                break
        active_rows = _global_active_rows(tracker, device)
        after = _parameter_probe(dense, embedding)
        probe = _probe_delta(before, after)
        dense_update_observed = bool(
            (resume or {}).get("dense_update_observed", False)
            or probe["dense"]["changed"]
        )
        projection_update_observed = bool(
            (resume or {}).get("projection_update_observed", False)
            or probe["projection"]["changed"]
        )
        dense_parameters = sum(parameter.numel() for parameter in dense.parameters())
        fixed_bytes = (
            spec.global_embedding_bytes_fp32
            + spec.projection_bytes_fp32
            + dense_parameters * 4
        )
        minimum_active = int(document["data"]["minimum_optimizer_active_rows"])
        gate_passed = active_rows >= minimum_active
        active_admission = str(
            execution.get("optimizer_active_admission", "required")
        )
        if active_admission not in ("required", "report_only"):
            raise ValueError("QK theta0 optimizer-active admission differs")
        checkpoint_admitted = gate_passed or active_admission == "report_only"
        canary = not bool(execution["commit_checkpoint"])
        status = "canary_complete" if canary else (
            "complete"
            if checkpoint_admitted
            else "optimizer_active_gate_failed"
        )
        final_manifest = None
        if bool(execution["commit_checkpoint"]):
            state = {
                "complete": checkpoint_admitted,
                "next_epoch": epochs,
                "next_step": 0,
                "total_steps": total_steps,
                "epoch_stats": epoch_stats,
                "optimizer_active_rows": active_rows,
                "optimizer_active_gate_passed": gate_passed,
                "dense_update_observed": dense_update_observed,
                "projection_update_observed": projection_update_observed,
            }
            final_manifest = _save_resume(
                document,
                spec,
                dense,
                embedding,
                tracker,
                optimizers,
                state,
                rank=rank,
                config_path=config_path,
                corpus_binding=corpus_binding,
            )
            if checkpoint_admitted:
                dist.barrier()
                if rank == 0:
                    pointer = json.loads(
                        _resume_pointer(checkpoint_root).read_text()
                    )
                    work_root, slot_root, _, _ = _resume_paths(
                        checkpoint_root, int(pointer["slot"])
                    )
                    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    (slot_root / "theta_0").replace(final_checkpoint)
                    shutil.rmtree(work_root)
                dist.barrier()
        hbm = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        result = {
            "protocol": protocol,
            "status": status,
            "scientific_result": False,
            "formal_result": False,
            "dataset": "tenrec-qk",
            "objective": "sampled next-item cross entropy",
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "corpus": corpus_binding,
            "model": {
                "spec": asdict(spec),
                "dense_parameters": dense_parameters,
                "fixed_model_bytes_fp32": fixed_bytes,
                "fixed_model_gib_fp32": fixed_bytes / (1 << 30),
                "embedding_layout": "modulo row-sharded FP32",
                "projection_layout": "replicated owner-side E4096-to-H1536 FP32",
            },
            "training": {
                "candidate_name": None,
                "target_policy": training.get(
                    "target_policy", "all_effective"
                ),
                "targets_per_record": training.get(
                    "targets_per_record"
                ),
                "expected_targets_per_epoch": training.get(
                    "expected_targets_per_epoch"
                ),
                "epochs": epochs,
                "epochs_completed": len(epoch_stats),
                "total_steps": total_steps,
                "epoch_stats": [
                    {
                        **value,
                        "global_mean_loss": value["global_loss_sum"]
                        / max(1, value["global_targets"]),
                    }
                    for value in epoch_stats
                ],
                "dense_optimizer": "AdamW",
                "projection_optimizer": "AdamW",
                "embedding_optimizer": "owner-local sparse SGD without momentum",
                "optimizer_state_continuity": bool(execution["commit_checkpoint"]),
            },
            "gates": {
                "minimum_optimizer_active_rows": minimum_active,
                "observed_optimizer_active_rows": active_rows,
                "optimizer_active_gate_passed": gate_passed,
                "optimizer_active_admission": active_admission,
                "checkpoint_admission_passed": checkpoint_admitted,
                "dense_probe_changed": dense_update_observed,
                "projection_probe_changed": projection_update_observed,
            },
            "parameter_probe": probe,
            "execution": {
                "world_size": world_size,
                "local_rank": local_rank,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "batch_size_per_rank": batch_size,
                "preflight": preflight,
                "hbm": hbm,
                "runtime_seconds": time.perf_counter() - started,
                "development_canary": canary,
            },
            "checkpoint": {
                "committed": bool(
                    checkpoint_admitted and execution["commit_checkpoint"]
                ),
                "root": str(checkpoint_root),
                "version": 0,
                "manifest": final_manifest,
                "optimizer_resume_retained": bool(execution["commit_checkpoint"]),
            },
        }
        if rank == 0:
            _write_result(output, result)
        if not canary and not checkpoint_admitted:
            raise RuntimeError("QK theta0 optimizer-active row gate failed")
        return result if rank == 0 else None
    finally:
        dist.destroy_process_group()


def main() -> None:
    result = run(parse_args().config)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
