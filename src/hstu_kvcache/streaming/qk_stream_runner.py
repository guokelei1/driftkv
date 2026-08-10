from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..data.qk_stream_chain import load_corpus
from .qk_stream_version import (
    QKStreamEdgeBinding,
    build_training_batch,
    cache_relative_error,
    candidate_batches,
    canonical_json_sha256,
    distributed_projected_candidate_scores,
    evaluation_suffix,
    file_sha256,
    fp16_storage_fp32_consumption,
    local_role_records,
    paired_quality_summary,
    prefix_inputs,
    snapshot_source_prefixes,
    summarize_record_scores,
    training_record_order,
    validate_binding_document,
)
from .sharded_edge import ExternalEmbeddingHSTU, modulo_local_rows
from .xp_projected_edge import (
    OptimizerActiveRowTracker,
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
    load_xp_projected_checkpoint,
    save_xp_projected_checkpoint,
    sparse_embedding_sgd,
)
from .xp_version_training import (
    global_tracker_delta,
    tracker_count_snapshot,
    xp_projected_next_item_train_step,
)


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


def _runtime() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != 2 or not torch.cuda.is_available():
        raise RuntimeError("QK stream edge requires two CUDA ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=device,
    )
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("QK stream distributed identity differs")
    return rank, world_size, local_rank, device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _manifest_spec(root: Path, version: int) -> XPProjectedModelSpec:
    manifest = json.loads((root / f"theta_{version}" / "manifest.json").read_text())
    return XPProjectedModelSpec(**manifest["spec"])


def _load_model(
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
    spec = _manifest_spec(root, version)
    dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=torch.empty(
            (
                modulo_local_rows(spec.num_embeddings, rank, world_size),
                spec.embedding_width,
            ),
            dtype=torch.float32,
            device=device,
        ),
        projection_weight=torch.empty(
            (spec.hidden_size, spec.embedding_width),
            dtype=torch.float32,
            device=device,
        ),
        num_embeddings=spec.num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    tracker = OptimizerActiveRowTracker(
        num_embeddings=spec.num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    manifest = load_xp_projected_checkpoint(
        root, version, spec, dense, embedding, tracker
    )
    return spec, dense, embedding, tracker, manifest


def _release_model(*values: object) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def _dense_state(dense: ExternalEmbeddingHSTU) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in dense.state_dict().items()
    }


def _parameter_probe(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
) -> dict[str, torch.Tensor]:
    def sampled(parameters: list[torch.Tensor]) -> torch.Tensor:
        values = []
        for parameter in parameters:
            flat = parameter.detach().reshape(-1)
            count = min(4, flat.numel())
            if count:
                indices = torch.linspace(
                    0,
                    flat.numel() - 1,
                    steps=count,
                    dtype=torch.float64,
                    device=flat.device,
                ).long()
                values.append(flat.index_select(0, indices).cpu())
        if not values:
            raise ValueError("QK parameter probe group is empty")
        return torch.cat(values).clone()

    named = list(dense.named_parameters())
    kv_parameters = [
        parameter
        for name, parameter in named
        if ".attn.k_proj." in name or ".attn.v_proj." in name
    ]
    non_kv_parameters = [
        parameter
        for name, parameter in named
        if ".attn.k_proj." not in name and ".attn.v_proj." not in name
    ]
    dense_parameters = [parameter for _, parameter in named]
    projection_flat = embedding.projection_weight.detach().reshape(-1)
    projection_indices = torch.linspace(
        0,
        projection_flat.numel() - 1,
        steps=min(64, projection_flat.numel()),
        dtype=torch.float64,
        device=projection_flat.device,
    ).long()
    return {
        "dense": sampled(dense_parameters),
        "direct_kv": sampled(kv_parameters),
        "non_kv_dense": sampled(non_kv_parameters),
        "projection": projection_flat.index_select(
            0, projection_indices
        ).cpu().clone(),
    }


def _probe_delta(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, dict[str, object]]:
    result = {}
    for name in before:
        delta = (after[name] - before[name]).abs()
        result[name] = {
            "sample_values": len(delta),
            "changed": bool(torch.any(delta > 0).item()),
            "maximum_absolute_delta": float(delta.max().item()),
        }
    return result


def _training_evaluation_readiness(
    *,
    parameters_finite: bool,
    epoch_mean_losses: list[float],
    total_targets: int,
) -> dict[str, bool]:
    return {
        "all_dense_projection_parameters_finite": parameters_finite,
        "training_losses_finite": all(
            math.isfinite(value) for value in epoch_mean_losses
        ),
        "global_targets_positive": total_targets > 0,
    }


def _optimizers(
    binding: QKStreamEdgeBinding,
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    return (
        torch.optim.AdamW(
            dense.parameters(),
            lr=binding.dense_learning_rate,
            weight_decay=1e-4,
            foreach=False,
        ),
        torch.optim.AdamW(
            [embedding.projection_weight],
            lr=binding.projection_learning_rate,
            weight_decay=1e-4,
            foreach=False,
        ),
        sparse_embedding_sgd(
            embedding, binding.embedding_learning_rate
        ),
    )


def _load_optimizer_continuity(
    source_root: Path,
    source_version: int,
    optimizers: tuple[
        torch.optim.Optimizer,
        torch.optim.Optimizer,
        torch.optim.Optimizer,
    ],
) -> dict[str, object]:
    directory = source_root / f"theta_{source_version}"
    state_path = directory / "training_state.json"
    state = json.loads(state_path.read_text())
    descriptor = state.get("optimizer_resume")
    if not isinstance(descriptor, dict):
        raise ValueError("QK stream source optimizer descriptor is absent")
    optimizer_path = directory / str(descriptor.get("path", ""))
    if (
        not optimizer_path.is_file()
        or optimizer_path.stat().st_size != int(descriptor.get("bytes", -1))
        or file_sha256(optimizer_path) != descriptor.get("sha256")
    ):
        raise ValueError("QK stream source optimizer artifact differs")
    payload = torch.load(
        optimizer_path, map_location="cpu", weights_only=True
    )
    dense_optimizer, projection_optimizer, _ = optimizers
    dense_optimizer.load_state_dict(payload["dense_optimizer"])
    projection_optimizer.load_state_dict(payload["projection_optimizer"])
    return {
        "training_state_path": str(state_path),
        "training_state_sha256": file_sha256(state_path),
        "optimizer_resume_sha256": descriptor["sha256"],
    }


def _set_learning_rates(
    binding: QKStreamEdgeBinding,
    optimizers: tuple[
        torch.optim.Optimizer,
        torch.optim.Optimizer,
        torch.optim.Optimizer,
    ],
) -> None:
    for optimizer, rate in zip(
        optimizers,
        (
            binding.dense_learning_rate,
            binding.projection_learning_rate,
            binding.embedding_learning_rate,
        ),
        strict=True,
    ):
        for group in optimizer.param_groups:
            group["lr"] = rate


def _train_edge(
    document: dict[str, object],
    binding: QKStreamEdgeBinding,
    corpus,
    spec: XPProjectedModelSpec,
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[
    dict[str, object],
    tuple[
        torch.optim.Optimizer,
        torch.optim.Optimizer,
        torch.optim.Optimizer,
    ],
]:
    execution = document["execution"]
    batch_size = int(execution["batch_size_per_rank"])
    global_batch = batch_size * world_size
    bucket_records = int(execution["length_bucket_records"])
    progress_every = int(execution["progress_every_steps"])
    before_counts = tracker_count_snapshot(tracker)
    before_probe = _parameter_probe(dense, embedding)
    optimizers = _optimizers(binding, dense, embedding)
    continuity = _load_optimizer_continuity(
        Path(document["source_checkpoint"]["root"]),
        binding.source_version,
        optimizers,
    )
    _set_learning_rates(binding, optimizers)
    epochs = []
    total_steps = 0
    dense_gradient_observed = False
    projection_gradient_observed = False
    started = time.perf_counter()
    for epoch in range(binding.epochs):
        order = training_record_order(
            corpus,
            binding.edge,
            binding.training_seed,
            epoch,
            bucket_records,
        )
        steps = math.ceil(len(order) / global_batch)
        loss_sum = 0.0
        targets = 0
        for step in range(steps):
            left = step * global_batch + rank * batch_size
            right = min(left + batch_size, len(order))
            selected = order[left:right] if left < len(order) else order[:0]
            batch = build_training_batch(
                corpus, selected, batch_size, binding.edge
            )
            loss, local_targets, global_targets, _ = (
                xp_projected_next_item_train_step(
                    dense,
                    embedding,
                    tracker,
                    batch,
                    optimizers[0],
                    optimizers[1],
                    optimizers[2],
                    device=device,
                    num_prediction_items=spec.num_prediction_items,
                    negative_count=binding.train_negative_count,
                    negative_seed=(
                        binding.negative_seed
                        + epoch * 100_000_007
                        + step * world_size
                        + rank
                    ),
                )
            )
            if not dense_gradient_observed:
                dense_gradient_observed = any(
                    parameter.grad is not None
                    and bool(torch.any(parameter.grad != 0).item())
                    for parameter in dense.parameters()
                )
            if not projection_gradient_observed:
                gradient = embedding.projection_weight.grad
                projection_gradient_observed = bool(
                    gradient is not None
                    and torch.any(gradient != 0).item()
                )
            values = torch.tensor(
                [loss * local_targets, local_targets],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            if int(values[1].item()) != global_targets:
                raise RuntimeError("QK stream training target reduction differs")
            loss_sum += float(values[0].item())
            targets += int(values[1].item())
            total_steps += 1
            if rank == 0 and (
                step == 0
                or step + 1 == steps
                or (step + 1) % progress_every == 0
            ):
                print(
                    f"phase=qk_theta{binding.target_version}_train "
                    f"epoch={epoch + 1}/{binding.epochs} "
                    f"step={step + 1}/{steps} "
                    f"loss={loss_sum / max(1, targets):.6f} "
                    f"targets={targets:,}",
                    flush=True,
                )
        epochs.append(
            {
                "epoch": epoch + 1,
                "steps": steps,
                "global_targets": targets,
                "global_loss_sum": loss_sum,
                "global_mean_loss": loss_sum / targets,
            }
        )
    delta = global_tracker_delta(tracker, before_counts)
    probe = _probe_delta(before_probe, _parameter_probe(dense, embedding))
    local_finite = all(
        bool(torch.all(torch.isfinite(parameter)).item())
        for parameter in (
            *dense.parameters(),
            embedding.projection_weight,
        )
    )
    finite = torch.tensor(int(local_finite), dtype=torch.int64, device=device)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    gradients = torch.tensor(
        [dense_gradient_observed, projection_gradient_observed],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(gradients, op=dist.ReduceOp.MIN)
    total_targets = sum(value["global_targets"] for value in epochs)
    evaluation_readiness = _training_evaluation_readiness(
        parameters_finite=bool(finite.item()),
        epoch_mean_losses=[
            float(value["global_mean_loss"]) for value in epochs
        ],
        total_targets=total_targets,
    )
    diagnostics = {
        "optimizer_active_embedding_rows_positive": bool(
            delta["global_updated_rows"] > 0
        ),
        "dense_nonzero_gradient_observed_on_all_ranks": bool(
            gradients[0].item()
        ),
        "projection_nonzero_gradient_observed_on_all_ranks": bool(
            gradients[1].item()
        ),
        "dense_parameter_sample_changed": bool(probe["dense"]["changed"]),
        "direct_kv_parameter_sample_changed": bool(
            probe["direct_kv"]["changed"]
        ),
        "projection_parameter_sample_changed": bool(
            probe["projection"]["changed"]
        ),
    }
    admitted = bool(all(evaluation_readiness.values()))
    if rank == 0:
        print(
            f"phase=qk_theta{binding.target_version}_admission "
            f"evaluation_ready={admitted} readiness="
            f"{json.dumps(evaluation_readiness, sort_keys=True)} "
            f"diagnostics={json.dumps(diagnostics, sort_keys=True)} "
            f"dense_probe_max={probe['dense']['maximum_absolute_delta']:.9g} "
            f"direct_kv_probe_max="
            f"{probe['direct_kv']['maximum_absolute_delta']:.9g} "
            f"projection_probe_max="
            f"{probe['projection']['maximum_absolute_delta']:.9g} "
            f"updated_rows={delta['global_updated_rows']:,}",
            flush=True,
        )
    return {
        "source_version": binding.source_version,
        "target_version": binding.target_version,
        "edge": binding.edge,
        "candidate_name": binding.candidate_name,
        "epochs": epochs,
        "total_steps": total_steps,
        "optimizer_continuity": continuity,
        "optimizer_active_delta": delta,
        "parameter_probe": probe,
        "gradient_probe": {
            "dense_nonzero_observed_on_all_ranks": bool(
                gradients[0].item()
            ),
            "projection_nonzero_observed_on_all_ranks": bool(
                gradients[1].item()
            ),
        },
        "evaluation_readiness": evaluation_readiness,
        "evaluation_ready": admitted,
        "diagnostic_checks": diagnostics,
        "all_dense_projection_parameters_finite": bool(finite.item()),
        "checkpoint_admission_passed": admitted,
        "runtime_seconds": time.perf_counter() - started,
    }, optimizers


def _save_candidate(
    document: dict[str, object],
    config_path: Path,
    binding: QKStreamEdgeBinding,
    corpus,
    spec: XPProjectedModelSpec,
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    optimizers: tuple[
        torch.optim.Optimizer,
        torch.optim.Optimizer,
        torch.optim.Optimizer,
    ],
    training: dict[str, object],
    *,
    rank: int,
) -> dict[str, object]:
    work_root = Path(document["outputs"]["work_checkpoint_root"])
    manifest = save_xp_projected_checkpoint(
        work_root,
        binding.target_version,
        spec,
        dense,
        embedding,
        tracker,
        provenance={
            "protocol": document["protocol"],
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "source_checkpoint": document["source_checkpoint"],
            "corpus": {
                "path": str(corpus.path),
                "sha256": corpus.file_sha256,
                "content_sha256": corpus.content_sha256,
            },
            "edge": asdict(binding),
        },
    )
    directory = work_root / f"theta_{binding.target_version}"
    if rank == 0:
        optimizer_path = directory / "optimizer_resume.pt"
        _atomic_torch_save(
            optimizer_path,
            {
                "protocol": document["protocol"],
                "dense_optimizer": optimizers[0].state_dict(),
                "projection_optimizer": optimizers[1].state_dict(),
            },
        )
        _atomic_json(
            directory / "training_state.json",
            {
                "protocol": document["protocol"],
                "complete": True,
                "source_version": binding.source_version,
                "target_version": binding.target_version,
                "config_sha256": file_sha256(config_path),
                "corpus_file_sha256": corpus.file_sha256,
                "checkpoint_manifest_sha256": file_sha256(
                    directory / "manifest.json"
                ),
                "optimizer_resume": _artifact(optimizer_path),
                "edge_training": training,
            },
        )
    dist.barrier()
    return manifest


def _load_candidate_training(
    work_root: Path,
    target_version: int,
) -> dict[str, object]:
    state = json.loads(
        (work_root / f"theta_{target_version}" / "training_state.json").read_text()
    )
    training = state.get("edge_training")
    if state.get("complete") is not True or not isinstance(training, dict):
        raise ValueError("QK stream candidate training state differs")
    return training


def _broadcast(value: object, rank: int) -> object:
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


@torch.no_grad()
def _evaluate_role(
    document: dict[str, object],
    binding: QKStreamEdgeBinding,
    role: str,
    corpus,
    spec: XPProjectedModelSpec,
    current_dense: ExternalEmbeddingHSTU,
    current_embedding: TrainableProjectedModuloEmbedding,
    source_dense: ExternalEmbeddingHSTU,
    source_vectors: list[torch.Tensor],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    records = local_role_records(corpus, role, rank, world_size)
    if len(records) != len(source_vectors):
        raise ValueError("QK stream source snapshot role coverage differs")
    target_chunk = int(document["quality"]["target_chunk"])
    progress_every = int(document["execution"]["quality_progress_every_records"])
    local_results = []
    candidate_digest = hashlib.sha256()
    started = time.perf_counter()
    for ordinal, (raw_record, old_vectors) in enumerate(
        zip(records, source_vectors, strict=True)
    ):
        record = int(raw_record)
        prefix_items, prefix_behaviors, prefix_deltas, prefix_length = (
            prefix_inputs(corpus, record, binding.edge)
        )
        lengths = torch.tensor([prefix_length], dtype=torch.int64, device=device)
        current_vectors = current_embedding(
            prefix_items.unsqueeze(0).to(device), lengths
        )
        old_cache = source_dense.core.compute_kv_from_item_embeddings(
            old_vectors.unsqueeze(0).to(device),
            prefix_behaviors.unsqueeze(0).to(device),
            prefix_deltas.unsqueeze(0).to(device),
            lengths,
        )
        exact_cache = current_dense.core.compute_kv_from_item_embeddings(
            current_vectors,
            prefix_behaviors.unsqueeze(0).to(device),
            prefix_deltas.unsqueeze(0).to(device),
            lengths,
        )
        old_cache = fp16_storage_fp32_consumption(old_cache)
        exact_cache = fp16_storage_fp32_consumption(exact_cache)
        kv_error = cache_relative_error(old_cache, exact_cache)
        suffix_items, suffix_behaviors, suffix_deltas, targets, labels = (
            evaluation_suffix(corpus, record, binding.edge)
        )
        suffix_lengths = torch.tensor(
            [len(suffix_items)], dtype=torch.int64, device=device
        )
        suffix_vectors = current_embedding(
            suffix_items.unsqueeze(0).to(device), suffix_lengths
        )
        reuse_hidden, _ = current_dense.core.forward_with_cache_from_item_embeddings(
            old_cache,
            suffix_vectors,
            suffix_behaviors.unsqueeze(0).to(device),
            suffix_deltas.unsqueeze(0).to(device),
        )
        exact_hidden, _ = current_dense.core.forward_with_cache_from_item_embeddings(
            exact_cache,
            suffix_vectors,
            suffix_behaviors.unsqueeze(0).to(device),
            suffix_deltas.unsqueeze(0).to(device),
        )
        mask = labels.to(device)
        positive_ids = targets.to(device)[mask]
        reuse_positive = reuse_hidden[0][mask]
        exact_positive = exact_hidden[0][mask]
        maximum_targets = torch.tensor(
            len(positive_ids), dtype=torch.int64, device=device
        )
        dist.all_reduce(maximum_targets, op=dist.ReduceOp.MAX)
        local_batches, local_hash = candidate_batches(
            positive_ids,
            num_prediction_items=spec.num_prediction_items,
            negative_count=binding.quality_negative_count,
            seed=binding.quality_seed + binding.edge * 1_000_003 + record,
            target_chunk=target_chunk,
            device=device,
        )
        candidate_digest.update(
            np.asarray([record, binding.edge], dtype="<i8").tobytes()
        )
        candidate_digest.update(local_hash.encode())
        steps = math.ceil(int(maximum_targets.item()) / target_chunk)
        reuse_scores = []
        exact_scores = []
        for step in range(steps):
            if step < len(local_batches):
                candidates, real = local_batches[step]
            else:
                candidates = torch.zeros(
                    (target_chunk, binding.quality_negative_count + 1),
                    dtype=torch.int64,
                    device=device,
                )
                real = 0
            start = step * target_chunk
            hidden = torch.zeros(
                (2, target_chunk, spec.hidden_size),
                dtype=torch.float32,
                device=device,
            )
            hidden[0, :real] = reuse_positive[start : start + real]
            hidden[1, :real] = exact_positive[start : start + real]
            scores = distributed_projected_candidate_scores(
                current_embedding, hidden, candidates, real
            )
            reuse_scores.append(scores[0])
            exact_scores.append(scores[1])
        if reuse_scores:
            reuse_value = torch.cat(reuse_scores)
            exact_value = torch.cat(exact_scores)
        else:
            reuse_value = torch.empty(
                (0, binding.quality_negative_count + 1), device=device
            )
            exact_value = reuse_value.clone()
        metrics = summarize_record_scores(reuse_value, exact_value)
        metrics.update(
            {
                "record": record,
                "user_id": int(corpus.arrays["record_user_ids"][record]),
                "history_length": prefix_length,
                "append_length": len(suffix_items),
                "prefix_cache_relative_error": kv_error,
            }
        )
        local_results.append(metrics)
        del (
            current_vectors,
            old_cache,
            exact_cache,
            suffix_vectors,
            reuse_hidden,
            exact_hidden,
            reuse_value,
            exact_value,
        )
        if ordinal == 0 or (ordinal + 1) % progress_every == 0 or ordinal + 1 == len(records):
            print(
                f"phase=qk_theta{binding.target_version}_{role} "
                f"rank={rank} record={ordinal + 1}/{len(records)}",
                flush=True,
            )
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    digests: list[object] = [None] * world_size
    dist.all_gather_object(digests, candidate_digest.hexdigest())
    if rank != 0:
        return {}
    combined = [value for rank_values in gathered for value in rank_values]
    combined.sort(key=lambda value: int(value["record"]))
    summary = paired_quality_summary(
        combined,
        epsilon_ce=binding.quality_epsilon_ce,
        bootstrap_samples=binding.bootstrap_samples,
        bootstrap_seed=(
            binding.bootstrap_seed
            + binding.edge * 1_000_033
            + (0 if role == "fit_tuning" else 10_000_019)
        ),
    )
    errors = np.asarray(
        [value["prefix_cache_relative_error"] for value in combined],
        dtype=np.float64,
    )
    histories = np.asarray(
        [value["history_length"] for value in combined], dtype=np.int64
    )
    appends = np.asarray(
        [value["append_length"] for value in combined], dtype=np.int64
    )
    summary.update(
        {
            "role": role,
            "candidate_sha256": canonical_json_sha256(digests),
            "candidate_sha256_per_rank": digests,
            "endpoint": "FP16 cache storage followed by FP32 consumption",
            "scorer": "algebraically equivalent owner-sharded direct projected scorer",
            "natural_unpadded_het": True,
            "prefix_cache_relative_error": {
                "mean": float(errors.mean()),
                "median": float(np.median(errors)),
                "p95": float(np.quantile(errors, 0.95)),
            },
            "history_length": {
                "minimum": int(histories.min()),
                "median": float(np.median(histories)),
                "p95": float(np.quantile(histories, 0.95)),
                "maximum": int(histories.max()),
                "unique": int(len(np.unique(histories))),
            },
            "append_length": {
                "minimum": int(appends.min()),
                "median": float(np.median(appends)),
                "p95": float(np.quantile(appends, 0.95)),
                "maximum": int(appends.max()),
                "unique": int(len(np.unique(appends))),
            },
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    return summary


def _cleanup_work(path: Path, rank: int) -> None:
    dist.barrier()
    if rank == 0 and path.exists():
        shutil.rmtree(path)
    dist.barrier()


def run_edge(
    config_path: Path,
    binding: QKStreamEdgeBinding,
) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    validate_binding_document(document, binding)
    rank, world_size, local_rank, device = _runtime()
    started = time.perf_counter()
    try:
        _seed_everything(binding.training_seed)
        torch.set_float32_matmul_precision("high")
        output_path = Path(document["outputs"]["result"])
        final_root = Path(document["outputs"]["checkpoint_root"])
        work_root = Path(document["outputs"]["work_checkpoint_root"])
        final_checkpoint = final_root / f"theta_{binding.target_version}"
        work_checkpoint = work_root / f"theta_{binding.target_version}"
        if output_path.exists() or final_checkpoint.exists():
            raise FileExistsError("QK stream valid target already exists")
        if work_checkpoint.exists() and not (
            work_checkpoint / "manifest.json"
        ).is_file():
            _cleanup_work(work_root, rank)
        resumed_candidate = (work_checkpoint / "manifest.json").is_file()
        source = document["source_checkpoint"]
        source_root = Path(source["root"])
        source_manifest_path = (
            source_root / f"theta_{binding.source_version}" / "manifest.json"
        )
        if file_sha256(source_manifest_path) != source["manifest_sha256"]:
            raise ValueError("QK stream source checkpoint hash differs")
        data = document["data"]
        data_config = Path(data["config"])
        if file_sha256(data_config) != data["config_sha256"]:
            raise ValueError("QK stream data config hash differs")
        corpus = load_corpus(data["corpus"])
        summary_path = Path(data["summary"])
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("status") != "pass"
            or summary.get("content_sha256") != corpus.content_sha256
            or summary.get("artifact", {}).get("sha256")
            != corpus.file_sha256
        ):
            raise ValueError("QK stream data summary differs")
        spec, source_dense, source_embedding, source_tracker, source_manifest = (
            _load_model(
                source_root,
                binding.source_version,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        )
        if source_manifest != json.loads(source_manifest_path.read_text()):
            raise ValueError("QK stream source checkpoint changed during load")
        old_dense_state = _dense_state(source_dense)
        snapshots = snapshot_source_prefixes(
            corpus,
            source_embedding,
            ("fit_tuning", "qualification"),
            binding.edge,
            rank,
            world_size,
            device,
            int(document["execution"]["snapshot_batch_size_per_rank"]),
        )
        if resumed_candidate:
            training = _load_candidate_training(
                work_root, binding.target_version
            )
            del source_dense, source_embedding, source_tracker
            gc.collect()
            torch.cuda.empty_cache()
            spec, dense, embedding, tracker, candidate_manifest = _load_model(
                work_root,
                binding.target_version,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        else:
            dense = source_dense
            embedding = source_embedding
            tracker = source_tracker
            training, optimizers = _train_edge(
                document,
                binding,
                corpus,
                spec,
                dense,
                embedding,
                tracker,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if not training["checkpoint_admission_passed"]:
                raise RuntimeError(
                    "QK stream candidate is not evaluation-ready: "
                    + json.dumps(
                        training["evaluation_readiness"], sort_keys=True
                    )
                )
            candidate_manifest = _save_candidate(
                document,
                config_path,
                binding,
                corpus,
                spec,
                dense,
                embedding,
                tracker,
                optimizers,
                training,
                rank=rank,
            )
            del optimizers
            gc.collect()
            torch.cuda.empty_cache()
        source_dense_for_quality = ExternalEmbeddingHSTU(
            spec.hstu_config()
        ).to(device)
        source_dense_for_quality.load_state_dict(old_dense_state)
        source_dense_for_quality.eval()
        dense.eval()
        embedding.eval()
        tuning_local = _evaluate_role(
            document,
            binding,
            "fit_tuning",
            corpus,
            spec,
            dense,
            embedding,
            source_dense_for_quality,
            snapshots["fit_tuning"],
            rank=rank,
            world_size=world_size,
            device=device,
        )
        tuning = _broadcast(tuning_local if rank == 0 else None, rank)
        qualification = None
        if tuning["practical_gate_passed"]:
            qualification_local = _evaluate_role(
                document,
                binding,
                "qualification",
                corpus,
                spec,
                dense,
                embedding,
                source_dense_for_quality,
                snapshots["qualification"],
                rank=rank,
                world_size=world_size,
                device=device,
            )
            qualification = _broadcast(
                qualification_local if rank == 0 else None, rank
            )
        accepted = bool(
            tuning["practical_gate_passed"]
            and qualification is not None
            and qualification["practical_gate_passed"]
        )
        if accepted:
            dist.barrier()
            if rank == 0:
                final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                work_checkpoint.replace(final_checkpoint)
                if work_root.exists() and not any(work_root.iterdir()):
                    work_root.rmdir()
            dist.barrier()
            status = "complete_development_qualified"
        else:
            _cleanup_work(work_root, rank)
            status = (
                "complete_tuning_gate_failed"
                if not tuning["practical_gate_passed"]
                else "complete_qualification_gate_failed"
            )
        hbm = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        result = {
            "protocol": document["protocol"],
            "status": status,
            "scientific_result": False,
            "formal_result": False,
            "dataset": "tenrec-qk",
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "edge": asdict(binding),
            "source_checkpoint": {
                **source,
                "training_state_sha256": file_sha256(
                    source_root
                    / f"theta_{binding.source_version}"
                    / "training_state.json"
                ),
            },
            "data": {
                "config": data,
                "corpus_sha256": corpus.file_sha256,
                "corpus_content_sha256": corpus.content_sha256,
                "summary_sha256": file_sha256(summary_path),
            },
            "training": training,
            "quality": {
                "same_current_model": binding.target_version,
                "reuse_source_version": binding.source_version,
                "exact_version": binding.target_version,
                "tuning": tuning,
                "qualification": qualification,
                "qualification_consumed": qualification is not None,
                "final_consumed": False,
                "labels_used_for_routing": False,
            },
            "checkpoint": {
                "committed": accepted,
                "root": str(final_root),
                "version": binding.target_version,
                "manifest_sha256": (
                    file_sha256(final_checkpoint / "manifest.json")
                    if accepted
                    else None
                ),
                "optimizer_resume_retained": accepted,
                "candidate_manifest": candidate_manifest,
            },
            "execution": {
                "world_size": world_size,
                "local_rank": local_rank,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "hbm": hbm,
                "runtime_seconds": time.perf_counter() - started,
                "resumed_candidate": resumed_candidate,
            },
        }
        if rank == 0:
            _atomic_json(output_path, result)
        return result if rank == 0 else None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
