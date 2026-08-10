from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..models import HSTUConfig, HSTUKVCache
from .kuairand_query_multiversion import _admitted, _edge_config, _passing_metrics
from .kuairand_query_transition import (
    _atomic_json,
    _candidate_metrics_tie_aware,
    _collate,
    _collate_true_next_item,
    _relative_rows,
    _summary,
    _training_candidates,
    _true_next_item,
    build_workload,
    file_sha256,
    load_config,
)
from .multifield_projected import lookup_multifield_projected
from .sharded_edge import ExternalEmbeddingHSTU, modulo_local_rows
from .xp_projected_edge import (
    OptimizerActiveRowTracker,
    TrainableProjectedModuloEmbedding,
)

PROTOCOL = "evokv_kuairand_projected_scale_chain_v0"


def load_projected_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    model = document.get("model")
    training = document.get("training")
    evaluation = document.get("evaluation")
    execution = document.get("execution")
    selection = document.get("selection")
    transitions = document.get("transitions")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(
            isinstance(value, dict)
            for value in (parent, model, training, evaluation, execution, selection)
        )
        or not isinstance(transitions, list)
        or not transitions
        or int(model.get("hidden_size", 0)) < 64
        or int(model.get("embedding_width", 0)) < int(model.get("hidden_size", 0))
        or not 1 <= int(model.get("embedding_replicas", 1)) <= 16
        or not 1 <= int(model.get("embedding_capacity_multiplier", 1)) <= 16
        or (
            int(model.get("embedding_replicas", 1)) > 1
            and int(model.get("embedding_capacity_multiplier", 1)) > 1
        )
        or (
            (
                int(model.get("embedding_replicas", 1)) > 1
                or int(model.get("embedding_capacity_multiplier", 1)) > 1
            )
            and int(model.get("embedding_width", 0)) != int(model.get("hidden_size", 0))
        )
        or not 2 <= int(model.get("num_layers", 0)) <= 24
        or int(model.get("num_heads", 0)) < 1
        or int(model.get("hidden_size", 0)) % int(model.get("num_heads", 0)) != 0
        or int(training.get("calibration_examples", -1)) < 0
        or int(training.get("maximum_update_examples", 0)) < 1
        or int(training.get("update_epochs", 0)) < 1
        or min(
            float(training.get("embedding_lr", 0)),
            float(training.get("projection_lr", 0)),
            float(training.get("dense_lr", 0)),
        )
        <= 0
        or any(
            float(training.get(field, 1.0)) <= 0
            for field in (
                "calibration_embedding_lr",
                "calibration_projection_lr",
                "calibration_dense_lr",
                "calibration_kv_lr",
            )
        )
        or int(evaluation.get("targets_per_user", 0)) not in (2, 4, 8)
        or int(evaluation.get("candidate_count", 50)) not in (50, 100)
        or int(evaluation.get("lineage_depth", 1)) < 1
        or int(evaluation.get("lineage_depth", 1)) > len(transitions)
        or int(evaluation.get("local_batch_size", 0)) < 1
        or int(execution.get("world_size", 0)) not in (1, 2)
        or selection.get("metrics")
        not in (
            ["mrr", "ndcg_at_10", "hit_rate_at_10"],
            ["mrr", "ndcg_at_5", "hit_rate_at_5"],
        )
        or float(selection.get("minimum_relative_percent", 0)) != 3.0
        or int(selection.get("minimum_metrics", 0)) != 2
    ):
        raise ValueError("KuaiRand projected-scale config differs")
    for field in ("base_config", "theta0"):
        artifact = parent.get(field)
        artifact_path = Path(artifact.get("path", "")) if isinstance(artifact, dict) else Path()
        if (
            not isinstance(artifact, dict)
            or not artifact_path.is_file()
            or file_sha256(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError("KuaiRand projected-scale parent differs")
    base_document = json.loads(Path(parent["base_config"]["path"]).read_text())
    base_model = base_document.get("model", {})
    if (
        int(base_model.get("hidden_size", 0)) != int(model["hidden_size"])
        or int(base_model.get("num_layers", 0)) != int(model["num_layers"])
        or int(base_model.get("num_heads", 0)) != int(model["num_heads"])
    ):
        raise ValueError("KuaiRand projected-scale parent geometry differs")
    expected = 0
    for transition in transitions:
        if (
            int(transition.get("source_version", -1)) != expected
            or int(transition.get("target_version", -1)) != expected + 1
        ):
            raise ValueError("KuaiRand projected-scale transition differs")
        override = transition.get("training", {})
        if not isinstance(override, dict) or any(
            key not in {
                "update_epochs",
                "maximum_update_examples",
                "embedding_lr",
                "projection_lr",
                "dense_lr",
                "kv_lr",
            }
            for key in override
        ):
            raise ValueError("KuaiRand projected-scale transition training differs")
        expected += 1
    return document


def _distributed(document: dict[str, Any]) -> tuple[int, int, torch.device]:
    expected = int(document["execution"]["world_size"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != expected:
        raise RuntimeError("KuaiRand projected-scale world size differs")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and (
        dist.get_world_size() != world_size or dist.get_rank() != rank
    ):
        raise RuntimeError("KuaiRand projected-scale process group differs")
    return rank, world_size, device


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capacity_physical_ids(
    semantic_ids: torch.Tensor,
    multiplier: int,
) -> torch.Tensor:
    if multiplier == 1:
        return semantic_ids
    slots = torch.remainder(semantic_ids * 5 + 3, multiplier)
    mapped = (semantic_ids - 1) * multiplier + 1 + slots
    return torch.where(semantic_ids > 0, mapped, 0)


@torch.no_grad()
def _copy_semantic_weight_to_embedding(
    embedding: TrainableProjectedModuloEmbedding,
    source_weight: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    semantic_rows = int(getattr(embedding, "semantic_rows", source_weight.shape[0] - 1))
    replicas = int(getattr(embedding, "embedding_replicas", 1))
    capacity_multiplier = int(getattr(embedding, "embedding_capacity_multiplier", 1))
    if source_weight.shape[0] != semantic_rows + 1:
        raise ValueError("KuaiRand capacity source rows differ")
    embedding.local_weight.zero_()
    if capacity_multiplier > 1:
        for start in range(1, semantic_rows + 1, 65_536):
            end = min(semantic_rows + 1, start + 65_536)
            semantic_ids = torch.arange(start, end, dtype=torch.long)
            physical_ids = _capacity_physical_ids(semantic_ids, capacity_multiplier)
            owned = torch.remainder(physical_ids, world_size) == rank
            if bool(torch.any(owned)):
                local_ids = torch.div(
                    physical_ids[owned],
                    world_size,
                    rounding_mode="floor",
                ).to(device)
                values = source_weight.index_select(0, semantic_ids[owned]).to(device)
                embedding.local_weight.index_copy_(0, local_ids, values)
    else:
        for start in range(0, embedding.local_rows, 65_536):
            end = min(embedding.local_rows, start + 65_536)
            global_ids = torch.arange(start, end, dtype=torch.long) * world_size + rank
            semantic_ids = torch.remainder(global_ids - 1, semantic_rows) + 1
            values = source_weight.index_select(0, semantic_ids) / math.sqrt(replicas)
            values[global_ids == 0].zero_()
            embedding.local_weight[start:end, : source_weight.shape[1]].copy_(
                values.to(device)
            )
    if rank == 0:
        embedding.local_weight[0].zero_()


def _initialize_model(
    document: dict[str, Any],
    base_config: dict[str, Any],
    embedding_rows: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    width = int(document["model"]["embedding_width"])
    hidden = int(document["model"]["hidden_size"])
    embedding_replicas = int(document["model"].get("embedding_replicas", 1))
    embedding_capacity_multiplier = int(
        document["model"].get("embedding_capacity_multiplier", 1)
    )
    expansion = max(embedding_replicas, embedding_capacity_multiplier)
    num_embeddings = embedding_rows * expansion + 1
    local_rows = modulo_local_rows(num_embeddings, rank, world_size)
    local_bytes = local_rows * width * 4
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserve = int(document["execution"].get("minimum_workspace_bytes", 4 << 30))
    if free_bytes < local_bytes + reserve:
        raise RuntimeError("KuaiRand projected-scale HBM preflight failed")
    payload = torch.load(
        document["parent"]["theta0"]["path"], map_location="cpu", weights_only=True
    )
    source_state = payload["state_dict"]
    source_weight = source_state["item_emb.weight"]
    if (
        source_weight.shape != (embedding_rows + 1, hidden)
        or width < hidden
        or embedding_replicas < 1
        or embedding_capacity_multiplier < 1
        or (embedding_replicas > 1 and embedding_capacity_multiplier > 1)
        or (expansion > 1 and width != hidden)
    ):
        raise ValueError("KuaiRand projected-scale source embedding differs")
    generator = torch.Generator(device=device).manual_seed(
        int(document["model"]["embedding_seed"]) + rank * 1_000_003
    )
    local_weight = torch.empty(local_rows, width, dtype=torch.float32, device=device)
    local_weight.normal_(
        mean=0.0,
        std=float(document["model"]["extra_embedding_std"]),
        generator=generator,
    )
    projection = torch.zeros(hidden, width, dtype=torch.float32, device=device)
    projection[:, :hidden].copy_(torch.eye(hidden, device=device))
    if width > hidden:
        projection_generator = torch.Generator(device=device).manual_seed(
            int(document["model"]["projection_seed"])
        )
        projection[:, hidden:].normal_(
            mean=0.0,
            std=float(document["model"]["extra_projection_std"]),
            generator=projection_generator,
        )
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=local_weight,
        projection_weight=projection,
        num_embeddings=num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    embedding.semantic_rows = embedding_rows
    embedding.embedding_replicas = embedding_replicas
    embedding.embedding_capacity_multiplier = embedding_capacity_multiplier
    _copy_semantic_weight_to_embedding(
        embedding,
        source_weight,
        rank,
        world_size,
        device,
    )
    model_config = base_config["model"]
    core_config = HSTUConfig(
        num_items=embedding_rows,
        num_prediction_items=embedding_rows,
        num_behaviors=1,
        hidden_size=hidden,
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        head_dim=hidden // int(model_config["num_heads"]),
        max_seq_len=int(model_config["max_seq_len"]),
        input_dropout=float(model_config["input_dropout"]),
        activation=str(model_config["activation"]),
        qk_scale=float(model_config["qk_scale"]),
        gating=str(model_config["gating"]),
        block_variant=str(model_config.get("block_variant", "legacy")),
        relative_position_bias=bool(
            model_config.get("relative_position_bias", False)
        ),
        causal_diagonal=str(model_config.get("causal_diagonal", "inclusive")),
    )
    dense = ExternalEmbeddingHSTU(core_config).to(device)
    dense_state = {
        name: value for name, value in source_state.items() if name != "item_emb.weight"
    }
    dense.core.load_state_dict(dense_state, strict=True)
    dense.core.query_mode = str(
        base_config["model"].get("query_mode", "history_only_zero")
    )
    del payload, source_state, source_weight
    tracker = OptimizerActiveRowTracker(
        num_embeddings=num_embeddings, rank=rank, world_size=world_size
    )
    global_embedding_bytes = num_embeddings * width * 4
    projection_bytes = hidden * width * 4
    dense_bytes = sum(value.numel() * value.element_size() for value in dense.parameters())
    model_bytes = global_embedding_bytes + projection_bytes + dense_bytes
    return dense, embedding, tracker, {
        "num_embeddings": num_embeddings,
        "semantic_embedding_rows": embedding_rows,
        "embedding_replicas": embedding_replicas,
        "embedding_capacity_multiplier": embedding_capacity_multiplier,
        "embedding_capacity_mapping": "strided_hash_v0"
        if embedding_capacity_multiplier > 1
        else "identity",
        "optimizer_reachable_embedding_rows": embedding_rows,
        "cold_capacity_embedding_rows": num_embeddings - embedding_rows - 1,
        "lookup_fields": 2 * embedding_replicas,
        "embedding_width": width,
        "hidden_size": hidden,
        "global_embedding_bytes": global_embedding_bytes,
        "projection_bytes": projection_bytes,
        "dense_bytes": dense_bytes,
        "global_model_parameter_bytes": model_bytes,
        "global_model_parameter_gib": model_bytes / (1 << 30),
        "single_gpu_total_bytes": total_bytes,
        "single_gpu_parameter_overflow": model_bytes > total_bytes,
        "local_embedding_rows": local_rows,
        "local_embedding_bytes": local_bytes,
        "initial_free_bytes": free_bytes,
        "workspace_reserve_bytes": reserve,
    }


def _lookup(
    embedding: TrainableProjectedModuloEmbedding,
    item_ids: torch.Tensor,
    lengths: torch.Tensor,
    author_by_item: torch.Tensor,
) -> torch.Tensor:
    authors = author_by_item.index_select(0, item_ids.reshape(-1)).reshape_as(item_ids)
    features = torch.stack((item_ids, authors), dim=-1)
    replicas = int(getattr(embedding, "embedding_replicas", 1))
    capacity_multiplier = int(getattr(embedding, "embedding_capacity_multiplier", 1))
    semantic_rows = int(getattr(embedding, "semantic_rows", embedding.num_embeddings - 1))
    if capacity_multiplier > 1:
        features = _capacity_physical_ids(features, capacity_multiplier)
    if replicas > 1:
        offsets = (
            torch.arange(replicas, dtype=features.dtype, device=features.device)
            * semantic_rows
        )
        expanded = features.unsqueeze(-1) + offsets
        features = torch.where(features.unsqueeze(-1) > 0, expanded, 0).flatten(2)
    return lookup_multifield_projected(embedding, features, lengths)


def _forward_query(dense, vectors, behaviors, deltas, lengths):
    embedded = dense.core.combine_input_features(vectors, behaviors, deltas)
    if getattr(dense.core, "query_mode", "learned_token") == "history_only_zero":
        rows = torch.arange(len(lengths), device=embedded.device)
        embedded[rows, lengths - 1] = 0
    hidden, _ = dense.core.forward_embedded(embedded, lengths=lengths)
    return dense.core.last_hidden(hidden, lengths)


def _forward_true_next_item(dense, vectors, behaviors, deltas, lengths):
    embedded = dense.core.combine_input_features(vectors, behaviors, deltas)
    hidden, _ = dense.core.forward_embedded(embedded, lengths=lengths)
    return dense.core.last_hidden(hidden, lengths)


def _projected_query_embedding(
    dense,
    embedding: TrainableProjectedModuloEmbedding,
    batch_size: int,
    author_by_item: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    items = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    lengths = torch.ones(batch_size, dtype=torch.long, device=device)
    vectors = _lookup(embedding, items, lengths, author_by_item)
    result = dense.core.combine_input_features(
        vectors,
        torch.ones_like(items),
        torch.zeros_like(items, dtype=torch.float32),
    )
    if getattr(dense.core, "query_mode", "learned_token") == "history_only_zero":
        result.zero_()
    return result


def _score(
    hidden: torch.Tensor,
    candidates: torch.Tensor,
    dense,
    embedding: TrainableProjectedModuloEmbedding,
    author_by_item: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    lengths = torch.full(
        (len(candidates),), candidates.shape[1], dtype=torch.long, device=candidates.device
    )
    candidate_vectors = _lookup(embedding, candidates, lengths, author_by_item)
    return torch.einsum(
        "nh,nch->nc", F.normalize(hidden, dim=-1), F.normalize(candidate_vectors, dim=-1)
    ) / temperature


def _all_reduce_dense(dense) -> None:
    for parameter in dense.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not bool(torch.all(torch.isfinite(parameter.grad))):
            raise RuntimeError("KuaiRand projected-scale dense gradient differs")
        if dist.is_initialized():
            dist.all_reduce(parameter.grad)


def _sparse_step(
    embedding: TrainableProjectedModuloEmbedding,
    optimizer: torch.optim.Optimizer,
    tracker: OptimizerActiveRowTracker,
) -> int:
    gradient = embedding.local_weight.grad
    if gradient is None or not gradient.is_sparse:
        raise RuntimeError("KuaiRand projected-scale embedding gradient differs")
    gradient = gradient.coalesce()
    values = gradient.values()
    if not bool(torch.all(torch.isfinite(values))):
        raise RuntimeError("KuaiRand projected-scale embedding gradient is nonfinite")
    nonzero_rows = torch.any(values != 0, dim=1)
    rows = gradient.indices()[0][nonzero_rows]
    active_dimensions = int(torch.count_nonzero(torch.any(values != 0, dim=0)).item())
    optimizer.step()
    tracker.mark_local_rows(rows)
    if embedding.rank == 0 and embedding.local_rows:
        with torch.no_grad():
            embedding.local_weight[0].zero_()
    return active_dimensions


def _epoch_indices(
    total: int, maximum: int, global_batch: int, seed: int
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    if maximum:
        order = order[: min(maximum, len(order))]
    unique = len(order)
    padded = math.ceil(unique / global_batch) * global_batch
    if padded > unique:
        order = np.concatenate((order, np.resize(order, padded - unique)))
    return order, unique


def _recent_per_user_epoch_indices(
    examples,
    maximum: int,
    global_batch: int,
    seed: int,
    examples_per_user: int,
) -> tuple[np.ndarray, int]:
    grouped = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[int(example[0])].append(index)
    selected = np.asarray(
        [
            index
            for user in sorted(grouped)
            for index in grouped[user][-examples_per_user:]
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    selected = selected[rng.permutation(len(selected))]
    if maximum:
        selected = selected[: min(maximum, len(selected))]
    unique = len(selected)
    padded = math.ceil(unique / global_batch) * global_batch
    if padded > unique:
        selected = np.concatenate(
            (selected, np.resize(selected, padded - unique))
        )
    return selected, unique


def _train_epochs(
    dense,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    examples,
    workload: dict[str, Any],
    base_config: dict[str, Any],
    document: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
    phase: str,
    seed: int,
) -> dict[str, Any]:
    training = document["training"]
    epochs = 1 if phase == "calibration" else int(training["update_epochs"])
    maximum = int(
        training["calibration_examples"]
        if phase == "calibration"
        else training["maximum_update_examples"]
    )
    if maximum == 0:
        return {
            "phase": phase,
            "epochs": [],
            "unique_examples": 0,
            "processed_examples": 0,
            "minimum_active_embedding_dimensions": 0,
        }
    local_batch = int(training["local_batch_size"])
    global_batch = local_batch * world_size
    if phase == "calibration":
        dense_lr = float(training.get("calibration_dense_lr", training["dense_lr"]))
        kv_lr = float(training.get("calibration_kv_lr", dense_lr))
        projection_lr = float(
            training.get("calibration_projection_lr", training["projection_lr"])
        )
        embedding_lr = float(
            training.get("calibration_embedding_lr", training["embedding_lr"])
        )
    else:
        dense_lr = float(training["dense_lr"])
        kv_lr = float(training.get("kv_lr", dense_lr))
        projection_lr = float(training["projection_lr"])
        embedding_lr = float(training["embedding_lr"])
    dense_update_scope = str(training.get("dense_update_scope", "full"))
    if dense_update_scope not in ("full", "qkv_only", "frozen"):
        raise ValueError("KuaiRand projected dense update scope differs")
    named_dense = list(dense.named_parameters())
    trainable_names = {
        name
        for name, _ in named_dense
        if dense_update_scope == "full"
        or any(
            f".attn.{projection}." in name
            for projection in ("q_proj", "k_proj", "v_proj")
        )
        and dense_update_scope == "qkv_only"
    }
    for name, parameter in named_dense:
        parameter.requires_grad_(name in trainable_names)
    kv_parameters = [
        parameter
        for name, parameter in named_dense
        if name in trainable_names
        and (".attn.k_proj." in name or ".attn.v_proj." in name)
    ]
    non_kv_parameters = [
        parameter
        for name, parameter in named_dense
        if name in trainable_names
        and ".attn.k_proj." not in name
        and ".attn.v_proj." not in name
    ]
    if dense_update_scope != "frozen" and (
        not kv_parameters or not non_kv_parameters
    ):
        raise RuntimeError("KuaiRand projected dense parameter groups differ")
    dense_optimizer = (
        None
        if dense_update_scope == "frozen"
        else torch.optim.AdamW(
            [
                {"params": non_kv_parameters, "lr": dense_lr},
                {"params": kv_parameters, "lr": kv_lr},
            ],
            weight_decay=float(training["weight_decay"]),
            foreach=False,
        )
    )
    projection_optimizer = torch.optim.AdamW(
        [embedding.projection_weight],
        lr=projection_lr,
        weight_decay=float(training["weight_decay"]),
        foreach=False,
    )
    embedding_optimizer = torch.optim.SGD(
        [embedding.local_weight],
        lr=embedding_lr,
        momentum=0.0,
        weight_decay=0.0,
        foreach=False,
    )
    negative_size = min(
        int(base_config["training"]["negative_pool_size"]),
        len(workload["popular_ids"]),
    )
    negative_pool = torch.as_tensor(
        np.asarray(workload["popular_ids"][:negative_size]).copy(),
        dtype=torch.long,
        device=device,
    )
    rank_by_item = torch.as_tensor(
        workload["rank_by_item"], dtype=torch.long, device=device
    )
    author_by_item = torch.as_tensor(
        np.asarray(workload["author_by_item"]).copy(), dtype=torch.long, device=device
    )
    candidate_generator = torch.Generator(device=device).manual_seed(seed + rank * 1009)
    epoch_results = []
    minimum_dimensions = embedding.embedding_width
    total_processed = 0
    unique_examples = 0
    sampling = str(training.get("sampling", "global_random"))
    examples_per_user = int(training.get("examples_per_user", 0))
    final_epoch_examples = int(training.get("final_epoch_examples", 0))
    for epoch in range(epochs):
        epoch_maximum = (
            final_epoch_examples
            if final_epoch_examples > 0 and epoch == epochs - 1
            else maximum
        )
        if sampling == "global_random":
            order, unique = _epoch_indices(
                len(examples), epoch_maximum, global_batch, seed + epoch * 100003
            )
        elif sampling == "recent_per_user" and examples_per_user > 0:
            order, unique = _recent_per_user_epoch_indices(
                examples,
                epoch_maximum,
                global_batch,
                seed + epoch * 100003,
                examples_per_user,
            )
        else:
            raise ValueError("KuaiRand projected training sampling differs")
        unique_examples = unique
        loss_sum = 0.0
        steps = len(order) // global_batch
        epoch_minimum_dimensions = embedding.embedding_width
        epoch_target_horizon = 0
        for step in range(steps):
            global_indices = order[step * global_batch : (step + 1) * global_batch]
            local_indices = global_indices[rank * local_batch : (rank + 1) * local_batch]
            batch = [examples[int(index)] for index in local_indices]
            items, behaviors, deltas, lengths, targets = (
                _collate_true_next_item(batch, device)
                if _true_next_item(base_config)
                else _collate(batch, device)
            )
            if dense_optimizer is not None:
                dense_optimizer.zero_grad(set_to_none=True)
            projection_optimizer.zero_grad(set_to_none=True)
            embedding_optimizer.zero_grad(set_to_none=True)
            vectors = _lookup(embedding, items, lengths, author_by_item)
            hidden = (
                _forward_true_next_item(
                    dense, vectors, behaviors, deltas, lengths
                )
                if _true_next_item(base_config)
                else _forward_query(dense, vectors, behaviors, deltas, lengths)
            )
            target_horizon = targets.shape[1] if targets.ndim == 2 else 1
            if epoch_target_horizon not in (0, target_horizon):
                raise RuntimeError("KuaiRand projected target horizon differs")
            epoch_target_horizon = target_horizon
            if targets.ndim == 2:
                scoring_hidden = hidden.unsqueeze(1).expand(
                    -1, target_horizon, -1
                ).reshape(-1, hidden.shape[1])
                scoring_targets = targets.reshape(-1)
            else:
                scoring_hidden = hidden
                scoring_targets = targets
            candidates = _training_candidates(
                scoring_targets,
                negative_pool,
                int(base_config["training"]["negative_samples"]),
                candidate_generator,
                str(base_config["training"]["negative_source"]),
                embedding.num_embeddings - 1,
                rank_by_item,
            )
            scores = _score(
                scoring_hidden,
                candidates,
                dense,
                embedding,
                author_by_item,
                float(base_config["training"]["temperature"]),
            )
            loss = F.cross_entropy(
                scores,
                torch.zeros(
                    len(scoring_targets), dtype=torch.long, device=device
                ),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("KuaiRand projected-scale loss is nonfinite")
            (loss / world_size).backward()
            _all_reduce_dense(dense)
            projection_gradient = embedding.projection_weight.grad
            if projection_gradient is None or not bool(
                torch.all(torch.isfinite(projection_gradient))
            ):
                raise RuntimeError("KuaiRand projected-scale projection gradient differs")
            if dense_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in named_dense if parameter.requires_grad],
                    1.0,
                )
            torch.nn.utils.clip_grad_norm_([embedding.projection_weight], 1.0)
            if dense_optimizer is not None:
                dense_optimizer.step()
            projection_optimizer.step()
            active_dimensions = _sparse_step(embedding, embedding_optimizer, tracker)
            dimension_tensor = torch.tensor(active_dimensions, device=device)
            if dist.is_initialized():
                dist.all_reduce(dimension_tensor, op=dist.ReduceOp.MIN)
            epoch_minimum_dimensions = min(
                epoch_minimum_dimensions, int(dimension_tensor.item())
            )
            loss_tensor = torch.tensor(float(loss.detach().item()), device=device)
            if dist.is_initialized():
                dist.all_reduce(loss_tensor)
            loss_sum += float(loss_tensor.item()) / world_size
        total_processed += steps * global_batch
        minimum_dimensions = min(minimum_dimensions, epoch_minimum_dimensions)
        epoch_results.append(
            {
                "epoch": epoch + 1,
                "mean_local_batch_cross_entropy": loss_sum / steps,
                "steps": steps,
                "processed_examples": steps * global_batch,
                "effective_targets": steps
                * global_batch
                * epoch_target_horizon,
                "unique_examples": unique,
                "minimum_active_embedding_dimensions": epoch_minimum_dimensions,
            }
        )
        if rank == 0:
            print(
                f"phase=kuairand_projected_{phase} epoch={epoch + 1}/{epochs} "
                f"loss={loss_sum / steps:.6f} examples={unique}",
                flush=True,
            )
    return {
        "phase": phase,
        "epochs": epoch_results,
        "unique_examples": unique_examples,
        "processed_examples": total_processed,
        "minimum_active_embedding_dimensions": minimum_dimensions,
        "sampling": sampling,
        "examples_per_user": examples_per_user,
        "final_epoch_examples": final_epoch_examples,
        "dense_lr": dense_lr,
        "kv_lr": kv_lr,
        "dense_update_scope": dense_update_scope,
        "temperature": float(base_config["training"]["temperature"]),
    }


def _evaluation_batches(
    workload: dict[str, Any], local_batch: int, rank: int, world_size: int
) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for key in workload["evaluation_keys"]:
        grouped[len(workload["evaluation"][key]["history"])].append(key)
    batches = []
    global_batch = local_batch * world_size
    for history_length, keys in sorted(grouped.items()):
        for start in range(0, len(keys), global_batch):
            chunk = keys[start : start + global_batch]
            valid_count = len(chunk)
            while len(chunk) < global_batch:
                chunk.append(chunk[len(chunk) % valid_count])
            local_keys = chunk[rank * local_batch : (rank + 1) * local_batch]
            local_valid = [
                rank * local_batch + index < valid_count
                for index in range(local_batch)
            ]
            items = torch.stack(
                [
                    torch.as_tensor(
                        workload["evaluation"][key]["history"], dtype=torch.long
                    )
                    for key in local_keys
                ]
            )
            candidates = torch.stack(
                [torch.as_tensor(workload["candidate_maps"][key], dtype=torch.long) for key in local_keys]
            )
            batches.append(
                {
                    "keys": local_keys,
                    "valid": local_valid,
                    "history_length": history_length,
                    "items": items,
                    "candidates": candidates,
                }
            )
    return batches


@torch.no_grad()
def _capture_old(
    dense,
    embedding: TrainableProjectedModuloEmbedding,
    batches: list[dict[str, Any]],
    workload: dict[str, Any],
    base_config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    dense.eval()
    embedding.eval()
    author_by_item = torch.as_tensor(
        np.asarray(workload["author_by_item"]).copy(), dtype=torch.long, device=device
    )
    captured = []
    for batch in batches:
        items = batch["items"].to(device)
        candidates = batch["candidates"].to(device)
        lengths = torch.full(
            (len(items),), items.shape[1], dtype=torch.long, device=device
        )
        query_mode = base_config["model"].get("query_mode", "learned_token")
        if _true_next_item(base_config) or query_mode == "latest_item_query":
            prefix_items = items[:, :-1]
            latest_items = items[:, -1:]
            prefix_lengths = lengths - 1
            latest_lengths = torch.ones_like(lengths)
            prefix_vectors = _lookup(
                embedding, prefix_items, prefix_lengths, author_by_item
            )
            prefix_behaviors = torch.ones_like(prefix_items)
            prefix_deltas = torch.zeros_like(prefix_items, dtype=torch.float32)
            cache = dense.core.compute_kv_from_item_embeddings(
                prefix_vectors,
                prefix_behaviors,
                prefix_deltas,
                prefix_lengths,
            )
            latest_vectors = _lookup(
                embedding, latest_items, latest_lengths, author_by_item
            )
            latest_embedded = dense.core.combine_input_features(
                latest_vectors,
                torch.ones_like(latest_items),
                torch.zeros_like(latest_items, dtype=torch.float32),
            )
            if query_mode == "latest_item_query":
                suffix = torch.cat(
                    (
                        latest_embedded,
                        _projected_query_embedding(
                            dense,
                            embedding,
                            len(items),
                            author_by_item,
                            device,
                        ),
                    ),
                    dim=1,
                )
            else:
                suffix = latest_embedded
            hidden, _ = dense.core.forward_with_cache_embedded(cache, suffix)
        else:
            vectors = _lookup(embedding, items, lengths, author_by_item)
            behaviors = torch.ones_like(items)
            deltas = torch.zeros_like(items, dtype=torch.float32)
            cache = dense.core.compute_kv_from_item_embeddings(
                vectors, behaviors, deltas, lengths
            )
            query = _projected_query_embedding(
                dense,
                embedding,
                len(items),
                author_by_item,
                device,
            )
            hidden, _ = dense.core.forward_with_cache_embedded(cache, query)
        scores = _score(
            hidden[:, -1],
            candidates,
            dense,
            embedding,
            author_by_item,
            float(base_config["training"]["temperature"]),
        )
        metrics = _candidate_metrics_tie_aware(
            scores, torch.zeros(len(items), dtype=torch.long, device=device)
        )
        captured.append(
            {
                **batch,
                "cache": HSTUKVCache(
                    k=cache.k.detach().cpu(),
                    v=cache.v.detach().cpu(),
                    seq_len=cache.seq_len,
                ),
                "previous_metrics": {
                    metric: value.detach().cpu() for metric, value in metrics.items()
                },
            }
        )
    return captured


@torch.no_grad()
def _evaluate_captured(
    dense,
    embedding: TrainableProjectedModuloEmbedding,
    captured: list[dict[str, Any]],
    workload: dict[str, Any],
    base_config: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    dense.eval()
    embedding.eval()
    author_by_item = torch.as_tensor(
        np.asarray(workload["author_by_item"]).copy(), dtype=torch.long, device=device
    )
    details = []
    maximum_hidden_error = 0.0
    maximum_score_error = 0.0
    for batch in captured:
        items = batch["items"].to(device)
        candidates = batch["candidates"].to(device)
        lengths = torch.full(
            (len(items),), items.shape[1], dtype=torch.long, device=device
        )
        old_cache = HSTUKVCache(
            k=batch["cache"].k.to(device),
            v=batch["cache"].v.to(device),
            seq_len=batch["cache"].seq_len,
        )
        query_mode = base_config["model"].get("query_mode", "learned_token")
        if _true_next_item(base_config) or query_mode == "latest_item_query":
            prefix_items = items[:, :-1]
            latest_items = items[:, -1:]
            prefix_lengths = lengths - 1
            latest_lengths = torch.ones_like(lengths)
            prefix_vectors = _lookup(
                embedding, prefix_items, prefix_lengths, author_by_item
            )
            current_cache = dense.core.compute_kv_from_item_embeddings(
                prefix_vectors,
                torch.ones_like(prefix_items),
                torch.zeros_like(prefix_items, dtype=torch.float32),
                prefix_lengths,
            )
            latest_vectors = _lookup(
                embedding, latest_items, latest_lengths, author_by_item
            )
            latest_embedded = dense.core.combine_input_features(
                latest_vectors,
                torch.ones_like(latest_items),
                torch.zeros_like(latest_items, dtype=torch.float32),
            )
            if query_mode == "latest_item_query":
                query = _projected_query_embedding(
                    dense,
                    embedding,
                    len(items),
                    author_by_item,
                    device,
                )
                suffix = torch.cat((latest_embedded, query), dim=1)
            else:
                suffix = latest_embedded
            reuse_hidden, _ = dense.core.forward_with_cache_embedded(
                old_cache, suffix
            )
            fresh_hidden, _ = dense.core.forward_with_cache_embedded(
                current_cache, suffix
            )
            full_vectors = _lookup(embedding, items, lengths, author_by_item)
            full_embedded = dense.core.combine_input_features(
                full_vectors,
                torch.ones_like(items),
                torch.zeros_like(items, dtype=torch.float32),
            )
            if query_mode == "latest_item_query":
                full_embedded = torch.cat((full_embedded, query), dim=1)
                full_lengths = lengths + 1
            else:
                full_lengths = lengths
        else:
            vectors = _lookup(embedding, items, lengths, author_by_item)
            behaviors = torch.ones_like(items)
            deltas = torch.zeros_like(items, dtype=torch.float32)
            current_cache = dense.core.compute_kv_from_item_embeddings(
                vectors, behaviors, deltas, lengths
            )
            query = _projected_query_embedding(
                dense,
                embedding,
                len(items),
                author_by_item,
                device,
            )
            reuse_hidden, _ = dense.core.forward_with_cache_embedded(
                old_cache, query
            )
            fresh_hidden, _ = dense.core.forward_with_cache_embedded(
                current_cache, query
            )
            full_embedded = torch.cat(
                (
                    dense.core.combine_input_features(
                        vectors, behaviors, deltas
                    ),
                    query,
                ),
                dim=1,
            )
            full_lengths = lengths + 1
        empty_cache = HSTUKVCache(
            k=current_cache.k[:, :, :0], v=current_cache.v[:, :, :0], seq_len=0
        )
        no_prefix_hidden, _ = dense.core.forward_with_cache_embedded(
            empty_cache,
            suffix
            if _true_next_item(base_config) or query_mode == "latest_item_query"
            else query,
        )
        full_hidden, _ = dense.core.forward_embedded(
            full_embedded, lengths=full_lengths
        )
        method_hidden = {
            "reuse": reuse_hidden[:, -1],
            "recompute": full_hidden[:, -1],
            "no_prefix": no_prefix_hidden[:, -1],
        }
        method_scores = {
            method: _score(
                value,
                candidates,
                dense,
                embedding,
                author_by_item,
                float(base_config["training"]["temperature"]),
            )
            for method, value in method_hidden.items()
        }
        incremental_scores = _score(
            fresh_hidden[:, -1],
            candidates,
            dense,
            embedding,
            author_by_item,
            float(base_config["training"]["temperature"]),
        )
        maximum_hidden_error = max(
            maximum_hidden_error,
            float((fresh_hidden[:, -1] - full_hidden[:, -1]).abs().max().item()),
        )
        maximum_score_error = max(
            maximum_score_error,
            float((incremental_scores - method_scores["recompute"]).abs().max().item()),
        )
        metrics = {
            method: _candidate_metrics_tie_aware(
                scores, torch.zeros(len(items), dtype=torch.long, device=device)
            )
            for method, scores in method_scores.items()
        }
        fresh_scores = method_scores["recompute"]
        reuse_scores = method_scores["reuse"]
        topk = min(10, fresh_scores.shape[1])
        fresh_topk = torch.topk(fresh_scores, k=topk, dim=1).indices
        reuse_topk = torch.topk(reuse_scores, k=topk, dim=1).indices
        topk_overlap = (
            fresh_topk.unsqueeze(2) == reuse_topk.unsqueeze(1)
        ).any(dim=2).sum(dim=1).float() / topk
        fresh_log_probabilities = F.log_softmax(fresh_scores, dim=1)
        reuse_log_probabilities = F.log_softmax(reuse_scores, dim=1)
        fresh_probabilities = fresh_log_probabilities.exp()
        score_relative_error = torch.linalg.vector_norm(
            (reuse_scores - fresh_scores).double(), dim=1
        ) / torch.linalg.vector_norm(fresh_scores.double(), dim=1).clamp_min(1e-12)
        score_kl = (
            fresh_probabilities
            * (fresh_log_probabilities - reuse_log_probabilities)
        ).sum(dim=1)
        score_cosine = F.cosine_similarity(reuse_scores, fresh_scores, dim=1)
        hidden_cosine = F.cosine_similarity(
            reuse_hidden[:, -1], full_hidden[:, -1], dim=1
        )
        no_prefix_hidden = method_hidden["no_prefix"]
        fresh_history_hidden = full_hidden[:, -1].double() - no_prefix_hidden.double()
        reuse_history_hidden = reuse_hidden[:, -1].double() - no_prefix_hidden.double()
        hidden_history_denominator = fresh_history_hidden.square().sum(dim=1).clamp_min(1e-12)
        hidden_history_projection = (
            reuse_history_hidden * fresh_history_hidden
        ).sum(dim=1) / hidden_history_denominator
        hidden_history_orthogonal = torch.linalg.vector_norm(
            reuse_history_hidden
            - hidden_history_projection.unsqueeze(1) * fresh_history_hidden,
            dim=1,
        ) / torch.sqrt(hidden_history_denominator)
        no_prefix_scores = method_scores["no_prefix"].double()
        fresh_history_scores = fresh_scores.double() - no_prefix_scores
        reuse_history_scores = reuse_scores.double() - no_prefix_scores
        score_history_denominator = fresh_history_scores.square().sum(dim=1).clamp_min(1e-12)
        score_history_projection = (
            reuse_history_scores * fresh_history_scores
        ).sum(dim=1) / score_history_denominator
        score_history_orthogonal = torch.linalg.vector_norm(
            reuse_history_scores
            - score_history_projection.unsqueeze(1) * fresh_history_scores,
            dim=1,
        ) / torch.sqrt(score_history_denominator)
        cache_k_error_by_layer = _relative_rows(old_cache.k, current_cache.k)
        cache_v_error_by_layer = _relative_rows(old_cache.v, current_cache.v)
        cache_k_error = cache_k_error_by_layer.mean(dim=0)
        cache_v_error = cache_v_error_by_layer.mean(dim=0)
        hidden_error = torch.linalg.vector_norm(
            (reuse_hidden[:, -1] - full_hidden[:, -1]).double(), dim=1
        ) / torch.linalg.vector_norm(full_hidden[:, -1].double(), dim=1).clamp_min(1e-12)
        for row, key in enumerate(batch["keys"]):
            if not batch["valid"][row]:
                continue
            source = workload["evaluation"][key]
            details.append(
                {
                    "user_id": int(source["user_id"]),
                    "query_ordinal": int(source["query_ordinal"]),
                    "history_length": int(batch["history_length"]),
                    "cache_prefix_length": int(current_cache.seq_len),
                    "metrics": {
                        "previous_fresh": {
                            metric: float(batch["previous_metrics"][metric][row].item())
                            for metric in batch["previous_metrics"]
                        },
                        **{
                            method: {
                                metric: float(value[metric][row].item())
                                for metric in value
                            }
                            for method, value in metrics.items()
                        },
                    },
                    "cache_k_relative_error": float(cache_k_error[row].item()),
                    "cache_v_relative_error": float(cache_v_error[row].item()),
                    "cache_k_relative_error_by_layer": [
                        float(value)
                        for value in cache_k_error_by_layer[:, row].tolist()
                    ],
                    "cache_v_relative_error_by_layer": [
                        float(value)
                        for value in cache_v_error_by_layer[:, row].tolist()
                    ],
                    "hidden_relative_error": float(hidden_error[row].item()),
                    "fidelity": {
                        "hidden_cosine": float(hidden_cosine[row].item()),
                        "score_cosine": float(score_cosine[row].item()),
                        "score_relative_error": float(score_relative_error[row].item()),
                        "score_kl_from_fresh": float(score_kl[row].item()),
                        "top10_overlap_with_fresh": float(topk_overlap[row].item()),
                        "hidden_history_projection": float(
                            hidden_history_projection[row].item()
                        ),
                        "hidden_history_orthogonal_relative_error": float(
                            hidden_history_orthogonal[row].item()
                        ),
                        "score_history_projection": float(
                            score_history_projection[row].item()
                        ),
                        "score_history_orthogonal_relative_error": float(
                            score_history_orthogonal[row].item()
                        ),
                    },
                }
            )
    sanity_tensor = torch.tensor(
        [maximum_hidden_error, maximum_score_error], dtype=torch.float64, device=device
    )
    if dist.is_initialized():
        dist.all_reduce(sanity_tensor, op=dist.ReduceOp.MAX)
    gathered: list[Any] = [None for _ in range(world_size)]
    if dist.is_initialized():
        dist.all_gather_object(gathered, details)
    else:
        gathered[0] = details
    sanity = {
        "maximum_same_model_incremental_hidden_absolute_error": float(
            sanity_tensor[0].item()
        ),
        "maximum_same_model_incremental_score_absolute_error": float(
            sanity_tensor[1].item()
        ),
        "passed": bool(sanity_tensor[0].item() <= 1e-4 and sanity_tensor[1].item() <= 1e-4),
    }
    if rank != 0:
        return None, sanity
    evaluation = {
        "records": [record for shard in gathered for record in shard],
        "sanity": sanity,
    }
    return _summary(evaluation, base_config), evaluation


def _active_rows(tracker: OptimizerActiveRowTracker, device: torch.device) -> int:
    value = torch.tensor(tracker.local_active_count, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(value)
    return int(value.item())


def run_projected_chain(config_path: str | Path) -> dict[str, Any]:
    document = load_projected_config(config_path)
    rank, world_size, device = _distributed(document)
    output_root = Path(document["outputs"]["root"])
    result_path = output_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        validate_projected_result(result, document)
        return result
    seed = int(document["training"]["seed"])
    _seed(seed)
    base_config = load_config(document["parent"]["base_config"]["path"])
    edge_documents = []
    workloads = []
    for transition in document["transitions"]:
        edge_document = _edge_config(base_config, transition, 1.0)
        edge_document["data"]["evaluation_targets_per_user"] = int(
            document["evaluation"]["targets_per_user"]
        )
        edge_document["evaluation"]["candidate_count"] = int(
            document["evaluation"].get("candidate_count", 50)
        )
        edge_document["data"]["user_limit"] = document["data"].get("user_limit")
        edge_documents.append(edge_document)
        workloads.append(build_workload(edge_document))
    embedding_rows = int(workloads[0]["metadata"]["embedding_rows"])
    dense, embedding, tracker, geometry = _initialize_model(
        document, base_config, embedding_rows, rank, world_size, device
    )
    if bool(document["model"]["require_single_card_overflow"]) != bool(
        geometry["single_gpu_parameter_overflow"]
    ):
        raise RuntimeError("KuaiRand projected-scale capacity gate differs")
    started = time.monotonic()
    calibration = _train_epochs(
        dense,
        embedding,
        tracker,
        workloads[0]["base_examples"],
        workloads[0],
        base_config,
        document,
        rank,
        world_size,
        device,
        "calibration",
        seed + 1009,
    )
    edge_results = []
    lineage_captures: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    lineage_depth = int(document["evaluation"].get("lineage_depth", 1))
    for edge_index, (transition, workload, edge_document) in enumerate(
        zip(document["transitions"], workloads, edge_documents, strict=True)
    ):
        for target_edge in range(
            edge_index, min(len(workloads), edge_index + lineage_depth)
        ):
            target_batches = _evaluation_batches(
                workloads[target_edge],
                int(document["evaluation"]["local_batch_size"]),
                rank,
                world_size,
            )
            lineage_captures[target_edge][edge_index] = _capture_old(
                dense,
                embedding,
                target_batches,
                workloads[target_edge],
                base_config,
                device,
            )
            del target_batches
        active_before = tracker.local_bitmap.clone()
        edge_training_document = json.loads(json.dumps(document))
        edge_training_document["training"].update(transition.get("training", {}))
        training = _train_epochs(
            dense,
            embedding,
            tracker,
            workload["update_examples"],
            workload,
            base_config,
            edge_training_document,
            rank,
            world_size,
            device,
            "update",
            seed + 2003 + edge_index * 100003,
        )
        compact = None
        evaluation = None
        lineage = []
        for source_version, captured in sorted(lineage_captures.pop(edge_index).items()):
            lineage_compact, lineage_evaluation = _evaluate_captured(
                dense,
                embedding,
                captured,
                workload,
                edge_document,
                rank,
                world_size,
                device,
            )
            if rank == 0:
                assert lineage_compact is not None and lineage_evaluation is not None
                lineage_passing = _passing_metrics(
                    lineage_compact, document["selection"]
                )
                lineage_admitted = _admitted(
                    lineage_compact,
                    lineage_passing,
                    int(document["selection"]["minimum_metrics"]),
                )
                lineage.append(
                    {
                        "source_version": source_version,
                        "target_version": int(transition["target_version"]),
                        "cache_age": int(transition["target_version"])
                        - source_version,
                        "summary": lineage_compact,
                        "passing_metrics": lineage_passing,
                        "admitted": lineage_admitted,
                    }
                )
                if source_version == edge_index:
                    compact = lineage_compact
                    evaluation = lineage_evaluation
                else:
                    stale = lineage_compact["comparisons"]["recompute_over_reuse"]
                    print(
                        f"phase=kuairand_projected_lineage edge={edge_index} "
                        f"source={source_version} "
                        f"age={int(transition['target_version']) - source_version} "
                        f"mrr={stale['mrr']['relative_percent']:.3f}% "
                        f"ndcg5={stale['ndcg_at_5']['relative_percent']:.3f}% "
                        f"hr5={stale['hit_rate_at_5']['relative_percent']:.3f}% "
                        f"admitted={lineage_admitted}",
                        flush=True,
                    )
            del captured
        new_local = int(
            torch.count_nonzero((tracker.local_bitmap != 0) & (active_before == 0)).item()
        )
        new_tensor = torch.tensor(new_local, dtype=torch.int64, device=device)
        if dist.is_initialized():
            dist.all_reduce(new_tensor)
        cumulative_active = _active_rows(tracker, device)
        if rank == 0:
            assert compact is not None and evaluation is not None
            passing = _passing_metrics(compact, document["selection"])
            cell = {
                "edge_index": edge_index,
                "transition": transition,
                "workload": workload["metadata"],
                "training": training,
                "summary": compact,
                "lineage": lineage,
                "passing_metrics": passing,
                "admitted": _admitted(
                    compact, passing, int(document["selection"]["minimum_metrics"])
                ),
                "new_optimizer_active_rows": int(new_tensor.item()),
                "cumulative_optimizer_active_rows": cumulative_active,
                "records": evaluation["records"],
            }
            cell_path = output_root / "cells" / f"edge_{edge_index}.json"
            _atomic_json(cell_path, cell)
            edge_results.append(
                {
                    key: value for key, value in cell.items() if key != "records"
                }
                | {"path": str(cell_path), "sha256": file_sha256(cell_path)}
            )
            stale = compact["comparisons"]["recompute_over_reuse"]
            print(
                f"phase=kuairand_projected_edge edge={edge_index} "
                f"mrr={stale['mrr']['relative_percent']:.3f}% "
                f"ndcg10={stale['ndcg_at_10']['relative_percent']:.3f}% "
                f"hr10={stale['hit_rate_at_10']['relative_percent']:.3f}% "
                f"admitted={cell['admitted']}",
                flush=True,
            )
        del active_before
        torch.cuda.empty_cache()
    memory = {
        "rank": rank,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_name": torch.cuda.get_device_name(device),
    }
    memories: list[Any] = [None for _ in range(world_size)]
    if dist.is_initialized():
        dist.all_gather_object(memories, memory)
    else:
        memories[0] = memory
    if rank == 0:
        result = {
            "protocol": PROTOCOL,
            "round_id": document["round_id"],
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "parent": document["parent"],
            "geometry": geometry,
            "calibration": calibration,
            "edges": edge_results,
            "hardware": memories,
            "decision": {
                "all_edges_admitted": all(value["admitted"] for value in edge_results),
                "all_edges_have_admitted_lineage": all(
                    any(lineage["admitted"] for lineage in value["lineage"])
                    for value in edge_results
                ),
                "capacity_forced": geometry["single_gpu_parameter_overflow"],
                "next": "freeze_large_lineage_result"
                if all(
                    any(lineage["admitted"] for lineage in value["lineage"])
                    for value in edge_results
                )
                and (
                    geometry["single_gpu_parameter_overflow"]
                    or not document["model"]["require_single_card_overflow"]
                )
                else "revise_projected_training",
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        validate_projected_result(result, document)
        _atomic_json(result_path, result)
    if dist.is_initialized():
        dist.barrier()
        payload: list[Any] = [json.loads(result_path.read_text()) if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        dist.destroy_process_group()
        return result
    return json.loads(result_path.read_text())


def validate_projected_result(result: dict[str, Any], document: dict[str, Any]) -> None:
    edges = result.get("edges")
    if (
        result.get("protocol") != PROTOCOL
        or result.get("round_id") != document["round_id"]
        or result.get("status") != "complete"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or not isinstance(edges, list)
        or len(edges) != len(document["transitions"])
        or bool(result.get("geometry", {}).get("single_gpu_parameter_overflow"))
        != bool(document["model"]["require_single_card_overflow"])
    ):
        raise ValueError("KuaiRand projected-scale result differs")
    explicit_lineage = "lineage_depth" in document["evaluation"]
    lineage_depth = int(document["evaluation"].get("lineage_depth", 1))
    for edge_index, edge in enumerate(edges):
        path = Path(edge["path"])
        if (
            not path.is_file()
            or file_sha256(path) != edge["sha256"]
            or not edge.get("summary", {}).get("sanity", {}).get("passed")
        ):
            raise ValueError("KuaiRand projected-scale cell binding differs")
        if explicit_lineage:
            lineage = edge.get("lineage")
            expected_sources = list(
                range(max(0, edge_index + 1 - lineage_depth), edge_index + 1)
            )
            if (
                not isinstance(lineage, list)
                or [int(value.get("source_version", -1)) for value in lineage]
                != expected_sources
                or any(
                    int(value.get("target_version", -1)) != edge_index + 1
                    or int(value.get("cache_age", -1))
                    != edge_index + 1 - int(value["source_version"])
                    or not value.get("summary", {}).get("sanity", {}).get("passed")
                    for value in lineage
                )
            ):
                raise ValueError("KuaiRand projected-scale lineage binding differs")
