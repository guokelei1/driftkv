from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from ..models import HSTU, HSTUConfig
from .trainer import build_next_item_targets

SHARDED_EDGE_CHECKPOINT_SCHEMA = "evokv_sharded_edge_checkpoint_v1"


def modulo_local_rows(
    num_embeddings: int,
    rank: int,
    world_size: int,
) -> int:
    if num_embeddings < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid modulo embedding layout")
    if rank >= num_embeddings:
        return 0
    return (num_embeddings - 1 - rank) // world_size + 1


class _ModuloEmbeddingLookup(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        local_weight: torch.Tensor,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
        num_embeddings: int,
        rank: int,
        world_size: int,
        process_group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        if (
            local_weight.ndim != 2
            or not local_weight.is_floating_point()
            or item_ids.ndim != 2
            or lengths.shape != (item_ids.shape[0],)
            or item_ids.device != local_weight.device
            or lengths.device != item_ids.device
        ):
            raise ValueError("row-sharded embedding lookup shape differs")
        if local_weight.shape[0] != modulo_local_rows(
            num_embeddings,
            rank,
            world_size,
        ):
            raise ValueError("local embedding shard differs from modulo layout")
        width = item_ids.shape[1]
        lengths_long = lengths.long()
        if bool(torch.any(lengths_long < 0)) or bool(torch.any(lengths_long > width)):
            raise ValueError("embedding lookup lengths exceed item width")
        valid_width = torch.arange(width, device=item_ids.device).unsqueeze(
            0
        ) < lengths_long.unsqueeze(1)
        flat_ids = item_ids.reshape(-1).long()
        valid_positions = torch.nonzero(
            valid_width.reshape(-1),
            as_tuple=False,
        ).flatten()
        valid_ids = flat_ids.index_select(0, valid_positions)
        if valid_ids.numel() and (
            bool(torch.any(valid_ids < 0)) or bool(torch.any(valid_ids >= num_embeddings))
        ):
            raise ValueError("embedding lookup item id exceeds vocabulary")
        nonpadding = valid_ids > 0
        positions = valid_positions[nonpadding]
        requested_ids = valid_ids[nonpadding]
        owners = requested_ids.remainder(world_size)
        local_mask = owners == rank
        remote_mask = ~local_mask
        local_ids = requested_ids[local_mask]
        local_positions = positions[local_mask]
        local_rows = local_ids.div(world_size, rounding_mode="floor")
        vectors = torch.zeros(
            (item_ids.numel(), local_weight.shape[1]),
            dtype=local_weight.dtype,
            device=local_weight.device,
        )
        if local_rows.numel():
            vectors.index_copy_(
                0,
                local_positions,
                local_weight.index_select(0, local_rows),
            )
        if world_size == 1:
            ordered_positions = positions[:0]
            received_local_rows = local_rows[:0]
            send_splits = (0,)
            receive_splits = (0,)
        else:
            if (
                not dist.is_available()
                or not dist.is_initialized()
                or dist.get_world_size(group=process_group) != world_size
                or dist.get_rank(group=process_group) != rank
            ):
                raise RuntimeError("distributed process group differs from embedding layout")
            send_counts = torch.bincount(
                owners[remote_mask],
                minlength=world_size,
            ).to(device=item_ids.device, dtype=torch.int64)
            receive_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(
                receive_counts,
                send_counts,
                group=process_group,
            )
            remote_ids = requested_ids[remote_mask]
            remote_positions = positions[remote_mask]
            remote_owners = owners[remote_mask]
            order = torch.argsort(remote_owners, stable=True)
            ordered_ids = remote_ids.index_select(0, order)
            ordered_positions = remote_positions.index_select(0, order)
            send_local_rows = ordered_ids.div(
                world_size,
                rounding_mode="floor",
            ).contiguous()
            send_splits = tuple(int(value) for value in send_counts.tolist())
            receive_splits = tuple(int(value) for value in receive_counts.tolist())
            received_local_rows = torch.empty(
                int(receive_counts.sum().item()),
                dtype=torch.int64,
                device=item_ids.device,
            )
            dist.all_to_all_single(
                received_local_rows,
                send_local_rows,
                output_split_sizes=receive_splits,
                input_split_sizes=send_splits,
                group=process_group,
            )
            if received_local_rows.numel() and (
                bool(torch.any(received_local_rows < 0))
                or bool(torch.any(received_local_rows >= local_weight.shape[0]))
            ):
                raise RuntimeError("received embedding row exceeds shard")
            response_vectors = local_weight.index_select(
                0,
                received_local_rows,
            )
            received_vectors = torch.empty(
                (send_local_rows.numel(), local_weight.shape[1]),
                dtype=local_weight.dtype,
                device=local_weight.device,
            )
            dist.all_to_all_single(
                received_vectors,
                response_vectors,
                output_split_sizes=send_splits,
                input_split_sizes=receive_splits,
                group=process_group,
            )
            if ordered_positions.numel():
                vectors.index_copy_(
                    0,
                    ordered_positions,
                    received_vectors,
                )
        ctx.local_weight_shape = tuple(local_weight.shape)
        ctx.world_size = world_size
        ctx.process_group = process_group
        ctx.send_splits = send_splits
        ctx.receive_splits = receive_splits
        ctx.save_for_backward(
            local_rows,
            local_positions,
            ordered_positions,
            received_local_rows,
        )
        return vectors.reshape(*item_ids.shape, local_weight.shape[1])

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            local_rows,
            local_positions,
            ordered_positions,
            received_local_rows,
        ) = ctx.saved_tensors
        grad_flat = grad_output.reshape(
            -1,
            ctx.local_weight_shape[1],
        ).contiguous()
        local_values = grad_flat.index_select(0, local_positions)
        if ctx.world_size == 1:
            remote_values = local_values[:0]
        else:
            requested_remote_values = grad_flat.index_select(
                0,
                ordered_positions,
            ).contiguous()
            remote_values = torch.empty(
                (
                    received_local_rows.numel(),
                    ctx.local_weight_shape[1],
                ),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            dist.all_to_all_single(
                remote_values,
                requested_remote_values,
                output_split_sizes=ctx.receive_splits,
                input_split_sizes=ctx.send_splits,
                group=ctx.process_group,
            )
        rows = torch.cat([local_rows, received_local_rows])
        values = torch.cat([local_values, remote_values])
        if rows.numel():
            indices = rows.unsqueeze(0)
        else:
            indices = torch.empty(
                (1, 0),
                dtype=torch.int64,
                device=grad_output.device,
            )
            values = torch.empty(
                (0, ctx.local_weight_shape[1]),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
        with torch.sparse.check_sparse_tensor_invariants(enable=True):
            gradient = torch.sparse_coo_tensor(
                indices,
                values,
                size=ctx.local_weight_shape,
                dtype=grad_output.dtype,
                device=grad_output.device,
            ).coalesce()
        return gradient, None, None, None, None, None, None


class TrainableModuloRowShardedEmbedding(nn.Module):
    def __init__(
        self,
        local_weight: torch.Tensor,
        num_embeddings: int,
        rank: int,
        world_size: int,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        if (
            local_weight.ndim != 2
            or not local_weight.is_floating_point()
            or local_weight.shape[0] != modulo_local_rows(num_embeddings, rank, world_size)
        ):
            raise ValueError("trainable embedding shard differs from layout")
        self.num_embeddings = num_embeddings
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.local_weight = nn.Parameter(local_weight.contiguous())

    @classmethod
    def initialize(
        cls,
        num_embeddings: int,
        hidden_size: int,
        rank: int,
        world_size: int,
        device: torch.device | str,
        seed: int,
        std: float = 0.02,
        process_group: dist.ProcessGroup | None = None,
    ) -> TrainableModuloRowShardedEmbedding:
        if hidden_size < 1 or std <= 0:
            raise ValueError("invalid embedding initialization")
        target = torch.device(device)
        generator = torch.Generator(device=target)
        generator.manual_seed(seed + rank * 1_000_003)
        local_weight = torch.empty(
            (
                modulo_local_rows(num_embeddings, rank, world_size),
                hidden_size,
            ),
            dtype=torch.float32,
            device=target,
        )
        local_weight.normal_(mean=0.0, std=std, generator=generator)
        if rank == 0 and local_weight.shape[0]:
            local_weight[0].zero_()
        return cls(
            local_weight,
            num_embeddings,
            rank,
            world_size,
            process_group=process_group,
        )

    @property
    def hidden_size(self) -> int:
        return self.local_weight.shape[1]

    @property
    def local_rows(self) -> int:
        return self.local_weight.shape[0]

    def forward(
        self,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return _ModuloEmbeddingLookup.apply(
            self.local_weight,
            item_ids,
            lengths,
            self.num_embeddings,
            self.rank,
            self.world_size,
            self.process_group,
        )


class _UnavailableItemEmbedding(nn.Module):
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("dense edge model has no item embedding")

    def score(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        raise RuntimeError("dense edge model has no item embedding")


class ExternalEmbeddingHSTU(nn.Module):
    def __init__(self, cfg: HSTUConfig) -> None:
        super().__init__()
        seed_cfg = HSTUConfig(
            **{
                **asdict(cfg),
                "num_items": 1,
                "num_prediction_items": 1,
            }
        )
        core = HSTU(seed_cfg)
        core.cfg = cfg
        core.item_emb = _UnavailableItemEmbedding()
        self.cfg = cfg
        self.core = core

    def forward(
        self,
        item_vectors: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.core.combine_input_features(
            item_vectors,
            behaviors,
            time_deltas,
        )
        hidden, _ = self.core.forward_embedded(
            embedded,
            return_kv=False,
            return_hidden=True,
            lengths=lengths,
        )
        return hidden


@dataclass(frozen=True)
class ShardedEdgeModelSpec:
    num_embeddings: int
    num_prediction_items: int
    num_behaviors: int
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int

    def __post_init__(self) -> None:
        if (
            self.num_embeddings < 2
            or not 1 <= self.num_prediction_items < self.num_embeddings
            or self.num_behaviors < 1
            or self.hidden_size < 1
            or self.num_layers < 1
            or self.num_heads < 1
            or self.head_dim < 1
            or self.num_heads * self.head_dim != self.hidden_size
            or self.max_seq_len < 2
        ):
            raise ValueError("invalid sharded edge model specification")

    @property
    def num_items(self) -> int:
        return self.num_embeddings - 1

    @property
    def embedding_bytes_fp32(self) -> int:
        return self.num_embeddings * self.hidden_size * 4

    @property
    def kv_bytes_fp16_per_record(self) -> int:
        return 2 * self.num_layers * self.max_seq_len * self.hidden_size * 2

    def hstu_config(self) -> HSTUConfig:
        return HSTUConfig(
            num_items=self.num_items,
            num_prediction_items=self.num_prediction_items,
            num_behaviors=self.num_behaviors,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            activation="relu",
            input_dropout=0.0,
        )


def sparse_sgd(
    embedding: TrainableModuloRowShardedEmbedding,
    learning_rate: float,
) -> torch.optim.SGD:
    if learning_rate <= 0:
        raise ValueError("embedding learning rate must be positive")
    return torch.optim.SGD(
        embedding.parameters(),
        lr=learning_rate,
        momentum=0.0,
        weight_decay=0.0,
        foreach=False,
    )


def fixed_candidate_ids(
    positive_ids: torch.Tensor,
    num_prediction_items: int,
    negative_count: int,
    seed: int,
) -> torch.Tensor:
    if positive_ids.ndim != 1 or negative_count < 1 or num_prediction_items < 2:
        raise ValueError("invalid fixed candidate request")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    negatives = torch.randint(
        1,
        num_prediction_items + 1,
        (positive_ids.numel(), negative_count),
        generator=generator,
        dtype=torch.int64,
    ).to(positive_ids.device)
    positives = positive_ids.long().unsqueeze(1)
    negatives = torch.where(
        negatives == positives,
        negatives.remainder(num_prediction_items) + 1,
        negatives,
    )
    return torch.cat([positives, negatives], dim=1)


def sharded_edge_train_step(
    dense_model: nn.Module,
    embedding: TrainableModuloRowShardedEmbedding,
    batch: dict[str, torch.Tensor],
    dense_optimizer: torch.optim.Optimizer,
    embedding_optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_prediction_items: int,
    negative_count: int,
    negative_seed: int,
) -> tuple[float, int, int]:
    dense_model.train()
    embedding.train()
    item_ids = batch["item_ids"].to(device)
    behaviors = batch["behaviors"].to(device)
    time_deltas = batch["time_deltas"].to(device)
    lengths = batch["lengths"].to(device)
    labels = batch.get("labels")
    train_mask = batch.get("train_mask")
    labels = None if labels is None else labels.to(device)
    train_mask = None if train_mask is None else train_mask.to(device)
    dense_optimizer.zero_grad(set_to_none=True)
    embedding_optimizer.zero_grad(set_to_none=True)
    item_vectors = embedding(item_ids, lengths)
    hidden = dense_model(
        item_vectors,
        behaviors,
        time_deltas,
        lengths,
    )
    targets, valid = build_next_item_targets(
        item_ids,
        lengths,
        labels,
        train_mask,
    )
    positive_ids = targets[valid]
    candidates = fixed_candidate_ids(
        positive_ids,
        num_prediction_items,
        negative_count,
        negative_seed,
    )
    candidate_lengths = torch.full(
        (candidates.shape[0],),
        candidates.shape[1],
        dtype=torch.int64,
        device=device,
    )
    candidate_vectors = embedding(candidates, candidate_lengths)
    if positive_ids.numel():
        source_hidden = hidden[:, :-1][valid]
        logits = torch.einsum(
            "nh,nch->nc",
            source_hidden,
            candidate_vectors,
        )
        loss = torch.nn.functional.cross_entropy(
            logits,
            torch.zeros(
                positive_ids.numel(),
                dtype=torch.int64,
                device=device,
            ),
        )
    else:
        loss = hidden.sum() * 0.0 + candidate_vectors.sum() * 0.0
    local_targets = torch.tensor(
        positive_ids.numel(),
        dtype=torch.float64,
        device=device,
    )
    global_targets = local_targets.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(
            global_targets,
            op=dist.ReduceOp.SUM,
            group=embedding.process_group,
        )
    if global_targets.item() > 0:
        gradient_scale = embedding.world_size * local_targets / global_targets
    else:
        gradient_scale = torch.zeros_like(global_targets)
    (loss * gradient_scale.to(loss.dtype)).backward()
    dense_parameters = [value for value in dense_model.parameters() if value.grad is not None]
    if dense_parameters:
        torch.nn.utils.clip_grad_norm_(dense_parameters, 1.0)
    if embedding.local_weight.grad is None or not embedding.local_weight.grad.is_sparse:
        raise RuntimeError("embedding gradient must remain owner-local sparse")
    embedding.local_weight.grad._values().div_(embedding.world_size)
    dense_optimizer.step()
    embedding_optimizer.step()
    if embedding.rank == 0 and embedding.local_rows:
        with torch.no_grad():
            embedding.local_weight[0].zero_()
    return (
        float(loss.detach().item()),
        int(positive_ids.numel()),
        int(global_targets.item()),
    )


@dataclass(frozen=True)
class FixedHeldoutBatch:
    batch: dict[str, torch.Tensor]
    candidates: torch.Tensor


def make_fixed_heldout_batch(
    batch: dict[str, torch.Tensor],
    num_prediction_items: int,
    negative_count: int,
    seed: int,
) -> FixedHeldoutBatch:
    targets, valid = build_next_item_targets(
        batch["item_ids"],
        batch["lengths"],
        batch.get("labels"),
        batch.get("train_mask"),
    )
    candidates = fixed_candidate_ids(
        targets[valid],
        num_prediction_items,
        negative_count,
        seed,
    ).cpu()
    return FixedHeldoutBatch(batch=batch, candidates=candidates)


@torch.no_grad()
def evaluate_fixed_heldout(
    dense_model: nn.Module,
    embedding: TrainableModuloRowShardedEmbedding,
    heldout: list[FixedHeldoutBatch],
    device: torch.device,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, float | int]:
    dense_model.eval()
    embedding.eval()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for value in heldout:
        batch = value.batch
        item_ids = batch["item_ids"].to(device)
        behaviors = batch["behaviors"].to(device)
        time_deltas = batch["time_deltas"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch.get("labels")
        train_mask = batch.get("train_mask")
        labels = None if labels is None else labels.to(device)
        train_mask = None if train_mask is None else train_mask.to(device)
        item_vectors = embedding(item_ids, lengths)
        hidden = dense_model(
            item_vectors,
            behaviors,
            time_deltas,
            lengths,
        )
        targets, valid = build_next_item_targets(
            item_ids,
            lengths,
            labels,
            train_mask,
        )
        positive_count = int(valid.sum().item())
        candidates = value.candidates.to(device)
        if candidates.shape[0] != positive_count:
            raise ValueError("heldout candidates differ from valid targets")
        candidate_lengths = torch.full(
            (positive_count,),
            candidates.shape[1],
            dtype=torch.int64,
            device=device,
        )
        candidate_vectors = embedding(candidates, candidate_lengths)
        if positive_count == 0:
            continue
        logits = torch.einsum(
            "nh,nch->nc",
            hidden[:, :-1][valid],
            candidate_vectors,
        )
        positive_scores = logits[:, :1]
        ranks = 1 + (logits[:, 1:] >= positive_scores).sum(dim=1)
        hit = ranks <= 10
        ndcg = torch.where(
            hit,
            torch.reciprocal(torch.log2(ranks.double() + 1.0)),
            torch.zeros_like(ranks, dtype=torch.float64),
        )
        reciprocal_rank = torch.reciprocal(ranks.double())
        loss = torch.nn.functional.cross_entropy(
            logits,
            torch.zeros(
                positive_count,
                dtype=torch.int64,
                device=device,
            ),
            reduction="sum",
        )
        totals += torch.stack(
            [
                torch.tensor(
                    float(positive_count),
                    dtype=torch.float64,
                    device=device,
                ),
                hit.double().sum(),
                ndcg.sum(),
                reciprocal_rank.sum(),
                loss.double(),
            ]
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=process_group)
    count = int(totals[0].item())
    if count == 0:
        raise RuntimeError("fixed heldout set has no positive targets")
    return {
        "positive_targets": count,
        "hit_rate_at_10": float((totals[1] / count).item()),
        "ndcg_at_10": float((totals[2] / count).item()),
        "mean_reciprocal_rank": float((totals[3] / count).item()),
        "sampled_cross_entropy": float((totals[4] / count).item()),
    }


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _dense_module(value: nn.Module) -> ExternalEmbeddingHSTU:
    module = value.module if hasattr(value, "module") else value
    if not isinstance(module, ExternalEmbeddingHSTU):
        raise TypeError("dense checkpoint module differs")
    return module


def save_sharded_edge_checkpoint(
    root: str | Path,
    version: int,
    spec: ShardedEdgeModelSpec,
    dense_model: nn.Module,
    embedding: TrainableModuloRowShardedEmbedding,
) -> dict[str, object]:
    if version < 0:
        raise ValueError("checkpoint version must be nonnegative")
    if embedding.num_embeddings != spec.num_embeddings or embedding.hidden_size != spec.hidden_size:
        raise ValueError("checkpoint embedding differs from model spec")
    directory = Path(root) / f"theta_{version}"
    shard_path = directory / f"embedding_rank_{embedding.rank:05d}.pt"
    _atomic_torch_save(
        {
            "schema": SHARDED_EDGE_CHECKPOINT_SCHEMA,
            "version": version,
            "rank": embedding.rank,
            "world_size": embedding.world_size,
            "num_embeddings": embedding.num_embeddings,
            "hidden_size": embedding.hidden_size,
            "global_row_start": embedding.rank,
            "global_row_stride": embedding.world_size,
            "local_rows": embedding.local_rows,
            "local_weight": embedding.local_weight.detach().cpu(),
        },
        shard_path,
    )
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=embedding.process_group)
    dense_path = directory / "dense.pt"
    if embedding.rank == 0:
        dense = _dense_module(dense_model)
        _atomic_torch_save(
            {
                "schema": SHARDED_EDGE_CHECKPOINT_SCHEMA,
                "version": version,
                "config": asdict(dense.cfg),
                "state_dict": dense.state_dict(),
            },
            dense_path,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=embedding.process_group)
    manifest_path = directory / "manifest.json"
    if embedding.rank == 0:
        shards = []
        for rank in range(embedding.world_size):
            path = directory / f"embedding_rank_{rank:05d}.pt"
            shards.append(
                {
                    "rank": rank,
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _artifact_sha256(path),
                    "local_rows": modulo_local_rows(
                        spec.num_embeddings,
                        rank,
                        embedding.world_size,
                    ),
                    "global_row_start": rank,
                    "global_row_stride": embedding.world_size,
                }
            )
        manifest = {
            "schema": SHARDED_EDGE_CHECKPOINT_SCHEMA,
            "version": version,
            "world_size": embedding.world_size,
            "spec": asdict(spec),
            "dense": {
                "path": dense_path.name,
                "bytes": dense_path.stat().st_size,
                "sha256": _artifact_sha256(dense_path),
            },
            "embedding_shards": shards,
        }
        _atomic_json(manifest, manifest_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=embedding.process_group)
    return json.loads(manifest_path.read_text())


def load_sharded_edge_checkpoint(
    root: str | Path,
    version: int,
    spec: ShardedEdgeModelSpec,
    dense_model: nn.Module,
    embedding: TrainableModuloRowShardedEmbedding,
) -> dict[str, object]:
    directory = Path(root) / f"theta_{version}"
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema") != SHARDED_EDGE_CHECKPOINT_SCHEMA
        or manifest.get("version") != version
        or manifest.get("world_size") != embedding.world_size
        or manifest.get("spec") != asdict(spec)
    ):
        raise ValueError("sharded checkpoint manifest differs")
    dense_payload = torch.load(
        directory / manifest["dense"]["path"],
        map_location=embedding.local_weight.device,
        weights_only=True,
    )
    if dense_payload.get("schema") != SHARDED_EDGE_CHECKPOINT_SCHEMA:
        raise ValueError("dense checkpoint schema differs")
    _dense_module(dense_model).load_state_dict(dense_payload["state_dict"])
    shard_record = manifest["embedding_shards"][embedding.rank]
    shard_payload = torch.load(
        directory / shard_record["path"],
        map_location=embedding.local_weight.device,
        weights_only=True,
    )
    if (
        shard_payload.get("schema") != SHARDED_EDGE_CHECKPOINT_SCHEMA
        or shard_payload.get("rank") != embedding.rank
        or shard_payload.get("world_size") != embedding.world_size
        or shard_payload.get("num_embeddings") != spec.num_embeddings
        or shard_payload.get("hidden_size") != spec.hidden_size
        or shard_payload["local_weight"].shape != embedding.local_weight.shape
    ):
        raise ValueError("embedding checkpoint shard differs")
    with torch.no_grad():
        embedding.local_weight.copy_(shard_payload["local_weight"])
    return manifest


def model_memory_estimate(
    spec: ShardedEdgeModelSpec,
    world_size: int,
    kv_records: int,
) -> dict[str, int | float | str]:
    if world_size < 1 or kv_records < 1:
        raise ValueError("invalid memory estimate request")
    dense_parameters = (1 + 5 * spec.num_layers) * spec.hidden_size**2 + (
        spec.num_layers + spec.num_behaviors + 34
    ) * spec.hidden_size
    dense_bytes = dense_parameters * 4
    maximum_local_rows = max(
        modulo_local_rows(spec.num_embeddings, rank, world_size) for rank in range(world_size)
    )
    local_embedding_bytes = maximum_local_rows * spec.hidden_size * 4
    local_dense_training_bytes = dense_bytes * 4
    return {
        "global_embedding_bytes_fp32": spec.embedding_bytes_fp32,
        "maximum_local_embedding_bytes_fp32": local_embedding_bytes,
        "dense_parameters": dense_parameters,
        "dense_bytes_fp32": dense_bytes,
        "per_rank_dense_adamw_parameter_gradient_state_bytes": (local_dense_training_bytes),
        "per_rank_embedding_sgd_persistent_bytes": (local_embedding_bytes),
        "per_rank_embedding_dense_gradient_upper_bound_bytes": (local_embedding_bytes),
        "per_rank_parameter_optimizer_and_dense_embedding_gradient_bytes": (
            2 * local_embedding_bytes + local_dense_training_bytes
        ),
        "embedding_gradient_layout": (
            "sparse COO over rows touched in the step; dense-size value is "
            "a conservative bound, not a standing allocation"
        ),
        "excluded_from_training_state_estimate": (
            "activations, temporary collectives, CUDA context, allocator "
            "fragmentation, and DDP buckets"
        ),
        "kv_bytes_fp16_per_record": spec.kv_bytes_fp16_per_record,
        "old_plus_target_kv_bytes_fp16": (2 * spec.kv_bytes_fp16_per_record * kv_records),
        "old_plus_target_kv_gib": (2 * spec.kv_bytes_fp16_per_record * kv_records / 2**30),
    }
