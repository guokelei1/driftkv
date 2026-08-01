from __future__ import annotations

import hashlib
import json
import os
import sys
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from ..models import HSTUConfig
from .sharded_edge import (
    ExternalEmbeddingHSTU,
    modulo_local_rows,
)

XP_PROJECTED_CHECKPOINT_SCHEMA = (
    "evokv_xp_projected_sharded_checkpoint_development_v0"
)


def _validate_group(
    rank: int,
    world_size: int,
    process_group: dist.ProcessGroup | None,
) -> None:
    if world_size == 1:
        if dist.is_initialized() and (
            dist.get_world_size(group=process_group) != 1
            or dist.get_rank(group=process_group) != rank
        ):
            raise ValueError("XP projected process group differs")
        return
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size(group=process_group) != world_size
        or dist.get_rank(group=process_group) != rank
    ):
        raise RuntimeError("XP projected process group is not initialized")


class _ProjectedModuloLookup(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        local_weight: torch.Tensor,
        projection_weight: torch.Tensor,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
        num_embeddings: int,
        rank: int,
        world_size: int,
        process_group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        if (
            local_weight.ndim != 2
            or projection_weight.ndim != 2
            or projection_weight.shape[1] != local_weight.shape[1]
            or item_ids.ndim != 2
            or lengths.shape != (item_ids.shape[0],)
            or item_ids.device != local_weight.device
            or lengths.device != item_ids.device
            or projection_weight.device != local_weight.device
            or local_weight.shape[0]
            != modulo_local_rows(num_embeddings, rank, world_size)
        ):
            raise ValueError("XP projected lookup shape differs")
        _validate_group(rank, world_size, process_group)
        width = item_ids.shape[1]
        lengths_long = lengths.long()
        if bool(torch.any(lengths_long < 0)) or bool(
            torch.any(lengths_long > width)
        ):
            raise ValueError("XP projected lookup lengths differ")
        valid = (
            torch.arange(width, device=item_ids.device).unsqueeze(0)
            < lengths_long.unsqueeze(1)
        )
        positions = torch.nonzero(
            valid.reshape(-1),
            as_tuple=False,
        ).flatten()
        requested_ids = item_ids.reshape(-1).index_select(
            0,
            positions,
        ).long()
        if requested_ids.numel() and (
            bool(torch.any(requested_ids < 0))
            or bool(torch.any(requested_ids >= num_embeddings))
        ):
            raise ValueError("XP projected lookup item id exceeds table")
        owners = requested_ids.remainder(world_size)
        local_mask = owners == rank
        remote_mask = ~local_mask
        local_positions = positions[local_mask]
        local_rows = requested_ids[local_mask].div(
            world_size,
            rounding_mode="floor",
        )
        output = torch.zeros(
            (item_ids.numel(), projection_weight.shape[0]),
            dtype=local_weight.dtype,
            device=local_weight.device,
        )
        if local_rows.numel():
            output.index_copy_(
                0,
                local_positions,
                torch.nn.functional.linear(
                    local_weight.index_select(0, local_rows),
                    projection_weight,
                ),
            )
        if world_size == 1:
            ordered_positions = positions[:0]
            received_local_rows = local_rows[:0]
            send_splits = (0,)
            receive_splits = (0,)
        else:
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
            send_splits = tuple(
                int(value) for value in send_counts.tolist()
            )
            receive_splits = tuple(
                int(value) for value in receive_counts.tolist()
            )
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
                or bool(
                    torch.any(
                        received_local_rows >= local_weight.shape[0]
                    )
                )
            ):
                raise RuntimeError("XP projected owner row exceeds shard")
            response = torch.nn.functional.linear(
                local_weight.index_select(0, received_local_rows),
                projection_weight,
            )
            received = torch.empty(
                (send_local_rows.numel(), projection_weight.shape[0]),
                dtype=local_weight.dtype,
                device=local_weight.device,
            )
            dist.all_to_all_single(
                received,
                response.contiguous(),
                output_split_sizes=send_splits,
                input_split_sizes=receive_splits,
                group=process_group,
            )
            if ordered_positions.numel():
                output.index_copy_(0, ordered_positions, received)
        ctx.local_weight_shape = tuple(local_weight.shape)
        ctx.world_size = world_size
        ctx.process_group = process_group
        ctx.send_splits = send_splits
        ctx.receive_splits = receive_splits
        ctx.save_for_backward(
            local_weight,
            projection_weight,
            local_rows,
            local_positions,
            ordered_positions,
            received_local_rows,
        )
        return output.reshape(
            *item_ids.shape,
            projection_weight.shape[0],
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            local_weight,
            projection_weight,
            local_rows,
            local_positions,
            ordered_positions,
            received_local_rows,
        ) = ctx.saved_tensors
        grad_flat = grad_output.reshape(
            -1,
            projection_weight.shape[0],
        ).contiguous()
        local_projected_grad = grad_flat.index_select(
            0,
            local_positions,
        )
        if ctx.world_size == 1:
            remote_projected_grad = local_projected_grad[:0]
        else:
            requested_remote_grad = grad_flat.index_select(
                0,
                ordered_positions,
            ).contiguous()
            remote_projected_grad = torch.empty(
                (
                    received_local_rows.numel(),
                    projection_weight.shape[0],
                ),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            dist.all_to_all_single(
                remote_projected_grad,
                requested_remote_grad,
                output_split_sizes=ctx.receive_splits,
                input_split_sizes=ctx.send_splits,
                group=ctx.process_group,
            )
        rows = torch.cat([local_rows, received_local_rows])
        projected_grad = torch.cat(
            [local_projected_grad, remote_projected_grad]
        )
        if rows.numel():
            inputs = local_weight.index_select(0, rows)
            embedding_values = projected_grad.matmul(
                projection_weight,
            )
            projection_gradient = projected_grad.transpose(
                0,
                1,
            ).matmul(inputs)
            indices = rows.unsqueeze(0)
        else:
            embedding_values = torch.empty(
                (0, ctx.local_weight_shape[1]),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            projection_gradient = torch.zeros_like(
                projection_weight,
            )
            indices = torch.empty(
                (1, 0),
                dtype=torch.int64,
                device=grad_output.device,
            )
        if ctx.world_size > 1:
            dist.all_reduce(
                projection_gradient,
                op=dist.ReduceOp.SUM,
                group=ctx.process_group,
            )
        with torch.sparse.check_sparse_tensor_invariants(enable=True):
            embedding_gradient = torch.sparse_coo_tensor(
                indices,
                embedding_values,
                size=ctx.local_weight_shape,
                dtype=grad_output.dtype,
                device=grad_output.device,
            ).coalesce()
        return (
            embedding_gradient,
            projection_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@dataclass(frozen=True)
class XPProjectedModelSpec:
    num_embeddings: int
    embedding_width: int
    hidden_size: int
    num_prediction_items: int
    num_behaviors: int
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int

    def __post_init__(self) -> None:
        if (
            self.num_embeddings < 2
            or self.embedding_width < 1
            or self.hidden_size < 1
            or not 1 <= self.num_prediction_items < self.num_embeddings
            or self.num_behaviors < 1
            or self.num_layers < 1
            or self.num_heads < 1
            or self.head_dim < 1
            or self.num_heads * self.head_dim != self.hidden_size
            or self.max_seq_len < 2
        ):
            raise ValueError("XP projected model specification is invalid")

    @property
    def global_embedding_bytes_fp32(self) -> int:
        return self.num_embeddings * self.embedding_width * 4

    @property
    def projection_bytes_fp32(self) -> int:
        return self.embedding_width * self.hidden_size * 4

    def hstu_config(self) -> HSTUConfig:
        return HSTUConfig(
            num_items=self.num_embeddings - 1,
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


class TrainableProjectedModuloEmbedding(nn.Module):
    def __init__(
        self,
        *,
        local_weight: torch.Tensor,
        projection_weight: torch.Tensor,
        num_embeddings: int,
        rank: int,
        world_size: int,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        if (
            local_weight.ndim != 2
            or projection_weight.ndim != 2
            or not local_weight.is_floating_point()
            or projection_weight.dtype != local_weight.dtype
            or projection_weight.device != local_weight.device
            or projection_weight.shape[1] != local_weight.shape[1]
            or local_weight.shape[0]
            != modulo_local_rows(num_embeddings, rank, world_size)
        ):
            raise ValueError("trainable XP projected layout differs")
        _validate_group(rank, world_size, process_group)
        self.num_embeddings = int(num_embeddings)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.process_group = process_group
        self.local_weight = nn.Parameter(local_weight.contiguous())
        self.projection_weight = nn.Parameter(
            projection_weight.contiguous()
        )

    @classmethod
    def initialize(
        cls,
        *,
        num_embeddings: int,
        embedding_width: int,
        hidden_size: int,
        rank: int,
        world_size: int,
        device: torch.device | str,
        embedding_seed: int,
        projection_seed: int,
        std: float = 0.02,
        process_group: dist.ProcessGroup | None = None,
    ) -> TrainableProjectedModuloEmbedding:
        if embedding_width < 1 or hidden_size < 1 or std <= 0:
            raise ValueError("XP projected initialization differs")
        target = torch.device(device)
        embedding_generator = torch.Generator(device=target)
        embedding_generator.manual_seed(
            embedding_seed + rank * 1_000_003
        )
        local_weight = torch.empty(
            (
                modulo_local_rows(
                    num_embeddings,
                    rank,
                    world_size,
                ),
                embedding_width,
            ),
            dtype=torch.float32,
            device=target,
        )
        local_weight.normal_(
            mean=0.0,
            std=std,
            generator=embedding_generator,
        )
        if rank == 0 and local_weight.shape[0]:
            local_weight[0].zero_()
        projection_generator = torch.Generator(device=target)
        projection_generator.manual_seed(projection_seed)
        projection_weight = torch.empty(
            (hidden_size, embedding_width),
            dtype=torch.float32,
            device=target,
        )
        projection_weight.normal_(
            mean=0.0,
            std=std,
            generator=projection_generator,
        )
        return cls(
            local_weight=local_weight,
            projection_weight=projection_weight,
            num_embeddings=num_embeddings,
            rank=rank,
            world_size=world_size,
            process_group=process_group,
        )

    @property
    def embedding_width(self) -> int:
        return self.local_weight.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.projection_weight.shape[0]

    @property
    def local_rows(self) -> int:
        return self.local_weight.shape[0]

    def forward(
        self,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return _ProjectedModuloLookup.apply(
            self.local_weight,
            self.projection_weight,
            item_ids,
            lengths,
            self.num_embeddings,
            self.rank,
            self.world_size,
            self.process_group,
        )


class OptimizerActiveRowTracker:
    def __init__(
        self,
        *,
        num_embeddings: int,
        rank: int,
        world_size: int,
    ) -> None:
        local_rows = modulo_local_rows(
            num_embeddings,
            rank,
            world_size,
        )
        self.num_embeddings = int(num_embeddings)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_bitmap = torch.zeros(
            local_rows,
            dtype=torch.uint8,
            device="cpu",
        )
        self.local_update_counts = torch.zeros(
            local_rows,
            dtype=torch.int32,
            device="cpu",
        )

    @property
    def local_active_count(self) -> int:
        return int(torch.count_nonzero(self.local_bitmap).item())

    def mark_local_rows(self, local_rows: torch.Tensor) -> None:
        rows = local_rows.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        if rows.ndim != 1 or (
            rows.numel()
            and (
                bool(torch.any(rows < 0))
                or bool(torch.any(rows >= len(self.local_bitmap)))
            )
        ):
            raise ValueError("optimizer active local rows differ")
        if rows.numel():
            global_rows = rows * self.world_size + self.rank
            rows = rows[
                (global_rows > 0)
                & (global_rows < self.num_embeddings)
            ]
            rows = torch.unique(rows)
            self.local_bitmap[rows] = 1
            self.local_update_counts[rows] += 1

    def load_activity(
        self,
        bitmap_value: torch.Tensor,
        count_value: torch.Tensor,
    ) -> None:
        bitmap = bitmap_value.detach().to(
            device="cpu",
            dtype=torch.uint8,
        )
        counts = count_value.detach().to(
            device="cpu",
            dtype=torch.int32,
        )
        if (
            bitmap.shape != self.local_bitmap.shape
            or counts.shape != self.local_update_counts.shape
            or bool(torch.any((bitmap != 0) & (bitmap != 1)))
            or bool(torch.any(counts < 0))
            or not torch.equal(bitmap, (counts > 0).to(torch.uint8))
        ):
            raise ValueError("optimizer active ledger differs")
        if self.rank == 0 and bitmap.numel() and (
            bitmap[0].item() != 0 or counts[0].item() != 0
        ):
            raise ValueError("padding row cannot be optimizer active")
        self.local_bitmap.copy_(bitmap)
        self.local_update_counts.copy_(counts)

    def local_global_row_ids(self) -> tuple[int, ...]:
        local = torch.nonzero(
            self.local_bitmap,
            as_tuple=False,
        ).flatten()
        return tuple(
            int(value)
            for value in (
                local * self.world_size + self.rank
            ).tolist()
            if 0 < int(value) < self.num_embeddings
        )


def sparse_embedding_sgd(
    embedding: TrainableProjectedModuloEmbedding,
    learning_rate: float,
) -> torch.optim.SGD:
    if learning_rate <= 0:
        raise ValueError("embedding learning rate must be positive")
    return torch.optim.SGD(
        [embedding.local_weight],
        lr=learning_rate,
        momentum=0.0,
        weight_decay=0.0,
        foreach=False,
    )


def tracked_sparse_optimizer_step(
    embedding: TrainableProjectedModuloEmbedding,
    optimizer: torch.optim.Optimizer,
    tracker: OptimizerActiveRowTracker,
) -> tuple[int, ...]:
    parameters = [
        value
        for group in optimizer.param_groups
        for value in group["params"]
    ]
    if (
        len(parameters) != 1
        or parameters[0] is not embedding.local_weight
        or tracker.num_embeddings != embedding.num_embeddings
        or tracker.rank != embedding.rank
        or tracker.world_size != embedding.world_size
    ):
        raise ValueError("tracked sparse optimizer binding differs")
    gradient = embedding.local_weight.grad
    if gradient is None or not gradient.is_sparse:
        raise RuntimeError("XP embedding gradient must be sparse")
    coalesced = gradient.coalesce()
    values = coalesced.values()
    if not bool(torch.all(torch.isfinite(values))):
        raise RuntimeError("XP embedding gradient is not finite")
    nonzero = (
        torch.any(values != 0, dim=1)
        if values.numel()
        else torch.zeros(
            0,
            dtype=torch.bool,
            device=values.device,
        )
    )
    updated_local_rows = coalesced.indices()[0][nonzero]
    optimizer.step()
    tracker.mark_local_rows(updated_local_rows)
    if embedding.rank == 0 and embedding.local_rows:
        with torch.no_grad():
            embedding.local_weight[0].zero_()
    return tracker.local_global_row_ids()


def active_row_ids_sha256(row_ids: Sequence[int]) -> str:
    resolved = tuple(int(value) for value in row_ids)
    if (
        any(value < 1 for value in resolved)
        or resolved != tuple(sorted(set(resolved)))
    ):
        raise ValueError("active row ids are invalid")
    digest = hashlib.sha256()
    for start in range(0, len(resolved), 131_072):
        payload = array("Q", resolved[start : start + 131_072])
        if sys.byteorder != "little":
            payload.byteswap()
        digest.update(payload.tobytes())
    return digest.hexdigest()


def active_row_update_counts_sha256(
    row_ids: Sequence[int],
    update_counts: Sequence[int],
) -> str:
    resolved_ids = tuple(int(value) for value in row_ids)
    resolved_counts = tuple(int(value) for value in update_counts)
    if (
        len(resolved_ids) != len(resolved_counts)
        or any(value < 1 for value in resolved_ids)
        or any(value < 1 for value in resolved_counts)
        or resolved_ids != tuple(sorted(set(resolved_ids)))
    ):
        raise ValueError("active row update counts are invalid")
    digest = hashlib.sha256()
    for start in range(0, len(resolved_ids), 131_072):
        ids = array("Q", resolved_ids[start : start + 131_072])
        counts = array(
            "I",
            resolved_counts[start : start + 131_072],
        )
        if sys.byteorder != "little":
            ids.byteswap()
            counts.byteswap()
        digest.update(ids.tobytes())
        digest.update(counts.tobytes())
    return digest.hexdigest()


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(
            lambda: source.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _descriptor(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _artifact_sha256(path),
    }


def _unwrap_dense(value: nn.Module) -> ExternalEmbeddingHSTU:
    module = value.module if hasattr(value, "module") else value
    if not isinstance(module, ExternalEmbeddingHSTU):
        raise TypeError("XP projected dense module differs")
    return module


def _validate_model_binding(
    spec: XPProjectedModelSpec,
    dense_model: nn.Module,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
) -> None:
    dense = _unwrap_dense(dense_model)
    if (
        asdict(dense.cfg) != asdict(spec.hstu_config())
        or embedding.num_embeddings != spec.num_embeddings
        or embedding.embedding_width != spec.embedding_width
        or embedding.hidden_size != spec.hidden_size
        or tracker.num_embeddings != spec.num_embeddings
        or tracker.rank != embedding.rank
        or tracker.world_size != embedding.world_size
    ):
        raise ValueError("XP projected model binding differs")


def _active_ledger(
    directory: Path,
    spec: XPProjectedModelSpec,
    world_size: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    descriptors = []
    global_ids = []
    global_counts = []
    histogram: dict[int, int] = {}
    for rank in range(world_size):
        path = directory / f"active_bitmap_rank_{rank:05d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        bitmap = payload.get("local_bitmap")
        update_counts = payload.get("local_update_counts")
        if (
            payload.get("schema") != XP_PROJECTED_CHECKPOINT_SCHEMA
            or payload.get("rank") != rank
            or payload.get("world_size") != world_size
            or payload.get("num_embeddings") != spec.num_embeddings
            or not isinstance(bitmap, torch.Tensor)
            or not isinstance(update_counts, torch.Tensor)
            or bitmap.shape
            != (
                modulo_local_rows(
                    spec.num_embeddings,
                    rank,
                    world_size,
                ),
            )
            or bitmap.dtype != torch.uint8
            or update_counts.shape != bitmap.shape
            or update_counts.dtype != torch.int32
            or bool(torch.any((bitmap != 0) & (bitmap != 1)))
            or bool(torch.any(update_counts < 0))
            or not torch.equal(
                bitmap,
                (update_counts > 0).to(torch.uint8),
            )
        ):
            raise ValueError("XP active bitmap artifact differs")
        local = torch.nonzero(bitmap, as_tuple=False).flatten()
        ids = local * world_size + rank
        valid = (ids > 0) & (ids < spec.num_embeddings)
        ids = ids[valid]
        counts = update_counts.index_select(0, local)[valid]
        global_ids.extend(int(value) for value in ids.tolist())
        global_counts.extend(int(value) for value in counts.tolist())
        unique_counts, unique_frequency = torch.unique(
            counts,
            return_counts=True,
        )
        for count, frequency in zip(
            unique_counts.tolist(),
            unique_frequency.tolist(),
            strict=True,
        ):
            histogram[int(count)] = (
                histogram.get(int(count), 0) + int(frequency)
            )
        descriptors.append(
            {
                "rank": rank,
                **_descriptor(path),
                "local_rows": len(bitmap),
                "active_rows": len(ids),
                "global_row_start": rank,
                "global_row_stride": world_size,
            }
        )
    ordered = sorted(
        zip(global_ids, global_counts, strict=True),
        key=lambda value: value[0],
    )
    global_ids = [value[0] for value in ordered]
    global_counts = [value[1] for value in ordered]
    if len(global_ids) != len(set(global_ids)):
        raise RuntimeError("XP active row shards overlap")
    return descriptors, {
        "definition": (
            "non-padding semantic rows with a finite nonzero sparse "
            "embedding gradient recorded only after a successful optimizer step"
        ),
        "bitmap_dtype": "uint8_binary",
        "global_active_rows": len(global_ids),
        "global_active_fraction": (
            len(global_ids) / (spec.num_embeddings - 1)
        ),
        "global_row_ids_sha256": active_row_ids_sha256(global_ids),
        "global_row_update_counts_sha256": (
            active_row_update_counts_sha256(
                global_ids,
                global_counts,
            )
        ),
        "row_id_hash_encoding": "sorted uint64 little-endian",
        "row_update_count_hash_encoding": (
            "chunks of sorted uint64 row ids followed by aligned uint32 "
            "counts, little-endian, chunk size 131072"
        ),
        "global_optimizer_update_events": sum(global_counts),
        "active_update_count_minimum": (
            min(global_counts) if global_counts else 0
        ),
        "active_update_count_maximum": (
            max(global_counts) if global_counts else 0
        ),
        "active_update_count_mean": (
            sum(global_counts) / len(global_counts)
            if global_counts
            else 0.0
        ),
        "active_update_count_histogram": {
            str(count): histogram[count]
            for count in sorted(histogram)
        },
        "padding_row_excluded": True,
        "bitmap_shards": descriptors,
    }


def save_xp_projected_checkpoint(
    root: str | Path,
    version: int,
    spec: XPProjectedModelSpec,
    dense_model: nn.Module,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if version < 0:
        raise ValueError("XP checkpoint version must be nonnegative")
    _validate_model_binding(spec, dense_model, embedding, tracker)
    _validate_group(
        embedding.rank,
        embedding.world_size,
        embedding.process_group,
    )
    if (
        embedding.rank == 0
        and embedding.local_rows
        and bool(torch.any(embedding.local_weight[0] != 0))
    ):
        raise ValueError("XP padding embedding row must remain zero")
    projection_sha256 = _tensor_sha256(
        embedding.projection_weight,
    )
    replica_hashes: list[object] = [None] * embedding.world_size
    if embedding.world_size > 1:
        dist.all_gather_object(
            replica_hashes,
            projection_sha256,
            group=embedding.process_group,
        )
    else:
        replica_hashes[0] = projection_sha256
    if len(set(str(value) for value in replica_hashes)) != 1:
        raise ValueError("XP projection replicas differ")
    directory = Path(root) / f"theta_{version}"
    shard_path = (
        directory / f"embedding_rank_{embedding.rank:05d}.pt"
    )
    bitmap_path = (
        directory / f"active_bitmap_rank_{embedding.rank:05d}.pt"
    )
    _atomic_torch_save(
        {
            "schema": XP_PROJECTED_CHECKPOINT_SCHEMA,
            "version": version,
            "rank": embedding.rank,
            "world_size": embedding.world_size,
            "num_embeddings": spec.num_embeddings,
            "embedding_width": spec.embedding_width,
            "global_row_start": embedding.rank,
            "global_row_stride": embedding.world_size,
            "local_rows": embedding.local_rows,
            "local_weight": embedding.local_weight.detach().cpu(),
        },
        shard_path,
    )
    _atomic_torch_save(
        {
            "schema": XP_PROJECTED_CHECKPOINT_SCHEMA,
            "version": version,
            "rank": embedding.rank,
            "world_size": embedding.world_size,
            "num_embeddings": spec.num_embeddings,
            "local_bitmap": tracker.local_bitmap.clone(),
            "local_update_counts": (
                tracker.local_update_counts.clone()
            ),
        },
        bitmap_path,
    )
    if embedding.world_size > 1:
        dist.barrier(group=embedding.process_group)
    manifest_path = directory / "manifest.json"
    if embedding.rank == 0:
        dense = _unwrap_dense(dense_model)
        dense_path = directory / "dense.pt"
        projection_path = directory / "projection.pt"
        _atomic_torch_save(
            {
                "schema": XP_PROJECTED_CHECKPOINT_SCHEMA,
                "version": version,
                "config": asdict(dense.cfg),
                "state_dict": dense.state_dict(),
            },
            dense_path,
        )
        _atomic_torch_save(
            {
                "schema": XP_PROJECTED_CHECKPOINT_SCHEMA,
                "version": version,
                "embedding_width": spec.embedding_width,
                "hidden_size": spec.hidden_size,
                "bias": False,
                "projection_weight": (
                    embedding.projection_weight.detach().cpu()
                ),
            },
            projection_path,
        )
        embedding_shards = []
        for rank in range(embedding.world_size):
            path = directory / f"embedding_rank_{rank:05d}.pt"
            embedding_shards.append(
                {
                    "rank": rank,
                    **_descriptor(path),
                    "local_rows": modulo_local_rows(
                        spec.num_embeddings,
                        rank,
                        embedding.world_size,
                    ),
                    "global_row_start": rank,
                    "global_row_stride": embedding.world_size,
                }
            )
        bitmap_shards, active = _active_ledger(
            directory,
            spec,
            embedding.world_size,
        )
        active["bitmap_shards"] = bitmap_shards
        manifest = {
            "schema": XP_PROJECTED_CHECKPOINT_SCHEMA,
            "protocol": XP_PROJECTED_CHECKPOINT_SCHEMA,
            "version": version,
            "world_size": embedding.world_size,
            "spec": asdict(spec),
            "scientific_result": False,
            "formal_design2": False,
            "formal_design3": False,
            "artifact_role": (
                "successor_xp_projected_checkpoint_development"
            ),
            "embedding_layout": "modulo_row_sharded_fp32",
            "projection_layout": (
                "replicated_bias_free_owner_side_fp32"
            ),
            "dense": _descriptor(dense_path),
            "projection": {
                **_descriptor(projection_path),
                "replica_tensor_sha256": projection_sha256,
            },
            "embedding_shards": embedding_shards,
            "optimizer_active_rows": active,
        }
        if provenance is not None:
            manifest["provenance"] = json.loads(
                json.dumps(provenance, sort_keys=True)
            )
        _atomic_json(manifest, manifest_path)
    if embedding.world_size > 1:
        dist.barrier(group=embedding.process_group)
    return json.loads(manifest_path.read_text())


def _resolve_artifact(
    directory: Path,
    value: Mapping[str, object],
) -> Path:
    path = directory / str(value.get("path", ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(value.get("bytes", -1))
        or _artifact_sha256(path) != str(value.get("sha256", ""))
    ):
        raise ValueError("XP checkpoint artifact binding differs")
    return path


def load_xp_projected_checkpoint(
    root: str | Path,
    version: int,
    spec: XPProjectedModelSpec,
    dense_model: nn.Module,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
) -> dict[str, object]:
    _validate_model_binding(spec, dense_model, embedding, tracker)
    directory = Path(root) / f"theta_{version}"
    manifest = json.loads(
        (directory / "manifest.json").read_text()
    )
    if (
        manifest.get("schema") != XP_PROJECTED_CHECKPOINT_SCHEMA
        or manifest.get("protocol") != XP_PROJECTED_CHECKPOINT_SCHEMA
        or manifest.get("version") != version
        or manifest.get("world_size") != embedding.world_size
        or manifest.get("spec") != asdict(spec)
        or manifest.get("scientific_result") is not False
        or manifest.get("formal_design2") is not False
        or manifest.get("formal_design3") is not False
    ):
        raise ValueError("XP checkpoint manifest differs")
    dense_record = manifest.get("dense")
    projection_record = manifest.get("projection")
    embedding_records = manifest.get("embedding_shards")
    active = manifest.get("optimizer_active_rows")
    if (
        not isinstance(dense_record, Mapping)
        or not isinstance(projection_record, Mapping)
        or not isinstance(embedding_records, list)
        or len(embedding_records) != embedding.world_size
        or not isinstance(active, Mapping)
    ):
        raise ValueError("XP checkpoint manifest layout differs")
    dense_path = _resolve_artifact(directory, dense_record)
    projection_path = _resolve_artifact(
        directory,
        projection_record,
    )
    shard_record = embedding_records[embedding.rank]
    if (
        not isinstance(shard_record, Mapping)
        or shard_record.get("rank") != embedding.rank
    ):
        raise ValueError("XP checkpoint shard descriptor differs")
    shard_path = _resolve_artifact(directory, shard_record)
    bitmap_records = active.get("bitmap_shards")
    if (
        not isinstance(bitmap_records, list)
        or len(bitmap_records) != embedding.world_size
        or not isinstance(
            bitmap_records[embedding.rank],
            Mapping,
        )
    ):
        raise ValueError("XP active bitmap descriptors differ")
    bitmap_path = _resolve_artifact(
        directory,
        bitmap_records[embedding.rank],
    )
    observed_bitmaps, observed_active = _active_ledger(
        directory,
        spec,
        embedding.world_size,
    )
    expected_active = dict(active)
    expected_active["bitmap_shards"] = observed_bitmaps
    if observed_active != expected_active:
        raise ValueError("XP global active row ledger differs")
    dense_payload = torch.load(
        dense_path,
        map_location="cpu",
        weights_only=True,
    )
    projection_payload = torch.load(
        projection_path,
        map_location="cpu",
        weights_only=True,
    )
    shard_payload = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=True,
    )
    bitmap_payload = torch.load(
        bitmap_path,
        map_location="cpu",
        weights_only=True,
    )
    if (
        dense_payload.get("schema")
        != XP_PROJECTED_CHECKPOINT_SCHEMA
        or dense_payload.get("version") != version
        or dense_payload.get("config")
        != asdict(spec.hstu_config())
        or projection_payload.get("schema")
        != XP_PROJECTED_CHECKPOINT_SCHEMA
        or projection_payload.get("version") != version
        or projection_payload.get("bias") is not False
        or projection_payload.get("projection_weight").shape
        != embedding.projection_weight.shape
        or shard_payload.get("schema")
        != XP_PROJECTED_CHECKPOINT_SCHEMA
        or shard_payload.get("version") != version
        or shard_payload.get("rank") != embedding.rank
        or shard_payload.get("world_size") != embedding.world_size
        or shard_payload.get("num_embeddings")
        != spec.num_embeddings
        or shard_payload.get("embedding_width")
        != spec.embedding_width
        or shard_payload.get("local_weight").shape
        != embedding.local_weight.shape
        or bitmap_payload.get("schema")
        != XP_PROJECTED_CHECKPOINT_SCHEMA
        or bitmap_payload.get("version") != version
        or not isinstance(
            bitmap_payload.get("local_update_counts"),
            torch.Tensor,
        )
    ):
        raise ValueError("XP checkpoint payload differs")
    _unwrap_dense(dense_model).load_state_dict(
        dense_payload["state_dict"]
    )
    with torch.no_grad():
        embedding.local_weight.copy_(
            shard_payload["local_weight"]
        )
        embedding.projection_weight.copy_(
            projection_payload["projection_weight"]
        )
    tracker.load_activity(
        bitmap_payload["local_bitmap"],
        bitmap_payload["local_update_counts"],
    )
    return manifest
