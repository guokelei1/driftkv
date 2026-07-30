from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..models import HSTU, HSTUConfig
from ..models.embeddings import ItemEmbedding
from ..streaming.sharded_edge import (
    SHARDED_EDGE_CHECKPOINT_SCHEMA,
    ExternalEmbeddingHSTU,
)
from .design2_embedding import (
    D2ShardedHSTU,
    ModuloRowShardedEmbedding,
    build_modulo_sharded_hstu_from_cpu,
)
from .design2_plan import file_sha256


@dataclass(frozen=True)
class D3VersionCheckpoint:
    version: int
    layout: str
    identity_path: Path
    identity_sha256: str
    full_path: Path | None = None
    manifest: Mapping[str, object] | None = None
    dense_path: Path | None = None
    shard_paths: tuple[Path, ...] = ()

    def descriptor(self) -> dict[str, object]:
        value: dict[str, object] = {
            "version": f"theta{self.version}",
            "layout": self.layout,
            "path": str(self.identity_path),
            "sha256": self.identity_sha256,
            "bytes": self.identity_path.stat().st_size,
        }
        if self.dense_path is not None:
            value["dense_path"] = str(self.dense_path)
            value["embedding_shard_paths"] = [
                str(path) for path in self.shard_paths
            ]
        return value


def training_model_config(
    training: Mapping[str, object],
) -> HSTUConfig:
    model = training.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("M1 training model boundary is missing")
    dense = model.get("dense_config", model)
    if not isinstance(dense, Mapping):
        raise ValueError("M1 dense model configuration is missing")
    cfg = HSTUConfig(**dict(dense))
    spec = model.get("spec")
    if isinstance(spec, Mapping):
        expected = {
            "num_embeddings": cfg.num_items + 1,
            "num_prediction_items": cfg.num_prediction_items,
            "num_behaviors": cfg.num_behaviors,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "head_dim": cfg.head_dim,
            "max_seq_len": cfg.max_seq_len,
        }
        if dict(spec) != expected:
            raise ValueError("M1 sharded model spec differs from dense config")
    return cfg


def _load_json(path: Path) -> dict[str, object]:
    with path.open() as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _validate_artifact(
    directory: Path,
    value: object,
    verify_hash: bool,
) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint artifact descriptor is invalid")
    path_value = value.get("path")
    if not isinstance(path_value, str):
        raise ValueError("checkpoint artifact path is invalid")
    path = directory / path_value
    if (
        not path.is_file()
        or int(value.get("bytes", -1)) != path.stat().st_size
        or (
            verify_hash
            and str(value.get("sha256", "")) != file_sha256(path)
        )
    ):
        raise ValueError(f"checkpoint artifact binding differs: {path}")
    return path


def resolve_version_checkpoint(
    training: Mapping[str, object],
    checkpoint_dir: str | Path,
    version: int,
    verify_payload_hashes: bool = True,
    verify_shard_ranks: Sequence[int] | None = None,
) -> D3VersionCheckpoint:
    if version not in {0, 1}:
        raise ValueError("M1 checkpoint version is unsupported")
    root = Path(checkpoint_dir)
    descriptors = training.get("checkpoints")
    if isinstance(descriptors, list):
        expected = next(
            (
                value
                for value in descriptors
                if isinstance(value, Mapping)
                and value.get("version") == f"theta{version}"
            ),
            None,
        )
        path = root / f"theta_{version}.pt"
        digest = file_sha256(path) if path.is_file() else ""
        if (
            not isinstance(expected, Mapping)
            or not path.is_file()
            or str(expected.get("sha256", "")) != digest
        ):
            raise ValueError(
                f"M1 theta{version} checkpoint binding differs"
            )
        return D3VersionCheckpoint(
            version=version,
            layout="full_table_single_file",
            identity_path=path,
            identity_sha256=digest,
            full_path=path,
        )
    if not isinstance(descriptors, Mapping):
        raise ValueError("M1 training checkpoint descriptors are missing")
    expected_manifest = descriptors.get(f"theta{version}")
    directory = root / f"theta_{version}"
    manifest_path = directory / "manifest.json"
    if not isinstance(expected_manifest, Mapping) or not manifest_path.is_file():
        raise ValueError(
            f"M1 theta{version} sharded checkpoint is missing"
        )
    manifest = _load_json(manifest_path)
    cfg = training_model_config(training)
    model = training["model"]
    spec = model.get("spec") if isinstance(model, Mapping) else None
    if (
        not isinstance(spec, Mapping)
        or manifest != dict(expected_manifest)
        or manifest.get("schema") != SHARDED_EDGE_CHECKPOINT_SCHEMA
        or int(manifest.get("version", -1)) != version
        or manifest.get("spec") != spec
    ):
        raise ValueError(
            f"M1 theta{version} sharded manifest binding differs"
        )
    world_size = int(manifest.get("world_size", 0))
    raw_shards = manifest.get("embedding_shards")
    if (
        world_size < 1
        or not isinstance(raw_shards, list)
        or len(raw_shards) != world_size
    ):
        raise ValueError("M1 sharded checkpoint layout is invalid")
    dense_path = _validate_artifact(
        directory,
        manifest.get("dense"),
        verify_payload_hashes,
    )
    selected_ranks = (
        set(range(world_size))
        if verify_shard_ranks is None
        else set(verify_shard_ranks)
    )
    if any(not 0 <= rank < world_size for rank in selected_ranks):
        raise ValueError("M1 checkpoint shard verification rank differs")
    shard_paths = []
    for rank, value in enumerate(raw_shards):
        if (
            not isinstance(value, Mapping)
            or int(value.get("rank", -1)) != rank
            or int(value.get("global_row_start", -1)) != rank
            or int(value.get("global_row_stride", -1)) != world_size
        ):
            raise ValueError("M1 embedding shard layout differs")
        shard_paths.append(
            _validate_artifact(
                directory,
                value,
                verify_payload_hashes and rank in selected_ranks,
            )
        )
    if cfg.num_items + 1 != int(spec.get("num_embeddings", 0)):
        raise ValueError("M1 checkpoint vocabulary differs")
    return D3VersionCheckpoint(
        version=version,
        layout="modulo_row_sharded_manifest",
        identity_path=manifest_path,
        identity_sha256=file_sha256(manifest_path),
        manifest=manifest,
        dense_path=dense_path,
        shard_paths=tuple(shard_paths),
    )


def _load_dense_core(
    cfg: HSTUConfig,
    checkpoint: D3VersionCheckpoint,
) -> HSTU:
    if checkpoint.dense_path is None:
        raise ValueError("sharded checkpoint lacks a dense payload")
    payload = torch.load(
        checkpoint.dense_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if (
        payload.get("schema") != SHARDED_EDGE_CHECKPOINT_SCHEMA
        or int(payload.get("version", -1)) != checkpoint.version
        or payload.get("config") != asdict(cfg)
        or not isinstance(payload.get("state_dict"), Mapping)
    ):
        raise ValueError("M1 dense checkpoint payload differs")
    dense = ExternalEmbeddingHSTU(cfg)
    dense.load_state_dict(payload["state_dict"])
    dense.eval()
    return dense.core


def _load_embedding_shard(
    cfg: HSTUConfig,
    checkpoint: D3VersionCheckpoint,
    rank: int,
) -> torch.Tensor:
    if checkpoint.manifest is None or not 0 <= rank < len(
        checkpoint.shard_paths
    ):
        raise ValueError("M1 embedding shard request is invalid")
    world_size = len(checkpoint.shard_paths)
    payload = torch.load(
        checkpoint.shard_paths[rank],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    weight = payload.get("local_weight")
    expected_rows = int(
        checkpoint.manifest["embedding_shards"][rank]["local_rows"]
    )
    if (
        payload.get("schema") != SHARDED_EDGE_CHECKPOINT_SCHEMA
        or int(payload.get("version", -1)) != checkpoint.version
        or int(payload.get("rank", -1)) != rank
        or int(payload.get("world_size", -1)) != world_size
        or int(payload.get("num_embeddings", -1)) != cfg.num_items + 1
        or int(payload.get("hidden_size", -1)) != cfg.hidden_size
        or not isinstance(weight, torch.Tensor)
        or weight.shape != (expected_rows, cfg.hidden_size)
    ):
        raise ValueError("M1 embedding checkpoint payload differs")
    return weight


def load_runtime_sharded_hstu(
    cfg: HSTUConfig,
    checkpoint: D3VersionCheckpoint,
    rank: int,
    world_size: int,
    device: torch.device | str,
    process_group=None,
) -> D2ShardedHSTU:
    if checkpoint.layout == "full_table_single_file":
        if checkpoint.full_path is None:
            raise ValueError("full checkpoint path is missing")
        model = HSTU(cfg)
        model.load_state_dict(
            torch.load(
                checkpoint.full_path,
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        return build_modulo_sharded_hstu_from_cpu(
            model,
            rank,
            world_size,
            device,
            process_group=process_group,
        )
    if (
        checkpoint.manifest is None
        or int(checkpoint.manifest["world_size"]) != world_size
    ):
        raise ValueError("M1 runtime and checkpoint shard counts differ")
    dense = _load_dense_core(cfg, checkpoint).to(device)
    local_weight = _load_embedding_shard(
        cfg,
        checkpoint,
        rank,
    ).to(device=device, dtype=torch.float32)
    embedding = ModuloRowShardedEmbedding(
        local_weight,
        cfg.num_items + 1,
        rank,
        world_size,
        process_group=process_group,
    )
    return D2ShardedHSTU(
        dense_model=dense,
        item_embedding=embedding,
        rank=rank,
        world_size=world_size,
    )


def load_compact_hstu(
    cfg: HSTUConfig,
    checkpoint: D3VersionCheckpoint,
    used_global_item_ids: Sequence[int],
    device: torch.device | str,
) -> HSTU:
    ids = torch.as_tensor(
        used_global_item_ids,
        dtype=torch.long,
        device="cpu",
    )
    if (
        ids.ndim != 1
        or ids.numel() < 2
        or int(ids[0]) != 0
        or not bool(torch.all(ids[1:] > ids[:-1]))
        or int(ids[-1]) > cfg.num_items
    ):
        raise ValueError("compact item IDs must be sorted and include padding")
    if checkpoint.layout == "full_table_single_file":
        if checkpoint.full_path is None:
            raise ValueError("full checkpoint path is missing")
        full = HSTU(cfg)
        full.load_state_dict(
            torch.load(
                checkpoint.full_path,
                map_location="cpu",
                weights_only=True,
            )
        )
        rows = full.item_emb.weight.detach().index_select(0, ids)
        dense = full
    else:
        dense = _load_dense_core(cfg, checkpoint)
        rows = torch.empty(
            (ids.numel(), cfg.hidden_size),
            dtype=torch.float32,
        )
        world_size = len(checkpoint.shard_paths)
        for rank in range(world_size):
            positions = torch.nonzero(
                ids.remainder(world_size) == rank,
                as_tuple=False,
            ).flatten()
            if positions.numel() == 0:
                continue
            local_rows = ids.index_select(0, positions).div(
                world_size,
                rounding_mode="floor",
            )
            shard = _load_embedding_shard(cfg, checkpoint, rank)
            rows.index_copy_(
                0,
                positions,
                shard.index_select(0, local_rows),
            )
    compact_cfg = HSTUConfig(
        **{
            **asdict(cfg),
            "num_items": ids.numel() - 1,
            "num_prediction_items": ids.numel() - 1,
        }
    )
    embedding = ItemEmbedding(
        compact_cfg.num_items,
        compact_cfg.hidden_size,
    )
    with torch.no_grad():
        embedding.weight.copy_(rows)
    dense.cfg = compact_cfg
    dense.item_emb = embedding
    dense.eval()
    return dense.to(device)
