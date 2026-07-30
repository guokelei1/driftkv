from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from hstu_kvcache.migration.design2_plan import file_sha256
from hstu_kvcache.migration.design3_checkpoint import (
    load_compact_hstu,
    load_runtime_sharded_hstu,
    resolve_version_checkpoint,
    training_model_config,
)
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming.sharded_edge import (
    SHARDED_EDGE_CHECKPOINT_SCHEMA,
    ExternalEmbeddingHSTU,
    modulo_local_rows,
)


def _cfg() -> HSTUConfig:
    return HSTUConfig(
        num_items=10,
        num_prediction_items=7,
        num_behaviors=3,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        max_seq_len=6,
        input_dropout=0.0,
    )


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _write_sharded(
    root: Path,
    cfg: HSTUConfig,
) -> tuple[dict[str, object], torch.Tensor]:
    directory = root / "theta_0"
    directory.mkdir(parents=True)
    dense = ExternalEmbeddingHSTU(cfg)
    dense_path = directory / "dense.pt"
    torch.save(
        {
            "schema": SHARDED_EDGE_CHECKPOINT_SCHEMA,
            "version": 0,
            "config": asdict(cfg),
            "state_dict": dense.state_dict(),
        },
        dense_path,
    )
    full_weight = torch.arange(
        (cfg.num_items + 1) * cfg.hidden_size,
        dtype=torch.float32,
    ).view(cfg.num_items + 1, cfg.hidden_size)
    full_weight[0].zero_()
    shard_records = []
    for rank in range(2):
        path = directory / f"embedding_rank_{rank:05d}.pt"
        local_weight = full_weight[rank::2].contiguous()
        torch.save(
            {
                "schema": SHARDED_EDGE_CHECKPOINT_SCHEMA,
                "version": 0,
                "rank": rank,
                "world_size": 2,
                "num_embeddings": cfg.num_items + 1,
                "hidden_size": cfg.hidden_size,
                "global_row_start": rank,
                "global_row_stride": 2,
                "local_rows": local_weight.shape[0],
                "local_weight": local_weight,
            },
            path,
        )
        shard_records.append(
            {
                "rank": rank,
                **_artifact(path),
                "local_rows": modulo_local_rows(
                    cfg.num_items + 1,
                    rank,
                    2,
                ),
                "global_row_start": rank,
                "global_row_stride": 2,
            }
        )
    spec = {
        "num_embeddings": cfg.num_items + 1,
        "num_prediction_items": cfg.num_prediction_items,
        "num_behaviors": cfg.num_behaviors,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "head_dim": cfg.head_dim,
        "max_seq_len": cfg.max_seq_len,
    }
    manifest = {
        "schema": SHARDED_EDGE_CHECKPOINT_SCHEMA,
        "version": 0,
        "world_size": 2,
        "spec": spec,
        "dense": _artifact(dense_path),
        "embedding_shards": shard_records,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True)
    )
    training = {
        "model": {
            "spec": spec,
            "dense_config": asdict(cfg),
            "embedding_layout": "modulo_row_sharded",
        },
        "checkpoints": {"theta0": manifest},
    }
    return training, full_weight


def test_sharded_manifest_loads_only_compact_rows_and_local_runtime_shard(
    tmp_path: Path,
) -> None:
    cfg = _cfg()
    training, full_weight = _write_sharded(tmp_path, cfg)
    checkpoint = resolve_version_checkpoint(
        training,
        tmp_path,
        0,
    )
    used = (0, 1, 4, 9)
    compact = load_compact_hstu(
        cfg,
        checkpoint,
        used,
        "cpu",
    )
    runtime = load_runtime_sharded_hstu(
        cfg,
        checkpoint,
        rank=1,
        world_size=2,
        device="cpu",
    )

    assert checkpoint.layout == "modulo_row_sharded_manifest"
    assert training_model_config(training) == cfg
    assert torch.equal(
        compact.item_emb.weight,
        full_weight[list(used)],
    )
    assert compact.item_emb.weight.shape == (4, cfg.hidden_size)
    assert torch.equal(
        runtime.item_embedding.local_weight,
        full_weight[1::2],
    )


def test_legacy_single_file_checkpoint_remains_supported(
    tmp_path: Path,
) -> None:
    cfg = _cfg()
    model = HSTU(cfg)
    path = tmp_path / "theta_0.pt"
    torch.save(model.state_dict(), path)
    training = {
        "model": asdict(cfg),
        "checkpoints": [
            {
                "version": "theta0",
                "sha256": file_sha256(path),
            }
        ],
    }
    checkpoint = resolve_version_checkpoint(
        training,
        tmp_path,
        0,
    )
    compact = load_compact_hstu(
        cfg,
        checkpoint,
        (0, 2, 5),
        "cpu",
    )
    runtime = load_runtime_sharded_hstu(
        cfg,
        checkpoint,
        rank=0,
        world_size=1,
        device="cpu",
    )

    assert checkpoint.layout == "full_table_single_file"
    assert torch.equal(
        compact.item_emb.weight,
        model.item_emb.weight[[0, 2, 5]],
    )
    assert torch.equal(
        runtime.item_embedding.local_weight,
        model.item_emb.weight,
    )
