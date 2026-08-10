from __future__ import annotations

import torch
from torch.nn import functional as F

from hstu_kvcache.streaming.sharded_edge import ExternalEmbeddingHSTU
from hstu_kvcache.streaming.xp_projected_edge import (
    OptimizerActiveRowTracker,
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
    active_row_ids_sha256,
    load_xp_projected_checkpoint,
    save_xp_projected_checkpoint,
    sparse_embedding_sgd,
    tracked_sparse_optimizer_step,
)


def _spec() -> XPProjectedModelSpec:
    return XPProjectedModelSpec(
        num_embeddings=11,
        embedding_width=5,
        hidden_size=4,
        num_prediction_items=6,
        num_behaviors=3,
        num_layers=1,
        num_heads=2,
        head_dim=2,
        max_seq_len=8,
    )


def test_world_one_projected_training_and_checkpoint_roundtrip(
    tmp_path,
) -> None:
    spec = _spec()
    weight = torch.arange(
        spec.num_embeddings * spec.embedding_width,
        dtype=torch.float32,
    ).reshape(spec.num_embeddings, spec.embedding_width) / 31.0
    weight[0].zero_()
    projection = torch.arange(
        spec.hidden_size * spec.embedding_width,
        dtype=torch.float32,
    ).reshape(spec.hidden_size, spec.embedding_width) / 19.0
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=weight.clone(),
        projection_weight=projection.clone(),
        num_embeddings=spec.num_embeddings,
        rank=0,
        world_size=1,
    )
    torch.manual_seed(77)
    dense = ExternalEmbeddingHSTU(spec.hstu_config())
    tracker = OptimizerActiveRowTracker(
        num_embeddings=spec.num_embeddings,
        rank=0,
        world_size=1,
    )
    item_ids = torch.tensor(
        [[0, 2, 4, 99], [3, 7, 99, 99]],
        dtype=torch.int64,
    )
    lengths = torch.tensor([3, 2], dtype=torch.int64)
    output = embedding(item_ids, lengths)
    expected = torch.zeros_like(output)
    expected[0, :3] = F.linear(weight[[0, 2, 4]], projection)
    expected[1, :2] = F.linear(weight[[3, 7]], projection)
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    assert embedding.local_weight.grad is not None
    assert embedding.local_weight.grad.is_sparse
    optimizer = sparse_embedding_sgd(embedding, learning_rate=0.1)
    active = tracked_sparse_optimizer_step(
        embedding,
        optimizer,
        tracker,
    )
    assert active == (2, 3, 4, 7)
    assert torch.count_nonzero(embedding.local_weight[0]) == 0
    manifest = save_xp_projected_checkpoint(
        tmp_path,
        1,
        spec,
        dense,
        embedding,
        tracker,
    )
    ledger = manifest["optimizer_active_rows"]
    assert ledger["global_active_rows"] == 4
    assert ledger["global_row_ids_sha256"] == active_row_ids_sha256(
        active
    )
    assert ledger["padding_row_excluded"] is True
    assert manifest["scientific_result"] is False
    reloaded_embedding = TrainableProjectedModuloEmbedding(
        local_weight=torch.zeros_like(embedding.local_weight),
        projection_weight=torch.zeros_like(
            embedding.projection_weight
        ),
        num_embeddings=spec.num_embeddings,
        rank=0,
        world_size=1,
    )
    torch.manual_seed(91)
    reloaded_dense = ExternalEmbeddingHSTU(spec.hstu_config())
    reloaded_tracker = OptimizerActiveRowTracker(
        num_embeddings=spec.num_embeddings,
        rank=0,
        world_size=1,
    )
    loaded = load_xp_projected_checkpoint(
        tmp_path,
        1,
        spec,
        reloaded_dense,
        reloaded_embedding,
        reloaded_tracker,
    )
    assert loaded == manifest
    torch.testing.assert_close(
        reloaded_embedding.local_weight,
        embedding.local_weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        reloaded_embedding.projection_weight,
        embedding.projection_weight,
        rtol=0,
        atol=0,
    )
    assert reloaded_tracker.local_global_row_ids() == active
    for name, value in dense.state_dict().items():
        torch.testing.assert_close(
            reloaded_dense.state_dict()[name],
            value,
            rtol=0,
            atol=0,
        )
