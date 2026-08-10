from __future__ import annotations

import math

import torch

from hstu_kvcache.streaming.kuairand_projected_scale import (
    _capacity_physical_ids,
    _lookup,
)
from hstu_kvcache.streaming.xp_projected_edge import (
    TrainableProjectedModuloEmbedding,
)


def _embedding(source: torch.Tensor, replicas: int) -> TrainableProjectedModuloEmbedding:
    semantic_rows = source.shape[0] - 1
    weights = torch.zeros(semantic_rows * replicas + 1, source.shape[1])
    for replica in range(replicas):
        start = replica * semantic_rows + 1
        weights[start : start + semantic_rows] = source[1:] / math.sqrt(replicas)
    result = TrainableProjectedModuloEmbedding(
        local_weight=weights,
        projection_weight=torch.eye(source.shape[1]),
        num_embeddings=len(weights),
        rank=0,
        world_size=1,
    )
    result.semantic_rows = semantic_rows
    result.embedding_replicas = replicas
    result.embedding_capacity_multiplier = 1
    return result


def _capacity_embedding(
    source: torch.Tensor,
    multiplier: int,
) -> TrainableProjectedModuloEmbedding:
    semantic_rows = source.shape[0] - 1
    weights = torch.zeros(semantic_rows * multiplier + 1, source.shape[1])
    semantic_ids = torch.arange(1, semantic_rows + 1)
    physical_ids = _capacity_physical_ids(semantic_ids, multiplier)
    weights.index_copy_(0, physical_ids, source[1:])
    result = TrainableProjectedModuloEmbedding(
        local_weight=weights,
        projection_weight=torch.eye(source.shape[1]),
        num_embeddings=len(weights),
        rank=0,
        world_size=1,
    )
    result.semantic_rows = semantic_rows
    result.embedding_replicas = 1
    result.embedding_capacity_multiplier = multiplier
    return result


def test_replicated_rows_preserve_multifield_lookup_and_sgd_step() -> None:
    source = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [0.5, 1.5, 2.5],
            [3.0, 2.0, 1.0],
        ]
    )
    original = _embedding(source, 1)
    replicated = _embedding(source, 4)
    items = torch.tensor([[1, 2], [2, 1]])
    lengths = torch.tensor([2, 2])
    author_by_item = torch.tensor([0, 3, 4])
    original_values = _lookup(original, items, lengths, author_by_item)
    replicated_values = _lookup(replicated, items, lengths, author_by_item)
    torch.testing.assert_close(replicated_values, original_values)
    original_optimizer = torch.optim.SGD([original.local_weight], lr=0.05)
    replicated_optimizer = torch.optim.SGD([replicated.local_weight], lr=0.05)
    original_values.square().sum().backward()
    replicated_values.square().sum().backward()
    original_optimizer.step()
    replicated_optimizer.step()
    torch.testing.assert_close(
        _lookup(replicated, items, lengths, author_by_item),
        _lookup(original, items, lengths, author_by_item),
        rtol=1e-5,
        atol=1e-6,
    )


def test_capacity_rows_preserve_two_field_lookup_exactly() -> None:
    source = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [0.5, 1.5, 2.5],
            [3.0, 2.0, 1.0],
        ]
    )
    original = _embedding(source, 1)
    expanded = _capacity_embedding(source, 8)
    items = torch.tensor([[1, 2], [2, 1]])
    lengths = torch.tensor([2, 2])
    author_by_item = torch.tensor([0, 3, 4])
    torch.testing.assert_close(
        _lookup(expanded, items, lengths, author_by_item),
        _lookup(original, items, lengths, author_by_item),
        rtol=0.0,
        atol=0.0,
    )
    physical_ids = _capacity_physical_ids(torch.arange(1, 1001), 8)
    assert int(physical_ids.max()) > 7900
    assert set(torch.remainder(physical_ids, 2).tolist()) == {0, 1}
