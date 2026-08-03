from __future__ import annotations

import math

import torch

from hstu_kvcache.streaming.multifield_projected import (
    lookup_multifield_projected,
)
from hstu_kvcache.streaming.xp_projected_edge import (
    TrainableProjectedModuloEmbedding,
)


def test_multifield_lookup_matches_merged_reference() -> None:
    embedding = TrainableProjectedModuloEmbedding.initialize(
        num_embeddings=17,
        embedding_width=8,
        hidden_size=4,
        rank=0,
        world_size=1,
        device="cpu",
        embedding_seed=3,
        projection_seed=5,
    )
    feature_ids = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 0, 0]]],
        dtype=torch.int64,
    )
    lengths = torch.tensor([2, 1], dtype=torch.int64)
    actual = lookup_multifield_projected(embedding, feature_ids, lengths)
    flattened = feature_ids.reshape(2, 6)
    expected = embedding(flattened, lengths * 3).reshape(2, 2, 3, 4).sum(dim=2)
    expected = expected / math.sqrt(3)
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[1, 1]).item() == 0
