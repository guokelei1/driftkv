from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn import functional as F

from hstu_kvcache.migration.foundation_projection import (
    FoundationProjectedModuloEmbedding,
    foundation_projection_capacity,
)


def test_foundation_projection_capacity_formula() -> None:
    rank_zero = foundation_projection_capacity(
        num_embeddings=11,
        embedding_width=5,
        output_width=3,
        rank=0,
        world_size=4,
    )
    rank_three = foundation_projection_capacity(
        num_embeddings=11,
        embedding_width=5,
        output_width=3,
        rank=3,
        world_size=4,
    )
    assert rank_zero.local_embedding_rows == 3
    assert rank_three.local_embedding_rows == 2
    assert rank_zero.global_embedding_parameter_bytes == 11 * 5 * 4
    assert rank_zero.local_embedding_parameter_bytes == 3 * 5 * 4
    assert rank_three.local_embedding_parameter_bytes == 2 * 5 * 4
    assert rank_zero.projection_parameter_bytes == 3 * 5 * 4


def test_world_one_gloo_project_before_return_matches_dense(
    tmp_path,
) -> None:
    rendezvous = tmp_path / "foundation_projection_gloo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        weight = torch.arange(
            35,
            dtype=torch.float32,
        ).reshape(7, 5) / 13.0
        projection = torch.tensor(
            [
                [0.3, -0.2, 0.1, 0.4, -0.5],
                [-0.4, 0.2, 0.6, -0.1, 0.3],
                [0.1, 0.5, -0.3, 0.2, 0.4],
            ],
            dtype=torch.float32,
        )
        embedding = FoundationProjectedModuloEmbedding(
            local_weight=weight,
            projection_weight=projection,
            num_embeddings=7,
            rank=0,
            world_size=1,
            response_dtype=torch.float32,
        )
        item_ids = torch.tensor(
            [[0, 2, 99], [3, 6, -7]],
            dtype=torch.long,
        )
        lengths = torch.tensor([2, 2], dtype=torch.long)
        result = embedding.lookup(item_ids, lengths)
        expected = torch.zeros((2, 3, 3), dtype=torch.float32)
        expected[0, :2] = F.linear(weight[[0, 2]], projection)
        expected[1, :2] = F.linear(weight[[3, 6]], projection)
        torch.testing.assert_close(result.item_vectors, expected)
        assert embedding.embedding_width == 5
        assert embedding.output_width == 3
        assert embedding.capacity.local_embedding_parameter_bytes == (
            7 * 5 * 4
        )
        assert embedding.capacity.projection_parameter_bytes == 3 * 5 * 4
        metrics = result.metrics
        assert metrics.requested_tokens == 4
        assert metrics.local_requested_tokens == 4
        assert metrics.remote_requested_tokens == 0
        assert metrics.returned_tensor_bytes == 2 * 3 * 3 * 4
        assert metrics.returned_valid_vector_bytes == 4 * 3 * 4
        assert metrics.counts_collective_tensor_bytes == 0
        assert metrics.id_collective_tensor_bytes == 0
        assert metrics.vector_collective_tensor_bytes == 0
        assert metrics.collective_tensor_bytes == 0
        assert metrics.off_diagonal_bytes == 0
        assert metrics.collective_calls == 0
        assert metrics.lookup_seconds >= metrics.collective_seconds
    finally:
        dist.destroy_process_group()


def test_world_one_response_dtype_is_independent_of_table_dtype() -> None:
    weight = torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(6, 4) / 9.0
    projection = torch.arange(
        12,
        dtype=torch.float32,
    ).reshape(3, 4) / 7.0
    embedding = FoundationProjectedModuloEmbedding(
        local_weight=weight,
        projection_weight=projection,
        num_embeddings=6,
        rank=0,
        world_size=1,
        response_dtype=torch.float16,
    )
    item_ids = torch.tensor([[1, 5]], dtype=torch.long)
    lengths = torch.tensor([2], dtype=torch.long)
    result = embedding.lookup(item_ids, lengths)
    expected = F.linear(weight[[1, 5]], projection).to(torch.float16)
    torch.testing.assert_close(result.item_vectors[0], expected)
    assert result.item_vectors.dtype == torch.float16
    assert result.metrics.response_element_bytes == 2
    assert result.metrics.returned_tensor_bytes == 2 * 3 * 2
