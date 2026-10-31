from __future__ import annotations

import torch
import pytest

from hstu_kvcache.models import HSTU, HSTUConfig


def test_append_only_one_token_matches_dense_cache_append_without_prefix_copy() -> None:
    torch.manual_seed(71)
    model = HSTU(HSTUConfig(
        num_items=32, num_behaviors=4, hidden_size=16, num_layers=2,
        num_heads=2, max_seq_len=8, input_dropout=0.0, attn_dropout=0.0,
    )).eval()
    items = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    behaviors = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
    deltas = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    prefix = model.compute_kv(items[:, :-1], behaviors[:, :-1], deltas[:, :-1])
    dense_hidden, dense_cache = model.forward_with_cache(
        prefix, items[:, -1:], behaviors[:, -1:], deltas[:, -1:]
    )
    append_hidden, appended = model.forward_with_cache_new_kv(
        prefix, items[:, -1:], behaviors[:, -1:], deltas[:, -1:]
    )
    assert appended.seq_len == 1
    assert torch.allclose(append_hidden, dense_hidden, atol=1e-6, rtol=1e-5)
    assert torch.allclose(appended.k, dense_cache.k[:, :, -1:, :], atol=1e-6, rtol=1e-5)
    assert torch.allclose(appended.v, dense_cache.v[:, :, -1:, :], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("variant,diagonal", [
    ("hstu_reference", "inclusive"), ("legacy", "exclusive"),
])
def test_one_token_append_preserves_reference_divisor_and_causal_boundary(variant, diagonal):
    from hstu_kvcache.models.attention import PointwiseAttention, PointwiseAttentionConfig

    torch.manual_seed(17)
    attention = PointwiseAttention(PointwiseAttentionConfig(
        hidden_size=32, num_heads=2, max_seq_len=16,
        block_variant=variant, activation="silu" if variant == "hstu_reference" else "elu_plus1",
        relative_position_bias=variant == "hstu_reference", causal_diagonal=diagonal,
    )).eval()
    prefix = torch.randn(2, 7, 32)
    token = torch.randn(2, 1, 32)
    with torch.no_grad():
        _, (k, v) = attention(prefix, return_kv=True)
        expected, (all_k, all_v) = attention.forward_with_cache(token, k, v)
        actual, (new_k, new_v) = attention.forward_one_with_cache_new_kv(token, k, v)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(new_k, all_k[:, -1:])
    torch.testing.assert_close(new_v, all_v[:, -1:])
