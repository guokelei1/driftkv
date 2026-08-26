from __future__ import annotations

import torch

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
