from __future__ import annotations

import torch
from insight_two.common_projection_response import (
    build_common_projection_response_memory,
    factorized_prefix_heads,
    intervene_common_projection_response,
    medium_common_projection_cost,
)
from insight_two.cone_response_memory import _native_prefix_heads
from insight_two.mode_space_replay import FactorizedCacheLayer, FactorizedReplay

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


def _model() -> HSTU:
    torch.manual_seed(13)
    return HSTU(
        HSTUConfig(
            num_items=31,
            num_behaviors=2,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            max_seq_len=8,
            input_dropout=0.0,
        )
    ).eval()


def _full_rank_replay(cache: HSTUKVCache) -> FactorizedReplay:
    length = cache.seq_len
    identity = torch.eye(length).unsqueeze(0)
    layers = tuple(
        FactorizedCacheLayer(identity, cache.k[layer], cache.v[layer])
        for layer in range(cache.k.shape[0])
    )
    return FactorizedReplay(cache=cache, layers=layers, block_input_factors=())


def test_factorized_native_response_matches_dense() -> None:
    model = _model()
    attention = model.blocks[0].attn
    left = torch.randn(1, 5, 3)
    key_core = torch.randn(1, 3, attention.inner)
    value_core = torch.randn_like(key_core)
    layer = FactorizedCacheLayer(left, key_core, value_core)
    q = torch.randn(4, attention.num_heads, 1, attention.head_dim)
    key, value = layer.materialize()
    expected = _native_prefix_heads(attention, q, key, value)
    observed = factorized_prefix_heads(attention, q, layer)
    assert torch.allclose(observed, expected, atol=2e-5, rtol=2e-5)


def test_full_rank_common_projection_recovers_current_reader() -> None:
    model = _model()
    items = torch.tensor([[1, 2, 3, 4]])
    actions = torch.tensor([[0, 1, 0, 1]])
    times = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    current = model.compute_kv(items, actions, times)
    parent = HSTUKVCache(
        k=current.k + 0.03 * torch.randn_like(current.k),
        v=current.v + 0.03 * torch.randn_like(current.v),
        seq_len=current.seq_len,
    )
    memory = build_common_projection_response_memory(
        _full_rank_replay(current), parent
    )
    candidates = torch.tensor([[5, 6, 7]])
    query_time = torch.tensor([5.0])
    observed = intervene_common_projection_response(
        model, parent, memory, candidates, query_time
    )
    expected = model.score_cc_reuse(current, candidates, query_time)
    assert torch.allclose(observed.scores, expected, atol=2e-5, rtol=2e-5)


def test_conservative_medium_cost_is_below_twenty_percent() -> None:
    cost = medium_common_projection_cost()
    assert cost.total_constructor_flops == 930_118_850
    assert cost.within_twenty_percent
    assert 0.194 < cost.constructor_fraction < 0.196
    assert cost.incremental_reader_flops_per_query == 2_435_328
