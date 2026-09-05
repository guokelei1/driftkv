from __future__ import annotations

import torch
from insight_two.kv_response_coupling import (
    decompose_prefix_response,
    intervene_kv_response_coupling,
)

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


def _model() -> HSTU:
    torch.manual_seed(7)
    model = HSTU(
        HSTUConfig(
            num_items=31,
            num_behaviors=2,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            max_seq_len=8,
            input_dropout=0.0,
        )
    )
    return model.eval()


def test_finite_kv_response_decomposition_is_exact() -> None:
    model = _model()
    attention = model.blocks[0].attn
    q = torch.randn(3, attention.num_heads, 1, attention.head_dim)
    parent_k = torch.randn(1, 5, attention.inner)
    parent_v = torch.randn_like(parent_k)
    current_k = torch.randn_like(parent_k)
    current_v = torch.randn_like(parent_k)
    components = decompose_prefix_response(
        attention,
        q,
        current_k,
        current_v,
        parent_k,
        parent_v,
        candidates=3,
    )
    reconstructed = (
        components.parent
        + components.key
        + components.value
        + components.interaction
    )
    assert torch.allclose(reconstructed, components.current, atol=2e-6, rtol=2e-6)


def test_current_and_reuse_modes_match_native_reader() -> None:
    model = _model()
    items = torch.tensor([[1, 2, 3, 4]])
    actions = torch.tensor([[0, 1, 0, 1]])
    times = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    current_cache = model.compute_kv(items, actions, times)
    parent_cache = HSTUKVCache(
        k=current_cache.k + 0.03 * torch.randn_like(current_cache.k),
        v=current_cache.v + 0.03 * torch.randn_like(current_cache.v),
        seq_len=current_cache.seq_len,
    )
    candidates = torch.tensor([[5, 6, 7]])
    query_times = torch.tensor([5.0])
    current = intervene_kv_response_coupling(
        model,
        current_cache,
        parent_cache,
        candidates,
        query_times,
        mode="current",
    )
    reuse = intervene_kv_response_coupling(
        model,
        current_cache,
        parent_cache,
        candidates,
        query_times,
        mode="reuse",
    )
    assert torch.allclose(
        current.scores,
        model.score_cc_reuse(current_cache, candidates, query_times),
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.allclose(
        reuse.scores,
        model.score_cc_reuse(parent_cache, candidates, query_times),
        atol=2e-5,
        rtol=2e-5,
    )


def test_no_interaction_equals_current_when_keys_or_values_do_not_change() -> None:
    model = _model()
    items = torch.tensor([[1, 2, 3, 4]])
    actions = torch.tensor([[0, 1, 0, 1]])
    times = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    current_cache = model.compute_kv(items, actions, times)
    parent_cache = HSTUKVCache(
        k=current_cache.k.clone(),
        v=current_cache.v + 0.05 * torch.randn_like(current_cache.v),
        seq_len=current_cache.seq_len,
    )
    candidates = torch.tensor([[5, 6, 7]])
    query_times = torch.tensor([5.0])
    current = intervene_kv_response_coupling(
        model,
        current_cache,
        parent_cache,
        candidates,
        query_times,
        mode="current",
    )
    additive = intervene_kv_response_coupling(
        model,
        current_cache,
        parent_cache,
        candidates,
        query_times,
        mode="additive_no_interaction",
    )
    assert torch.allclose(additive.scores, current.scores, atol=2e-5, rtol=2e-5)
