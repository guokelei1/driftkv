"""Compact contractions preserve the materialized Item+Response read."""

import torch

from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.design_one.compact import CompactItemResponseAdapter, CompactItemState
from hstu_kvcache.design_one.composite import FrozenItemResponseAdapter
from hstu_kvcache.design_one.item_kv import ItemKVReadViewAdapter
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


def test_compact_read_score_native_state_and_lifecycle(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(1902)
    item = ItemKVReadViewAdapter(2, 2, 3, hidden_width=4, max_length=5)
    base = FrozenItemResponseAdapter.from_item_adapter(item, response_hidden_width=5)
    with torch.no_grad():
        for layer in (*base.item.layers, *base.response.layers):
            layer.input_center.normal_(std=.3)
            layer.input_scale.uniform_(.5, 2.)
            layer.output_scale.uniform_(.5, 2.)
            layer.output.weight.normal_(std=.08)
            layer.output.bias.normal_(std=.04)
    base.eval().requires_grad_(False)
    compact = CompactItemResponseAdapter(base).eval()
    model = HSTU(HSTUConfig(num_items=12, num_behaviors=2, hidden_size=6,
                            num_layers=2, num_heads=2, max_seq_len=5,
                            input_dropout=0., attn_dropout=0.)).eval()
    cache = HSTUKVCache(torch.randn(2, 2, 5, 6), torch.randn(2, 2, 5, 6), 5)
    features = torch.randn(2, 5, 6)
    untouched = (cache.k.clone(), cache.v.clone(), features.clone())
    state = compact.encode_cache(cache, features)
    mapped = base.map_cache(cache, features)
    # Several queries per owner check the head-specific contractions and both
    # nonzero affine biases. Nontrivial scales catch incorrectly folded units.
    query = torch.randn(2, 2, 3, 3) * .7
    expected = history_read(model.blocks[0].attn, query, mapped.k[0], mapped.v[0])
    actual = compact.history_read(model.blocks[0].attn, 0, query, cache, state)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    candidates, deltas = torch.tensor([[1, 2], [3, 4]]), torch.tensor([1., 2.])
    expected_logits = score(model, cache, candidates, deltas,
                            history_override=base.make_history_override(model, mapped))[0]
    actual_logits = score(model, cache, candidates, deltas,
                          history_override=compact.make_history_override(model, cache, state))[0]
    torch.testing.assert_close(actual_logits, expected_logits, atol=2e-6, rtol=2e-6)
    # Appends encode only arriving native rows; removing two old rows yields
    # the same state as encoding the corresponding native cache from scratch.
    new = HSTUKVCache(torch.randn(2, 2, 2, 6), torch.randn(2, 2, 2, 6), 2)
    new_features = torch.randn(2, 2, 6)
    suffix = compact.encode_cache(new, new_features)
    advanced = CompactItemState(torch.cat((state.hidden[:, :, 2:], suffix.hidden), dim=2), 5)
    next_cache = HSTUKVCache(torch.cat((cache.k[:, :, 2:], new.k), dim=2),
                            torch.cat((cache.v[:, :, 2:], new.v), dim=2), 5)
    reference = compact.encode_cache(next_cache, torch.cat((features[:, 2:], new_features), dim=1))
    # GEMM row-count changes may alter FP32 rounding when encoding a suffix.
    torch.testing.assert_close(advanced.hidden, reference.hidden, atol=1e-7, rtol=2e-6)
    owners = torch.tensor([1, 0, 1])
    torch.testing.assert_close(advanced.select(owners).hidden,
                               advanced.hidden[:, owners], atol=0, rtol=0)
    assert state.storage_bytes() == state.hidden.numel() * 4
    assert not hasattr(state, "k") and not hasattr(state, "v")
    for value, original in zip((cache.k, cache.v, features), untouched, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)


def test_compact_cost_and_storage_at_medium_dimensions():
    base = FrozenItemResponseAdapter.from_item_adapter(
        ItemKVReadViewAdapter(6, 6, 32, hidden_width=64), response_hidden_width=256)
    compact = CompactItemResponseAdapter(base)
    state = CompactItemState(torch.empty(6, 1, 1024, 64), 1024)
    native_bytes = 2 * 6 * 1024 * 192 * 4
    assert state.storage_bytes() * 6 == native_bytes
    assert compact.projection_storage_bytes() == compact.setup_flops() * 4
    cost = compact.estimate_flops(tokens=1024, candidates=1)
    old = base.estimate_flops(tokens=1024, candidates=1)
    assert cost["token_transform"] < old["token_transform"]
    assert cost["candidate_reads"] == old["candidate_reads"]
    # The algebraic contractions must cost less than reconstructing a dense
    # 64->384 residual at every position, then running the ordinary read.
    dense_decode = 6 * 1024 * (2 * 64 * 384 + 3 * 384)
    assert old["extra_history_read"] < cost["extra_history_read"] < old["extra_history_read"] + dense_decode
    # With repeated reads, encoding once plus contracted reads is cheaper than
    # the full Item MLP rerun independently for every query.
    queries = 10
    repeated = compact.estimate_flops(tokens=1024, candidates=queries)
    compact_service = cost["token_transform"] + repeated["extra_history_read"] + repeated["candidate_reads"]
    naive = queries * sum(old.values())
    assert compact_service < naive
