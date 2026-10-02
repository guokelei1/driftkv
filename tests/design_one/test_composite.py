"""One tiny composition, checkpoint, native-storage and arithmetic check."""

from io import BytesIO

import torch

from hstu_kvcache.design_one.composite import FrozenItemResponseAdapter
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.design_one.item_kv import ItemKVReadViewAdapter
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


def test_composite_read_identity_checkpoint_and_cost(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(1861)
    item = ItemKVReadViewAdapter(2, 2, 3, hidden_width=4, max_length=5).double()
    with torch.no_grad():
        for layer in item.layers:
            layer.output.weight.normal_(std=.03)
            layer.output.bias.normal_(std=.03)
    inherited = {name: value.clone() for name, value in item.state_dict().items()}
    adapter = FrozenItemResponseAdapter.from_item_adapter(item, response_hidden_width=5)
    assert all(not value.requires_grad for value in adapter.item.parameters())
    assert all(value.requires_grad for value in adapter.response.parameters())
    model = HSTU(HSTUConfig(num_items=12, num_behaviors=2, hidden_size=6,
                            num_layers=2, num_heads=2, max_seq_len=5,
                            input_dropout=0., attn_dropout=0.)).double().eval()
    cache = HSTUKVCache(torch.randn(2, 2, 5, 6, dtype=torch.double),
                        torch.randn(2, 2, 5, 6, dtype=torch.double), 5)
    features = torch.randn(2, 5, 6, dtype=torch.double)
    untouched = (cache.k.clone(), cache.v.clone(), features.clone())
    mapped = adapter.map_cache(cache, features)
    query = torch.randn(2, 2, 3, 3, dtype=torch.double) * .2
    native = history_read(model.blocks[0].attn, query, cache.k[0], cache.v[0])
    mapped_read = history_read(model.blocks[0].attn, query, mapped.k[0], mapped.v[0])
    callback = adapter.make_history_override(model, mapped)
    torch.testing.assert_close(callback(0, query, native), mapped_read, atol=0, rtol=0)
    assert not torch.allclose(mapped_read, native)
    with torch.no_grad():
        for layer in adapter.response.layers:
            layer.output.weight.normal_(std=.02)
            layer.output.bias.normal_(std=.01)
    counts = torch.tensor([5, 5])
    expected = adapter.response.layers[0](query, mapped_read, counts)
    actual = callback(0, query, native)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(adapter.make_history_override(model, mapped, fitted_layers=0)(
        0, query, native), mapped_read, atol=0, rtol=0)
    # Fitting on mapped caches and serving through the two-read callback must
    # give the same actual higher-layer query path and final logits.
    candidates, deltas = torch.tensor([[1, 2], [3, 4]]), torch.tensor([1., 2.])
    direct = score(model, mapped, candidates, deltas,
                   history_override=adapter.response.make_history_override(counts))[0]
    composed = score(model, cache, candidates, deltas, history_override=callback)[0]
    torch.testing.assert_close(composed, direct, atol=1e-14, rtol=1e-14)
    saved = BytesIO()
    torch.save(adapter.export_state(), saved)
    saved.seek(0)
    restored = FrozenItemResponseAdapter.from_state_dict(torch.load(saved, weights_only=True))
    restored_read = restored.make_history_override(model, restored.map_cache(cache, features))
    torch.testing.assert_close(restored_read(0, query, native), expected, atol=0, rtol=0)
    for name, value in restored.item.state_dict().items():
        torch.testing.assert_close(value, inherited[name], atol=0, rtol=0)
    assert all(not value.requires_grad for value in restored.item.parameters())
    cost = adapter.estimate_flops(batch=2, tokens=5, candidates=3)
    item_cost = item.estimate_flops(batch=2, tokens=5, candidates=3)
    response_cost = adapter.response.estimate_flops(batch=2, candidates=3)
    assert cost == {**item_cost, **response_cost}
    assert set(cost) == {"token_transform", "extra_history_read", "candidate_reads"}
    for value, original in zip((cache.k, cache.v, features), untouched, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
