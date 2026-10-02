"""Producer scale formula, frozen base, native preservation and checkpoint."""

from io import BytesIO

import torch

from hstu_kvcache.design_one.item_kv import ItemKVReadViewAdapter
from hstu_kvcache.design_one.producer_scaled_item import ProducerScaledItemKVAdapter
from hstu_kvcache.models import HSTUKVCache


def test_producer_scale_formula_gradient_and_checkpoint():
    torch.manual_seed(1845)
    base = ItemKVReadViewAdapter(2, 2, 2, hidden_width=5, max_length=8).double()
    cache = HSTUKVCache(torch.randn(2, 2, 3, 4, dtype=torch.double),
                        torch.randn(2, 2, 3, 4, dtype=torch.double), 3)
    features = torch.randn(2, 3, 4, dtype=torch.double)
    mask = torch.tensor([[False, True, True], [False, False, True]])
    original = (cache.k.clone(), cache.v.clone(), features.clone())
    identity = ProducerScaledItemKVAdapter.from_item_adapter(base).map_cache(cache, features, mask)
    torch.testing.assert_close(identity.k, cache.k, atol=0, rtol=0)
    torch.testing.assert_close(identity.v, cache.v, atol=0, rtol=0)
    with torch.no_grad():
        for layer in base.layers:
            layer.output.weight.normal_(std=.1)
            layer.output.bias.normal_(std=.03)
            layer.set_normalization(torch.randn(12, dtype=torch.double),
                                    torch.rand(12, dtype=torch.double) + .5,
                                    torch.rand(8, dtype=torch.double) + .5)
    adapter = ProducerScaledItemKVAdapter.from_item_adapter(base)
    reference = base.map_cache(cache, features)
    same = adapter.map_cache(cache, features, mask)
    torch.testing.assert_close(same.k, reference.k, atol=0, rtol=0)
    torch.testing.assert_close(same.v, reference.v, atol=0, rtol=0)
    assert [name for name, value in adapter.named_parameters() if value.requires_grad] == [
        "layers.0.native_scale", "layers.1.native_scale"]
    with torch.no_grad():
        adapter.layers[0].native_scale.fill_(0.)
        adapter.layers[1].native_scale.fill_(-.4)
    actual = adapter.map_cache(cache, features, mask)
    for index, layer in enumerate(adapter.layers):
        source = torch.cat((cache.k[index], cache.v[index]), -1)
        delta = base.layers[index]._delta(torch.cat((source, features), -1))
        expected = source + torch.where(mask[..., None], delta * layer.native_scale, delta)
        torch.testing.assert_close(torch.cat((actual.k[index], actual.v[index]), -1), expected)
        torch.testing.assert_close(actual.k[index][~mask], reference.k[index][~mask], atol=0, rtol=0)
    torch.testing.assert_close(actual.k[0][mask], cache.k[0][mask], atol=0, rtol=0)
    (actual.k.sum() + actual.v.sum()).backward()
    for index, layer in enumerate(adapter.layers):
        source = torch.cat((cache.k[index], cache.v[index]), -1)
        expected_grad = base.layers[index]._delta(torch.cat((source, features), -1))[mask].sum()
        torch.testing.assert_close(layer.native_scale.grad, expected_grad)
        assert layer.native_scale.grad.abs() > 0
        assert all(value.grad is None for name, value in layer.named_parameters() if name != "native_scale")
    current_rows = int(mask.sum())
    cost = adapter.estimate_flops(batch=2, tokens=3, current_rows=current_rows)
    base_cost = base.estimate_flops(batch=2, tokens=3)
    assert cost["token_transform"] - base_cost["token_transform"] == 2 * 8 * current_rows
    assert cost["native_scale_flops"] == 2 * 8 * current_rows
    saved = BytesIO()
    torch.save(adapter.export_state(), saved)
    saved.seek(0)
    restored = ProducerScaledItemKVAdapter.from_state_dict(torch.load(saved, weights_only=True))
    loaded = restored.map_cache(cache, features, mask)
    torch.testing.assert_close(loaded.k, actual.k, atol=0, rtol=0)
    torch.testing.assert_close(loaded.v, actual.v, atol=0, rtol=0)
    assert sum(value.numel() for value in restored.parameters() if value.requires_grad) == 2
    for value, untouched in zip((cache.k, cache.v, features), original, strict=True):
        torch.testing.assert_close(value, untouched, atol=0, rtol=0)
