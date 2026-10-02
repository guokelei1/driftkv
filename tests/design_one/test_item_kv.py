"""One focused formula/checkpoint check for the item-feature map."""

from io import BytesIO

import torch
from torch.nn import functional as F

from hstu_kvcache.models import HSTUKVCache


from hstu_kvcache.design_one import item_kv


def test_item_kv_identity_formula_and_checkpoint_preserve_native_inputs():
    torch.manual_seed(1831)
    adapter = item_kv.ItemKVReadViewAdapter(1, 2, 2, hidden_width=5, max_length=8).double()
    cache = HSTUKVCache(torch.randn(1, 2, 3, 4, dtype=torch.double),
                        torch.randn(1, 2, 3, 4, dtype=torch.double), 3)
    item_features = torch.randn(2, 3, 4, dtype=torch.double)
    untouched = (cache.k.clone(), cache.v.clone(), item_features.clone())
    identity = adapter.map_cache(cache, item_features)
    torch.testing.assert_close(identity.k, cache.k, atol=0, rtol=0)
    torch.testing.assert_close(identity.v, cache.v, atol=0, rtol=0)
    layer = adapter.layers[0]
    layer.set_normalization(torch.randn(12, dtype=torch.double),
                            torch.rand(12, dtype=torch.double) + .5,
                            torch.rand(8, dtype=torch.double) + .5)
    with torch.no_grad():
        layer.output.weight.normal_(std=.1)
        layer.output.bias.normal_(std=.03)
    source = torch.cat((cache.k[0], cache.v[0]), -1)
    normalized = (torch.cat((source, item_features), -1) - layer.input_center) / layer.input_scale
    hidden = F.silu(F.linear(normalized, layer.input.weight, layer.input.bias))
    expected = source + F.linear(hidden, layer.output.weight, layer.output.bias) * layer.output_scale
    actual = adapter.map_cache(cache, item_features)
    torch.testing.assert_close(torch.cat((actual.k[0], actual.v[0]), -1), expected)
    # Item features are an actual learned input, not unused metadata.
    changed = adapter.map_cache(cache, item_features + 1.)
    assert not torch.allclose(changed.k, actual.k)
    saved = BytesIO()
    torch.save(adapter.export_state(), saved)
    saved.seek(0)
    restored = item_kv.ItemKVReadViewAdapter.from_state_dict(torch.load(saved, weights_only=True))
    loaded = restored.map_cache(cache, item_features)
    torch.testing.assert_close(loaded.k, actual.k)
    torch.testing.assert_close(loaded.v, actual.v)
    for value, original in zip((cache.k, cache.v, item_features), untouched, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
