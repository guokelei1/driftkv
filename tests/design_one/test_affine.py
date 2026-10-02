"""One CPU formula/checkpoint check for the full-rank affine view."""

from io import BytesIO

import torch

from hstu_kvcache.models import HSTUKVCache


from hstu_kvcache.design_one import affine


def test_affine_identity_physical_units_context_formula_and_checkpoint():
    torch.manual_seed(1830)
    adapter = affine.AffineKVReadViewAdapter(1, 2, 2, max_length=8).double()
    cache = HSTUKVCache(torch.randn(1, 2, 3, 4, dtype=torch.double),
                        torch.randn(1, 2, 3, 4, dtype=torch.double), 3)
    context = torch.tensor([[[1., 1.], [0., .75], [0., .5]],
                            [[1., 1.], [1., 1.], [0., .5]]], dtype=torch.double)
    untouched = (cache.k.clone(), cache.v.clone())
    identity = adapter.map_cache(cache, context)
    torch.testing.assert_close(identity.k, cache.k, atol=0, rtol=0)
    torch.testing.assert_close(identity.v, cache.v, atol=0, rtol=0)
    layer = adapter.layers[0]
    center = torch.randn(10, dtype=torch.double)
    scale = torch.rand(10, dtype=torch.double) + .5
    weight = torch.randn(10, 8, dtype=torch.double) * .05
    bias = torch.randn(8, dtype=torch.double) * .1
    layer.set_parameters(center, scale, weight, bias)
    source = torch.cat((cache.k[0], cache.v[0]), dim=-1)
    expected = source + ((torch.cat((source, context), -1) - center) / scale) @ weight + bias
    mapped = adapter.map_cache(cache, context)
    torch.testing.assert_close(torch.cat((mapped.k[0], mapped.v[0]), -1), expected)
    saved = BytesIO()
    torch.save(adapter.export_state(), saved)
    saved.seek(0)
    restored = affine.AffineKVReadViewAdapter.from_state_dict(torch.load(saved, weights_only=True))
    loaded = restored.map_cache(cache, context)
    torch.testing.assert_close(loaded.k, mapped.k)
    torch.testing.assert_close(loaded.v, mapped.v)
    torch.testing.assert_close(cache.k, untouched[0], atol=0, rtol=0)
    torch.testing.assert_close(cache.v, untouched[1], atol=0, rtol=0)
    assert len(list(adapter.parameters())) == 0
    assert adapter.estimate_flops(batch=2, tokens=3)["token_transform"] == layer.token_forward_flops(batch=2, length=3)
