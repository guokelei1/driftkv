import pytest
import torch

from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.models import HSTU, HSTUConfig


@pytest.mark.parametrize("width", [1, 4, 32])
@pytest.mark.parametrize("initial_length", [3, 8])
def test_band_cache_and_actual_boundaries_match_scalar_append_eviction(width, initial_length):
    torch.manual_seed(17)
    parent = HSTU(HSTUConfig(num_items=50, num_behaviors=3, hidden_size=12,
                            num_layers=3, num_heads=3, max_seq_len=64,
                            input_dropout=0.2, relative_position_bias=False)).eval()
    torch.manual_seed(29)
    current = HSTU(parent.cfg).eval()
    total = initial_length + width
    items = torch.arange(2 * total).reshape(2, total) % 49 + 1
    behaviors = torch.ones_like(items)
    deltas = torch.arange(total).float()[None].expand(2, -1)
    raw = (items, behaviors, deltas)
    initial = lr.capture_state(parent, *(v[:, :initial_length] for v in raw))
    old_keys = initial.cache.k.clone()
    current.train()
    actual = lr.append_band(current, initial, *(v[:, initial_length:] for v in raw), max_length=8)
    assert current.training
    scalar = initial
    for position in range(initial_length, total):
        if scalar.cache.seq_len >= 8:
            scalar = lr.retain_latest(scalar, 7)
        scalar = lr.append(current, scalar, *(v[:, position:position + 1] for v in raw))
    assert actual.cache.seq_len == scalar.cache.seq_len == min(8, total)
    torch.testing.assert_close(actual.cache.k, scalar.cache.k, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual.cache.v, scalar.cache.v, atol=3e-6, rtol=3e-5)
    for layer in actual.boundary_inputs:
        torch.testing.assert_close(actual.boundary_inputs[layer], scalar.boundary_inputs[layer],
                                   atol=3e-6, rtol=3e-5)
    assert torch.equal(initial.cache.k, old_keys)
