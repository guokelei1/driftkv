"""Regression for Adam's absolute step size across very different response units."""
from pathlib import Path
import sys

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[3] / "scripts"),
               str(Path(__file__).resolve().parents[3] / "src")]

from hstu_kvcache.read_correction import QueryCorrection
from read_correction_2026_09.history_conditioned.fit import fit


def data():
    generator = torch.Generator().manual_seed(109)
    query = torch.randn(12, 2, 3, 2, generator=generator)
    keys = torch.randn(12, 6, 4, generator=generator)
    values = torch.randn(12, 6, 4, generator=generator)
    target = values.reshape(12, 6, 2, 2).mean(1)[:, :, None].expand(-1, -1, 3, -1).clone()
    counts = torch.full((12,), 6)
    return query, keys, values, target, counts


def test_foldback_preserves_affine_units_and_input_module():
    torch.set_num_threads(1)
    query, keys, values, target, counts = data()
    initial = QueryCorrection(2, 2)
    units = torch.tensor([1e-4, 1e4])
    with torch.no_grad():
        initial.weight.copy_(torch.eye(2)[None].expand(2, -1, -1)*units[:, None, None])
        initial.bias.copy_(torch.ones(2, 2)*units[:, None])
    previous = {name: value.clone() for name, value in initial.state_dict().items()}
    expected = initial.rate(query)
    module, stats = fit(query, target*units[None, :, None, None], keys, values, counts,
                        initial_query=initial, width=8, epochs=0, device="cpu")
    torch.testing.assert_close(module.rate(query, keys, values, counts), expected, rtol=3e-6, atol=1e-8)
    for name, value in initial.state_dict().items():
        torch.testing.assert_close(value, previous[name], rtol=0, atol=0)
    assert stats["output_scale_per_head"][1] / stats["output_scale_per_head"][0] > 1e7


def test_training_is_invariant_to_large_independent_head_units():
    torch.set_num_threads(1)
    query, keys, values, target, counts = data()
    units = torch.tensor([1e-4, 1e4])
    settings = dict(initial_query=QueryCorrection(2, 2), width=8, epochs=120,
                    learning_rate=.03, weight_decay=0., batch_size=12, seed=17, device="cpu")
    ordinary, a = fit(query, target, keys, values, counts, **settings)
    scaled, b = fit(query, target*units[None, :, None, None], keys, values, counts, **settings)
    with torch.no_grad():
        unscaled_prediction = scaled.rate(query, keys, values, counts) / units[None, :, None, None]
        torch.testing.assert_close(unscaled_prediction, ordinary.rate(query, keys, values, counts), rtol=3e-4, atol=3e-5)
    for before, after in zip(b["initial_rate_mse_per_head"], b["final_rate_mse_per_head"]):
        assert after < .1*before
    assert a["output_normalization_flops"] == b["output_normalization_flops"]
