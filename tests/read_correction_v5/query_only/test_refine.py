from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from read_correction_v5.query_only import refine

import pytest
import torch

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.read_correction import score_corrected
from hstu_kvcache.read_correction_v5.query_only import QueryFeatureCorrection


@pytest.mark.parametrize('mode', ['head_phi', 'cross_phi'])
def test_output_units_fold_and_gradient_through_frozen_reader(mode):
    torch.manual_seed(66)
    model = HSTU(HSTUConfig(num_items=30, num_behaviors=3, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., temporal_num_freqs=2)).eval().requires_grad_(False)
    original = [QueryFeatureCorrection(2,4,mode) for _ in model.blocks]
    with torch.no_grad():
        for module in original:
            module.weight.normal_(std=.002)
            module.bias.normal_(std=.001)
    initial = [{name: value.clone() for name,value in m.state_dict().items()} for m in original]
    wrappers = torch.nn.ModuleList([refine.FeatureOutputUnits(m, .003) for m in original])
    cache = HSTUKVCache(torch.randn(2,2,5,8)*.1, torch.randn(2,2,5,8)*.1, 5)
    before = (cache.k.clone(), cache.v.clone())
    candidates, delta, counts = torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([2.,3.]), torch.tensor([5,5])
    args = (model, cache, candidates, delta)
    original_score = score_corrected(*args, original, counts)[0].detach()
    wrapped_score = score_corrected(*args, wrappers, counts)[0]
    torch.testing.assert_close(wrapped_score, original_score, atol=2e-6, rtol=2e-5)
    wrapped_score.square().mean().backward()
    for wrapper in wrappers:
        for parameter in (wrapper.module.weight,wrapper.module.bias):
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            assert parameter.grad.count_nonzero() > 0
        assert wrapper.units.grad is None
        assert wrapper.module.input_mean.grad is None
        assert wrapper.module.input_scale.grad is None
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        for parameter in wrappers.parameters():
            parameter.add_(parameter.grad, alpha=-.001)
    folded = [wrapper.folded() for wrapper in wrappers]
    actual = score_corrected(*args, folded, counts)[0]
    wanted = score_corrected(*args, wrappers, counts)[0]
    torch.testing.assert_close(actual, wanted, atol=2e-6, rtol=2e-5)
    assert torch.equal(cache.k, before[0]) and torch.equal(cache.v, before[1])
    for module, snapshot in zip(original,initial):
        assert all(torch.equal(value, snapshot[name]) for name,value in module.state_dict().items())
