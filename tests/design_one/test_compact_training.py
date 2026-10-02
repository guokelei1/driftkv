"""Live compact training preserves both Item MLP affine-map gradients."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from design_one.calibrate_nonlinear import aggregate_loss, compact_item_read, compact_item_fit_flops
from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.design_one.item_kv import ItemKVViewLayer
from hstu_kvcache.models import HSTU, HSTUConfig


@pytest.mark.parametrize("dtype", (torch.float64, torch.float32))
def test_compact_output_and_all_parameter_gradients_match_materialized(dtype):
    torch.manual_seed(2002)
    module = ItemKVViewLayer(2, 3, hidden_width=4).to(dtype)
    with torch.no_grad():
        module.input_center.normal_(std=.3)
        module.input_scale.uniform_(.5, 2.)
        module.output_scale.uniform_(.5, 2.)
        module.output.weight.normal_(std=.08)
        module.output.bias.normal_(std=.04)
    reference = copy.deepcopy(module)
    model = HSTU(HSTUConfig(num_items=12, num_behaviors=2, hidden_size=6,
        num_layers=1, num_heads=2, max_seq_len=5, input_dropout=0., attn_dropout=0.)).to(dtype).eval()
    model.requires_grad_(False)
    q = torch.randn(2, 2, 3, 3, dtype=dtype)*.7
    k, v, items = [torch.randn(2, 5, 6, dtype=dtype) for _ in range(3)]
    counts = torch.tensor([5, 3], dtype=dtype)
    valid = torch.arange(5)[None] < counts[:, None]
    native = history_read(model.blocks[0].attn, q, k, v, count=valid)
    mapped_k, mapped_v = reference.map_tokens(k, v, items)
    expected = history_read(model.blocks[0].attn, q, mapped_k, mapped_v, count=valid)
    actual = compact_item_read(module, model.blocks[0].attn, q, k, v, items, valid)
    tolerance = 1e-11 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    target = torch.randn_like(actual)
    for prediction in (actual, expected):
        rate = (prediction-native)/counts[:, None, None, None]
        aggregate_loss(rate, target, counts, torch.tensor(2., dtype=dtype), 5).backward()
    for (name, parameter), (other_name, other) in zip(module.named_parameters(), reference.named_parameters(), strict=True):
        assert name == other_name and parameter.grad is not None and parameter.grad.abs().sum() > 0
        torch.testing.assert_close(parameter.grad, other.grad, atol=tolerance, rtol=tolerance)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(value.grad is None for value in (q, k, v, items))
    # A changed decoder must be folded live on the next prediction.
    with torch.no_grad():
        module.output.weight.add_(.01)
        reference.output.weight.add_(.01)
    k2, v2 = reference.map_tokens(k, v, items)
    torch.testing.assert_close(compact_item_read(module, model.blocks[0].attn, q, k, v, items, valid),
        history_read(model.blocks[0].attn, q, k2, v2, count=valid), atol=tolerance, rtol=tolerance)
    # A batch split repeats its decoder fold; token/read work stays linear.
    joined = compact_item_fit_flops(module, batch=2, length=5, queries=3)
    split = compact_item_fit_flops(module, batch=1, length=5, queries=3)
    fold = module.output.weight.numel()+module.output.bias.numel()
    assert all(2*split[direction]-joined[direction] == fold for direction in ("forward", "backward"))
