"""Existing causal flags gate both Key/Value corrections in one code."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from design_system.evaluate_lightweight_support_mask import mask_codes
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.design_one.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.design_three.lifecycle import InheritedSupport


def test_clear_codes_give_exact_native_key_and_value_read():
    torch.manual_seed(872)
    adapter = LightweightFullItemKVAdapter(1, 2, 4, rank=2).double()
    with torch.no_grad():
        adapter.layers[0].key_decoder.normal_(); adapter.layers[0].decoder.normal_()
    q = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    k, v = (torch.randn(1, 5, 8, dtype=torch.float64) for _ in range(2))
    state = LightweightValueState(torch.randn(1, 1, 5, 2, 2, dtype=torch.float64), 5)
    attention = SimpleNamespace(position_bias=None, training=False, block_variant='legacy',
        cfg=SimpleNamespace(activation='elu_plus1'), scale=.5,
        _activate=lambda x: F.elu(x) + 1, attn_dropout=lambda x: x)
    original = state.codes.clone()
    mask_codes(state, np.ones((1, 5), dtype=bool))
    assert torch.equal(state.codes, original)
    mask_codes(state, np.zeros((1, 5), dtype=bool))
    actual = adapter.make_history_read(state)(0, attention, q, k, v)
    keys = k.reshape(1, 5, 2, 4).transpose(1, 2)
    values = v.reshape(1, 5, 2, 4).transpose(1, 2)
    expected = (F.elu((q @ keys.transpose(-2, -1)) * .5) + 1) @ values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_append_mask_then_fifo_eviction_equals_full_mask():
    support = InheritedSupport(3, 4, 4, inherited=True)
    old = torch.randn(3, 1, 4, 2, 2)
    arriving = torch.randn(3, 1, 2, 2, 2)
    support.append(2)
    new = mask_codes(LightweightValueState(arriving.clone(), 2), support.flags[:, -2:]).codes
    incremental = torch.cat((old, new), dim=2)[:, :, -4:]
    full = mask_codes(LightweightValueState(torch.cat((old, arriving), dim=2)[:, :, -4:].clone(), 4), support.flags)
    assert torch.equal(incremental, full.codes)
    assert not full.codes[0, :, -2:].any()
    assert torch.equal(full.codes[1:, :, -2:], arriving[1:])
