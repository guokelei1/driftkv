"""Operation-wide gate preparation preserves candidate-local frozen gates."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]

from implementation_2026_10.backend import ImplementationConfig, InferenceAdapter
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.models import PointwiseAttention, PointwiseAttentionConfig


def adapter():
    torch.manual_seed(20261015)
    obj = UnifiedRelayAdapter(3, mode='A', xi_width=1, read_shared=3,
        read_private=3, gate_normalization='rms', gate_source='item')
    with torch.no_grad():
        for layer in obj.layers:
            layer.key_gate_weight.normal_(std=.05)
            layer.value_gate_weight.normal_(std=.05)
            layer.key_decoder.normal_(std=.03)
            layer.decoder.normal_(std=.03)
    return obj.eval().requires_grad_(False)


@torch.no_grad()
def test_packed_gates_match_per_layer_and_do_not_mix_candidates():
    original = adapter()
    configured = InferenceAdapter(original, ImplementationConfig(reader_backend='torch'))
    item = torch.randn(2,6,5,32)
    gates = configured.prepare_gates(item)
    for layer,(key,value) in zip(original.layers,gates,strict=True):
        wanted_key,wanted_value = layer.query_gates(item)
        torch.testing.assert_close(key,wanted_key,rtol=2e-6,atol=2e-7)
        torch.testing.assert_close(value,wanted_value,rtol=2e-6,atol=2e-7)
    changed = item.clone()
    changed[:,:,4].normal_()
    after = configured.prepare_gates(changed)
    for old,new in zip(gates,after,strict=True):
        for a,b in zip(old,new,strict=True):
            torch.testing.assert_close(a[:,:,:4],b[:,:,:4],rtol=0,atol=0)
    assert configured.describe()['extra_persistent_row_bytes'] == 0


@torch.no_grad()
@pytest.mark.parametrize('code_count',[0,3,7])
def test_configured_torch_callback_matches_original(code_count):
    original = adapter()
    configured = InferenceAdapter(original,ImplementationConfig(reader_backend='torch'))
    attention = PointwiseAttention(PointwiseAttentionConfig(192,6,head_dim=32)).eval()
    q,item = torch.randn(1,6,4,32)*.1,torch.randn(1,6,4,32)
    k,v = torch.randn(1,7,192)*.1,torch.randn(1,7,192)*.1
    state = LightweightValueState(torch.randn(3,1,7,6,8)*.1,7)
    mask = torch.arange(7)[None] < torch.tensor([1,3,5,7])[:,None]
    reference = original.make_history_read(state,valid_mask=mask,code_count=code_count,gate_query=item)
    current = configured.make_history_read(state,valid_mask=mask,code_count=code_count,gate_query=item)
    for layer in range(3):
        torch.testing.assert_close(current(layer,attention,q,k,v),reference(layer,attention,q,k,v),
                                   atol=3e-6,rtol=2e-5)
    assert configured.counters['packed_gate_operations'] == int(code_count>0)
    assert configured.counters['torch_history_calls'] == 3


def test_inference_configuration_requires_frozen_geometry_and_explicit_backend():
    with pytest.raises(ValueError,match='explicitly'):
        ImplementationConfig(reader_backend='automatic')
    with pytest.raises(ValueError,match='positive'):
        ImplementationConfig(row_chunk=0)
    original=adapter()
    original.train()
    with pytest.raises(ValueError,match='freeze and eval'):
        InferenceAdapter(original)
    original.eval()
    configured=InferenceAdapter(original)
    with pytest.raises(ValueError,match='no_grad'):
        configured.prepare_gates(torch.randn(1,6,1,32))
