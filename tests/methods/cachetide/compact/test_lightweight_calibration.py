"""Cached local fitting must implement the deployed Value correction."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


from experiments.archive.compact_read.calibrate_lightweight import cached_prediction, feature_statistics
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueLayer


def test_cached_training_and_folded_decoder_match_deployed_value_read():
    torch.manual_seed(3)
    b, n, h, d, rank, q = 2, 7, 2, 3, 2, 4
    layer = LightweightValueLayer(h, d, rank).double()
    with torch.no_grad():
        layer.decoder.normal_(0, .2)
        layer.input_center.normal_()
        layer.input_scale.uniform_(.5, 2)
    k, v, item = (torch.randn(b, n, h*d, dtype=torch.float64) for _ in range(3))
    query = torch.randn(b, h, q, d, dtype=torch.float64)
    attention = SimpleNamespace(position_bias=None, training=False, block_variant="legacy",
        cfg=SimpleNamespace(activation="elu_plus1"), scale=d**-.5,
        _activate=lambda value: torch.nn.functional.elu(value)+1,
        attn_dropout=lambda value:value)
    weights = attention._activate(query@k.reshape(b,n,h,d).transpose(1,2).transpose(-2,-1)*attention.scale)
    native = weights@v.reshape(b,n,h,d).transpose(1,2)
    features = torch.cat([x.reshape(b,n,h,d) for x in (k,v,item)], -1)
    normalized = (features-layer.input_center)/layer.input_scale
    predicted = cached_prediction(layer, normalized, weights/n)*n
    direct = layer.read(attention, query, k, v, layer.encode(k,v,item))-native
    torch.testing.assert_close(predicted, direct, rtol=1e-10, atol=1e-10)
    probe = torch.randn_like(predicted)
    grads_cached = torch.autograd.grad((predicted*probe).sum(), tuple(layer.parameters()), retain_graph=True)
    grads_direct = torch.autograd.grad((direct*probe).sum(), tuple(layer.parameters()))
    for a,bgrad in zip(grads_cached, grads_direct):
        torch.testing.assert_close(a,bgrad,rtol=1e-9,atol=1e-9)
    scales = torch.tensor([.03, .5], dtype=torch.float64)
    target_in_physical_units = predicted.detach()*scales[None,:,None,None]
    with torch.no_grad():
        layer.decoder.mul_(scales[:,None,None])
    folded = layer.read(attention, query, k, v, layer.encode(k,v,item))-native
    torch.testing.assert_close(folded,target_in_physical_units,rtol=1e-10,atol=1e-10)


def test_feature_moments_exclude_padding():
    features = torch.tensor([[[[1.,2.]],[[3.,6.]],[[999.,999.]]],
                             [[[5.,4.]],[[999.,999.]],[[999.,999.]]]])
    center, scale = feature_statistics(features, torch.tensor([2,1]))
    actual = torch.tensor([[1.,2.],[3.,6.],[5.,4.]])
    torch.testing.assert_close(center[0],actual.mean(0))
    torch.testing.assert_close(scale[0],actual.std(0,correction=0))
