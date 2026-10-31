"""Local dynamic attention must preserve masks, gradients and fitted Key origins."""
from collections import defaultdict
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from experiments.archive.compact_read.local_kv_fit import cached_prediction, fit_layer
from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem import LightweightFullItemAdapter
from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.methods.cachetide.refinement.joint_local_kv_refinement import JointLocalKVRefinement
from hstu_kvcache.models import HSTUKVCache


def data_batch(batch=3):
    b,n,h,d,q= batch,5,2,4,3
    k,v,item=(torch.randn(b,n,h*d,dtype=torch.float64) for _ in range(3))
    query=torch.randn(b,h,q,d,dtype=torch.float64)
    lengths=torch.tensor([3,5,4][:b])
    mask=torch.arange(n)[None]<lengths[:,None]
    kh,vh=k.reshape(b,n,h,d).transpose(1,2),v.reshape(b,n,h,d).transpose(1,2)
    raw=query@kh.transpose(-2,-1)*.5
    a=(torch.nn.functional.elu(raw)+1)*mask[:,None,None,:]/lengths[:,None,None,None]
    teacher=(torch.nn.functional.elu(query@(kh+.1).transpose(-2,-1)*.5)+1)
    teacher=teacher*mask[:,None,None,:]/lengths[:,None,None,None]
    return dict(features=torch.cat((k.reshape(b,n,h,d),v.reshape(b,n,h,d),
        item[:,:,None,:].expand(b,n,h,h*d)),-1),q=query,raw_scores=raw,v=vh,
        lengths=lengths,native_rate=a@vh,teacher_rate=teacher@(vh+.1),attention_scale=.5)


def test_local_nonzero_key_value_dense_gradient_and_padding_equivalence():
    torch.manual_seed(960)
    adapter=LightweightFullItemKVAdapter.from_value(LightweightFullItemAdapter(1,2,4,rank=2).double())
    layer=adapter.layers[0]
    with torch.no_grad():layer.key_decoder.normal_(std=.1);layer.decoder.normal_(std=.2)
    data=data_batch(2)
    outscale=torch.tensor([.6,1.3],dtype=torch.float64)
    keyscale=torch.tensor([.7,1.2],dtype=torch.float64)[:,None,None]
    def predict(x):
        return cached_prediction(layer,x["features"],x["q"],x["raw_scores"],x["v"],
            x["lengths"],x["native_rate"],outscale,keyscale,.5)
    actual=predict(data)
    z=torch.nn.functional.silu(data["features"].transpose(1,2)@layer.input_weight[None]+layer.input_bias[None,:,None])
    keys=data["features"][...,:4].transpose(1,2)+z@(layer.key_decoder*keyscale)[None]
    values=data["v"]+z@(layer.decoder*outscale[:,None,None])[None]
    mask=torch.arange(5)[None]<data["lengths"][:,None]
    a=(torch.nn.functional.elu(data["q"]@keys.transpose(-2,-1)*.5)+1)*mask[:,None,None,:]/data["lengths"][:,None,None,None]
    expected=(a@values-data["native_rate"])/outscale[None,:,None,None]
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-12)
    probe=torch.randn_like(actual)
    left=torch.autograd.grad((actual*probe).sum(),tuple(layer.parameters()),retain_graph=True)
    right=torch.autograd.grad((expected*probe).sum(),tuple(layer.parameters()))
    for x,y in zip(left,right,strict=True):
        assert x.norm()>0
        torch.testing.assert_close(x,y,atol=2e-11,rtol=2e-11)
    changed={key:value.clone() if isinstance(value,torch.Tensor) else value for key,value in data.items()}
    changed["raw_scores"][0,:,:,3:]=10000
    changed["features"][0,3:]=1000
    changed["v"][0,:,3:]=1000
    torch.testing.assert_close(predict(changed),actual,atol=0,rtol=0)


def test_zero_decoders_then_encoder_gradient_and_local_fold():
    torch.manual_seed(961)
    adapter=LightweightFullItemKVAdapter.from_value(LightweightFullItemAdapter(1,2,4,rank=2).double())
    layer=adapter.layers[0]
    costs=defaultdict(int)
    record=fit_layer(layer,data_batch(),2,epochs=2,batch_size=2,checks=[0,1,2],costs=costs)
    first,second=record["first_two_gradient_norms"]["1"],record["first_two_gradient_norms"]["2"]
    assert first["input_weight"]==0 and first["input_bias"]==0
    assert first["key_decoder"]>0 and first["decoder"]>0
    assert second["input_weight"]>0 and second["input_bias"]>0
    assert sum(costs.values())>0 and costs["initial_fit_encoder_flops"]>0
    assert torch.isfinite(layer.key_decoder).all() and torch.isfinite(layer.decoder).all()


def test_joint_zero_coordinates_preserve_nonzero_local_key_and_fold():
    torch.manual_seed(962)
    adapter=LightweightFullItemKVAdapter.from_value(LightweightFullItemAdapter(2,2,4,rank=2).double())
    with torch.no_grad():
        for layer in adapter.layers:layer.key_decoder.normal_(std=.2);layer.decoder.normal_(std=.3)
    adapter.requires_grad_(False)
    refiner=JointLocalKVRefinement(adapter,[[1.,1.],[1.,1.]],[[2.,2.],[2.,2.]])
    assert torch.equal(refiner.key_decoders(),torch.stack([layer.key_decoder for layer in adapter.layers]))
    with torch.no_grad():
        for p in refiner.parameters():p.normal_(std=.01)
    cache=HSTUKVCache(torch.randn(2,1,5,8,dtype=torch.float64),torch.randn(2,1,5,8,dtype=torch.float64),5)
    item=torch.randn(1,5,8,dtype=torch.float64);q=torch.randn(1,2,3,4,dtype=torch.float64)
    a=SimpleNamespace(position_bias=None,training=False,block_variant="legacy",cfg=SimpleNamespace(activation="elu_plus1"),
        scale=.5,_activate=lambda x:torch.nn.functional.elu(x)+1,attn_dropout=lambda x:x)
    state=refiner.encode_cache(adapter,cache,item)
    before=[refiner.make_history_read(adapter,state)(i,a,q,cache.k[i],cache.v[i]) for i in range(2)]
    sum(x.square().mean() for x in before).backward()
    assert all(p.grad is not None and p.grad.norm()>0 for p in refiner.parameters())
    refiner.fold_into(adapter);published=adapter.encode_cache(cache,item)
    assert torch.equal(state.codes,published.codes)
    for i,wanted in enumerate(before):
        torch.testing.assert_close(adapter.make_history_read(published)(i,a,q,cache.k[i],cache.v[i]),wanted,atol=0,rtol=0)
