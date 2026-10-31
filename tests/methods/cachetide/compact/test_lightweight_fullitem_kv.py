"""Nonzero factored KV, its gradients, and zero-start publication semantics."""
from types import SimpleNamespace

import torch

from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem import LightweightFullItemAdapter
from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.methods.cachetide.refinement.joint_kv_refinement import JointKVRefinement, fit_key_scales
from hstu_kvcache.models import HSTUKVCache


def attention():
    return SimpleNamespace(position_bias=None, training=False, block_variant="legacy",
        cfg=SimpleNamespace(activation="elu_plus1"), scale=.5,
        _activate=lambda x: torch.nn.functional.elu(x)+1, attn_dropout=lambda x: x)


def test_zero_key_conversion_preserves_rng_tensors_codes_and_read():
    torch.manual_seed(950)
    value=LightweightFullItemAdapter(2,2,4,rank=2).double()
    with torch.no_grad():
        for layer in value.layers: layer.decoder.normal_(std=.2)
    before=torch.get_rng_state().clone()
    kv=LightweightFullItemKVAdapter.from_value(value)
    assert torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(x,kv.state_dict()[name]) for name,x in value.state_dict().items())
    cache=HSTUKVCache(torch.randn(2,1,5,8,dtype=torch.float64),
        torch.randn(2,1,5,8,dtype=torch.float64),5)
    item=torch.randn(1,5,8,dtype=torch.float64)
    q=torch.randn(1,2,3,4,dtype=torch.float64)
    old,new=value.encode_cache(cache,item),kv.encode_cache(cache,item)
    assert torch.equal(old.codes,new.codes)
    assert old.storage_bytes()==new.storage_bytes()
    for i in range(2):
        torch.testing.assert_close(value.make_history_read(old)(i,attention(),q,cache.k[i],cache.v[i]),
            kv.make_history_read(new)(i,attention(),q,cache.k[i],cache.v[i]),atol=0,rtol=0)


def test_nonzero_factored_key_value_matches_dense_read_and_all_gradients():
    torch.manual_seed(951)
    kv=LightweightFullItemKVAdapter.from_value(LightweightFullItemAdapter(1,2,4,rank=2).double())
    layer=kv.layers[0]
    with torch.no_grad():
        layer.key_decoder.normal_(std=.15); layer.decoder.normal_(std=.2)
        layer.input_center.normal_(std=.1);layer.input_scale.uniform_(.5,2.)
    k,v,item=(torch.randn(2,5,8,dtype=torch.float64) for _ in range(3))
    q=torch.randn(2,2,3,4,dtype=torch.float64,requires_grad=True)
    oldk,oldv=k.clone(),v.clone()
    z=layer.encode(k,v,item)
    actual=layer.read(attention(),q,k,v,z)
    zh=z.transpose(1,2)
    kh=k.reshape(2,5,2,4).transpose(1,2)+zh@layer.key_decoder[None]
    vh=v.reshape(2,5,2,4).transpose(1,2)+zh@layer.decoder[None]
    expected=attention()._activate(q@kh.transpose(-2,-1)*.5)@vh
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-12)
    probe=torch.randn_like(actual)
    parameters=(*layer.parameters(),q)
    left=torch.autograd.grad((actual*probe).sum(),parameters,retain_graph=True)
    right=torch.autograd.grad((expected*probe).sum(),parameters)
    for x,y in zip(left,right,strict=True):
        assert x.norm()>0 and torch.isfinite(x).all()
        torch.testing.assert_close(x,y,atol=2e-11,rtol=2e-11)
    assert torch.equal(k,oldk) and torch.equal(v,oldv)


def test_joint_key_fit_only_statistics_and_nonzero_fold():
    torch.manual_seed(952)
    value=LightweightFullItemAdapter(2,2,4,rank=2).double()
    with torch.no_grad():
        for layer in value.layers:layer.decoder.normal_(std=.2)
    kv=LightweightFullItemKVAdapter.from_value(value).requires_grad_(False)
    rows={}
    for uid in range(3):
        cache=HSTUKVCache(torch.randn(2,1,5,8,dtype=torch.float64),
            torch.randn(2,1,5,8,dtype=torch.float64),5)
        item=torch.randn(1,5,8,dtype=torch.float64)
        rows[uid]=dict(cache=cache,state=kv.encode_cache(cache,item),item=item)
    scales,stats=fit_key_scales(kv,rows,[0,1])
    rows[2]["cache"].k.mul_(1000);rows[2]["state"].codes.mul_(1000)
    assert torch.equal(scales,fit_key_scales(kv,rows,[0,1])[0])
    assert stats["fit_tokens"]==10 and stats["fit_users"]==2
    refiner=JointKVRefinement(kv,[[1.,1.],[1.,1.]],scales)
    with torch.no_grad():
        for p in refiner.parameters():p.normal_(std=.02)
    cache,item=rows[0]["cache"],rows[0]["item"]
    q=torch.randn(1,2,3,4,dtype=torch.float64)
    state=refiner.encode_cache(kv,cache,item)
    reads=[refiner.make_history_read(kv,state)(i,attention(),q,cache.k[i],cache.v[i]) for i in range(2)]
    sum(x.square().mean() for x in reads).backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in refiner.parameters())
    refiner.fold_into(kv)
    published=kv.encode_cache(cache,item)
    assert torch.equal(state.codes,published.codes)
    for i,wanted in enumerate(reads):
        torch.testing.assert_close(kv.make_history_read(published)(i,attention(),q,cache.k[i],cache.v[i]),wanted,atol=0,rtol=0)
    full=LightweightFullItemKVAdapter(6,6,32,rank=8)
    assert sum(p.numel() for p in full.parameters())==92448
    assert full.estimate_flops()["token_transform"]==166464
    base=LightweightFullItemAdapter(6,6,32,rank=8)
    assert full.estimate_flops(tokens=100)["extra_history_read"]-base.estimate_flops(tokens=100)["extra_history_read"]==6*(2*6*32*8+(2*6*8+6)*100)
