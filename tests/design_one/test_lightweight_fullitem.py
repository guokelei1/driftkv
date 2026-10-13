"""Full item features must expand information without changing the read path."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))

from design_system.calibrate_lightweight import cached_prediction
from hstu_kvcache.design_one.lightweight import LightweightValueAdapter
from hstu_kvcache.design_one.lightweight_fullitem import LightweightFullItemAdapter
from hstu_kvcache.design_two.decoder_refinement import DecoderRefinement
from hstu_kvcache.models import HSTUKVCache


def attention(d):
    return SimpleNamespace(position_bias=None,training=False,block_variant="legacy",
        cfg=SimpleNamespace(activation="elu_plus1"),scale=d**-.5,
        _activate=lambda x:torch.nn.functional.elu(x)+1,attn_dropout=lambda x:x)


def test_fullitem_contains_original_head_local_encoder_and_same_read():
    torch.manual_seed(801)
    h,d,r,b,n=2,4,2,2,5
    local=LightweightValueAdapter(1,h,d,rank=r,response_hidden_width=0).double()
    full=LightweightFullItemAdapter(1,h,d,rank=r).double()
    with torch.no_grad():
        old,new=local.layers[0],full.layers[0]
        old.input_center.normal_(std=.1)
        old.input_scale.uniform_(.5,2)
        old.decoder.normal_(std=.2)
        new.input_weight.zero_()
        new.decoder.copy_(old.decoder)
        new.input_bias.copy_(old.input_bias)
        for head in range(h):
            chosen=list(range(2*d))+list(range(2*d+head*d,2*d+(head+1)*d))
            new.input_weight[head,chosen]=old.input_weight[head]
            new.input_center[head,chosen]=old.input_center[head]
            new.input_scale[head,chosen]=old.input_scale[head]
    cache=HSTUKVCache(torch.randn(1,b,n,h*d,dtype=torch.float64),
                      torch.randn(1,b,n,h*d,dtype=torch.float64),n)
    item=torch.randn(b,n,h*d,dtype=torch.float64)
    old_state,new_state=local.encode_cache(cache,item),full.encode_cache(cache,item)
    torch.testing.assert_close(new_state.codes,old_state.codes,atol=1e-12,rtol=1e-12)
    assert new_state.storage_bytes()==old_state.storage_bytes()
    q=torch.randn(b,h,3,d,dtype=torch.float64)
    a=attention(d)
    old_read=local.make_history_read(old_state)(0,a,q,cache.k[0],cache.v[0])
    new_read=full.make_history_read(new_state)(0,a,q,cache.k[0],cache.v[0])
    torch.testing.assert_close(new_read,old_read,atol=1e-12,rtol=1e-12)
    # A head can now use an item coordinate outside its former raw slice.
    with torch.no_grad():
        new.input_weight[0,2*d+d,0]=.25
    changed=item.clone()
    changed[:,:,d]+=1
    torch.testing.assert_close(local.encode_cache(cache,changed).codes[0,:,:,0],old_state.codes[0,:,:,0])
    assert not torch.allclose(full.encode_cache(cache,changed).codes[0,:,:,0],
                              full.encode_cache(cache,item).codes[0,:,:,0])


def test_cached_fullitem_gradient_scale_and_decoder_fold_match_deployment():
    torch.manual_seed(802)
    b,n,h,d,r,q=2,5,2,4,2,3
    adapter=LightweightFullItemAdapter(1,h,d,rank=r).double()
    layer=adapter.layers[0]
    with torch.no_grad():
        layer.decoder.normal_(std=.2)
        layer.input_center.normal_(std=.1)
        layer.input_scale.uniform_(.5,2)
    k,v,item=(torch.randn(b,n,h*d,dtype=torch.float64) for _ in range(3))
    query=torch.randn(b,h,q,d,dtype=torch.float64)
    a=attention(d)
    weights=a._activate(query@k.reshape(b,n,h,d).transpose(1,2).transpose(-2,-1)*a.scale)
    features=torch.cat((k.reshape(b,n,h,d),v.reshape(b,n,h,d),
        item[:,:,None,:].expand(b,n,h,h*d)),-1)
    normalized=(features-layer.input_center)/layer.input_scale
    cached=cached_prediction(layer,normalized,weights/n)*n
    codes=layer.encode(k,v,item)
    native=weights@v.reshape(b,n,h,d).transpose(1,2)
    direct=layer.read(a,query,k,v,codes)-native
    torch.testing.assert_close(cached,direct,atol=1e-11,rtol=1e-11)
    probe=torch.randn_like(cached)
    left=torch.autograd.grad((cached*probe).sum(),tuple(layer.parameters()),retain_graph=True)
    right=torch.autograd.grad((direct*probe).sum(),tuple(layer.parameters()))
    for x,y in zip(left,right,strict=True):
        torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)
    # The same decoder-only offline module accepts full-item codes unchanged.
    adapter.requires_grad_(False)
    cache=HSTUKVCache(k[None],v[None],n)
    state=adapter.encode_cache(cache,item)
    refiner=DecoderRefinement(adapter,[[.2,.3]])
    with torch.no_grad():
        refiner.delta.normal_(std=.1)
    before=refiner.make_history_read(adapter,state)(0,a,query,k,v)
    refiner.fold_into(adapter)
    after=adapter.make_history_read(state)(0,a,query,k,v)
    torch.testing.assert_close(before,after,atol=0,rtol=0)
