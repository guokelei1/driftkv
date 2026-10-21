"""Semantic checks for the optimized current-reader path, not training tests."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from implementation_2026_10.reader import fused_relay_read,reader_work_model
from query_relay_2026_10.adapter import UnifiedRelayAdapter


def attention():
    return SimpleNamespace(position_bias=None,block_variant='legacy',
        cfg=SimpleNamespace(activation='elu_plus1'),training=False,
        attn_dropout=torch.nn.Identity(),scale=32**-.5,
        _activate=lambda s:torch.nn.functional.elu(s)+1)


def layer(device, heads=6):
    torch.manual_seed(17)
    result=UnifiedRelayAdapter(1,heads=heads,mode='A',read_shared=3,read_private=3,
        gate_normalization='rms',gate_source='item').to(device).layers[0]
    for p in (result.key_decoder,result.decoder,result.key_gate_weight,result.value_gate_weight):
        torch.nn.init.normal_(p,std=.1)
    return result


def test_model_separates_fusion_from_flop_reduction():
    x=reader_work_model(batch=1,queries=16,history=1024,code_count=512)
    assert x['tensor_bytes']['fused_history_sized_temporary']==0
    assert x['tensor_bytes']['eager_score_or_weight_each']==6*16*1024*4
    assert x['logical_matmul_flops']['correction']>0
    assert x['requested_history_read_bytes']['code']==6*512*6*4


@torch.no_grad()
def test_cpu_requires_explicit_torch_backend():
    l=layer('cpu');q=torch.randn(1,6,2,32);k=torch.randn(1,3,192);z=torch.randn(1,3,6,8)
    with pytest.raises(ValueError,match='requires CUDA'):
        fused_relay_read(l,attention(),q,k,k,z,gate_q=q)
    out,stats=fused_relay_read(l,attention(),q,k,k,z,gate_q=q,backend='torch',return_stats=True)
    assert out.shape==q.shape and stats['backend']=='torch' and stats['fallback'] is False


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires an explicitly allocated GPU')
@pytest.mark.parametrize('heads',[6,10])
@pytest.mark.parametrize('b,n,q,c,mask_kind',[
    (1,0,16,0,'none'),(1,1,1,1,'none'),(1,67,17,35,'causal'),
    (1,1024,32,1024,'none'),(1,99,1,0,'none'),(2,19,7,17,'empty'),
    (2,127,3,1,'numeric'),(1,513,16,259,'fifo'),
    (1,1024,2,1024,'none'),(1,65,4,1,'causal'),(1,64,5,64,'none')])
@torch.no_grad()
def test_fused_native_and_code_semantics(b,n,q,c,mask_kind,heads):
    l=layer('cuda',heads);torch.manual_seed(97+n+q)
    query=torch.randn(b,q,heads,32,device='cuda').transpose(1,2)
    keys=torch.randn(b,n+5,heads*32,device='cuda')[:,5:]
    values=torch.randn_like(keys)
    codes=torch.randn(b,c,*l.code_shape,device='cuda')
    gate_query=torch.randn_like(query)
    mask=None
    if mask_kind in ('causal','fifo'):
        pos=torch.arange(q,device='cuda')+max(0,n-q)
        col=torch.arange(n,device='cuda')
        mask=col[None,:]<pos[:,None]
        if mask_kind=='fifo':mask&=col[None,:]>pos[:,None]-71
    elif mask_kind=='empty':mask=torch.zeros(b,q,n,device='cuda',dtype=torch.bool)
    elif mask_kind=='numeric':mask=torch.rand(b,heads,q,n,device='cuda')
    expected=fused_relay_read(l,attention(),query,keys,values,codes,
        valid_mask=mask,code_count=c,gate_q=gate_query,backend='torch')
    actual=fused_relay_read(l,attention(),query,keys,values,codes,
        valid_mask=mask,code_count=c,gate_q=gate_query)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    if c:
        gates=l.query_gates(gate_query)
        prepared=fused_relay_read(l,attention(),query,keys,values,codes,
            valid_mask=mask,code_count=c,prepared_gates=gates)
        torch.testing.assert_close(prepared,actual,atol=0,rtol=0)
