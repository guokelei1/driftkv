"""All heads/fields stay separate; streamed Gram agrees with direct SVD."""
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.kv_delta_spectrum import (
    Moments, head_matrix, spectrum, batch_inputs, native_batches, prefill, direct_svd_check,
    native_spectral_check, NAMES,
)
from hstu_kvcache.models import HSTU,HSTUConfig,HSTUKVCache


def test_streamed_fields_heads_and_centered_spectrum_match_svd():
    torch.manual_seed(622);torch.set_num_threads(1)
    a=HSTUKVCache(torch.randn(2,3,15,12),torch.randn(2,3,15,12),15)
    b=HSTUKVCache(a.k*1.2+torch.randn_like(a.k)*.04,a.v*.7+torch.randn_like(a.v)*.11,15)
    before=a.k.clone(),a.v.clone();mom=Moments(6,4,'cpu');mom.add(a,b,heads=3,chunk_rows=7)
    assert mom.rows==45 and mom.gram.dtype==torch.float64
    expected={}
    for field in ('k','v'):
        old,new=head_matrix(getattr(a,field),3).double(),head_matrix(getattr(b,field),3).double()
        expected['source_'+field.upper()]=old;expected['rebuilt_'+field.upper()]=new;expected['delta_'+field.upper()]=new-old
    for index,name in enumerate(NAMES):
        x=expected[name]
        torch.testing.assert_close(mom.gram[index],x.transpose(1,2).bmm(x),atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(mom.total[index],x.sum(1),atol=1e-12,rtol=1e-12)
        for centered in (False,True):
            matrix=x-x.mean(1,keepdim=True) if centered else x
            result=spectrum(mom.gram[index],mom.total[index],mom.rows,centered)
            torch.testing.assert_close(result['eigenvalues'],torch.linalg.svdvals(matrix).square(),atol=1e-10,rtol=1e-10)
            assert torch.equal(result['valid'],torch.ones(6,dtype=torch.bool))
            torch.testing.assert_close(result['curve'][:,-1],torch.ones(6,dtype=torch.float64))
    assert torch.equal(a.k,before[0]) and torch.equal(a.v,before[1])
    assert all(r['passed'] for r in direct_svd_check(a,b,3))


def test_zero_delta_is_undefined_not_claimed_low_rank():
    a=HSTUKVCache(torch.arange(24,dtype=torch.float32).reshape(1,1,3,8),torch.ones(1,1,3,8),3)
    mom=Moments(2,4,'cpu');mom.add(a,a,2,chunk_rows=1)
    for centered in (False,True):
        result=spectrum(mom.gram,mom.total,mom.rows,centered)
        assert not bool(result['valid'][4:].any())
        assert torch.isnan(result['curve'][4:]).all()
        assert not bool(result['ranks']['0.9'][4:].any())


def test_spectral_gate_accepts_roundoff_but_rejects_a_distorted_head():
    torch.manual_seed(624);torch.set_num_threads(1)
    old=HSTUKVCache(torch.randn(2,2,20,96),torch.randn(2,2,20,96),20)
    new=HSTUKVCache(old.k*.9+torch.randn_like(old.k)*.3,old.v*.8+torch.randn_like(old.v)*.3,20)
    actual=HSTUKVCache(old.k+torch.randn_like(old.k)*1e-7,old.v+torch.randn_like(old.v)*1e-7,20)
    assert native_spectral_check(actual,new,old,new,3)['passed']
    distorted=HSTUKVCache(actual.k.clone(),actual.v.clone(),20)
    distorted.k[0,:,:,:32].add_(.001)
    with pytest.raises(ValueError,match='per-head relative RMS'):
        native_spectral_check(distorted,new,old,new,3)


def test_exact_length_batch_matches_native_causal_prefix(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch');torch.set_num_threads(1);torch.manual_seed(623)
    model=HSTU(HSTUConfig(num_items=64,num_behaviors=3,hidden_size=8,num_heads=2,
        num_layers=2,max_seq_len=1024,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    rows={1:(np.array([1,4,9,10]),np.array([2,3,4,5]),np.array([1,2,1,2])),
          2:(np.array([2,5,8,10,11]),np.array([6,7,8,9,10]),np.array([1,2,1,2,1])),
          3:(np.array([3,10]),np.array([11,12]),np.array([1,1]))}
    histories=types.SimpleNamespace(rows=rows);groups,batches,meta=native_batches(histories,[1,2,3],10,8)
    assert groups[3]==[1,2] and meta['1']['last_cached_timestamp']==9 and meta['3']['history_rows']==1
    inputs=batch_inputs(histories,[1,2],10,'cpu')
    assert torch.equal(inputs[2],torch.tensor([[0.,3.,5.],[0.,3.,3.]]))
    actual=prefill(model,inputs)
    for i in range(2):
        single=prefill(model,tuple(x[i:i+1] for x in inputs))
        torch.testing.assert_close(actual.k[:,i:i+1],single.k,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(actual.v[:,i:i+1],single.v,atol=1e-6,rtol=1e-5)
