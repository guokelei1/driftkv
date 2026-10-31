"""Snapshot lineage endpoints, actual branch inputs, and fitted map roundtrip."""
from pathlib import Path
import copy
import sys

import numpy as np
import torch

from experiments.ch3_motivation.code.state_lineage_probe import (
    RowResidualMap, branch_inputs, fit_map, load_bank, replay_rows, replay_caches, row_indices,
)
from experiments.ch3_motivation.code.simple_kv_maps import head_rows, rows_to_cache
from hstu_kvcache.models import HSTUKVCache


def test_endpoints_and_anchor_reads_same_chain_previous():
    assert np.array_equal(row_indices(1024,128)[[0,-1]],[0,1023])
    assert len(np.unique(row_indices(9,128)))==9
    original=torch.arange(24,dtype=torch.float32).reshape(2,3,4);previous=original+10
    for kind,expected in [('chain',(previous,previous)),('direct',(original,original)),('anchored',(original,previous))]:
        a,b=branch_inputs(original,previous,kind);assert a is expected[0] and b is expected[1]
        model=RowResidualMap(2,4,8)
        torch.testing.assert_close(model(a,b),b,atol=0,rtol=0)
    bank={2:{'chain':lambda a,b:b+1,'direct':lambda a,b:a+50,'anchored':lambda a,b:a+100},
          3:{'chain':lambda a,b:2*b,'direct':lambda a,b:5*a,'anchored':lambda a,b:a+3*b}}
    outputs=replay_rows(original,bank,3)
    torch.testing.assert_close(outputs['chain'],2*(original+1))
    torch.testing.assert_close(outputs['anchored'],original+3*(original+1))
    torch.testing.assert_close(outputs['direct'],5*original)
    bank[4]={'chain':lambda a,b:3*b,'direct':lambda a,b:7*a,'anchored':lambda a,b:a+4*b}
    bank['short']={'chain':lambda a,b:5*b,'anchored':lambda a,b:a+6*b}
    final=replay_rows(original,bank,4)
    torch.testing.assert_close(final['chain'],6*(original+1))
    torch.testing.assert_close(final['short_chain'],5*(original+1))
    torch.testing.assert_close(final['short_anchored'],original+6*(original+1))


def test_fitted_head_map_checkpoint_and_full_cache_composition(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(119)
    a,b=torch.randn(2,48,4),torch.randn(2,48,4);target=b+.3*a
    model,report=fit_map(a[:,:32],b[:,:32],target[:,:32],a[:,32:],b[:,32:],target[:,32:],
        hidden=8,steps=32,batch_rows=16,lr=.02,seed=17,validate_every=8)
    assert report['selected_step']>0
    assert report['validation_kv_mse']<report['trace'][0]['validation_kv_mse']
    path=tmp_path/'maps.pt'
    torch.save(dict(format='state_lineage_maps_v1',pairs=2,width=4,hidden=8,
        maps={2:{kind:model.state_dict() for kind in ('chain','direct','anchored')}}),path)
    bank,_=load_bank(path,'cpu')
    for kind in bank[2]:torch.testing.assert_close(bank[2][kind](a,b),model(a,b),atol=0,rtol=0)
    cache=HSTUKVCache(torch.randn(1,1,6,4),torch.randn(1,1,6,4),6)
    original_k=cache.k.clone();original_v=cache.v.clone()
    result=replay_caches(cache,bank,2,heads=2)
    for kind,value in result.items():
        rows=head_rows(cache,2);expected=rows_to_cache(model(rows,rows),cache,2)
        torch.testing.assert_close(value.k,expected.k);torch.testing.assert_close(value.v,expected.v)
    torch.testing.assert_close(cache.k,original_k,atol=0,rtol=0)
    torch.testing.assert_close(cache.v,original_v,atol=0,rtol=0)


@torch.no_grad()
def test_fixed_snapshot_scoring_preserves_real_gaps_and_native_endpoints(monkeypatch):
    from experiments.ch3_motivation.code.score_state_lineage import score_user
    from hstu_kvcache.models import HSTU,HSTUConfig
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch');torch.set_num_threads(1);torch.manual_seed(913)
    old=HSTU(HSTUConfig(num_items=100,num_behaviors=2,hidden_size=8,num_heads=2,
        num_layers=2,max_seq_len=1024,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    current=copy.deepcopy(old)
    for p in current.parameters():p.add_(torch.randn_like(p)*.02)
    bank={}
    for version in (2,3,4):
        bank[version]={kind:RowResidualMap(4,8,8).eval() for kind in ('chain','direct','anchored')}
        for m in bank[version].values():m.w2.normal_(std=.01)
    bank['short']={kind:RowResidualMap(4,8,8).eval() for kind in ('chain','anchored')}
    for m in bank['short'].values():m.w2.normal_(std=.01)
    times=np.array([86370,86380,86390,86400,86401,86403])
    events=times,np.arange(1,7),np.arange(6)%2
    requests={86400:[dict(item_idx=40,label=0)],86402:[dict(item_idx=41,label=1)],
        86404:[dict(item_idx=42,label=0),dict(item_idx=43,label=1)]}
    for target,expected_methods in ((3,4),(4,4)):
        rows,stats=score_user(old,current,bank,events,requests,16,target,86400,True)
        assert len(stats['method_names'])==expected_methods and len(rows)==4*expected_methods
        assert stats['max_cache_abs_error']==0 and stats['max_score_abs_error']<1e-6
        for row in rows:
            assert row['appended_events']==0 and row['history_rows']==3
            assert row['query_gap']=={86400:10,86402:1,86404:1}[row['query_timestamp']]
            if row['method']=='rebuild':assert row['kv_mse']==0
        assert stats['initial_mse']['reuse']['kv_mse']>0
