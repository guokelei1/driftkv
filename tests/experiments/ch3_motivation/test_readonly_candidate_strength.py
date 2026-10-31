"""One causal write lineage, exact inherited endpoints, read-only strengths."""
import copy
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.ch3_motivation.code.run_readonly_candidate_strength import (
    GRID, ReadonlyStrengthTrajectory, replay_user,
)
from hstu_kvcache.models import HSTU,HSTUConfig,HSTUKVCache


def scenario(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch');torch.set_num_threads(1);torch.manual_seed(611)
    old=HSTU(HSTUConfig(num_items=100,num_behaviors=2,hidden_size=8,num_heads=2,
        num_layers=2,max_seq_len=4,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    current=copy.deepcopy(old)
    with torch.no_grad():
        for p in current.parameters():p.add_(torch.randn_like(p)*.03)
    times=np.array([86370,86380,86390,86400,86401,86401,86402,86404,86405])
    events=times,np.arange(1,len(times)+1),np.arange(len(times))%2
    cfg=dict(edge=2,release_day=1,end_day=2,history_length=4,append_band_size=2,query_batch=2)
    calls=[]
    def linear(c):
        calls.append(1)
        return HSTUKVCache(c.k*1.1+.17,c.v*.9-.11,c.seq_len)
    return old,current,linear,events,cfg,calls


@torch.no_grad()
def test_endpoint_fifo_same_new_rows_and_no_writeback(monkeypatch):
    old,current,linear,events,cfg,calls=scenario(monkeypatch)
    state=ReadonlyStrengthTrajectory(old,current,linear,events,cfg,True)
    initial_k,initial_v=state.original.k.clone(),state.original.v.clone()
    for time,expected_rows in ((86400,3),(86401,3),(86402,1),(86403,0),(86406,0)):
        state.advance(time);assert state.inherited==expected_rows
        before=(state.cache.k.clone(),state.cache.v.clone())
        read=state.read_cache();n=state.inherited
        if n:
            for field in ('k','v'):
                value=getattr(read,field);native=getattr(state.cache,field);mapped=getattr(state.mapped,field)
                torch.testing.assert_close(value[:,0:1],native,atol=0,rtol=0)
                torch.testing.assert_close(value[:,4:5,:n],mapped[:,:,state.initial_length-n:],atol=0,rtol=0)
                for index in range(len(GRID)):
                    torch.testing.assert_close(value[:,index:index+1,n:],native[:,:,n:],atol=0,rtol=0)
        else:assert read is state.cache
        scores=state.score(torch.tensor([[41,52]]),torch.tensor([1.]))
        if n==0:assert torch.equal(scores,scores[:1].expand_as(scores))
        torch.testing.assert_close(state.cache.k,before[0],atol=0,rtol=0)
        torch.testing.assert_close(state.cache.v,before[1],atol=0,rtol=0)
    assert calls==[1]
    torch.testing.assert_close(state.original.k,initial_k,atol=0,rtol=0)
    torch.testing.assert_close(state.original.v,initial_v,atol=0,rtol=0)
    assert state.max_cache_abs_error==0 and state.max_score_abs_error<1e-6


@torch.no_grad()
def test_real_rows_keep_labels_ties_gaps_and_mark_flat(monkeypatch):
    old,current,linear,events,cfg,_=scenario(monkeypatch)
    requests={86400:[dict(item_idx=41,label=0)],86402:[dict(item_idx=42,label=1)],
        86403:[dict(item_idx=43,label=0),dict(item_idx=44,label=1)]}
    rows,stats=replay_user(old,current,linear,events,requests,15,cfg,True)
    assert len(rows)==4*9 and stats['persistent_native_write_paths']==1
    assert stats['flat_no_inherited_events']==2
    for row in rows:
        assert row['query_gap']=={86400:10,86402:1,86403:1}[row['query_timestamp']]
        assert row['appended_events']=={86400:0,86402:3,86403:4}[row['query_timestamp']]
        assert row['inherited_rows']=={86400:3,86402:1,86403:0}[row['query_timestamp']]
        assert row['label']==int(row['item_idx'] in (42,44))
