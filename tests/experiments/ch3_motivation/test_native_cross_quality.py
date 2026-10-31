"""Native K/V crosses use their actual two causal trajectories, never learned F."""
import copy
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.ch3_motivation.code.native_cross_quality import (
    NativeCross, METHODS, crossed_cache, replay_user,
)
from experiments.ch3_motivation.code.real_quality_paths import _member
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTU,HSTUConfig,HSTUKVCache


def scenario(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch')
    torch.set_num_threads(1);torch.manual_seed(193)
    old=HSTU(HSTUConfig(num_items=300,num_behaviors=2,hidden_size=8,num_heads=2,
        num_layers=2,max_seq_len=4,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    current=copy.deepcopy(old)
    with torch.no_grad():
        for parameter in current.parameters():parameter.add_(torch.randn_like(parameter)*.05)
    times=np.array([86370,86380,86390,86400,86401,86401,86402,86404,86405])
    events=times,np.arange(1,len(times)+1),np.arange(len(times))%2
    cfg=dict(edge=2,release_day=1,end_day=3,history_length=4,append_band_size=2,query_batch=16)
    return old,current,events,cfg


@torch.no_grad()
def test_full_native_cross_endpoint_scores_and_fifo_do_not_write_back(monkeypatch):
    old,current,events,cfg=scenario(monkeypatch)
    state=NativeCross(old,current,events,cfg,check_independent=True)
    reuse=NativeTrajectory(old,1,events,86400,max_length=4,append_band_size=2)
    rebuild=copy.copy(reuse);rebuild.ledger=copy.copy(reuse.ledger)
    reuse.switch(current,2,86400)
    rebuild.switch(current,2,86400,rebuild=True)
    assert METHODS==('oldKoldV','teacherKoldV','oldKteacherV','teacherKteacherV')
    items=torch.tensor([[31,52,78]])
    for timestamp in (86400,86401,86402,86403,86406):
        state.advance(timestamp);reuse.advance(timestamp);rebuild.advance(timestamp)
        before=state.paths.cache.k.clone(),state.paths.cache.v.clone()
        r,t=reuse.cache,rebuild.cache
        caches=(r,HSTUKVCache(t.k,r.v,r.seq_len),HSTUKVCache(r.k,t.v,r.seq_len),t)
        combined=crossed_cache(r,t)
        for i,c in enumerate(caches):
            torch.testing.assert_close(_member(combined,i).k,c.k,atol=0,rtol=0)
            torch.testing.assert_close(_member(combined,i).v,c.v,atol=0,rtol=0)
        delta=torch.tensor([timestamp-reuse.last_timestamp],dtype=torch.float32)
        actual=state.score(items,delta)
        for i,c in enumerate(caches):
            torch.testing.assert_close(actual[i:i+1],native_score(current,c,items,delta)[0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(before[0],state.paths.cache.k,atol=0,rtol=0)
        torch.testing.assert_close(before[1],state.paths.cache.v,atol=0,rtol=0)
    assert state.paths.appended_tokens==6 and state.paths.evicted_tokens==5
    assert state.max_score_abs_error<1e-6 and state.max_cache_abs_error<1e-6


@torch.no_grad()
def test_cutover_pool64_preserves_following_real_requests_and_ties(monkeypatch):
    old,current,events,cfg=scenario(monkeypatch)
    requests={86400:[dict(item_idx=41,label=0)],
        86401:[dict(item_idx=42,label=1)],86402:[dict(item_idx=43,label=0)]}
    panel=dict(query_delta=10,history_length=3,pool_ids=np.arange(31,95),
        panel_indices=np.stack((np.r_[0:8,56:64],np.arange(16))))
    rows,stats,ledger,candidate=replay_user(old,current,events,requests,15,cfg,
        check_independent=True,panel=panel)
    plain,plain_stats,_,no_candidate=replay_user(old,current,events,requests,15,cfg,
        check_independent=True)
    assert candidate['scores'].shape==(4,64) and no_candidate is None
    assert stats['appended_events']==plain_stats['appended_events']==3
    assert stats['candidate_panel'] and not ledger['learned_f_used']
    assert ledger['identity_duplicate_reuse_member'] and not ledger['deployment_cost_claim']
    assert len(rows)==12
    for a,b in zip(rows,plain,strict=True):
        assert {k:v for k,v in a.items() if k!='score'}=={k:v for k,v in b.items() if k!='score'}
        assert a['appended_events']=={86400:0,86401:1,86402:3}[a['query_timestamp']]
        torch.testing.assert_close(torch.tensor(a['score']),torch.tensor(b['score']),atol=1e-6,rtol=1e-5)
