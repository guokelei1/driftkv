"""Strength recipes share one linear call but retain independent causal state."""
import copy
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.ch3_motivation.code.run_candidate_strength_sweep import STRENGTHS, strength_maps
from experiments.ch3_motivation.code.candidate_robust_paths import replay_user
from hstu_kvcache.models import HSTU,HSTUConfig,HSTUKVCache


@torch.no_grad()
def test_shared_linear_exact_endpoint_and_reset():
    torch.manual_seed(712)
    cache=HSTUKVCache(torch.randn(2,1,3,8),torch.randn(2,1,3,8),3)
    calls=[]
    def linear(c):
        value=HSTUKVCache(c.k*1.13+.17,c.v*.91-.08,c.seq_len)
        calls.append(value)
        return value
    transforms=strength_maps(linear)
    for _ in range(2):
        results={name:fn(cache,None) for name,fn in transforms.items()}
        assert results['alpha1'] is calls[-1]
        for name,a in STRENGTHS:
            torch.testing.assert_close(results[name].k,cache.k+a*(calls[-1].k-cache.k),atol=1e-7,rtol=1e-6)
            torch.testing.assert_close(results[name].v,cache.v+a*(calls[-1].v-cache.v),atol=1e-7,rtol=1e-6)
    assert len(calls)==2


@torch.no_grad()
def test_eight_strength_paths_match_independent_native_and_base(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch')
    torch.set_num_threads(1);torch.manual_seed(719)
    old=HSTU(HSTUConfig(num_items=100,num_behaviors=2,hidden_size=8,num_heads=2,
        num_layers=2,max_seq_len=4,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    current=copy.deepcopy(old)
    for p in current.parameters():p.add_(torch.randn_like(p)*.04)
    times=np.array([86370,86380,86390,86400,86401,86401,86402,86404,86405])
    events=times,np.arange(1,len(times)+1),np.arange(len(times))%2
    cfg=dict(edge=2,release_day=1,end_day=2,history_length=4,append_band_size=2,query_batch=2)
    requests={86400:[dict(item_idx=41,label=0)],86401:[dict(item_idx=42+i,label=i%2) for i in range(3)],
        86402:[dict(item_idx=51,label=1)],86406:[dict(item_idx=61,label=0)]}
    def linear(c):return HSTUKVCache(c.k*1.13+.17,c.v*.91-.08,c.seq_len)
    rows,stats,_=replay_user(old,current,strength_maps(linear),events,requests,15,cfg,check_independent=True)
    base,_,_=replay_user(old,current,{'linear256_rows':lambda c,ids:linear(c)},events,requests,15,cfg,
        check_independent=True)
    assert len(stats['method_names'])==8 and stats['appended_events']==6
    assert stats['max_score_abs_error']<1e-6
    index={(r['method'],r['query_timestamp'],r['item_idx']):r for r in rows}
    for expected in base:
        name='alpha1' if expected['method']=='linear256_rows' else expected['method']
        actual=index[name,expected['query_timestamp'],expected['item_idx']]
        for field in ('label','appended_events','history_rows','query_gap'):assert actual[field]==expected[field]
        torch.testing.assert_close(torch.tensor(actual['score']),torch.tensor(expected['score']),atol=1e-6,rtol=1e-5)
