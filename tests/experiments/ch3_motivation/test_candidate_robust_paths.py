"""Static callbacks produce only their requested independent causal paths."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from experiments.ch3_motivation.code.candidate_robust_paths import StaticPaths, replay_user
from experiments.ch3_motivation.code.kv_evolution import clone_native_trajectory
from experiments.ch3_motivation.code.real_quality_paths import _member
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTU,HSTUConfig,HSTUKVCache


@pytest.fixture
def scenario(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch')
    torch.set_num_threads(1);torch.manual_seed(319)
    old=HSTU(HSTUConfig(num_items=300,num_behaviors=2,hidden_size=8,num_heads=2,
        num_layers=2,max_seq_len=4,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    current=copy.deepcopy(old)
    with torch.no_grad():
        for parameter in current.parameters():parameter.add_(torch.randn_like(parameter)*.04)
    times=np.array([86370,86380,86390,86400,86401,86401,86402,86404,86405,9*86400])
    events=times,np.arange(1,len(times)+1),np.arange(len(times))%2
    cfg=dict(edge=2,release_day=1,end_day=12,history_length=4,append_band_size=2,query_batch=2)
    calls={};memo={}
    def counted(name,fn):
        def transform(cache,ids):
            calls[name]=calls.get(name,0)+1
            assert tuple(ids.shape)==(1,3) and ids.tolist()==[[1,2,3]]
            result=fn(cache);memo[name]=result
            return result
        return transform
    maps=dict(f1=counted('f1',lambda c:HSTUKVCache(c.k+.2,c.v-.1,c.seq_len)),
        f2=counted('f2',lambda c:HSTUKVCache(c.k+2*(memo['f1'].k-c.k),c.v+2*(memo['f1'].v-c.v),c.seq_len)))
    for i in range(4):
        maps[f'linear{i}']=counted(f'linear{i}',lambda c,a=i:HSTUKVCache(
            (1+.03*a)*c.k+.02*c.v,(1-.02*a)*c.v-.03*c.k,c.seq_len))
    return old,current,events,cfg,maps,calls,memo


@torch.no_grad()
def test_exact_eight_members_match_independent_native_fifo_and_scores(scenario):
    old,current,events,cfg,maps,calls,memo=scenario
    paths=StaticPaths(old,current,maps,events,cfg,check_independent=True)
    assert paths.methods==['reuse','f1','f2','linear0','linear1','linear2','linear3','rebuild']
    assert paths.cache.k.shape[1]==8 and calls=={name:1 for name in maps}
    source=NativeTrajectory(old,1,events,86400,max_length=4,append_band_size=2)
    teacher=source.fresh_cache(current)
    for index,expected in ((0,source.cache),(1,memo['f1']),(2,memo['f2']),(7,teacher)):
        actual=_member(paths.cache,index)
        torch.testing.assert_close(actual.k,expected.k,atol=0,rtol=0)
        torch.testing.assert_close(actual.v,expected.v,atol=0,rtol=0)
    references=[]
    for i in range(8):
        native=clone_native_trajectory(source);native.switch(current,2,86400)
        native.cache=_member(paths.cache,i);references.append(native)
    items=torch.tensor([[31,52,78]])
    saw_distinct_new_row=False
    for timestamp in (86400,86401,86402,86403,86405,86406,9*86400+1):
        paths.advance(timestamp)
        for i,native in enumerate(references):
            native.advance(timestamp)
            actual=_member(paths.cache,i)
            torch.testing.assert_close(actual.k,native.cache.k,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(actual.v,native.cache.v,atol=1e-6,rtol=1e-5)
            assert (paths.cursor,paths.appended,paths.evicted)==(native.cursor,native.appended_tokens,native.evicted_tokens)
        before=paths.cache.k.clone(),paths.cache.v.clone()
        delta=torch.tensor([timestamp-references[0].last_timestamp],dtype=torch.float32)
        scores=paths.score(items,delta)
        for i,native in enumerate(references):
            torch.testing.assert_close(scores[i:i+1],native_score(current,native.cache,items,delta)[0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(paths.cache.k,before[0],atol=0,rtol=0)
        torch.testing.assert_close(paths.cache.v,before[1],atol=0,rtol=0)
        if paths.appended:
            saw_distinct_new_row|=bool((paths.cache.k[1,0,-1]!=paths.cache.k[1,2,-1]).any())
    assert saw_distinct_new_row and paths.cursor==len(events[0])
    assert calls=={name:1 for name in maps}
    assert paths.ledger()['event_inputs']['per_append_scalar_tensor_constructions']==0
    json.dumps(paths.ledger())


@torch.no_grad()
def test_real_rows_preserve_labels_ties_and_no_map_native_endpoints(scenario):
    old,current,events,cfg,maps,_,_=scenario
    requests={86400:[dict(item_idx=41,label=0)],
        86401:[dict(item_idx=42+i,label=i%2) for i in range(5)],
        86402:[dict(item_idx=51,label=1)]}
    rows,stats,ledger=replay_user(old,current,maps,events,requests,15,cfg,check_independent=True)
    plain,_,_=replay_user(old,current,{},events,requests,15,cfg,check_independent=True)
    assert len(rows)==7*8 and stats['method_names']==list(ledger['persistent_methods'])
    assert stats['feedback_events']==7 and stats['appended_events']==3
    assert stats['max_batch_abs_error']<1e-6
    index={(r['method'],r['query_timestamp'],r['item_idx']):r for r in rows}
    for r in plain:
        expected=index[r['method'],r['query_timestamp'],r['item_idx']]
        torch.testing.assert_close(torch.tensor(r['score']),torch.tensor(expected['score']),atol=1e-6,rtol=1e-5)
    for r in rows:
        assert r['appended_events']=={86400:0,86401:1,86402:3}[r['query_timestamp']]
        assert r['query_gap']=={86400:10,86401:1,86402:1}[r['query_timestamp']]
        assert r['kv_mse'] is None
    for timestamp,values in requests.items():
        assert [r['label'] for r in rows if r['method']=='f1' and r['query_timestamp']==timestamp]==[r['label'] for r in values]
    json.dumps((stats,ledger))
    def mutate(cache,ids):
        cache.k.add_(1)
        return cache
    with pytest.raises(ValueError,match='mutated'):
        StaticPaths(old,current,{'bad':mutate},events,cfg)
