"""Only cached-prefix time, with causal cutoff ties, changes the gap input."""
import copy
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.ch3_motivation.code.score_state_lineage_cached_gap import cached_prefix_gaps, score_user
from experiments.ch3_motivation.code.score_state_lineage import score_user as raw_gap_score_user
from experiments.ch3_motivation.code.state_lineage_probe import RowResidualMap
from hstu_kvcache.models import HSTU,HSTUConfig


def events(times):
    return np.array(times,dtype=np.int64),np.arange(1,len(times)+1),np.zeros(len(times),dtype=np.int64)


def test_gap_depends_only_on_last_strictly_precutoff_event_without_clipping():
    queries=[40,45,1_000_000]
    expected=[10,15,999_970]
    assert cached_prefix_gaps(events([10,20,30,40,50]),queries,40)==expected
    # Adding tied-cutoff events or replacing all later behavior cannot alter it.
    assert cached_prefix_gaps(events([10,20,30,40,40,41,60,200]),queries,40)==expected
    assert cached_prefix_gaps(events([10,20,31,40,50]),queries,40)==[9,14,999_969]
    assert cached_prefix_gaps(events([]),queries,40)==[0,0,0]
    assert cached_prefix_gaps(events([40,41]),queries,40)==[0,0,0]


@torch.no_grad()
def test_cached_gap_preserves_caches_real_events_labels_and_four_paths(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch');torch.set_num_threads(1);torch.manual_seed(721)
    old=HSTU(HSTUConfig(num_items=100,num_behaviors=2,hidden_size=8,num_heads=2,
        num_layers=2,max_seq_len=1024,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    current=copy.deepcopy(old)
    for p in current.parameters():p.add_(torch.randn_like(p)*.02)
    bank={v:{kind:RowResidualMap(4,8,8).eval() for kind in ('chain','direct')} for v in (2,3)}
    history=events([86370,86380,86390,86400,86401,86403])
    requests={86400:[dict(item_idx=41,label=0)],86404:[dict(item_idx=42,label=1)]}
    raw,raw_stats=raw_gap_score_user(old,current,bank,history,requests,19,3,86400,True)
    cached,stats=score_user(old,current,bank,history,requests,19,3,86400,True)
    assert stats['method_names']==raw_stats['method_names']==['reuse','direct','chain','rebuild']
    assert stats['initial_mse']==raw_stats['initial_mse']
    assert stats['query_gap_protocol']=='cached_prefix_gap'
    for a,b in zip(raw,cached,strict=True):
        for field in ('uid','query_timestamp','item_idx','label','method','history_rows','appended_events','kv_mse','k_mse','v_mse'):
            assert a[field]==b[field]
        assert b['query_gap']=={86400:10,86404:14}[b['query_timestamp']]
        if b['query_timestamp']==86400:assert a['score']==b['score']
        else:assert a['query_gap']==1
    assert stats['max_cache_abs_error']==0 and stats['max_score_abs_error']<1e-6
