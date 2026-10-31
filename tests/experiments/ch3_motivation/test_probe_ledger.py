import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.ch3_motivation.code.probe_ledger import full_probe_ledger
from experiments.ch3_motivation.code.probe_replay import evaluate_probe
from experiments.ch3_motivation.code.probes import ProbeConfig, ProbeLadder
from hstu_kvcache.models import HSTU,HSTUConfig


@pytest.mark.parametrize("stage",["f","fg","fgh"])
@torch.no_grad()
def test_cpu_shape_ledger_matches_complete_real_replay_with_idle_users(stage,monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    torch.set_num_threads(1)
    model=HSTU(HSTUConfig(num_items=40,num_behaviors=2,hidden_size=8,num_heads=2,
        num_layers=2,max_seq_len=4,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    old=copy.deepcopy(model)
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=4,append_band_size=2,query_batch=2)
    times=86400+np.arange(-6,9)*10
    history=SimpleNamespace(rows={uid:(times,np.arange(1,len(times)+1),np.zeros(len(times),dtype=np.int64))
        for uid in [1,2,3]})
    rows={1:{86405:[dict(uid=1,query_timestamp=86405,item_idx=21,raw_item_id=21,label=1)],
        86455:[dict(uid=1,query_timestamp=86455,item_idx=22,raw_item_id=22,label=0)]},
        3:{86400:[dict(uid=3,query_timestamp=86400,item_idx=23,raw_item_id=23,label=1)]}}
    probe=ProbeLadder(ProbeConfig.from_model(model,hidden_width=7,depth=1,row_chunk=4),stage)
    _,actual=evaluate_probe(old,model,probe,history,rows,[1,2,3],cfg)
    predicted=full_probe_ledger(history,rows,[1,2,3],cfg,stage=stage)
    assert actual["counts"]==predicted["counts"]
    canonical=lambda values: sorted(tuple(sorted(row.items())) for row in values)
    for name in ["initial_shapes","prepare_shapes","read_shapes","append_shapes"]:
        assert canonical(actual[name])==canonical(predicted[name])
    assert predicted["counts"]["appended_rows"]==27


def test_padded_cpu_ledger_includes_last_inherited_row_eviction():
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=4,append_band_size=2,query_batch=16)
    history=SimpleNamespace(rows={1:(86400+np.arange(-4,6),np.arange(10),np.zeros(10))})
    result=full_probe_ledger(history,{},[1],cfg,stage="fgh",padded=True)
    assert sum(row["calls"] for row in result["padded_read_shapes"])==4
    assert any(row["inherited_rows"]==0 for row in result["padded_read_shapes"])
    assert result["counts"].get("candidate_reads",0)==0
