"""Projected preparation follows full event lifecycles, not scored GPU tails."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.probe_variants import build_variant
from experiments.ch3_motivation.code.probe_ledger import full_probe_ledger
from experiments.ch3_motivation.code.probe_cost import padded_read_extra_flops
from experiments.ch3_motivation.code.probe_projected_fast import (
    projected_read_extra_flops, projected_row_preparation_flops,
)
from experiments.ch3_motivation.code.probe_projected_cost import (
    full_projected_preparation, deployment_projected_graph_setup, projected_padded_read_delta,
)
from experiments.ch3_motivation.code.probe_projected_fast import (
    ProjectedFastProbeTrajectory, ProjectedGraphPool,
)
from hstu_kvcache.models import HSTU,HSTUConfig


@pytest.mark.parametrize("variant",["head","full_item","layerwise_kv"])
def test_full_projection_cost_matches_actual_complete_trajectories_including_idle_tails(variant,monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        model=HSTU(HSTUConfig(num_items=32,num_behaviors=2,hidden_size=8,num_heads=2,
            num_layers=1,max_seq_len=5,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
        probe=build_variant(variant,model,stage="fgh",hidden_width=7,depth=2)
        cfg=dict(edge=1,release_day=1,end_day=2,history_length=5,append_band_size=2,query_batch=3)
        rows={}
        # Full idle; full behavior-only; short ending just full; short with a
        # full-state behavior; short reaching full then scored there.
        for uid,(before,after) in enumerate(((5,0),(5,1),(2,3),(2,4),(2,3))):
            times=np.r_[np.arange(86400-before,86400),np.arange(86400,86400+after)]
            rows[uid]=(times,np.arange(1,len(times)+1),np.ones(len(times),dtype=np.int64))
        by_user={4:{86404:[dict(item_idx=15)]}}
        histories=SimpleNamespace(rows=rows)
        expected=full_projected_preparation(probe,histories,by_user,list(rows),cfg)
        actual=0
        for uid,events in rows.items():
            trajectory=ProjectedFastProbeTrajectory(model,model,probe,events,cfg,
                pool=ProjectedGraphPool(model,probe,window=5,capture=False))
            for timestamp,requests in by_user.get(uid,{}).items():
                trajectory.advance(timestamp)
                trajectory.score(torch.tensor([[request["item_idx"] for request in requests]]),torch.ones(1))
            trajectory.advance(2*86400)
            actual+=trajectory.report()["projected_execution"]["extra_row_preparation_flops"]
        assert expected["extra_row_preparation_flops"]==actual
        assert expected["projected_users"]==3
        assert expected["projected_rows"]==9  # 5 + 2 + 2, including behavior-only users.
        assert expected["preparation_shapes"]==[dict(inherited_rows=2,calls=2),dict(inherited_rows=5,calls=1)]
    finally:
        torch.set_num_threads(previous)


def test_projected_startup_and_signed_read_replacement_bill_whole_tail_mlp():
    model=HSTU(HSTUConfig(num_items=8,num_behaviors=2,hidden_size=64,num_heads=2,
        num_layers=2,max_seq_len=1024,input_dropout=0.,attn_dropout=0.))
    probe=build_variant("layerwise_kv",model,stage="fgh",hidden_width=128,depth=2)
    execution=dict(read_shapes=[dict(kind="query",history_length=1024,inherited_rows=1,queries=16,calls=2),
        dict(kind="behavior",history_length=1023,inherited_rows=0,queries=1,calls=1)],
        append_shapes=[dict(before_length=1024,width=1,inherited_rows=0,calls=1)],
        padded_read_shapes=[dict(kind="query",physical_read_rows=1024,inherited_rows=1,queries=16,calls=2),
            dict(kind="behavior",physical_read_rows=1023,inherited_rows=0,queries=1,calls=1)])
    setup=deployment_projected_graph_setup(probe,execution,model)
    assert {(row["operation"],row["queries"]) for row in setup["required_graph_keys"]}=={("append",1),("score",16)}
    for row in setup["per_key"]:
        assert row["probe_forward_flops_per_call"]==projected_read_extra_flops(probe,row["physical_read_rows"],row["queries"])
        assert row["setup_flops"]==3*(row["native_forward_flops_per_call"]+row["probe_forward_flops_per_call"])
    delta=projected_padded_read_delta(probe,execution,padded_read_extra_flops)
    physical=2*projected_read_extra_flops(probe,1024,16)+projected_read_extra_flops(probe,1023,1)
    old=2*padded_read_extra_flops(probe,1024,16)+padded_read_extra_flops(probe,1023,1)
    assert delta["projected_physical_padded_read_flops"]==physical
    assert delta["projected_minus_raw_padded_read_flops"]==physical-old<0
