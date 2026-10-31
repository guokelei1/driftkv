"""E5 projected integration follows complete trajectories and its 230 reference."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.evaluate_probes import (
    correction_cost, apply_projected_costs, merge_execution,
)
from experiments.ch3_motivation.code.probe_variants import build_variant
from experiments.ch3_motivation.code.probe_fast import evaluate_fast_probe, GraphProbePool
from experiments.ch3_motivation.code.probe_projected_fast import evaluate_projected_probe, ProjectedGraphPool
from experiments.ch3_motivation.code.probe_ledger import full_probe_ledger
from experiments.ch3_motivation.code.probe_cost import padded_read_extra_flops
from experiments.ch3_motivation.code.probe_projected_cost import (
    full_projected_preparation, deployment_projected_graph_setup, projected_padded_read_delta,
)
from hstu_kvcache.evaluation.baselines import event_shapes
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.models import HSTU,HSTUConfig


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    threads=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fixture(variant,stage):
    torch.manual_seed(23)
    model=HSTU(HSTUConfig(num_items=64,num_behaviors=2,hidden_size=64,num_heads=2,
        num_layers=2,max_seq_len=5,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    probe=build_variant(variant,model,stage=stage,hidden_width=7,depth=2,row_chunk=5)
    with torch.no_grad():
        for terms in (probe.f,probe.g,probe.h):
            if terms is not None:
                for layer in terms:
                    weight=layer.weights[-1] if hasattr(layer,"weights") else layer.network[-1].weight
                    weight.normal_(std=.001)
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=5,append_band_size=2,
             query_batch=16,repair_query_batch=16,attention_backend="torch")
    rows={}
    for uid,(before,after) in enumerate(((5,0),(5,1),(2,3),(2,4),(2,3),(1,0)),start=11):
        times=np.r_[np.arange(86400-before,86400),np.arange(86400,86400+after)]
        rows[uid]=times,np.arange(1,len(times)+1),np.ones(len(times),dtype=np.int64)
    def requests(uid,timestamp,count):
        return [dict(uid=uid,query_timestamp=timestamp,raw_item_id=100+j,item_idx=15+j,label=j%2)
                for j in range(count)]
    by_user={14:{86400:requests(14,86400,20)},15:{86404:requests(15,86404,1)},16:{86400:requests(16,86400,3)}}
    return model,probe,SimpleNamespace(rows=rows),by_user,cfg,CostModel(64,2,2,"torch")


@pytest.mark.parametrize("variant",["full_item","layerwise_kv"])
@pytest.mark.parametrize("stage",["f","fg","fgh"])
def test_complete_raw_and_projected_trajectories_match_ledger_and_added_cost(variant,stage):
    model,probe,histories,by_user,cfg,cm=fixture(variant,stage)
    users=list(histories.rows)
    with torch.no_grad():
        raw_scores,raw_execution=evaluate_fast_probe(model,model,probe,histories,by_user,users,cfg,
            pool=GraphProbePool(model,probe,window=5,capture=False),stop_after_last_query=False)
        new_scores,new_execution=evaluate_projected_probe(model,model,probe,histories,by_user,users,cfg,
            pool=ProjectedGraphPool(model,probe,window=5,capture=False),stop_after_last_query=False)
    assert len(raw_scores)==len(new_scores)==24
    assert max(abs(a["score"]-b["score"]) for a,b in zip(raw_scores,new_scores,strict=True))<3e-5
    ledger=full_probe_ledger(histories,by_user,users,cfg,stage=stage,padded=True)
    ledger["padded_execution"]=True
    for key in ("initial_shapes","prepare_shapes","read_shapes","append_shapes"):
        assert ledger[key]==raw_execution[key]==new_execution[key]
    if stage!="f":
        assert ledger["padded_read_shapes"]==raw_execution["padded_read_shapes"]==new_execution["padded_read_shapes"]
    shapes=event_shapes(histories,by_user,users,cfg)
    raw=correction_cost(probe,ledger,17,cm,shapes,cfg,len(users),model=model)
    new=apply_projected_costs(raw,probe,ledger,histories,by_user,users,cfg,model)
    preparation=full_projected_preparation(probe,histories,by_user,users,cfg)
    assert new_execution["projected_execution"]["extra_row_preparation_flops"]==preparation["extra_row_preparation_flops"]
    delta=projected_padded_read_delta(probe,ledger,padded_read_extra_flops)
    startup=deployment_projected_graph_setup(probe,ledger,model,window=5)
    assert new["complete_added_flops"]==(raw["complete_added_flops"]+preparation["extra_row_preparation_flops"]+
        delta["projected_minus_raw_padded_read_flops"]+startup["shared_graph_setup_flops"]-raw["shared_graph_setup_flops"])
    assert new["complete_added_flops"]==sum(new[name] for name in ("calibration_flops","release_preparation_flops",
        "extra_read_flops","extra_native_writer_flops","shared_graph_setup_flops"))
    if stage in ("f","fg"):
        for name in ("calibration_flops","release_preparation_flops","extra_read_flops","extra_native_writer_flops",
                     "shared_graph_setup_flops","complete_added_flops","percent_of_once_rebuild"):
            assert new[name]==raw[name]
        for name in ("release_preparation_flops","extra_candidate_read_flops","actual_behavior_read_extra_flops",
                     "shared_graph_setup_flops","complete_added_flops","percent_of_once_rebuild"):
            assert new["candidate_reads_230"][name]==raw["candidate_reads_230"][name]


@pytest.mark.parametrize("variant",["full_item","layerwise_kv"])
def test_230_projection_preparation_and_keys_follow_release_source_reads_not_removed_actual_queries(variant):
    model,probe,histories,by_user,cfg,cm=fixture(variant,"fgh")
    users=list(histories.rows)
    ledger=full_probe_ledger(histories,by_user,users,cfg,stage="fgh",padded=True)
    ledger["padded_execution"]=True
    raw=correction_cost(probe,ledger,17,cm,event_shapes(histories,by_user,users,cfg),cfg,len(users),model=model)
    new=apply_projected_costs(raw,probe,ledger,histories,by_user,users,cfg,model)
    # Every hypothetical user reads 230 candidates on its release-source
    # history. Keep real behavior evolution. User15's only actual full graph
    # was a late query AFTER its final short write filled the window: this
    # graph disappears in the declared release-source-read sensitivity.
    normalized_by_user={uid:{86400:[dict(item_idx=15)]*230} for uid in users}
    normalized_ledger=full_probe_ledger(histories,normalized_by_user,users,cfg,stage="fgh",padded=True)
    expected=full_projected_preparation(probe,histories,normalized_by_user,users,cfg)
    assert expected["projected_rows"]==12  # idle full5 + behavior full5 + short-full behavior2.
    norm=new["candidate_reads_230"]
    assert norm["extra_row_projection_preparation_flops"]==expected["extra_row_preparation_flops"]
    assert norm["release_preparation_flops"]==raw["candidate_reads_230"]["release_preparation_flops"]+expected["extra_row_preparation_flops"]
    expected_setup=deployment_projected_graph_setup(probe,normalized_ledger,model,window=5)
    assert norm["shared_graph_setup_flops"]==expected_setup["shared_graph_setup_flops"]
    assert {(row["operation"],row["queries"]) for row in norm["graph_setup"]["required_graph_keys"]}=={("append",1),("score",6),("score",16)}
    assert norm["complete_added_flops"]==sum(norm[name] for name in ("calibration_flops","release_preparation_flops",
        "extra_candidate_read_flops","actual_behavior_read_extra_flops","extra_native_writer_flops","shared_graph_setup_flops"))


def test_resumed_projected_per_block_metadata_retains_preparation_and_byte_scopes():
    empty=dict(counts={},initial_shapes=[],prepare_shapes=[],read_shapes=[],append_shapes=[],padded_read_shapes=[])
    parts=[dict(empty,projected_execution=dict(extra_row_preparation_flops=value,
        max_observed_end_state_bytes=dict(padded_projected_table_logical_bytes=size),cost_scope="quality execution only"))
        for value,size in ((17,1024),(23,2048))]
    merged=merge_execution(parts)
    assert [row["extra_row_preparation_flops"] for row in merged["block_projected_execution"]]==[17,23]
    assert [row["max_observed_end_state_bytes"]["padded_projected_table_logical_bytes"]
        for row in merged["block_projected_execution"]]==[1024,2048]
