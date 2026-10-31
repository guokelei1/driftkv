"""E5 integration: padded work and shared cold setup survive final accounting."""
import copy
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.evaluate_probes import (
    correction_cost, cached_read_extra_flops, merge_execution,
)
from experiments.ch3_motivation.code.probe_cost import (
    cached_preparation_flops, padded_read_extra_flops, scalar_behavior_flops,
)
from experiments.ch3_motivation.code.probe_variants import build_variant
from hstu_kvcache.models import HSTU,HSTUConfig
from hstu_kvcache.evaluation.cost import CostModel


def fixture(variant,stage):
    model=HSTU(HSTUConfig(num_items=8,num_behaviors=2,hidden_size=64,num_heads=2,
        num_layers=2,max_seq_len=1024,input_dropout=0.,attn_dropout=0.))
    probe=build_variant(variant,model,stage=stage,hidden_width=7,depth=1,row_chunk=1024)
    cfg=dict(history_length=1024,query_batch=16)
    cm=CostModel(64,2,2,"torch")
    # One full user scores 3 candidates, appends one behavior, then scores 1.
    # A second one-row user is idle: the actual ledger never prepares it,
    # whereas the 230-read reference must prepare both users.
    reads=[dict(kind="query",history_length=1024,inherited_rows=1024,queries=3,calls=1),
           dict(kind="query",history_length=1024,inherited_rows=1023,queries=1,calls=1)]
    physical=[]
    if stage!="f":
        reads.append(dict(kind="behavior",history_length=1023,inherited_rows=1023,queries=1,calls=1))
        physical=[dict(kind=row["kind"],physical_read_rows=row["history_length"],
                       inherited_rows=row["inherited_rows"],queries=row["queries"],calls=1) for row in reads]
    execution=dict(counts={},padded_execution=True,
        initial_shapes=[dict(history_length=1024,calls=1),dict(history_length=1,calls=1)],
        prepare_shapes=[dict(history_length=1024,calls=1)],read_shapes=reads,
        append_shapes=[dict(before_length=1024,width=1,inherited_rows=0 if stage=="f" else 1023,calls=1)],
        padded_read_shapes=physical)
    shapes=dict(first=Counter({(1024,"query",3):1}),appends=Counter({(1024,1,True):1}))
    return model,probe,cfg,cm,execution,shapes


@pytest.mark.parametrize("variant",["full_item","layerwise_kv"])
@pytest.mark.parametrize("stage",["f","fg","fgh"])
def test_actual_padded_total_startup_and_independent_230_query_keys(variant,stage):
    model,probe,cfg,cm,execution,shapes=fixture(variant,stage)
    cost=correction_cost(probe,execution,17,cm,shapes,cfg,2,model=model)
    reads=padded_read_extra_flops(probe,1024,3)+padded_read_extra_flops(probe,1024,1)
    behavior=padded_read_extra_flops(probe,1023,1) if stage!="f" else 0
    assert cost["extra_read_flops"]==reads+behavior
    assert cost["release_preparation_flops"]==cached_preparation_flops(probe,1024)
    actual_keys={(row["operation"],row["queries"]) for row in cost["graph_setup"]["required_graph_keys"]}
    assert actual_keys==({("score",1),("score",3)}|({("append",1)} if stage!="f" else set()))
    startup=cost["shared_graph_setup_flops"]
    assert startup>0  # f also captures its native score shapes.
    assert startup==sum(row["setup_flops"] for row in cost["graph_setup"]["per_key"])
    assert all(row["setup_flops"]==3*row["gross_forward_flops_per_call"]
               for row in cost["graph_setup"]["per_key"])
    expected_writer=(scalar_behavior_flops(model,1023)-cm.band_append(1024,1,window_size=1024)
                     if stage!="f" else 0)
    assert cost["extra_native_writer_flops"]==expected_writer
    assert cost["complete_added_flops"]==(17+cached_preparation_flops(probe,1024)+
        reads+behavior+expected_writer+startup)

    normalized=cost["candidate_reads_230"]
    normalized_keys={(row["operation"],row["queries"]) for row in normalized["graph_setup"]["required_graph_keys"]}
    assert normalized_keys==({("score",6),("score",16)}|({("append",1)} if stage!="f" else set()))
    assert normalized_keys!=actual_keys
    expected_reads=(padded_read_extra_flops(probe,1024,230)+cached_read_extra_flops(probe,1,1,230))
    assert normalized["extra_candidate_read_flops"]==expected_reads
    assert normalized["actual_behavior_read_extra_flops"]==behavior
    assert normalized["release_preparation_flops"]==cached_preparation_flops(probe,1024)+cached_preparation_flops(probe,1)
    assert normalized["complete_added_flops"]==(17+normalized["release_preparation_flops"]+
        expected_reads+behavior+expected_writer+normalized["shared_graph_setup_flops"])


@pytest.mark.parametrize("variant",["full_item","layerwise_kv"])
def test_startup_is_shared_once_and_reference_path_has_none(variant):
    model,probe,cfg,cm,execution,shapes=fixture(variant,"fgh")
    first=correction_cost(probe,execution,17,cm,shapes,cfg,2,model=model)
    doubled=copy.deepcopy(execution)
    for name in ("initial_shapes","prepare_shapes","read_shapes","append_shapes","padded_read_shapes"):
        for row in doubled[name]:
            row["calls"]*=2
    doubled_shapes={name:Counter({key:2*count for key,count in values.items()}) for name,values in shapes.items()}
    second=correction_cost(probe,doubled,17,cm,doubled_shapes,cfg,4,model=model)
    startup=first["shared_graph_setup_flops"]
    assert second["shared_graph_setup_flops"]==startup
    assert second["complete_added_flops"]-startup-17==2*(first["complete_added_flops"]-startup-17)
    reference=copy.deepcopy(execution)
    reference.update(padded_execution=False,padded_read_shapes=[])
    cost=correction_cost(probe,reference,17,cm,shapes,cfg,2,model=model)
    assert cost["shared_graph_setup_flops"]==0
    assert cost["candidate_reads_230"]["shared_graph_setup_flops"]==0


def test_resumed_graph_snapshots_are_retained_without_summing_cumulative_setup():
    _,_,_,_,execution,_=fixture("full_item","fgh")
    parts=[]
    for generation,replays in (("old-process",7),("new-process",2)):
        parts.append(dict(execution,graph_pool_generation=generation,
            graph_pool=[dict(operation="score",queries=16,replays=replays,capture_forward_calls=1)],
            quality_shortcut=dict(stop_after_last_query=True,skipped_no_feedback_users=1)))
    merged=merge_execution(parts)
    assert [row["generation"] for row in merged["block_graph_pool_snapshots"]]==["old-process","new-process"]
    assert [row["graphs"][0]["replays"] for row in merged["block_graph_pool_snapshots"]]==[7,2]
    assert len(merged["block_quality_shortcuts"])==2
    assert merged["padded_read_shapes"] and all(row["calls"]==2 for row in merged["padded_read_shapes"])
