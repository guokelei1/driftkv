"""Checks for separate per-request denominators and fitting-only selection."""
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.run_rule_sensitivity import merge_reference_parts, request_local_costs
from hstu_kvcache.evaluation.cost import CostModel


def test_request_local_full_denominator_is_per_request_not_once():
    cm=CostModel(32,2,1,"torch")
    cfg=dict(history_length=4)
    policies=[dict(tag="reuse",method="reuse",calibration_flops=0),
        dict(tag="rebuild",method="rebuild",calibration_flops=0),
        dict(tag="layer",method="layer",interval=(0,0),calibration_flops=123)]
    requests=[dict(shape=[4,"query",2],count=3)]
    shapes=dict(first=Counter({(4,"query",2):1}),appends=Counter({(4,1,True):2}))
    result=request_local_costs(policies,requests,shapes,cm,cfg)
    assert result["per_request_full_repair_flops"]==3*cm.full_cache(4)
    assert result["standalone_full_minus_reuse_flops"]==3*cm.full_cache(4)-2*cm.band_append(4,1,window_size=4)
    rows={row["method"]:row for row in result["results"]}
    assert rows["rebuild"]["percent_of_per_request_full_repair"]==100
    assert rows["rebuild"]["percent_of_standalone_full_minus_reuse"]==100
    assert rows["layer"]["shared_reference_added_flops"]==123+3*cm.layer_rebuild(4,(0,0))


def test_no_positive_full_minus_reuse_denominator_is_not_forced_to_ratio():
    cm=CostModel(32,2,1,"torch")
    cfg=dict(history_length=4)
    policies=[dict(tag="reuse",method="reuse",calibration_flops=0),dict(tag="rebuild",method="rebuild",calibration_flops=0)]
    result=request_local_costs(policies,[dict(shape=[4,"query",1],count=1)],
        dict(appends=Counter({(4,1,True):100})),cm,cfg)
    assert not result["standalone_denominator_positive"]
    assert all(row["percent_of_standalone_full_minus_reuse"] is None for row in result["results"])


def test_merge_retains_every_shard_request_and_native_write():
    parts=[dict(observed=dict(layer=dict(prepare_calls=2,appended_rows=5)),
                request_shapes=[dict(shape=[4,"query",1],count=2)]),
           dict(observed=dict(layer=dict(prepare_calls=3,appended_rows=7)),
                request_shapes=[dict(shape=[4,"query",1],count=3)])]
    merged=merge_reference_parts(parts)
    assert merged["observed"]["layer"]==dict(prepare_calls=5,appended_rows=12)
    assert merged["request_shapes"]==[dict(shape=[4,"query",1],count=5)]
