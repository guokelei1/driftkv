"""The final E3 analysis preserves clusters, denominators and unfinished work."""
from pathlib import Path
import csv
import json
import sys

import pytest

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.analyze_selective_quality import (
    aggregate_quality, aggregate_costs, compare_fresh_anchors, analyze,
)
from experiments.ch3_motivation.code import analyze_selective_quality as analysis


def row(uid,method,score):
    return dict(uid=uid,query_timestamp=1,raw_item_id=uid,item_idx=uid,label=uid%2,method=method,score=score)


def test_opposite_edges_cancel_in_every_shared_uid_draw():
    users=list(range(512))
    forward,reverse=[],[]
    for uid in users:
        a=((uid*17)%31)/31
        b=((uid*13)%29)/29
        for tag,value,opposite in (("reuse",a,b),("rebuild",b,a),("layer_2layers",b,a)):
            forward.append(row(uid,tag,value))
            reverse.append(row(uid,tag,opposite))
    summary=aggregate_quality([forward,reverse],users,replicates=32)
    result={value["method"]:value for value in summary["results"]}["layer_2layers"]
    assert result["mean_auc_gain_over_reuse"]==0.
    assert result["mean_auc_gain_over_reuse_ci95"]==[0.,0.]
    assert result["mean_auc_minus_rebuild_ci95"]==[0.,0.]
    assert summary["confidence"]["shared_uid_multiplicity_across_releases"]


def test_recovery_is_ratio_of_release_mean_gains():
    users=list(range(512))
    edges=[]
    for rebuild_positive_high in (192,256):
        rows=[]
        for uid in users:
            positive_index=uid//2
            scores=dict(reuse=.5,
                rebuild=float(positive_index<rebuild_positive_high) if uid%2 else .5,
                layer_2layers=float(positive_index<160) if uid%2 else .5)
            rows.extend(row(uid,tag,value) for tag,value in scores.items())
        edges.append(rows)
    summary=aggregate_quality(edges,users,replicates=32)
    result={value["method"]:value for value in summary["results"]}["layer_2layers"]
    assert result["per_edge_auc"]==[.625,.625]
    assert result["mean_auc_gain_over_reuse"]==.125
    assert summary["mean_reference_gap"]==.375
    assert result["recovery_percent"]==pytest.approx(100/3)
    assert result["recovery_percent"]!=pytest.approx((50+25)/2)
    assert len(result["mean_auc_ci95"])==len(result["mean_auc_gain_over_reuse_ci95"])==2
    assert len(result["mean_auc_minus_rebuild_ci95"])==2


def test_aggregate_ratio_interval_is_suppressed_for_zero_reference_gap():
    users=list(range(512))
    rows=[row(uid,tag,float(uid%2) if tag=="layer_2layers" else .5)
          for uid in users for tag in ("reuse","rebuild","layer_2layers")]
    summary=aggregate_quality([rows],users,replicates=32)
    assert summary["confidence"]["nonpositive_or_invalid_reference_gap_replicates"]==32
    result={value["method"]:value for value in summary["results"]}["layer_2layers"]
    assert result["recovery_percent"] is None and result["recovery_percent_ci95"] is None
    assert result["mean_auc_gain_over_reuse_ci95"]==[.5,.5]


def test_actual_and_normalized_costs_sum_flops_before_dividing():
    def ledger(denominator,added):
        return dict(once_rebuild_flops=denominator,common_native_append_flops=7,
            results=[dict(method="reuse",added_flops=0),dict(method="rebuild",added_flops=denominator),
                     dict(method="layer_2layers",added_flops=added)])
    costs=[dict(actual_event_ledger=ledger(10,5),candidate_reads_230=ledger(20,10)),
           dict(actual_event_ledger=ledger(90,9),candidate_reads_230=ledger(80,8))]
    summary=aggregate_costs(costs,["reuse","rebuild","layer_2layers"])
    actual={value["method"]:value for value in summary["actual_event_ledger"]["results"]}["layer_2layers"]
    normalized={value["method"]:value for value in summary["candidate_reads_230"]["results"]}["layer_2layers"]
    assert actual["added_flops"]==14 and actual["percent_of_once_rebuild"]==14
    assert normalized["added_flops"]==18 and normalized["percent_of_once_rebuild"]==18
    assert actual["percent_of_once_rebuild"]!=(50+10)/2
    assert summary["actual_event_ledger"]["common_native_append_flops"]==14


def test_small_panel_never_produces_quality_result():
    with pytest.raises(ValueError,match="512"):
        aggregate_quality([[row(1,"reuse",.5),row(1,"rebuild",.6)]],[1],replicates=8)


def test_fresh_anchor_comparison_checks_every_event_and_tolerance():
    anchors=[row(uid,tag,.5) for uid in (1,2) for tag in ("reuse","rebuild")]
    actual=[dict(value,score=value["score"]+1e-6) for value in anchors if value["uid"]==1]
    result=compare_fresh_anchors(actual,anchors,[1])
    assert result["scores"]==2 and result["passed"]
    actual[0]["score"]+=1e-3
    with pytest.raises(ValueError,match="tolerance"):
        compare_fresh_anchors(actual,anchors,[1])
    actual[0]["raw_item_id"]+=1
    with pytest.raises(ValueError,match="coverage"):
        compare_fresh_anchors(actual,anchors,[1])


def test_incomplete_population_queue_cannot_write_partial_analysis(tmp_path):
    output=tmp_path/"analysis"
    with pytest.raises(ValueError,match="all ten E3 runs"):
        analyze(tmp_path/"runs",tmp_path/"qualification",output)
    assert not output.exists()


def test_complete_driver_writes_both_panels_and_all_partial_release_rows(tmp_path,monkeypatch):
    """Exercise the actual output plumbing; score/block math is tested above."""
    root=tmp_path/"runs"
    users=list(range(2000))
    methods=["reuse","rebuild"]+[f"method_{i}" for i in range(22)]
    for panel in ("main","confirmation"):
        for edge in range(1,6):
            folder=root/f"edge_{edge}"/panel/"attempt_1"
            folder.mkdir(parents=True)
            (folder/"summary.json").write_text("{}")
            (folder/"status.json").write_text(json.dumps(dict(state="complete")))
    def fake_edge(folder,manifest,edge,code,cache):
        rows=[dict(method=method,setting={},auc=.7,auc_gain_over_reuse=.1,recovery_percent=50.,
            added_flops=5,percent_of_once_rebuild=50.,normalized_230_percent_of_once_rebuild=25.,
            auc_ci95=[.6,.8],auc_gain_over_reuse_ci95=[.01,.2],recovery_percent_ci95=[10.,90.]) for method in methods]
        ledger=dict(once_rebuild_flops=10,results=[dict(method=method,added_flops=5) for method in methods])
        costs=dict(actual_event_ledger=ledger,candidate_reads_230=ledger)
        record=dict(edge=edge,quality_users=2000,feedback_users=1000,feedback_events=1000,
            evaluation_window=dict(name="E14_partial" if edge==5 else "E14",complete_source_horizon=edge!=5),
            results=rows,block_audit=dict(blocks=8))
        return users,[],costs,record
    def fake_aggregate(rows,population,**kwargs):
        assert len(rows)==5 and population==users and kwargs["replicates"]==1000
        return dict(quality_users=2000,edges=5,confidence=dict(replicates=1000),
            results=[dict(method=method,mean_auc=.7,mean_auc_gain_over_reuse=.1,mean_auc_minus_rebuild=-.1,
                recovery_percent=50.,mean_auc_ci95=[.6,.8],mean_auc_gain_over_reuse_ci95=[.01,.2],
                mean_auc_minus_rebuild_ci95=[-.2,-.01],recovery_percent_ci95=[10.,90.]) for method in methods])
    monkeypatch.setattr(analysis,"_audit_edge",fake_edge)
    monkeypatch.setattr(analysis,"aggregate_quality",fake_aggregate)
    monkeypatch.setattr(analysis.run_selective_population,"sources",lambda:{})
    output=tmp_path/"analysis"
    result=analyze(root,tmp_path/"qualification",output)
    assert result["audit"]["blocks"]==80 and result["audit"]["points"]==240
    for panel in ("main","confirmation"):
        assert result["panels"][panel]["edges"]==5
        assert len(result["panels"][panel]["edge_records"])==5
    saved=json.loads((output/"summary.json").read_text())
    assert saved["panels"]["confirmation"]["edge_records"][-1]["evaluation_window"]["name"]=="E14_partial"
    with (output/"metrics_by_edge.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==240 and sum(row["evaluation_window"]=="E14_partial" for row in rows)==48
    with (output/"metrics_by_panel.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==48 and {row["panel"] for row in rows}=={"main","confirmation"}
