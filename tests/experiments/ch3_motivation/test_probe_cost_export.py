"""CPU cost-only fixtures: actual typed tensors, no quality scores or claims."""
import copy
import csv
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.probe_cost_export import (
    collect_costs, write_cost_csvs, ACTUAL_COMPONENTS, REFERENCE_COMPONENTS,
)
from experiments.ch3_motivation.code.analyze_probe_quality_multiedge import _training_record
from experiments.ch3_motivation.code.probe_variants import build_variant, save_variant
from hstu_kvcache.models import HSTU,HSTUConfig
from hstu_kvcache.common.artifacts import sha256


def fixture(tmp_path,*,family="ladder",panels=("main",),seeds=(17,),sizes=(64,)):
    # These synthetic costs exercise export arithmetic only. No formal quality
    # result is made: the input report here stands in for the caller's audit.
    model=HSTU(HSTUConfig(num_items=16,num_behaviors=2,hidden_size=8,num_heads=2,num_layers=2,
        max_seq_len=1024,input_dropout=0.,attn_dropout=0.)).eval()
    stage="fgh" if family=="ladder" else "f"
    probe=build_variant("full_item",model,stage=stage,hidden_width=7,depth=1,query_chunk=16,row_chunk=1024)
    loaded=[];weight_cache={}
    for panel in panels:
        for seed in seeds:
            if panel=="confirmation" and seed!=17:continue
            for n in sizes:
                for edge in range(1,6):
                    path=tmp_path/f"weights/fit{n}/seed{seed}/edge{edge}/{stage}.pt"
                    if (edge,seed,n) not in weight_cache:
                        identity=dict(edge=edge,seed=seed,variant="full_item",fit_users=n,fit_uids=list(range(n)),
                            validation_users=list(range(1000,1016)),models={f"old{edge}":"old",f"new{edge}":"new"},
                            input_binding_sha256=f"input{edge}",calibration_candidate_sha256=f"candidates{edge}",
                            learning_rate=.00003,steps=2,queries=16,stages="f,fg,fgh" if family=="ladder" else "f",
                            controls=family=="ladder",numerical_sources={
                                f"experiments/ch3_motivation/code/{name}":sha256(ROOT/"experiments/ch3_motivation/code"/name)
                                for name in ("probes.py","probe_variants.py")})
                        metadata=dict(identity=identity,stage=stage,method_label=stage,seed=seed,training_users=n,
                            best_step=0,completed_steps=2,objective=dict(read_weight=1.,score_weight=1.),
                            fit_pool_loss_scales=dict(version="fit_pool_equal_user_length_normalized_v1",source="fitting_users_only"),
                            cumulative_calibration_flops_estimate=100000+edge*10000+seed*100+n,
                            total_calibration_flops_estimate=90000,operation_counts=dict(training_probe_rows=7,
                                validation_probe_rows=11,training_sample_calls=2,training_backward_calls=2))
                        save_variant(path,probe,metadata)
                        payload=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
                        weight_cache[edge,seed,n]=payload
                    payload=weight_cache[edge,seed,n];identity=payload["metadata"]["identity"]
                    calibration=payload["metadata"]["cumulative_calibration_flops_estimate"]
                    multiplier=2 if panel=="confirmation" else 1
                    actual=dict(calibration_flops=calibration,release_preparation_flops=1000*edge,
                        extra_read_flops=10000*(6-edge),extra_native_writer_flops=-100*edge,shared_graph_setup_flops=500*edge,
                        once_rebuild_flops=100000*edge**2*multiplier)
                    reference=dict(calibration_flops=calibration,release_preparation_flops=2000*edge,
                        extra_candidate_read_flops=700*edge,actual_behavior_read_extra_flops=300*edge,
                        extra_native_writer_flops=-100*edge,shared_graph_setup_flops=800*edge,
                        once_rebuild_flops=200000*(6-edge)*multiplier)
                    for account,components in ((actual,ACTUAL_COMPONENTS),(reference,REFERENCE_COMPONENTS)):
                        account["complete_added_flops"]=sum(account[key] for key in components)
                        account["percent_of_once_rebuild"]=100*account["complete_added_flops"]/account["once_rebuild_flops"]
                    actual["candidate_reads_230"]=reference
                    start=10000 if panel=="main" else 20000
                    summary=dict(edge=edge,seed=seed,variant="full_item",fit_users=n,width=7,depth=1,
                        method_label=stage,quality_users=2000,costs=actual,
                        identity=dict(panel=panel,edge=edge,backend="projected",users=list(range(start,start+2000)),
                            models=identity["models"],input_binding_sha256=identity["input_binding_sha256"],
                            checkpoint=str(path),checkpoint_sha256=sha256(path)))
                    summary_path=tmp_path/f"summaries/{panel}/fit{n}/seed{seed}/edge{edge}.json"
                    summary_path.parent.mkdir(parents=True,exist_ok=True);summary_path.write_text(json.dumps(summary))
                    training=_training_record(summary,payload)
                    training["formal_main_calibration_bound" if family=="ladder" else "formal_curve_calibration_bound"]=True
                    loaded.append(dict(path=str(summary_path),summary=summary,training=training))
    if family=="ladder":
        analysis=dict(status="complete",panels={panel:dict(primary_aggregate_computed=True,formal_plan_prepared=True,
            edge_records=[dict(methods=[dict(source_summary=item["path"]) for item in loaded
                if item["summary"]["identity"]["panel"]==panel])]) for panel in panels})
    else:
        analysis=dict(status="complete",primary_aggregate_computed=True,formal_curve_plan_bound=True,
            edge_records=[dict(points=[dict(source_summary=item["path"]) for item in loaded])])
    return loaded,analysis


def test_typed_nonzero_correction_preserves_sources_components_and_ratio_of_sums(tmp_path):
    loaded,analysis=fixture(tmp_path)
    before=copy.deepcopy(loaded)
    hashes={str(path):sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    result=collect_costs(loaded,analysis,family="ladder")
    assert result["status"]=="complete" and result["point_count"]==5 and result["aggregate_count"]==2
    assert loaded==before
    assert hashes=={str(path):sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    for point in result["points"]:
        assert point["calibration_correction"]["shared_row_normalization_deduction_flops"]==4096
        for raw,corrected,components in ((point["raw"],point["corrected"],ACTUAL_COMPONENTS),
                (point["raw"]["candidate_reads_230"],point["corrected"]["candidate_reads_230"],REFERENCE_COMPONENTS)):
            assert corrected["complete_added_flops"]==raw["complete_added_flops"]-4096
            assert corrected["complete_added_flops"]==sum(corrected[key] for key in components)
            assert corrected["once_rebuild_flops"]==raw["once_rebuild_flops"]
            assert all(corrected[key]==raw[key] for key in components if key!="calibration_flops")
    actual=next(row for row in result["aggregates"] if row["ledger"]=="actual_event")
    accounts=[point["corrected"] for point in result["points"]]
    assert actual["percent_of_once_rebuild"]==pytest.approx(
        100*sum(row["complete_added_flops"] for row in accounts)/sum(row["once_rebuild_flops"] for row in accounts))
    assert abs(actual["percent_of_once_rebuild"]-sum(row["percent_of_once_rebuild"] for row in accounts)/5)>1
    outputs=write_cost_csvs(result,tmp_path/"export")
    assert {Path(path).name for path in outputs}=={"cost_points.csv","cost_aggregates.csv"}
    with Path(outputs[0]).open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==10 and {row["ledger"] for row in rows}=={"actual_event","analytic230"}
    assert all(int(row["raw_complete_added_flops"])-int(row["complete_added_flops"])==4096 for row in rows)


def test_populations_seeds_and_fitting_sizes_keep_distinct_five_edge_aggregates(tmp_path):
    loaded,analysis=fixture(tmp_path,family="f_population_curve",seeds=(17,23),sizes=(64,128))
    result=collect_costs(loaded,analysis,family="f_population_curve")
    assert len(result["aggregates"])==8
    assert {(row["seed"],row["fit_users"],row["ledger"]) for row in result["aggregates"]}=={
        (seed,n,ledger) for seed in (17,23) for n in (64,128) for ledger in ("actual_event","analytic230")}
    assert all(row["calibration_normalization_deduction_flops"]==0 for row in result["aggregates"])
    main,report=fixture(tmp_path/"panels",panels=("main","confirmation"),seeds=(17,23))
    result=collect_costs(main,report,family="ladder")
    assert len(result["aggregates"])==6
    assert {(r["panel"],r["seed"]) for r in result["aggregates"]}=={("main",17),("main",23),("confirmation",17)}


@pytest.mark.parametrize("fault",["component_sum","raw_calibration","weight_sha","training_identity","unbound","missing_edge"])
def test_complete_export_rejects_inconsistent_costs_or_checkpoint_bindings(tmp_path,fault):
    loaded,analysis=fixture(tmp_path)
    item=loaded[0]
    if fault=="component_sum":item["summary"]["costs"]["complete_added_flops"]+=1
    elif fault=="raw_calibration":
        item["summary"]["costs"]["calibration_flops"]+=1
        item["summary"]["costs"]["complete_added_flops"]+=1
        item["summary"]["costs"]["percent_of_once_rebuild"]=100*item["summary"]["costs"]["complete_added_flops"]/item["summary"]["costs"]["once_rebuild_flops"]
    elif fault=="weight_sha":item["summary"]["identity"]["checkpoint_sha256"]="a"*64
    elif fault=="training_identity":item["training"]["identity"]["learning_rate"]=.1
    elif fault=="unbound":item["training"]["formal_main_calibration_bound"]=False
    else:
        loaded.pop();analysis["panels"]["main"]["edge_records"][0]["methods"].pop()
    with pytest.raises(ValueError):collect_costs(loaded,analysis,family="ladder")


def test_pending_grid_does_not_read_weights_or_emit_cost_csv(tmp_path):
    result=collect_costs([dict(path="absent",summary={})],dict(status="not_ready"),family="ladder")
    assert result["status"]=="pending_complete_audited_grid" and not result["points"] and not result["aggregates"]
    assert write_cost_csvs(result,tmp_path)==[] and not list(tmp_path.glob("*.csv"))


@pytest.mark.parametrize("family",["ladder","f_population_curve"])
def test_actual_analyzer_entry_keeps_empty_grid_and_costs_pending(tmp_path,family):
    if family=="ladder":
        from experiments.ch3_motivation.code.analyze_probe_quality_multiedge import analyze
    else:
        from experiments.ch3_motivation.code.analyze_probe_population_curve import analyze
    result=analyze([],formal_plan_path=tmp_path/"not_selected.json",replicates=1)
    assert result["status"]=="not_ready"
    assert result["costs"]["status"]=="pending_complete_audited_grid"
    assert result["costs"]["points"]==[] and result["costs"]["aggregates"]==[]
