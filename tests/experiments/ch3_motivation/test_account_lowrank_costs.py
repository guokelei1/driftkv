"""Actual geometry accounting is independent of model tensor storage."""
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import account_lowrank_costs as costs
from experiments.ch3_motivation.code.probe_variants import build_variant
from hstu_kvcache.models import HSTU,HSTUConfig
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def geometry(tmp_path,*,status="formal_tiny_checkpoint",progress=1.):
    model=HSTU(HSTUConfig(num_items=40,num_behaviors=2,hidden_size=32,num_heads=1,num_layers=1,
        max_seq_len=4,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    path=tmp_path/"actual_tiny.pt"
    torch.save(dict(status=status,progress=progress,version="v1",config=asdict(model.cfg),model=model.state_dict()),path)
    return model,path


@pytest.mark.parametrize("stage",["fg","fgh"])
def test_meta_and_cpu_actual_geometry_have_identical_complete_flops_and_full_tails(tmp_path,stage):
    model,path=geometry(tmp_path)
    meta,info=costs.load_geometry(path,sha256(path))
    assert all(value.device.type=="meta" for value in meta.parameters())
    assert info["weight_tensors_loaded_into_model"] is False and info["progress"]==1.
    probe=build_variant("full_item",model,stage=stage,hidden_width=7,depth=2,row_chunk=4,query_chunk=2)
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=4,append_band_size=2,query_batch=2,attention_backend="torch")
    at=86400
    history=SimpleNamespace(rows={
        1:(at+np.arange(-4,7),np.arange(1,12),np.ones(11,dtype=np.int64)),
        2:(at+np.array([-2,-1,0,1,4]),np.arange(1,6),np.ones(5,dtype=np.int64))})
    queries={1:{at+2:[None]},2:{at+1:[None]}}
    fit=dict(cumulative_calibration_flops_estimate=12345)
    cpu,ledger=costs.method_cost(probe,fit,model,history,queries,[1,2],cfg)
    predicted,meta_ledger=costs.method_cost(probe,fit,meta,history,queries,[1,2],cfg)
    assert cpu==predicted and ledger==meta_ledger
    assert cpu["calibration_flops"]==12345
    assert cpu["complete_added_flops"]==sum(cpu[key] for key in ("calibration_flops","release_preparation_flops",
        "extra_read_flops","extra_native_writer_flops","shared_graph_setup_flops"))
    analytic=cpu["candidate_reads_230"]
    assert analytic["users"]==2 and analytic["candidate_reads_per_user"]==230
    assert analytic["complete_added_flops"]!=cpu["complete_added_flops"]
    assert ledger["padded_execution"] and cpu["graph_setup"]["required_graph_keys"]
    timeline=costs.timeline_counts(history,queries,[1,2],cfg)
    assert timeline["complete_behavior_appended_rows"]==10
    assert timeline["before_last_feedback_behavior_rows"]==3
    assert timeline["post_last_feedback_behavior_rows"]==7
    assert ledger["counts"]["appended_rows"]==10
    truncated=SimpleNamespace(rows={uid:tuple(column[:np.searchsorted(events[0],max(queries[uid]),side="left")]
        for column in events) for uid,events in history.rows.items()})
    shortcut,short=costs.method_cost(probe,fit,meta,truncated,queries,[1,2],cfg)
    assert short["counts"]["appended_rows"]==3
    assert cpu["complete_added_flops"]>shortcut["complete_added_flops"]


@pytest.mark.parametrize("status,progress",[("canary_checkpoint",1.),("formal_tiny_checkpoint",.5)])
def test_geometry_rejects_nonformal_or_incomplete_actual_checkpoint(tmp_path,status,progress):
    _,path=geometry(tmp_path,status=status,progress=progress)
    with pytest.raises(ValueError,match="completed formal backbone"):costs.load_geometry(path)


def test_missing_formal_weights_wait_without_selected_cost_or_research_rank_claim(tmp_path):
    path=tmp_path/"pending.json";costs.write_json(path,dict(status="waiting_selection",weight_bindings={}))
    result=costs.generate(path)
    assert result["status"]=="waiting_formal_weights" and not result["cost_bindings"]
    assert not result["complete_serving_cost_ready"] and not result["gpu_used"]
    assert result["research_overhead"]["svd_flops"] is None
    assert result["research_overhead"]["hindsight_rank_read_intervention_flops"] is None
    assert result["calibration_accounting"]["format"]=="e6_raw_and_corrected_calibration_v1"
    bindings={f"edge_{e}_seed_{s}":dict(trained_fg_checkpoint=str(tmp_path/f"missing_fg_{e}_{s}.pt"),
        full_h_checkpoint=str(tmp_path/f"missing_h_{e}_{s}.pt")) for e in range(1,6) for s in (17,23,29)}
    costs.write_json(path,dict(status="prepared_for_root_review",weight_bindings=bindings))
    result=costs.generate(path)
    assert len(result["missing_dependencies"])==30 and not result["cost_bindings"]


def test_current_source_cannot_overwrite_plan_freeze_via_alias():
    path=Path(costs.__file__).resolve();relative=str(path.relative_to(ROOT.resolve()))
    digest=sha256(path)
    assert costs._merge({str(path):digest},{relative:digest})=={str(path):digest}
    with pytest.raises(ValueError,match="frozen SHA conflict"):
        costs._merge({str(path):"0"*64},{relative:digest})


def test_actual15_bindings_pair_audit_meta_accounting_and_csv(tmp_path,tmp_path_factory):
    # Reuse the handoff's real sampling manifests/NPZ/Parquet and typed probe
    # metadata. The fixture's numerical values are synthetic test data only.
    pass  # imports use the installed package / repository namespace
    import tests.experiments.ch3_motivation.test_plan_lowrank_formal as handoff
    assets=handoff.assets.__wrapped__(tmp_path_factory)
    world=handoff.world.__wrapped__(tmp_path_factory,assets)
    model=HSTU(HSTUConfig(num_items=64,num_behaviors=2,hidden_size=32,num_heads=1,num_layers=1,
        max_seq_len=1024,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    fg=build_variant("full_item",model,stage="fg",hidden_width=128,depth=1,query_chunk=16,row_chunk=1024)
    full=fg.expanded("fgh")
    calibrations=dict(fg=10000000,fgh=20000000)
    # Transparent synthetic executed-shape counters:1024 one-user steps on
    # two retained rows; step0 plus every8step validate16 two-row histories.
    train_rows=2*1024
    validation_rows=2*16*(1+1024//8)
    deduction=2*(2*32+32)*(3*train_rows+validation_rows)
    models={}
    for version in range(6):
        path=tmp_path/f"v{version}.pt"
        torch.save(dict(status="formal_tiny_checkpoint",progress=1.,version=f"v{version}",
            config=asdict(model.cfg),model=model.state_dict()),path)
        models[version]=(str(path),sha256(path))
    formal=copy.deepcopy(world.formal)
    for edge in range(1,6):
        folder=world.root/f"fit256/edge_{edge}";binding_path=folder/"inputs.json"
        raw=json.loads(binding_path.read_text())
        raw["settings"].update(append_band_size=128,query_batch=16,attention_backend="torch")
        costs.write_json(binding_path,raw);binding_sha=sha256(binding_path)
        edge_jobs=[j for j in formal["main_jobs"] if j["edge"]==edge]
        manifest_path=Path(edge_jobs[0]["sampling_manifest"])
        manifest=json.loads(manifest_path.read_text());manifest["sourcebinding"]["sha256"]=binding_sha
        costs.write_json(manifest_path,manifest);manifest_sha=sha256(manifest_path)
        formal["frozen_sha256"][str(binding_path)]=binding_sha
        formal["frozen_sha256"][str(manifest_path)]=manifest_sha
        fit_binding=formal["main_input_bindings"][str(edge)]
        fit_binding.update(input_binding_sha256=binding_sha,sampling_manifest_sha256=manifest_sha,gap_manifest_sha256=manifest_sha)
        model_map=dict(models[v] for v in (edge-1,edge))
        for job in edge_jobs:
            job.update(sampling_manifest_sha256=manifest_sha,gap_manifest_sha256=manifest_sha)
            for method,probe in (("fg",fg),("fgh",full)):
                path=costs.quality.checkpoint_path(world.main_root,job,method)
                metadata=torch.load(path,map_location="cpu",weights_only=False)["metadata"]
                metadata["identity"].update(models=model_map,input_binding_sha256=binding_sha,
                    sampling_manifest_sha256=manifest_sha,gap_manifest_sha256=manifest_sha)
                metadata["identity"]["numerical_sources"]["experiments/ch3_motivation/code/probe_variants.py"]=sha256(
                    ROOT/"experiments/ch3_motivation/code/probe_variants.py")
                metadata.update(stage=method,cumulative_calibration_flops_estimate=calibrations[method],
                    total_calibration_flops_estimate=5000000 if method=="fg" else 10000000,
                    operation_counts=dict(training_probe_rows=train_rows,validation_probe_rows=validation_rows,
                        training_sample_calls=1024,training_backward_calls=1024))
                from experiments.ch3_motivation.code.probe_variants import save_variant
                save_variant(path,probe,metadata)
        scenario=world.scenario_root/f"edge_{edge}/feedback_union_2000"
        summary_path=scenario/"summary.json";summary=json.loads(summary_path.read_text())
        summary["identity"].update(models=model_map,input_binding_sha256=binding_sha)
        costs.write_json(summary_path,summary)
        path=scenario/"scenario_manifest.json";population=json.loads(path.read_text())
        population["input_binding"]["sha256"]=binding_sha
        population["qualification"]["summary_sha256"]=sha256(summary_path)
        costs.write_json(path,population)
    costs.write_json(world.formal_path,formal)
    bindings={};populations={}
    for job in formal["main_jobs"]:
        edge,seed=job["edge"],job["seed"]
        popfile=world.scenario_root/f"edge_{edge}/feedback_union_2000/scenario_manifest.json"
        if edge not in populations:populations[edge]=costs.load_population(popfile,edge)
        _,users,population,provenance,_,_=populations[edge]
        full_path=costs.quality.checkpoint_path(world.main_root,job,"fgh")
        fg_path=costs.quality.checkpoint_path(world.main_root,job,"fg")
        probe,metadata=costs.load_variant(full_path)
        model_map=dict(models[v] for v in (edge-1,edge))
        _,_,paired=costs.load_fg_control(fg_path,probe,metadata,source_population=population,evaluation_uids=users,
            edge=edge,models=model_map,calibration_binding_sha256=provenance["calibration_input_binding_sha256"])
        key=f"edge_{edge}_seed_{seed}"
        bindings[key]=dict(id=key,edge=edge,seed=seed,full_h_checkpoint=str(full_path),full_h_checkpoint_sha256=sha256(full_path),
            trained_fg_checkpoint=str(fg_path),trained_fg_checkpoint_sha256=sha256(fg_path),population_file=str(popfile),
            population_sha256=sha256(popfile),uids=users,uids_sha256=costs._uid_hash(users),users=2000,
            population_provenance=provenance,models=model_map,paired_control=paired,
            cumulative_calibration_flops_estimate=calibrations)
    path=tmp_path/"actual_plan_schema.json"
    costs.write_json(path,dict(status="prepared_for_root_review",formal_plan=str(world.formal_path),
        formal_plan_sha256=sha256(world.formal_path),weight_bindings=bindings,frozen_sha256=formal["frozen_sha256"]))
    output=tmp_path/"costs"
    result=costs.main(SimpleNamespace(plan=path,output=output,prepare_cohorts=False))
    assert result["status"]=="complete" and result["complete_serving_cost_ready"] and len(result["cost_bindings"])==15
    assert set(result["cost_bindings"])==set(bindings) and not result["gpu_used"]
    for key,value in result["cost_bindings"].items():
        assert value["full_h_checkpoint_sha256"]==bindings[key]["full_h_checkpoint_sha256"]
        assert value["calibration_accounting"]["format"]=="e6_raw_and_corrected_calibration_v1"
        for ledger in ("actual_event","analytic230"):
            for method in ("fg","fgh"):
                raw=value["raw_"+ledger][method];corrected=value[ledger][method]
                difference=deduction if method=="fgh" else 0
                assert raw["calibration_flops"]==calibrations[method]
                assert corrected["calibration_flops"]==calibrations[method]-difference
                assert corrected["complete_added_flops"]==raw["complete_added_flops"]-difference
                assert corrected["percent_of_once_rebuild"]==pytest.approx(
                    100*corrected["complete_added_flops"]/raw["once_rebuild_flops"])
                assert corrected["calibration_accounting"]==costs.CALIBRATION_ACCOUNTING
                for field in ("release_preparation_flops","extra_native_writer_flops","shared_graph_setup_flops","once_rebuild_flops"):
                    assert corrected[field]==raw[field]
                read_fields=("extra_read_flops",) if ledger=="actual_event" else (
                    "extra_candidate_read_flops","actual_behavior_read_extra_flops")
                assert all(corrected[field]==raw[field] for field in read_fields)
                correction=value["calibration_cost_corrections"][method]
                assert correction["shared_row_normalization_deduction_flops"]==difference
                assert correction["checkpoint_sha256"]==bindings[key][
                    "full_h_checkpoint_sha256" if method=="fgh" else "trained_fg_checkpoint_sha256"]
        assert value["calibration_cost_corrections"]["fgh"]["shared_row_normalization_deduction_flops"]>0
        assert value["analytic230"]["fg"]["users"]==2000
        assert all(row["parameter_device"]=="meta" for row in value["model_geometry"])
        for field in ("full_h_checkpoint","trained_fg_checkpoint"):
            assert sha256(Path(bindings[key][field]))==bindings[key][field+"_sha256"]
    with (output/"costs.csv").open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==60 and {r["ledger"] for r in rows}=={"actual_event","analytic230"}
    for row in rows:
        method,ledger=row["method"],row["ledger"]
        difference=deduction if method=="fgh" else 0
        value=result["cost_bindings"][row["binding_id"]][ledger][method]
        assert int(row["raw_calibration_flops"])==calibrations[method]
        assert int(row["calibration_normalization_deduction_flops"])==difference
        assert int(row["calibration_flops"])==calibrations[method]-difference
        assert int(row["complete_added_flops"])==int(row["raw_complete_added_flops"])-difference
        assert int(row["complete_added_flops"])==value["complete_added_flops"]
        assert float(row["percent_of_once_rebuild"])==value["percent_of_once_rebuild"]
        assert row["calibration_accounting"]==costs.CALIBRATION_ACCOUNTING
