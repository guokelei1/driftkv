"""Formal plans bind real causal manifests without choosing a recipe or GPU."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    calibration_sampling as sampling, plan_probe_calibration_formal as planner,
    plan_probe_quality as quality,
)
from experiments.ch3_motivation.code.calibration_gaps import fit_gap_law
from experiments.ch3_motivation.code.probe_initialization import initialization_spec
from experiments.ch3_motivation.code.select_probe_common import sha256, write_json


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    folder=tmp_path_factory.mktemp("formal_sampling_assets")
    root=folder/"fit256"
    fit=list(range(1,513));val=list(range(1001,1017))
    population=dict(fit=fit,validation=val,main=list(range(10000,12000)),confirmation=list(range(20000,22000)))
    write_json(root/"population.json",population)
    manifests={}
    for edge in planner.EDGES:
        directory=root/f"edge_{edge}";directory.mkdir()
        for name in ("history","requests"):(directory/f"{name}.pack").write_text(f"{name}edge{edge}")
        cutoff=(20+(edge-1)*14)*86400
        binding=dict(population_sha256=sha256(root/"population.json"),settings=dict(edge=edge,release_day=cutoff//86400),
            **{kind:dict(path=f"{kind}.pack",sha256=sha256(directory/f"{kind}.pack")) for kind in ("history","requests")})
        write_json(directory/"inputs.json",binding)
        candidate=dict(population_sha256=sha256(root/"population.json"),known_items=64,
            fitting_pool_uids=fit[:256],validation_uids=val,
            recent_items={str(uid):list(range(1,17)) for uid in fit[:256]+val})
        write_json(directory/"calibration_feedback_candidates.json",candidate)
        law=fit_gap_law({uid:[cutoff-20] for uid in fit[:256]},
            {uid:[cutoff-60,cutoff-30] for uid in fit[:256]},fit[:256],cutoff)
        manifest=dict(schema_version=1,status="prepared",edge=edge,source_cutoff=cutoff,candidate_count=16,
            candidate_seed=17,gap_seed=17,catalog_source_users=256,time_law_source_users=256,
            candidate_catalog_source_fit_uids=fit[:256],time_law_source_fit_uids=fit[:256],target_fit_uids=fit[:256],
            validation_uids=val,gap_law=law,time_rule="strict-past feedback gap",candidate_order="feedback prefix then uniform",
            views=sampling.make_sampling_views(fit[:256],val,candidate,{uid:30 for uid in fit[:256]+val},law,edge=edge,cutoff=cutoff),
            cpu_seconds=dict(total=.01),source_sha256={},sourcebinding=dict(input_root=str(root),
                population_sha256=sha256(root/"population.json"),sha256=sha256(directory/"inputs.json"),
                candidate_sha256=sha256(directory/"calibration_feedback_candidates.json"),
                history_path=str(directory/"history.pack"),history_sha256=sha256(directory/"history.pack")))
        path=folder/f"manifest_edge{edge}.json";write_json(path,manifest);manifests[edge]=path
    scope=folder/"scope.json"
    write_json(scope,json.loads((ROOT/"experiments/ch3_motivation/configs/formal_quality_scope.json").read_text()))
    return SimpleNamespace(root=root,population=population,manifests=manifests,scope=scope)


def selection(tmp_path,assets,*,driver="run_probe_calibration.py",mode="balanced",rule="gap_mix50",n=128):
    population=assets.population
    config=dict(num_layers=10,num_heads=10,head_dim=32,item_dim=320,hidden_width=128,depth=1,
                include_item=True,normalize=True,query_chunk=16,row_chunk=1024)
    source_names=[f"experiments/ch3_motivation/code/{driver}","experiments/ch3_motivation/code/probes.py"]
    if driver.endswith("_init.py"):source_names.append("experiments/ch3_motivation/code/probe_initialization.py")
    numerical={name:sha256(ROOT/name) for name in source_names}
    source_driver_sha=sha256(ROOT/"experiments/ch3_motivation/code"/driver)
    points={};provenance={};representative=None
    for edge in planner.EDGES:
        binding=planner._input_binding(assets.root,edge)
        _,prov=sampling.load_sampling_manifest(assets.manifests[edge],assets.root,edge,.75,rule,fit_users=n)
        identity=dict(variant="full_item",width=128,depth=1,fit_users=n,queries=16,learning_rate=.00003,
            steps=2048,patience=1024,validation_every=8,read_weight=1.,score_weight=1.,candidate_source="feedback",
            recent_fraction=.75,candidate_seed=17,row_chunk=1024,query_chunk=16,weight_decay=.0001,gradient_clip=5.,
            edge=edge,seed=17,models={f"version{edge-1}":f"model{edge-1}",f"version{edge}":f"model{edge}"},
            fit_uids=population["fit"][:n],validation_users=population["validation"],stages="f,fg,fgh",controls=False,
            input_binding_sha256=binding["input_binding_sha256"],population_sha256=binding["population_sha256"],
            calibration_candidate_sha256=binding["candidate_file_sha256"],numerical_sources=numerical,
            fitting_driver=driver,fitting_driver_sha256=source_driver_sha,query_delta_rule=rule,
            sampling_manifest=str(assets.manifests[edge]),sampling_manifest_sha256=sha256(assets.manifests[edge]),
            gap_manifest=str(assets.manifests[edge]),gap_manifest_sha256=sha256(assets.manifests[edge]),
            time_rule=prov["time_rule"],source_cutoff=prov["source_cutoff"],candidate_order=prov["candidate_order"],
            query_delta_layout="[B,16]",candidate_catalog_source_users=256,time_law_source_users=256,
            gradient_fitting_users=n,teacher_fitting_users=n,fitting_uses_future_feedback=False)
        if driver.endswith("_init.py"):
            spec=initialization_spec(32,384,mode)
            identity.update(h_initialization=mode,h_initialization_spec=spec,initialization_helper_sha256=spec["helper_sha256"])
        metadata=dict(identity=identity,method_label="fgh",seed=17,best_step=0,completed_steps=1024,
            training_users=n,validation_users=16,teacher_preparation_users=n+16,actual_teacher_users_captured=n,
            actual_validation_users_captured=16,fit_pool_loss_scales=dict(users=n,source="fitting_users_only",
                version="fit_pool_equal_user_length_normalized_v1",validation_uses_fitting_scales=True))
        path=tmp_path/f"edge_{edge}/fgh.pt";path.parent.mkdir(parents=True)
        # Plan/audit fixtures exercise typed metadata and exact file bindings;
        # they are never loaded by a model or used as quality weights.
        torch.save(dict(format="ch3_probe_variant_v1",variant="full_item",stage="fgh",config=config,
                        metadata=metadata,state_dict={}),path)
        points[str(edge)]=dict(checkpoint=str(path),checkpoint_sha256=sha256(path),status="validated",best_step=0,completed_steps=1024)
        provenance[str(edge)]={key:identity[key] for key in planner.SAMPLING_FIELDS if key not in ("fitting_driver","fitting_driver_sha256")}
        if edge==1:representative=dict(identity,input_root=str(assets.root))
    chosen=dict(recipe="synthetic_selected",method_label="fgh",seed=17,variant="full_item",width=128,depth=1,
        fit_users=n,edges=planner.EDGES,correction_seeds=planner.FORMAL_SEEDS,step_zero_edges=[1,2,3,4,5],
        settings=representative,checkpoints_by_edge=points,sampling_provenance_by_edge=provenance,
        fitting_driver=driver,fitting_driver_sha256=source_driver_sha,query_delta_rule=rule)
    value=dict(status="recommendation_for_review",all_five_edges_complete=True,selection_seed=17,recommendation=chosen)
    path=tmp_path/"selection.json";write_json(path,value)
    return SimpleNamespace(path=path,value=value)


@pytest.mark.parametrize("driver,rule",[("run_probe_calibration.py","cutover"),("run_probe_calibration_init.py","gap_empirical100")])
def test_real_manifest_main_curve_grids_and_quality_schema(tmp_path,assets,driver,rule):
    chosen=selection(tmp_path,assets,driver=driver,rule=rule)
    result=planner.generate(chosen.path,assets.scope)
    assert result["status"]=="prepared_for_root_review" and result["gpu_launch_performed"] is False
    assert result["recipe_selection_performed"] is False
    assert len(result["main_jobs"])==15 and len(result["curve_jobs"])==45
    assert result["job_counts"]["main_stage_weights_with_controls"]==75
    assert result["confirmation_quality"]["users"]==2000 and result["confirmation_quality"]["seeds"]==[17]
    for job in result["main_jobs"]:
        edge=str(job["edge"])
        assert job["controls"] is True and job["stages"]=="f,fg,fgh" and job["fit_users"]==128
        assert job["fitting_driver"]==driver and job["gap_rule"]==job["query_delta_rule"]==rule
        assert job["sampling_manifest_sha256"]==result["main_input_bindings"][edge]["sampling_manifest_sha256"]
        assert job["sampling_manifest"]==job["gap_manifest"]==str(assets.manifests[job["edge"]])
        assert job["gradient_fitting_users"]==job["teacher_fitting_users"]==128
        assert job["candidate_catalog_source_users"]==job["time_law_source_users"]==256
        assert ("h_initialization" in job)==driver.endswith("_init.py")
    for job in result["curve_jobs"]:
        assert job["seed"] in (17,23,29) and job["fit_users"] in (64,128,256)
        assert job["gradient_fitting_users"]==job["teacher_fitting_users"]==job["fit_users"]
        assert job["stages"]=="f" and not job["controls"] and job["steps"]==2048 and job["patience"]==0
        assert job["sampling_manifest_sha256"]==result["main_input_bindings"][str(job["edge"])]["sampling_manifest_sha256"]
    policy=result["population_curve_policy"]
    assert policy["candidate_catalog_source_users"]==policy["time_law_source_users"]==256
    assert "own-N" in policy["training_statistics"]
    assert "own-plus-other" in policy["feedback_fraction_semantics"]
    assert result["selected_h_step_zero_edges"]==[1,2,3,4,5]
    # The actual quality planner accepts the generated fields. With missing
    # formal weights it writes waiting_weights, never pretends they are ready.
    out=tmp_path/"formal";write_json(out/"plan.json",result)
    prepared=quality.generate(out/"plan.json",tmp_path/"future_main",tmp_path/"future_curve",tmp_path/"quality")
    assert prepared["status"]=="waiting_weights"
    assert {name:row["points"] for name,row in prepared["families"].items()}==dict(main=75,confirmation=15,f_population_curve=45)
    assert {Path(p).name for p in result["planner_sources_sha256"]}>={"plan_probe_calibration_formal.py","sampling_identity.py","calibration_sampling.py","calibration_gaps.py","prepare_candidates.py","probe_initialization.py","run_probe_calibration.py","run_probe_calibration_init.py"}
    # Check generated identities through the real quality gate, including a
    # different training seed/control flag and a different curve teacher N.
    for job in (next(j for j in result["main_jobs"] if j["edge"]==2 and j["seed"]==23),
                next(j for j in result["curve_jobs"] if j["edge"]==2 and j["seed"]==29 and j["fit_users"]==256)):
        prototype=torch.load(chosen.value["recommendation"]["checkpoints_by_edge"]["2"]["checkpoint"],map_location="cpu",weights_only=False)
        metadata=prototype["metadata"];identity=metadata["identity"]
        n=job["fit_users"];stage="f" if job["stages"]=="f" else "fgh"
        identity.update({key:job[key] for key in quality.FIELDS})
        identity.update(seed=job["seed"],fit_users=n,fit_uids=assets.population["fit"][:n],gradient_fitting_users=n,teacher_fitting_users=n)
        metadata.update(method_label=stage,seed=job["seed"],training_users=n,teacher_preparation_users=n+16,
            actual_teacher_users_captured=n,completed_steps=job["steps"])
        metadata["fit_pool_loss_scales"]["users"]=n
        prototype["stage"]=stage
        path=tmp_path/f"schema_seed{job['seed']}_{stage}.pt";torch.save(prototype,path)
        checked=quality.audit_weight(path,job,result["main_input_bindings"]["2"],stage)
        assert checked["checkpoint_sha256"]==sha256(path)


@pytest.mark.parametrize("fault",["manifest_from_edge1","mode","driver_sha","fit_roles","recipe_lr"])
def test_wrong_selected_edge_or_recipe_rejected(tmp_path,assets,fault):
    chosen=selection(tmp_path,assets,driver="run_probe_calibration_init.py")
    if fault=="manifest_from_edge1":
        chosen.value["recommendation"]["sampling_provenance_by_edge"]["2"]=copy.deepcopy(chosen.value["recommendation"]["sampling_provenance_by_edge"]["1"])
        write_json(chosen.path,chosen.value)
    else:
        ref=chosen.value["recommendation"]["checkpoints_by_edge"]["2"]
        path=Path(ref["checkpoint"]);payload=torch.load(path,map_location="cpu",weights_only=False)
        identity=payload["metadata"]["identity"]
        if fault=="mode":identity["h_initialization"]="standard"
        elif fault=="driver_sha":identity["fitting_driver_sha256"]="0"*64
        elif fault=="fit_roles":identity["fit_uids"]=list(reversed(identity["fit_uids"]))
        else:identity["learning_rate"] = .00001
        torch.save(payload,path);ref["checkpoint_sha256"]=sha256(path);write_json(chosen.path,chosen.value)
    with pytest.raises(ValueError):planner.generate(chosen.path,assets.scope)


def test_pending_selection_and_missing_source_never_generate_jobs(tmp_path,assets):
    result=planner.generate(tmp_path/"missing_selection.json",assets.scope)
    assert result["status"]=="waiting_selection" and not result["main_jobs"] and not result["curve_jobs"]
    chosen=selection(tmp_path,assets)
    Path(chosen.value["recommendation"]["checkpoints_by_edge"]["5"]["checkpoint"]).unlink()
    result=planner.generate(chosen.path,assets.scope)
    assert result["status"]=="waiting_selected_weights" and not result["main_jobs"] and not result["curve_jobs"]


def test_curve_cannot_silently_change_catalog_law_root(tmp_path,assets):
    chosen=selection(tmp_path,assets)
    with pytest.raises(ValueError,match="retain the selected fixed256"):
        planner.generate(chosen.path,assets.scope,curve_input_root=tmp_path/"other")


def test_old_recipe_delegates_without_reinterpreting_driver(tmp_path,assets,monkeypatch):
    chosen=selection(tmp_path,assets)
    chosen.value["recommendation"]["fitting_driver"]="run_probe_temporal.py"
    write_json(chosen.path,chosen.value)
    observed=[]
    def legacy(*args,**kwargs):
        observed.append(args);return dict(status="prepared_for_root_review",main_jobs=["old"],curve_jobs=[],curve_plan_status="pending_temporal_curve_driver")
    monkeypatch.setattr(planner.plan_probe_temporal_formal,"generate",legacy)
    result=planner.generate(chosen.path,assets.scope)
    assert observed and result["main_jobs"]==["old"] and result["curve_plan_status"]=="pending_temporal_curve_driver"
