"""Fixed2000 natural membership and exact formal weights, without inference."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    common, plan_probe_natural_quality as planner, plan_probe_calibration_formal as formal,
)
from hstu_kvcache.common.artifacts import sha256
import tests.experiments.ch3_motivation.test_plan_probe_calibration_formal as fitting_fixture


@pytest.fixture
def natural(tmp_path,tmp_path_factory):
    assets=fitting_fixture.assets.__wrapped__(tmp_path_factory)
    users=assets.population["main"]  # Deliberate overlap: not independent confirmation.
    all_users=assets.population["fit"]+assets.population["validation"]+users
    models={}
    for version in range(6):
        path=tmp_path/f"models/v{version}.pt";path.parent.mkdir(exist_ok=True)
        path.write_text(f"model binding fixture{version}");models[version]=(str(path),sha256(path))
    strata=tmp_path/"strata.json"
    common.write_json(strata,dict(subgroups_by_edge={str(e):dict(long_feedback_active=dict(first_2000_uid_hashes=users)) for e in range(1,6)}))
    anchors=tmp_path/"anchors"
    for edge in range(1,6):
        directory=assets.root/f"edge_{edge}";binding_path=directory/"inputs.json"
        binding=json.loads(binding_path.read_text());cutoff=binding["settings"]["release_day"]*86400
        history=directory/"history.pack";requests=directory/"requests.pack"
        # Small structural inputs validate the planner's frozen membership and
        # history bindings; no feature thresholds or model scores are derived.
        histories=SimpleNamespace(rows={u:(np.array([cutoff-30]),np.array([1]),np.array([1])) for u in all_users})
        common.save_histories(history,histories,all_users)
        pq.write_table(pa.Table.from_pylist([dict(uid=u,query_timestamp=cutoff+1,raw_item_id=1,item_idx=1,label=0) for u in users[:2]]),requests)
        binding.update(history=dict(path=history.name,sha256=sha256(history)),requests=dict(path=requests.name,sha256=sha256(requests)),
            candidate_panels=dict(long_feedback_active=users))
        common.write_json(binding_path,binding)
        manifest=json.loads(assets.manifests[edge].read_text())
        manifest["sourcebinding"].update(sha256=sha256(binding_path),history_sha256=sha256(history))
        common.write_json(assets.manifests[edge],manifest)
        auc=dict(old_full=.5,reuse=.6,rebuild=.4 if edge==2 else .7)
        qualifies=edge!=2
        qualification=dict(new_rebuild_minus_old_full=auc["rebuild"]-auc["old_full"],
            new_rebuild_minus_reuse=auc["rebuild"]-auc["reuse"],point_qualifies=qualifies,uid95_qualifies=qualifies)
        model_map=dict(models[e] for e in (edge-1,edge))
        folder=anchors/f"edge_{edge}";summary_path=folder/"summary.json"
        common.write_json(summary_path,dict(status="complete",edge=edge,panel="long_feedback_active",quality_users=2000,
            quality_uids=users,quality_uids_sha256=hashlib.sha256(json.dumps(users).encode()).hexdigest(),
            auc=auc,qualification=qualification,**qualification,
            new_rebuild_minus_old_full_ci95=[.1,.3] if qualifies else [-.2,0.],
            new_rebuild_minus_reuse_ci95=[.01,.2] if qualifies else [-.3,-.1],
            identity=dict(experiment="fixed_natural_three_anchors",input_binding_sha256=sha256(binding_path),models=model_map)))
        common.write_json(folder/"scenario_manifest.json",dict(schema_version=1,edge=edge,uids=users,
            roles=dict(quality=users,e6=users[:2]),input_binding=dict(path=str(binding_path),sha256=sha256(binding_path)),
            history=dict(path=str(history),sha256=sha256(history)),requests=dict(path=str(requests),sha256=sha256(requests)),
            source_population=dict(path=str(assets.root/"population.json"),sha256=sha256(assets.root/"population.json")),
            qualification=dict(summary_path=str(summary_path),summary_sha256=sha256(summary_path),scope="fixed pre-release cohort",**qualification),
            selection=dict(panel_name="long_feedback_active",definition=planner.DEFINITION,
                strata_manifest_path=str(strata),strata_manifest_sha256=sha256(strata))))
    chosen=fitting_fixture.selection(tmp_path/"selected",assets)
    for edge in range(1,6):
        ref=chosen.value["recommendation"]["checkpoints_by_edge"][str(edge)]
        path=Path(ref["checkpoint"]);payload=torch.load(path,map_location="cpu",weights_only=False)
        payload["metadata"]["identity"]["models"]=dict(models[e] for e in (edge-1,edge))
        if edge==1:chosen.value["recommendation"]["settings"]["models"]=payload["metadata"]["identity"]["models"]
        torch.save(payload,path);ref["checkpoint_sha256"]=sha256(path)
    common.write_json(chosen.path,chosen.value)
    plan=formal.generate(chosen.path,assets.scope);main_root=tmp_path/"main_weights"
    for job in (j for j in plan["main_jobs"] if j["seed"]==17):
        source=chosen.value["recommendation"]["checkpoints_by_edge"][str(job["edge"])]["checkpoint"]
        for method in planner.METHODS:
            # Typed metadata only: no fitting/inference weights or scores are
            # manufactured by this planner fixture.
            payload=torch.load(source,map_location="cpu",weights_only=False)
            payload["stage"]=method;payload["metadata"]["method_label"]=method
            payload["metadata"]["identity"].update({key:job[key] for key in planner.quality.FIELDS})
            path=planner.quality.checkpoint_path(main_root,job,method);path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(payload,path)
    plan_path=tmp_path/"formal.json";common.write_json(plan_path,plan)
    return SimpleNamespace(assets=assets,anchors=anchors,plan=plan,plan_path=plan_path,main_root=main_root,users=users)


def test_all15_exact_weights_negative_anchor_and_copied_calibration_root(natural,tmp_path):
    copied=tmp_path/"same_binding_new_root";shutil.copytree(natural.assets.root,copied)
    for binding in natural.plan["main_input_bindings"].values():binding["input_root"]=str(copied)
    common.write_json(natural.plan_path,natural.plan)
    result=planner.generate(natural.plan_path,natural.main_root,anchors_root=natural.anchors)
    assert result["status"]=="prepared_for_root_review" and len(result["jobs"])==15
    assert result["gpu_launch_performed"] is False and result["independent_confirmation"] is False
    assert result["seeds"]==[17] and result["main_and_confirmation_plans_modified"] is False
    assert result["validated_edges"]==[1,2,3,4,5] and result["available_weights"]==15
    assert not result["edges"]["2"]["scenario"]["point_qualifies"]
    assert all(row["scenario"]["users"]==2000 and row["scenario"]["overlap_with_main"]==2000 for row in result["edges"].values())
    assert {j["name"] for j in result["jobs"]}=={f"natural_edge_{e}_seed_17_{m}" for e in range(1,6) for m in planner.METHODS}
    allowed={"edge","checkpoint","name","input_root","panel","population_file","backend"}
    assert all(set(job)==allowed and job["panel"]=="natural_active" and job["backend"]=="projected" for job in result["jobs"])
    assert all(Path(job["population_file"])==natural.anchors/f"edge_{job['edge']}/scenario_manifest.json" for job in result["jobs"])


def test_partial_anchors_or_weights_never_release_subset_jobs(natural):
    (natural.anchors/"edge_5/scenario_manifest.json").unlink()
    job=next(j for j in natural.plan["main_jobs"] if j["edge"]==4 and j["seed"]==17)
    planner.quality.checkpoint_path(natural.main_root,job,"fgh").unlink()
    result=planner.generate(natural.plan_path,natural.main_root,anchors_root=natural.anchors)
    assert result["status"]=="waiting_prerequisites" and result["jobs"]==[]
    assert len(result["planned_jobs"])==15 and len(result["missing_weights"])==len(result["missing_anchor_manifests"])==1
    assert result["validated_edges"]==[1,2,3]


@pytest.mark.parametrize("fault",["models","uids"])
def test_mismatched_models_or_rescreened_uids_rejected(natural,fault,tmp_path):
    if fault=="models":
        job=next(j for j in natural.plan["main_jobs"] if j["edge"]==3 and j["seed"]==17)
        path=planner.quality.checkpoint_path(natural.main_root,job,"fg")
        payload=torch.load(path,map_location="cpu",weights_only=False)
        payload["metadata"]["identity"]["models"]={"wrong-model":"0"*64}
        torch.save(payload,path)
    else:
        manifest_path=natural.anchors/"edge_3/scenario_manifest.json"
        manifest=json.loads(manifest_path.read_text());strata=Path(manifest["selection"]["strata_manifest_path"])
        value=json.loads(strata.read_text());value["subgroups_by_edge"]["3"]["long_feedback_active"]["first_2000_uid_hashes"]=list(reversed(natural.users))
        common.write_json(strata,value)
        # Updating this recorded hash cannot authorize different frozen UIDs.
        for edge in range(1,6):
            path=natural.anchors/f"edge_{edge}/scenario_manifest.json";data=json.loads(path.read_text())
            data["selection"]["strata_manifest_sha256"]=sha256(strata);common.write_json(path,data)
    result=planner.generate(natural.plan_path,natural.main_root,anchors_root=natural.anchors)
    assert result["status"]=="failed_prerequisites" and not result["jobs"]
    assert any(row["edge"]==3 for row in result["failed_prerequisites"])


def test_missing_selection_does_not_make_quality_jobs(tmp_path):
    path=tmp_path/"formal.json";common.write_json(path,dict(status="waiting_selection"))
    result=planner.generate(path,tmp_path/"weights",anchors_root=tmp_path/"anchors")
    assert result["status"]=="waiting_selection" and result["jobs"]==[] and not result["gpu_launch_performed"]
