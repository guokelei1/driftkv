"""Final selection retains original recipes and rejects context/typed drift."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import select_probe_calibration_followup_common as selector
from tests.experiments.ch3_motivation.test_select_probe_calibration_common import candidate
from tests.experiments.ch3_motivation.test_validate_probe_calibration_followup_common import fixture
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_actual_phase_b_row_keeps_init_identity_step0_and_identical_contexts(tmp_path,monkeypatch):
    fx=fixture(tmp_path,monkeypatch,n=128,width=256)
    point=selector.followup.evaluate_weight(fx.contexts,fx.spec,fx.training)
    contexts={rule:context.identity for rule,context in fx.contexts.items()}
    settings=selector._check_point(point,fx.spec,fx.training,contexts,False)
    assert settings["fit_users"]==128 and settings["width"]==256
    assert settings["h_initialization"]=="balanced" and point["step_zero"]
    changed=copy.deepcopy(point);changed["contexts"]["gap_mix50"]["common_validation_sha256"]="another-problem"
    with pytest.raises(ValueError,match="different common fingerprint"):
        selector._check_point(changed,fx.spec,fx.training,contexts,False)
    changed=copy.deepcopy(point);changed["training_identity"]["h_initialization"]="standard"
    with pytest.raises(ValueError,match="actual typed checkpoint"):
        selector._check_point(changed,fx.spec,fx.training,contexts,False)


def test_matched_failure_keeps_measurement_but_blocks_both_control_recipes(tmp_path,monkeypatch):
    fx=fixture(tmp_path,monkeypatch,n=256,mode="standard")
    point=selector.followup.evaluate_weight(fx.contexts,fx.spec,fx.training)
    point.update(selection_eligible=False,matched_initialization_status="failed")
    contexts={rule:context.identity for rule,context in fx.contexts.items()}
    assert selector._check_point(point,fx.spec,fx.training,contexts,True)["h_initialization"]=="standard"
    with pytest.raises(ValueError,match="eligibility differs"):
        selector._check_point(point,fx.spec,fx.training,contexts,False)
    assert point["mean_context_loss"]>=0  # A failed matched control does not erase the measured losses.


def test_final_global_one_percent_retains_old_driver_law_and_tie_rule():
    old=candidate("legacy/law64",1.005,users=64,cost=1)
    old["settings"].update(fitting_driver="run_probe_temporal.py",time_law_source_users=64)
    new=candidate("calibration_followup/balanced_h_init",1.,users=256,cost=100)
    new["settings"].update(fitting_driver="run_probe_calibration_init.py",time_law_source_users=256,h_initialization="balanced")
    result=selector.choose([old,new],True,True,["failed-preserved.pt"],{},old)
    assert result["recommendation"]["recipe"]=="legacy/law64"
    assert result["recommendation"]["settings"]["time_law_source_users"]==64
    assert result["recommendation"]["fitting_driver"]=="run_probe_temporal.py"
    assert result["phase_b_completed"] and result["final_candidate_for_three_seeds"] and result["needs_failure_review"]
    assert result["expected_weights"]==475 and result["expected_context_cells"]==1425
    assert selector.choose([old,new],True,False,[],{},old)["recommendation"] is None
    out=candidate("outside-fixed-set",1.011,users=32)
    assert selector.choose([old,new,out],True,True,[],{},old)["recommendation"]["recipe"]=="legacy/law64"
    cheaper=candidate("new-cheaper",1.009,users=64,cost=.5)
    assert selector.choose([old,new,cheaper],True,True,[],{},old)["recommendation"]["recipe"]=="new-cheaper"


def test_matched_diagnostics_are_separate_and_use_complete_five_edge_means():
    groups=[]
    for name,loss in (("reference",2.),("fit64",1.5),("failed",None)):
        groups.append(dict(recipe="calibration_followup/"+name,method_label="fgh",seed=17,fit_users=64,width=128,
            complete_five_edges=loss is not None,selection_eligible=loss is not None,mean_common_loss=loss,
            common_loss_by_edge={str(edge):loss for edge in selector.EDGES} if loss is not None else {},
            step_zero_edges=[5],failed_edges=[3] if loss is None else [],pending_edges=[],
            complete_calibration_flops_estimate_sum=100,settings=dict(h_initialization="standard")))
    result=selector.matched_diagnostics(groups,{1:dict(status="verified_equal")})
    byname={row["recipe"]:row for row in result["recipes"]}
    assert byname["calibration_followup/fit64"]["mean_common_loss_minus_reference"]==-.5
    assert set(byname["calibration_followup/fit64"]["common_loss_minus_reference_by_edge"].values())=={-.5}
    assert byname["calibration_followup/failed"]["mean_common_loss_minus_reference"] is None
    assert byname["calibration_followup/failed"]["failed_edges"]==[3]


def test_final_rollup_retains475_rows_and_pending_phase_b_inventory(tmp_path,monkeypatch):
    common,followup=tmp_path/"common.json",tmp_path/"followup.json"
    common.write_text("{}");followup.write_text("{}")
    contexts={rule:dict(token="same",query_delta_rule=rule) for rule in selector.followup.RULES}
    base=dict(points=[dict(checkpoint=f"/old/{i}.pt",validation=dict(status="missing")) for i in range(370)],
        groups=[],summary_sha256={},observed_weights=0,common_contexts_by_edge={"1":contexts},
        selection=dict(recommendation=None,all_five_edges_complete=False,sampling_phase_a={}))
    delegated=[]
    def previous(*args):delegated.append(args);return base
    monkeypatch.setattr(selector.previous,"rollup",previous)
    def load_plan(path,common,edge):
        prepared=[]
        for recipe in ("reference","balanced_h_init","fit64","fit128","lr1e5","width256","cap1024"):
            for stage in selector.followup.STAGES:
                spec=dict(recipe=recipe,edge=edge,seed=17,method_label=stage,
                    checkpoint=str((tmp_path/recipe/f"edge_{edge}/{stage}.pt").resolve()))
                prepared.append((spec,{}))
        return dict(expected_weights=105),dict(selection={}),prepared,{}
    monkeypatch.setattr(selector.followup,"load_plan",load_plan)
    monkeypatch.setattr(selector.previous.old,"_check_contexts",lambda summary,*args:summary["contexts"])
    prepared=load_plan(None,None,1)[2]
    points=[dict(spec,status="pending",contexts={},selection_eligible=False) for spec,_ in prepared]
    folder=tmp_path/"edge_1";folder.mkdir()
    config=dict(format=selector.followup.FORMAT,mode="common_validation",edge=1,contexts=contexts,
        followup_plan_sha256=sha256(followup),common_plan_sha256=sha256(common),frozen_sha256={})
    config_path=folder/"configuration.json";config_path.write_text(json.dumps(config))
    summary=dict(format=selector.followup.FORMAT,edge=1,followup_plan_sha256=sha256(followup),common_plan_sha256=sha256(common),
        configuration_sha256=sha256(config_path),contexts=contexts,points=points,status="waiting_weights",pending=21,validated=0,failed=0,
        paired_initialization=selector.followup.check_paired_predecessors(prepared,points))
    path=folder/"summary.json";path.write_text(json.dumps(summary))
    result=selector.rollup(tmp_path/"sampling.json",common,followup,[],[],[path])
    assert len(delegated)==1 and len(result["points"])==475
    assert result["observed_weights"]==21 and result["phase_b_expected_weights"]==105
    assert result["selection"]["recommendation"] is None and not result["selection"]["phase_b_completed"]
    assert len(result["matched_diagnostics"]["recipes"])==7
    assert all(row["pending_edges"]==selector.EDGES for row in result["matched_diagnostics"]["recipes"])
    json.dumps(result,allow_nan=False)
    summary["contexts"]=copy.deepcopy(contexts);summary["contexts"]["cutover"]["token"]="different"
    config["contexts"]=summary["contexts"];config_path.write_text(json.dumps(config))
    summary["configuration_sha256"]=sha256(config_path);path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="previous370 use different"):
        selector.rollup(tmp_path/"sampling.json",common,followup,[],[],[path])

