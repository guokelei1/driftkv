"""Combined search keeps strict common cells, global ties and incomplete rows."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    select_probe_calibration_common as selector, validate_probe_calibration_common as validator,
)
from tests.experiments.ch3_motivation.test_validate_probe_calibration_common import setup
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def candidate(name,loss,*,users=256,width=128,cost=100.,eligible=True):
    return dict(recipe=name,variant="full_item",method_label="fgh",seed=17,selection_eligible=eligible,
        fit_users=users,width=width,mean_common_loss=loss,complete_calibration_flops_estimate_sum=cost,
        step_zero_edges=[1],settings=dict(fitting_driver="run_probe_calibration.py",fitting_driver_sha256="sha",query_delta_rule="cutover"))


def test_real_typed_new_point_passes_only_on_identical_context_fingerprints(tmp_path,monkeypatch):
    fx=setup(tmp_path,monkeypatch)
    point=validator.evaluate_sampling_weight(fx.contexts,fx.spec,fx.training)
    contexts={rule:context.identity for rule,context in fx.contexts.items()}
    settings=selector._check_point(point,fx.spec,fx.training,contexts)
    assert settings["sampling_manifest_sha256"]==fx.metadata["identity"]["sampling_manifest_sha256"]
    assert settings["time_law_source_users"]==settings["gradient_fitting_users"]==256
    changed=copy.deepcopy(point)
    changed["contexts"]["gap_mix50"]["common_validation_sha256"]="another-problem"
    with pytest.raises(ValueError,match="different common fingerprint"):
        selector._check_point(changed,fx.spec,fx.training,contexts)
    changed=copy.deepcopy(point)
    changed["best_step"]=1
    with pytest.raises(ValueError,match="typed checkpoint"):
        selector._check_point(changed,fx.spec,fx.training,contexts)


def test_verified_edge_fields_only_are_removed_and_initialization_is_preserved():
    identity=dict(edge=1,models={"v0":"sha"},input_binding_sha256="binding",calibration_candidate_sha256="candidate",
        sampling_manifest="/edge_1/sampling.json",sampling_manifest_sha256="sampling",gap_manifest="/edge_1/sampling.json",
        gap_manifest_sha256="sampling",source_cutoff=231*86400,recent_fraction=.75,query_delta_rule="gap_mix50",
        time_law_source_users=256,gradient_fitting_users=256,fitting_driver_sha256="driver",
        numerical_sources={"helper":"sha"},init_mode="explicit",h_initialization_spec=dict(mode="zero_output"))
    normalized=selector.normalized_sampling_identity(identity)
    assert not selector.SAMPLING_EDGE_FIELDS&set(normalized)
    for key in ("recent_fraction","query_delta_rule","time_law_source_users","gradient_fitting_users",
                "fitting_driver_sha256","numerical_sources","init_mode","h_initialization_spec"):
        assert normalized[key]==identity[key]


def test_fixed_one_percent_set_is_global_not_transitive():
    groups=[candidate("minimum",1.,users=256),candidate("near",1.009,users=128),candidate("outside",1.018,users=64)]
    result=selector.choose(groups,True,[],{})
    assert result["recommendation"]["recipe"]=="near"
    assert set(result["within_one_percent"])=={"minimum","near"}
    assert result["fixed_one_percent_threshold"]==1.01
    assert "PhaseB" in result["selection_scope"] and not result["development_seed_robustness_established"]


def test_tie_order_is_population_width_then_complete_calibration_cost():
    groups=[candidate("wide-cheap",1.,users=64,width=256,cost=1),candidate("narrow",1.005,users=64,width=128,cost=100),
            candidate("narrow-cheaper",1.009,users=64,width=128,cost=50),candidate("larger-pool",1.,users=128,width=128,cost=1)]
    result=selector.choose(groups,True,["failed-other-recipe.pt"],{})
    assert result["recommendation"]["recipe"]=="narrow-cheaper" and result["needs_failure_review"]
    assert result["recommendation"]["step_zero_edges"]==[1]
    assert result["recommendation"]["correction_seeds"]==[17,23,29]


def test_zero_minimum_only_ties_zero_and_not_ready_never_selects():
    groups=[candidate("zero",0.,users=256),candidate("positive",1e-9,users=64)]
    assert selector.choose(groups,True,[],{})["recommendation"]["recipe"]=="zero"
    assert selector.choose(groups,False,[],{})["recommendation"] is None


def phase_a_candidates():
    groups=[candidate(f"sampling/view{i}",1+i*.001,cost=100-i) for i in range(6)]
    for group in groups:
        group.update(complete_five_edges=True,validated_edges=selector.EDGES,failed_edges=[],pending_edges=[],
            checkpoints_by_edge={str(edge):dict(checkpoint=f"/edge_{edge}/{group['recipe']}.pt",checkpoint_sha256="sha")
                for edge in selector.EDGES},sampling_provenance_by_edge={str(edge):dict(sampling_manifest_sha256=f"sha{edge}")
                for edge in selector.EDGES})
        group["settings"].update(candidate_catalog_source_users=256,time_law_source_users=256)
    return groups


def test_sampling_phase_a_keeps_raw_best_separate_from_global_one_percent_choice():
    groups=phase_a_candidates()
    legacy=candidate("old/law64",.99,users=64,cost=1)
    legacy["settings"].update(fitting_driver="run_probe_temporal.py",time_law_source_users=64)
    assert selector.choose([legacy]+groups,True,[],{})["recommendation"]["recipe"]=="old/law64"
    selected=selector.choose_sampling_inputs(groups,True,[])
    assert selected["status"]=="recommendation_for_review" and selected["all_five_edges_complete"]
    assert selected["selection_seed"]==17 and selected["expected_input_recipes"]==6
    assert selected["recommendation"]["recipe"]=="sampling/view0"
    assert [group["recipe"] for group in selected["ranked_recipes"]]==[f"sampling/view{i}" for i in range(6)]
    assert len(selected["recipes"])==6 and selected["recommendation"]["step_zero_edges"]==[1]
    assert selected["recommendation"]["sampling_provenance_by_edge"]["5"]["sampling_manifest_sha256"]=="sha5"
    assert selected["recommendation"]["correction_seeds"]==[17,23,29]


def test_sampling_phase_a_exact_ties_cost_name_and_incomplete_never_recommend():
    groups=phase_a_candidates()
    groups[0]["mean_common_loss"]=groups[1]["mean_common_loss"]
    assert selector.choose_sampling_inputs(groups,True,[])["recommendation"]["recipe"]=="sampling/view1"
    groups[0]["complete_calibration_flops_estimate_sum"]=groups[1]["complete_calibration_flops_estimate_sum"]
    assert selector.choose_sampling_inputs(groups,True,[])["recommendation"]["recipe"]=="sampling/view0"
    assert selector.choose_sampling_inputs(groups,False,[])["recommendation"] is None
    groups[4].update(selection_eligible=False,failed_edges=[3],mean_common_loss=None,complete_five_edges=False)
    result=selector.choose_sampling_inputs(groups,True,["failed.pt"])
    assert result["status"]=="incomplete_eligible_input_recipes" and result["recommendation"] is None
    assert result["needs_failure_review"] and len(result["recipes"])==6
    assert result["recipes"][4]["rank"] is None and result["recipes"][4]["failed_edges"]==[3]
    groups[4].update(selection_eligible=True,mean_common_loss=1.)
    groups[4]["settings"]["time_law_source_users"]=64
    with pytest.raises(ValueError,match="actual new law256"):
        selector.choose_sampling_inputs(groups,True,[])


def sampling_rows():
    rows=[]
    for edge in selector.EDGES:
        settings=dict(edge=edge,models={str(edge):"sha"},input_binding_sha256=str(edge),calibration_candidate_sha256=str(edge),
            sampling_manifest=f"/edge_{edge}/sampling.json",sampling_manifest_sha256=f"sampling{edge}",
            gap_manifest=f"/edge_{edge}/sampling.json",gap_manifest_sha256=f"sampling{edge}",source_cutoff=edge*86400,
            candidate_order="fixed",time_rule="strict-past",query_delta_layout="[B,16]",query_delta_rule="gap_mix50",
            candidate_catalog_source_users=256,time_law_source_users=256,gradient_fitting_users=256,teacher_fitting_users=256,
            fitting_driver="run_probe_calibration.py",fitting_driver_sha256="sha",input_root="/fit256")
        point=dict(status="validated",variant="full_item",config=dict(hidden_width=128,depth=1),fitting_users=256,
            mean_context_loss=2.,contexts={rule:dict(metrics=dict(loss=float(i+1))) for i,rule in enumerate(selector.old.RULES)},
            cumulative_calibration_flops_estimate=100,selection_eligible=True,step_zero=edge==5,checkpoint_sha256="sha",best_step=0,completed_steps=128)
        rows.append(dict(recipe="sampling",method_label="fgh",seed=17,edge=edge,checkpoint=f"/edge_{edge}/fgh.pt",
            validation=point,settings=settings,normalized_identity=selector.normalized_sampling_identity(settings),
            source_binding=dict(input_root="/fit256",edge=edge),sampling_view_key="feedback75_gap_mix50"))
    return rows


def test_group_requires_all_five_edges_and_keeps_each_manifest_binding():
    rows=sampling_rows()
    group=selector._group_sampling(rows)[0]
    assert group["complete_five_edges"] and group["mean_common_loss"]==2
    assert group["mean_loss_by_context"]==dict(cutover=1.,gap_mix50=2.,gap_empirical100=3.)
    assert group["complete_calibration_flops_estimate_sum"]==500 and group["step_zero_edges"]==[5]
    assert group["sampling_provenance_by_edge"]["5"]["sampling_manifest_sha256"]=="sampling5"
    assert group["sampling_provenance_by_edge"]["5"]["query_delta_rule"]=="gap_mix50"
    rows[2]["validation"]["status"]="failed"
    group=selector._group_sampling(rows)[0]
    assert not group["selection_eligible"] and group["mean_common_loss"] is None
    assert group["failed_edges"]==[3] and group["validated_edges"]==[1,2,4,5]


def test_group_rejects_non_edge_initialization_changes():
    rows=sampling_rows()
    rows[2]["settings"]["init_mode"]="another"
    rows[2]["normalized_identity"]=selector.normalized_sampling_identity(rows[2]["settings"])
    with pytest.raises(ValueError,match="non-edge fitting/init"):
        selector._group_sampling(rows)


def test_missing_plans_remain_pending_without_a_recommendation(tmp_path):
    result=selector.rollup(tmp_path/"missing_new.json",tmp_path/"missing_old.json",[],[])
    assert result["expected_weights"]==370 and result["expected_context_cells"]==1110
    assert result["points"]==[] and result["selection"]["recommendation"] is None
    assert result["selection"]["sampling_phase_a"]["recommendation"] is None


def test_combined_driver_retains_all370_rows_and_pending90_with_one_shared_context(tmp_path,monkeypatch):
    fitting_path,common_path=tmp_path/"new.json",tmp_path/"old.json"
    fitting_path.write_text("{}");common_path.write_text("{}")
    context={rule:dict(token="same",query_delta_rule=rule) for rule in selector.old.RULES}
    legacy=dict(summary_sha256={},common_contexts_by_edge={1:context},observed_weights=0,groups=[],
        selection=dict(all_five_edges_complete=False,missing_edges=[2,3,4,5]),
        points=[dict(checkpoint=f"/old/{i}.pt",validation=dict(status="missing")) for i in range(280)])
    calls=[]
    def old_rollup(path,summaries):
        calls.append((path,summaries))
        return legacy
    def plan(path,common,edge):
        prepared=[]
        for view in range(6):
            for stage in ("f","fg","fgh"):
                spec=dict(origin="sampling256_90",recipe=f"sampling/view{view}",edge=edge,seed=17,method_label=stage,
                    checkpoint=str((tmp_path/f"new/edge_{edge}/view_{view}/{stage}.pt").resolve()))
                prepared.append((spec,{}))
        return {},dict(selection={}),prepared,{}
    monkeypatch.setattr(selector.old,"rollup",old_rollup)
    monkeypatch.setattr(selector.sampling,"load_plan",plan)
    monkeypatch.setattr(selector.old,"_check_contexts",lambda summary,*args:summary["contexts"])
    folder=tmp_path/"edge_1"
    folder.mkdir()
    config=dict(mode="common_validation",format=validator.FORMAT,edge=1,contexts=context,
        fitting_plan_sha256=sha256(fitting_path),common_plan_sha256=sha256(common_path),frozen_sha256={})
    config_path=folder/"configuration.json"
    config_path.write_text(json.dumps(config))
    specs=[spec for spec,_ in plan(None,None,1)[2]]
    summary=dict(edge=1,format=validator.FORMAT,fitting_plan_sha256=sha256(fitting_path),common_plan_sha256=sha256(common_path),
        configuration_sha256=sha256(config_path),contexts=context,status="waiting_weights",validated=0,pending=18,failed=0,
        points=[dict(spec,status="pending",contexts={},selection_eligible=False) for spec in specs])
    summary_path=folder/"summary.json"
    summary_path.write_text(json.dumps(summary))
    result=selector.rollup(fitting_path,common_path,[],[summary_path])
    assert calls==[(common_path,[])]  # The strict old280 rollup stays delegated.
    assert len(result["points"])==370 and result["observed_weights"]==18
    assert len([row for row in result["points"] if row["validation"]["status"]=="pending"])==18
    assert list(result["common_contexts_by_edge"])==["1"]
    assert result["selection"]["recommendation"] is None
    phase_a=result["selection"]["sampling_phase_a"]
    assert phase_a["recommendation"] is None and len(phase_a["recipes"])==6
    assert all(group["rank"] is None for group in phase_a["recipes"])
    json.dumps(result,allow_nan=False)
    summary["contexts"]=copy.deepcopy(context)
    summary["contexts"]["cutover"]["token"]="different"
    config["contexts"]=summary["contexts"]
    config_path.write_text(json.dumps(config))
    summary["configuration_sha256"]=sha256(config_path)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="different common fingerprints"):
        selector.rollup(fitting_path,common_path,[],[summary_path])
