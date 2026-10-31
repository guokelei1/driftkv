"""New sampling identities use unchanged real three-context validation math."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code import (
    validate_probe_calibration_common as driver, validate_probe_temporal_common as temporal,
    calibration_sampling as sampling,
)
from experiments.ch3_motivation.code.calibration_gaps import fit_gap_law
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant, save_variant
from hstu_kvcache.common.artifacts import sha256
from tests.experiments.ch3_motivation.test_validate_probe_temporal_common import fixture as old_fixture


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup(tmp_path,monkeypatch,*,rule="gap_mix50",fraction=.75,stage="fgh"):
    fx=old_fixture(tmp_path,monkeypatch)
    contexts=temporal.prepare_contexts(fx.root,1,torch.device("cpu"),fx.manifest)
    root=tmp_path/"fit256"
    folder=root/"edge_1"
    folder.mkdir(parents=True)
    for name in ("population.json","edge_1/inputs.json","edge_1/history.pack","edge_1/requests.pack"):
        (root/name).write_bytes((fx.root/name).read_bytes())
    population=json.loads((root/"population.json").read_text())
    fit,val=population["fit"][:256],population["validation"]
    candidates=dict(fx.candidates,fitting_pool_uids=fit,
        recent_items={str(uid):list(range(64,48,-1)) for uid in fit+val})
    candidate=folder/"calibration_feedback_candidates.json"
    candidate.write_text(json.dumps(candidates))
    cutoff=231*86400
    law=fit_gap_law({fit[0]:[cutoff-500,cutoff-200]},
        {fit[0]:[cutoff-1000,cutoff-600]},fit,cutoff)
    manifest=dict(schema_version=1,status="prepared",edge=1,source_cutoff=cutoff,release_day=231,
        candidate_count=16,candidate_seed=17,gap_seed=17,catalog_source_users=256,time_law_source_users=256,
        candidate_catalog_source_fit_uids=fit,time_law_source_fit_uids=fit,target_fit_uids=fit,validation_uids=val,
        gap_law=law,time_rule=sampling.TIME_RULE,candidate_order="fixed feedback prefix then unique uniform; fixture",
        views=sampling.make_sampling_views(fit,val,candidates,{uid:100000. for uid in fit+val},law,edge=1,cutoff=cutoff),
        sourcebinding=dict(input_root=str(root),population_path=str(root/"population.json"),population_sha256=sha256(root/"population.json"),
            path=str(folder/"inputs.json"),sha256=sha256(folder/"inputs.json"),candidate_path=str(candidate),candidate_sha256=sha256(candidate),
            history_path=str(folder/"history.pack"),history_sha256=sha256(folder/"history.pack")),
        source_sha256={},cpu_seconds=dict(total=0.))
    path=tmp_path/"sampling_manifest.json"
    path.write_text(json.dumps(manifest))
    job=dict(name="fixture",input_root=str(root),output=str(tmp_path/"weights"),sampling_manifest=str(path),edge=1,
        recent_fraction=fraction,gap_rule=rule,fit_users=256,width=128,learning_rate=.00003,steps=2048,patience=1024,
        validation_every=8,stages="f,fg,fgh",controls=False,seed=17)
    sources={name:sha256(ROOT/name) for name in driver.FITTING_SOURCES}
    training=driver.prepare_training_binding(job,sources)
    expected=dict(training["expected_identity"],models=contexts["cutover"].identity["models"])
    probe=build_variant("full_item",fx.model,stage=stage,hidden_width=128,depth=1,query_chunk=16,row_chunk=1024)
    with torch.no_grad():
        probe.statistics_initialized.fill_(True)
        probe.row_scale.fill_(2.)
        probe.query_scale.fill_(1.5)
        for group in (probe.f,probe.g,probe.h):
            if group is not None:
                for layer in group:
                    layer.weights[-1].normal_(0,.0001)
    scales=dict(contexts["cutover"].identity["loss_scales"],users=256)
    metadata=dict(identity=expected,method_label=stage,sampling_provenance=training["sampling_provenance"],
        best_step=0,completed_steps=64,best_validation_loss=float("nan"),fit_pool_loss_scales=scales,
        training_users=256,validation_users=16,actual_teacher_users_captured=256,actual_validation_users_captured=16,
        quality_auc_computed=False,fitting_uses_future_feedback=False,cumulative_calibration_flops_estimate=123456)
    checkpoint=Path(job["output"])/f"{stage}.pt"
    save_variant(checkpoint,probe,metadata)
    spec=dict(origin="sampling256_90",recipe="fixture",edge=1,seed=17,method_label=stage,checkpoint=str(checkpoint),
        fitting_driver="run_probe_calibration.py",query_delta_rule=rule,recent_fraction=fraction,
        training_sampling_manifest=str(path),training_sampling_manifest_sha256=sha256(path))
    return SimpleNamespace(contexts=contexts,training=training,spec=spec,probe=probe,metadata=metadata,fx=fx,job=job)


def test_new_sampling_weights_reuse_exact_old_common_math_and_keep_step0(tmp_path,monkeypatch):
    fx=setup(tmp_path,monkeypatch)
    before=sha256(Path(fx.spec["checkpoint"]))
    row=driver.evaluate_sampling_weight(fx.contexts,fx.spec,fx.training)
    assert row["status"]=="validated" and row["step_zero"] and row["selection_eligible"]
    assert row["original_validation_loss"]=="nan"
    direct=load_variant(fx.spec["checkpoint"])[0]
    for rule,context in fx.contexts.items():
        assert row["contexts"][rule]["common_validation_sha256"]==context.fingerprint
        assert row["contexts"][rule]["metrics"]==temporal.evaluate_common_probe(context,direct)
        assert context.identity["loss_scales"]["users"]==192
        assert context.identity["recent_fraction"]==.5  # Training uses .75 and256 users.
    assert row["mean_context_loss"]==sum(row["contexts"][rule]["metrics"]["loss"] for rule in driver.RULES)/3
    reloaded,_=load_variant(fx.spec["checkpoint"])
    for name,tensor in fx.probe.state_dict().items():
        torch.testing.assert_close(tensor,reloaded.state_dict()[name],rtol=0,atol=0)
    assert sha256(Path(fx.spec["checkpoint"]))==before
    with pytest.raises(ValueError):
        temporal._check_planned_weight(fx.contexts["cutover"],fx.probe,fx.metadata,fx.spec)


@pytest.mark.parametrize("fault",["candidate_fraction","manifest_alias","source","fit_role","catalog_count","sampling_view"])
def test_actual_training_identity_mismatch_is_retained_failed(tmp_path,monkeypatch,fault):
    fx=setup(tmp_path,monkeypatch,rule="cutover",fraction=.5)
    metadata=copy.deepcopy(fx.metadata)
    identity=metadata["identity"]
    if fault=="candidate_fraction":identity["recent_fraction"]=.75
    elif fault=="manifest_alias":identity["gap_manifest_sha256"]="wrong"
    elif fault=="source":identity["numerical_sources"][driver.FITTING_SOURCES[0]]="wrong"
    elif fault=="fit_role":identity["fit_uids"].reverse()
    elif fault=="catalog_count":identity["candidate_catalog_source_users"]=64
    else:metadata["sampling_provenance"]["candidate_ids"][str(fx.metadata["identity"]["fit_uids"][0])][0]=1
    save_variant(fx.spec["checkpoint"],fx.probe,metadata)
    row=driver.evaluate_sampling_weight(fx.contexts,fx.spec,fx.training)
    assert row["status"]=="failed" and not row["selection_eligible"] and row["contexts"]=={}
    assert row["step_zero"] and "mean_context_loss" not in row


def test_cpu_identity_check_computes_no_validation_metrics(tmp_path,monkeypatch):
    fx=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(driver,"evaluate_common_probe",lambda *args:pytest.fail("CPU identity audit must not score"))
    row=driver.evaluate_sampling_weight({},fx.spec,fx.training,check_only=True,models=fx.metadata["identity"]["models"])
    assert row["status"]=="identity_checked" and row["contexts"]=={} and not row["selection_eligible"]


def test_late_context_failure_never_creates_partial_mean(tmp_path,monkeypatch):
    fx=setup(tmp_path,monkeypatch)
    actual=driver.evaluate_common_probe
    def evaluate(context,probe):
        if context.identity["query_delta_rule"]=="gap_empirical100":raise ValueError("fixture failure")
        return actual(context,probe)
    monkeypatch.setattr(driver,"evaluate_common_probe",evaluate)
    row=driver.evaluate_sampling_weight(fx.contexts,fx.spec,fx.training)
    assert row["status"]=="failed" and set(row["contexts"])=={"cutover","gap_mix50"}
    assert "mean_context_loss" not in row and not row["selection_eligible"]


def test_missing_weight_is_pending_without_context_allocation(tmp_path):
    row=driver.evaluate_sampling_weight({},dict(checkpoint=str(tmp_path/"missing.pt")),{})
    assert row["status"]=="pending" and row["contexts"]=={} and not row["selection_eligible"]


def test_missing_plan_is_pending_without_cuda_initialization(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,"set_device",lambda *args:pytest.fail("missing plans must never initialize CUDA"))
    args=SimpleNamespace(fitting_plan=tmp_path/"missing.json",common_plan=tmp_path/"old.json",output=tmp_path/"pending",check=True)
    result=driver.run(args)
    assert result["status"]=="waiting_plans" and result["points"]==[] and not result["gpu_started"]
    assert result["quality_auc_computed"] is False and result["post_release_labels_loaded"] is False


def test_run_driver_binds_old_reference_contexts_and_preserves_its_file(tmp_path,monkeypatch):
    fx=setup(tmp_path,monkeypatch)
    fitting,common=tmp_path/"new_plan.json",tmp_path/"common_plan.json"
    fitting.write_text("{}");common.write_text("{}")
    reference=tmp_path/"old_results/edge_1/summary.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps(dict(format=temporal.FORMAT,plan_sha256=sha256(common),
        contexts={rule:context.identity for rule,context in fx.contexts.items()})))
    before=sha256(reference)
    frozen={str(fitting):sha256(fitting),str(common):sha256(common)}
    monkeypatch.setattr(driver,"load_plan",lambda *a:({},dict(edge_bindings={"1":dict(input_root=str(fx.fx.root),gap_manifest=str(fx.fx.manifest))}),[(fx.spec,fx.training)],frozen))
    monkeypatch.setattr(temporal,"prepare_contexts",lambda *a,**k:fx.contexts)
    args=SimpleNamespace(fitting_plan=fitting,common_plan=common,edge=1,output=tmp_path/"new_results",check=False,
        device="cpu",gpu=1,backend="torch",reference_results=reference.parents[1])
    result=driver.run(args)
    assert result["status"]=="complete" and result["reference_280"]["status"]=="same_contexts"
    assert result["validated"]==1 and result["independent_fitting_users"]==64 and result["training_time_law_users"]==256
    assert sha256(reference)==before and not result["gpu_started"]
    for rule,context in fx.contexts.items():
        assert result["points"][0]["contexts"][rule]["common_validation_sha256"]==context.fingerprint


def test_frozen_source_change_is_rejected(tmp_path):
    path=tmp_path/"source.py"
    path.write_text("original")
    frozen={str(path):sha256(path)}
    path.write_text("changed")
    with pytest.raises(ValueError,match="source/input changed"):
        driver.verify_hashes(frozen)
