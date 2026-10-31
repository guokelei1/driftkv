"""PhaseB preserves actual-N/init identity and the original common math."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    validate_probe_calibration_followup_common as driver, validate_probe_temporal_common as temporal,
)
from experiments.ch3_motivation.code.probe_initialization import initialization_spec
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant, save_variant
from experiments.ch3_motivation.code.sampling_identity import validate_sampling_identity
from tests.experiments.ch3_motivation.test_validate_probe_calibration_common import setup as phase_a_setup
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(tmp_path,monkeypatch,*,n=64,width=128,mode="balanced",stage="fgh"):
    fx=phase_a_setup(tmp_path,monkeypatch,stage=stage)
    probe=build_variant("full_item",fx.fx.model,stage=stage,hidden_width=width,depth=1,query_chunk=16,row_chunk=1024)
    with torch.no_grad():
        probe.statistics_initialized.fill_(True);probe.row_scale.fill_(2.);probe.query_scale.fill_(1.5)
        for group in (probe.f,probe.g,probe.h):
            if group is not None:
                for network in group:network.weights[-1].normal_(0,.0001)
    identity=copy.deepcopy(fx.metadata["identity"])
    h_spec=initialization_spec(probe.cfg.head_dim,probe.cfg.row_dim,mode)
    identity.update(fit_users=n,fit_uids=identity["fit_uids"][:n],gradient_fitting_users=n,teacher_fitting_users=n,
        width=width,h_initialization=mode,h_initialization_spec=h_spec,initialization_helper_sha256=h_spec["helper_sha256"],
        fitting_driver=driver.DRIVER,fitting_driver_sha256=sha256(ROOT/driver.FITTING_SOURCES[0]),
        numerical_sources={name:sha256(ROOT/name) for name in driver.FITTING_SOURCES})
    job=dict(fx.job,fit_users=n,width=width,h_initialization=mode,fitting_driver=driver.DRIVER,
        fitting_driver_sha256=identity["fitting_driver_sha256"],reuse_existing_weights=False,expected_identity=identity,
        sampling_provenance=validate_sampling_identity(identity,config=asdict(probe.cfg)))
    training=driver.prepare_training_binding(job)
    metadata=dict(fx.metadata,identity=identity,sampling_provenance=training["sampling_provenance"],
        training_users=n,actual_teacher_users_captured=n,teacher_preparation_users=n+16,
        fit_pool_loss_scales=dict(users=n,source="fitting_users_only",version="fit_pool_equal_user_length_normalized_v1",
            validation_uses_fitting_scales=True),h_initialization_weight_rescale_flops=sum(
            network.weights[0].numel() for network in probe.h) if mode=="balanced" and stage=="fgh" else 0)
    spec=dict(fx.spec,origin="calibration_followup",recipe="fit64" if n==64 else "reference",fitting_driver=driver.DRIVER,
        h_initialization=mode,fit_users=n,width=width)
    save_variant(spec["checkpoint"],probe,metadata)
    return SimpleNamespace(fx=fx,contexts=fx.contexts,job=job,training=training,metadata=metadata,probe=probe,spec=spec)


@pytest.mark.parametrize("n,width,mode",[(64,128,"standard"),(128,256,"balanced"),(256,128,"balanced")])
def test_actual_n_width_init_use_identical_common_contexts_and_preserve_weights(tmp_path,monkeypatch,n,width,mode):
    fx=fixture(tmp_path,monkeypatch,n=n,width=width,mode=mode)
    before=sha256(Path(fx.spec["checkpoint"]))
    row=driver.evaluate_weight(fx.contexts,fx.spec,fx.training)
    assert row["status"]=="validated" and row["selection_eligible"] and row["step_zero"]
    assert row["fitting_users"]==n and row["config"]["hidden_width"]==width
    assert row["training_identity"]["h_initialization"]==mode
    assert row["original_loss_scales"]["users"]==n and row["original_validation_loss"]=="nan"
    direct=load_variant(fx.spec["checkpoint"])[0]
    for rule,context in fx.contexts.items():
        assert row["contexts"][rule]["metrics"]==temporal.evaluate_common_probe(context,direct)
        assert row["contexts"][rule]["common_validation_sha256"]==context.fingerprint
        assert context.identity["loss_scales"]["users"]==192
    assert row["mean_context_loss"]==sum(row["contexts"][rule]["metrics"]["loss"] for rule in driver.RULES)/3
    assert sha256(Path(fx.spec["checkpoint"]))==before


@pytest.mark.parametrize("fault",["teacher_n","statistics_n","initialization_helper","actual_geometry","phase_a_driver","manifest_alias","sampling_view"])
def test_mismatched_actual_identity_capture_or_init_is_failed_not_partial(tmp_path,monkeypatch,fault):
    fx=fixture(tmp_path,monkeypatch)
    metadata=copy.deepcopy(fx.metadata);probe=fx.probe
    if fault=="teacher_n":metadata["actual_teacher_users_captured"]=256
    elif fault=="statistics_n":metadata["fit_pool_loss_scales"]["users"]=256
    elif fault=="initialization_helper":metadata["identity"]["initialization_helper_sha256"]="wrong"
    elif fault=="actual_geometry":probe=build_variant("full_item",fx.fx.fx.model,stage="fgh",hidden_width=256,row_chunk=1024)
    elif fault=="phase_a_driver":metadata["identity"]["fitting_driver"]="run_probe_calibration.py"
    elif fault=="manifest_alias":metadata["identity"]["gap_manifest_sha256"]="wrong"
    else:metadata["sampling_provenance"]["candidate_ids"][str(metadata["identity"]["fit_uids"][0])][0]=1
    save_variant(fx.spec["checkpoint"],probe,metadata)
    row=driver.evaluate_weight(fx.contexts,fx.spec,fx.training)
    assert row["status"]=="failed" and not row["selection_eligible"] and row["contexts"]=={}
    assert "mean_context_loss" not in row


def test_check_only_never_scores_and_late_failure_never_averages_partial_contexts(tmp_path,monkeypatch):
    fx=fixture(tmp_path,monkeypatch)
    actual=driver.evaluate_common_probe
    monkeypatch.setattr(driver,"evaluate_common_probe",lambda *args:pytest.fail("CPU check cannot score"))
    row=driver.evaluate_weight({},fx.spec,fx.training,check_only=True,models=fx.metadata["identity"]["models"])
    assert row["status"]=="identity_checked" and not row["selection_eligible"] and row["contexts"]=={}
    def fail_late(context,probe):
        if context.identity["query_delta_rule"]=="gap_empirical100":raise ValueError("late fixture failure")
        return actual(context,probe)
    monkeypatch.setattr(driver,"evaluate_common_probe",fail_late)
    row=driver.evaluate_weight(fx.contexts,fx.spec,fx.training)
    assert row["status"]=="failed" and len(row["contexts"])==2 and "mean_context_loss" not in row


def test_matched_f_fg_tensor_and_buffer_check_catches_normalization_difference(tmp_path,monkeypatch):
    fx=fixture(tmp_path,monkeypatch,n=256,mode="standard",stage="fg")
    prepared=[];rows=[]
    for stage in ("f","fg"):
        for recipe,mode in (("reference","standard"),("balanced_h_init","balanced")):
            probe=fx.probe if stage=="fg" else build_variant("full_item",fx.fx.fx.model,stage=stage,hidden_width=128,row_chunk=1024)
            if stage=="f":torch.manual_seed(17)  # Save the same freshly constructed tensors for the paired mode below.
            metadata=copy.deepcopy(fx.metadata)
            metadata["identity"].update(h_initialization=mode,h_initialization_spec=initialization_spec(probe.cfg.head_dim,probe.cfg.row_dim,mode))
            metadata["method_label"]=stage
            path=tmp_path/recipe/f"{stage}.pt"
            # Copy the standard f tensors exactly rather than relying on a seed assertion.
            if stage=="f" and mode=="balanced":probe=load_variant(tmp_path/"reference/f.pt")[0]
            save_variant(path,probe,metadata)
            spec=dict(recipe=recipe,method_label=stage,checkpoint=str(path))
            prepared.append((spec,{}));rows.append(dict(spec,status="validated",checkpoint_sha256=sha256(path),selection_eligible=True))
    result=driver.check_paired_predecessors(prepared,rows)
    assert result["status"]=="verified_equal" and len(result["stages"])==2
    path=tmp_path/"balanced_h_init/fg.pt"
    probe,metadata=load_variant(path)
    with torch.no_grad():probe.row_scale.add_(1)
    save_variant(path,probe,metadata)
    rows[-1]["checkpoint_sha256"]=sha256(path)
    result=driver.check_paired_predecessors(prepared,rows)
    assert result["status"]=="failed" and all(not row["selection_eligible"] for row in rows)


def test_missing_plans_and_weights_are_pending_without_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,"set_device",lambda *args:pytest.fail("pending input must not initialize CUDA"))
    args=SimpleNamespace(followup_plan=tmp_path/"missing.json",common_plan=tmp_path/"old.json",output=tmp_path/"pending",check=True)
    result=driver.run(args)
    assert result["status"]=="waiting_plans" and result["points"]==[] and not result["gpu_started"]
    assert driver.evaluate_weight({},dict(checkpoint=str(tmp_path/"missing.pt")),{})["status"]=="pending"


def test_run_plumbing_matches_both_frozen_references_and_retains_no_auc(tmp_path,monkeypatch):
    fx=fixture(tmp_path,monkeypatch)
    followup,common=tmp_path/"followup.json",tmp_path/"common.json"
    followup.write_text("{}");common.write_text("{}")
    contexts={rule:context.identity for rule,context in fx.contexts.items()}
    for name,fmt,key in (("old",temporal.FORMAT,"plan_sha256"),("new",driver.phase_a.FORMAT,"common_plan_sha256")):
        path=tmp_path/name/"edge_1/summary.json";path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(format=fmt,contexts=contexts,**{key:sha256(common)})))
    frozen={str(followup):sha256(followup),str(common):sha256(common)}
    monkeypatch.setattr(driver,"load_plan",lambda *args:(dict(expected_weights=105),dict(edge_bindings={"1":dict(
        input_root=str(fx.fx.fx.root),gap_manifest=str(fx.fx.fx.manifest))}),[(fx.spec,fx.training)],frozen))
    monkeypatch.setattr(temporal,"prepare_contexts",lambda *args,**kwargs:fx.contexts)
    args=SimpleNamespace(followup_plan=followup,common_plan=common,edge=1,output=tmp_path/"result",check=False,
        device="cpu",gpu=1,backend="torch",reference_results=tmp_path/"old",sampling_reference_results=tmp_path/"new")
    result=driver.run(args)
    assert result["status"]=="complete" and result["validated"]==1 and not result["gpu_started"]
    assert all(row["status"]=="same_contexts" for row in result["references"].values())
    assert result["expected_weights"]==105 and result["expected_context_cells"]==315
    assert not result["quality_auc_computed"] and result["independent_fitting_users"]==64
    changed=copy.deepcopy(contexts);changed["cutover"]["source_cutoff"]=0
    path=tmp_path/"new/edge_1/summary.json"
    path.write_text(json.dumps(dict(format=driver.phase_a.FORMAT,contexts=changed,common_plan_sha256=sha256(common))))
    args.output=tmp_path/"mismatch"
    with pytest.raises(ValueError,match="fingerprints differ"):driver.run(args)


def test_real_plan_inventory_uses_own_n_and_freezes_actual_request_pack(tmp_path,monkeypatch):
    fx=fixture(tmp_path,monkeypatch)
    config_path=tmp_path/"recipes.json"
    config=json.loads((ROOT/"experiments/ch3_motivation/configs/probe_calibration_followup.json").read_text())
    config_path.write_text(json.dumps(config))
    root=Path(fx.job["input_root"])
    population=json.loads((root/"population.json").read_text())
    common_path=tmp_path/"common_plan.json"
    candidate=fx.fx.fx.root/"edge_1/calibration_feedback_candidates.json"
    binding=dict(input_root=str(fx.fx.fx.root),gap_manifest=str(fx.fx.fx.manifest),gap_manifest_sha256=sha256(fx.fx.fx.manifest),
        candidate_file=str(candidate),candidate_file_sha256=sha256(candidate),population_sha256=fx.metadata["identity"]["population_sha256"],
        input_binding_sha256=fx.metadata["identity"]["input_binding_sha256"])
    common=dict(format="ch3_temporal_common_plan_v1",rules=driver.RULES,weights=[dict(checkpoint=f"old{i}.pt") for i in range(280)],
        source_fitting_plans_sha256={},executor_sources_sha256={},planner_sha256=sha256(ROOT/"experiments/ch3_motivation/code/prepare_temporal_common_plan.py"),
        edge_bindings={str(edge):binding for edge in driver.EDGES})
    common_path.write_text(json.dumps(common))
    jobs=[]
    for edge in driver.EDGES:
        for row in config["recipes"]:
            recipe={key:value for key,value in row.items() if key!="name"}
            identity=copy.deepcopy(fx.metadata["identity"])
            identity.update(recipe,edge=edge,fit_uids=population["fit"][:recipe["fit_users"]],
                gradient_fitting_users=recipe["fit_users"],teacher_fitting_users=recipe["fit_users"],
                h_initialization_spec=initialization_spec(fx.probe.cfg.head_dim,fx.probe.cfg.row_dim,recipe["h_initialization"]))
            output=tmp_path/"new_weights"/row["name"]/f"edge_{edge}"
            job=dict(fx.job,**recipe)
            job.update(name=row["name"],edge=edge,output=str(output),expected_identity=identity,
                expected_checkpoints={stage:str(output/f"{stage}.pt") for stage in driver.STAGES})
            if edge==1:job["sampling_provenance"]=validate_sampling_identity(identity,config=asdict(fx.probe.cfg))
            jobs.append(job)
    plan=dict(format="probe_calibration_followup_v1",status="prepared_for_root_review",common_plan_path=str(common_path),
        common_plan_sha256=sha256(common_path),source_config=str(config_path),source_config_sha256=sha256(config_path),
        planner_sources_sha256={},frozen_sha256={},expected_ladders=35,expected_weights=105,jobs=jobs)
    path=tmp_path/"plan.json";path.write_text(json.dumps(plan))
    _,_,prepared,frozen=driver.load_plan(path,common_path,1)
    assert len(prepared)==21 and {training["expected_identity"]["fit_users"] for _,training in prepared}=={64,128,256}
    assert {spec["width"] for spec,_ in prepared}=={128,256}
    request=root/"edge_1/requests.pack"
    assert frozen[str(request.resolve())]==sha256(request)
    request.write_bytes(b"changed-request-pack")
    with pytest.raises(ValueError,match="source/input changed"):driver.load_plan(path,common_path,1)

