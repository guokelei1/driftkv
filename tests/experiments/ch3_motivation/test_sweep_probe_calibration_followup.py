"""PhaseB sequential fitting uses real optimizer steps and exact typed resume."""
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    sweep_probe_calibration_followup as sweep, run_probe_calibration as phase_a,
    run_probe_calibration_init as driver,
)
from experiments.ch3_motivation.code.prepare_probe_calibration_followup import _recipes, DEFAULT_CONFIG
from experiments.ch3_motivation.code.probe_initialization import initialization_spec
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.common.artifacts import sha256
import tests.experiments.ch3_motivation.test_run_probe_calibration as integration


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(integration,"driver",driver)
    fx=integration.fixture(tmp_path,monkeypatch,count=256)
    fx.args.h_initialization="standard";fx.args.controls=False;fx.args.steps=2048;fx.args.patience=1024
    baseline=driver.run(fx.args)
    recipes=_recipes(json.loads(DEFAULT_CONFIG.read_text()));jobs=[]
    for name in sweep.ORDER:
        recipe=recipes[name];identity=copy.deepcopy(baseline["identity"])
        n=recipe["fit_users"];spec=initialization_spec(32,128,recipe["h_initialization"])
        identity.update(recipe,fit_uids=fx.fit[:n],gradient_fitting_users=n,teacher_fitting_users=n,
            h_initialization_spec=spec,initialization_helper_sha256=spec["helper_sha256"])
        output=tmp_path/"jobs"/name/"edge_1/seed_17"
        job=dict(vars(fx.args));job.update(recipe,name=name,output=str(output),
            fitting_driver="run_probe_calibration_init.py",fitting_driver_sha256=identity["fitting_driver_sha256"],
            expected_identity=identity,reuse_existing_weights=False,
            expected_checkpoints={stage:str(output/f"{stage}.pt") for stage in ("f","fg","fgh")})
        jobs.append(job)
    plan=dict(weight_reuse_policy="new actual init driver, no PhaseA relabelling")
    args=SimpleNamespace(output=tmp_path/"sweep",gpu=1)
    return SimpleNamespace(fx=fx,jobs=jobs,plan=plan,args=args)


def test_seven_real_recipes_cost_cache_and_partial_complete_resume(tmp_path,monkeypatch):
    setup=fixture(tmp_path,monkeypatch);fx=setup.fx
    result=sweep._execute(setup.args,setup.plan,setup.jobs,dict(test="bounded CPU edge1"),{})
    assert result["status"]=="complete" and result["jobs"]==7 and len(result["points"])==21
    assert result["quality_auc_computed"] is False
    assert [len(uids) for uids in fx.captures]==[272,272,80,144]
    assert [row["raw_preparation_reused_this_invocation"] for row in result["completed"]]==[False,True,True,True,True,False,False]
    assert all(check["exact_tensor_equality"] for check in result["paired_initialization_checks"][0]["checks"])
    assert result["complete_standalone_calibration_flops_estimate_sum"]==sum(row["complete_ladder_calibration_flops_estimate"] for row in result["completed"])
    for job in setup.jobs:
        payload=json.loads((Path(job["output"])/"summary.json").read_text())
        assert payload["identity"]==job["expected_identity"]
        prep=payload["calibration_preparation"]["total_model_preparation_flops"]
        assert prep>0 and payload["stages"][0]["source_teacher_and_target_preparation_flops"]==prep
    reference=Path(setup.jobs[0]["output"])
    integration.assert_equal_states(integration.states(fx.args.output),integration.states(reference))
    before={job["name"]:integration.states(Path(job["output"])) for job in setup.jobs}
    count=len(fx.fits)
    again=sweep._execute(setup.args,setup.plan,setup.jobs,dict(test="bounded CPU edge1"),{})
    assert len(fx.fits)==count and again["points"]==result["points"]
    # A genuine stage-boundary interruption retains trained f, then constructs
    # and optimizes new g/h with the original RNG trajectory in both modes.
    for job in setup.jobs[:2]:
        for stage in ("fg","fgh"):
            for suffix in ("pt","json"):(Path(job["output"])/f"{stage}.{suffix}").unlink()
    sweep._execute(setup.args,setup.plan,setup.jobs,dict(test="bounded CPU edge1"),{})
    assert len(fx.fits)==count+4
    for job in setup.jobs:integration.assert_equal_states(before[job["name"]],integration.states(Path(job["output"])))


def test_phase_a_checkpoint_cannot_masquerade_as_new_standard_driver(tmp_path,monkeypatch):
    setup=fixture(tmp_path,monkeypatch);fx=setup.fx
    for name in ("load_models","restore_pack","capture_samples","prepare_training_samples","fit_probe"):
        monkeypatch.setattr(phase_a,name,getattr(driver,name))
    original=SimpleNamespace(**{**vars(fx.args),"output":tmp_path/"phase_a","stages":"f"})
    phase_a.run(original)
    path=Path(setup.jobs[0]["output"]);path.mkdir(parents=True)
    shutil.copyfile(original.output/"f.pt",path/"f.pt")
    _,saved=load_variant(path/"f.pt")
    assert saved["identity"]["fitting_driver"]=="run_probe_calibration.py"
    with pytest.raises(ValueError,match="identity/branch differs"):
        sweep._execute(setup.args,setup.plan,setup.jobs[:1],dict(test="reject old actual driver"),{})


def test_full_plan_partition_exact_sources_and_resume_identity(tmp_path,monkeypatch):
    digest=sha256(Path(driver.__file__));jobs=[]
    for edge in range(1,6):
        for name in sweep.ORDER:
            output=tmp_path/name/f"edge_{edge}/seed_17"
            jobs.append(dict(name=name,edge=edge,gpu=1 if edge%2 else 0,fitting_driver=sweep.DRIVER,
                fitting_driver_sha256=digest,reuse_existing_weights=False,stages="f,fg,fgh",controls=False,seed=17,
                output=str(output),expected_checkpoints={stage:str(output/f"{stage}.pt") for stage in ("f","fg","fgh")}))
    frozen_file=tmp_path/"source.py";frozen_file.write_text("actual source")
    plan=dict(format="probe_calibration_followup_v1",status="prepared_for_root_review",expected_ladders=35,
        expected_weights=105,jobs=jobs,gpu_jobs={str(g):[j for j in jobs if j["gpu"]==g] for g in (0,1)},
        frozen_sha256={str(frozen_file):sha256(frozen_file)})
    plan_path=tmp_path/"plan.json";plan_path.write_text(json.dumps(plan));calls=[]
    monkeypatch.setattr(sweep,"_execute",lambda args,plan,jobs,identity,frozen:calls.append((jobs,identity)) or dict(status="complete"))
    args=SimpleNamespace(plan=plan_path,output=tmp_path/"sweep",gpu=1)
    sweep.main(args)
    assert len(calls[0][0])==21 and {j["edge"] for j in calls[0][0]}=={1,3,5}
    sweep.main(args);assert calls[1][1]==calls[0][1]
    frozen_file.write_text("changed source")
    with pytest.raises(ValueError,match="frozen source/input changed"):sweep.main(args)
    frozen_file.write_text("actual source")
    plan["gpu_jobs"]["1"][0]["fitting_driver"]="run_probe_calibration.py"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match="driver/branch differs"):sweep.main(args)
