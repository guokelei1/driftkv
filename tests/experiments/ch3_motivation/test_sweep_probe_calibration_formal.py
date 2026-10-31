"""Actual tiny fitting for the two formal sampling drivers, never population AUC."""
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
    sweep_probe_calibration_formal as sweep, plan_probe_calibration_formal as formal,
)
from experiments.ch3_motivation.code.probe_initialization import initialization_spec
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.probes import fit_probe as real_fit
from hstu_kvcache.common.artifacts import sha256
import tests.experiments.ch3_motivation.test_run_probe_calibration as integration


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(tmp_path,monkeypatch,name):
    driver=sweep.DRIVERS[name]
    monkeypatch.setattr(integration,"driver",driver)
    config=integration.HSTUConfig
    monkeypatch.setattr(integration,"HSTUConfig",lambda **kw:config(**{**kw,"hidden_size":32,"num_heads":1,"num_layers":1}))
    fx=integration.fixture(tmp_path,monkeypatch)
    _,provenance=driver.calibration_sampling.load_sampling_manifest(fx.args.sampling_manifest,fx.args.input_root,
        1,fx.args.recent_fraction,fx.args.gap_rule,fit_users=64)
    job={key:getattr(fx.args,key) for key in sweep.RUNTIME_FIELDS}
    job.update(name="selected_main",variant="full_item",depth=1,queries=16,read_weight=1.,score_weight=1.,
        candidate_source="feedback",fitting_driver=name,fitting_driver_sha256=sha256(Path(driver.__file__)),
        query_delta_rule=fx.args.gap_rule,sampling_manifest=str(fx.args.sampling_manifest.resolve()),
        sampling_manifest_sha256=sha256(fx.args.sampling_manifest),gap_manifest=str(fx.args.sampling_manifest.resolve()),
        gap_manifest_sha256=sha256(fx.args.sampling_manifest),time_rule=provenance["time_rule"],
        source_cutoff=provenance["source_cutoff"],candidate_order=provenance["candidate_order"],query_delta_layout="[B,16]",
        candidate_catalog_source_users=256,time_law_source_users=256,gradient_fitting_users=64,teacher_fitting_users=64)
    if name.endswith("_init.py"):
        spec=initialization_spec(32,96,"balanced")
        job.update(h_initialization="balanced",h_initialization_spec=spec,initialization_helper_sha256=spec["helper_sha256"])
    binding=dict(formal._input_binding(fx.args.input_root,1),sampling_manifest=job["sampling_manifest"],
        sampling_manifest_sha256=job["sampling_manifest_sha256"])
    plan=dict(main_input_bindings={"1":binding},curve_input_bindings={"1":binding})
    args=SimpleNamespace(output=tmp_path/"formal_weights",family="main",gpu=1,device="cpu")
    return SimpleNamespace(fx=fx,driver=driver,job=job,plan=plan,args=args)


@pytest.mark.parametrize("name",list(sweep.DRIVERS))
def test_actual_main_resume_curve_fullcap_and_quality_audit(tmp_path,monkeypatch,name):
    setup=fixture(tmp_path,monkeypatch,name);job=setup.job;args=setup.args;fx=setup.fx
    main=sweep._execute(args,setup.plan,[job],dict(test="CPU actualmain"),{})
    assert main["jobs"]==1 and len(main["points"])==5 and main["quality_auc_computed"] is False
    folder=sweep._folder(args.output,"main",job)
    before=integration.states(folder)
    assert {row["method_label"] for row in main["points"]}==set(sweep.quality.MAIN_METHODS)
    for row in main["completed"][0]["actual_weight_audit"]:
        assert row["sampling_identity"]["fitting_driver"]==name
        assert row["sampling_identity"]["gradient_fitting_users"]==64
    fits=len(fx.fits)
    sweep._execute(args,setup.plan,[job],dict(test="CPU actualmain"),{})
    assert len(fx.fits)==fits
    for stage in ("f_extended","fg","fg_extended","fgh"):
        for suffix in ("pt","json"):(folder/f"{stage}.{suffix}").unlink()
    sweep._execute(args,setup.plan,[job],dict(test="CPU actualmain"),{})
    integration.assert_equal_states(before,integration.states(folder))
    # Unlike bounded main checks, patience0 curves execute all declared512 real
    # optimizer steps. The unmodified quality gate verifies that exact count.
    actual=[]
    def curve_fit(model,probe,train,validation,**kwargs):
        kwargs["progress_callback"]=None
        result=real_fit(model,probe,train,validation,**kwargs)
        actual.append((len(train),kwargs["seed"],result["completed_steps"]))
        return result
    monkeypatch.setattr(setup.driver,"fit_probe",curve_fit)
    curves=[]
    for n,seed in ((256,17),(64,17),(64,23)):
        curves.append({**job,"name":f"selected_f_curve_fit{n}","fit_users":n,"gradient_fitting_users":n,
            "teacher_fitting_users":n,"seed":seed,"stages":"f","controls":False,"patience":0})
    curve_args=SimpleNamespace(**{**vars(args),"family":"curve"})
    curve=sweep._execute(curve_args,setup.plan,curves,dict(test="CPU fullcap curve"),{})
    assert actual==[(256,17,512),(64,17,512),(64,23,512)]
    assert [row["raw_preparation_reused_this_invocation"] for row in curve["completed"]]==[False,False,True]
    assert all(row["completed_steps"]==512 and row["method_label"]=="f" for row in curve["points"])
    for planned in curves:
        path=sweep.quality.checkpoint_path(args.output/"curve",planned,"f")
        _,metadata=load_variant(path)
        identity=metadata["identity"]
        assert identity["stages"]=="f" and identity["controls"] is False and identity["patience"]==0
        assert identity["seed"]==planned["seed"]
        assert metadata["fit_pool_loss_scales"]["users"]==planned["fit_users"]
        assert identity["teacher_fitting_users"]==planned["fit_users"] and identity["time_law_source_users"]==256
    assert main["quality_roots"]==curve["quality_roots"]
    integration.assert_equal_states(before,integration.states(folder))
    count=len(actual);sweep._execute(curve_args,setup.plan,curves,dict(test="CPU fullcap curve"),{})
    assert len(actual)==count


def test_actual_phase_b_seed17_controls_false_cannot_resume_formal_main(tmp_path,monkeypatch):
    setup=fixture(tmp_path,monkeypatch,"run_probe_calibration_init.py");fx=setup.fx
    phase_b=SimpleNamespace(**{**vars(fx.args),"h_initialization":"balanced","controls":False})
    setup.driver.run(phase_b)
    folder=sweep._folder(setup.args.output,"main",setup.job);folder.mkdir(parents=True)
    shutil.copyfile(phase_b.output/"f.pt",folder/"f.pt")
    count=len(fx.loads)
    with pytest.raises(ValueError,match="fitting recipe differs"):
        sweep._execute(setup.args,setup.plan,[setup.job],dict(test="refuse PhaseB relabel"),{})
    assert len(fx.loads)==count  # Reject before any expensive model preparation.


def test_complete_grids_partition_paths_and_legacy_rejection(tmp_path,monkeypatch):
    setup=fixture(tmp_path,monkeypatch,"run_probe_calibration.py");base=setup.job
    main=[{**base,"edge":e,"seed":s} for e in range(1,6) for s in sweep.SEEDS]
    curves=[{**base,"name":f"curve_fit{n}","edge":e,"seed":s,"fit_users":n,
        "gradient_fitting_users":n,"teacher_fitting_users":n,"stages":"f","controls":False,"patience":0}
        for e in range(1,6) for s in sweep.SEEDS for n in (64,128,256)]
    plan=dict(status="prepared_for_root_review",main_jobs=main,curve_jobs=curves)
    picked=sweep._jobs(plan,"main",1,setup.args.output)
    assert len(picked)==9 and {row["edge"] for row in picked}=={1,3,5}
    picked=sweep._jobs(plan,"curve",0,setup.args.output)
    assert len(picked)==18 and [(j["fit_users"],j["seed"]) for j in picked[:9]]==[(n,s) for n in (256,128,64) for s in sweep.SEEDS]
    assert sweep._folder(setup.args.output,"main",base)!=sweep._folder(setup.args.output,"curve",base)
    broken=copy.deepcopy(plan);broken["main_jobs"][0]["controls"]=False
    with pytest.raises(ValueError,match="control-enabled identity"):sweep._jobs(broken,"main",1,setup.args.output)
    broken=copy.deepcopy(plan);broken["main_jobs"][0]["output"]=str(sweep._folder(setup.args.output,"curve",base))
    with pytest.raises(ValueError,match="separate main/curve"):sweep._jobs(broken,"main",1,setup.args.output)
    broken=copy.deepcopy(plan);broken["main_jobs"][0]["fitting_driver"]="run_probe_temporal.py"
    with pytest.raises(ValueError,match="legacy fitting driver"):sweep._jobs(broken,"main",1,setup.args.output)


def test_plan_freeze_cannot_be_overwritten_by_current_quality_source(tmp_path,monkeypatch):
    source=Path(sweep.quality.__file__).resolve()
    relative=str(source.relative_to(ROOT.resolve()))
    actual=sha256(source)
    # Both actual key spellings name the same file. Matching aliases collapse;
    # a current-source snapshot must never replace the plan's older digest.
    merged=sweep._merge_frozen({str(source):actual},{relative:actual})
    assert merged=={str(source):actual}
    plan_path=tmp_path/"plan.json"
    plan_path.write_text(json.dumps(dict(frozen_sha256={str(source):"0"*64})))
    monkeypatch.setattr(sweep,"_jobs",lambda *args:[])
    monkeypatch.setattr(sweep.quality,"sources",lambda:{relative:actual})
    monkeypatch.setattr(sweep,"_execute",lambda *args:pytest.fail("conflicting freeze reached fitting"))
    args=SimpleNamespace(plan=plan_path,family="main",gpu=1,output=tmp_path/"new_weights",device="cpu")
    with pytest.raises(ValueError,match="formal freeze SHA conflict"):
        sweep.main(args)
    assert not (args.output/"main/execution/gpu_1/configuration.json").exists()
