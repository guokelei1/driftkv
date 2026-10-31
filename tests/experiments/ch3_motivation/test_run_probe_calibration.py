"""Real bounded optimizer integration of the new configurable calibration driver."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import run_probe_calibration as driver
from experiments.ch3_motivation.code.calibration_gaps import fit_gap_law, STALE_FIELDS
from experiments.ch3_motivation.code.probes import (
    fit_probe as real_fit, prepare_training_samples as real_prepare,
)
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTU,HSTUConfig
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(tmp_path,monkeypatch,*,count=64):
    root=tmp_path/"fit256";folder=root/"edge_1";folder.mkdir(parents=True)
    fit=list(range(1000,1512));val=list(range(2000,2016));cutoff=20*86400
    population=dict(fit=fit,validation=val)
    (root/"population.json").write_text(json.dumps(population))
    for kind in ("history","requests"):(folder/f"{kind}.pack").write_text(kind)
    binding=dict(population_sha256=sha256(root/"population.json"),
        settings=dict(edge=1,release_day=20,history_length=8,attention_backend="torch"),
        **{kind:dict(path=f"{kind}.pack",sha256=sha256(folder/f"{kind}.pack")) for kind in ("history","requests")})
    (folder/"inputs.json").write_text(json.dumps(binding))
    candidates=dict(population_sha256=sha256(root/"population.json"),fitting_pool_uids=fit[:256],
        validation_uids=val,known_items=64,recent_items={str(uid):list(range(1,17)) for uid in fit[:256]+val})
    candidate_path=folder/"calibration_feedback_candidates.json";candidate_path.write_text(json.dumps(candidates))
    law=fit_gap_law({uid:[cutoff-20] for uid in fit[:256]},
                    {uid:[cutoff-60,cutoff-30] for uid in fit[:256]},fit[:256],cutoff)
    sampling=driver.calibration_sampling
    manifest=dict(schema_version=1,status="prepared",edge=1,source_cutoff=cutoff,candidate_count=16,
        candidate_seed=17,gap_seed=17,catalog_source_users=256,time_law_source_users=256,
        candidate_catalog_source_fit_uids=fit[:256],time_law_source_fit_uids=fit[:256],target_fit_uids=fit[:256],
        validation_uids=val,gap_law=law,time_rule="strict-past feedback gap",candidate_order="feedback prefix then uniform",
        views=sampling.make_sampling_views(fit[:256],val,candidates,{uid:30 for uid in fit[:256]+val},law,edge=1,cutoff=cutoff),
        cpu_seconds=dict(total=.01),source_sha256={},sourcebinding=dict(input_root=str(root),
            population_sha256=sha256(root/"population.json"),sha256=sha256(folder/"inputs.json"),
            candidate_sha256=sha256(candidate_path),history_path=str(folder/"history.pack"),history_sha256=sha256(folder/"history.pack")))
    path=tmp_path/"sampling.json";path.write_text(json.dumps(manifest))
    torch.manual_seed(812)
    cfg=HSTUConfig(num_items=320,num_behaviors=2,hidden_size=64,num_heads=2,num_layers=2,
        max_seq_len=8,input_dropout=0.,attn_dropout=0.)
    old=HSTU(cfg).eval().requires_grad_(False);current=copy.deepcopy(old)
    with torch.no_grad():current.blocks[0].attn.k_proj.weight.add_(.03*torch.randn_like(current.blocks[0].attn.k_proj.weight))
    original={key:value.clone() for key,value in current.state_dict().items()}
    histories=SimpleNamespace(rows={uid:(np.array([cutoff-31,cutoff-30,cutoff+1]),np.array([1,2,63]),np.ones(3,dtype=np.int64)) for uid in fit[:256]+val})
    captures=[];preparations=[];actual_fits=[];loads=[]
    real_capture=driver.capture_samples
    def capture(parent,model,values,uids,settings,device):
        assert uids==fit[:len(uids)-16]+val
        samples=real_capture(parent,model,values,uids,settings,device)
        for uid,s in samples.items():
            assert 63 not in s["item_ids"]
            for name in STALE_FIELDS:s[name]="stale"
        captures.append(list(uids));return samples
    def prepare(model,samples):
        assert all(not (set(STALE_FIELDS)&set(s)) and s["candidates"].shape==s["query_delta"].shape==(1,16) for s in samples)
        result=real_prepare(model,samples)
        wanted,_=score(model,result[0]["teacher_cache"],result[0]["candidates"],result[0]["query_delta"])
        torch.testing.assert_close(result[0]["teacher_scores"],wanted)
        preparations.append(result);return result
    def bounded_fit(model,probe,train,validation,**kwargs):
        requested=kwargs["steps"];kwargs["steps"]=2*(requested//512)
        kwargs["validation_every"]=1;kwargs["patience"]=0
        assert len(validation)==16 and len(train) in (64,128,256)
        before={k:v.clone() for k,v in probe.state_dict().items()}
        result=real_fit(model,probe,train,validation,**kwargs)
        assert result["completed_steps"]==kwargs["steps"]
        assert all(torch.equal(value,current.state_dict()[key]) for key,value in original.items())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in probe.parameters())
        actual_fits.append(dict(stage=probe.stage,requested=requested,actual=result["completed_steps"],before=before))
        return result
    def load_models(*args,**kwargs):loads.append(1);return old,current,{"old":"sha-old","current":"sha-current"}
    monkeypatch.setattr(driver,"load_models",load_models)
    monkeypatch.setattr(driver,"restore_pack",lambda _:histories)
    monkeypatch.setattr(driver,"capture_samples",capture)
    monkeypatch.setattr(driver,"prepare_training_samples",prepare)
    monkeypatch.setattr(driver,"fit_probe",bounded_fit)
    args=SimpleNamespace(input_root=root,output=tmp_path/"fresh",sampling_manifest=path,edge=1,
        recent_fraction=.75,gap_rule="gap_mix50",fit_users=count,width=128,learning_rate=.00003,
        steps=512,patience=512 if count==256 else 128,validation_every=8,stages="f,fg,fgh",controls=True,
        seed=17,gpu=0,device="cpu")
    return SimpleNamespace(args=args,captures=captures,preparations=preparations,fits=actual_fits,loads=loads,
                           model=current,original=original,fit=fit,val=val)


@pytest.mark.parametrize("count",[64,128,256])
def test_actual_teacher_N_views_before_targets_and_standalone_cost(tmp_path,monkeypatch,count):
    fx=fixture(tmp_path,monkeypatch,count=count)
    result=driver.run(fx.args)
    assert [len(uids) for uids in fx.captures]==[count+16]
    assert [len(s) for s in fx.preparations]==[count,16]
    identity=result["identity"]
    assert identity["gradient_fitting_users"]==identity["teacher_fitting_users"]==count
    assert identity["candidate_catalog_source_users"]==identity["time_law_source_users"]==256
    assert [r["method_label"] for r in result["stages"]]==["f","f_extended","fg","fg_extended","fgh"]
    assert [r["actual"] for r in fx.fits]==[2,4,2,2,2]
    prep=driver.preparation_cost(fx.model,fx.preparations[0]+fx.preparations[1],
        dict(attention_backend="torch"),fx.args.sampling_manifest)
    assert result["calibration_preparation"]==prep
    total=prep["total_model_preparation_flops"]
    for row in result["stages"]:
        previous=total
        assert row["cumulative_calibration_flops_estimate"]==previous+row["total_calibration_flops_estimate"]
        if not row["method_label"].endswith("extended"):total=row["cumulative_calibration_flops_estimate"]


def states(folder):
    return {stage:load_variant(folder/f"{stage}.pt")[0].state_dict() for stage in ("f","fg","fgh")}


def assert_equal_states(left,right):
    assert left.keys()==right.keys()
    for stage in left:
        assert left[stage].keys()==right[stage].keys()
        assert all(torch.equal(left[stage][name],right[stage][name]) for name in left[stage]),stage


def test_partial_full_resume_and_controls_preserve_real_optimizer_trajectory(tmp_path,monkeypatch):
    fx=fixture(tmp_path,monkeypatch)
    driver.run(fx.args);fresh=states(fx.args.output)
    # Partial stage resume from the actual saved f, with fresh g/h training.
    partial=fx.args.output
    for label in ("f_extended","fg","fg_extended","fgh"):
        for suffix in ("pt","json"):(partial/f"{label}.{suffix}").unlink()
    driver.run(fx.args)
    assert_equal_states(fresh,states(partial))
    count=len(fx.fits)
    driver.run(SimpleNamespace(**{**vars(fx.args),"output":partial}))
    assert len(fx.fits)==count  # Complete resume performs no optimizer step.
    assert_equal_states(fresh,states(partial))
    for suffix in ("pt","json"):(partial/f"fgh.{suffix}").unlink()
    driver.run(fx.args)  # f and fg restored, then a newly trained h stage.
    assert_equal_states(fresh,states(partial))
    # Sibling controls must not change the primary f -> fg -> fgh path.
    without=tmp_path/"without_controls"
    driver.run(SimpleNamespace(**{**vars(fx.args),"output":without,"controls":False}))
    assert_equal_states(fresh,states(without))


def test_shared_preparation_reuses_only_same_edge_actual_N(tmp_path,monkeypatch):
    fx=fixture(tmp_path,monkeypatch)
    cache={};base=dict(vars(fx.args),stages="f",controls=False)
    first=driver.run(SimpleNamespace(**{**base,"output":tmp_path/"first"}),preparation_cache=cache)
    second=driver.run(SimpleNamespace(**{**base,"output":tmp_path/"other_view","gap_rule":"gap_empirical100"}),preparation_cache=cache)
    assert len(fx.loads)==1 and len(fx.captures)==1
    assert not first["stages"][0]["actual_search_preparation_reused"]
    assert second["stages"][0]["actual_search_preparation_reused"]
    # Reuse discounts actual search work, never independent-method model prep.
    assert first["calibration_preparation"]==second["calibration_preparation"]
    driver.run(SimpleNamespace(**{**base,"output":tmp_path/"otherN","fit_users":128}),preparation_cache=cache)
    assert len(fx.loads)==2 and [len(uids) for uids in fx.captures]==[80,144]
    # A different edge with the same UID roles is a different model/cache key.
    root=fx.args.input_root;second_folder=root/"edge_2";second_folder.mkdir()
    for path in (root/"edge_1").iterdir():(second_folder/path.name).write_bytes(path.read_bytes())
    binding=json.loads((second_folder/"inputs.json").read_text());binding["settings"]["edge"]=2
    (second_folder/"inputs.json").write_text(json.dumps(binding))
    manifest=json.loads(fx.args.sampling_manifest.read_text());manifest["edge"]=2
    manifest["sourcebinding"].update(sha256=sha256(second_folder/"inputs.json"),
        history_path=str(second_folder/"history.pack"),history_sha256=sha256(second_folder/"history.pack"))
    other_manifest=tmp_path/"sampling_edge2.json";other_manifest.write_text(json.dumps(manifest))
    driver.run(SimpleNamespace(**{**base,"output":tmp_path/"otherEdge","edge":2,
        "sampling_manifest":other_manifest}),preparation_cache=cache)
    assert len(fx.loads)==3 and [len(uids) for uids in fx.captures]==[80,144,80]
