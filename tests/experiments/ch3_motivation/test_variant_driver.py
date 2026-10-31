"""CPU driver smoke: real variants/checkpoints/targets, mocked experiment IO.

The optimizer is replaced by a deterministic parameter increment so the tests
can inspect which fitted branch each extended-training control inherits.
"""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code import run_probe_variants as driver
from experiments.ch3_motivation.code.probe_variants import (
    FullItemProbeLadder, LayerwiseKVProbeLadder, load_variant,
)
from hstu_kvcache.models import HSTU,HSTUConfig
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class CPUTorch:
    """Change only this driver's CUDA placement; tensor code remains real."""
    cuda=SimpleNamespace(set_device=lambda _:None,empty_cache=lambda:None)

    def __getattr__(self,name):
        return getattr(torch,name)

    @staticmethod
    def device(_):
        return torch.device("cpu")

    @staticmethod
    def manual_seed(seed):
        return torch.random.default_generator.manual_seed(seed)


def output_weight(probe):
    layer=probe.f[0]
    return layer.weights[-1] if hasattr(layer,"weights") else layer.network[-1].weight


def setup_experiment(tmp_path,monkeypatch,variant="full_item",candidate_source="feedback"):
    input_root=tmp_path/"inputs"
    edge=input_root/"edge_1"
    edge.mkdir(parents=True)
    population=dict(fit=[11,12],validation=[21])
    (input_root/"population.json").write_text(json.dumps(population))
    for name in ("history","requests"):
        (edge/f"{name}.pack").write_text(name)
    binding=dict(population_sha256=sha256(input_root/"population.json"),
        settings=dict(attention_backend="torch"),
        **{name:dict(path=f"{name}.pack",sha256=sha256(edge/f"{name}.pack"))
           for name in ("history","requests")})
    (edge/"inputs.json").write_text(json.dumps(binding))
    (edge/"calibration_feedback_candidates.json").write_text(json.dumps(dict(
        population_sha256=binding["population_sha256"],
        recent_items={str(uid):[4,2] for uid in (11,12,21)})))

    torch.manual_seed(812)
    cfg=HSTUConfig(num_items=260,num_behaviors=2,hidden_size=64,num_heads=2,
        num_layers=1,max_seq_len=4,input_dropout=0.,attn_dropout=0.)
    parent=HSTU(cfg).eval().requires_grad_(False)
    current=copy.deepcopy(parent)
    with torch.no_grad():
        current.item_emb.weight[1:].add_(torch.randn_like(current.item_emb.weight[1:])*.02)
    events=[]
    fits=[]
    builds=[]
    real_prepare=driver.prepare_training_samples
    real_build=driver.build_variant

    def capture(old,new,histories,uids,settings,device):
        assert old is parent and new is current
        assert uids==[11,12,21]
        assert settings["calibration_queries_per_user"]==4
        events.append("capture")
        ids=torch.tensor([[1,3,4]])
        behaviors=torch.ones_like(ids)
        delta=torch.ones_like(ids,dtype=torch.float32)
        samples={}
        for uid in uids:
            samples[uid]=dict(uid=uid,parent_cache=old.compute_kv(ids,behaviors,delta),
                teacher_cache=new.compute_kv(ids,behaviors,delta),item_ids=ids.clone(),
                history_count=3,candidates=torch.full((1,4),257,dtype=torch.long),
                query_delta=torch.ones(1))
        return samples

    def prepare(model,samples):
        assert events[0]=="capture"
        for sample in samples:
            values=sample["candidates"][0].tolist()
            # This includes the final known ID K=4 and excludes capture's
            # sentinel, before real teacher-score/read targets are computed.
            assert set(values)=={1,2,3,4}
            assert values[:2]==([4,2] if candidate_source=="feedback" else [4,3])
        events.append(("targets",[sample["uid"] for sample in samples]))
        return real_prepare(model,samples)

    def build(name,model,**kwargs):
        value=real_build(name,model,**kwargs)
        builds.append(value)
        return value

    def fit(model,module,train,val,*,steps,progress_callback,**kwargs):
        assert [sample["uid"] for sample in train]==[11,12]
        assert [sample["uid"] for sample in val]==[21]
        assert all("teacher_scores" in sample for sample in train+val)
        before=float(output_weight(module).detach().flatten()[0])
        with torch.no_grad():
            output_weight(module).add_(steps)
        fits.append(dict(stage=module.stage,before=before,steps=steps,
                         after=float(output_weight(module).detach().flatten()[0]),cls=type(module)))
        progress_callback(dict(step=steps,validation_loss=1.))
        return dict(total_calibration_flops_estimate=10*steps,completed_steps=steps,
            seed=kwargs["seed"],training_users=len(train),validation_users=len(val),history=[])

    monkeypatch.setattr(driver,"torch",CPUTorch())
    monkeypatch.setattr(driver,"load_edge",lambda _: (binding,object(),{}))
    monkeypatch.setattr(driver,"load_models",lambda *_: (parent,current,{"old":"fixture-old","current":"fixture-current"}))
    monkeypatch.setattr(driver,"capture_samples",capture)
    monkeypatch.setattr(driver,"prepare_training_samples",prepare)
    monkeypatch.setattr(driver,"build_variant",build)
    monkeypatch.setattr(driver,"fit_probe",fit)
    args=SimpleNamespace(input_root=input_root,output=tmp_path/"output",edge=1,
        variant=variant,gpu=0,seed=17,fit_users=2,queries=4,width=7,depth=1,steps=3,
        learning_rate=.0001,read_weight=1.,score_weight=1.,patience=0,validation_every=1,
        stages="f,fg,fgh",controls=True,candidate_source=candidate_source,recent_fraction=.5)
    return args,fits,events,builds


@pytest.mark.parametrize("variant,candidate_source",[
    ("full_item","feedback"),("layerwise_kv","feedback"),("full_item","listen")])
def test_driver_targets_candidates_and_independent_extended_branches(tmp_path,monkeypatch,variant,candidate_source):
    args,fits,events,builds=setup_experiment(tmp_path,monkeypatch,variant,candidate_source)
    driver.run(args)
    assert events==["capture",("targets",[11,12]),("targets",[21])]
    expected_type=FullItemProbeLadder if variant=="full_item" else LayerwiseKVProbeLadder
    assert len(builds)==1 and type(builds[0]) is expected_type
    assert [(row["stage"],row["before"],row["steps"],row["after"]) for row in fits]==[
        ("f",0.,3,3.),("f",3.,6,9.),("fg",3.,3,6.),("fg",6.,3,9.),("fgh",6.,3,9.)]
    assert all(row["cls"] is expected_type for row in fits)
    summary=json.loads((args.output/"summary.json").read_text())
    labels=["f","f_extended","fg","fg_extended","fgh"]
    assert [row["method_label"] for row in summary["stages"]]==labels
    shared=summary["stages"][0]["source_teacher_and_target_preparation_flops"]
    assert shared>0
    assert [row["cumulative_calibration_flops_estimate"]-shared for row in summary["stages"]]==[30,90,60,90,90]
    for label,marker in zip(labels,(3.,9.,6.,9.,9.),strict=True):
        checkpoint=args.output/f"{label}.pt"
        loaded,metadata=load_variant(checkpoint)
        row=json.loads((args.output/f"{label}.json").read_text())
        assert type(loaded) is expected_type
        assert float(output_weight(loaded).detach().flatten()[0])==marker
        assert metadata["variant"]==variant and metadata["method_label"]==label
        assert row["checkpoint_sha256"]==sha256(checkpoint)
        assert row["quality_auc_computed"] is False


@pytest.mark.parametrize("variant",["full_item","layerwise_kv"])
def test_complete_driver_resume_loads_every_branch_without_refitting(tmp_path,monkeypatch,variant):
    args,fits,_,_=setup_experiment(tmp_path,monkeypatch,variant)
    driver.run(args)
    original=json.loads((args.output/"summary.json").read_text())
    checkpoint_hashes={path.name:sha256(path) for path in args.output.glob("*.pt")}
    assert len(fits)==5

    def no_refit(*args,**kwargs):
        pytest.fail("a complete saved variant branch must not be fitted on resume")

    monkeypatch.setattr(driver,"fit_probe",no_refit)
    driver.run(args)
    assert json.loads((args.output/"summary.json").read_text())==original
    assert {path.name:sha256(path) for path in args.output.glob("*.pt")}==checkpoint_hashes
    args.width+=1
    with pytest.raises(ValueError,match="configuration changed"):
        driver.run(args)


def test_shared_preparation_across_fair_variant_comparisons(tmp_path,monkeypatch):
    args,fits,events,_=setup_experiment(tmp_path,monkeypatch)
    shared={}
    driver.run(args,preparation_cache=shared)
    first_events=list(events)
    args.output=tmp_path/"layerwise"
    args.variant="layerwise_kv"
    driver.run(args,preparation_cache=shared)
    assert events==first_events  # no second old/teacher/target construction
    summary=json.loads((args.output/"summary.json").read_text())
    assert all(row["actual_search_preparation_reused"] for row in summary["stages"])
    assert len(fits)==10
