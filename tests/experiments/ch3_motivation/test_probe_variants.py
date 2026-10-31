"""Information access, nesting, training and replay contracts of strong probes."""
import copy
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.probes import ProbeConfig, ProbeLadder, fit_probe
from experiments.ch3_motivation.code.probe_variants import (
    FullItemProbeLadder, LayerwiseKVProbeLadder, build_variant, load_variant, save_variant,
)
from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory
from experiments.ch3_motivation.code.probe_fast import FastProbeTrajectory, GraphProbePool
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTU,HSTUConfig,HSTUKVCache


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(839)
    cfg=HSTUConfig(num_items=32,num_behaviors=2,hidden_size=8,num_heads=2,num_layers=2,
        max_seq_len=6,input_dropout=0.,attn_dropout=0.)
    parent=HSTU(cfg).eval().requires_grad_(False)
    current=copy.deepcopy(parent)
    with torch.no_grad():
        current.item_emb.weight[1:].add_(torch.randn_like(current.item_emb.weight[1:])*.03)
    ids=torch.arange(1,6)[None]
    behaviors=torch.ones_like(ids)
    delta=torch.ones_like(ids,dtype=torch.float32)
    sample=dict(parent_cache=parent.compute_kv(ids,behaviors,delta),
        teacher_cache=current.compute_kv(ids,behaviors,delta),item_ids=ids,
        candidates=torch.tensor([[13,17,19]]),query_delta=torch.ones(1))
    return parent,current,sample


def nonzero(probe):
    with torch.no_grad():
        for modules in (probe.f,probe.g,probe.h):
            if modules is not None:
                for layer in modules:
                    if hasattr(layer,"weights"):
                        layer.weights[-1].normal_(std=.01)
                    else:
                        layer.network[-1].weight.normal_(std=.01)


def test_full_item_is_visible_to_every_head_including_nonlocal_coordinates():
    _,model,sample=fixture()
    full=FullItemProbeLadder.from_model(model,hidden_width=9,depth=1,normalize=False)
    local=ProbeLadder(ProbeConfig.from_model(model,hidden_width=9,depth=1,normalize=False))
    assert full.cfg.item_dim==model.cfg.hidden_size==8
    assert full.cfg.row_dim==16
    embedded=model.lookup_item_embeddings(sample["item_ids"]).detach()
    features=full._items(model,sample["item_ids"])
    torch.testing.assert_close(features[:,0],embedded)
    torch.testing.assert_close(features[:,1],embedded)
    changed=embedded.clone()
    changed[...,7]+=.5  # Outside head zero's local four-coordinate slice.
    nonzero(full)
    nonzero(local)
    with torch.no_grad():
        a=full.prepare_cache(model,sample["parent_cache"],sample["item_ids"],item_embeddings=embedded)
        b=full.prepare_cache(model,sample["parent_cache"],sample["item_ids"],item_embeddings=changed)
        assert (a.k[...,:4]-b.k[...,:4]).abs().max()>1e-7
        local_a=local.prepare_cache(model,sample["parent_cache"],sample["item_ids"],item_embeddings=embedded)
        local_b=local.prepare_cache(model,sample["parent_cache"],sample["item_ids"],item_embeddings=changed)
        torch.testing.assert_close(local_a.k[...,:4],local_b.k[...,:4],atol=0,rtol=0)


@pytest.mark.parametrize("kind",["full_item","layerwise_kv"])
def test_variant_nested_zero_additions_preserve_f_and_statistics(kind):
    _,model,sample=fixture()
    probe=build_variant(kind,model,hidden_width=7,depth=1)
    probe.fit_input_statistics(model,[sample])
    nonzero(probe)
    fg=probe.expanded("fg")
    fgh=fg.expanded("fgh")
    assert type(probe) is type(fg) is type(fgh)
    predictions=[]
    with torch.no_grad():
        for value in (probe,fg,fgh):
            result,_=value.score(model,sample["parent_cache"],sample["item_ids"],sample["candidates"],sample["query_delta"])
            predictions.append(result)
        torch.testing.assert_close(predictions[0],predictions[1])
        torch.testing.assert_close(predictions[0],predictions[2])
        for name,value in probe.state_dict().items():
            torch.testing.assert_close(value,fgh.state_dict()[name],atol=0,rtol=0)
    with pytest.raises(ValueError):
        fgh.expanded("f")


def test_layerwise_f_can_map_another_heads_value_into_head_zero_key():
    _,model,sample=fixture()
    probe=LayerwiseKVProbeLadder.from_model(model,hidden_width=5,depth=1,normalize=False)
    assert probe.cfg.row_dim==24  # Full K8 + V8 + item8.
    with torch.no_grad():
        for layer in probe.f:
            for module in layer.network:
                if isinstance(module,torch.nn.Linear):
                    module.weight.zero_(); module.bias.zero_()
            # Head1's last value -> hidden0 -> head0's first key.
            layer.network[0].weight[0,15]=1.
            layer.network[-1].weight[0,0]=1.
        original=probe.prepare_cache(model,sample["parent_cache"],sample["item_ids"])
        modified=HSTUKVCache(sample["parent_cache"].k.clone(),sample["parent_cache"].v.clone(),5)
        modified.v[...,7]+=.5
        altered=probe.prepare_cache(model,modified,sample["item_ids"])
    assert (original.k[...,0]-altered.k[...,0]).abs().max()>1e-4


@pytest.mark.parametrize("kind",["head","full_item","layerwise_kv"])
def test_variant_checkpoint_roundtrip_keeps_class_and_predictions(tmp_path,kind):
    _,model,sample=fixture()
    probe=build_variant(kind,model,hidden_width=7,depth=1,stage="fgh")
    probe.fit_input_statistics(model,[sample])
    nonzero(probe)
    path=tmp_path/f"{kind}.pt"
    save_variant(path,probe,dict(test="roundtrip",role="synthetic"))
    restored,metadata=load_variant(path,"cpu")
    assert type(restored) is type(probe)
    assert metadata==dict(test="roundtrip",role="synthetic")
    assert restored.cfg.row_dim==probe.cfg.row_dim
    with torch.no_grad():
        expected,_=probe.score(model,sample["parent_cache"],sample["item_ids"],sample["candidates"],sample["query_delta"])
        actual,_=restored.score(model,sample["parent_cache"],sample["item_ids"],sample["candidates"],sample["query_delta"])
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


@pytest.mark.parametrize("kind",["full_item","layerwise_kv"])
def test_variants_fit_all_stages_and_leave_backbone_frozen(kind):
    _,model,sample=fixture()
    validation=dict(sample,candidates=torch.tensor([[14,18,21]]))
    probe=build_variant(kind,model,hidden_width=7,depth=1)
    before={name:value.clone() for name,value in model.state_dict().items()}
    for stage in ("f","fg","fgh"):
        probe=probe if stage=="f" else probe.expanded(stage)
        report=fit_probe(model,probe,[sample],[validation],steps=2,validation_every=1,
                         learning_rate=.0001,patience=0)
        assert report["completed_steps"]==2
        assert all(torch.isfinite(torch.tensor(row["validation_loss"])) for row in report["history"])
        assert report["fit_pool_loss_scales"]["users"]==1
        assert report["arithmetic"]["training_probe_forward_flops"]>0
    assert all(torch.equal(value,before[name]) for name,value in model.state_dict().items())
    assert all(value.grad is None for value in model.parameters())


@pytest.mark.parametrize("kind",["full_item","layerwise_kv"])
def test_nonzero_variant_append_and_fixed_mask_replay_match_reference(kind):
    parent,model,sample=fixture()
    probe=build_variant(kind,model,hidden_width=7,depth=1,stage="fgh")
    probe.fit_input_statistics(model,[sample])
    nonzero(probe)
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=5,append_band_size=2)
    times=np.array([86390,86391,86392,86393,86394,86400,86401,86402])
    events=times,np.arange(1,9),np.ones(8,dtype=np.int64)
    pool=GraphProbePool(model,probe,window=5,capture=False)
    fast=FastProbeTrajectory(parent,model,probe,events,cfg,pool=pool)
    reference=ProbeTrajectory(parent,model,probe,events,cfg)
    with torch.no_grad():
        for timestamp in (86400,86401,86402,86403):
            fast.advance(timestamp); reference.advance(timestamp)
            delta=torch.tensor([timestamp-reference.native.last_timestamp],dtype=torch.float32)
            actual=fast.score(sample["candidates"],delta)
            wanted=reference.score(sample["candidates"],delta)
            torch.testing.assert_close(actual,wanted,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(fast.cache.k,reference.cache.k,atol=1e-6,rtol=1e-5)
    assert fast.report()["counts"]["prepare_calls"]==1
    assert fast.report()["counts"]["fixed_graph_append_calls"]==3


def test_layerwise_cost_matches_its_single_joint_row_mlp():
    _,model,_=fixture()
    probe=LayerwiseKVProbeLadder.from_model(model,hidden_width=7,depth=1,normalize=False)
    cost=probe.forward_flops(5,3)
    # L2 * N5: one [row24 -> hidden7 -> output16] MLP, plus residual16.
    per_row=2*24*7+7+2*7*16+16+2*7+16
    assert cost["f_prepare_flops"]==2*5*per_row
    assert cost["total_correction_forward_flops"]==cost["f_prepare_flops"]


def test_actual_320_width_ten_head_geometry_and_per_head_native_output_scales():
    torch.manual_seed(921)
    model=HSTU(HSTUConfig(num_items=12,num_behaviors=2,hidden_size=320,num_heads=10,
        num_layers=1,max_seq_len=4,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    head_scale=torch.arange(1,11).repeat_interleave(32)
    source=HSTUKVCache(torch.randn(1,1,4,320)*head_scale,
                       torch.randn(1,1,4,320)*(head_scale+20),4)
    ids=torch.arange(1,5)[None]
    sample=dict(parent_cache=source,teacher_cache=source,item_ids=ids,
                candidates=torch.tensor([[7,8]]),query_delta=torch.ones(1))
    full=FullItemProbeLadder.from_model(model,hidden_width=8,depth=1)
    layerwise=LayerwiseKVProbeLadder.from_model(model,hidden_width=8,depth=1,stage="fgh")
    assert full.cfg.item_dim==layerwise.cfg.item_dim==320
    assert full.cfg.row_dim==384
    assert layerwise.cfg.row_dim==960
    layerwise.fit_input_statistics(model,[sample])
    expected_k=source.k[0].std((0,1),unbiased=False).reshape(10,32)
    expected_v=source.v[0].std((0,1),unbiased=False).reshape(10,32)
    torch.testing.assert_close(layerwise.kv_output_scale[0],torch.cat((expected_k,expected_v),-1),
                               atol=2e-5,rtol=2e-5)
    with torch.no_grad():
        actual,_=layerwise.score(model,source,ids,sample["candidates"],sample["query_delta"])
        wanted,_=score(model,source,sample["candidates"],sample["query_delta"])
    torch.testing.assert_close(actual,wanted,atol=1e-6,rtol=1e-5)


def test_loader_accepts_actual_untagged_head_checkpoint_and_rejects_untagged_new_variants(tmp_path):
    _,model,sample=fixture()
    for kind in ("head","full_item","layerwise_kv"):
        probe=build_variant(kind,model,hidden_width=7,depth=1)
        checkpoint=tmp_path/f"untagged_{kind}.pt"
        torch.save(dict(config=asdict(probe.cfg),stage=probe.stage,state_dict=probe.state_dict(),
                        metadata=dict(existing_head_format=True)),checkpoint)
        if kind!="head":
            with pytest.raises(ValueError,match="explicit tag"):
                load_variant(checkpoint)
        else:
            loaded,metadata=load_variant(checkpoint)
            assert type(loaded) is ProbeLadder
            assert metadata["existing_head_format"]
            with torch.no_grad():
                actual,_=loaded.score(model,sample["parent_cache"],sample["item_ids"],
                    sample["candidates"],sample["query_delta"])
                expected,_=probe.score(model,sample["parent_cache"],sample["item_ids"],
                    sample["candidates"],sample["query_delta"])
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
