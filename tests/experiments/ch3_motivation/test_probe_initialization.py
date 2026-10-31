"""Initialization changes only new h features; paired real fitting is causal."""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    run_probe_calibration_init as driver, run_probe_calibration as phase_a,
)
from experiments.ch3_motivation.code.probe_initialization import apply_h_initialization, initialization_spec
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant
from hstu_kvcache.models import HSTU,HSTUConfig
import tests.experiments.ch3_motivation.test_run_probe_calibration as integration


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("variant",["head","full_item","layerwise_kv"])
def test_only_h_first_weight_changes_and_zero_output_preserves_fg(variant):
    torch.manual_seed(71)
    model=HSTU(HSTUConfig(num_items=40,num_behaviors=2,hidden_size=8,num_heads=2,num_layers=2,
        max_seq_len=8,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    ids=torch.tensor([[1,2,3,4,5]])
    cache=model.compute_kv(ids,torch.ones_like(ids),torch.ones_like(ids,dtype=torch.float32))
    fg=build_variant(variant,model,stage="fg",hidden_width=8,depth=2)
    with torch.no_grad():
        for group in (fg.f,fg.g):
            for layer in group:
                (layer.weights[-1] if hasattr(layer,"weights") else layer.network[-1].weight).normal_(0,.002)
    standard=fg.expanded("fgh");balanced=copy.deepcopy(standard)
    before={name:value.clone() for name,value in standard.state_dict().items()}
    normal=apply_h_initialization(standard,"standard")
    spec=apply_h_initialization(balanced,"balanced")
    assert normal["weight_rescale_flops"]==0
    assert spec["weight_rescale_flops"]==sum(network.weights[0].numel() for network in balanced.h)
    for name,value in before.items():
        assert torch.equal(value,standard.state_dict()[name])
        if name.startswith("h.") and name.endswith("weights.0"):
            changed=balanced.state_dict()[name]
            d=fg.cfg.head_dim
            torch.testing.assert_close(changed[:,:d],value[:,:d]*spec["query_gain"],rtol=0,atol=0)
            torch.testing.assert_close(changed[:,d:],value[:,d:]*spec["row_gain"],rtol=0,atol=0)
        else:assert torch.equal(value,balanced.state_dict()[name])
    candidates=torch.tensor([[12,17,20]]);delta=torch.tensor([[1.,20.,600.]])
    expected=fg.score(model,cache,ids,candidates,delta)[0]
    for probe in (standard,balanced):
        actual=probe.score(model,cache,ids,candidates,delta)[0]
        torch.testing.assert_close(actual,expected,atol=5e-7,rtol=5e-6)


def test_actual_large_gain_and_trained_output_guard():
    spec=initialization_spec(32,384,"balanced")
    assert spec["query_gain"]==math.sqrt(6.5)
    assert spec["row_gain"]==math.sqrt(416/768)
    assert spec["expected_query_energy_fraction"]==.5
    model=HSTU(HSTUConfig(num_items=20,num_behaviors=2,hidden_size=8,num_heads=2,num_layers=1,max_seq_len=8))
    probe=build_variant("full_item",model,stage="fgh",hidden_width=8,depth=1)
    with torch.no_grad():probe.h[0].weights[-1].fill_(.01)
    with pytest.raises(ValueError,match="untouched zero output"):apply_h_initialization(probe,"balanced")


def setup_driver(tmp_path,monkeypatch):
    monkeypatch.setattr(integration,"driver",driver)
    fx=integration.fixture(tmp_path,monkeypatch)
    fx.args.h_initialization="paired"
    return fx


def test_pair_real_optimizer_same_fg_identity_and_initialization_cost(tmp_path,monkeypatch):
    fx=setup_driver(tmp_path,monkeypatch)
    result=driver.run_paired(fx.args)
    assert result["status"]=="paired_fitted" and result["f_and_fg_exactly_shared"]
    assert len(fx.loads)==1 and len(fx.captures)==1
    assert [r["actual"] for r in fx.fits]==[2,4,2,2,2]*2
    for mode in ("standard","balanced"):
        path=fx.args.output/mode
        for label in ("f","fg","fgh"):
            probe,metadata=load_variant(path/f"{label}.pt")
            identity=metadata["identity"]
            assert identity["fitting_driver"]=="run_probe_calibration_init.py"
            assert identity["h_initialization"]==mode
            assert identity["initialization_helper_sha256"]==identity["h_initialization_spec"]["helper_sha256"]
            assert identity["candidate_catalog_source_users"]==identity["time_law_source_users"]==256
            cost=metadata["h_initialization_weight_rescale_flops"]
            assert (cost>0)==(label=="fgh" and mode=="balanced")
            if label=="fgh":
                _,fg_metadata=load_variant(path/"fg.pt")
                assert metadata["cumulative_calibration_flops_estimate"]==fg_metadata["cumulative_calibration_flops_estimate"]+metadata["total_calibration_flops_estimate"]+cost
    left,_=load_variant(fx.args.output/"standard/fgh.pt")
    right,_=load_variant(fx.args.output/"balanced/fgh.pt")
    assert not torch.equal(left.h[0].weights[0],right.h[0].weights[0])


def test_both_modes_partial_and_complete_resume_are_exact(tmp_path,monkeypatch):
    fx=setup_driver(tmp_path,monkeypatch)
    driver.run_paired(fx.args)
    states={mode:integration.states(fx.args.output/mode) for mode in ("standard","balanced")}
    for mode in states:
        for label in ("f_extended","fg","fg_extended","fgh"):
            for suffix in ("pt","json"):(fx.args.output/mode/f"{label}.{suffix}").unlink()
    driver.run_paired(fx.args)
    for mode in states:integration.assert_equal_states(states[mode],integration.states(fx.args.output/mode))
    executed=len(fx.fits)
    driver.run_paired(fx.args)
    assert len(fx.fits)==executed
    for mode in states:
        integration.assert_equal_states(states[mode],integration.states(fx.args.output/mode))
        for suffix in ("pt","json"):(fx.args.output/mode/f"fgh.{suffix}").unlink()
    driver.run_paired(fx.args)
    for mode in states:integration.assert_equal_states(states[mode],integration.states(fx.args.output/mode))


def test_standard_checkpoint_cannot_resume_as_balanced(tmp_path,monkeypatch):
    fx=setup_driver(tmp_path,monkeypatch)
    args=SimpleNamespace(**{**vars(fx.args),"h_initialization":"standard"})
    driver.run(args)
    with pytest.raises(ValueError,match="identity changed"):
        driver.run(SimpleNamespace(**{**vars(args),"h_initialization":"balanced"}))


def test_standard_real_fitting_matches_phase_a_weights_but_has_own_driver_identity(tmp_path,monkeypatch):
    fx=setup_driver(tmp_path,monkeypatch)
    for name in ("load_models","restore_pack","capture_samples","prepare_training_samples","fit_probe"):
        monkeypatch.setattr(phase_a,name,getattr(driver,name))
    standard=SimpleNamespace(**{**vars(fx.args),"h_initialization":"standard"})
    driver.run(standard)
    original=SimpleNamespace(**{**vars(fx.args),"output":tmp_path/"phase_a"})
    phase_a.run(original)
    integration.assert_equal_states(integration.states(standard.output),integration.states(original.output))
    _,new_metadata=load_variant(standard.output/"fgh.pt")
    _,old_metadata=load_variant(original.output/"fgh.pt")
    assert new_metadata["identity"]["fitting_driver"]=="run_probe_calibration_init.py"
    assert old_metadata["identity"]["fitting_driver"]=="run_probe_calibration.py"
    assert new_metadata["total_calibration_flops_estimate"]==old_metadata["total_calibration_flops_estimate"]
    assert new_metadata["cumulative_calibration_flops_estimate"]==old_metadata["cumulative_calibration_flops_estimate"]
