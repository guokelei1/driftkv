"""Actual optimizer, warm-start provenance, and partial resume of the supplement."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    run_probe_calibration_init as reference, run_probe_frozen_predecessors as driver, probes,
)
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.common.artifacts import sha256
import tests.experiments.ch3_motivation.test_run_probe_calibration as integration


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch')
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(integration,'driver',reference)
    fx=integration.fixture(tmp_path,monkeypatch)
    fx.args.controls=False;fx.args.h_initialization='standard'
    reference.run(fx.args)
    source=fx.args.output/'f.pt'
    args=SimpleNamespace(**{**vars(fx.args),'source_f':source,'source_f_sha256':sha256(source),'output':tmp_path/'new_term'})
    for name in ('load_models','restore_pack','capture_samples','prepare_training_samples'):
        monkeypatch.setattr(driver,name,getattr(reference,name))
    original=probes.fit_probe
    calls=[]
    def bounded(model,probe,train,validation,**kwargs):
        kwargs.update(steps=2,validation_every=1,patience=0)
        calls.append(probe.stage)
        return original(model,probe,train,validation,**kwargs)
    monkeypatch.setattr(probes,'fit_probe',bounded)
    return fx,args,calls


def test_actual_inheritance_freezing_cost_and_partial_resume(tmp_path,monkeypatch):
    fx,args,calls=setup(tmp_path,monkeypatch)
    result=driver.run(args)
    assert calls==['fg','fgh']
    identity=result['identity']
    assert identity['fitting_driver']=='run_probe_frozen_predecessors.py'
    assert identity['training_protocol']['format']=='e5_new_term_only_v1'
    source,source_meta=load_variant(args.source_f)
    loaded={name:load_variant(args.output/f'{name}.pt') for name in ('f','fg','fgh')}
    assert all(torch.equal(v,loaded['f'][0].state_dict()[key]) for key,v in source.state_dict().items())
    assert loaded['f'][1]['cumulative_calibration_flops_estimate']==source_meta['cumulative_calibration_flops_estimate']
    assert loaded['f'][1]['training_protocol']['inherited_source_only']
    total=source_meta['cumulative_calibration_flops_estimate']
    for stage,prior in (('fg','f'),('fgh','fg')):
        current,metadata=loaded[stage]
        prior_state=loaded[prior][0].state_dict()
        assert all(torch.equal(v,current.state_dict()[k]) for k,v in prior_state.items())
        assert metadata['training_protocol']['trainable_term']=={'fg':'g','fgh':'h'}[stage]
        assert metadata['execution_adapter']['binding']['format']=='resident_probe_fitting_v1'
        total+=metadata['total_calibration_flops_estimate']
        assert metadata['cumulative_calibration_flops_estimate']==total
        assert metadata['identity']==identity and metadata['quality_auc_computed'] is False
    wanted={k:v.clone() for k,v in loaded['fgh'][0].state_dict().items()}
    wanted_history=loaded['fgh'][1]['history']
    (args.output/'fgh.pt').unlink();(args.output/'fgh.json').unlink()
    driver.run(args)
    assert calls==['fg','fgh','fgh']
    resumed,meta=load_variant(args.output/'fgh.pt')
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in wanted.items())
    assert meta['history']==wanted_history
    assert source_meta['identity']['fitting_driver']=='run_probe_calibration_init.py'


def test_changed_source_hash_is_rejected_before_model_or_data_capture(tmp_path,monkeypatch):
    fx,args,calls=setup(tmp_path,monkeypatch)
    loads=len(fx.loads);captures=len(fx.captures)
    args.source_f_sha256='0'*64
    with pytest.raises(ValueError,match='source f checkpoint changed'):driver.run(args)
    assert len(fx.loads)==loads and len(fx.captures)==captures and not calls
