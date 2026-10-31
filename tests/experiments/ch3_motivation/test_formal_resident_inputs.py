"""Real CPU optimizer/resume for explicit residency; no GPU or quality scores."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    probes, resident_probe_fitting, sweep_probe_calibration_formal as sweep,
    plan_probe_quality as quality,
)
from experiments.ch3_motivation.code.probe_variants import load_variant, save_variant
from experiments.ch3_motivation.code.probes import fit_probe as real_fit
from hstu_kvcache.common.artifacts import sha256
from tests.experiments.ch3_motivation.test_sweep_probe_calibration_formal import fixture
from tests.experiments.ch3_motivation.test_plan_probe_quality import fixture as audit_fixture
import tests.experiments.ch3_motivation.test_run_probe_calibration as integration


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch')
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('name',list(sweep.DRIVERS))
def test_actual_main_same_math_full_partial_resume_and_preprep_mode_guards(tmp_path,monkeypatch,name):
    setup=fixture(tmp_path,monkeypatch,name);original=setup.driver.fit_probe
    reference=sweep._execute(setup.args,setup.plan,[setup.job],dict(test='reference'),{})
    reference_folder=sweep._folder(setup.args.output,'main',setup.job)
    reference_states=integration.states(reference_folder)
    reference_rng=torch.random.get_rng_state().clone()
    # The fixture uses the real optimizer for a bounded two-step main stage.
    # The resident helper still calls that same core with identical kwargs.
    calls=[]
    def bounded_core(model,probe,train,validation,**kwargs):
        requested=kwargs['steps'];kwargs['steps']=2*(requested//512)
        kwargs['validation_every']=1;kwargs['patience']=0
        value=real_fit(model,probe,train,validation,**kwargs)
        calls.append((probe.stage,value['completed_steps']))
        return value
    monkeypatch.setattr(probes,'fit_probe',bounded_core)
    args=SimpleNamespace(**{**vars(setup.args),'resident_inputs':True,'output':tmp_path/'resident'})
    resident=sweep._execute(args,setup.plan,[setup.job],dict(test='resident'),{})
    folder=sweep._folder(args.output,'main',setup.job)
    assert setup.driver.fit_probe is original
    integration.assert_equal_states(reference_states,integration.states(folder))
    assert torch.equal(reference_rng,torch.random.get_rng_state())
    binding=quality.resident_execution_binding()
    for expected,actual in zip(reference['completed'][0]['actual_weight_audit'],resident['completed'][0]['actual_weight_audit'],strict=True):
        method=actual['method_label']
        _,left=load_variant(reference_folder/f'{method}.pt')
        _,right=load_variant(folder/f'{method}.pt')
        assert left['identity']==right['identity']  # Original numerical driver/controls remain the identity.
        assert left['operation_counts']==right['operation_counts'] and left['arithmetic']==right['arithmetic']
        assert left['cumulative_calibration_flops_estimate']==right['cumulative_calibration_flops_estimate']
        adapter=right['execution_adapter']
        assert adapter==actual['execution_adapter'] and adapter['binding']==binding
        assert adapter['metrics']['model_device']=='cpu' and adapter['metrics']['cuda_memory'] is None
        stage=json.loads((folder/f'{method}.json').read_text())
        assert stage['execution_adapter']==adapter
        full=json.loads((folder/'summary.json').read_text())
        assert next(row for row in full['stages'] if row['method_label']==method)['execution_adapter']==adapter
        assert expected.get('execution_adapter') is None
    assert resident['completed'][0]['expected_execution_adapter']=={'binding':binding}
    count=len(calls)
    sweep._execute(args,setup.plan,[setup.job],dict(test='resident'),{})
    assert len(calls)==count and setup.driver.fit_probe is original
    before=integration.states(folder)
    for method in ('fg','fg_extended','fgh'):
        for suffix in ('pt','json'):(folder/f'{method}.{suffix}').unlink()
    sweep._execute(args,setup.plan,[setup.job],dict(test='resident'),{})
    integration.assert_equal_states(before,integration.states(folder))
    assert setup.driver.fit_probe is original
    # All three cases must fail before any model/cache preparation occurs.
    loads=len(setup.fx.loads)
    with pytest.raises(ValueError,match='missing.*execution adapter'):
        sweep._execute(SimpleNamespace(**{**vars(setup.args),'resident_inputs':True}),setup.plan,[setup.job],{}, {})
    with pytest.raises(ValueError,match='requires reference fitting'):
        sweep._execute(SimpleNamespace(**{**vars(args),'resident_inputs':False}),setup.plan,[setup.job],{}, {})
    bad=torch.load(folder/'f.pt',map_location='cpu',weights_only=False)
    bad['metadata']['execution_adapter']['binding']['helper_source']['sha256']='0'*64
    torch.save(bad,folder/'f.pt')
    with pytest.raises(ValueError,match='execution adapter binding differs'):
        sweep._execute(args,setup.plan,[setup.job],{}, {})
    assert len(setup.fx.loads)==loads and setup.driver.fit_probe is original


def test_actual_patience0_curve512_fullcap_resume_and_stage_isolation(tmp_path,monkeypatch):
    setup=fixture(tmp_path,monkeypatch,'run_probe_calibration_init.py')
    # Curves execute their actual full512 cap in both modes, unlike bounded
    # main diagnostics. No hook or altered step budget is used for this test.
    monkeypatch.setattr(setup.driver,'fit_probe',real_fit)
    original=setup.driver.fit_probe
    job=dict(setup.job,name='curve_fit64',seed=23,stages='f',controls=False,patience=0)
    base=SimpleNamespace(**{**vars(setup.args),'family':'curve','output':tmp_path/'reference'})
    reference=sweep._execute(base,setup.plan,[job],{}, {})
    args=SimpleNamespace(**{**vars(base),'resident_inputs':True,'output':tmp_path/'resident'})
    resident=sweep._execute(args,setup.plan,[job],{}, {})
    assert setup.driver.fit_probe is original
    path=sweep._folder(args.output,'curve',job)/'f.pt'
    expected,metadata=load_variant(sweep._folder(base.output,'curve',job)/'f.pt')
    actual,placed=load_variant(path)
    assert all(torch.equal(value,actual.state_dict()[key]) for key,value in expected.state_dict().items())
    assert placed['completed_steps']==metadata['completed_steps']==512
    assert placed['identity']==metadata['identity'] and placed['identity']['seed']==23
    assert placed['identity']['stages']=='f' and placed['identity']['controls'] is False
    assert placed['operation_counts']==metadata['operation_counts'] and placed['arithmetic']==metadata['arithmetic']
    before=sha256(path)
    def forbidden(*args,**kwargs):raise AssertionError('complete curve resume attempted fitting')
    monkeypatch.setattr(probes,'fit_probe',forbidden)
    resumed=sweep._execute(args,setup.plan,[job],{}, {})
    assert sha256(path)==before and resumed['points'][0]['completed_steps']==512
    assert setup.driver.fit_probe is original
    assert not (Path(args.output)/'main').exists()


def test_scoped_alias_restored_even_when_driver_raises(tmp_path,monkeypatch):
    setup=fixture(tmp_path,monkeypatch,'run_probe_calibration.py')
    original=setup.driver.fit_probe
    other=sweep.DRIVERS['run_probe_calibration_init.py'].fit_probe
    def failure(*args,**kwargs):
        assert setup.driver.fit_probe is resident_probe_fitting.fit_probe_resident
        assert sweep.DRIVERS['run_probe_calibration_init.py'].fit_probe is other
        raise RuntimeError('declared diagnostic failure')
    monkeypatch.setattr(setup.driver,'run',failure)
    args=SimpleNamespace(**{**vars(setup.args),'resident_inputs':True})
    with pytest.raises(RuntimeError,match='declared diagnostic failure'):
        sweep._execute(args,setup.plan,[setup.job],{}, {})
    assert setup.driver.fit_probe is original


def test_main_records_explicit_mode_binding_and_frozen_helper_core(tmp_path,monkeypatch):
    setup=fixture(tmp_path,monkeypatch,'run_probe_calibration.py')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'frozen_sha256':{}}))
    monkeypatch.setattr(sweep,'_jobs',lambda *args:[setup.job])
    calls=[]
    def execute(args,plan,jobs,identity,frozen):
        calls.append((jobs,identity,frozen));return identity
    monkeypatch.setattr(sweep,'_execute',execute)
    args=SimpleNamespace(**{**vars(setup.args),'plan':plan,'resident_inputs':True})
    identity=sweep.main(args)
    binding=quality.resident_execution_binding()
    assert identity['resident_inputs'] and identity['execution_adapter']=={'binding':binding}
    assert calls[0][0][0]['execution_adapter']=={'binding':binding}
    assert 'execution_adapter' not in setup.job  # No formal-plan or method identity mutation.
    for field in ('helper_source','reference_source'):
        assert identity['frozen_sha256'][binding[field]['path']]==binding[field]['sha256']
    with pytest.raises(ValueError,match='execution identity changed'):
        sweep.main(SimpleNamespace(**{**vars(args),'resident_inputs':False}))


def test_quality_reads_actual_adapter_without_plan_expectation_and_refuses_wrong_sources(tmp_path):
    path,probe,metadata,job,binding=audit_fixture(tmp_path)
    assert 'execution_adapter' not in quality.audit_weight(path,job,binding,'fg')
    expected=quality.resident_execution_binding()
    metadata['execution_adapter']=dict(binding=expected,metrics=dict(diagnostic_cpu_fixture=True))
    save_variant(path,probe,metadata)
    audited=quality.audit_weight(path,job,binding,'fg')
    assert audited['execution_adapter']==metadata['execution_adapter']
    assert quality.audit_weight(path,job,binding,'fg',expected_execution_adapter=expected)['execution_adapter']==metadata['execution_adapter']
    wrong=copy.deepcopy(expected);wrong['reference_source']['sha256']='0'*64
    with pytest.raises(ValueError,match='expected execution adapter'):
        quality.audit_weight(path,job,binding,'fg',expected_execution_adapter=wrong)
    metadata['execution_adapter']['binding']=wrong;save_variant(path,probe,metadata)
    with pytest.raises(ValueError,match='execution adapter binding differs'):
        quality.audit_weight(path,job,binding,'fg')
