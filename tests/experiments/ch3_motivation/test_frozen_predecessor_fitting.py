"""Actual optimizer checks for the separate new-term-only E5 protocol."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import probes
from experiments.ch3_motivation.code.frozen_predecessor_fitting import fit_probe_new_term
from experiments.ch3_motivation.code.probe_variants import build_variant
from tests.experiments.ch3_motivation.test_resident_probe_fitting import world


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch')
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('stage',['f','fg','fgh'])
@pytest.mark.parametrize('resident',[False,True])
def test_real_optimizer_preserves_predecessors_and_matches_manual_gradient_policy(stage,resident):
    model,train,validation=world()
    probe=build_variant('full_item',model,stage=stage,hidden_width=8,depth=1,query_chunk=2,row_chunk=3)
    # Nonzero predecessors expose unintended AdamW decay as well as gradients.
    term={'f':'f','fg':'g','fgh':'h'}[stage]
    with torch.no_grad():
        for name in ('f','g'):
            module=getattr(probe,name)
            if module is not None and name!=term:
                for network in module:network.weights[-1].normal_(std=.01)
    oracle=copy.deepcopy(probe)
    original={name:value.clone() for name,value in probe.named_parameters()}
    flags={name:value.requires_grad for name,value in probe.named_parameters()}
    term={'f':'f','fg':'g','fgh':'h'}[stage]
    gradients=[]
    hooks=[value.register_hook(lambda g,name=name:gradients.append((name,float(g.abs().max())))) for name,value in probe.named_parameters()]
    for name,parameter in oracle.named_parameters():parameter.requires_grad_(name.startswith(term+'.'))
    kwargs=dict(steps=12,validation_every=1,patience=0,learning_rate=.001,weight_decay=.1,seed=17)
    # Shared data here is solely a small optimizer/numerical oracle, not held-out quality evidence.
    expected=probes.fit_probe(model,oracle,train,train,**kwargs)
    actual=fit_probe_new_term(model,probe,train,train,resident_inputs=resident,**kwargs)
    policy=actual.pop('training_protocol')
    actual.pop('execution_adapter',None)
    assert actual==expected
    assert actual['best_step']>0
    assert gradients and all(name.startswith(term+'.') for name,_ in gradients)
    assert any(value>0 for _,value in gradients)
    for name,value in probe.named_parameters():
        assert value.requires_grad==flags[name]
        assert torch.equal(value,dict(oracle.named_parameters())[name])
        if not name.startswith(term+'.'):
            assert torch.equal(value,original[name]) and value.grad is None
    assert any(not torch.equal(value,original[name]) for name,value in probe.named_parameters() if name.startswith(term+'.'))
    assert policy['trainable_term']==term
    assert policy['binding']['format']=='e5_new_term_only_v1'
    for hook in hooks:hook.remove()


def test_flags_restored_when_real_fit_rejects_nonfinite_target():
    model,train,validation=world()
    probe=build_variant('full_item',model,stage='fgh',hidden_width=8)
    flags=[p.requires_grad for p in probe.parameters()]
    bad=[dict(s,teacher_scores=torch.full_like(s['teacher_scores'],float('nan'))) for s in train]
    with pytest.raises((ValueError,FloatingPointError)):
        fit_probe_new_term(model,probe,bad,validation,steps=1,validation_every=1)
    assert [p.requires_grad for p in probe.parameters()]==flags
