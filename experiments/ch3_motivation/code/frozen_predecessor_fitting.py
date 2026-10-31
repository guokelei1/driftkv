"""Supplementary E5 training protocol: fit only the newly added term.

The existing joint-training protocol remains the reference. This diagnostic
keeps architecture, causal samples and read+score loss unchanged, but freezes
f during FG and f/g during FGH to isolate drift in trained predecessors.
It is a scientific protocol variant, not a numerically equivalent execution
optimization, and must retain its own identity if used in quality experiments.
"""
from __future__ import annotations

from pathlib import Path

from experiments.ch3_motivation.code import probes
from experiments.ch3_motivation.code.resident_probe_fitting import fit_probe_resident
from hstu_kvcache.common.artifacts import sha256


def protocol_binding():
    return dict(format='e5_new_term_only_v1',helper_sha256=sha256(Path(__file__)),
        core_sha256=sha256(Path(probes.__file__)),
        trainable_terms_by_stage={'f':['f'],'fg':['g'],'fgh':['h']},
        loss='unchanged fixed-scale same-query read alignment plus final-score MSE',
        scope='supplementary protocol; does not replace or inherit a joint-training result identity',
        arithmetic='original full-forward ledger retained; 2x full-forward backward estimate is conservative for frozen predecessors, not measured actual backward work')


def fit_probe_new_term(model,probe,train_samples,validation_samples,*,resident_inputs=False,**kwargs):
    """Temporarily disable predecessor parameter gradients around the real fitter.

    Do not detach model queries or the new-term input: gradients through the
    frozen backbone to the active term still follow the original objective.
    Existing forward trajectories and train/validation data are unchanged.
    """
    selected={'f':'f','fg':'g','fgh':'h'}[probe.stage]
    parameters=list(probe.named_parameters())
    flags={name:parameter.requires_grad for name,parameter in parameters}
    active=[name for name,parameter in parameters if name.startswith(selected+'.') and parameter.requires_grad]
    if not active:
        raise ValueError('new-term training requires trainable parameters in the added module')
    active=set(active)
    try:
        for name,parameter in parameters:
            parameter.requires_grad_(name in active)
            # Old optimizer gradients are not a valid update in this protocol.
            if name not in active:
                parameter.grad=None
        fit=fit_probe_resident if resident_inputs else probes.fit_probe
        result=fit(model,probe,train_samples,validation_samples,**kwargs)
    finally:
        for name,parameter in parameters:
            parameter.requires_grad_(flags[name])
    result['training_protocol']=dict(binding=protocol_binding(),
        trainable_term=selected,trainable_parameter_count=sum(parameter.numel() for name,parameter in parameters if name in active),
        frozen_parameter_count=sum(parameter.numel() for name,parameter in parameters if name not in active),
        resident_inputs=bool(resident_inputs))
    return result
