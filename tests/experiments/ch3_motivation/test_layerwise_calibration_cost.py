"""Compare ledger deduction against normalization calls in a real tiny fit."""
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import torch
import pytest
ROOT=Path(__file__).resolve().parents[3]
from tests.experiments.ch3_motivation.test_probes import fixture, cpu_backend
from experiments.ch3_motivation.code import (
    probes, layerwise_query_centering as centered, layerwise_calibration_cost as cost,
)
from experiments.ch3_motivation.code.probe_variants import LayerwiseKVProbeLadder
from hstu_kvcache.common.artifacts import sha256

@pytest.mark.parametrize('factorized',[False,True])
def test_executed_normalization_count_and_affine_delta_preservation(factorized,monkeypatch):
    model,sample=fixture()
    train=probes.prepare_training_samples(model,[sample]);val=copy.deepcopy(train)
    probe=LayerwiseKVProbeLadder.from_model(model,stage='fgh',hidden_width=7,depth=1)
    stats=centered.fit_statistics(probe,model,train,'center_std')
    actual={'training':0,'validation':0}
    normalize=probe._normalize_rows
    def counted(layer,rows):
        actual['training' if torch.is_grad_enabled() else 'validation']+=2*rows.numel()
        return normalize(layer,rows)
    monkeypatch.setattr(probe,'_normalize_rows',counted)
    report=centered.fit_query_conditioned(model,probe,train,val,mode='center_std',statistics=stats,
        factorized_h=factorized,steps=2,patience=0,validation_every=1,learning_rate=1e-3,seed=17)
    report['method_label']='fgh';report['cumulative_calibration_flops_estimate']=report['total_calibration_flops_estimate']+1234
    report['identity']=dict(edge=1,seed=17,numerical_sources={name:sha256(ROOT/name) for name in (cost.original.CORE_SOURCE,cost.original.VARIANT_SOURCE)})
    payload=dict(format='ch3_probe_variant_v1',variant='layerwise_kv',stage='fgh',config=asdict(probe.cfg),metadata=report)
    before=copy.deepcopy(payload)
    correction=cost.correction_record(payload)
    assert correction['deduction_components_flops']==dict(training_forward=actual['training'],validation_forward=actual['validation'],training_backward_estimate=2*actual['training'])
    assert correction['corrected_stage_calibration_flops_estimate']==sum(report['arithmetic'].values())-sum(correction['deduction_components_flops'].values())
    assert correction['corrected_cumulative_calibration_flops_estimate']-correction['corrected_stage_calibration_flops_estimate']==1234
    assert payload==before
