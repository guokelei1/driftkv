"""Extend the retained normalization-ledger correction to actual layerwise KV.

The reference f and h share one normalized row table. Their legacy arithmetic
charges its normalization twice, including for the enlarged all-head row.
This sidecar preserves checkpoints and the already-applied affine deltas.
"""
import hashlib
import json
from pathlib import Path
import torch
from experiments.ch3_motivation.code import calibration_cost_correction as original
from hstu_kvcache.common.artifacts import sha256


def correction_record(payload):
    if payload.get('variant')!='layerwise_kv' or payload['metadata']['method_label']!='fgh':
        return original.correction_record(payload)
    meta,cfg=payload['metadata'],payload['config'];ident=meta['identity']
    if payload.get('format')!='ch3_probe_variant_v1' or payload['stage']!='fgh' or meta['stage']!='fgh':
        raise ValueError('need actual layerwise FGH payload')
    sources=ident.get('numerical_sources',{})
    if not original.audited_sources(sources):
        raise ValueError('layerwise source is not the audited shared-normalization implementation')
    counts=meta['operation_counts'];integer=original._integer
    steps=integer(meta['completed_steps'],'completed_steps')
    if any(integer(counts.get(k),k)!=steps for k in ('training_sample_calls','training_backward_calls')):
        raise ValueError('actual training calls disagree with completed steps')
    for key in ('num_layers','num_heads','head_dim','item_dim'):
        if integer(cfg.get(key),key)==0:raise ValueError('positive actual layerwise dimensions required')
    if type(cfg.get('normalize')) is not bool or type(cfg.get('include_item')) is not bool:raise ValueError('actual normalization flags required')
    row_dim=2*cfg['num_heads']*cfg['head_dim']+(cfg['item_dim'] if cfg['include_item'] else 0)
    unit=2*row_dim*cfg['num_layers']*cfg['num_heads'] if cfg['normalize'] else 0
    tr=integer(counts.get('training_probe_rows'),'training_probe_rows');val=integer(counts.get('validation_probe_rows'),'validation_probe_rows')
    parts=dict(training_forward=unit*tr,validation_forward=unit*val,training_backward_estimate=2*unit*tr)
    deduction=sum(parts.values())
    own=integer(meta.get('total_calibration_flops_estimate'),'stage cost');raw=integer(meta.get('cumulative_calibration_flops_estimate'),'cumulative cost')
    if not deduction<=own<=raw:raise ValueError('invalid normalization deduction or cumulative cost')
    return dict(format='layerwise_calibration_cost_correction_v1',edge=ident['edge'],seed=ident['seed'],method_label='fgh',
        variant='layerwise_kv',config=dict(cfg),row_dim=row_dim,raw_cumulative_calibration_flops_estimate=raw,
        raw_stage_calibration_flops_estimate=own,shared_row_normalization_deduction_flops=deduction,
        deduction_components_flops=parts,duplicated_forward_flops_per_probe_row=unit,
        corrected_cumulative_calibration_flops_estimate=raw-deduction,corrected_stage_calibration_flops_estimate=own-deduction,
        formula='row_dim=2*heads*head_dim+item_dim; 2*row_dim*layers*heads*(3*training_rows+validation_rows)',
        backward_convention='original estimate2x forward; first-affine optimization already applied and retained',
        fitting_identity_sha256=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest(),fitting_identity=ident,
        correction_source_sha256={str(Path(__file__).relative_to(original.ROOT)):sha256(Path(__file__)),
            str(Path(original.__file__).relative_to(original.ROOT)):sha256(Path(original.__file__))},
        original_metadata_and_weights_modified=False)


def checkpoint_correction(path,*,expected_sha256=None):
    path=Path(path).resolve();digest=sha256(path)
    if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('actual checkpoint changed')
    record=correction_record(torch.load(path,map_location='cpu',mmap=True,weights_only=False))
    return dict(record,checkpoint=str(path),checkpoint_sha256=digest)
