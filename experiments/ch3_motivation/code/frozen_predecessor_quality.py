"""Actual weight/input audits for the supplementary fixed-predecessor protocol.

The original sampling auditor remains unchanged. Its source-f audit establishes
the original causal recipe; this module verifies the new protocol explicitly,
rather than presenting new weights as joint-training checkpoints.
"""
from __future__ import annotations
from pathlib import Path
import json
import torch
from experiments.ch3_motivation.code import plan_probe_quality as quality
from experiments.ch3_motivation.code.common import ROOT
from experiments.ch3_motivation.code.frozen_predecessor_fitting import protocol_binding
from hstu_kvcache.common.artifacts import sha256


def audit_ladder(job,binding):
    source_job=job['source_job']
    source_path=Path(source_job['output'])/'f.pt'
    source_summary=json.loads(source_path.with_suffix('.json').read_text())
    if source_summary['identity']!=source_job['expected_identity']:
        raise ValueError('source f differs from its original planned identity')
    # Execution jobs omit fixed identity fields such as read_weight. The
    # quality audit needs the full planned recipe, not only runtime options.
    audit_job=dict(source_job['expected_identity'])
    audit_job.update(source_job)
    source_audit=quality.audit_weight(source_path,audit_job,binding,'f')
    if source_summary['checkpoint_sha256']!=source_audit['checkpoint_sha256']:
        raise ValueError('source f committed hash differs')
    original=torch.load(source_path,map_location='cpu',weights_only=False,mmap=True)
    source_identity=original['metadata']['identity']
    driver='run_probe_frozen_predecessors.py'
    code=dict(source_identity['numerical_sources'])
    del code['experiments/ch3_motivation/code/run_probe_calibration_init.py']
    for name in (driver,'frozen_predecessor_fitting.py','resident_probe_fitting.py'):
        code['experiments/ch3_motivation/code/'+name]=sha256(ROOT/'experiments/ch3_motivation/code'/name)
    expected=dict(source_identity,training_protocol=protocol_binding(),resident_inputs=True,
        source_f_checkpoint=str(source_path.resolve()),source_f_sha256=source_audit['checkpoint_sha256'],
        fitting_driver=driver,fitting_driver_sha256=code['experiments/ch3_motivation/code/'+driver],numerical_sources=code)
    for name,digest in code.items():
        if sha256(ROOT/name)!=digest:raise ValueError('supplementary numerical source changed')
    previous=original['state_dict'];cumulative=original['metadata']['cumulative_calibration_flops_estimate']
    audited=[]
    for stage in ('f','fg','fgh'):
        path=Path(job['output'])/f'{stage}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        metadata=payload['metadata'];state=payload['state_dict'];digest=sha256(path)
        summary=json.loads(path.with_suffix('.json').read_text())
        if summary!={**metadata,'checkpoint':str(path),'checkpoint_sha256':digest}:
            raise ValueError('supplementary stage summary/hash differs from checkpoint')
        if (payload.get('format')!='ch3_probe_variant_v1' or payload.get('variant')!='full_item' or
                payload['stage']!=stage or metadata['method_label']!=stage or
                payload['config']!=original['config'] or metadata['identity']!=expected or
                metadata['seed']!=source_job['seed'] or metadata['training_users']!=source_job['fit_users'] or
                metadata['validation_users']!=16 or metadata['quality_auc_computed'] or
                metadata.get('diagnostic_only') or metadata.get('quality_use_permitted') is False or
                metadata.get('fitting_uses_future_feedback')):
            raise ValueError('supplementary architecture/recipe/role/protocol identity differs')
        if any(key not in state or not torch.equal(value,state[key]) for key,value in previous.items()):
            raise ValueError('supplementary frozen predecessor or statistics changed')
        history=metadata['history'];best=min(history,key=lambda row:row['validation_loss'])
        if (best['step']!=metadata['best_step'] or best['validation_loss']!=metadata['best_validation_loss'] or
                history[-1]['step']!=metadata['completed_steps'] or not 0<metadata['completed_steps']<=source_job['steps']):
            raise ValueError('supplementary validation selection/budget differs')
        scales=metadata['fit_pool_loss_scales']
        if (scales.get('users')!=source_job['fit_users'] or scales.get('source')!='fitting_users_only' or
                scales.get('validation_uses_fitting_scales') is not True or
                metadata['calibration_preparation']!=original['metadata']['calibration_preparation']):
            raise ValueError('supplementary own-N scales or preparation differs')
        policy=metadata['training_protocol']
        if policy['binding']!=protocol_binding():raise ValueError('supplementary protocol binding differs')
        if stage=='f':
            if (set(state)!=set(previous) or not policy['inherited_source_only'] or
                    policy['source_checkpoint_sha256']!=source_audit['checkpoint_sha256'] or
                    metadata['cumulative_calibration_flops_estimate']!=cumulative):
                raise ValueError('source f reuse/cost differs')
        else:
            term={'fg':'g','fgh':'h'}[stage]
            active=sum(value.numel() for name,value in state.items() if name.startswith(term+'.'))
            frozen=sum(value.numel() for name,value in state.items() if name.startswith(('f.','g.','h.')))-active
            if (policy['trainable_term']!=term or policy['trainable_parameter_count']!=active or
                    policy['frozen_parameter_count']!=frozen or not policy['resident_inputs'] or
                    metadata.get('execution_adapter',{}).get('binding')!=quality.resident_execution_binding()):
                raise ValueError('supplementary fitted term or execution adapter differs')
            cumulative+=metadata['total_calibration_flops_estimate']
            if metadata['cumulative_calibration_flops_estimate']!=cumulative or metadata['h_initialization_weight_rescale_flops']!=0:
                raise ValueError('supplementary cumulative calibration ledger differs')
        audited.append(dict(method_label=stage,checkpoint=str(path),checkpoint_sha256=digest,
            best_step=metadata['best_step'],completed_steps=metadata['completed_steps'],
            predecessors_exact=True,training_protocol=policy,
            standalone_calibration_flops_estimate=cumulative))
        previous=state
    return dict(edge=source_job['edge'],identity=expected,source_f_audit=source_audit,
        source_f_checkpoint=str(source_path),weights=audited,
        cost_convention='full forward plus conservative 2x full-forward backward estimate; frozen-backward saving not claimed')
