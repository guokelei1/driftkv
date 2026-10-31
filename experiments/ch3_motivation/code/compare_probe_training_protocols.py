#!/usr/bin/env python3
"""Paired same-population comparison of joint and new-term-only E5 ladders.

All six committed points are required. This computes the difference between
h increments directly under shared UID draws, rather than subtracting CIs.
"""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from pathlib import Path
import argparse,csv,json,sys
ROOT=Path(__file__).resolve().parents[3]
import numpy as np
import torch
from experiments.ch3_motivation.code.analyze_probe_quality import (
    ShardLoader, _context, _same_events, _interval,
)
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.frozen_predecessor_fitting import protocol_binding
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256

METHODS=('f','fg','fgh')
SHARED=('edge','seed','models','variant','width','depth','fit_users','fit_uids','validation_users',
    'queries','learning_rate','steps','patience','validation_every','read_weight','score_weight',
    'candidate_source','recent_fraction','candidate_seed','row_chunk','query_chunk','weight_decay',
    'gradient_clip','stages','controls','input_binding_sha256','population_sha256',
    'calibration_candidate_sha256','query_delta_rule','sampling_manifest_sha256','gap_manifest_sha256')


def contrast_draws(values):
    result={}
    for protocol in ('joint','new_term_only'):
        result[protocol+'_g']=values[protocol,'fg']-values[protocol,'f']
        result[protocol+'_h']=values[protocol,'fgh']-values[protocol,'fg']
    for stage in METHODS:
        result['new_minus_joint_'+stage]=values['new_term_only',stage]-values['joint',stage]
    for term in ('g','h'):
        result['new_minus_joint_'+term+'_increment']=result['new_term_only_'+term]-result['joint_'+term]
    return result


def analyze(joint_paths,new_paths,*,replicates=1000,seed=17051):
    if replicates<1:raise ValueError('positive bootstrap replicates required')
    loader=ShardLoader();points={};identities={};f_states={}
    for protocol,paths in (('joint',joint_paths),('new_term_only',new_paths)):
        if len(paths)!=3:raise ValueError('require exactly three actual stage summaries per protocol')
        for path in paths:
            point=loader.load(path);summary=point['summary'];stage=summary['method_label']
            if stage not in METHODS or (protocol,stage) in points:raise ValueError('duplicate/unknown protocol stage')
            checkpoint=Path(summary['identity']['checkpoint'])
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
            identity=payload['metadata']['identity']
            if protocol=='new_term_only':
                if identity.get('training_protocol')!=protocol_binding():raise ValueError('new protocol identity differs')
            elif 'training_protocol' in identity:raise ValueError('joint reference cannot carry another training protocol')
            points[protocol,stage]=point;identities[protocol,stage]=identity
            if stage=='f':f_states[protocol]=payload['state_dict']
    first=points['joint','f'];context=_context(first['summary']['identity']);base=first['anchors']['reuse']
    reference={key:identities['joint','f'].get(key) for key in SHARED}
    for key,point in points.items():
        if _context(point['summary']['identity'])!=context or not _same_events(point['rows'],base):
            raise ValueError('protocol population/events/models/input/backend differ')
        if {name:identities[key].get(name) for name in SHARED}!=reference:
            raise ValueError('shared fitting recipe differs beyond the training protocol')
        for anchor in ('reuse','rebuild'):
            if point['anchors'][anchor]!=first['anchors'][anchor]:raise ValueError('protocol matched anchors differ')
    left,right=f_states['joint'],f_states['new_term_only']
    if left.keys()!=right.keys() or any(not torch.equal(value,right[key]) for key,value in left.items()):
        raise ValueError('protocols did not start from exactly the same trained f')
    population=np.asarray(sorted(context['users']),dtype=np.int64)
    labels=np.asarray([r['label'] for r in base],dtype=np.float64)
    indices=np.searchsorted(population,np.asarray([r['uid'] for r in base],dtype=np.int64))
    rng=np.random.default_rng(seed)
    weights=np.asarray([np.bincount(rng.integers(len(population),size=len(population)),minlength=len(population)) for _ in range(replicates)],dtype=np.uint16)
    weights=np.r_[np.ones((1,len(population)),dtype=np.uint16),weights]
    values={}
    for key,point in points.items():
        values[key]=weighted_auc(np.asarray([r['score'] for r in point['rows']]),labels,indices,weights)
        if abs(values[key][0]-point['summary']['auc'])>1e-12:raise ValueError('summary AUC differs from verified scores')
    rows=[]
    for name,draws in contrast_draws(values).items():
        ci=_interval(draws[1:])
        rows.append(dict(contrast=name,auc_difference=float(draws[0]),auc_ci95_low=ci[0] if ci else None,
            auc_ci95_high=ci[1] if ci else None,valid_replicates=int(np.isfinite(draws[1:]).sum())))
    return dict(status='complete_paired_training_protocol_comparison',edge=context['edge'],
        quality_users=len(population),feedback_events=len(base),training_seed=reference['seed'],
        contrasts=rows,point_auc={f'{protocol}/{stage}':float(value[0]) for (protocol,stage),value in values.items()},
        reference_gap=first['summary']['rebuild_auc']-first['summary']['reuse_auc'],
        same_trained_f_exact=True,source_summaries={f'{protocol}/{stage}':point['path'] for (protocol,stage),point in points.items()},
        confidence=dict(method='paired UID-cluster percentile bootstrap',replicates=replicates,seed=seed,
            scope='conditional on these six fitted weights, all eligible UIDs including no-feedback users; direct contrasts share identical UID multiplicities'),
        interpretation='development comparison only; no independent confirmation, multi-seed or five-edge conclusion; absolute AUC contrasts avoid unstable recovery denominators',
        verification=dict(artifact_sha256=loader.hashes,shared_fitting_recipe=reference,
            analysis_sources={str(p):sha256(p) for p in (Path(__file__),ROOT/'experiments/ch3_motivation/code/analyze_probe_quality.py',ROOT/'src/hstu_kvcache/evaluation/statistics.py')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--joint',type=Path,nargs=3,required=True)
    p.add_argument('--new-term-only',type=Path,nargs=3,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=1000);p.add_argument('--bootstrap-seed',type=int,default=17051)
    args=p.parse_args();value=analyze(args.joint,args.new_term_only,replicates=args.replicates,seed=args.bootstrap_seed)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'contrasts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(value['contrasts'][0]));writer.writeheader();writer.writerows(value['contrasts'])
    write_json(args.output/'summary.json',value)
    print(json.dumps(dict(status=value['status'],edge=value['edge'],users=value['quality_users'],output=str(args.output))))
