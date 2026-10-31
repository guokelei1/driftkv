#!/usr/bin/env python3
"""Exploratory1/3/7/14-day prefixes of a complete causal E5 ladder.

Preserve the fixed full UID population and native trajectories. This computes
quality only: whole-E14 FLOPs are never relabeled as prefix-window costs.
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
    ShardLoader, _context, _same_events, _interval, _metric,
)
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256


def analyze(paths,config_path):
    cfg=json.loads(config_path.read_text())
    if cfg['days_after_release']!=[1,3,7,14] or cfg['methods']!=['f','fg','fgh']:
        raise ValueError('use all four predeclared windows and the complete ladder')
    loader=ShardLoader();points={};identities={}
    if len(paths)!=3:raise ValueError('require three complete stage summaries')
    for path in paths:
        point=loader.load(path);method=point['summary']['method_label']
        if method not in cfg['methods'] or method in points:raise ValueError('missing/duplicate method')
        points[method]=point
        payload=torch.load(point['summary']['identity']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        identities[method]=payload['metadata']['identity']
    context=_context(points['f']['summary']['identity']);anchors=points['f']['anchors'];base=anchors['reuse']
    cutoff=identities['f']['source_cutoff']
    recipe=identities['f']
    for method,point in points.items():
        if (_context(point['summary']['identity'])!=context or identities[method]!=recipe or
                point['anchors']!=anchors or not _same_events(point['rows'],base)):
            raise ValueError('prefix methods need the same recipe/UIDs/events/anchors/model/input/backend')
    if any(not cutoff<=row['query_timestamp']<cutoff+14*86400 for row in base):
        raise ValueError('source scores outside the stated causalE14 handover window')
    population=np.asarray(sorted(context['users']),dtype=np.int64)
    seed=cfg['confidence']['bootstrap_seed'];replicates=cfg['confidence']['replicates']
    rng=np.random.default_rng(seed)
    weights=np.asarray([np.bincount(rng.integers(len(population),size=len(population)),minlength=len(population)) for _ in range(replicates)],dtype=np.uint16)
    weights=np.r_[np.ones((1,len(population)),dtype=np.uint16),weights]
    reports=[];rows=[]
    def number(value):return float(value) if np.isfinite(value) else None
    for days in cfg['days_after_release']:
        mask=np.asarray([row['query_timestamp']<cutoff+days*86400 for row in base],dtype=bool)
        selected=[row for row,keep in zip(base,mask) if keep]
        labels=np.asarray([row['label'] for row in selected],dtype=np.float64)
        indices=np.searchsorted(population,np.asarray([row['uid'] for row in selected],dtype=np.int64))
        def auc(source):
            scores=np.asarray([row['score'] for row,keep in zip(source,mask) if keep],dtype=np.float64)
            if len(set(labels.tolist()))<2:return np.full(1+replicates,np.nan)
            return weighted_auc(scores,labels,indices,weights)
        values={method:auc(point['rows']) for method,point in points.items()}
        native={method:auc(values) for method,values in anchors.items()}
        gap=native['rebuild']-native['reuse']
        report=dict(days=days,quality_users=len(population),users_with_feedback=len({row['uid'] for row in selected}),
            feedback_events=len(selected),start_timestamp=cutoff,end_timestamp=cutoff+days*86400,
            status='defined_auc' if np.isfinite(gap[0]) else 'undefined_auc_fewer_than_two_labels',
            anchors={method:dict(auc=number(value[0]),ci95=_interval(value[1:])) for method,value in native.items()},
            gap=number(gap[0]),gap_ci95=_interval(gap[1:]),methods={},increments={})
        for method,value in values.items():
            report['methods'][method]=dict(auc=number(value[0]),auc_ci95=_interval(value[1:]))
            ci=report['methods'][method]['auc_ci95'] or [None,None]
            rows.append(dict(days=days,method=method,quality_users=len(population),users_with_feedback=report['users_with_feedback'],
                feedback_events=len(selected),auc=number(value[0]),auc_ci95_low=ci[0],auc_ci95_high=ci[1],
                reuse_auc=number(native['reuse'][0]),rebuild_auc=number(native['rebuild'][0]),reference_gap=number(gap[0])))
        if report['status']=='defined_auc':
            for term,after,before in (('g','fg','f'),('h','fgh','fg')):
                delta=values[after]-values[before]
                report['increments'][term]=_metric(delta[0],delta[1:],gap[0],gap[1:])
        if days==14:
            for method,value in values.items():
                if not np.isfinite(value[0]) or abs(value[0]-points[method]['summary']['auc'])>1e-12:
                    raise ValueError('full14 prefix AUC differs from original committed summary')
        reports.append(report)
    return dict(status='complete_exploratory_handover_windows',edge=context['edge'],quality_users=len(population),
        source_evaluation_window=points['f']['summary']['evaluation_window'],windows=reports,rows=rows,
        confidence=cfg['confidence'],config=str(config_path.resolve()),config_sha256=sha256(config_path),
        source_summaries={method:point['path'] for method,point in points.items()},
        verification=dict(artifact_sha256=loader.hashes,full14_auc_matches_original=True,
            sources={str(p):sha256(p) for p in (Path(__file__),ROOT/'experiments/ch3_motivation/code/analyze_probe_quality.py',ROOT/'src/hstu_kvcache/evaluation/statistics.py')}),
        cost_scope='quality-only diagnostic; no prefix FLOPs claimed, fullE14 cost must not be reused',
        interpretation=cfg['scope'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,nargs=3,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=analyze(args.summary,args.config);args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'points.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['rows'][0]));writer.writeheader();writer.writerows(result['rows'])
    increments=[]
    for report in result['windows']:
        for term,value in report['increments'].items():
            ci=value['auc_difference_ci95'] or [None,None]
            increments.append(dict(days=report['days'],term=term,auc_difference=value['auc_difference'],
                auc_ci95_low=ci[0],auc_ci95_high=ci[1],feedback_events=report['feedback_events'],
                users_with_feedback=report['users_with_feedback'],quality_users=report['quality_users']))
    if increments:
        with (args.output/'increments.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(increments[0]));writer.writeheader();writer.writerows(increments)
    write_json(args.output/'summary.json',result)
    print(json.dumps(dict(status=result['status'],edge=result['edge'],output=str(args.output))))
