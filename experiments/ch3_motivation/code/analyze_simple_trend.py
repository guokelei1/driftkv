#!/usr/bin/env python3
"""Complete five-edge, single-seed simple_trend2000 ladder and full cost export.

This does not bypass the formal analyzer's2000-user/three-seed/controls gates.
It is a separate explicitly exploratory artifact for one complete simple_trend
configuration; all five paired release differences are retained.
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
from experiments.ch3_motivation.code.analyze_probe_quality_multiedge import (
    _training_record, _method_metric, _contrast_metric,
)
from experiments.ch3_motivation.code.layerwise_calibration_cost import checkpoint_correction
from experiments.ch3_motivation.code.calibration_cost_correction import apply_to_costs
from experiments.ch3_motivation.code.probe_cost_export import (
    ACTUAL_COMPONENTS, REFERENCE_COMPONENTS, COMPONENTS, _validate_account,
)
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256

METHODS=('f','fg','fgh')
FIELDS=('variant','width','depth','fit_users','queries','learning_rate','steps','patience','validation_every',
    'read_weight','score_weight','candidate_source','recent_fraction','candidate_seed','row_chunk','query_chunk',
    'weight_decay','gradient_clip','stages','controls','query_delta_rule','fit_uids','validation_users',
    'candidate_catalog_source_users','time_law_source_users','gradient_fitting_users','teacher_fitting_users')


def analyze(paths,label):
    if len(paths)!=15:raise ValueError('require all five simple_trend edges times three actual stages')
    loader=ShardLoader();points={};users=None;recipe=None;h_policy=None;cost_rows=[];cost_details=[]
    for path in paths:
        item=loader.load(path);s=item['summary'];key=s['edge'],s['method_label']
        if key in points or key[0] not in range(1,6) or key[1] not in METHODS:raise ValueError('duplicate/unexpected simple_trend point')
        if s['seed']!=17 or s['quality_users']!=2000 or s['identity']['panel']!='main':raise ValueError('this explicit simple_trend aggregate requires fixed2000/seed17')
        if users is not None and s['identity']['users']!=users:raise ValueError('simple_trend UID population differs across releases/stages')
        users=s['identity']['users']
        payload=torch.load(s['identity']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        training=_training_record(s,payload);identity=training['identity']
        current={name:identity.get(name) for name in FIELDS}
        current.update(config=training['config'],objective=training['objective'])
        if recipe is not None and current!=recipe:raise ValueError('simple_trend recipes differ beyond release-specific inputs/model bindings')
        recipe=current
        if key[1]=='fgh':
            policy=dict(h_initialization=identity.get('h_initialization','standard'),training_protocol=identity.get('training_protocol'))
            if h_policy is not None and policy!=h_policy:raise ValueError('H initialization/training policy differs across releases')
            h_policy=policy
        if points:
            previous=next(iter(points.values()))['summary']['identity']
            if any(s['identity'][name]!=previous[name] for name in ('sources','backend')):raise ValueError('simple_trend runtime sources/backend differ')
        correction=checkpoint_correction(s['identity']['checkpoint'],expected_sha256=s['identity']['checkpoint_sha256'])
        accounts=apply_to_costs(s['costs'],correction)
        cost_details.append(dict(edge=key[0],method=key[1],source_summary=item['path'],**accounts))
        for ledger,field,components in (('actual_event',None,ACTUAL_COMPONENTS),('analytic230','candidate_reads_230',REFERENCE_COMPONENTS)):
            raw=accounts['raw'] if field is None else accounts['raw'][field]
            fixed=accounts['corrected'] if field is None else accounts['corrected'][field]
            _validate_account(raw,components);_validate_account(fixed,components)
            cost_rows.append(dict(edge=key[0],method=key[1],ledger=ledger,quality_users=2000,
                raw_calibration_flops=raw['calibration_flops'],calibration_flops=fixed['calibration_flops'],
                calibration_normalization_deduction_flops=correction['shared_row_normalization_deduction_flops'],
                raw_complete_added_flops=raw['complete_added_flops'],complete_added_flops=fixed['complete_added_flops'],
                once_rebuild_flops=fixed['once_rebuild_flops'],raw_percent_of_once_rebuild=raw['percent_of_once_rebuild'],
                percent_of_once_rebuild=fixed['percent_of_once_rebuild'],source_summary=item['path'],
                checkpoint_sha256=s['identity']['checkpoint_sha256'],**{name:fixed.get(name) for name in COMPONENTS if name!='calibration_flops'}))
        points[key]=item
    if set(points)!={(e,m) for e in range(1,6) for m in METHODS}:raise ValueError('incomplete simple_trend matrix')
    population=np.asarray(sorted(users),dtype=np.int64);rng=np.random.default_rng(17051)
    weights=np.r_[np.ones((1,2000),dtype=np.uint16),np.asarray([np.bincount(rng.integers(2000,size=2000),minlength=2000) for _ in range(1000)],dtype=np.uint16)]
    values={name:[] for name in ('reuse','rebuild',*METHODS)};edges=[];point_rows=[]
    for edge in range(1,6):
        first=points[edge,'f'];anchors=first['anchors'];context=_context(first['summary']['identity']);base=anchors['reuse']
        labels=np.asarray([row['label'] for row in base],dtype=np.float64)
        indices=np.searchsorted(population,np.asarray([row['uid'] for row in base],dtype=np.int64))
        def auc(rows):return weighted_auc(np.asarray([row['score'] for row in rows]),labels,indices,weights)
        measured={method:auc(rows) for method,rows in anchors.items()}
        for method in METHODS:
            item=points[edge,method]
            if _context(item['summary']['identity'])!=context or item['anchors']!=anchors or not _same_events(item['rows'],base):raise ValueError('same-edge events/anchors/context differ')
            measured[method]=auc(item['rows'])
            if abs(measured[method][0]-item['summary']['auc'])>1e-12:raise ValueError('verified score AUC differs from committed summary')
        gap=measured['rebuild']-measured['reuse']
        report=dict(edge=edge,evaluation_window=first['summary']['evaluation_window'],quality_users=2000,feedback_events=len(base),
            anchors=dict(reuse_auc=float(measured['reuse'][0]),rebuild_auc=float(measured['rebuild'][0]),gap=float(gap[0]),gap_ci95=_interval(gap[1:])),
            methods={},increments={})
        for method in METHODS:
            report['methods'][method]=dict(source_summary=points[edge,method]['path'],**_method_metric(measured[method],measured['reuse'],measured['rebuild']))
            point_rows.append(dict(edge=edge,method=method,quality_users=2000,feedback_events=len(base),auc=float(measured[method][0]),reuse_auc=float(measured['reuse'][0]),rebuild_auc=float(measured['rebuild'][0]),recovery_percent=report['methods'][method]['recovery_percent'],source_summary=points[edge,method]['path']))
        for term,after,before in (('g','fg','f'),('h','fgh','fg')):report['increments'][term]=_contrast_metric(measured[after]-measured[before],gap)
        for method,array in measured.items():values[method].append(array)
        edges.append(report)
    means={name:np.stack(items).mean(0) for name,items in values.items()};gap=means['rebuild']-means['reuse']
    aggregate=dict(methods={method:_method_metric(means[method],means['reuse'],means['rebuild']) for method in METHODS},
        increments={term:_contrast_metric(means[after]-means[before],gap) for term,after,before in (('g','fg','f'),('h','fgh','fg'))},
        mean_native_gap=float(gap[0]),mean_native_gap_ci95=_interval(gap[1:]),nonpositive_gap_replicates=int((~(np.isfinite(gap[1:])&(gap[1:]>0))).sum()),
        positive_h_point_edges=[e['edge'] for e in edges if e['increments']['h']['mean_auc_difference']>0])
    totals=[]
    for method in METHODS:
        for ledger in ('actual_event','analytic230'):
            rows=[r for r in cost_rows if r['method']==method and r['ledger']==ledger]
            numerator=sum(r['complete_added_flops'] for r in rows);denominator=sum(r['once_rebuild_flops'] for r in rows)
            totals.append(dict(method=method,ledger=ledger,complete_added_flops=numerator,once_rebuild_flops=denominator,
                percent_of_once_rebuild=100*numerator/denominator,calibration_flops=sum(r['calibration_flops'] for r in rows),
                raw_complete_added_flops=sum(r['raw_complete_added_flops'] for r in rows),calibration_normalization_deduction_flops=sum(r['calibration_normalization_deduction_flops'] for r in rows)))
    return dict(status='complete_five_edge_simple_trend_single_seed',label=label,quality_users=2000,seed=17,points=15,
        recipe=recipe,h_policy=h_policy,edges=edges,aggregate=aggregate,point_rows=point_rows,cost_rows=cost_rows,cost_aggregates=totals,cost_details=cost_details,
        confidence=dict(replicates=1000,seed=17051,method='same paired UID multiplicities across all methods and releases; equal-release AUC differences before recovery ratio',scope='conditional on one fixed fitting seed, simple_trend only, not independent confirmation or training-seed robustness'),
        formal_e5_complete=False,independent_confirmation=False,fit_test_ratio=.05,validation_users_extra=16,trend_round_only=True,verification=dict(artifacts=loader.hashes,sources={str(p):sha256(p) for p in (Path(__file__),ROOT/'experiments/ch3_motivation/code/analyze_probe_quality.py',ROOT/'experiments/ch3_motivation/code/analyze_probe_quality_multiedge.py',ROOT/'experiments/ch3_motivation/code/calibration_cost_correction.py',ROOT/'experiments/ch3_motivation/code/layerwise_calibration_cost.py',ROOT/'experiments/ch3_motivation/code/probe_cost_export.py',ROOT/'src/hstu_kvcache/evaluation/statistics.py')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,nargs=15,required=True);p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=analyze(args.summary,args.label);args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in (('points',result['point_rows']),('cost_points',result['cost_rows']),('cost_aggregates',result['cost_aggregates'])):
        with (args.output/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(args.output/'summary.json',result)
    print(json.dumps(dict(status=result['status'],output=str(args.output),aggregate=result['aggregate'])))
