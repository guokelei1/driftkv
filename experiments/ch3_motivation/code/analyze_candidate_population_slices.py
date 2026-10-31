#!/usr/bin/env python3
"""One development sensitivity: history-class AUC on original versus added UIDs."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import csv
import json
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.analyze_candidate_robustness import (
    load_scores, load_attributes, auc_summary, class_differences,
)
from experiments.ch3_motivation.code.analyze_real_quality import uid_weights, sha
import numpy as np


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def direction(value):return None if value is None else ('positive' if value>0 else 'negative' if value<0 else 'zero')


def analyze(root,edge):
    population_path=root/'population.json';folder=root/f'quality/edge_{edge}'
    slices_path=root/'population_slices.json';coverage_path=root/'population_slice_coverage.json'
    slices=json.loads(slices_path.read_text());coverage=json.loads(coverage_path.read_text())
    if slices['population_sha256']!=sha(population_path) or coverage['slice_sha256']!=sha(slices_path):
        raise ValueError('frozen source slices changed')
    config,population,groups=load_scores(folder,population_path)
    _,_,flags=load_attributes(root/f'edge_{edge}',population_path,edge,groups['reuse'])
    uids=np.asarray([r['uid'] for r in groups['reuse']],dtype=np.uint64)
    results={};csv_rows=[];comparison_rows=[]
    for name,slice_entry in slices['slices'].items():
        fixed=slice_entry['users'];members=set(fixed)
        common=sorted({int(u) for u,keep in zip(uids,flags['history_seen']) if keep and int(u) in members}
            &{int(u) for u,keep in zip(uids,flags['history_unseen']) if keep and int(u) in members})
        prescribed=coverage['edges'][str(edge)]['slices'][name]['attributes']['history_familiarity']['common_uids']
        if common!=prescribed:raise ValueError('sameUID class support differs from frozen coverage')
        if len(common)<512:raise ValueError('history paired source slice below512; only coverage is permitted')
        mask=np.isin(uids,common);selected={m:[r for r,keep in zip(rows,mask,strict=True) if keep] for m,rows in groups.items()}
        index,weights=uid_weights(common,uids[mask]);points={};draws={}
        for category in ['history_seen','history_unseen']:
            point,samples=auc_summary(selected,flags[category][mask],index,weights)
            for method,metrics in point['methods'].items():
                metrics.pop('recovery_percent',None)
                row=dict(slice=name,common_uid_count=len(common),category=category,method=method,
                    feedback_users=point['feedback_users'],events=point['events'],mechanism_use_permitted=point['mechanism_use_permitted'])
                for key,value in dict(metrics,teacher_minus_reuse_AUC=point['teacher_minus_reuse_AUC']).items():
                    row[key]=value['value'];row[key+'_ci_low'],row[key+'_ci_high']=value['ci95'] or [None,None]
                csv_rows.append(row)
            points[category],draws[category]=point,samples
        differences=class_differences(points['history_seen'],points['history_unseen'],draws['history_seen'],draws['history_unseen'])
        for method,metrics in differences['methods'].items():
            row=dict(slice=name,common_uid_count=len(common),method=method)
            for key,value in dict(metrics,native_gap_difference=differences['native_gap_difference']).items():
                row[key]=value['value'];row[key+'_ci_low'],row[key+'_ci_high']=value['ci95'] or [None,None]
                row[key+'_direction']=direction(value['value'])
            comparison_rows.append(row)
        results[name]=dict(residents=len(fixed),common_feedback_users=len(common),common_uids=common,
            groups=points,class_differences=differences)
        print(json.dumps(dict(edge=edge,slice=name,common_feedback_users=len(common),status='analyzed')),flush=True)
    result=dict(format='candidate_development_population_sensitivity_v1',edge=edge,slices=results,
        scope='History-class AUC on sameUID intersections within disjoint original4000 and added3000 development slices; added3000 is not pristine confirmation.',
        source_quality_configuration_sha256=sha(folder/'configuration.json'),source_quality_summary_sha256=sha(folder/'summary.json'),
        population_sha256=sha(population_path),population_slices_sha256=sha(slices_path),coverage_sha256=sha(coverage_path),analysis_sha256=sha(__file__),
        metrics_sources={str(Path(__file__).with_name(name).resolve()):sha(Path(__file__).with_name(name))
            for name in ['analyze_candidate_robustness.py','analyze_real_quality.py','analyze_candidate_class_auc.py']},
        direction_convention='First class history_seen minus second history_unseen; report positive and negative point directions without selecting maps or users.',
        limitations=['Both source slices are development data and share the same frozen models/calibration; neither is pristine confirmation.',
            'Each slice conditions on actual feedback in both candidate classes; comparisons cannot be generalized to absent-feedback users.',
            '500 shared UID-bootstrap draws pair methods and candidate classes within each slice; no new cross-slice significance test.',
            'All configured static maps are retained, irrespective of the direction; no new GPU, thresholds, figures or metrics.'])
    output=root/f'population_sensitivity/edge_{edge}';output.mkdir(parents=True,exist_ok=True)
    write(output/'summary.json',result)
    for filename,rows in [('metrics.csv',csv_rows),('history_class_differences.csv',comparison_rows)]:
        with (output/filename).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[f'# Disjoint development population sensitivity: edge{edge}','',result['scope'],'',
        '| Slice | Common UID | Static map | Remaining-gap seen−unseen | Map-gain seen−unseen | Native-gap seen−unseen |',
        '|---|---:|---|---:|---:|---:|']
    for row in comparison_rows:
        values=[row[k] for k in ['remaining_gap_difference','map_gain_difference','native_gap_difference']]
        lines.append(f"| {row['slice']} | {row['common_uid_count']} | {row['method']} | "+' | '.join('NA' if x is None else f'{x:.6f}' for x in values)+' |')
    lines+=['','Each class AUC, native gap, map gain and remaining gap with95% UID intervals appears in metrics.csv. Both positive and negative directions are retained.','',*['- '+v for v in result['limitations']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--wait',action='store_true');args=p.parse_args()
    for edge in [2,3]:
        summary=args.root/f'quality/edge_{edge}/summary.json'
        while args.wait and not summary.exists():
            failure=args.root/'batch_exit.json'
            if failure.exists() and json.loads(failure.read_text())['exit_code']!=0:raise ValueError('source quality batch failed; no partial-score analysis')
            time.sleep(5)
        analyze(args.root,edge)
    print(json.dumps(dict(status='complete',edges=[2,3],question='original versus added development population history-class direction')),flush=True)
