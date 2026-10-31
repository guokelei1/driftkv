#!/usr/bin/env python3
"""Paired AUC changes on the complete predeclared static-strength grid.

Every strength materializes once at cutover, then has its own causal native
writes. These are global migration policies, not isolated read-time effects.
"""
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

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.analyze_candidate_robustness import (
    ATTRIBUTES, load_scores, load_attributes, auc_summary, class_differences, write_outputs,
)
from experiments.ch3_motivation.code.analyze_real_quality import uid_weights, interval, sha
import numpy as np

ALPHAS=(.25,.5,1.,2.,4.,8.)


def strength_grid(config):
    mapping=config['method_strengths']
    if config['base_map_name']!='linear256_rows':raise ValueError('the final frozen base must be linear256_rows')
    if len(mapping)!=6 or set(map(float,mapping.values()))!=set(ALPHAS):
        raise ValueError('exact predeclared six strengths required')
    if set(config['method_names'])!=set(mapping)|{'reuse','rebuild'}:
        raise ValueError('method names differ from predeclared strengths and anchors')
    return [(alpha,next(name for name,value in mapping.items() if float(value)==alpha)) for alpha in ALPHAS]


def adjacent_results(points,draws,grid,pairs):
    """Use the exact same UID draw columns across methods and both classes."""
    output=[]
    for (before,first),(after,second) in zip(grid[:-1],grid[1:]):
        changes={}
        for category,point in points.items():
            a=point['methods'][first]['AUC']['value'];b=point['methods'][second]['AUC']['value']
            value=b-a if a is not None and b is not None else np.nan
            changes[category]=dict(change=interval(value,draws[category][second]-draws[category][first]),
                feedback_users=point['feedback_users'],mechanism_use_permitted=point['mechanism_use_permitted'])
        contrasts={}
        for attribute,(a,b) in pairs.items():
            left=changes[a]['change']['value'];right=changes[b]['change']['value']
            value=left-right if left is not None and right is not None else np.nan
            samples=(draws[a][second]-draws[a][first])-(draws[b][second]-draws[b][first])
            contrasts[attribute]=dict(first_group=a,second_group=b,change_difference=interval(value,samples),
                opposite_point_slopes=bool(left*right<0) if left is not None and right is not None else None,
                mechanism_use_permitted=changes[a]['mechanism_use_permitted'] and changes[b]['mechanism_use_permitted'],
                secondary=attribute=='seen_repetition')
        output.append(dict(alpha_before=before,alpha_after=after,method_before=first,method_after=second,
            class_changes=changes,class_change_differences=contrasts))
    return output


def analyze_scopes(groups,population,flags,grid,slices=None):
    uids=np.asarray([r['uid'] for r in groups['reuse']],dtype=np.uint64)
    points={};contrasts={};scopes={};adjacent={}
    def scope(name,residents,mask,pairs,include_all=False):
        selected={m:[r for r,keep in zip(rows,mask,strict=True) if keep] for m,rows in groups.items()}
        index,weights=uid_weights(residents,uids[mask]);local={};draws={}
        scopes[name]=dict(resident_users=len(residents),uids=list(map(int,residents)),bootstrap_replicates=500,
            scope='Same UID draw multiplicities across every strength and candidate class; conditional on this fixed resident scope.')
        categories=[category for pair in pairs.values() for category in pair]
        if include_all:categories=['all']+categories
        for category in categories:
            category_mask=np.ones(int(mask.sum()),dtype=bool) if category=='all' else flags[category][mask]
            point,samples=auc_summary(selected,category_mask,index,weights)
            attribute=next((a for a,pair in pairs.items() if category in pair),'all')
            points[name+'/'+category]=dict(point,scope=name,attribute=attribute,category=category,secondary=attribute=='seen_repetition')
            local[category],draws[category]=point,samples
        for attribute,(a,b) in pairs.items():
            contrasts[name+'/'+attribute]=dict(class_differences(local[a],local[b],draws[a],draws[b]),
                scope=name,attribute=attribute,first_group=a,second_group=b,secondary=attribute=='seen_repetition',
                mechanism_use_permitted=local[a]['mechanism_use_permitted'] and local[b]['mechanism_use_permitted'])
        adjacent[name]=adjacent_results(local,draws,grid,pairs)
        print(json.dumps(dict(status='analyzed_strength_scope',scope=name,residents=len(residents))),flush=True)
    scope('all7000',population['users'],np.ones(len(uids),dtype=bool),ATTRIBUTES,True)
    for attribute,(a,b) in ATTRIBUTES.items():
        common=sorted(set(map(int,uids[flags[a]]))&set(map(int,uids[flags[b]])))
        if len(common)>=512:scope('common_'+attribute,common,np.isin(uids,common),{attribute:(a,b)})
        else:scopes['common_'+attribute]=dict(resident_users=len(common),uids=common,skipped_reason='Common UID count below512')
    if slices is not None:
        for name,entry in slices['slices'].items():
            members=set(entry['users'])
            common=sorted({int(u) for u in uids[flags['history_seen']] if int(u) in members}
                &{int(u) for u in uids[flags['history_unseen']] if int(u) in members})
            if len(common)>=512:scope(name+'_common_history',common,np.isin(uids,common),{'history_familiarity':ATTRIBUTES['history_familiarity']})
            else:scopes[name+'_common_history']=dict(resident_users=len(common),uids=common,skipped_reason='Common UID count below512')
    return points,contrasts,scopes,adjacent


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(folder,population_path,attributes,output,slices_path=None):
    config,population,groups=load_scores(folder,population_path);grid=strength_grid(config)
    definition,manifest,flags=load_attributes(attributes,population_path,config['edge'],groups['reuse'])
    slices=json.loads(slices_path.read_text()) if slices_path else None
    if slices is not None and slices['population_sha256']!=sha(population_path):raise ValueError('population slices changed')
    points,comparisons,scopes,adjacent=analyze_scopes(groups,population,flags,grid,slices)
    limitations=['All six strengths and the five adjacent comparisons were fixed before new quality scoring; no winning strength is selected here.',
        'The base is the frozen linear256_rows map. A strength materializes once at cutover; each policy subsequently generates its own native append states.',
        'Candidate-class AUC slopes are whole migration-policy effects, not pure read-time interventions or validation of a candidate gate.',
        'History familiarity and popularity are primary; seen-once/repeated is secondary; all original rules and thresholds are preserved.',
        'UID pointwise intervals condition on frozen weights, have no multiplicity adjustment, and do not provide training-seed confirmation.',
        'Opposite point slopes are descriptive. Class-change difference alone does not imply both individual changes exclude zero.',
        'Original4000 and added3000 are disjoint development slices; added3000 is not pristine confirmation.',
        'Recovery intervals are suppressed when any bootstrap native gap is nonpositive or undefined; ordinary AUC changes are primary.',
        'No tradeoff and adverse trends are retained; this bounded grid cannot prove every static map incapable or the final Design effective.']
    result=dict(edge=config['edge'],method_names=config['method_names'],method_strengths=config['method_strengths'],base_map_name=config['base_map_name'],
        groups=points,class_differences=comparisons,scopes=scopes,candidate_definition=definition,
        provenance=dict(source=str(folder.resolve()),configuration_sha256=sha(folder/'configuration.json'),source_summary_sha256=sha(folder/'summary.json'),
            population_sha256=sha(population_path),attribute_manifest_sha256=sha(attributes/'manifest.json'),analysis_sha256=sha(__file__),
            sweep_source_sha256=sha(__file__),writer_source_sha256=sha(Path(__file__).with_name('analyze_candidate_robustness.py')),
            source_archive_roles={'strength_source.py':'executed sweep analysis source; matches analysis_sha256',
                'analysis_source.py':'unchanged shared writer/AUC helper source; matches writer_source_sha256'},
            slices_sha256=sha(slices_path) if slices_path else None),limitations=limitations)
    # Existing writer preserves every AUC, native gap, map gain, remaining gap
    # and unbounded recovery ratio. No second AUC/bootstrap pass is needed.
    write_outputs(result,output/'all_auc')
    (output/'all_auc/strength_source.py').write_bytes(Path(__file__).read_bytes())
    change_rows=[];difference_rows=[]
    for scope,records in adjacent.items():
        for record in records:
            prefix=dict(scope=scope,alpha_before=record['alpha_before'],alpha_after=record['alpha_after'],method_before=record['method_before'],method_after=record['method_after'])
            for category,value in record['class_changes'].items():
                metric=value['change'];lo,hi=metric['ci95'] or [None,None]
                change_rows.append(dict(prefix,category=category,feedback_users=value['feedback_users'],mechanism_use_permitted=value['mechanism_use_permitted'],
                    AUC_change=metric['value'],ci_low=lo,ci_high=hi))
            for attribute,value in record['class_change_differences'].items():
                metric=value['change_difference'];lo,hi=metric['ci95'] or [None,None]
                difference_rows.append(dict(prefix,attribute=attribute,first_group=value['first_group'],second_group=value['second_group'],
                    secondary=value['secondary'],mechanism_use_permitted=value['mechanism_use_permitted'],opposite_point_slopes=value['opposite_point_slopes'],
                    first_minus_second_AUC_change=metric['value'],ci_low=lo,ci_high=hi))
    write_csv(output/'adjacent_auc_changes.csv',change_rows)
    write_csv(output/'adjacent_class_change_differences.csv',difference_rows)
    final=dict(result,format='candidate_strength_sweep_analysis_v1',adjacent_strengths=adjacent,
        adjacent_pairs=[[a,b] for a,b in zip(ALPHAS[:-1],ALPHAS[1:])],base_auc_summary_sha256=sha(output/'all_auc/summary.json'))
    (output/'summary.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    lines=[f'# Global migration strength sweep: edge{config["edge"]}','','All predeclared strengths are retained. Adjacent changes use500 shared UID draws; no best strength is selected.','',
        '| Scope | Attribute | Alpha change | First class | Second class | First−second AUC change | 95% CI |','|---|---|---|---|---|---:|---|']
    for row in difference_rows:
        def f(v):return 'NA' if v is None else f'{v:.6f}'
        lines.append(f"| {row['scope']} | {row['attribute']} | {row['alpha_before']}→{row['alpha_after']} | {row['first_group']} | {row['second_group']} | {f(row['first_minus_second_AUC_change'])} | [{f(row['ci_low'])},{f(row['ci_high'])}] |")
    lines+=['','All class AUC/native gap/map gain/remaining gap estimates are in all_auc/metrics.csv. Individual adjacent AUC changes are in adjacent_auc_changes.csv; shared-UID class-change differences are in adjacent_class_change_differences.csv.','',*['- '+v for v in limitations]]
    (output/'report.md').write_text('\n'.join(lines)+'\n');(output/'strength_analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status='complete',edge=config['edge'],scopes=len(scopes),adjacent_comparisons=len(difference_rows))),flush=True)
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--population',type=Path,required=True)
    parser.add_argument('--attributes',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--population-slices',type=Path)
    args=parser.parse_args();run(args.input,args.population,args.attributes,args.output,args.population_slices)
