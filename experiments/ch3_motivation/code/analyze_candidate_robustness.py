#!/usr/bin/env python3
"""CPU AUC analysis of fixed candidate classes and cheap static map recipes."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.analyze_real_quality import uid_weights, interval, sha, weighted_auc
from experiments.ch3_motivation.code.analyze_candidate_class_auc import recovery
import numpy as np
import pyarrow.parquet as pq

ATTRIBUTES={'popularity':('popular','long_tail'),
    'history_familiarity':('history_seen','history_unseen'),
    'seen_repetition':('history_once','history_repeated')}


def align(rows,methods):
    groups={m:[] for m in methods}
    if len(groups)!=len(methods) or not {'reuse','rebuild'}<=set(methods):
        raise ValueError('unique static methods and reuse/rebuild anchors required')
    if any('oracle' in m or m in ['f_key','f_value','fg','fgh'] for m in methods):
        raise ValueError('this study contains static maps only')
    for row in rows:
        if row['method'] not in groups:raise ValueError('unconfigured quality method')
        groups[row['method']].append(row)
    expected=labels=None
    for method,values in groups.items():
        values.sort(key=lambda r:(r['uid'],r['query_timestamp'],r['item_idx']))
        actual=[(r['uid'],r['query_timestamp'],r['item_idx']) for r in values]
        y=[r['label'] for r in values]
        if not values or len(set(actual))!=len(actual):raise ValueError('missing method or duplicate event key')
        if expected is None:expected,labels=actual,y
        elif expected!=actual or labels!=y:raise ValueError('static maps differ in observed event keys/labels')
        if not np.isin(y,[0,1]).all() or not np.isfinite([r['score'] for r in values]).all():
            raise ValueError('requires binary observed labels and finite raw logits')
    return groups


def auc_summary(groups,mask,index,weights):
    y=np.array([r['label'] for r in groups['reuse']],dtype=float)[mask];ix=index[mask]
    if not len(y):raise ValueError('empty candidate class')
    auc={};draws={};points={}
    for method,rows in groups.items():
        score=np.array([r['score'] for r in rows])[mask]
        auc[method]=weighted_auc(score,y,ix,np.ones((1,weights.shape[1]),dtype=np.uint16))[0]
        draws[method]=weighted_auc(score,y,ix,weights)
    baseline=auc['rebuild']-auc['reuse'];baseline_draws=draws['rebuild']-draws['reuse']
    for method in groups:
        gain=auc[method]-auc['reuse'];remaining=auc['rebuild']-auc[method]
        gain_draws=draws[method]-draws['reuse'];remaining_draws=draws['rebuild']-draws[method]
        points[method]=dict(AUC=interval(auc[method],draws[method]),
            teacher_minus_map_AUC=interval(remaining,remaining_draws),
            map_minus_reuse_AUC=interval(gain,gain_draws),
            recovery_percent=recovery(gain,baseline,gain_draws,baseline_draws))
    positive=int(y.sum());negative=len(y)-positive;users=len(np.unique(ix))
    return dict(feedback_users=users,events=len(y),positive_events=positive,negative_events=negative,
        mechanism_use_permitted=users>=512 and positive>0 and negative>0,
        teacher_minus_reuse_AUC=interval(baseline,baseline_draws),methods=points),draws


def class_differences(first,second,first_draws,second_draws):
    """All standard AUC contrasts use the same UID draws across the two classes."""
    def difference(point_a,point_b,draw_a,draw_b):
        point=point_a-point_b if point_a is not None and point_b is not None else np.nan
        return interval(point,draw_a-draw_b)
    native_a=first['teacher_minus_reuse_AUC']['value'];native_b=second['teacher_minus_reuse_AUC']['value']
    native_samples=(first_draws['rebuild']-first_draws['reuse'])-(second_draws['rebuild']-second_draws['reuse'])
    baseline=difference(native_a,native_b,first_draws['rebuild']-first_draws['reuse'],second_draws['rebuild']-second_draws['reuse'])
    methods={}
    for method in first['methods']:
        if method in ['reuse','rebuild']:continue
        a,b=first['methods'][method],second['methods'][method]
        gain_samples=(first_draws[method]-first_draws['reuse'])-(second_draws[method]-second_draws['reuse'])
        remaining_samples=(first_draws['rebuild']-first_draws[method])-(second_draws['rebuild']-second_draws[method])
        finite=np.isfinite(native_samples)&np.isfinite(gain_samples)&np.isfinite(remaining_samples)
        if not np.allclose(remaining_samples[finite],(native_samples-gain_samples)[finite],atol=1e-14,rtol=0):
            raise ValueError('paired AUC decomposition identity failed')
        methods[method]=dict(
            remaining_gap_difference=difference(a['teacher_minus_map_AUC']['value'],b['teacher_minus_map_AUC']['value'],
                first_draws['rebuild']-first_draws[method],second_draws['rebuild']-second_draws[method]),
            map_gain_difference=difference(a['map_minus_reuse_AUC']['value'],b['map_minus_reuse_AUC']['value'],
                first_draws[method]-first_draws['reuse'],second_draws[method]-second_draws['reuse']))
    return dict(native_gap_difference=baseline,methods=methods,
        identity='remaining gap difference = native gap difference − map gain difference; all are ordinary paired AUC differences.')


def load_scores(folder,population_path):
    config=json.loads((folder/'configuration.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    population=json.loads(population_path.read_text());methods=config['method_names']
    if population['original_roles_modified'] or population['score_values_read']:
        raise ValueError('fixed development population must preserve roles without reading scores')
    if config['diagnostic'] or summary['status']!='complete' or not summary['quality_use_permitted']:
        raise ValueError('requires completed nondiagnostic static-map quality')
    if config['settings']['history_length']!=1024 or config['population_sha256']!=sha(population_path) or config['users']!=population['users']:
        raise ValueError('requires fixed development7000 cap1024 inputs')
    if len(config['users'])!=7000 or len(set(config['users']))!=7000:raise ValueError('requires7000 fixed residents')
    frozen=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    rows=[]
    for artifact in summary['artifacts']:
        block=folder/artifact['path'];manifest=folder/artifact['manifest']
        if sha(block)!=artifact['sha256'] or sha(manifest)!=artifact['manifest_sha256']:
            raise ValueError('static quality block or manifest changed')
        record=json.loads(manifest.read_text())
        if record['configuration_sha256']!=frozen or record['score_sha256']!=sha(block):raise ValueError('static score configuration changed')
        values=pq.read_table(block,columns=['uid','query_timestamp','item_idx','label','method','score']).to_pylist()
        if len(values)!=record['score_rows']:raise ValueError('static score row count changed')
        rows.extend(values)
    if len(rows)!=summary['score_rows']:raise ValueError('static total score count changed')
    groups=align(rows,methods)
    if set(r['uid'] for r in groups['reuse'])!=set(config['feedback_users']):raise ValueError('static feedback UID set changed')
    if len(config['feedback_users'])<512:raise ValueError('needs at least512 feedback UIDs')
    return config,population,groups


def load_attributes(folder,population_path,edge,base):
    manifest=json.loads((folder/'manifest.json').read_text())
    for field in ['definition','event_mapping','global_frequency']:
        if sha(Path(manifest[field]['path']))!=manifest[field]['sha256']:raise ValueError('fixed candidate attribute artifact changed')
    definition=json.loads(Path(manifest['definition']['path']).read_text())
    if manifest['population_sha256']!=sha(population_path) or definition['population_sha256']!=sha(population_path) or definition['edge']!=edge:
        raise ValueError('candidate attributes belong to a different fixed population or edge')
    if definition['score_values_read'] or definition['threshold_search_performed'] or not definition['repeated_split_is_natural_integer_boundary']:
        raise ValueError('candidate rules must not use model scores or alter roles')
    original=definition['original_attribute_definition']
    if sha(Path(original['path']))!=original['sha256']:raise ValueError('original attribute definition changed')
    if not json.loads(Path(original['path']).read_text())['group_rules_frozen_before_future_requests_parsed']:
        raise ValueError('original candidate rules not preregistered')
    rows=pq.read_table(manifest['event_mapping']['path']).to_pylist()
    rows.sort(key=lambda r:(r['uid'],r['query_timestamp'],r['item_idx']))
    keys=lambda values:[(r['uid'],r['query_timestamp'],r['item_idx']) for r in values]
    if len(rows)!=manifest['event_mapping']['rows'] or keys(rows)!=keys(base) or len(set(keys(rows)))!=len(rows):
        raise ValueError('attributes must map every observed event exactly once')
    if any(not isinstance(r[f],bool) for r in rows for f in ['popular','history_seen']):raise ValueError('missing attribute flags')
    counts=np.array([r['history_item_occurrences'] for r in rows],dtype=int)
    seen=np.array([r['history_seen'] for r in rows]);popular=np.array([r['popular'] for r in rows])
    if (counts<0).any() or not np.array_equal(seen,counts>0):raise ValueError('seen flag disagrees with prerelease history occurrences')
    return definition,manifest,dict(popular=popular,long_tail=~popular,history_seen=seen,history_unseen=~seen,
        history_once=counts==1,history_repeated=counts>=2)


def analyze_scopes(groups,population,flags):
    uids=np.array([r['uid'] for r in groups['reuse']],dtype=np.uint64)
    points={};comparisons={};scopes={}
    def scope(name,residents,mask,pairs,include_all=False):
        selected={m:[r for r,keep in zip(values,mask,strict=True) if keep] for m,values in groups.items()}
        index,weights=uid_weights(residents,uids[mask]);local={};draws={}
        scopes[name]=dict(resident_users=len(residents),uids=list(map(int,residents)),bootstrap_replicates=500,
            scope='Identical UID multiplicities for every method and class within this resident scope.')
        if include_all:
            point,_=auc_summary(selected,np.ones(mask.sum(),dtype=bool),index,weights)
            points[name+'/all']=dict(point,scope=name,attribute='all',category='all',secondary=False)
        for attribute,categories in pairs.items():
            for category in categories:
                point,samples=auc_summary(selected,flags[category][mask],index,weights)
                points[name+'/'+category]=dict(point,scope=name,attribute=attribute,category=category,secondary=attribute=='seen_repetition')
                local[category],draws[category]=point,samples
            a,b=categories
            comparisons[name+'/'+attribute]=dict(class_differences(local[a],local[b],draws[a],draws[b]),
                scope=name,attribute=attribute,first_group=a,second_group=b,secondary=attribute=='seen_repetition',
                mechanism_use_permitted=local[a]['mechanism_use_permitted'] and local[b]['mechanism_use_permitted'])
        print(json.dumps(dict(status='analyzed_scope',scope=name,residents=len(residents))),flush=True)
    scope('all7000',population['users'],np.ones(len(uids),dtype=bool),ATTRIBUTES,True)
    for attribute,(first,second) in ATTRIBUTES.items():
        common=sorted(set(map(int,uids[flags[first]]))&set(map(int,uids[flags[second]])))
        name='common_'+attribute
        if len(common)>=512:scope(name,common,np.isin(uids,common),{attribute:(first,second)})
        else:scopes[name]=dict(resident_users=len(common),uids=common,skipped_reason='Common UID count below512; no paired quality analysis.')
    return points,comparisons,scopes


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)


def flatten(row,key,value):
    row[key]=value['value'];row[key+'_ci_low'],row[key+'_ci_high']=value['ci95'] or [None,None]


def write_outputs(result,output):
    output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=[];contrasts=[]
    for group in result['groups'].values():
        for method,metrics in group['methods'].items():
            row={k:group[k] for k in ['scope','attribute','category','secondary','feedback_users','events','mechanism_use_permitted']}
            row['method']=method;flatten(row,'teacher_minus_reuse_AUC',group['teacher_minus_reuse_AUC'])
            for key,value in metrics.items():flatten(row,key,value)
            row['ratio_ci_permitted']=metrics['recovery_percent']['ratio_ci_permitted']
            row['nonpositive_denominator_draws']=metrics['recovery_percent']['nonpositive_or_invalid_denominator_draws'];rows.append(row)
    for comparison in result['class_differences'].values():
        for method,metrics in comparison['methods'].items():
            row={k:comparison[k] for k in ['scope','attribute','first_group','second_group','secondary','mechanism_use_permitted']}
            row['method']=method;flatten(row,'native_gap_difference',comparison['native_gap_difference'])
            for key,value in metrics.items():flatten(row,key,value)
            contrasts.append(row)
    write_csv(output/'metrics.csv',rows);write_csv(output/'paired_class_differences.csv',contrasts)
    def f(x):return 'NA' if x is None else f'{x:.6f}'
    lines=[f'# Candidate robustness: edge{result["edge"]}','','Main comparisons use ordinary paired AUC differences. Recovery is descriptive and never clipped.','',
        '| scope | class | UID | method | native gap | map gain | remaining gap | recovery % |','|---|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+' | '.join(str(row[k]) for k in ['scope','category','feedback_users','method'])+' | '+
            ' | '.join(f(row[k]) for k in ['teacher_minus_reuse_AUC','map_minus_reuse_AUC','teacher_minus_map_AUC','recovery_percent'])+' |')
    lines += ['','Remaining-gap difference = native-gap difference − map-gain difference. Different native difficulty can explain class residuals; class residuals alone do not prove candidate gates are necessary.','',
        *['- '+x for x in result['limitations']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n');(output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())


def run(folder,population_path,attributes,output):
    config,population,groups=load_scores(folder,population_path)
    definition,manifest,flags=load_attributes(attributes,population_path,config['edge'],groups['reuse'])
    points,differences,scopes=analyze_scopes(groups,population,flags)
    result=dict(edge=config['edge'],method_names=config['method_names'],groups=points,class_differences=differences,scopes=scopes,
        candidate_definition=definition,provenance=dict(source=str(folder.resolve()),configuration_sha256=sha(folder/'configuration.json'),
            source_summary_sha256=sha(folder/'summary.json'),population_sha256=sha(population_path),
            attribute_manifest_sha256=sha(attributes/'manifest.json'),analysis_sha256=sha(__file__)),
        limitations=['Fixed7000 development users overlap earlier4000; this is not independent confirmation.',
            'Mappings are compared dynamically without outcome-based method selection; F1/F2 remain static sensitivities.',
            'Primary quantities are teacher−map remaining AUC, teacher−reuse native AUC gap, and map−reuse AUC gain.',
            'Recovery uses an unclipped ratio; no ratio interval if any bootstrap denominator or point denominator is nonpositive/invalid.',
            'All methods and candidate classes share UID weights within each population scope; common-UID controls reduce user composition differences.',
            'Pooled observed-feedback AUC, not macro-user AUC; fewer than512 feedback UIDs or single-label groups are diagnostic only.',
            'Remaining AUC gaps can scale with baseline difficulty; residual heterogeneity does not establish the necessity or efficacy of a gate.',
            'Seen once versus repeatedly seen is secondary and uses only seen candidates; no crossproduct strata are formed.',
            'Pointwise exploratory intervals are conditional on frozen weights; no multiplicity-adjusted or training-seed confirmation.',
            'No GPU work, oracle/native K/V split or f/fg/fgh ladder is launched by this analysis.'])
    write_outputs(result,output);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--population',type=Path,required=True)
    p.add_argument('--attributes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=run(a.input,a.population,a.attributes,a.output)
    print(json.dumps(dict(status='complete',edge=result['edge'])))
