#!/usr/bin/env python3
"""Simple real AUC by fixed prerelease candidate attributes; CPU only."""
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
import numpy as np
import pyarrow.parquet as pq

METHODS=('reuse','f1','kv_linear','rebuild','f2')
MAPS=('f1','kv_linear','f2')
ATTRIBUTES={'popularity':('popular','long_tail'),'history_familiarity':('history_seen','history_unseen')}


def recovery(gain,gap,gain_draws,gap_draws):
    """Keep the unbounded point ratio; suppress unstable denominator intervals."""
    gain_draws=np.asarray(gain_draws,dtype=float);gap_draws=np.asarray(gap_draws,dtype=float)
    point=100*gain/gap if np.isfinite(gain) and np.isfinite(gap) and gap!=0 else np.nan
    invalid=~np.isfinite(gap_draws)|(gap_draws<=0)
    permitted=np.isfinite(gap) and gap>0 and not invalid.any() and np.isfinite(gain_draws).all()
    band=np.percentile(100*gain_draws/gap_draws,[2.5,97.5]).tolist() if permitted else None
    return dict(value=float(point) if np.isfinite(point) else None,ci95=band,
        ratio_ci_permitted=bool(permitted),nonpositive_or_invalid_denominator_draws=int(invalid.sum()),
        rule='Unclipped100*(methodAUC−ReuseAUC)/(RebuildAUC−ReuseAUC); no ratio interval if point gap or any UID-bootstrap gap is nonpositive/invalid.')


def align(rows):
    groups={m:[] for m in METHODS}
    for row in rows:
        if row['method'] in groups:groups[row['method']].append(row)
    expected=None;expected_labels=None
    for method,values in groups.items():
        if not values:raise ValueError('missing fixed candidate-quality method')
        values.sort(key=lambda x:(x['uid'],x['query_timestamp'],x['item_idx']))
        keys=[(r['uid'],r['query_timestamp'],r['item_idx']) for r in values];labels=[r['label'] for r in values]
        if len(set(keys))!=len(keys):raise ValueError('duplicate or conflicting observed event key')
        if expected is None:expected,expected_labels=keys,labels
        elif keys!=expected or labels!=expected_labels:raise ValueError('methods differ in real event keys or labels')
        if not np.isin(labels,[0,1]).all() or not np.isfinite([r['score'] for r in values]).all():
            raise ValueError('binary labels and finite raw logits required')
    return groups


def summarize(groups,mask,uid_index,weights):
    labels=np.array([r['label'] for r in groups['reuse']],dtype=float)[mask]
    index=uid_index[mask];auc={};draws={}
    if not len(labels):raise ValueError('empty candidate attribute group')
    for method,rows in groups.items():
        scores=np.array([r['score'] for r in rows])[mask]
        auc[method]=weighted_auc(scores,labels,index,np.ones((1,weights.shape[1]),dtype=np.uint16))[0]
        draws[method]=weighted_auc(scores,labels,index,weights)
    gap=auc['rebuild']-auc['reuse'];gap_draws=draws['rebuild']-draws['reuse']
    point={}
    for method in METHODS:
        gain=auc[method]-auc['reuse'];gain_draws=draws[method]-draws['reuse']
        point[method]=dict(AUC=interval(auc[method],draws[method]),
            AUC_gain_over_reuse=interval(gain,gain_draws),
            recovery_percent=recovery(gain,gap,gain_draws,gap_draws))
    feedback_users=len(np.unique(index));positives=int(labels.sum());negatives=len(labels)-positives
    return dict(feedback_users=feedback_users,events=len(labels),positive_events=positives,negative_events=negatives,
        mechanism_use_permitted=feedback_users>=512 and positives>0 and negatives>0,
        rebuild_minus_reuse_AUC=interval(gap,gap_draws),methods=point),draws


def effect_difference(first,second,first_draws,second_draws):
    """Both groups must share the same full4000 resident UID draws."""
    out={}
    for method in MAPS:
        a=first['methods'][method]['AUC_gain_over_reuse']['value']
        b=second['methods'][method]['AUC_gain_over_reuse']['value']
        point=a-b if a is not None and b is not None else np.nan
        sample=(first_draws[method]-first_draws['reuse'])-(second_draws[method]-second_draws['reuse'])
        out[method]=interval(point,sample)
    return out


def load_scores(folder,population_path):
    config=json.loads((folder/'configuration.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    population=json.loads(population_path.read_text())
    if config['diagnostic'] or summary['status']!='complete' or not summary['quality_use_permitted']:
        raise ValueError('requires completed non-diagnostic cap1024 observed-feedback run')
    if config['settings']['history_length']!=1024 or config['population_sha256']!=sha(population_path) or config['users']!=population['users']:
        raise ValueError('requires the fixed cap1024 main+sensA4000 population')
    if len(config['users'])!=4000 or len(set(config['users']))!=4000:raise ValueError('requires4000 residents')
    frozen=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    rows=[];all_rows=0
    for artifact in summary['artifacts']:
        block=folder/artifact['path'];manifest=folder/artifact['manifest']
        if sha(block)!=artifact['sha256'] or sha(manifest)!=artifact['manifest_sha256']:
            raise ValueError('candidate source block or manifest changed')
        record=json.loads(manifest.read_text())
        if record['configuration_sha256']!=frozen or record['score_sha256']!=sha(block):
            raise ValueError('candidate source belongs to another frozen configuration')
        values=pq.read_table(block,columns=['uid','query_timestamp','item_idx','label','method','score']).to_pylist()
        if len(values)!=record['score_rows']:raise ValueError('candidate source block row count changed')
        rows.extend(r for r in values if r['method'] in METHODS);all_rows+=len(values)
    if all_rows!=summary['score_rows']:raise ValueError('candidate source total row count changed')
    groups=align(rows)
    if set(r['uid'] for r in groups['reuse'])!=set(config['feedback_users']):
        raise ValueError('candidate source feedback UID set changed')
    return config,population,groups


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def write_outputs(result,output):
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=[];differences=[]
    for name,group in result['groups'].items():
        for method,metrics in group['methods'].items():
            row=dict(scope=group['scope'],attribute=group['attribute'],group=group['category'],method=method,feedback_users=group['feedback_users'],events=group['events'],
                mechanism_use_permitted=group['mechanism_use_permitted'],sensitivity=method=='f2')
            for metric,value in dict(metrics,rebuild_minus_reuse_AUC=group['rebuild_minus_reuse_AUC']).items():
                row[metric]=value['value'];row[metric+'_ci_low'],row[metric+'_ci_high']=value['ci95'] or [None,None]
            row['ratio_ci_permitted']=metrics['recovery_percent']['ratio_ci_permitted']
            row['nonpositive_or_invalid_denominator_draws']=metrics['recovery_percent']['nonpositive_or_invalid_denominator_draws']
            rows.append(row)
    for attribute,comparison in result['effect_differences'].items():
        for method,value in comparison['methods'].items():
            lo,hi=value['ci95'] or [None,None]
            differences.append(dict(scope=comparison['scope'],attribute=comparison['attribute'],first_group=comparison['first_group'],second_group=comparison['second_group'],
                method=method,AUC_gain_difference=value['value'],ci_low=lo,ci_high=hi,
                mechanism_use_permitted=comparison['mechanism_use_permitted']))
    write_csv(output/'metrics.csv',rows);write_csv(output/'group_effect_difference.csv',differences)
    def f(x):return 'NA' if x is None else f'{x:.6f}'
    lines=[f'# 候选分类真实AUC：edge{result["edge"]}','','固定cap1024、main+sensA4000；只使用既有真实请求分数，不启动GPU。F2仅为更强静态映射敏感性。','',
        '| 范围 | 分组 | 反馈UID | 方法 | AUC | 相对Reuse增益 | Rebuild−Reuse gap | 恢复率(%) | 恢复率区间 |',
        '|---|---|---:|---|---:|---:|---:|---:|---|']
    for row in rows:
        interval_text='不报告：分母不稳定' if not row['ratio_ci_permitted'] else '['+f(row['recovery_percent_ci_low'])+', '+f(row['recovery_percent_ci_high'])+']'
        lines.append(f'| {row["scope"]} | {row["group"]} | {row["feedback_users"]} | {row["method"]} | '+
            ' | '.join(f(row[x]) for x in ['AUC','AUC_gain_over_reuse','rebuild_minus_reuse_AUC','recovery_percent'])+' | '+interval_text+' |')
    lines += ['', '恢复率=100×(method AUC−Reuse AUC)/(Rebuild AUC−Reuse AUC)。点值不截断；只要分母点值非正或任一bootstrap分母非正/无效，即不报告有界ratio区间。不同组的分母不同，恢复率差不能直接称为候选异质性。','',
        '组间AUC增益差使用完全相同的4000 UID抽样权重，比较(groupA method−Reuse)−(groupB method−Reuse)，见group_effect_difference.csv。','',
        *['- '+x for x in result['limitations']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())


def run(folder,population_path,attributes,output):
    config,population,groups=load_scores(folder,population_path)
    manifest=json.loads((attributes/'manifest.json').read_text())
    for field in ['definition','event_mapping','frequency']:
        if sha(Path(manifest[field]['path']))!=manifest[field]['sha256']:
            raise ValueError('fixed prerelease candidate attribute artifact changed')
    definition=json.loads(Path(manifest['definition']['path']).read_text())
    if (manifest['population_sha256']!=sha(population_path) or definition['population_sha256']!=sha(population_path)
            or definition['edge']!=config['edge'] or definition['history_cap']!=1024):
        raise ValueError('candidate attributes belong to another population/edge/cap')
    if (definition['score_values_read'] or definition['threshold_search_performed'] or definition['original_roles_modified']
            or not definition['group_rules_frozen_before_future_requests_parsed']):
        raise ValueError('candidate attribute rules must be frozen without model scores or outcome selection')
    attribute_rows=pq.read_table(manifest['event_mapping']['path']).to_pylist()
    if len(attribute_rows)!=manifest['event_mapping']['rows']:raise ValueError('candidate attribute row count changed')
    attribute_rows.sort(key=lambda x:(x['uid'],x['query_timestamp'],x['item_idx']))
    keys=lambda rows:[(r['uid'],r['query_timestamp'],r['item_idx']) for r in rows]
    attribute_keys=keys(attribute_rows)
    if len(set(attribute_keys))!=len(attribute_keys) or attribute_keys!=keys(groups['reuse']):
        raise ValueError('candidate attributes must map every real observation exactly once')
    if any(not isinstance(r[field],bool) for r in attribute_rows for field in ['popular','history_seen']):
        raise ValueError('candidate attribute flags must be nonmissing booleans')
    flags={'popular':np.array([r['popular'] for r in attribute_rows]),'history_seen':np.array([r['history_seen'] for r in attribute_rows])}
    flags['long_tail']=~flags['popular'];flags['history_unseen']=~flags['history_seen']
    uids=np.array([r['uid'] for r in groups['reuse']],dtype=np.uint64)
    results={};differences={};scopes={}
    def analyze_scope(name,residents,event_mask,attribute_pairs,include_all=False):
        selected={m:[r for r,keep in zip(values,event_mask,strict=True) if keep] for m,values in groups.items()}
        selected_uids=uids[event_mask];index,weights=uid_weights(residents,selected_uids)
        scopes[name]=dict(resident_users=len(residents),feedback_users=len(np.unique(selected_uids)),uids=list(map(int,residents)),
            mechanism_use_permitted=len(np.unique(selected_uids))>=512,
            bootstrap='500 shared UID draws within this declared resident scope; identical draw multiplicities for every candidate class and method.')
        local_points={};local_draws={}
        if include_all:
            point,_=summarize(selected,np.ones(len(selected_uids),dtype=bool),index,weights)
            results[name+'/all']=dict(point,scope=name,attribute='all',category='all')
        for attribute,(first,second) in attribute_pairs.items():
            for category in [first,second]:
                mask=flags[category][event_mask]
                point,draws=summarize(selected,mask,index,weights)
                point.update(scope=name,attribute=attribute,category=category)
                results[name+'/'+category]=point;local_points[category]=point;local_draws[category]=draws
            differences[name+'/'+attribute]=dict(scope=name,attribute=attribute,first_group=first,second_group=second,
                mechanism_use_permitted=local_points[first]['mechanism_use_permitted'] and local_points[second]['mechanism_use_permitted'],
                methods=effect_difference(local_points[first],local_points[second],local_draws[first],local_draws[second]))
        print(json.dumps(dict(status='analyzed_scope',edge=config['edge'],scope=name,residents=len(residents))),flush=True)
    analyze_scope('all4000',population['users'],np.ones(len(uids),dtype=bool),ATTRIBUTES,True)
    for attribute,(first,second) in ATTRIBUTES.items():
        common=sorted(set(map(int,uids[flags[first]]))&set(map(int,uids[flags[second]])))
        declared=manifest['coverage']['paired_classes'][attribute]['uids']
        if common!=declared:raise ValueError('same-UID attribute support changed')
        name='common_'+attribute
        if len(common)>=512:
            analyze_scope(name,common,np.isin(uids,common),{attribute:(first,second)})
        else:scopes[name]=dict(resident_users=len(common),uids=common,mechanism_use_permitted=False,
            skipped_reason='Fewer than512 UIDs with observed requests in both candidate classes; no paired class quality analysis.')
    metric_source=Path(sys.modules[uid_weights.__module__].__file__)
    result=dict(edge=config['edge'],primary_mapping='kv_linear',static_sensitivities=['f1','f2'],
        method_order=list(METHODS),groups=results,effect_differences=differences,scopes=scopes,
        candidate_definition=definition,provenance=dict(source=str(folder.resolve()),configuration_sha256=sha(folder/'configuration.json'),
            source_summary_sha256=sha(folder/'summary.json'),attribute_manifest=str((attributes/'manifest.json').resolve()),
            attribute_manifest_sha256=sha(attributes/'manifest.json'),population_sha256=sha(population_path),analysis_sha256=sha(__file__),
            shared_metric_source_sha256=sha(metric_source)),
        limitations=['Primary static mapping is kv_linear; F1 and stronger F2 remain parallel sensitivities, never chosen from subgroup trends.',
            'Pooled observed-feedback AUC only; no macro-user AUC or synthetic candidates/labels.',
            'Class definitions use prerelease attributes and no model outcomes; fixed top20% popularity rule is unchanged despite inadequate tail/common coverage.',
            'Every all4000 comparison shares the same4000 UID resamples across classes and methods; common-UID sensitivity resamples its same shared UID subset.',
            'Common-UID groups condition on observed requests in both classes; they reduce user-composition differences but do not create a causal candidate intervention.',
            'Pointwise exploratory confidence intervals condition on frozen backbones/maps; no independent confirmation or multiple-comparison guarantee.',
            'AUC recovery is an unclipped ratio with an unstable denominator; inspect raw AUC gaps and paired gain differences before comparing recovery.',
            'Any class with fewer than512 feedback UIDs is diagnostic only; popular/long-tail common UID comparisons below512 are skipped.',
            'Popularity, familiarity, labels, time and item identity can be correlated; class-specific quality does not prove gate necessity or a specific implementation.',
            'All requests follow the original continuous cap1024 replay; no new GPU inference, fitting or source modification.'])
    write_outputs(result,output)
    (output/'shared_metrics_source.py').write_bytes(metric_source.read_bytes())
    return result



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--population',type=Path,required=True)
    p.add_argument('--attributes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=run(args.input,args.population,args.attributes,args.output)
    print(json.dumps(dict(status='complete',edge=result['edge'])))
