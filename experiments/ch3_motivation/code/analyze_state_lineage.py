#!/usr/bin/env python3
"""Paired real-label AUC for direct versus chained fixed-snapshot KV maps."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
import numpy as np
import pyarrow.parquet as pq
from experiments.ch3_motivation.code.analyze_real_quality import (
    binary_point_draws, uid_weights, interval, sha, write_csv, flatten_interval,
)
from experiments.ch3_motivation.code.analyze_candidate_robustness import align
from experiments.ch3_motivation.code.analyze_candidate_class_auc import recovery

METHODS=('reuse','direct','chain','rebuild')
KEYS=('uid','query_timestamp','item_idx')
MSE_FIELDS=('kv_mse','k_mse','v_mse')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_scores(folder,population_path):
    config=json.loads((folder/'configuration.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    population=json.loads(population_path.read_text())
    edge=int(config['final_target'])
    if edge not in (3,4) or config['diagnostic'] or summary['status']!='complete' or not summary['quality_use_permitted']:
        raise ValueError('requires completed nondiagnostic target3/4 quality')
    if config['population_sha256']!=sha(population_path) or config['users']!=population['users']:
        raise ValueError('fixed resident population differs')
    if len(config['users'])!=2048 or len(set(config['users']))!=2048:
        raise ValueError('requires the fixed2048 resident population')
    if set(config['method_names'])!=set(METHODS) or len(config['method_names'])!=len(METHODS):
        raise ValueError('only the reduced four-method comparison is permitted')
    if population['original_roles_modified'] or population['score_values_read'] or any(population['role_separation'].values()):
        raise ValueError('evaluation roles must remain frozen and separated')
    entry=population['edges'][str(edge)]
    if config['cutoff']!=entry['cutover_timestamp'] or config['input_binding_sha256']!=entry['inputs']['binding']['sha256']:
        raise ValueError('history cutoff/input identity differs')
    rows=[];frozen=digest(config)
    for artifact in summary['artifacts']:
        block=folder/artifact['path'];manifest=folder/artifact['manifest']
        if sha(block)!=artifact['sha256'] or sha(manifest)!=artifact['manifest_sha256']:
            raise ValueError('quality block or manifest changed')
        record=json.loads(manifest.read_text())
        if record['configuration_sha256']!=frozen or record['score_sha256']!=sha(block):
            raise ValueError('score block configuration differs')
        values=pq.read_table(block,columns=[*KEYS,'label','method','score','appended_events','history_rows','query_gap',*MSE_FIELDS]).to_pylist()
        if len(values)!=record['score_rows']:raise ValueError('block row count differs')
        rows.extend(values)
    if len(rows)!=summary['score_rows']:raise ValueError('total score row count differs')
    groups=align(rows,METHODS);base=groups['reuse']
    observed=pq.read_table(entry['inputs']['requests']['path'],columns=[*KEYS,'label']).to_pylist()
    selected=set(population['users'])
    expected=sorted(tuple(r[k] for k in (*KEYS,'label')) for r in observed if int(r['uid']) in selected)
    actual=[tuple(r[k] for k in (*KEYS,'label')) for r in base]
    if actual!=expected:raise ValueError('scores do not cover exactly the frozen real-label requests')
    feedback=len({r['uid'] for r in base})
    if feedback<512 or feedback!=summary['feedback_users']:raise ValueError('feedback UID coverage differs or is below512')
    for method,values in groups.items():
        for row,reference in zip(values,base,strict=True):
            uid=str(row['uid'])
            if (row['appended_events']!=0 or row['history_rows']!=entry['user_metadata'][uid]['history_count']
                or row['query_gap']!=reference['query_gap'] or row['query_gap']<0):
                raise ValueError('methods differ in fixed history or shared query gap')
        if method=='rebuild' and any(r[f]!=0 for r in values for f in MSE_FIELDS):
            raise ValueError('Rebuild snapshot must be its own zero-error teacher')
    return config,summary,population,groups


def summarize(groups,population):
    base=groups['reuse'];uids=np.asarray([r['uid'] for r in base],dtype=np.uint64)
    y=np.asarray([r['label'] for r in base],dtype=np.int8)
    index,weights=uid_weights(population,uids,replicates=500,seed=17)
    points={};draws={}
    for method,rows in groups.items():
        p,d=binary_point_draws(y,np.asarray([r['score'] for r in rows]),index,weights)
        points[method]={key:p[key] for key in ('AUC','log_loss')}
        draws[method]={key:d[key] for key in ('AUC','log_loss')}
        # Snapshot error is constant per UID; do not count frequent requests repeatedly.
        per_uid={}
        for row in rows:
            values=tuple(row[f] for f in MSE_FIELDS)
            if not np.isfinite(values).all() or any(v<0 for v in values):raise ValueError('invalid snapshot KV MSE')
            if row['uid'] in per_uid and per_uid[row['uid']]!=values:raise ValueError('snapshot error changed across requests')
            per_uid[row['uid']]=values
        points[method]['feedback_uid_mean_snapshot_MSE']={f:float(np.mean([v[j] for v in per_uid.values()])) for j,f in enumerate(MSE_FIELDS)}
    gap=points['rebuild']['AUC']['value']-points['reuse']['AUC']['value']
    gap_draws=draws['rebuild']['AUC']-draws['reuse']['AUC']
    for method,p in points.items():
        for metric in ('AUC','log_loss'):
            for anchor in ('reuse','rebuild'):
                p[metric+'_minus_'+anchor]=interval(p[metric]['value']-points[anchor][metric]['value'],draws[method][metric]-draws[anchor][metric])
        gain=p['AUC']['value']-points['reuse']['AUC']['value']
        p['recovery_percent']=recovery(gain,gap,draws[method]['AUC']-draws['reuse']['AUC'],gap_draws)
    contrasts={metric:interval(points['direct'][metric]['value']-points['chain'][metric]['value'],draws['direct'][metric]-draws['chain'][metric]) for metric in ('AUC','log_loss')}
    positive=int(y.sum());feedback=len(np.unique(uids))
    return dict(resident_users=len(population),feedback_users=feedback,events=len(y),positive_events=positive,
        negative_events=len(y)-positive,mechanism_use_permitted=feedback>=512 and 0<positive<len(y),
        rebuild_minus_reuse_AUC=interval(gap,gap_draws),methods=points,paired_direct_minus_chain=contrasts)


def run(folder,population_path,output):
    config,source_summary,population,groups=load_scores(folder,population_path)
    point=summarize(groups,population['users'])
    limitations=[
        'All caches, including Rebuild, remain at the same final-target cutover snapshot. Real request candidates/labels and the gap from all real pre-query history are shared; this is a counterfactual mechanism case, not serving replay.',
        'Direct and chain are ordinary fitted maps, not the final compact-state Design. Positive or negative AUC contrasts are both retained.',
        'Reuse means sourceV1 native KV read by the final-target model; it is not adjacent-version native Reuse.',
        'Target3 uses two chain migrations and target4 three; comparing different final targets does not establish same-target monotonic growth with migration count.',
        'Each fitted stage uses the same steps and template, but chain has more stages, total learned parameters and fitting/preparation work than direct; this is not a matched-total-budget superiority comparison.',
        'Five hundred paired UID bootstrap draws condition on the frozen weights. Intervals are pointwise and do not include training-seed uncertainty.',
        'Recovery is unclipped and can be negative or exceed100%; ordinary AUC and gain remain primary when the matched native gap is small/nonpositive.',
        'Snapshot KV MSE is an auxiliary, equally weighted mean over feedbackUID, not fit MSE or proof of task quality.']
    result=dict(format='state_lineage_real_quality_analysis_v1',edge=config['final_target'],method_names=list(METHODS),
        primary_contrast='direct_minus_chain: positive AUC favors direct; negative favors chain; no favorable direction is assumed',
        metrics=point,scope=config['comparison_scope'],limitations=limitations,
        provenance=dict(source=str(folder.resolve()),configuration_sha256=sha(folder/'configuration.json'),
            source_summary_sha256=sha(folder/'summary.json'),population_sha256=sha(population_path),analysis_sha256=sha(__file__),
            input_binding_sha256=config['input_binding_sha256'],checkpoint_sha256=config['checkpoint_sha256'],
            metric_helper_sha256=sha(Path(__file__).with_name('analyze_real_quality.py')),
            recovery_helper_sha256=sha(Path(__file__).with_name('analyze_candidate_class_auc.py'))))
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    metric_rows=[]
    for method,metrics in point['methods'].items():
        row=dict(method=method,feedback_users=point['feedback_users'],events=point['events'])
        for name,value in metrics.items():
            if name=='feedback_uid_mean_snapshot_MSE':row.update({'feedback_uid_mean_'+k:v for k,v in value.items()})
            else:flatten_interval(row,name,value)
        row['recovery_ratio_ci_permitted']=metrics['recovery_percent']['ratio_ci_permitted'];metric_rows.append(row)
    write_csv(output/'metrics.csv',metric_rows)
    contrast_row=dict(contrast='direct_minus_chain',feedback_users=point['feedback_users'])
    for metric,value in point['paired_direct_minus_chain'].items():flatten_interval(contrast_row,metric,value)
    write_csv(output/'paired_contrasts.csv',[contrast_row])
    def fmt(x):return 'NA' if x is None else f'{x:.6f}'
    lines=[f'# Direct versus chain: target{config["final_target"]}','','Fixed2048 residents, '+str(point['feedback_users'])+' actual feedbackUID. All statistics use observed labels and500 shared UID draws.','',
        '| Method | AUC | AUC−Reuse | Recovery % | Log loss | UID-mean snapshot KV MSE |','|---|---:|---:|---:|---:|---:|']
    for method,p in point['methods'].items():lines.append('| '+' | '.join([method,fmt(p['AUC']['value']),fmt(p['AUC_minus_reuse']['value']),fmt(p['recovery_percent']['value']),fmt(p['log_loss']['value']),fmt(p['feedback_uid_mean_snapshot_MSE']['kv_mse'])])+' |')
    lines+=['','Direct−chain (positive AUC favors direct):']
    for metric,value in point['paired_direct_minus_chain'].items():lines.append(f"- {metric}: {fmt(value['value'])};95% CI {value['ci95']}.")
    lines+=['','Per-method and contrast confidence intervals are recorded in metrics.csv and paired_contrasts.csv.','',*['- '+v for v in limitations]]
    (output/'report.md').write_text('\n'.join(lines)+'\n');(output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status='complete',edge=config['final_target'],feedback_users=point['feedback_users'],direct_minus_chain=point['paired_direct_minus_chain'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--population',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.input,a.population,a.output)
