#!/usr/bin/env python3
"""CPU metrics for request-local native K/V source counterfactuals.

Every method scores exactly the same observed feedback. The four cache pairs
are a read diagnostic; they do not define four persistent native write paths.
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
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.analyze_real_quality import (
    binary_point_draws, interval, uid_weights, sha,
)
import numpy as np
import pyarrow.parquet as pq

METHODS=('oldKoldV','teacherKoldV','oldKteacherV','teacherKteacherV')
BASE,KEY,VALUE,TEACHER=METHODS
# These coefficients are effects on a larger-is-better metric. For log loss
# they are negated, so positive always denotes reduced loss.
CONTRASTS={
    'key_effect_with_old_value':{KEY:1,BASE:-1},
    'key_effect_with_teacher_value':{TEACHER:1,VALUE:-1},
    'value_effect_with_old_key':{VALUE:1,BASE:-1},
    'value_effect_with_teacher_key':{TEACHER:1,KEY:-1},
    'joint_teacher_effect':{TEACHER:1,BASE:-1},
    'joint_minus_sum_of_single_effects':{TEACHER:1,KEY:-1,VALUE:-1,BASE:1},
}


def align(rows):
    groups={m:[] for m in METHODS}
    for row in rows:
        if row['method'] not in groups:raise ValueError('unexpected native cross method')
        groups[row['method']].append(row)
    expected_keys=expected_labels=None
    for method,values in groups.items():
        if not values:raise ValueError('missing native cross method')
        values.sort(key=lambda r:(r['uid'],r['query_timestamp'],r['item_idx']))
        keys=[(r['uid'],r['query_timestamp'],r['item_idx']) for r in values]
        labels=[r['label'] for r in values]
        if len(set(keys))!=len(keys):raise ValueError('duplicate or conflicting observed event key')
        if expected_keys is None:expected_keys,expected_labels=keys,labels
        elif keys!=expected_keys or labels!=expected_labels:
            raise ValueError('native cross methods differ in observed event keys or labels')
        if not np.isin(labels,[0,1]).all() or not np.isfinite([r['score'] for r in values]).all():
            raise ValueError('binary labels and finite raw logits required')
    return groups


def contrasts(points,draws):
    result={}
    for name,coefficients in CONTRASTS.items():
        result[name]={}
        for metric in ['AUC','log_loss']:
            direction=-1 if metric=='log_loss' else 1
            values=[points[m][metric]['value'] for m in coefficients]
            point=(sum(direction*c*points[m][metric]['value'] for m,c in coefficients.items())
                   if all(x is not None for x in values) else np.nan)
            samples=sum(direction*c*draws[m][metric] for m,c in coefficients.items())
            result[name][metric]=interval(point,samples)
    return result


def summarize(groups,population):
    base=groups[BASE];uids=np.array([r['uid'] for r in base],dtype=np.uint64)
    labels=np.array([r['label'] for r in base]);index,weights=uid_weights(population,uids)
    teacher=np.array([r['score'] for r in groups[TEACHER]])
    points={};draws={}
    for method,values in groups.items():
        score=np.array([r['score'] for r in values])
        points[method],draws[method]=binary_point_draws(labels,score,index,weights)
        points[method]['teacher_logit_mse']=float(np.mean((score-teacher)**2))
    effects=contrasts(points,draws)
    for method in METHODS:
        for metric in ['AUC','log_loss']:
            a,b=points[method][metric]['value'],points[BASE][metric]['value']
            sign=-1 if metric=='log_loss' else 1
            delta=sign*(a-b) if a is not None and b is not None else np.nan
            points[method][metric+'_gain_vs_oldKoldV']=interval(delta,sign*(draws[method][metric]-draws[BASE][metric]))
    users=len(set(map(int,uids)));positive=int(labels.sum());negative=len(labels)-positive
    return dict(resident_users=len(population),feedback_users=users,events=len(labels),
        positive_events=positive,negative_events=negative,positive_fraction=float(labels.mean()),
        majority_accuracy_reference=max(positive,negative)/len(labels),
        mechanism_use_permitted=users>=512 and positive>0 and negative>0,
        methods=points,effects=effects)


def select(groups,mask):
    return {m:[r for r,keep in zip(values,mask,strict=True) if keep] for m,values in groups.items()}


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def format_number(value):
    return 'NA' if value is None else f'{value:.6f}'


def write_outputs(result,output):
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    methods=[];effects=[]
    for group,point in result['groups'].items():
        support=dict(group=group,feedback_users=point['feedback_users'],events=point['events'],
            mechanism_use_permitted=point['mechanism_use_permitted'],majority_accuracy=point['majority_accuracy_reference'])
        for method,metrics in point['methods'].items():
            row=dict(support,method=method)
            for name,value in metrics.items():
                if isinstance(value,dict):
                    row[name]=value['value'];row[name+'_ci_low'],row[name+'_ci_high']=value['ci95'] or [None,None]
                    row[name+'_invalid_bootstrap_draws']=value['invalid_bootstrap_draws']
                else:row[name]=value
            methods.append(row)
        for contrast,metrics in point['effects'].items():
            for metric,value in metrics.items():
                lo,hi=value['ci95'] or [None,None]
                effects.append(dict(support,contrast=contrast,metric=metric,gain=value['value'],
                    ci_low=lo,ci_high=hi,invalid_bootstrap_draws=value['invalid_bootstrap_draws']))
    write_csv(output/'metrics.csv',methods);write_csv(output/'factorial_effects.csv',effects)
    lines=[f'# Native K/V来源交叉：edge{result["edge"]}','','同一请求分别读取old K/V、teacher K配old V、old K配teacher V、teacher K/V。交叉只用于read，不回写；不是新训练方法。','',
        'AUC为pooled observed-feedback AUC；log loss为稳定BCE，score是原始logit。准确率固定p≥0.5。500次同UID配对bootstrap；不足512反馈UID仅诊断。','',
        '| 分组 | 反馈UID | 方法 | AUC | log loss | accuracy | negative recall |',
        '|---|---:|---|---:|---:|---:|---:|']
    for group,point in result['groups'].items():
        for method,x in point['methods'].items():
            values=[format_number(x[k]['value']) for k in ['AUC','log_loss','accuracy_at_05','negative_recall']]
            lines.append(f'| {group} | {point["feedback_users"]} | {method} | '+' | '.join(values)+' |')
    lines += ['','四格效应统一为正值表示质量改善：AUC增加，log loss降低。interaction=joint−key-only−value-only；它表示该指标上的非加性，不等同模型内部独立性或模块必要性。','',
        '| 整体效应 | AUC gain [95% CI] | log loss reduction [95% CI] |','|---|---|---|']
    for name,x in result['groups']['all']['effects'].items():
        values=[]
        for metric in ['AUC','log_loss']:
            v=x[metric];band='NA' if v['ci95'] is None else ', '.join(format_number(z) for z in v['ci95'])
            values.append(format_number(v['value'])+' ['+band+']')
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['','- '+'\n- '.join(result['limitations'])]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())


def run(folder,population_path,output):
    config=json.loads((folder/'configuration.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    population=json.loads(population_path.read_text())
    if config['format']!='native_cross_quality_v1' or tuple(config['method_names'])!=METHODS:
        raise ValueError('requires the frozen four-cell native K/V source protocol')
    if summary['status']!='complete' or config['diagnostic'] or not summary['quality_use_permitted']:
        raise ValueError('requires completed non-diagnostic native cross quality')
    if sha(population_path)!=config['population_sha256'] or config['users']!=population['users']:
        raise ValueError('native cross population does not match frozen4000 cohort')
    if len(config['users'])!=4000 or len(set(config['users']))!=4000:
        raise ValueError('requires fixed4000 original residents')
    if any(sha(Path(path))!=digest for path,digest in config['sources'].items()):
        raise ValueError('native cross numerical source changed')
    frozen=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    rows=[]
    for artifact in summary['artifacts']:
        block=folder/artifact['path'];manifest=folder/artifact['manifest']
        if sha(block)!=artifact['sha256'] or sha(manifest)!=artifact['manifest_sha256']:
            raise ValueError('committed native cross block or manifest changed')
        record=json.loads(manifest.read_text())
        if record['configuration_sha256']!=frozen or record['score_sha256']!=sha(block):
            raise ValueError('native cross block belongs to a different frozen configuration')
        values=pq.read_table(block).to_pylist()
        if len(values)!=record['score_rows']:raise ValueError('native cross block row count changed')
        rows.extend(values)
    if len(rows)!=summary['score_rows']:raise ValueError('native cross total row count changed')
    groups=align(rows);base=groups[BASE]
    if set(r['uid'] for r in base)!=set(config['feedback_users']):
        raise ValueError('native cross scored a different feedback UID set')
    if len(set(r['uid'] for r in base))<512:
        raise ValueError('native cross quality requires at least512 feedback UIDs')
    if len(set(r['uid'] for r in base))!=summary['feedback_users']:
        raise ValueError('native cross feedback user count changed')
    uids=np.array([r['uid'] for r in base],dtype=np.uint64)
    elapsed=(np.array([r['query_timestamp'] for r in base])-config['settings']['release_day']*86400)/86400
    if ((elapsed<0)|(elapsed>=14)).any():raise ValueError('native cross event outside14day window')
    fixed=population['users'];metadata=population['edges'][str(config['edge'])]['user_metadata']
    masks={'all':np.ones(len(base),dtype=bool)};residents={'all':fixed}
    definitions={'all':'Frozen main+sensA4000, full observed-feedback window; not independent confirmation.'}
    for start,end in [(0,3),(3,7),(7,14)]:
        name=f'days_{start}_{end}';masks[name]=(elapsed>=start)&(elapsed<end);residents[name]=fixed
        definitions[name]=f'Elapsed days[{start},{end}), slicing continuous native trajectories with no reset.'
    for name,predicate,definition in [
        ('pre_gap_lt1d',lambda x:x['pre_gap_seconds']<86400,'Pre-release gap<1day'),
        ('pre_gap_ge1d',lambda x:x['pre_gap_seconds']>=86400,'Pre-release gap>=1day'),
        ('activity_lt256',lambda x:x['prerelease14activity']<256,'Pre-release14day events<256'),
        ('activity_ge256',lambda x:x['prerelease14activity']>=256,'Pre-release14day events>=256')]:
        chosen=[u for u in fixed if predicate(metadata[str(u)])]
        masks[name]=np.isin(uids,chosen);residents[name]=chosen;definitions[name]=definition
    points={}
    for name,mask in masks.items():
        if mask.any():points[name]=summarize(select(groups,mask),residents[name])
        else:points[name]=dict(resident_users=len(residents[name]),feedback_users=0,events=0,methods={},effects={},
            mechanism_use_permitted=False,majority_accuracy_reference=None,skipped_reason='No observed feedback')
        print(json.dumps(dict(status='analyzed_group',edge=config['edge'],group=name,
            feedback_users=points[name]['feedback_users'],mechanism_use_permitted=points[name]['mechanism_use_permitted'])),flush=True)
    metric_source=Path(sys.modules[binary_point_draws.__module__].__file__)
    result=dict(edge=config['edge'],scope='Four request-local native K/V source combinations; oracle read counterfactuals, no cross-cache writes.',
        methods=list(METHODS),groups=points,group_definitions=definitions,
        contrast_coefficients=CONTRASTS,contrast_direction='Positive denotes larger AUC or smaller log loss.',
        provenance=dict(source=str(folder.resolve()),configuration_sha256=sha(folder/'configuration.json'),
            source_summary_sha256=sha(folder/'summary.json'),population_sha256=sha(population_path),
            analysis_sha256=sha(__file__),shared_metric_source=str(metric_source),shared_metric_source_sha256=sha(metric_source)),
        limitations=['All effects and confidence intervals are exploratory pointwise comparisons, not multiplicity-adjusted confirmation.',
            'These are native source counterfactuals using a teacher trajectory, distinct from learned F key/value ablations.',
            'Mixed teacher/source K/V pairs are read only; they do not form native persistent trajectories or deployable corrections.',
            'K/V changes can affect later-layer queries; channel interventions do not uniquely isolate one attention-stage computation.',
            'The four cache combinations can be off the native joint K/V manifold; no module necessity claim follows.',
            'AUC is pooled over observed events, not macro-user AUC; label imbalance makes threshold accuracy insufficient.',
            'Fewer than512 feedback UID groups or undefined AUC are diagnostic only.',
            'The same UID multiplicities are shared by all four cells; bootstrap conditions on frozen native models.',
            'AUC factorial interaction is non-additivity of a pooled ranking metric; log-loss interaction is a paired mean-loss contrast.',
            'Temporal strata retain the same uninterrupted native replay; prerelease strata use no future labels.',
            'No correction is fitted, no affine/score-oracle controls are generated, and no cost superiority is claimed.'])
    write_outputs(result,output)
    (output/'shared_metrics_source.py').write_bytes(metric_source.read_bytes())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--population',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.input,args.population,args.output)
    print(json.dumps(dict(status='complete',edge=result['edge'],feedback_users=result['groups']['all']['feedback_users'])))
