#!/usr/bin/env python3
"""Real observed-feedback metrics and deliberately simple correction controls.

Scores are BCE logits. All metrics use true observed labels; teacher agreement
is a separate diagnostic. Score interpolation is an oracle readout control,
never a KV mapping. Affine controls fit teacher targets on prerelease fit users.
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

ROOT = Path(__file__).resolve().parents[3]
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.evaluation.binary_metrics import stable_log_loss
import numpy as np
import pyarrow.parquet as pq

MAPPING_FIELDS = ('kv_mse', 'k_mse', 'v_mse', 'initial_kv_mse')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(point, draws):
    a = np.asarray(draws, dtype=float); valid = np.isfinite(a)
    return dict(value=float(point) if np.isfinite(point) else None,
        ci95=np.percentile(a[valid], [2.5,97.5]).tolist() if valid.any() else None,
        invalid_bootstrap_draws=int((~valid).sum()))


def point_difference(a,b):
    return a-b if a is not None and b is not None else np.nan


def uid_weights(population, uids, *, replicates=500, seed=17):
    population = np.asarray(sorted(population), dtype=np.uint64)
    if len(np.unique(population)) != len(population):
        raise ValueError('duplicate resident UID')
    idx = np.searchsorted(population, np.asarray(uids,dtype=np.uint64))
    if (idx >= len(population)).any() or not np.array_equal(population[idx], uids):
        raise ValueError('scored UID outside fixed population')
    rng = np.random.default_rng(seed)
    w = np.asarray([np.bincount(rng.integers(len(population),size=len(population)),
                     minlength=len(population)) for _ in range(replicates)], dtype=np.uint16)
    return idx, w


def binary_point_draws(labels, logits, uid_index, weights):
    """Event-weighted estimands, resampled with shared UID multiplicities."""
    y = np.asarray(labels,dtype=np.int8); z = np.asarray(logits,dtype=np.float64)
    if not len(y) or y.shape != z.shape or not np.isin(y,[0,1]).all() or not np.isfinite(z).all():
        raise ValueError('requires finite aligned logits and binary observed labels')
    residents = weights.shape[1]
    counts = np.bincount(uid_index,minlength=residents)
    positives = np.bincount(uid_index,weights=y,minlength=residents)
    negatives = counts-positives
    predicted = z >= 0.
    loss = stable_log_loss(z,y)
    def mean(a):
        sums=np.bincount(uid_index,weights=a,minlength=residents)
        with np.errstate(divide='ignore',invalid='ignore'):
            return float(np.mean(a)), (weights@sums)/(weights@counts)
    ll,ll_draws=mean(loss)
    acc,acc_draws=mean(predicted==y)
    tp=np.bincount(uid_index,weights=predicted*y,minlength=residents)
    tn=np.bincount(uid_index,weights=(~predicted)*(1-y),minlength=residents)
    with np.errstate(divide='ignore',invalid='ignore'):
        tpr=(weights@tp)/(weights@positives)
        tnr=(weights@tn)/(weights@negatives)
        point_tpr=float(tp.sum()/positives.sum()) if positives.sum() else np.nan
        point_tnr=float(tn.sum()/negatives.sum()) if negatives.sum() else np.nan
        user_loss=np.divide(np.bincount(uid_index,weights=loss,minlength=residents),counts,
                            out=np.zeros(residents,dtype=float),where=counts>0)
        ul_draws=(weights@user_loss)/(weights@(counts>0))
    auc=weighted_auc(z,y.astype(float),uid_index,np.ones((1,residents),dtype=np.uint16))[0]
    auc_draws=weighted_auc(z,y.astype(float),uid_index,weights)
    return dict(AUC=interval(auc,auc_draws),log_loss=interval(ll,ll_draws),
        accuracy_at_05=interval(acc,acc_draws),negative_recall=interval(point_tnr,tnr),
        balanced_accuracy_at_05=interval((point_tpr+point_tnr)/2,(tpr+tnr)/2),
        user_mean_log_loss=interval(user_loss[counts>0].mean(),ul_draws)), dict(
            AUC=auc_draws,log_loss=ll_draws,accuracy_at_05=acc_draws,
            negative_recall=tnr,balanced_accuracy_at_05=(tpr+tnr)/2,user_mean_log_loss=ul_draws)


def fit_affine(x,y):
    x,y=np.asarray(x,dtype=float).reshape(-1),np.asarray(y,dtype=float).reshape(-1)
    variance=np.mean((x-x.mean())**2)
    slope=max(float(np.mean((x-x.mean())*(y-y.mean()))/variance),1e-8) if variance>0 else 1.
    return dict(slope=slope,bias=float(y.mean()-slope*x.mean())),dict(slope=1.,bias=float(np.mean(y-x)))


def affine_controls(edge):
    folder=ROOT/f'experiments/ch3_motivation/runs/insight_extensions/attempt_1/candidate_edge{edge}'
    config=json.loads((folder/'configuration.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    data={}
    for role in ['fit','validation']:
        actual=[];wanted=[];uids=[]
        for artifact in summary['artifacts']:
            p=folder/artifact['path']
            if p.parent.name!=role:continue
            if sha(p)!=artifact['sha256']:raise ValueError('affine calibration block changed')
            with np.load(p) as b:
                actual.append(b['f_scores'][:,0].astype(float));wanted.append(b['teacher_scores'][:,0].astype(float))
                uids.extend(map(int,b['uid']))
        if uids!=config['roles'][role]:raise ValueError('affine calibration UID order changed')
        data[role]=(np.concatenate(actual),np.concatenate(wanted))
    if len(config['roles']['fit'])!=100 or len(config['roles']['validation'])!=16:
        raise ValueError('requires reserved fit100/validation16 affine calibration')
    a,b=fit_affine(*data['fit'])
    records={}
    for name,parameters in [('affine_teacher_fit',a),('bias_teacher_fit',b)]:
        metrics={}
        for role,(x,y) in data.items():
            metrics[role+'_teacher_logit_mse']=float(np.mean((parameters['slope']*x+parameters['bias']-y)**2))
        records[name]=dict(parameters,**metrics)
    return records,dict(source=str(folder),configuration_sha256=sha(folder/'configuration.json'),
        checkpoint_sha256=config['checkpoint_sha256'],
        fit_uids=config['roles']['fit'],validation_uids=config['roles']['validation'],
        rule='Teacher-logit least squares on fit100 broad16 only; val16 records error but selects neither parameters nor method.')


def aligned_rows(rows):
    groups={}
    for row in rows:groups.setdefault(row['method'],[]).append(row)
    keys=None;output={}
    for method,values in groups.items():
        values.sort(key=lambda r:(r['uid'],r['query_timestamp'],r['item_idx']))
        actual=[(r['uid'],r['query_timestamp'],r['item_idx'],r['label']) for r in values]
        if len(set(actual))!=len(actual):raise ValueError('duplicate real observation within method')
        if keys is None:keys=actual
        elif actual!=keys:raise ValueError('methods do not share identical observed feedback')
        output[method]=values
    if not {'reuse','rebuild','f1'}<=set(output):raise ValueError('requires matched reuse/rebuild/f1 scores')
    return output


def add_score_controls(groups,controls):
    base,target=groups['f1'],groups['rebuild']
    f=np.array([r['score'] for r in base]);t=np.array([r['score'] for r in target])
    new={f'score_oracle{code}':(1-beta)*f+beta*t for code,beta in [('025',.25),('05',.5),('075',.75)]}
    new.update({name:fit['slope']*f+fit['bias'] for name,fit in controls.items()})
    for name,scores in new.items():
        # A score oracle has no single associated KV cache; affine keeps F's.
        groups[name]=[dict(row,method=name,score=float(score),
            **({key:None for key in MAPPING_FIELDS} if name.startswith('score_oracle') else {}))
            for row,score in zip(base,scores,strict=True)]
    return groups


def summarize(groups,population,*,seed=17):
    base=groups['reuse'];uids=np.array([r['uid'] for r in base],dtype=np.uint64)
    labels=np.array([r['label'] for r in base]);index,weights=uid_weights(population,uids,seed=seed)
    target=np.array([r['score'] for r in groups['rebuild']]);reuse=np.array([r['score'] for r in groups['reuse']]);points={};draws={}
    counts=np.bincount(index,minlength=len(population));denominator=weights@counts
    for method,rows in groups.items():
        scores=np.array([r['score'] for r in rows]);p,d=binary_point_draws(labels,scores,index,weights)
        p['teacher_logit_mse']=float(np.mean((scores-target)**2))
        delta=scores-reuse
        for metric,values in [('score_minus_reuse_mean',delta),('score_minus_reuse_mean_abs',np.abs(delta)),
                              ('score_minus_reuse_rms',delta**2)]:
            sums=np.bincount(index,weights=values,minlength=len(population))
            with np.errstate(divide='ignore',invalid='ignore'):sample=(weights@sums)/denominator
            point=values.mean()
            if metric.endswith('_rms'):sample=np.sqrt(sample);point=np.sqrt(point)
            p[metric]=interval(point,sample);d[metric]=sample
        for field in MAPPING_FIELDS:
            a=np.array([np.nan if r.get(field) is None else r[field] for r in rows],dtype=float)
            p['event_mean_'+field]=float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else None
        points[method]=p;draws[method]=d
    for method,p in points.items():
        for anchor in ['reuse','f1','rebuild']:
            for metric in ['AUC','log_loss']:
                p[metric+'_minus_'+anchor]=interval(point_difference(p[metric]['value'],points[anchor][metric]['value']),draws[method][metric]-draws[anchor][metric])
    for name in ['affine_teacher_fit','bias_teacher_fit']:
        if (name in points and points[name]['AUC']['value'] is not None and
                not np.isclose(points[name]['AUC']['value'],points['f1']['AUC']['value'],atol=1e-12,rtol=0)):
            raise ValueError('positive affine control unexpectedly changes AUC')
    positive=int(labels.sum());feedback_users=len(np.unique(uids))
    return dict(resident_users=len(population),feedback_users=feedback_users,events=len(labels),
        positive_events=positive,negative_events=len(labels)-positive,
        positive_fraction=float(labels.mean()),majority_accuracy_reference=float(max(positive,len(labels)-positive)/len(labels)),
        mechanism_use_permitted=feedback_users>=512 and 0<positive<len(labels),methods=points),draws


def event_gain_harm(groups,population,*,seed=17):
    """Benefit is anchor loss minus method loss, with entire UID clusters paired."""
    base=groups['reuse'];uids=np.array([r['uid'] for r in base],dtype=np.uint64)
    labels=np.array([r['label'] for r in base])
    index,weights=uid_weights(population,uids,seed=seed)
    counts=np.bincount(index,minlength=len(population));denominator=weights@counts
    loss={m:stable_log_loss(np.array([r['score'] for r in rows]),labels) for m,rows in groups.items()}
    output={}
    for anchor in ['reuse','f1']:
        output[anchor]={}
        for method,method_loss in loss.items():
            gain=loss[anchor]-method_loss
            quantities=dict(improved_fraction=(gain>0).astype(float),harmed_fraction=(gain<0).astype(float),
                unchanged_fraction=(gain==0).astype(float),mean_log_loss_gain=gain,
                benefit_log_loss_mass=np.maximum(gain,0),harm_log_loss_mass=np.maximum(-gain,0))
            metrics={}
            for name,values in quantities.items():
                total=np.bincount(index,weights=values,minlength=len(population))
                draws=weights@total
                point=values.sum() if name.endswith('_mass') else values.mean()
                if not name.endswith('_mass'):
                    with np.errstate(divide='ignore',invalid='ignore'):draws=draws/denominator
                metrics[name]=interval(point,draws)
            output[anchor][method]=metrics
    return output


def select(groups,mask):
    return {m:[r for r,keep in zip(rows,mask,strict=True) if keep] for m,rows in groups.items()}


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def flatten_interval(row,key,value):
    row[key]=value['value'];row[key+'_ci_low'],row[key+'_ci_high']=value['ci95'] or [None,None]


def fmt(value):
    return 'NA' if value is None else f'{value:.6f}'


def write_outputs(result,output):
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    rows=[]
    for group,point in result['groups'].items():
        for method,metrics in point['methods'].items():
            row=dict(group=group,method=method,feedback_users=point['feedback_users'],events=point['events'],
                mechanism_use_permitted=point['mechanism_use_permitted'],majority_accuracy=point['majority_accuracy_reference'])
            for key,value in metrics.items():
                if isinstance(value,dict):
                    flatten_interval(row,key,value)
                else:row[key]=value
            rows.append(row)
    write_csv(output/'metrics.csv',rows)
    write_csv(output/'mapping_vs_quality.csv',rows)
    harm=[]
    for group,anchors in result.get('event_gain_harm',{}).items():
        support=result['groups'][group]
        for anchor,methods in anchors.items():
            for method,metrics in methods.items():
                row=dict(group=group,anchor=anchor,method=method,feedback_users=support['feedback_users'],
                    mechanism_use_permitted=support['mechanism_use_permitted'])
                for metric,value in metrics.items():flatten_interval(row,metric,value)
                harm.append(row)
    write_csv(output/'event_gain_harm.csv',harm)
    write_csv(output/'channel_comparison.csv',[row for row in rows if row['method'] in ['reuse','f_key','f_value','f1','rebuild']])
    alpha={'f025':.25,'f05':.5,'f1':1.,'f15':1.5,'f2':2.}
    write_csv(output/'alpha_by_group.csv',[dict(row,alpha=alpha[row['method']]) for row in rows if row['method'] in alpha])
    winners=[]
    for group,point in result['groups'].items():
        candidates={m:x for m,x in point['methods'].items() if m in alpha}
        if not candidates:continue
        row=dict(group=group,feedback_users=point['feedback_users'],mechanism_use_permitted=point['mechanism_use_permitted'],candidate_alpha_count=len(candidates))
        for metric,maximize in [('AUC',True),('log_loss',False)]:
            valid={m:x[metric]['value'] for m,x in candidates.items() if x[metric]['value'] is not None}
            if not valid:continue
            best=max(valid.values()) if maximize else min(valid.values())
            row[metric+'_best_observed_alpha']=','.join(str(alpha[m]) for m,v in valid.items() if v==best)
            row[metric+'_best_observed_value']=best
        winners.append(row)
    write_csv(output/'alpha_exploratory_winners.csv',winners)
    for pair_name in ['paired_time','paired_fifo']:
        paired=[]
        for method,metrics in result.get(pair_name,{}).get('methods',{}).items():
            row=dict(method=method,feedback_users=result[pair_name]['feedback_users'])
            for key,value in metrics.items():flatten_interval(row,key,value)
            paired.append(row)
        write_csv(output/(pair_name+'.csv'),paired)
    lines=[f"# 真实反馈质量：edge{result['edge']}",'',result['scope'],'',
        'AUC用原始logit；log loss为稳定BCE。准确率阈值固定p≥0.5，附多数类比例、负例召回与balanced accuracy。区间为500次同UID配对bootstrap。', '',
        '| 分组 | 有反馈UID | 方法 | AUC | log loss | accuracy | negative recall | teacher logit MSE |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for group,point in result['groups'].items():
        for method,m in point['methods'].items():
            lines.append(f"| {group} | {point['feedback_users']} | {method} | {fmt(m['AUC']['value'])} | {fmt(m['log_loss']['value'])} | {fmt(m['accuracy_at_05']['value'])} | {fmt(m['negative_recall']['value'])} | {fmt(m['teacher_logit_mse'])} |")
    lines += ['', 'score_oracle在最终logit之间插值，使用主数据teacher分数，是诊断oracle而非KV映射。正affine及bias-only只用预发布fit100 teacher targets拟合；val16仅记录；两者应保持AUC不变。', '',
        'event_gain_harm.csv以anchor loss减method loss衡量逐请求收益；正、负总量均保留。channel_comparison.csv仅比较相同请求上的K/V干预；不能单独证明模块必要性。alpha_exploratory_winners.csv是固定分组内的观察值，不能作为独立确认或拟合后的无偏选择。', '',
        *['- '+x for x in result['limitations']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def pilot(edge,output):
    folder=ROOT/f'experiments/ch3_motivation/runs/insight_mechanisms/attempt_1/main_edge{edge}'
    config=json.loads((folder/'configuration.json').read_text())
    rows=[]
    for p in sorted((folder/'feedback').glob('*.parquet')):
        for row in pq.read_table(p).to_pylist():
            row['method']='f1' if row['method']=='f' else row['method'];rows.append(row)
    groups=aligned_rows(rows);controls,calibration=affine_controls(edge)
    groups=add_score_controls(groups,controls);point,_=summarize(groups,config['users'])
    result=dict(edge=edge,scope='CPU endpoint pilot: first/last/single-time observed feedback only, NOT all requests or a fresh GPU rerun.',
        groups={'all':point},score_controls=controls,calibration=calibration,
        provenance=dict(source=str(folder),configuration_sha256=sha(folder/'configuration.json'),analysis_sha256=sha(__file__)),
        limitations=['Endpoint pilot reuses saved real observations; not full-window quality.',
            'Event-weighted pooled AUC, not macro-user AUC; fewer than512 feedback UID groups are diagnostic only.',
            'Teacher agreement and real-label quality are distinct; no monotonicity is presumed.',
            'Accuracy can equal the ~90% majority rate if all logits predict positive.',
            'No held-out labels fit affine parameters, threshold or score oracle coefficients.',
            'UID intervals condition on fixed backbones and fitted corrections, not training randomness.'])
    write_outputs(result,output)
    return result


def paired_change(points,draws,first,last):
    output={}
    for method in points[first]['methods']:
        output[method]={}
        for metric in ['AUC','log_loss','score_minus_reuse_mean','score_minus_reuse_mean_abs','score_minus_reuse_rms']:
            a,b=points[first]['methods'][method][metric]['value'],points[last]['methods'][method][metric]['value']
            change=point_difference(b,a);samples=draws[last][method][metric]-draws[first][method][metric]
            output[method][metric+'_after_minus_before']=interval(change,samples)
            if metric not in ['AUC','log_loss']:continue
            anchor=point_difference(points[last]['methods']['reuse'][metric]['value'],points[first]['methods']['reuse'][metric]['value'])
            anchor_samples=draws[last]['reuse'][metric]-draws[first]['reuse'][metric]
            output[method][metric+'_change_in_effect_vs_reuse']=interval(change-anchor,samples-anchor_samples)
    return output


def full_requests(folder,population_path,output):
    """Analyze the single current long-table schema, after all GPU blocks commit."""
    config=json.loads((folder/'configuration.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    population=json.loads(population_path.read_text())
    if summary['status']!='complete' or config['diagnostic'] or not summary['quality_use_permitted']:
        raise ValueError('full analysis needs a completed non-diagnostic run')
    if sha(population_path)!=config['population_sha256'] or config['users']!=population['users']:
        raise ValueError('fixed4000 population changed or run used another cohort')
    if len(set(config['users']))!=4000 or len(config['users'])!=4000:
        raise ValueError('full analysis requires the frozen main+sensA4000 residents')
    rows=[]
    frozen=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    for artifact in summary['artifacts']:
        block=folder/artifact['path'];manifest=folder/artifact['manifest']
        if sha(block)!=artifact['sha256'] or sha(manifest)!=artifact['manifest_sha256']:
            raise ValueError('committed score block or manifest changed')
        record=json.loads(manifest.read_text())
        if record['configuration_sha256']!=frozen or record['score_sha256']!=sha(block):
            raise ValueError('score manifest belongs to a different frozen configuration')
        values=pq.read_table(block).to_pylist()
        if len(values)!=record['score_rows']:raise ValueError('block row count changed')
        rows.extend(values)
    if len(rows)!=summary['score_rows']:raise ValueError('score row count changed')
    groups=aligned_rows(rows);controls,calibration=affine_controls(config['edge'])
    if calibration['checkpoint_sha256']!=config['checkpoint_sha256']:
        raise ValueError('affine controls fit a different F checkpoint')
    if set(r['uid'] for r in groups['reuse'])!=set(config['feedback_users']):
        raise ValueError('saved scores omit or add configured feedback users')
    groups=add_score_controls(groups,controls)
    base=groups['reuse'];uids=np.array([r['uid'] for r in base],dtype=np.uint64)
    elapsed=(np.array([r['query_timestamp'] for r in base])-config['settings']['release_day']*86400)/86400
    if ((elapsed<0)|(elapsed>=14)).any():raise ValueError('request outside fixed14day window')
    fixed=population['users'];meta=population['edges'][str(config['edge'])]['user_metadata']
    masks={'all':np.ones(len(base),dtype=bool)};residents={'all':fixed}
    definitions={'all':'All frozen residents, known-item observed feedback throughout14days.'}
    for a,b in [(0,3),(3,7),(7,14)]:
        name=f'days_{a}_{b}';masks[name]=(elapsed>=a)&(elapsed<b);residents[name]=fixed
        definitions[name]=f'Predetermined elapsed days[{a},{b}); slice continuous replay without resetting cache.'
    specifications=[('pre_gap_lt1d',lambda m:m['pre_gap_seconds']<86400,'Pre-release gap<1day'),
        ('pre_gap_ge1d',lambda m:m['pre_gap_seconds']>=86400,'Pre-release gap>=1day'),
        ('activity_lt256',lambda m:m['prerelease14activity']<256,'Pre-release14day behavior events<256'),
        ('activity_ge256',lambda m:m['prerelease14activity']>=256,'Pre-release14day behavior events>=256')]
    for name,predicate,definition in specifications:
        chosen=[u for u in fixed if predicate(meta[str(u)])]
        residents[name]=chosen;masks[name]=np.isin(uids,chosen);definitions[name]=definition
    history_cap=config['settings']['history_length']
    after_fifo=np.array([r['appended_events'] for r in base])>=history_cap
    for name,mask in [('old_rows_present',~after_fifo),('old_rows_evicted',after_fifo)]:
        masks[name]=mask;residents[name]=fixed
        definitions[name]='appended_events '+('>=' if name.endswith('evicted') else '<')+str(history_cap)+'. FIFO removes cutover physical rows; inherited causal influence can continue through new rows.'
    fifo_common=sorted(set(map(int,uids[~after_fifo]))&set(map(int,uids[after_fifo])))
    fifo=dict(feedback_users=len(fifo_common),mechanism_use_permitted=len(fifo_common)>=512,methods={},
        rule='Strict same UID intersection before/after appended_events>=history_cap; old physical rows absent, causal influence may persist.')
    if len(fifo_common)>=512:
        for name,mask in [('paired_fifo_before',~after_fifo),('paired_fifo_after',after_fifo)]:
            masks[name]=mask&np.isin(uids,fifo_common);residents[name]=fifo_common;definitions[name]=fifo['rule']
    else:fifo['skipped_reason']='Fewer than512 feedback UIDs with requests both before and after all cutover rows are physically evicted.'
    early_uids=set(map(int,uids[elapsed<7]));late_uids=set(map(int,uids[elapsed>=7]))
    common=sorted(early_uids&late_uids)
    paired=dict(feedback_users=len(common),mechanism_use_permitted=len(common)>=512,
        rule='Strict same UID intersection with observed feedback in both[0,7) and[7,14); fixed weights, no reset.',methods={})
    if len(common)>=512:
        for name,mask in [('paired_early7',elapsed<7),('paired_late7',elapsed>=7)]:
            masks[name]=mask&np.isin(uids,common);residents[name]=common;definitions[name]=paired['rule']
    else:paired['skipped_reason']='Fewer than512 feedback UIDs in strict early/late intersection.'
    points={};draws={};harms={}
    for name,mask in masks.items():
        selected=select(groups,mask)
        if not mask.any():
            points[name]=dict(resident_users=len(residents[name]),feedback_users=0,events=0,methods={},
                mechanism_use_permitted=False,majority_accuracy_reference=None,skipped_reason='No observed feedback')
            continue
        points[name],draws[name]=summarize(selected,residents[name])
        harms[name]=event_gain_harm(selected,residents[name])
        print(json.dumps(dict(status='analyzed_group',edge=config['edge'],group=name,
            feedback_users=points[name]['feedback_users'],mechanism_use_permitted=points[name]['mechanism_use_permitted'])),flush=True)
    if len(common)>=512:
        early,late=points['paired_early7'],points['paired_late7']
        paired['mechanism_use_permitted']=early['mechanism_use_permitted'] and late['mechanism_use_permitted']
        paired['methods']=paired_change(points,draws,'paired_early7','paired_late7')
    if len(fifo_common)>=512:
        fifo['mechanism_use_permitted']=points['paired_fifo_before']['mechanism_use_permitted'] and points['paired_fifo_after']['mechanism_use_permitted']
        fifo['methods']=paired_change(points,draws,'paired_fifo_before','paired_fifo_after')
    result=dict(edge=config['edge'],scope=f'Fresh all-request observed-feedback replay; history cap{config["settings"]["history_length"]}; main+sensA4000 exploratory population, not confirmation.',
        groups=points,group_definitions=definitions,event_gain_harm=harms,paired_time=paired,paired_fifo=fifo,
        score_controls=controls,calibration=calibration,cost_reference=summary['cost_reference'],
        provenance=dict(source=str(folder.resolve()),configuration_sha256=sha(folder/'configuration.json'),
            source_summary_sha256=sha(folder/'summary.json'),population_sha256=sha(population_path),analysis_sha256=sha(__file__)),
        limitations=['AUC is event-weighted pooled AUC; no macro-user AUC (few users have both labels).',
            'Fewer than512 feedback UIDs or undefined AUC: diagnostic only, never mechanism evidence.',
            '500 fixed-seed UID-cluster bootstrap draws pair every method; intervals condition on frozen weights and backbones.',
            'Population is fixed main+sensA4000; this exploration is not independent confirmation and does not alter roles.',
            'Accuracy threshold is fixedp=0.5; always inspect majority baseline, negative recall and balanced accuracy.',
            'Teacher KV/logit MSE and actual label prediction are separate outcomes; monotonicity is not assumed.',
            'KV oracle blends request-local states using teacher; score oracle blends final logits; neither is deployable correction.',
            'Positive affine/bias controls fit broad16 teacher targets on100 reserved fit users, never evaluation labels.',
            'Best observed alpha by subgroup is descriptive exploration with no postselection confidence claim.',
            'K-only/V-only interventions describe asymmetry; interactions and persistent native writes prevent a necessity claim.',
            'All slices share continuous replay; pre-release strata are fixed, temporal slices can differ in event mix.',
            'Shift/linear raw-KV teacher objectives differ from F read-target fitting; comparison is not a matched objective ablation.',
            'Preparation cost reference excludes common native reads/appends and research-only oracle work; not total deployment cost.'])
    write_outputs(result,output)
    # Keep the exact analysis implementation associated with its archived digest.
    (output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-edge',type=int,choices=[2,3])
    p.add_argument('--input',type=Path)
    p.add_argument('--population',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.pilot_edge is not None:a=pilot(args.pilot_edge,args.output)
    elif args.input is not None and args.population is not None:
        a=full_requests(args.input,args.population,args.output)
    else:p.error('Provide --pilot-edge, or --input and --population for a completed current-schema GPU run')
    print(json.dumps(dict(status='complete',edge=a['edge'],feedback_users=a['groups']['all']['feedback_users'])))
