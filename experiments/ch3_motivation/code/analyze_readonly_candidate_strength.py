#!/usr/bin/env python3
"""Item loss curves on a shared causal write trajectory; no oracle policy AUC."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse,csv,json
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
import numpy as np
import pyarrow.parquet as pq
from experiments.ch3_motivation.code.analyze_real_quality import sha, uid_weights, stable_log_loss
from experiments.ch3_motivation.code.freeze_candidate_strength_support import context_half

GRID=np.arange(9,dtype=float)/4
EPS=1e-6
GROUPS=('near_zero','intermediate','near_one','above_one_strong','flat','ambiguous','invalid')
DEFAULT=ROOT/'experiments/ch3_motivation/runs/candidate_strength_direction/attempt_1'


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def group(value):
    if value<=.25:return 'near_zero'
    if value==.5:return 'intermediate'
    if .75<=value<=1.25:return 'near_one'
    return 'above_one_strong'


def preference(curve):
    curve=np.asarray(curve,dtype=float)
    if not np.isfinite(curve).all():return dict(valid=False,flat=False,representative=None,group='invalid')
    span=float(np.ptp(curve));indices=np.flatnonzero(curve<=curve.min()+EPS)
    minimizing=GRID[indices].tolist()
    if span<=EPS:return dict(valid=True,flat=True,loss_range=span,representative=None,group='flat',minimizing_lambdas=minimizing,
        minimum_set_span=[min(minimizing),max(minimizing)],tie_groups=sorted({group(v) for v in minimizing}),identified_preference=False)
    chosen=min(minimizing,key=lambda v:(abs(v-1),v));classes={group(v) for v in minimizing}
    return dict(valid=True,flat=False,loss_range=span,representative=chosen,group=next(iter(classes)) if len(classes)==1 else 'ambiguous',
        minimizing_lambdas=minimizing,minimum_set_span=[min(minimizing),max(minimizing)],tie_groups=sorted(classes),
        identified_preference=len(classes)==1,representative_group=group(chosen),strictly_above_one=chosen>1,
        all_near_minima_strictly_above_one=all(v>1 for v in minimizing),minima_cross_one=any(v<=1 for v in minimizing) and any(v>1 for v in minimizing))


def load(folder,support_root):
    plan_path=support_root/'plan.json';plan=json.loads(plan_path.read_text());population_path=support_root/'population.json'
    population=json.loads(population_path.read_text());support=json.loads((support_root/'support_coverage.json').read_text())
    config=json.loads((folder/'configuration.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    edge=int(config['edge'])
    if (plan['A']['lambdas']!=GRID.tolist() or plan['A']['flat_loss_range_max_nat']!=EPS or plan['A']['tie_min_loss_tolerance_nat']!=EPS
        or plan['A']['stability']['uid_bootstrap']!=200 or plan['A']['stability']['seed']!=17):raise ValueError('frozen statistical rules differ')
    if (config['diagnostic'] or summary['status']!='complete' or not summary['quality_use_permitted']
        or config['format']!='readonly_candidate_strength_v1'):raise ValueError('completed nondiagnostic read-only strength quality required')
    if (config['population_sha256']!=sha(population_path) or config['users']!=population['users']
        or support['population_sha256']!=sha(population_path) or plan['population']['sha256']!=sha(population_path)
        or config['plan_sha256']!=sha(plan_path)):raise ValueError('final population/plan binding differs')
    if plan['A']['support_plan_sha256']!=sha(Path(plan['A']['support_plan'])):raise ValueError('preflight support plan changed')
    methods=sorted(config['method_names'],key=lambda m:config['method_strengths'][m])
    if len(methods)!=9 or [config['method_strengths'][m] for m in methods]!=GRID.tolist():raise ValueError('nine frozen lambdas required')
    groups={m:[] for m in methods};frozen=digest(config);context_fields=('appended_events','history_rows','query_gap','inherited_rows','evicted_events')
    for a in summary['artifacts']:
        block=folder/a['path'];manifest=folder/a['manifest']
        if sha(block)!=a['sha256'] or sha(manifest)!=a['manifest_sha256']:raise ValueError('score artifact changed')
        record=json.loads(manifest.read_text())
        if record['configuration_sha256']!=frozen or record['score_sha256']!=sha(block):raise ValueError('block configuration differs')
        rows=pq.read_table(block,columns=['uid','query_timestamp','item_idx','label','method','lambda_value','score',*context_fields]).to_pylist()
        if len(rows)!=record['score_rows']:raise ValueError('block row count differs')
        for r in rows:
            if r['method'] not in groups or r['lambda_value']!=config['method_strengths'][r['method']]:raise ValueError('lambda label differs')
            groups[r['method']].append(r)
    if sum(map(len,groups.values()))!=summary['score_rows']:raise ValueError('total score row count differs')
    key=lambda r:(r['uid'],r['query_timestamp'],r['item_idx'])
    expected=None
    for values in groups.values():
        values.sort(key=key);actual=[(*key(r),r['label'],*[r[f] for f in context_fields]) for r in values]
        if len({key(r) for r in values})!=len(values):raise ValueError('duplicate observed item event')
        if expected is None:expected=actual
        elif actual!=expected:raise ValueError('lambdas do not share labels and one causal trajectory')
    base=groups[methods[0]];entry=population['edges'][str(edge)];request=Path(entry['inputs']['requests']['path'])
    if sha(request)!=entry['inputs']['requests']['sha256']:raise ValueError('canonical request source differs')
    residents=set(population['users']);observed=pq.read_table(request,columns=['uid','query_timestamp','item_idx','label']).to_pylist()
    wanted=sorted((*key(r),r['label']) for r in observed if r['uid'] in residents)
    if wanted!=[(*key(r),r['label']) for r in base]:raise ValueError('quality rows omit or alter canonical feedback')
    y=np.asarray([r['label'] for r in base],dtype=np.int8);uids=np.asarray([r['uid'] for r in base],dtype=np.uint64)
    if len(np.unique(uids))<512 or not np.isin(y,[0,1]).all():raise ValueError('insufficient real feedback support')
    scores=np.column_stack([[r['score'] for r in groups[m]] for m in methods]).astype(np.float64)
    if not np.isfinite(scores).all():raise ValueError('nonfinite logit')
    inherited=np.asarray([r['inherited_rows'] for r in base]);zero=inherited==0
    if not np.array_equal(scores[zero].view(np.uint64),np.repeat(scores[zero,:1],9,axis=1).view(np.uint64)):
        raise ValueError('lambda scores differ after all inherited rows disappear')
    loss=np.column_stack([stable_log_loss(scores[:,j],y) for j in range(9)])
    ref=support['edges'][str(edge)]['item_support_artifact'];support_table=pq.read_table(ref['path']).to_pylist()
    if sha(Path(ref['path']))!=ref['sha256']:raise ValueError('item support artifact differs')
    return plan,population,config,summary,support,support_table,base,loss,uids,y,inherited


def analyze_scope(mask,name,base,loss,uids,labels,inherited,uid_index,weights,frozen_items=None):
    items=np.asarray([r['item_idx'] for r in base],dtype=np.int64);halves=np.asarray([context_half(r['uid'],r['query_timestamp']) for r in base])
    candidates=sorted(set(map(int,items[mask])));records=[];curves=[];boot_arrays=[]
    for item in candidates:
        ix=np.flatnonzero(mask&(items==item));local_uid,local_index,counts=np.unique(uid_index[ix],return_inverse=True,return_counts=True)
        if len(ix)<20 or len(local_uid)<10:continue
        if frozen_items is not None and item not in frozen_items:raise ValueError('unexpected supported item outside frozen universe')
        sums=np.zeros((len(local_uid),9));np.add.at(sums,local_index,loss[ix]);w=weights[:,local_uid].astype(float);den=w@counts
        with np.errstate(divide='ignore',invalid='ignore'):samples=(w@sums)/den[:,None]
        point=preference(loss[ix].mean(0));decisions=[preference(v) for v in samples]
        valid=[v for v in decisions if v['valid']];states=[('flat' if v['flat'] else str(v['representative'])) for v in valid]
        frequencies=Counter(states);maximum=max(frequencies.values()) if frequencies else 0
        modes=[s for s,n in frequencies.items() if n==maximum]
        chosen=point['representative'];numeric=[v['representative'] for v in valid if not v['flat']]
        exact=sum(not v['flat'] and v['representative']==chosen for v in valid)/len(valid) if valid and chosen is not None else None
        nearby=sum(not v['flat'] and abs(v['representative']-chosen)<=.25 for v in valid)/len(valid) if valid and chosen is not None else None
        stable=len(valid)>=190 and nearby>=.70 if nearby is not None else None
        split={};half_pref=[]
        for half in (0,1):
            hi=ix[halves[ix]==half];support=len(hi)>=10 and len(np.unique(uid_index[hi]))>=5
            decision=preference(loss[hi].mean(0)) if len(hi) else None
            split[str(half)]=dict(events=len(hi),feedback_users=len(np.unique(uid_index[hi])),support=support,preference=decision)
            half_pref.append(decision)
        split_supported=all(v['support'] for v in split.values())
        split_available=split_supported and all(v is not None and not v['flat'] for v in half_pref)
        difference=abs(half_pref[0]['representative']-half_pref[1]['representative']) if split_available else None
        pos=int(labels[ix].sum());active=inherited[ix]>0
        record=dict(scope=name,item_idx=item,events=len(ix),feedback_users=len(local_uid),positive_events=pos,negative_events=len(ix)-pos,
            only_positive=pos==len(ix),only_negative=pos==0,both_labels_ge5=pos>=5 and len(ix)-pos>=5,
            inherited_active_events=int(active.sum()),inherited_active_users=len(np.unique(uids[ix][active])),zero_inherited_events=int((~active).sum()),
            preference=point,bootstrap=dict(draws=len(decisions),valid_draws=len(valid),invalid_zero_weight_draws=len(decisions)-len(valid),
                lambda_state_counts=dict(frequencies),mode_states=modes,mode_fraction=maximum/len(valid) if valid else None,
                point_exact_agreement=exact,point_within_one_grid_step_probability=nearby,stable_descriptive=stable,
                flat_draws=sum(v['flat'] for v in valid),ambiguous_draws=sum(v['group']=='ambiguous' for v in valid),
                nonflat_lambda_range=[min(numeric),max(numeric)] if numeric else None,
                point_is_in_mode_states=str(chosen) in modes if chosen is not None else None),
            context_split=dict(halves=split,support=split_supported,comparison_available=split_available,
                status='unsupported' if not split_supported else ('compared' if split_available else 'supported_but_flat_half'),
                representative_absolute_difference=difference,halves_within_one_grid_step=difference<=.25 if difference is not None else None,
                identified_halves=split_available and all(v['identified_preference'] for v in half_pref)))
        records.append(record)
        boot_arrays.append(np.asarray([np.nan if not v['valid'] or v['flat'] else v['representative'] for v in decisions]))
        for j,value in enumerate(GRID):
            sample=samples[:,j];sample=sample[np.isfinite(sample)]
            curves.append(dict(scope=name,item_idx=item,lambda_value=value,mean_log_loss=float(loss[ix,j].mean()),
                bootstrap_ci_low=float(np.quantile(sample,.025)) if len(sample) else None,bootstrap_ci_high=float(np.quantile(sample,.975)) if len(sample) else None,
                positive_mean_log_loss=float(loss[ix[labels[ix]==1],j].mean()) if pos else None,
                negative_mean_log_loss=float(loss[ix[labels[ix]==0],j].mean()) if pos<len(ix) else None))
    if frozen_items is not None and set(r['item_idx'] for r in records)!=set(frozen_items):raise ValueError('frozen primary item support changed')
    selected_items={r['item_idx'] for r in records};selected=mask&np.isin(items,list(selected_items))
    primary_groups=Counter(r['preference']['group'] for r in records)
    summary=dict(items=len(records),events=int(selected.sum()),feedback_users=len(np.unique(uids[selected])),
        aggregate_feedback_support_at_least512=len(np.unique(uids[selected]))>=512,identified_group_counts={g:primary_groups[g] for g in GROUPS},
        bootstrap_rule_satisfied_items=sum(r['bootstrap']['stable_descriptive'] is True for r in records),
        bootstrap_rule_satisfied_and_identified_items=sum(r['bootstrap']['stable_descriptive'] is True and r['preference']['identified_preference'] for r in records),
        split_supported_items=sum(r['context_split']['support'] for r in records),split_comparable_nonflat_items=sum(r['context_split']['comparison_available'] for r in records),
        split_within_one_step_items=sum(r['context_split']['halves_within_one_grid_step'] is True for r in records),
        representative_strictly_above_one_items=sum(r['preference'].get('strictly_above_one',False) for r in records),
        all_near_minima_strictly_above_one_items=sum(r['preference'].get('all_near_minima_strictly_above_one',False) for r in records),
        near_minimum_set_crosses_one_items=sum(r['preference'].get('minima_cross_one',False) for r in records),
        both_labels_ge5_items=sum(r['both_labels_ge5'] for r in records),positive_only_items=sum(r['only_positive'] for r in records))
    nonflat=sum(not r['preference']['flat'] for r in records)
    summary['representative_strictly_above_one_fraction_all_items']=summary['representative_strictly_above_one_items']/len(records) if records else None
    summary['representative_strictly_above_one_fraction_nonflat_items']=summary['representative_strictly_above_one_items']/nonflat if nonflat else None
    summary['all_near_minima_strictly_above_one_fraction_all_items']=summary['all_near_minima_strictly_above_one_items']/len(records) if records else None
    return records,curves,summary,np.stack(boot_arrays) if boot_arrays else np.empty((0,200))


def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(folder,support_root,output):
    plan,population,config,source_summary,support,support_table,base,loss,uids,labels,inherited=load(folder,support_root)
    feedback=sorted(map(int,np.unique(uids)));index,weights=uid_weights(feedback,uids,replicates=200,seed=17)
    frozen=set(support['edges'][str(config['edge'])]['primary_item_indices']);all_records=[];all_curves=[];summaries={};boot={}
    for name,mask in [('all_events',np.ones(len(base),dtype=bool)),('active_inherited_only',inherited>0)]:
        records,curves,summary,draws=analyze_scope(mask,name,base,loss,uids,labels,inherited,index,weights,frozen if name=='all_events' else None)
        if name=='all_events':
            prior={r['item_idx']:r for r in support_table}
            for r in records:
                if any(r[k]!=prior[r['item_idx']][k] for k in ('events','feedback_users','positive_events','negative_events')):raise ValueError('frozen item counts differ')
        if name=='active_inherited_only' and not set(r['item_idx'] for r in records)<=frozen:raise ValueError('active universe not a subset of primary')
        all_records.extend(records);all_curves.extend(curves);summaries[name]=summary
        boot[name+'_items']=np.asarray([r['item_idx'] for r in records],dtype=np.int64);boot[name+'_lambda']=draws
    output.mkdir(parents=True,exist_ok=True);np.savez_compressed(output/'bootstrap_preferences.npz',**boot)
    flat=[]
    for r in all_records:
        p,b,s=r['preference'],r['bootstrap'],r['context_split'];span=p.get('minimum_set_span',[None,None])
        flat.append(dict(scope=r['scope'],item_idx=r['item_idx'],events=r['events'],feedback_users=r['feedback_users'],positive_events=r['positive_events'],negative_events=r['negative_events'],
            inherited_active_events=r['inherited_active_events'],zero_inherited_events=r['zero_inherited_events'],only_positive=r['only_positive'],both_labels_ge5=r['both_labels_ge5'],
            flat=p['flat'],loss_range=p['loss_range'],identified_group=p['group'],representative_lambda=p['representative'],minimum_set=json.dumps(p['minimizing_lambdas']),minimum_span_low=span[0],minimum_span_high=span[1],
            valid_bootstrap_draws=b['valid_draws'],bootstrap_mode_states=json.dumps(b['mode_states']),bootstrap_mode_fraction=b['mode_fraction'],point_exact_agreement=b['point_exact_agreement'],
            point_is_in_mode_states=b['point_is_in_mode_states'],
            point_within_one_step_probability=b['point_within_one_grid_step_probability'],stable_descriptive=b['stable_descriptive'],flat_bootstrap_draws=b['flat_draws'],ambiguous_bootstrap_draws=b['ambiguous_draws'],
            split_supported=s['support'],split_comparison_available=s['comparison_available'],split_status=s['status'],split_half0_lambda=s['halves']['0']['preference']['representative'] if s['halves']['0']['preference'] else None,
            split_half1_lambda=s['halves']['1']['preference']['representative'] if s['halves']['1']['preference'] else None,split_absolute_difference=s['representative_absolute_difference'],split_within_one_step=s['halves_within_one_grid_step']))
    csv_write(output/'items.csv',flat);csv_write(output/'loss_curves.csv',all_curves)
    limitations=[
        'Item lambda preferences minimize observed-label mean loss on these same events; they are descriptive oracle sensitivities, not deployable item policies. No overall oracle-selected AUC/recovery/gain is computed.',
        'A uses one shared native-write trajectory and read-only inherited-row interpolation. It is not the older independently written alpha trajectories or final Design evaluation.',
        'Flat curves receive no representative lambda. Nonflat near-minimum ties are retained; cross-group ties are ambiguous, so a nearest1 representative cannot by itself establish preference for1.',
        'The root frozen plan is authoritative for200 joint feedbackUID bootstrap draws. It supersedes the earlier support-plan residentUID sampling wording without changing the bound preflight file.',
        'Bootstrap validity/mode/exact agreement/within-step frequency and context split agreement are all shown. The descriptive stable rule is valid>=190 and within-step probability>=.70; flat preferences are not called stable lambda choices.',
        'Context halves have disjoint UID/timestamp groups but can share users and are not independent confirmation. Unsupported or flat halves are marked unavailable, not automatically stable.',
        'Primary items need not contain both labels. Item preferences may follow label composition and realized label noise rather than stable candidate identity; stricter both-label counts remain exploratory.',
        'Active-only inherited_rows>0 sensitivity reuses the same20event/10UID support rule on state provenance, not scores. Its smaller coverage and flat/unidentified items remain visible.']
    result=dict(format='readonly_candidate_item_strength_analysis_v1',edge=config['edge'],lambdas=GRID.tolist(),epsilon_nat=EPS,
        population_users=len(population['users']),feedback_users=len(feedback),events=len(base),events_with_no_inherited_rows=int((inherited==0).sum()),
        zero_inherited_scores_bitexact=True,scopes=summaries,items=all_records,limitations=limitations,
        provenance=dict(source=str(folder.resolve()),configuration_sha256=sha(folder/'configuration.json'),source_summary_sha256=sha(folder/'summary.json'),
            population_sha256=sha(support_root/'population.json'),root_plan_sha256=sha(support_root/'plan.json'),
            support_coverage_sha256=sha(support_root/'support_coverage.json'),analysis_sha256=sha(__file__),bootstrap_replicates=200,bootstrap_seed=17,
            bootstrap_unit='joint distinct feedbackUID clusters from the full completed edge, common weights across all items/lambdas/scopes'))
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=[f'# Read-only item strength preferences: edge{config["edge"]}','','No oracle-selected overall AUC/recovery is reported. Loss curves, labels and uncertainty remain explicit.','',
        '| Scope | Items | FeedbackUID | Group counts | Bootstrap rule satisfied/identified | Split supported/comparable/within step |','|---|---:|---:|---|---:|---|']
    for name,s in summaries.items():lines.append(f"| {name} | {s['items']} | {s['feedback_users']} | {s['identified_group_counts']} | {s['bootstrap_rule_satisfied_items']}/{s['bootstrap_rule_satisfied_and_identified_items']} | {s['split_supported_items']}/{s['split_comparable_nonflat_items']}/{s['split_within_one_step_items']} |")
    lines+=['','items.csv preserves every supported item, including flat/ambiguous and unstable cases. loss_curves.csv contains all nine mean losses and UID-bootstrap intervals. bootstrap_preferences.npz records all200 representative draws (NaN for flat/invalid; those counts are separated in summary.json).','',*['- '+x for x in limitations]]
    (output/'report.md').write_text('\n'.join(lines)+'\n');(output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status='complete',edge=config['edge'],scopes=summaries)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--support-root',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.input,a.support_root,a.output)
