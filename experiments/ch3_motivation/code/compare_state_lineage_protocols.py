#!/usr/bin/env python3
"""Paired native-input versus actual-chain-input fit protocol comparison."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
import numpy as np
from experiments.ch3_motivation.code.analyze_state_lineage_cached_gap import load_scores
from experiments.ch3_motivation.code.analyze_real_quality import (
    binary_point_draws, uid_weights, interval, sha,
)

DEFAULT=ROOT/'experiments/ch3_motivation/runs/simple_config_sweep/attempt_1/native_pair'
ACTUAL=ROOT/'experiments/ch3_motivation/runs/lineage_candidate/attempt_1/state/cached_gap/quality'
POPULATION=ROOT/'experiments/ch3_motivation/runs/lineage_candidate/attempt_1/population.json'


def snapshot_mean(rows):
    values={}
    for r in rows:
        current=(r['kv_mse'],r['k_mse'],r['v_mse'])
        if r['uid'] in values and values[r['uid']]!=current:raise ValueError('snapshot error varies across requests')
        values[r['uid']]=current
    return {name:float(np.mean([v[i] for v in values.values()])) for i,name in enumerate(('kv_mse','k_mse','v_mse'))}


def run(root,actual_root,population_path):
    results={}
    for target in (3,4):
        folders={'native_pair':root/f'quality/final_{target}','actual_chain':actual_root/f'final_{target}'}
        loaded={name:load_scores(folder,population_path) for name,folder in folders.items()}
        native_config,_,population,native=loaded['native_pair'];actual_config,_,_,actual=loaded['actual_chain']
        for field in ('users','population_sha256','cutoff','models','input_binding_sha256','query_gap_protocol','method_names'):
            if native_config[field]!=actual_config[field]:raise ValueError(f'protocols differ in matched input: {field}')
        fields=('uid','query_timestamp','item_idx','label','query_gap','history_rows','appended_events')
        checks={}
        for method in ('reuse','direct','chain','rebuild'):
            if [tuple(r[f] for f in fields) for r in native[method]]!=[tuple(r[f] for f in fields) for r in actual[method]]:
                raise ValueError(f'protocols differ in event keys, labels or context: {method}')
            if method!='chain':
                a=np.asarray([r['score'] for r in native[method]],dtype=np.float64)
                b=np.asarray([r['score'] for r in actual[method]],dtype=np.float64)
                if not np.array_equal(a.view(np.uint64),b.view(np.uint64)):
                    raise ValueError(f'{method} scores are not bitexact; stop before protocol inference')
                checks[method]=dict(events=len(a),bitexact=True,maximum_absolute_score_difference=0.)
        uids=np.asarray([r['uid'] for r in native['chain']],dtype=np.uint64)
        labels=np.asarray([r['label'] for r in native['chain']],dtype=np.int8)
        index,weights=uid_weights(population['users'],uids,replicates=500,seed=17)
        points={};draws={};sources={}
        for name,groups in [('native_pair',native),('actual_chain',actual)]:
            scores=np.asarray([r['score'] for r in groups['chain']],dtype=float)
            p,d=binary_point_draws(labels,scores,index,weights)
            saved=json.loads((folders[name]/'analysis/summary.json').read_text())
            if (saved['provenance']['configuration_sha256']!=sha(folders[name]/'configuration.json')
                or saved['provenance']['source_summary_sha256']!=sha(folders[name]/'summary.json')):
                raise ValueError('prior CPU summary has changed source bindings')
            for metric in ('AUC','log_loss'):
                if not np.isclose(p[metric]['value'],saved['metrics']['methods']['chain'][metric]['value'],atol=1e-15,rtol=0):
                    raise ValueError('recomputed metric differs from saved protocol summary')
            mse=snapshot_mean(groups['chain'])
            for metric,value in mse.items():
                if not np.isclose(value,saved['metrics']['methods']['chain']['feedback_uid_mean_snapshot_MSE'][metric],atol=1e-15,rtol=0):raise ValueError('snapshot MSE differs from saved summary')
            points[name]=dict(AUC=p['AUC'],log_loss=p['log_loss'],feedback_uid_mean_snapshot_MSE=mse)
            draws[name]={metric:d[metric] for metric in ('AUC','log_loss')}
            sources[name]=dict(directory=str(folders[name].resolve()),configuration_sha256=sha(folders[name]/'configuration.json'),
                score_summary_sha256=sha(folders[name]/'summary.json'),analysis_summary_sha256=sha(folders[name]/'analysis/summary.json'))
        contrast={metric:interval(points['native_pair'][metric]['value']-points['actual_chain'][metric]['value'],
            draws['native_pair'][metric]-draws['actual_chain'][metric]) for metric in ('AUC','log_loss')}
        m_native=points['native_pair']['feedback_uid_mean_snapshot_MSE']['kv_mse']
        m_actual=points['actual_chain']['feedback_uid_mean_snapshot_MSE']['kv_mse']
        results[str(target)]=dict(resident_users=len(population['users']),feedback_users=len(np.unique(uids)),events=len(labels),
            unchanged_method_bitexact_checks=checks,protocols=points,native_pair_minus_actual_chain=contrast,
            raw_KV_MSE_increase=m_native-m_actual,raw_KV_MSE_relative_increase_percent=100*(m_native/m_actual-1),sources=sources)
    result=dict(format='state_lineage_training_input_protocol_comparison_v1',targets=results,
        contrast_definition='native-pair-trained chain minus actual-chain-input-trained chain; positive AUC favors native-pair; negative logloss favors native-pair.',
        pairing='One common set of500 UID bootstrap multiplicity vectors(seed17) per target, over all2048 fixed residents; subtraction occurs draw by draw, never between separate interval endpoints.',
        population_sha256=sha(population_path),analysis_sha256=sha(__file__),
        numerical_helper_sha256=sha(Path(__file__).with_name('analyze_real_quality.py')),
        loader_sha256=sha(Path(__file__).with_name('analyze_state_lineage_cached_gap.py')),
        limitations=[
            'Only the training input protocol changes for the chain. Direct, Reuse and Rebuild are asserted bitexact on every complete matched event, not merely close in aggregate metrics.',
            'Every method uses the fixed target-cutover cache and unchanged cached-prefix query gap. This is a controlled counterfactual, not native serving or final Design evaluation.',
            'UID intervals condition on the frozen fitted weights and omit training-seed uncertainty; they are pointwise and unadjusted for multiplicity.',
            'Larger raw KV error need not imply lower AUC or higher logloss. Native-pair training can create an input-distribution mismatch under chained deployment, but two cases do not establish a universal ranking law.',
            'No user, event, threshold, weight or checkpoint is selected using these quality results.'])
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Native-pair versus actual-chain-input fitting','',result['contrast_definition'],result['pairing'],'',
        '| Target | Native-pair / actual-chain AUC | Paired AUC difference pp [95% CI] | Paired logloss difference [95% CI] | Native-pair / actual-chain UID-mean KV MSE |','|---|---|---|---|---|']
    for target,r in results.items():
        n,a=r['protocols']['native_pair'],r['protocols']['actual_chain'];auc=r['native_pair_minus_actual_chain']['AUC'];ll=r['native_pair_minus_actual_chain']['log_loss']
        lines.append(f"| {target} | {n['AUC']['value']:.6f}/{a['AUC']['value']:.6f} | {100*auc['value']:+.4f} [{100*auc['ci95'][0]:+.4f},{100*auc['ci95'][1]:+.4f}] | {ll['value']:+.7f} [{ll['ci95'][0]:+.7f},{ll['ci95'][1]:+.7f}] | {n['feedback_uid_mean_snapshot_MSE']['kv_mse']:.6f}/{a['feedback_uid_mean_snapshot_MSE']['kv_mse']:.6f} |")
    lines+=['','All three unchanged methods pass complete-event bitexact checks in both targets. Larger native-pair KV error appears in both targets; the task-metric directions must be read from the paired table rather than inferred from MSE.','',*['- '+x for x in result['limitations']]]
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',results=results)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--actual-root',type=Path,default=ACTUAL);p.add_argument('--population',type=Path,default=POPULATION)
    a=p.parse_args();run(a.root,a.actual_root,a.population)
