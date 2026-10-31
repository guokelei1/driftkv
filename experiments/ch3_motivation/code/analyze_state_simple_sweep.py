#!/usr/bin/env python3
"""Collect all six prespecified Direct/Chain settings; no quality selection."""
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
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[3]
import numpy as np
from experiments.ch3_motivation.code.analyze_real_quality import sha, write_csv, flatten_interval
from experiments.ch3_motivation.code.state_lineage_probe import RowResidualMap
from hstu_kvcache.methods.baselines.protocol.cost import CostModel

DEFAULT=ROOT/'experiments/ch3_motivation/runs/simple_config_sweep/attempt_1'
PREVIOUS=ROOT/'experiments/ch3_motivation/runs/lineage_candidate/attempt_1'
SETTINGS=((32,256),(32,1024),(64,256),(64,1024),(128,256),(128,1024))


def map_flops(hidden,rows):
    return RowResidualMap.forward_flops(SimpleNamespace(pairs=100,width=64,hidden=hidden),rows)


def fit_details(summary):
    identity=summary['identity'];target=int(identity['final_target']);hidden=int(identity['hidden']);budget=int(identity['steps'])
    reports={};expected_parameters=100*(128*hidden+hidden+hidden*64+64)
    def record(name,r):
        trace=r['trace'];selected=int(r['selected_step']);steps=int(r['actual_optimizer_steps'])
        if steps!=budget or r['parameter_count']!=expected_parameters:raise ValueError('fit budget or map parameter count differs')
        if not trace or trace[0]['step']!=0 or trace[-1]['step']!=steps:raise ValueError('incomplete validation trace')
        selected_values=[v['validation_kv_mse'] for v in trace if v['step']==selected]
        best=float(r['validation_kv_mse'])
        if len(selected_values)!=1 or not np.isclose(selected_values[0],best,atol=1e-12,rtol=0):raise ValueError('selected checkpoint differs from validation record')
        if not np.isclose(min(v['validation_kv_mse'] for v in trace),best,atol=1e-12,rtol=0):raise ValueError('checkpoint was not the validation minimum')
        initial=float(trace[0]['validation_kv_mse']);last=float(trace[-1]['validation_kv_mse'])
        prefix=min(v['validation_kv_mse'] for v in trace if v['step']<=256)
        train=map_flops(hidden,r['sampled_rows_per_step'])*steps
        validation=map_flops(hidden,r['validation_rows'])*len(trace)
        reports[name]=dict(parameter_count=r['parameter_count'],actual_optimizer_steps=steps,
            selected_step=selected,selected_at_budget_boundary=selected==steps,
            initial_validation_kv_mse=initial,best_validation_kv_mse=best,last_validation_kv_mse=last,
            initial_to_best_validation_improvement=initial-best,
            initial_to_best_validation_improvement_percent=100*(initial-best)/initial if initial else None,
            best_validation_through_step256_in_this_trace=prefix,
            best_validation_improvement_after256_in_this_trace=prefix-best if steps>256 else None,
            last_validation_interval_improvement=float(trace[-2]['validation_kv_mse']-last) if len(trace)>1 else None,
            last_sampled_batch_training_kv_mse=trace[-1].get('training_kv_mse'),
            train_rows=r['train_rows'],validation_rows=r['validation_rows'],sampled_rows_per_step=r['sampled_rows_per_step'],
            validation_calls=len(trace),training_forward_flops=train,validation_forward_flops=validation,
            training_backward_validation_estimate=3*train+validation)
    for version,stage in summary['stages'].items():
        for kind,report in stage.items():
            if int(version)==2 and kind!='chain':continue  # Exact shared first-stage aliases.
            record(f'v{version}_{kind}',report)
    for kind,report in summary.get('short_stages',{}).items():record('short_v4_'+kind,report)
    endpoints={}
    for method,names in [('direct',[f'v{target}_direct']),('chain',[f'v{v}_chain' for v in range(2,target+1)])]:
        endpoint=[reports[n] for n in names]
        endpoints[method]=dict(fit_names=names,map_count=len(names),parameters_in_composed_path=sum(r['parameter_count'] for r in endpoint),
            actual_optimizer_steps_total=sum(r['actual_optimizer_steps'] for r in endpoint),
            training_backward_validation_estimate=sum(r['training_backward_validation_estimate'] for r in endpoint))
    dense,fused=CostModel(320,10,10,'torch'),CostModel(320,10,10,'triton')
    per_version_capture=sum((fused if n>=1024 else dense).full_cache(n) for n in summary['history_lengths'].values())
    return dict(per_stage_parameters=expected_parameters,stage_reports=reports,endpoints=endpoints,
        completed_unique_fits_in_attempt=len(reports),
        attempt_all_map_training_backward_validation_estimate=sum(r['training_backward_validation_estimate'] for r in reports.values()),
        chain_fit_row_materialization_forward_flops=map_flops(hidden,summary['sampled_fit_rows']+summary['sampled_validation_rows'])*(target-1),
        native_capture_required_versions=list(range(1,target+1)),native_capture_forward_reference=per_version_capture*target,
        standalone_direct_native_capture_forward_reference=per_version_capture*2,
        forward_flops_per_map_N1024=map_flops(hidden,1024),
        native_capture_reused=summary.get('backbone_capture_reused',False))


def collect(root,previous):
    plan_path=root/'plan.json';plan=json.loads(plan_path.read_text())
    expected={(f'w{w}_s{s}',w,s) for w,s in SETTINGS}
    if {(s['name'],s['width'],s['steps']) for s in plan['state_configs']}!=expected:raise ValueError('prespecified six-setting grid changed')
    population=Path(plan['population']);pop=json.loads(population.read_text())
    if sha(population)!=plan['population_sha256'] or len(pop['users'])!=2048:raise ValueError('fixed population differs')
    rows=[];details=[];sources=[]
    for width,steps in SETTINGS:
        name=f'w{width}_s{steps}'
        for target in (3,4):
            quality=(previous/f'state/cached_gap/quality/final_{target}' if (width,steps)==(64,256)
                else root/f'quality/{name}/final_{target}')
            analysis_path=quality/'analysis/summary.json'
            if not analysis_path.exists():raise ValueError(f'all12 completed quality points required; missing {analysis_path}')
            analysis=json.loads(analysis_path.read_text());config=json.loads((quality/'configuration.json').read_text())
            score_summary=json.loads((quality/'summary.json').read_text())
            if score_summary['status']!='complete' or config['diagnostic'] or not score_summary['quality_use_permitted']:raise ValueError('incomplete or diagnostic quality point')
            if config['users']!=pop['users'] or config['population_sha256']!=sha(population) or config['query_gap_protocol']!='cached_prefix_gap':raise ValueError('population or query input differs')
            if set(config['method_names'])!={'reuse','direct','chain','rebuild'}:raise ValueError('unplanned quality methods')
            provenance=analysis['provenance']
            if (provenance['configuration_sha256']!=sha(quality/'configuration.json')
                or provenance['source_summary_sha256']!=sha(quality/'summary.json')
                or provenance['population_sha256']!=sha(population)):raise ValueError('CPU analysis binding differs')
            fit_folder=Path(config['checkpoint']).parent;fit_path=fit_folder/'summary.json';fitted=json.loads(fit_path.read_text());identity=fitted['identity']
            if (fitted['status']!='complete' or identity['hidden']!=width or identity['steps']!=steps or identity['final_target']!=target
                or identity['population_sha256']!=sha(population) or fitted['checkpoint_sha256']!=config['checkpoint_sha256']):raise ValueError('fitted map identity differs')
            metrics=analysis['metrics']
            if metrics['feedback_users']!=plan['feedback_users'][str(target)] or not metrics['mechanism_use_permitted']:raise ValueError('fixed feedback coverage differs')
            budget=fit_details(fitted)
            row=dict(setting=name,width=width,steps_per_stage=steps,final_target=target,feedback_users=metrics['feedback_users'],events=metrics['events'],
                source='previous_cached_gap' if (width,steps)==(64,256) else 'new_sweep',per_stage_parameters=budget['per_stage_parameters'])
            flatten_interval(row,'direct_minus_chain_AUC',metrics['paired_direct_minus_chain']['AUC'])
            flatten_interval(row,'rebuild_minus_reuse_AUC',metrics['rebuild_minus_reuse_AUC'])
            for method in ('reuse','rebuild'):flatten_interval(row,method+'_AUC',metrics['methods'][method]['AUC'])
            for method in ('direct','chain'):
                m=metrics['methods'][method];last=budget['stage_reports'][f'v{target}_{method}'];endpoint=budget['endpoints'][method]
                for metric in ('AUC','AUC_minus_reuse','log_loss','recovery_percent'):flatten_interval(row,method+'_'+metric,m[metric])
                row[method+'_recovery_ratio_ci_permitted']=m['recovery_percent']['ratio_ci_permitted']
                for field,value in m['feedback_uid_mean_snapshot_MSE'].items():row[method+'_feedback_uid_mean_'+field]=value
                for field in ('selected_step','selected_at_budget_boundary','best_validation_kv_mse','initial_to_best_validation_improvement_percent',
                    'best_validation_through_step256_in_this_trace','best_validation_improvement_after256_in_this_trace','last_validation_interval_improvement'):
                    row[method+'_final_'+field]=last[field]
                for field in ('map_count','parameters_in_composed_path','actual_optimizer_steps_total','training_backward_validation_estimate'):
                    row[method+'_'+field]=endpoint[field]
            row.update(attempt_completed_unique_map_fits=budget['completed_unique_fits_in_attempt'],
                attempt_all_map_training_backward_validation_estimate=budget['attempt_all_map_training_backward_validation_estimate'],
                native_capture_forward_reference=budget['native_capture_forward_reference'],
                native_capture_reused=budget['native_capture_reused'])
            rows.append(row)
            details.append(dict(setting=name,final_target=target,quality=str(quality.resolve()),fit_summary=str(fit_path.resolve()),budget_and_traces=budget))
            for path in (analysis_path,quality/'configuration.json',quality/'summary.json',fit_path):sources.append(dict(path=str(path.resolve()),sha256=sha(path)))
    if len(rows)!=12:raise ValueError('all12 points required')
    changes=[]
    for width in (32,64,128):
        for target in (3,4):
            before=next(r for r in rows if r['width']==width and r['steps_per_stage']==256 and r['final_target']==target)
            after=next(r for r in rows if r['width']==width and r['steps_per_stage']==1024 and r['final_target']==target)
            changes.append(dict(width=width,final_target=target,from_steps=256,to_steps=1024,
                direct_AUC_point_change=after['direct_AUC']-before['direct_AUC'],chain_AUC_point_change=after['chain_AUC']-before['chain_AUC'],
                direct_minus_chain_AUC_point_change=after['direct_minus_chain_AUC']-before['direct_minus_chain_AUC'],
                ci95=None,scope='Point changes only; no cross-configuration paired bootstrap is computed. Separate per-config intervals must not be subtracted to claim significance.'))
    notes=[
        'All six prespecified settings and both final targets are retained. No best AUC setting is selected, and checkpoint selection uses validation KV MSE only.',
        'AUC is primary. Validation/held-out raw KV MSE and trace improvements are optimization diagnostics, not task-quality proxies.',
        'Within-setting Direct−Chain intervals reuse the existing500 pairedUID bootstrap analysis. Cross-setting256→1024 changes are descriptive points only.',
        'Changing the chain step budget also changes its selected incoming predecessor states; the final-stage1024 trace is not simply continuation of the old256 endpoint.',
        'Map training FLOPs use actual optimizer steps and sampled batch rows; validation forward calls use the complete recorded trace. Backward is an explicit approximate2× forward, so training total is3× forward.',
        'FLOPs exclude optimizer updates, normalization statistics, transfers and MSE arithmetic; they are arithmetic estimates, not measured runtime or complete serving costs.',
        'Direct uses one map while chain uses two/three. Per-stage parameter count and budget match, but total fitted capacity, training and materialization budgets differ.',
        'Native row capture is reused across the new grid: the same target capture must not be charged as independently executed for every configuration. The legacy W64/256 attempt had extra unused fits, retained in its attempt-level totals.',
        'All caches stay fixed at target cutoff; actual query timestamps use the cached-prefix gap without clipping. This is a counterfactual mechanism study, not native serving or final Design evaluation.',
        'A negative or near-zero native Rebuild−Reuse gap makes recovery difficult to interpret; unclipped point ratios and stability flags are retained, with ordinary AUC gains primary.']
    return dict(format='state_simple_sweep_summary_v1',grid=rows,point_count=12,setting_count=6,point_changes_256_to1024=changes,
        fit_details=details,notes=notes,sources=sources,
        provenance=dict(plan_path=str(plan_path.resolve()),plan_sha256=sha(plan_path),population_sha256=sha(population),
            analysis_sha256=sha(__file__),map_flops_helper_sha256=sha(Path(__file__).with_name('state_lineage_probe.py')),
            native_cost_helper_sha256=sha(ROOT/'src/hstu_kvcache/methods/baselines/protocol/cost.py')))


def run(root,previous,output):
    result=collect(root,previous);output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'grid.csv',result['grid']);write_csv(output/'point_changes_256_to1024.csv',result['point_changes_256_to1024'])
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Prespecified simple Direct/Chain grid','',
        '| Setting | Target | Direct AUC | Chain AUC | Direct−Chain pp [95% CI] | Direct/Chain UID-mean KV MSE | Final selected steps Direct/Chain |','|---|---|---:|---:|---|---:|---|']
    for r in result['grid']:
        lines.append(f"| {r['setting']} | {r['final_target']} | {r['direct_AUC']:.6f} | {r['chain_AUC']:.6f} | {100*r['direct_minus_chain_AUC']:+.4f} [{100*r['direct_minus_chain_AUC_ci_low']:+.4f},{100*r['direct_minus_chain_AUC_ci_high']:+.4f}] | {r['direct_feedback_uid_mean_kv_mse']:.6f}/{r['chain_feedback_uid_mean_kv_mse']:.6f} | {r['direct_final_selected_step']}/{r['chain_final_selected_step']} |")
    lines+=['','| Setting | Target | Parameters per stage | Direct/Chain stages | Direct/Chain total optimizer steps | Map train+backward+val estimate TFLOPs Direct/Chain |','|---|---|---:|---|---|---|']
    for r in result['grid']:lines.append(f"| {r['setting']} | {r['final_target']} | {r['per_stage_parameters']} | {r['direct_map_count']}/{r['chain_map_count']} | {r['direct_actual_optimizer_steps_total']}/{r['chain_actual_optimizer_steps_total']} | {r['direct_training_backward_validation_estimate']/1e12:.4f}/{r['chain_training_backward_validation_estimate']/1e12:.4f} |")
    lines+=['','The complete grid.csv also includes native anchors, unclipped recovery/stability, per-method AUC intervals and final-stage validation trace improvements. Detailed per-stage budgets and selected steps are in summary.json. point_changes_256_to1024.csv contains descriptive point changes, without significance claims.','',*['- '+x for x in result['notes']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n');(output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status='complete',points=len(result['grid']),output=str(output.resolve()))),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--previous',type=Path,default=PREVIOUS);p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.root,a.previous,a.output or a.root/'aggregate')
