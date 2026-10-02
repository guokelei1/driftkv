"""CPU-only descriptive audit of retained fitting artifacts; no model execution."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / 'results/read_correction_2026_09'
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(feature, budget, scale, edge):
    tail = Path('query_only/pure') / feature / f'c{budget}' / scale / edge
    for parent in ('population_run/calibration', 'resource_canary', ''):
        folder = BASE / 'v5' / parent / tail
        if (folder / 'calibration.pt').exists():
            return folder
    raise FileNotFoundError(tail)


def rms(value):
    return float(value.double().square().mean().sqrt())


def quantiles(value):
    return {str(q): float(torch.quantile(value.double(), q)) for q in (0., .1, .5, .9, 1.)}


def parameter_metrics(module, units):
    state = module['state_dict']
    w, bias, mean, scale = (state[name].double() for name in ('weight', 'bias', 'input_mean', 'input_scale'))
    width = len(scale) // 2
    near = scale <= 1e-5
    derivative = torch.exp(torch.minimum(mean[:width], torch.zeros_like(mean[:width])))
    jacobian = w[:width] / scale[:width, None] + derivative[:, None] * w[width:] / scale[width:, None]
    # A descriptive local linearization at the saved train mean, not a measured
    # rolling-query sensitivity and not proof that the feature causes failure.
    local_gain = float((jacobian * scale[:width, None]).norm() / (width**.5 * units))
    return {'finite': all(bool(torch.isfinite(t).all()) for t in (w,bias,mean,scale)),
        'raw_q_mean_rms': rms(mean[:width]), 'raw_q_std_rms': rms(scale[:width]),
        'raw_q_scale_quantiles': quantiles(scale[:width]), 'elu_scale_quantiles': quantiles(scale[width:]),
        'elu_mean_near_minus_one_count': int((mean[width:] <= -.999999).sum()),
        'nearconstant_count': int(near.sum()), 'nearconstant_raw_q_count': int(near[:width].sum()),
        'scale_floor_count': int((scale <= 1.01e-6).sum()),
        'weight_norm': float(w.norm()), 'weight_rms': rms(w), 'weight_max_abs': float(w.abs().max()),
        'weight_norm_in_output_units': float(w.norm() / units), 'bias_rms_in_output_units': rms(bias) / units,
        'nearconstant_weight_norm': float(w[near].norm()),
        'nearconstant_weight_energy_fraction': float(w[near].square().sum() / w.square().sum().clamp_min(1e-30)),
        'nearconstant_effective_slope_norm': float((w[near] / scale[near, None]).norm()),
        'local_jacobian_norm_at_train_mean': float(jacobian.norm()),
        'local_std_input_to_target_rms_gain': local_gain}


def main():
    torch.set_num_threads(2)
    records, models, source_records, csv_rows, layer_rows = [], {}, {}, [], []
    for scale_name in ('medium', 'large', 'max'):
        for number in range(1,6):
            edge = f'v{number-1}_to_v{number}'
            path = BASE / 'v5/population_run/evaluation' / scale_name / edge / 'summary.json'
            evaluated = json.loads(path.read_text())
            source_records[str(path.relative_to(ROOT))] = sha(path)
            full = evaluated['current_full']['ROC_AUC']
            reuse = evaluated['current_reuse']['ROC_AUC']
            historical = {}
            for name, history_path in (
                ('v2_query', BASE/'v2/query_only'/scale_name/edge/'summary.json'),
                ('v4_history', BASE/'v4/population_run/evaluation'/scale_name/edge/'summary.json')):
                historical_record = json.loads(history_path.read_text())
                assert historical_record['current_full']['ROC_AUC'] == full
                assert historical_record['current_reuse']['ROC_AUC'] == reuse
                historical[name] = [{key: point.get(key) for key in (
                    'budget','variant','users','requests','baseline_auc','recovery_percent','relative_flops_percent')}
                    for point in historical_record['points'] if point.get('kind') == 'measurement']
                source_records[str(history_path.relative_to(ROOT))] = sha(history_path)
            for point in evaluated['points']:
                if point['method'] != 'query_only':
                    continue
                budget = point['budget']
                raw_folder = artifact('cross_phi', budget, scale_name, edge)
                joint_folder = artifact('cross_phi_joint8', budget, scale_name, edge)
                metadata_path = joint_folder/'calibration.json'
                metadata = json.loads(metadata_path.read_text())
                raw = torch.load(raw_folder/'calibration.pt', map_location='cpu', weights_only=False)
                joint = torch.load(joint_folder/'calibration.pt', map_location='cpu', weights_only=False)
                source_records[str(metadata_path.relative_to(ROOT))] = sha(metadata_path)
                source_records[str(raw_folder.joinpath('calibration.pt').relative_to(ROOT))] = sha(raw_folder/'calibration.pt')
                actual_hash = sha(joint_folder/'calibration.pt')
                assert actual_hash == metadata['weights_sha256']
                source_records[str(joint_folder.joinpath('calibration.pt').relative_to(ROOT))] = actual_hash
                models[f'{scale_name}/{edge}'] = metadata['model_config']
                ridge, refined = metadata['fit']['ridge'], metadata['fit']['joint_refinement']
                layer_stats = []
                for layer, (old,new,diagnostic,units) in enumerate(zip(raw['modules'],joint['modules'],ridge['layers'],
                        refined['output_units_per_layer'],strict=True)):
                    old_state,new_state=old['state_dict'],new['state_dict']
                    assert torch.equal(old_state['input_mean'],new_state['input_mean'])
                    assert torch.equal(old_state['input_scale'],new_state['input_scale'])
                    item = {'layer':layer,'output_unit':units,'ridge_fit':diagnostic['fit'],
                        'diagnostics':diagnostic['diagnostics'],'raw':parameter_metrics(old,units),
                        'joint':parameter_metrics(new,units),
                        'weight_relative_change':float((new_state['weight'].double()-old_state['weight'].double()).norm()/old_state['weight'].double().norm().clamp_min(1e-30)),
                        'bias_change_rms_in_output_units':rms(new_state['bias']-old_state['bias'])/units}
                    for split in ('train','validation'):
                        d=diagnostic['diagnostics'][split]
                        item[f'{split}_rate_error_ratio'] = d['fitted_rate_mse']/d['baseline_rate_mse']
                        item[f'{split}_read_error_ratio'] = d['fitted_read_mse']/d['baseline_read_mse']
                    layer_stats.append(item)
                    layer_rows.append({'scale':scale_name,'edge':edge,'budget':budget,'layer':layer,
                        'output_unit':units,'train_read_ratio':item['train_read_error_ratio'],
                        'val_read_ratio':item['validation_read_error_ratio'],
                        'nearconstant_count':item['joint']['nearconstant_count'],
                        'q_mean_rms':item['joint']['raw_q_mean_rms'],'q_std_rms':item['joint']['raw_q_std_rms'],
                        'raw_weight_norm':item['raw']['weight_norm'],'joint_weight_norm':item['joint']['weight_norm'],
                        'raw_near_weight_norm':item['raw']['nearconstant_weight_norm'],
                        'joint_near_weight_norm':item['joint']['nearconstant_weight_norm'],
                        'joint_local_gain':item['joint']['local_std_input_to_target_rms_gain']})
                counts = np.array(list(ridge['counts_by_uid'].values()))
                summary = {'scale':scale_name,'edge':edge,'budget':budget,'users':point['users'],
                    'requests':point['requests'],'full_auc':full,'reuse_auc':reuse,'auc_gap':full-reuse,
                    'auc':point['baseline_auc'],'auc_delta_vs_reuse':point['baseline_auc']-reuse,
                    'recovery_percent':point['recovery_percent'],'selected_epoch':refined['selected_epoch'],
                    'initial_train_objective':refined['initial_train']['objective'],
                    'final_train_objective':refined['final_train']['objective'],
                    'initial_validation_objective':refined['epochs'][0]['validation']['objective'],
                    'selected_validation_objective':refined['selected_validation']['objective'],
                    'nearconstant_count':sum(l['joint']['nearconstant_count'] for l in layer_stats),
                    'nearconstant_layers':[l['layer'] for l in layer_stats if l['joint']['nearconstant_count']],
                    'validation_worse_layers':[l['layer'] for l in layer_stats if l['validation_read_error_ratio']>1],
                    'maximum_validation_read_error_ratio':max(l['validation_read_error_ratio'] for l in layer_stats),
                    'first_layer_output_unit':layer_stats[0]['output_unit'],
                    'maximum_output_unit':max(l['output_unit'] for l in layer_stats),
                    'maximum_joint_weight_norm_in_units':max(l['joint']['weight_norm_in_output_units'] for l in layer_stats),
                    'maximum_local_gain':max(l['joint']['local_std_input_to_target_rms_gain'] for l in layer_stats),
                    'clipped_fraction':refined['clipped_steps']/refined['optimizer_steps'],
                    'count_min':int(counts.min()),'count_median':float(np.median(counts)),
                    'count_max':int(counts.max()),'count_full_fraction':float(np.mean(counts==1024))}
                csv_rows.append(summary)
                records.append({**summary,'paths':{'raw':str(raw_folder.relative_to(ROOT)),
                    'joint':str(joint_folder.relative_to(ROOT))},'historical_controls':historical,
                    'objective_statistics':refined['objective_statistics'],
                    'initial_train':refined['initial_train'],'final_train':refined['final_train'],
                    'initial_validation':refined['epochs'][0]['validation'],
                    'selected_validation':refined['selected_validation'],
                    'epochs':refined['epochs'],'layers':layer_stats})
    result={'role':'CPU-only descriptive fitting diagnostic; all 15 frozen edges and all three v5 Q budgets retained',
        'nearconstant_definition':'input_scale <= 1e-5; reference floor is 1e-6; no threshold chosen on quality',
        'limitations':['Training/validation are prerelease pure-parent snapshots; final AUC uses causal rolling cache.',
            'Local Jacobian at saved train mean is a descriptive linearization, not actual rolling evidence.',
            'Large parameter/slope norms and heldout losses do not alone establish causes of AUC failure.',
            'Historical controls share exact Full/Reuse anchors; no score-dependent request filtering performed.'],
        'model_configs':models,'records':records,'source_sha256':source_records,
        'analysis_source_sha256':sha(Path(__file__))}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('overview.csv',csv_rows),('layers.csv',layer_rows)]:
        with (OUT/name).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'status':'complete','calibration_points':len(records),'layers':len(layer_rows),
                      'output':str(OUT/'analysis.json')}))


if __name__ == '__main__':
    main()
