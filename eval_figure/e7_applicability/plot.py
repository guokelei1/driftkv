"""SIMULATED: two simple controlled slices of the complete-FLOPs cost model.
All coefficients and groups come from data.json, never from experimental results.
Transfers are reported in a separate byte ledger, not converted to FLOPs/time.
"""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from style import finish, load_data, DATASET_COLORS, SCALE_STYLES


def costs(cfg, scenario, history, density):
    """One complete release; arrays broadcast across history/query-density grid."""
    p = cfg['flop_coefficients']
    f = cfg['fixed_cost_flops']
    n, r = np.asarray(history), np.asarray(density)
    users = cfg['active_population']
    appends = scenario['behaviors_per_user_per_release']
    queries = appends * r
    rebuild = p['rebuild_linear_flops_per_row']*n + p['rebuild_quadratic_flops_per_row_squared']*n*n
    score = p['native_score_fixed_flops'] + p['native_score_flops_per_history_row']*n
    append = p['native_append_fixed_flops'] + p['native_append_flops_per_history_row']*n
    materialize = p['materialize_fixed_flops'] + p['materialize_flops_per_row']*n
    handover = p['handover_fixed_flops'] + p['handover_flops_per_row']*n
    delta_query = p['correction_read_fixed_flops'] + p['correction_read_flops_per_inherited_row']*(n-appends/2)
    delta_append = p['correction_read_fixed_flops'] + p['correction_read_flops_per_inherited_row']*(n-(appends-1)/2)
    initialization = f['initialization_total']/scenario['releases_amortizing_initialization']
    fixed_fit = (initialization + f['calibration_per_release'] +
                 f['teacher_preparation_per_release'] + f['source_preparation_per_release'])
    catalogue = scenario['shared_table_replicas']*cfg['catalogue_items']*p['catalogue_flops_per_item']
    shared = fixed_fit + catalogue
    common = users*(queries*score + appends*append)
    added = users*(materialize + handover + queries*delta_query + appends*delta_append)
    return {'rebuild_flops':users*rebuild, 'common_event_flops':common,
            'initialization_allocated_flops':initialization,
            'calibration_flops':f['calibration_per_release'],
            'teacher_preparation_flops':f['teacher_preparation_per_release'],
            'source_preparation_flops':f['source_preparation_per_release'],
            'catalogue_preparation_flops':catalogue,
            'shared_flops':shared,
            'materialization_flops':users*materialize,
            'handover_flops':users*handover,
            'added_query_flops':users*queries*delta_query,
            'added_append_flops':users*appends*delta_append,
            'cachetide_added_flops':shared+added,
            'rebuild_total_flops':users*rebuild+common,
            'cachetide_total_flops':shared+added+common}


def transfers(cfg, scenario, n):
    """A separate assumed ingress/egress byte ledger, not a timing model."""
    u, layers, size = cfg['active_population'], scenario['layer_count'], scenario['scalar_bytes']
    native_row = layers*2*scenario['head_count']*scenario['head_dimension']*size
    code_row = layers*scenario['code_coordinates_per_row_layer']*size
    return {'common_native_state_bytes':scenario['native_cache_transfers_per_release']*u*n*native_row,
            'additional_code_state_bytes':scenario['code_transfers_per_release']*u*n*code_row,
            'shared_table_bytes':scenario['shared_table_replicas']*cfg['catalogue_items']*
                                 scenario['shared_tables_per_replica']*code_row}


def main():
    d = load_data(__file__)
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75), sharey=True)
    fig.subplots_adjust(left=.16, right=.985, bottom=.24, top=.72, wspace=.23)
    for cfg in d['configurations']:
        name = f"{cfg['dataset']}-{cfg['scale']}"
        for kind, history, density in [
            ('history', np.asarray(d['slices']['history']['history_length']),
             d['slices']['history']['fixed_queries_per_behavior']),
            ('density', d['slices']['density']['fixed_history_length'],
             np.asarray(d['slices']['density']['queries_per_behavior']))]:
            c = costs(cfg, d['scenario'], history, density)
            expected = 100*(1-c['cachetide_total_flops']/c['rebuild_total_flops'])
            if not np.allclose(expected, d['slices'][kind]['saving_pct'][name]):
                raise ValueError(f'{name}: stored slice no longer matches cost coefficients')
        style = dict(color=DATASET_COLORS[cfg['dataset']], **SCALE_STYLES[cfg['scale']],
                     linewidth=1.15, markersize=2.7, markeredgewidth=.45)
        axes[0].plot(d['slices']['history']['history_length'],
                     d['slices']['history']['saving_pct'][name], label=name, **style)
        axes[1].plot(d['slices']['density']['queries_per_behavior'],
                     d['slices']['density']['saving_pct'][name], label=name, **style)
    axes[0].set_title('(a)', loc='left', fontsize=7.4, pad=2)
    axes[1].set_title('(b)', loc='left', fontsize=7.4, pad=2)
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_ylim(-30, 90)
        ax.set_yticks([-20, 0, 40, 80])
        ax.axhline(0, color='#858F97', linestyle=':', linewidth=.7, zorder=0)
        ax.grid(axis='y', linewidth=.4)
        ax.tick_params(labelsize=7, length=2, pad=2)
        ax.minorticks_off()
    axes[0].set_xticks([128, 512, 2048], ['128', '512', '2048'])
    axes[1].set_xticks([1, 16, 256], ['1', '16', '256'])
    axes[0].set_ylabel('FLOPs saved (%)', fontsize=7.5, labelpad=3)
    axes[0].set_xlabel('History length', fontsize=7.3, labelpad=3)
    axes[1].set_xlabel('Queries / behavior', fontsize=7.3, labelpad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, 1.02),
               ncol=2, fontsize=6.8, columnspacing=.9, labelspacing=.3,
               handlelength=1.6, handletextpad=.45)
    finish(fig, __file__, '6-7', 'Compute sensitivity', layout=False, paper_column=True)
    ledger = {'synthetic': True, 'unit_flops': 'floating-point operations',
              'unit_transfers': 'bytes, no time/FLOPs conversion', 'configurations': []}
    for cfg in d['configurations']:
        cases = []
        for n, r in [(128, 1), (1024, 16), (1024, 256), (2048, 256)]:
            case = costs(cfg, d['scenario'], n, r)
            cases.append({'history_length': n, 'queries_per_behavior': r,
                **{key: float(value) for key, value in case.items()},
                'saving_pct': float(100*(1-case['cachetide_total_flops']/case['rebuild_total_flops'])),
                **transfers(cfg, d['scenario'], n)})
        ledger['configurations'].append({'dataset': cfg['dataset'], 'scale': cfg['scale'],
            'active_population': cfg['active_population'], 'catalogue_items': cfg['catalogue_items'],
            'reference_cases': cases})
    Path(__file__).with_name('model_ledger.json').write_text(json.dumps(ledger, indent=2)+'\n')


if __name__ == '__main__':
    main()
