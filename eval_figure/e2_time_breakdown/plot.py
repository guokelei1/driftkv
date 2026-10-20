"""SIMULATED: matched complete-trace time attribution, not online serving.

Each subplot is one independently calibrated model group. All native events
match; setup is paid once and every correction operation remains charged.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import finish, load_data
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

STAGES = ['Fit / tables', 'State prep', 'Service ref.',
          'Service Δ', 'Handover', 'Transfers']
COLORS = ['#B9A5D0', '#80AACD', '#CCD2D7', '#DAB674', '#79B7A2', '#8D9CA9']


def time_ledger(cfg, scenario):
    u = cfg['active_users']
    scores = u*scenario['scored_candidates_per_user']
    appends = u*scenario['behavior_appends_per_user']
    p = cfg['operation_ms']
    native = (scores*p['native_score']+appends*p['native_append'])/1000
    correction = (scores*(p['corrected_score']-p['native_score'])+
                  appends*(p['corrected_append']-p['native_append']))/1000
    common_transfer = cfg['transfer_seconds']['common']
    extra_transfer = cfg['transfer_seconds']['additional_codes_and_tables']
    rebuild = [0, u*p['rebuild_history']/1000, native, 0, 0, common_transfer]
    ct = [sum(cfg['first_release_shared_seconds'].values()),
          u*p['materialize_code']/1000, native, correction,
          u*p['reanchor_code']/1000, common_transfer+extra_transfer]
    assert min(rebuild+ct) >= 0
    common = native+common_transfer
    additional = sum(ct)-common
    assert np.isclose(sum(rebuild), common+rebuild[1])
    assert np.isclose(sum(ct)-sum(rebuild), additional-rebuild[1])
    return {'configuration': f"{cfg['dataset']}-{cfg['scale']}",
            'active_users': u, 'score_operations': scores, 'append_operations': appends,
            'stage_names': STAGES,
            'Rebuild_seconds': rebuild, 'CacheTide_seconds': ct,
            'common_reference_seconds': common,
            'native_service_reference_seconds': native,
            'corrected_service_seconds': native+correction,
            'additional_vs_native_reuse_seconds': additional,
            'Rebuild_total_seconds': sum(rebuild), 'CacheTide_total_seconds': sum(ct),
            'CacheTide_over_Rebuild': sum(ct)/sum(rebuild)}


def main():
    d = load_data(__file__)
    rows = [time_ledger(cfg, d['scenario']) for cfg in d['configurations']]
    plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7,
                         'axes.titlesize': 7.2, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
                         'hatch.linewidth': .45})
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.15))
    fig.subplots_adjust(left=.16, right=.985, top=.80, bottom=.14,
                        wspace=.52, hspace=.82)
    for i, (ax, row, cfg) in enumerate(zip(axes.flat, rows, d['configurations'])):
        for y, key in [(1, 'Rebuild_seconds'), (0, 'CacheTide_seconds')]:
            start = 0
            for j, seconds in enumerate(row[key]):
                if seconds <= 0:
                    continue
                value = seconds/1000
                ct_work = key == 'CacheTide_seconds' and j in [0, 1, 3, 4]
                ax.barh(y, value, left=start, height=.42, color=COLORS[j],
                        edgecolor='#69747D', lw=.32,
                        hatch='///' if ct_work else None, zorder=3)
                start += value
            ax.annotate(f'{start:.1f}', (start, y), xytext=(3, 0),
                        textcoords='offset points', va='center', fontsize=6.3,
                        color='#3B4853')
        total = max(row['Rebuild_total_seconds'], row['CacheTide_total_seconds'])/1000
        ax.set(xlim=(0, total*1.30), ylim=(-.65, 1.65),
               yticks=[1, 0], yticklabels=['Rebuild', 'CT'])
        ax.set_title(f'({chr(97+i)}) {row["configuration"]}', loc='left', pad=3)
        ax.tick_params(length=2, pad=2)
        ax.set_xticks([0, 5, 10] if cfg['scale'] == 'Large' else [0, 25, 50])
        ax.grid(axis='x', lw=.35)
        if i >= 2:
            ax.set_xlabel('Time (ks)', labelpad=2)
    handles = [Patch(facecolor=c, edgecolor='#69747D', lw=.32, label=s)
               for s, c in zip(STAGES, COLORS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, 1.005),
               ncol=3, columnspacing=.9, handlelength=1.25,
               handletextpad=.4, labelspacing=.55)
    finish(fig, __file__, '6-5-time', 'Complete controlled-release time breakdown',
           layout=False, paper_column=True)
    Path(__file__).with_name('time_ledger.json').write_text(json.dumps(
        {'synthetic': True, 'time_unit': 'seconds',
         'scenario': d['scenario'], 'configurations': rows}, indent=2)+'\n')


if __name__ == '__main__':
    main()
