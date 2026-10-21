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
    plt.rcParams.update({'font.size': 7, 'axes.labelsize': 6.7,
                         'axes.titlesize': 7.1, 'xtick.labelsize': 6.4,
                         'ytick.labelsize': 6.2, 'legend.fontsize': 6.1,
                         'axes.linewidth': .45})
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.30))
    fig.subplots_adjust(left=.14, right=.99, top=.80, bottom=.105,
                        wspace=.38, hspace=.72)
    for i, (ax, row, cfg) in enumerate(zip(axes.flat, rows, d['configurations'])):
        for x, key in [(0, 'Rebuild_seconds'), (1, 'CacheTide_seconds')]:
            bottom = 0
            for j, seconds in enumerate(row[key]):
                if seconds <= 0:
                    continue
                value = seconds/1000
                # Thin white separators expose the layers without covering
                # the short handover/transfer segments with dark outlines.
                ax.bar(x, value, bottom=bottom, width=.58, color=COLORS[j],
                       edgecolor='white', lw=.22, zorder=3)
                bottom += value
            ax.annotate(f'{bottom:.1f}', (x, bottom), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=6.2, color='#3B4853')
        # Same physical unit, but scale independently so each configuration's
        # small segments remain visible. No broken or nonlinear time axes.
        limit = 12 if cfg['scale'] == 'Large' else 70
        ticks = [0, 4, 8, 12] if cfg['scale'] == 'Large' else [0, 20, 40, 60]
        ax.set(xlim=(-.65, 1.65), ylim=(0, limit),
               xticks=[0, 1], xticklabels=['Rebuild', 'CacheTide'], yticks=ticks)
        ax.set_title(f'({chr(97+i)}) {row["configuration"]}', loc='left', pad=4)
        ax.tick_params(length=2, width=.45, pad=2)
        ax.grid(axis='y', lw=.25)
        if i % 2 == 0:
            ax.set_ylabel('Time (ks)', labelpad=2)
    handles = [Patch(facecolor=c, edgecolor='white', lw=.22, label=s)
               for s, c in zip(STAGES, COLORS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, 1.005),
               ncol=3, columnspacing=.85, handlelength=1.15,
               handletextpad=.35, labelspacing=.35, borderaxespad=0,
               fontsize=6.1)
    finish(fig, __file__, '6-5-time', 'Complete controlled-release time breakdown',
           layout=False, paper_column=True)
    Path(__file__).with_name('time_ledger.json').write_text(json.dumps(
        {'synthetic': True, 'time_unit': 'seconds',
         'scenario': d['scenario'], 'configurations': rows}, indent=2)+'\n')


if __name__ == '__main__':
    main()
