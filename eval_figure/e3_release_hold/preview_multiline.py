"""SIMULATED layout stress test, not additional dataset evidence.

Yambda uses the existing preview values. The neighboring purple trace is
deliberately invented to test annotation overlap, NOT a RecFlow prediction.
Outputs stay in paper/build and do not replace the manuscript figure/data.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot import cost_trajectory, load_data
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


def main():
    d = load_data(__file__)
    q = d['quality_preview']
    a = np.asarray(q['appends'], dtype=int)
    n = d['scenario']['window_length']
    x = cost_trajectory(d)['added_percent'][a]
    r = 100*(np.asarray(q['auc']['CacheTide'])-np.asarray(q['auc']['Reuse'])) / (
        np.asarray(q['auc']['Rebuild'])-np.asarray(q['auc']['Reuse']))
    # Pure layout perturbation: do not interpret these as operation counts or
    # a measured second dataset. The adjacent paths intentionally nearly overlap.
    traces = [('Yambda-Max', '#1764AB', '-', x, r),
              ('RecFlow-Max', '#8964A2', '--', x*1.03+1.2, r+2.8-1.2*a/(2*n))]
    stages = [(0, 'o', '0'), (4, '^', '0.5'), (6, 's', '1'), (8, 'D', '2')]
    plt.rcParams.update({'font.size': 7.4, 'axes.labelsize': 7.4,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 6.7, 'axes.linewidth': .65})
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    fig.subplots_adjust(left=.145, right=.985, bottom=.22, top=.82)
    for i, (name, col, ls, costs, rec) in enumerate(traces):
        ax.plot(costs, rec, color=col, ls=ls, lw=1.3, zorder=2+i)
        for j, marker, _ in stages:
            ax.plot(costs[j], rec[j], marker=marker, ls='',
                    ms=4.8 if i == 0 else 3.5, color=col,
                    mfc='white' if i == 0 else col,
                    mew=.85 if i == 0 else .5, zorder=5+i)
    dataset_handles = [Line2D([], [], color=c, ls=ls, lw=1.3, label=name)
                       for name, c, ls, _, _ in traces]
    fig.legend(handles=dataset_handles, loc='upper center',
               bbox_to_anchor=(.55, 1.015), ncol=2, columnspacing=1.2,
               handlelength=2.0, handletextpad=.5)
    stage_handles = [Line2D([], [], color='none', ls='', label='Appends / N:')]
    stage_handles += [Line2D([], [], color='#505A64', marker=m, ls='',
                             mfc='white', mew=.8, ms=3.6, label=lab)
                      for _, m, lab in stages]
    fig.legend(handles=stage_handles, loc='upper center',
               bbox_to_anchor=(.54, .92), ncol=5, columnspacing=.7,
               handlelength=.7, handletextpad=.3)
    ax.set(xlim=(0, 120), ylim=(0, 100), xticks=[0, 25, 50, 75, 100],
           yticks=[0, 25, 50, 75, 100], xlabel='Added FLOPs (% of Rebuild)',
           ylabel='Recovery (%)')
    ax.grid(False)
    ax.tick_params(length=2.5, pad=3, color='#737D87')
    for spine in ax.spines.values():
        spine.set_color('#737D87')
    out = Path('/home/gkl/work/paper/build/hold_multiline_20261020')
    out.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        meta = ({'Title': 'SIMULATED adjacent trajectory annotation preview'}
                if ext == 'pdf' else
                {'Description': 'SIMULATED layout only. Purple trace invented to test overlap.'})
        fig.savefig(out/f'shared_stage_key.{ext}', dpi=300,
                    bbox_inches='tight', pad_inches=.025, metadata=meta)
    plt.close(fig)
    output = {'synthetic': True,
              'purpose': 'Annotation stress test only; no independent RecFlow result or cost ledger.',
              'base_source': str(Path(__file__).with_name('data.json')),
              'second_trace': 'x=1.03*x_Yambda+1.2, recovery=recovery_Yambda+2.8-1.2*appends/(2N); invented for near-overlap.',
              'appends': a.tolist(),
              'traces': [{'label': name, 'added_cost_percent': c.tolist(),
                          'recovery_percent': rec.tolist()}
                         for name, _, _, c, rec in traces]}
    (out/'layout_only_data.json').write_text(json.dumps(output, indent=2)+'\n')
    print(out/'shared_stage_key.png')


if __name__ == '__main__':
    main()
