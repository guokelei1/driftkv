"""SIMULATED DATA: Figure 8, design analysis on Yambda-Max.

(a) Recovery against stored coordinates per row, one line per per-head read
width: rising lines show that capacity helps, close lines show that reading
more does not. (b) Recovery by candidate-specificity quintile for the dynamic
gate and a constant gate that folds into candidate-independent K/V.
All values in data.json are invented layout previews.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import COLORS, finish, load_data
import matplotlib.pyplot as plt

BLUE = COLORS['CacheTide']
READ_STYLES = {2: ('#9DBBD6', 's', (0, (3, 1.2))),
               6: (BLUE, 'o', '-'),
               12: ('#2B3F55', '^', (0, (1, 1)))}
CONST = '#B65757'


def main():
    d = load_data(__file__)
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.3,
                         'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'legend.fontsize': 6.2})
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55))
    fig.subplots_adjust(left=.13, right=.99, bottom=.25, top=.86, wspace=.42)

    c = d['capacity']
    stored = np.asarray(c['stored_coordinates'])
    ax = axes[0]
    for row in c['read_widths']:
        color, marker, dash = READ_STYLES[row['read']]
        ax.plot(stored, row['recovery_pct'], color=color, marker=marker,
                ls=dash, ms=2.6, lw=1.15 if row['read'] == 6 else .9,
                label=f"read {row['read']}: +{row['read_flops_pct']}%")
    default = c['read_widths'][[r['read'] for r in c['read_widths']].index(
        c['default_read'])]
    y = default['recovery_pct'][list(stored).index(c['default_stored'])]
    ax.scatter(c['default_stored'], y, s=34, facecolors='none',
               edgecolors=BLUE, lw=.7, zorder=6)
    ax.set_xscale('log', base=2)
    ax.set(xticks=stored, xticklabels=[str(x) for x in stored],
           xlim=(20, 170), ylim=(15, 100), yticks=[20, 40, 60, 80, 100],
           xlabel='Stored coordinates / row', ylabel='Recovery (%)')
    ax.minorticks_off()
    ax.set_title('(a) Capacity vs. read width', loc='left', pad=3)
    ax.legend(loc='lower right', handlelength=1.6, handletextpad=.35,
              labelspacing=.15, borderaxespad=.1, fontsize=5.9)

    s = d['specificity']
    q = np.arange(len(s['quintiles']))
    ax = axes[1]
    dyn = np.asarray(s['dynamic_gate_recovery_pct'])
    const = np.asarray(s['constant_gate_recovery_pct'])
    ax.fill_between(q, const, dyn, color=BLUE, alpha=.10, lw=0, zorder=1)
    ax.plot(q, dyn, color=BLUE, marker='o', ms=2.6, lw=1.15, zorder=4,
            label='Candidate gate')
    ax.plot(q, const, color=CONST, marker='s', ms=2.5, lw=.9,
            ls=(0, (3, 1.2)), zorder=3, label='Constant gate')
    ax.set(xticks=q, xticklabels=s['quintiles'], xlim=(-.3, len(q) - .7),
           ylim=(15, 100), yticks=[20, 40, 60, 80, 100],
           xlabel='Candidate specificity')
    ax.set_title('(b) Candidate conditioning', loc='left', pad=3)
    ax.legend(loc='lower left', handlelength=1.6, handletextpad=.35,
              labelspacing=.25, borderaxespad=.1)

    for ax in axes:
        ax.tick_params(length=2, pad=2)
        ax.grid(axis='y', lw=.35)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
    finish(fig, __file__, '6-3-design', 'Design analysis', layout=False)


if __name__ == '__main__':
    main()
