"""SIMULATED DATA: Figure 7, release continuity and returning users.

(a,b) follow one cache and one correction state through V1-V5 on the two Max
models. The reset arm restarts each release from a rebuilt source, so the gap
between the two recovery lines is the error accumulated across releases.
(c,d) withhold k consecutive releases from a fixed Yambda-Max cohort and
compare composed updates with step-by-step replay at the target release.
All values in data.json are invented layout previews.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import COLORS, finish, load_data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

BLUE = COLORS['CacheTide']
RESET = '#7FAED6'
REPLAY = '#BF7145'
REUSE = '#7F8890'
BAR, BAR_EDGE = '#D6E2EC', '#9CB2C4'


def main():
    d = load_data(__file__)
    plt.rcParams.update({'font.size': 7.2, 'axes.titlesize': 7.5,
                         'axes.labelsize': 7.2, 'xtick.labelsize': 6.8,
                         'ytick.labelsize': 6.8, 'legend.fontsize': 6.9})
    fig, axes = plt.subplots(1, 4, figsize=(7.05, 1.30))
    fig.subplots_adjust(left=.068, right=.995, top=.74, bottom=.27,
                        wspace=.42)

    versions = d['release_labels']
    xs = np.arange(len(versions))
    for i, (ax, s) in enumerate(zip(axes[:2], d['consecutive'])):
        ax.bar(xs, s['added_cost_pct'], width=.5, color=BAR,
               edgecolor=BAR_EDGE, lw=.4, zorder=2)
        ax.plot(xs, s['chained_recovery_pct'], color=BLUE, marker='o',
                ms=2.6, lw=1.15, zorder=4)
        # The reset arm is drawn as open rings around the chained points so
        # that coinciding values stay visible on the shared percent axis.
        ax.plot(xs, s['reset_recovery_pct'], color=RESET, ls='',
                marker='s', mfc='none', ms=4.6, mew=.75, zorder=5)
        ax.set(xlim=(-.55, len(xs) - .45), ylim=(0, 105), xticks=xs,
               xticklabels=versions, yticks=[0, 50, 100], xlabel='Release')
        ax.set_title(f'({chr(97 + i)}) {s["configuration"]}', pad=3)
    axes[0].set_ylabel('Recovery / cost (%)', labelpad=2)

    r = d['returning']
    k = np.asarray(r['skipped_releases'])
    ax = axes[2]
    ax.axhline(r['rebuild_auc'], color=COLORS['Rebuild'], lw=.6,
               ls=(0, (4, 2)), zorder=1)
    ax.plot(k, r['reuse_auc'], color=REUSE, marker='o', mfc='white',
            ms=2.6, mew=.7, lw=.85, ls=(0, (1, 1.2)), zorder=3)
    ax.plot(k, r['composed_auc'], color=BLUE, marker='o', ms=2.6, lw=1.15,
            zorder=4)
    ax.plot(k, r['replay_auc'], color=REPLAY, ls='', marker='D', mfc='none',
            ms=4.2, mew=.7, zorder=5)
    ax.set(xticks=k, ylim=(.733, .770), yticks=[.74, .75, .76, .77],
           xlabel='Skipped releases', ylabel='AUC')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title('(c) Returning-user AUC', pad=3)

    ax = axes[3]
    prep = r['preparation_pct_rebuild']
    ax.plot(k, prep['replay'], color=REPLAY, marker='D', mfc='none', ms=3.0,
            mew=.7, lw=.9, ls=(0, (3, 1.2)), zorder=3)
    ax.plot(k, prep['composed'], color=BLUE, marker='o', ms=2.6, lw=1.15,
            zorder=4)
    ax.set(xticks=k, ylim=(0, 21), yticks=[0, 10, 20],
           xlabel='Skipped releases', ylabel='Prep. / Rebuild (%)')
    ax.set_title('(d) Preparation cost', pad=3)

    for ax in axes:
        ax.tick_params(length=2, pad=1.5)
        ax.grid(axis='y', lw=.35)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2

    handles = [
        Line2D([], [], color=BLUE, marker='o', ms=2.8, lw=1.1,
               label='CacheTide'),
        Line2D([], [], color=RESET, marker='s', mfc='none', ms=4.0,
               mew=.75, ls='', label='CacheTide (reset)'),
        Line2D([], [], color=REPLAY, marker='D', mfc='none', ms=3.2,
               ls=(0, (3, 1.2)), lw=.9, label='Step-by-step replay'),
        Patch(facecolor=BAR, edgecolor=BAR_EDGE, lw=.4, label='Added cost'),
        Line2D([], [], color=REUSE, marker='o', mfc='white', ms=2.8,
               ls=(0, (1, 1.2)), lw=.85, label='Reuse'),
        Line2D([], [], color=COLORS['Rebuild'], ls=(0, (4, 2)), lw=.6,
               label='Rebuild'),
    ]
    fig.legend(handles=handles, ncol=len(handles), loc='upper center',
               bbox_to_anchor=(.53, 1.01), handlelength=2.0,
               handletextpad=.4, columnspacing=1.4, borderaxespad=0)
    finish(fig, __file__, '6-2-releases',
           'Consecutive releases and returning users', layout=False)


if __name__ == '__main__':
    main()
