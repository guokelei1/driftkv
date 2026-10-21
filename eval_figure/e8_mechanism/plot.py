"""SIMULATED DATA: two design-mechanism panels, not experimental evidence.

Manuscript captions and file metadata identify simulation, per user preference.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import COLORS, finish, load_data
import matplotlib.pyplot as plt


def main():
    d = load_data(__file__)
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.3,
                         'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5})
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.62))
    fig.subplots_adjust(left=.15, right=.985, bottom=.25, top=.78, wspace=.55)
    ax = axes[0]
    for row in d['read_structure']:
        default = row['label'] == '72/6'
        color = COLORS['CacheTide'] if row['stored'] == 72 else '#84929E'
        marker = 'o' if row['stored'] == 72 else 's'
        x, y = row['read_flops_pct'], row['gain_pp']
        ax.scatter(x, y, s=20, marker=marker, color=color, zorder=4)
        if default:
            ax.scatter(x, y, s=54, facecolors='none', edgecolors=color, lw=.6)
        ax.annotate(row['label'], (x, y), xytext=(4 if x == 12 else -4, 4),
                    textcoords='offset points', ha='left' if x == 12 else 'right',
                    fontsize=6.3, color=color)
    c = d['constant_gate']
    ax.scatter(c['read_flops_pct'], c['gain_pp'], marker='x', s=22,
               color='#B65757', linewidths=.9, zorder=4)
    ax.annotate('Const.', (c['read_flops_pct'], c['gain_pp']), xytext=(5, -4),
                textcoords='offset points', ha='left', va='top',
                fontsize=6.3, color='#B65757')
    ax.set(xlim=(7, 38), ylim=(1.10, 1.77), xticks=[12, 22, 32],
           yticks=[1.2, 1.4, 1.6], xlabel='Added read FLOPs (%)',
           ylabel='AUC gain (pp)')
    ax.set_title('(a) Read structure', loc='left', pad=4)

    ax = axes[1]
    cal = d['calibration']
    ax.plot(cal['fit_users'], cal['with_tail_recovery'], color=COLORS['CacheTide'],
            marker='o', ms=3, lw=1.2, label='With tail')
    ax.plot(cal['fit_users'], cal['without_tail_recovery'], color='#B65757',
            marker='s', ls='--', ms=2.8, lw=1.0, label='No tail')
    ax.set_xscale('log', base=2)
    ax.set(xlim=(12, 340), ylim=(48, 95), xticks=cal['fit_users'],
           xticklabels=[str(x) for x in cal['fit_users']], yticks=[50, 70, 90],
           xlabel='Fitting users', ylabel='Recovery (%)')
    ax.minorticks_off()
    ax.set_title('(b) Calibration', loc='left', pad=4)
    ax.legend(loc='lower right', fontsize=6.2, handlelength=1.3,
              handletextpad=.4, borderaxespad=.2, labelspacing=.4)
    for ax in axes:
        ax.tick_params(length=2, pad=2)
        ax.grid(axis='y', lw=.4)
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3
    finish(fig, __file__, '6-3-design', 'Design mechanism analysis', layout=False)


if __name__ == '__main__':
    main()
