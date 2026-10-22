"""SIMULATED DATA: Figure 9 (cost) and Table 3 (persistent storage).

(a) Release-attributable time per configuration, normalized to Rebuild.
(b) Steady-state scoring latency added over the native read, against
history length, for the fused read and the framework-operator read.
(c) Latency of each user's first read after a release, as a CDF across users.
storage_table() writes the Table 3 body used by main.tex.
All values in data.json are invented layout previews.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import COLORS, finish, load_data
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator

BLUE = COLORS['CacheTide']
SEGMENT_COLORS = ['#B9A5D0', '#80AACD', '#DAB674', '#79B7A2', '#8D9CA9']
LATENCY_STYLES = {
    'Native read': ('#7F8890', 'o', (0, (1, 1.2)), 'white'),
    'CacheTide (fused)': (BLUE, 'o', '-', None),
    'CacheTide (framework ops)': ('#9DBBD6', 's', (0, (3, 1.2)), None),
}
FIRST_READ_STYLES = {
    'Reuse': ('#7F8890', (0, (1, 1.2)), .9),
    'CacheTide': (BLUE, '-', 1.2),
    'CacheTide (per-row projection)': ('#9DBBD6', (0, (3, 1.2)), .95),
    'Rebuild on first read': (COLORS['Rebuild'], (0, (4, 1.5)), .95),
}
SHORT = {'Yambda-Large': 'Y-Large', 'Yambda-Max': 'Y-Max',
         'RecFlow-Large': 'R-Large', 'RecFlow-Max': 'R-Max'}


def plain(value, _):
    return f'{value:g}'


def release_time(ax, d):
    rows = d['configurations'][::-1]
    y = np.arange(len(rows))
    for i, row in enumerate(rows):
        left = 0.0
        for value, color in zip(row['cachetide_pct'], SEGMENT_COLORS):
            ax.barh(i, value, left=left, height=.58, color=color,
                    edgecolor='white', lw=.35, zorder=3)
            left += value
        ax.annotate(f'{left:.0f}%', (left, i), xytext=(3, 0),
                    textcoords='offset points', va='center', fontsize=6.3,
                    color='#3B4853')
    ax.set(yticks=y, yticklabels=[SHORT[r['configuration']] for r in rows],
           xlim=(0, 34), xticks=[0, 10, 20, 30],
           xlabel='Release time / Rebuild (%)')
    ax.grid(axis='x', lw=.35)
    ax.set_title('(a) Release processing', loc='left', pad=3)
    handles = [Patch(facecolor=c, edgecolor='white', lw=.3, label=s)
               for s, c in zip(d['segments'], SEGMENT_COLORS)]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-.02, 1.19),
              ncol=3, handlelength=.9, handletextpad=.3, columnspacing=.6,
              labelspacing=.25, borderaxespad=0, fontsize=6.0)


def steady_latency(ax, d):
    """Added latency of each corrected read over the native read (%)."""
    n = np.asarray(d['history_length'])
    native = np.asarray(d['median_ms']['Native read'])
    for name in ['CacheTide (fused)', 'CacheTide (framework ops)']:
        color, marker, dash, face = LATENCY_STYLES[name]
        added = 100*(np.asarray(d['median_ms'][name])/native - 1)
        ax.plot(n, added, color=color, marker=marker, ls=dash, ms=2.6,
                mfc=face or color, mew=.7,
                lw=1.2 if name == 'CacheTide (fused)' else .9,
                label={'CacheTide (fused)': 'Fused read',
                       'CacheTide (framework ops)': 'Framework operators'}[name])
    ax.set_xscale('log', base=2)
    ax.set(xticks=n, xticklabels=['256', '512', '1K', '2K', '4K'],
           ylim=(0, 120), yticks=[0, 40, 80, 120], xlabel='History length',
           ylabel='Added latency (%)')
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis='y', lw=.35)
    ax.set_title('(b) Steady-state request', loc='left', pad=3)
    ax.legend(loc='lower left', bbox_to_anchor=(-.02, 1.19), ncol=2,
              handlelength=1.8, handletextpad=.35, columnspacing=.8,
              labelspacing=.2, borderaxespad=0, fontsize=6.0)


def first_read(ax, d):
    p = np.asarray(d['percentiles'])
    for name, values in d['latency_ms'].items():
        color, dash, lw = FIRST_READ_STYLES[name]
        ax.plot(values, p, color=color, ls=dash, lw=lw,
                label={'CacheTide (per-row projection)': 'Per-row projection',
                       'Rebuild on first read': 'Rebuild on read'}.get(name, name))
    ax.set_xscale('log')
    ax.set(xlim=(.3, 60), ylim=(0, 100), yticks=[0, 50, 100],
           xticks=[.5, 1, 2, 5, 10, 20, 50], xlabel='First-read latency (ms)',
           ylabel='Users (%)')
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis='y', lw=.35)
    ax.set_title('(c) First read after release', loc='left', pad=3)
    ax.legend(loc='lower left', bbox_to_anchor=(-.02, 1.19), ncol=2,
              handlelength=1.8, handletextpad=.35, columnspacing=.7,
              labelspacing=.2, borderaxespad=0, fontsize=6.0)


def storage_table(d):
    s = d['storage']
    lines = []
    for i, name in enumerate(s['configurations']):
        lines.append(
            f"{name} & {s['native_kv_tb'][i]:.2f} & "
            f"{s['correction_state_gb'][i]:.0f} & {s['item_tables_gb'][i]:.1f} & "
            f"{s['added_pct_native'][i]:.1f} & {s['handover_direct_gb'][i]:.2f} & "
            f"{s['handover_blockwise_mb'][i]:.0f}\\\\")
    out = Path(__file__).with_name('table_6_4_storage_simulated.tex')
    out.write_text('% SIMULATED: generated by e12_cost/plot.py\n' +
                   '\n'.join(lines) + '\n')
    print(out)


def main():
    d = load_data(__file__)
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.3,
                         'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.62))
    fig.subplots_adjust(left=.065, right=.99, bottom=.24, top=.63,
                        wspace=.36)
    release_time(axes[0], d['release_time'])
    steady_latency(axes[1], d['steady_latency'])
    first_read(axes[2], d['first_read'])
    for ax in axes:
        ax.tick_params(length=2, pad=2)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
    finish(fig, __file__, '6-4-cost',
           'Release time, request latency and first-read latency',
           layout=False)
    storage_table(d)


if __name__ == '__main__':
    main()
