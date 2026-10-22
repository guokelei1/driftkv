"""SIMULATED DATA: Figure 10, applicability boundaries.

(a) Absolute AUC while one release keeps serving and appended feedback
replaces the inherited window, with and without short-tail supervision.
(b) Net FLOPs saved against candidate reads per user per release on the two
Max models, with the Figure 6 ledger marked. Read-count curves reuse the
complete-FLOPs coefficients of e7_applicability. All values are invented.
"""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from style import COLORS, DATASET_COLORS, finish, load_data
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullLocator

BLUE = COLORS['CacheTide']
HOLD_STYLES = {
    'Rebuild once': (COLORS['Rebuild'], (0, (4, 1.5)), None, .9),
    'CacheTide': (BLUE, '-', 'o', 1.2),
    'CacheTide (no tail)': ('#9DBBD6', (0, (3, 1.2)), 's', .95),
    'Reuse': ('#7F8890', (0, (1, 1.2)), 'o', .9),
}


def cost_model():
    spec = importlib.util.spec_from_file_location(
        'release_cost_reference', ROOT / 'e7_applicability/plot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    d = load_data(__file__)
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.3,
                         'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'legend.fontsize': 6.1})
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.62))
    fig.subplots_adjust(left=.15, right=.99, bottom=.24, top=.70, wspace=.50)

    h = d['fixed_release']
    x = np.asarray(h['appended_over_window'])
    ax = axes[0]
    ax.axvline(1.0, color='#B8BFC5', lw=.6, ls=(0, (2, 2)), zorder=1)
    ax.annotate('inherited rows\nevicted', (1.0, .7535), xytext=(3, 0),
                textcoords='offset points', fontsize=5.8, color='#69737C',
                va='bottom')
    for name, values in h['auc'].items():
        color, dash, marker, lw = HOLD_STYLES[name]
        ax.plot(x, values, color=color, ls=dash, marker=marker, ms=2.2,
                mfc='white' if name == 'Reuse' else color, mew=.6, lw=lw,
                label={'Rebuild once': 'Rebuild', 'CacheTide (no tail)': 'No tail'}
                .get(name, name), zorder=4 if name == 'CacheTide' else 3)
    ax.set(xlim=(-.05, 2.05), xticks=[0, .5, 1, 1.5, 2],
           xticklabels=['0', '.5N', 'N', '1.5N', '2N'], ylim=(.750, .770),
           yticks=[.750, .760, .770], xlabel='Appended feedback',
           ylabel='AUC')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_title('(a) Fixed release', loc='left', pad=3)
    ax.legend(loc='lower left', bbox_to_anchor=(-.05, 1.17), ncol=2,
              handlelength=1.8, handletextpad=.3, columnspacing=.6,
              labelspacing=.2, borderaxespad=0)

    rc = d['read_count']
    source = json.loads((Path(__file__).parent / rc['cost_source']).read_text())
    assert source['meta']['synthetic'] is True
    ops = cost_model()
    reads = np.geomspace(rc['reads_min'], rc['reads_max'], rc['points'])
    appends = source['scenario']['behaviors_per_user_per_release']
    ax = axes[1]
    ax.axhline(0, color='#969DA4', ls=':', lw=.6)
    ax.axvline(rc['ledger_reads'], color='#B8BFC5', lw=.6, ls=(0, (2, 2)))
    ax.annotate('Fig. 6\nledger', (rc['ledger_reads'], 22), xytext=(3, 0),
                textcoords='offset points', ha='left', fontsize=5.8,
                color='#69737C')
    for cfg in source['configurations']:
        name = f"{cfg['dataset']}-{cfg['scale']}"
        if name not in rc['configurations']:
            continue
        f = ops.costs(cfg, source['scenario'], rc['history_length'],
                      reads / appends)
        saved = 100 * (f['rebuild_flops'] - f['cachetide_added_flops']) / \
            f['rebuild_flops']
        ax.plot(reads, saved, color=DATASET_COLORS[cfg['dataset']],
                ls='-' if cfg['dataset'] == 'Yambda' else (0, (3, 1.2)),
                lw=1.05, label=name)
    ax.set_xscale('log', base=2)
    ax.set(xlim=(12, 10000), ylim=(-60, 100), xticks=[16, 256, 4096],
           xticklabels=['16', '256', '4K'], yticks=[-50, 0, 50, 100],
           xlabel='Reads / user / release', ylabel='FLOPs saved (%)')
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title('(b) Read count', loc='left', pad=3)
    ax.legend(loc='lower left', handlelength=1.8, handletextpad=.3,
              labelspacing=.2, borderaxespad=.2)

    for ax in axes:
        ax.tick_params(length=2, pad=2)
        ax.grid(axis='y', lw=.35)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
    finish(fig, __file__, '6-5-boundaries',
           'Fixed-release and read-count boundaries', layout=False)


if __name__ == '__main__':
    main()
