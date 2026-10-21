"""SIMULATED: combined fixed-release continuation and read-count cost boundary.

Reuses original preview data and arithmetic. The two panels have different
x variables and normalizations, explicitly described in the manuscript.
"""
from pathlib import Path
import sys
import json
import importlib.util

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from style import COLORS, DATASET_COLORS, SCALE_STYLES, finish, load_data


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    data = load_data(__file__)
    folder = Path(__file__).resolve().parent
    hold = json.loads((folder/data['hold_source']).read_text())
    cost = json.loads((folder/data['cost_source']).read_text())
    assert hold['meta']['synthetic'] and cost['meta']['synthetic']
    hold_ops = module_at('hold_cost_reference', ROOT/'e3_release_hold/plot.py')
    cost_ops = module_at('release_cost_reference', ROOT/'e7_applicability/plot.py')
    ledger = hold_ops.cost_trajectory(hold)
    q = hold['quality_preview']
    checkpoints = np.asarray(q['appends'], dtype=int)
    recovery = 100*(np.array(q['auc']['CacheTide'])-np.array(q['auc']['Reuse']))/(np.array(q['auc']['Rebuild'])-np.array(q['auc']['Reuse']))
    x = ledger['added_percent'][checkpoints]
    n = hold['scenario']['window_length']
    plt.rcParams.update({'font.size':7, 'axes.titlesize':7.1,
                         'axes.labelsize':6.7, 'xtick.labelsize':6.4,
                         'ytick.labelsize':6.4})
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.45))
    fig.subplots_adjust(left=.15, right=.99, bottom=.25, top=.71, wspace=.60)
    ax = axes[0]
    ax.plot(x, recovery, color=COLORS['CacheTide'], lw=1.2)
    for i, offset, align in [(0,(4,2),'left'),(4,(-3,5),'right'),
                              (6,(-4,3),'right'),(8,(-4,-6),'right')]:
        ax.plot(x[i], recovery[i], 'o', color=COLORS['CacheTide'], ms=2.7)
        ax.annotate(f'{checkpoints[i]/n:g}N', (x[i],recovery[i]),
                    xytext=offset, textcoords='offset points', ha=align,
                    fontsize=6.1, color=COLORS['CacheTide'])
    ax.set(xlim=(0,125), ylim=(-4,102), xticks=[0,50,100],
           yticks=[0,50,100], xlabel='Added FLOPs (%)', ylabel='Recovery (%)')
    ax.set_title('(a) Fixed release', loc='left', pad=4)
    cfg_slice = data['read_count_slice']
    reads = np.geomspace(cfg_slice['candidate_reads_per_user_release_min'],
                         cfg_slice['candidate_reads_per_user_release_max'],
                         cfg_slice['points'])
    appends = cost['scenario']['behaviors_per_user_per_release']
    ax = axes[1]
    output = []
    for cfg in cost['configurations']:
        f = cost_ops.costs(cfg, cost['scenario'], cfg_slice['history_length'], reads/appends)
        saved = 100*(f['rebuild_flops']-f['cachetide_added_flops'])/f['rebuild_flops']
        label = ('Y' if cfg['dataset']=='Yambda' else 'R')+'-'+cfg['scale'][0]
        ax.plot(reads, saved, color=DATASET_COLORS[cfg['dataset']],
                linestyle=SCALE_STYLES[cfg['scale']]['linestyle'], lw=1.0,
                marker=SCALE_STYLES[cfg['scale']]['marker'], ms=2.3,
                markevery=[0,24,48,64,80], label=label)
        output.append({'configuration':f"{cfg['dataset']}-{cfg['scale']}",
                       'reads_per_user_release':reads.tolist(),
                       'net_flops_saved_pct_rebuild':saved.tolist()})
    ax.set_xscale('log', base=2)
    ax.set(xlim=(12,10000), ylim=(-70,105), xticks=[16,256,4096],
           xticklabels=['16','256','4k'], yticks=[-50,0,100],
           xlabel='Reads / release', ylabel='FLOPs saved (%)')
    ax.minorticks_off()
    ax.axhline(0,color='#969DA4',ls=':',lw=.6)
    ax.set_title('(b) Read count', loc='left', pad=4)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(.995,1.015),
               ncol=4, handlelength=1.35, handletextpad=.3,
               columnspacing=.6, borderaxespad=0, fontsize=6.1)
    for ax in axes:
        ax.tick_params(length=2,pad=2)
        ax.xaxis.labelpad=3; ax.yaxis.labelpad=3
        ax.grid(False)
    finish(fig, __file__, '6-5-boundaries', 'Fixed-release and read-count boundaries', layout=False)
    (folder/'model_ledger.json').write_text(json.dumps({'synthetic':True,
        'left_source':'e3_release_hold: unchanged causal cost and mock quality',
        'right_source':'e7_applicability: unchanged coefficients, fixed history and appends',
        'right_normalization':cfg_slice['normalization'], 'configurations':output},indent=2)+'\n')


if __name__=='__main__':
    main()
