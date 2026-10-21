"""SIMULATED DATA: four configuration frontiers on a shallow two-column canvas.

frontier_data.json contains invented values, not measured results. The paper
caption and export metadata identify simulation; no watermark obscures artwork.
Historical data.json is retained for other preview figures and is not consumed.
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import COLORS, finish
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


# Keep system blue and reference gray consistent with the other figures.
# Baselines use both shape and dash pattern for grayscale readability.
METHODS = {
    'CacheTide': (COLORS['CacheTide'], 'o', '-'),
    'Layer-selective': ('#2D8B78', '^', '--'),
    'Tail': ('#A78735', 'v', ':'),
    'Deviation-guided': ('#8964A2', 'D', '-.'),
    'Query-guided': ('#BF7145', '>', (0, (4, 1, 1, 1))),
    'Query-only': ('#B65757', 's', (0, (3, 1))),
    'History-conditioned': ('#597C86', 'P', (0, (1, 1))),
}


def main():
    data = json.loads(Path(__file__).with_name('frontier_data.json').read_text())
    assert data['meta']['synthetic'] is True
    plt.rcParams.update({
        'font.size': 7.2, 'axes.titlesize': 7.5, 'axes.labelsize': 7.4,
        'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
        'legend.fontsize': 6.9,
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.05, 1.43))
    fig.subplots_adjust(left=.071, right=.995, bottom=.235, top=.675,
                        wspace=.38)

    for panel_index, (ax, cfg) in enumerate(zip(axes, data['configurations'])):
        panels = cfg['release_panels']
        reuse = np.mean([p['reuse_auc'] for p in panels])
        rebuild = np.mean([p['rebuild_auc'] for p in panels])
        for method, (color, marker, dash) in METHODS.items():
            runs = np.asarray([p['curves'][method] for p in panels])
            # Small translucent points are coordinates of synthetic release
            # observations, not confidence intervals or error bars.
            ax.scatter(runs[:, :, 0], runs[:, :, 1], color=color, marker=marker,
                       s=3.0, alpha=.24, linewidths=0, zorder=2)
            mean = runs.mean(axis=0)
            ax.plot(mean[:, 0], mean[:, 1], color=color, marker=marker,
                    ls=dash, lw=1.05 if method == 'CacheTide' else .8,
                    ms=2.4 if method == 'CacheTide' else 1.9, mew=.35,
                    zorder=5 if method == 'CacheTide' else 3)
            if method == 'CacheTide':
                x, y = mean[data['meta']['default_point_index']]
                ax.scatter(x, y, s=30, facecolors='none', edgecolors=color,
                           lw=.7, zorder=6)

        # References remain single mean operating points, never budget curves.
        ax.plot(0, reuse, marker='o', ls='', mfc='white', mec='#555D65',
                mew=.8, ms=3.1, zorder=6)
        ax.plot(100, rebuild, marker='s', ls='', color=COLORS['Rebuild'],
                ms=3.0, zorder=6)
        low = min(p['reuse_auc'] for p in panels)
        high = max(p['rebuild_auc'] for p in panels)
        span = high - low
        ax.set(xlim=(-5, 106), ylim=(low - .08*span, high + .08*span),
               xticks=[0, 50, 100])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.tick_params(length=2, pad=1.5)
        ax.grid(axis='y', lw=.35)
        ax.set_title(f"({chr(97 + panel_index)}) {cfg['dataset']}-{cfg['scale']}",
                     pad=3)
    axes[0].set_ylabel('AUC', labelpad=3)
    fig.text(.53, .025, 'Complete added FLOPs / Rebuild budget (%)',
             ha='center', va='bottom', fontsize=7.4)

    handles = [Line2D([], [], color=c, marker=m, ls=ls, lw=1.0, ms=2.7,
                      label=name)
               for name, (c, m, ls) in METHODS.items()]
    handles.extend([
        Line2D([], [], color='#555D65', marker='o', mfc='white', ls='',
               ms=3.2, label='Reuse'),
        Line2D([], [], color=COLORS['Rebuild'], marker='s', ls='',
               ms=3.1, label='Rebuild'),
    ])
    # Matplotlib fills legend columns first. Reorder so each row reads across
    # in method order; reference points finish the last row.
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4]
    fig.legend(handles=[handles[i] for i in order], ncol=5, loc='upper center',
               bbox_to_anchor=(.53, 1.015), handlelength=1.65,
               handletextpad=.4, columnspacing=1.1, labelspacing=.38,
               borderaxespad=0)
    finish(fig, __file__, '6-1', 'Four configuration quality-cost frontiers',
           layout=False)


if __name__ == '__main__':
    main()
