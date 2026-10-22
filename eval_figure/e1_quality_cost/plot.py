"""SIMULATED DATA: four configuration frontiers on a shallow two-column canvas.

frontier_data.json contains invented values, not measured results. The paper
caption and export metadata identify simulation; no watermark obscures artwork.
Five methods are drawn: CacheTide, the two correction alternatives (query-only
and per-release cache translation), and the two entry-selection baselines.
Layer-selective, tail and per-request history-conditioned curves remain in the
data file for the motivation study; add them to METHODS to draw them here.
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


# System blue and reference grays match the other figures. Baselines differ in
# shape and dash pattern as well as color for grayscale readability.
METHODS = {
    'CacheTide': (COLORS['CacheTide'], 'o', '-'),
    'Cache translation': ('#2D8B78', 'P', (0, (5, 1.5))),
    'Query-only': ('#B65757', 's', (0, (2.5, 1))),
    'Deviation-guided': ('#8964A2', 'D', '-.'),
    'Query-guided': ('#BF7145', '>', (0, (4, 1, 1, 1))),
}
REUSE_COLOR = '#7F8890'


def main():
    data = json.loads(Path(__file__).with_name('frontier_data.json').read_text())
    assert data['meta']['synthetic'] is True
    plt.rcParams.update({
        'font.size': 7.2, 'axes.titlesize': 7.5, 'axes.labelsize': 7.4,
        'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
        'legend.fontsize': 6.9,
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.05, 1.36))
    fig.subplots_adjust(left=.071, right=.995, bottom=.25, top=.76,
                        wspace=.38)

    for panel_index, (ax, cfg) in enumerate(zip(axes, data['configurations'])):
        panels = cfg['release_panels']
        reuse = np.mean([p['reuse_auc'] for p in panels])
        rebuild = np.mean([p['rebuild_auc'] for p in panels])
        # Reuse and Rebuild are single operating points. Thin horizontal
        # guides make the gap between them, the quantity every curve
        # recovers, visible across the whole budget axis.
        ax.axhline(rebuild, color=COLORS['Rebuild'], lw=.55, ls=(0, (4, 2)),
                   zorder=1)
        ax.axhline(reuse, color=REUSE_COLOR, lw=.55, ls=(0, (1, 1.5)),
                   zorder=1)
        for method, (color, marker, dash) in METHODS.items():
            runs = np.asarray([p['curves'][method] for p in panels])
            mean = runs.mean(axis=0)
            system = method == 'CacheTide'
            ax.plot(mean[:, 0], mean[:, 1], color=color, marker=marker,
                    ls=dash, lw=1.15 if system else .85,
                    ms=2.6 if system else 2.1, mew=.35,
                    zorder=5 if system else 3)
            if system:
                x, y = mean[data['meta']['default_point_index']]
                ax.scatter(x, y, s=30, facecolors='none', edgecolors=color,
                           lw=.7, zorder=6)
        ax.plot(0, reuse, marker='o', ls='', mfc='white', mec=REUSE_COLOR,
                mew=.8, ms=3.1, zorder=6)
        ax.plot(100, rebuild, marker='s', ls='', color=COLORS['Rebuild'],
                ms=3.0, zorder=6)
        span = rebuild - reuse
        ax.set(xlim=(-5, 106), ylim=(reuse - .12*span, rebuild + .12*span),
               xticks=[0, 50, 100])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.tick_params(length=2, pad=1.5)
        ax.grid(False)
        ax.set_title(f"({chr(97 + panel_index)}) {cfg['dataset']}-{cfg['scale']}",
                     pad=3)
    axes[0].set_ylabel('AUC', labelpad=3)
    fig.text(.53, .025, 'Complete added FLOPs / Rebuild budget (%)',
             ha='center', va='bottom', fontsize=7.4)

    handles = [Line2D([], [], color=c, marker=m, ls=ls, lw=1.0, ms=2.7,
                      label=name)
               for name, (c, m, ls) in METHODS.items()]
    handles.extend([
        Line2D([], [], color=REUSE_COLOR, marker='o', mfc='white',
               ls=(0, (1, 1.5)), lw=.6, ms=3.2, label='Reuse'),
        Line2D([], [], color=COLORS['Rebuild'], marker='s', ls=(0, (4, 2)),
               lw=.6, ms=3.1, label='Rebuild'),
    ])
    fig.legend(handles=handles, ncol=len(handles), loc='upper center',
               bbox_to_anchor=(.53, 1.005), handlelength=2.0,
               handletextpad=.4, columnspacing=1.15, borderaxespad=0)
    finish(fig, __file__, '6-1', 'Four configuration quality-cost frontiers',
           layout=False)


if __name__ == '__main__':
    main()
