"""SIMULATED DATA: compact two-panel, single-column implementation comparison.

Simulation is disclosed in the manuscript caption. All original values remain.
Conversion and auxiliary-space data are presented in paper tables.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import finish, load_data
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = load_data(__file__)
    plt.rcParams.update({'font.size': 7.4, 'axes.titlesize': 7.4,
                         'axes.labelsize': 7.3, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 7})
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.7), sharey=True)
    # Only the three defined cumulative stages belong in the paper.  The
    # retained JSON's fourth "Full" point has no specified added mechanism.
    stages = ['Base', '+P', '+F']
    x = np.arange(len(stages))
    axes[0].bar(x, data['path_relative_time']['Prepare'][:3], width=.65,
                color='#1764AB', edgecolor='white', linewidth=.35)
    axes[0].set_title('(a) Preparation', pad=4)
    width = .34
    for index, (key, label, color) in enumerate([
            ('Read', 'Score', '#2D8B78'), ('Append', 'Append', '#B65757')]):
        axes[1].bar(x+(index-.5)*width, data['path_relative_time'][key][:3],
                    width=width, color=color, label=label,
                    edgecolor='white', linewidth=.35)
    axes[1].set_title('(b) Score / append', pad=4)
    for ax in axes:
        ax.set_xticks(x, stages)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, .5, 1])
        ax.grid(axis='y')
        ax.tick_params(axis='both', pad=2, length=2.5)
    axes[0].set_ylabel('Time / base', labelpad=3)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.57, 1.01),
               ncol=2, handlelength=1.3, columnspacing=1.2, fontsize=7)
    fig.subplots_adjust(left=.17, right=.985, top=.74, bottom=.235, wspace=.22)
    finish(fig, __file__, '6-6', '', paper_column=True, layout=False)


if __name__ == '__main__':
    main()
