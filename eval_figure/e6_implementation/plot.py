"""SIMULATED DATA: preparation reuse and fused reading, one panel each.

New preview values are declared in optimization_preview; historical timing
arrays remain intact. Captions and metadata identify simulation.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import finish, load_data
import matplotlib.pyplot as plt
import numpy as np


def main():
    d = load_data(__file__)['optimization_preview']
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.3,
                         'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'legend.fontsize': 6.1})
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.58), sharey=True)
    fig.subplots_adjust(left=.15, right=.99, bottom=.23, top=.69, wspace=.25)
    x = np.arange(3)
    bottom = np.zeros(3)
    for (name, values), color in zip(d['preparation']['components'].items(),
                                    ['#1764AB', '#D5B86E', '#BDC3C8']):
        axes[0].bar(x, values, bottom=bottom, color=color, width=.58,
                    edgecolor='white', lw=.35,
                    label={'Row work':'Rows', 'Table build':'Build', 'Transfers':'Move'}[name])
        bottom += np.array(values)
    axes[0].set_xticks(x, d['preparation']['variants'])
    axes[0].set_title('(a) Preparation', loc='left', pad=4)
    for xi, total in zip(x, bottom):
        axes[0].text(xi, total+.035, f'{total:.2f}', ha='center', fontsize=6.1)
    width = .34
    for j, (key, label, color) in enumerate([
            ('score', 'Score', '#2D8B78'), ('append', 'Append', '#B65757')]):
        axes[1].bar(x+(j-.5)*width, d['reading'][key], width=width,
                    color=color, edgecolor='white', lw=.35, label=label)
    axes[1].set_xticks(x, ['Base', 'Fused', '+Disp.'])
    axes[1].set_title('(b) Reading', loc='left', pad=4)
    # Each panel owns its legend: preparation stages are distinct from
    # the score/append series in the reading comparison.
    for ax, ncol in zip(axes, [3, 2]):
        ax.legend(loc='upper left', bbox_to_anchor=(-.025, 1.60),
                  ncol=ncol, handlelength=.9, handletextpad=.3,
                  columnspacing=.55, borderaxespad=0, fontsize=6.2)
    for ax in axes:
        ax.set(ylim=(0, 1.2), yticks=[0, .5, 1])
        ax.tick_params(length=2, pad=2)
        ax.grid(axis='y', lw=.4)
    axes[0].set_ylabel('Time / reference', labelpad=3)
    finish(fig, __file__, '6-6', 'Implementation mechanism analysis', layout=False)


if __name__ == '__main__':
    main()
