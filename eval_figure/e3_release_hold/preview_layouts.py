"""Compare layouts using the UNCHANGED SIMULATED hold-release data.

These previews are not experiment results. Simulation is stated in the sheet
header and export metadata, outside the paper artwork, per the user's format.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot import cost_trajectory, draw_trend, load_data, COLORS


BLUE = COLORS['CacheTide']
COST = '#B7AA8B'
INK = '#454F59'


def series(data):
    q = data['quality_preview']
    n = data['scenario']['window_length']
    a = np.asarray(q['appends'], dtype=int)
    rb = np.asarray(q['auc']['Rebuild'])
    reuse = np.asarray(q['auc']['Reuse'])
    rec = 100*(np.asarray(q['auc']['CacheTide'])-reuse)/(rb-reuse)
    cost = cost_trajectory(data)['added_percent'][a]
    return a/n, rec, cost


def setup():
    plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'axes.linewidth': .65})


def tidy(ax):
    ax.tick_params(length=2.5, pad=3, color='#737D87')
    for spine in ax.spines.values():
        spine.set_color('#737D87')
    ax.grid(False)


def trajectory(data):
    """Ekya Fig3 inspired cost-quality plane, sharing the final renderer."""
    return draw_trend(data, cost_trajectory(data))[0]


def barline(data):
    """AssyLLM Fig20: separate bars and line, a common append coordinate."""
    setup()
    x, r, c = series(data)
    keep = np.array([0, 3, 4, 5, 6, 7, 8])
    fig, ax = plt.subplots(figsize=(3.35, 2.30))
    fig.subplots_adjust(left=.15, right=.84, bottom=.21, top=.84)
    cost_ax = ax.twinx()
    ax.set_zorder(cost_ax.get_zorder()+1)
    ax.patch.set_visible(False)
    cost_ax.bar(x[keep], c[keep], width=.105, color=COST, edgecolor='#8A7D61',
                lw=.45, zorder=1)
    ax.plot(x, r, color=BLUE, marker='o', ms=3.1, lw=1.5, zorder=3)
    ax.set(xlim=(-.12, 2.12), ylim=(0, 120), yticks=[0, 50, 100],
           xticks=[0, .5, 1, 1.5, 2], xticklabels=['0', '.5', '1', '1.5', '2'],
           xlabel='Appends / initial window', ylabel='Quality recovery (%)')
    cost_ax.set(ylim=(0, 120), yticks=[0, 50, 100],
                ylabel='Cumulative added cost (%)')
    ax.yaxis.label.set_color(BLUE)
    cost_ax.yaxis.label.set_color('#85764F')
    ax.tick_params(axis='y', labelcolor=BLUE)
    cost_ax.tick_params(axis='y', labelcolor='#85764F')
    handles = [Line2D([], [], color=BLUE, marker='o', ms=3, lw=1.3, label='Recovery'),
               Patch(facecolor=COST, edgecolor='#8A7D61', lw=.45, label='Added cost')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.02),
               ncol=2, handlelength=1.6, columnspacing=1.4)
    cost_ax.spines['right'].set_visible(True)
    tidy(ax)
    tidy(cost_ax)
    return fig


def aligned_bars(data):
    """AWStream Fig12: aligned stage rows, no invented distributions."""
    setup()
    x, r, c = series(data)
    keep = np.array([0, 3, 4, 5, 6, 8])
    y = np.arange(len(keep))
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(3.35, 2.30))
    fig.subplots_adjust(left=.16, right=.97, bottom=.18, top=.85, wspace=.25)
    for ax, vals, col in zip(axes, [r[keep], c[keep]], [BLUE, COST]):
        ax.barh(y, vals, height=.5, color=col)
        for yy, v in zip(y, vals):
            ax.text(v+3, yy, f'{v:.0f}', va='center', fontsize=6.7, color=INK)
        ax.set(xlim=(0, 130), xticks=[0, 50, 100])
        tidy(ax)
    axes[0].set_yticks(y, [f'{xx:g}N' for xx in x[keep]])
    axes[0].invert_yaxis()
    axes[0].set_ylabel('Appends', labelpad=2)
    axes[1].tick_params(axis='y', length=0)
    axes[0].set_title('Recovery (%)', color=BLUE, fontsize=7.5, pad=5)
    axes[1].set_title('Added cost (%)', color='#85764F', fontsize=7.5, pad=5)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('/home/gkl/work/paper/build/hold_trends_20261020'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_data(__file__)
    paths = []
    for name, draw in [('A_trajectory', trajectory), ('B_barline', barline),
                       ('C_aligned_bars', aligned_bars)]:
        fig = draw(data)
        for ext in ['png', 'pdf']:
            path = args.out/f'{name}.{ext}'
            meta = {'Title': 'Simulated no-update trend layout'} if ext == 'pdf' else {'Description': 'Simulated layout only'}
            fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.025, metadata=meta)
        plt.close(fig)
        paths.append(args.out/f'{name}.png')
    tiles = [Image.open(p).convert('RGB') for p in paths]
    w = 800
    h = max(round(t.height*w/t.width) for t in tiles)
    sheet = Image.new('RGB', (3*w+80, h+145), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 26)
    small = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 22)
    draw.text((22, 10), '同一组模拟数据 · 三种结构（仅排版比较）', font=small, fill='#606A74')
    for j, (t, title, sub) in enumerate(zip(tiles,
           ['A  单条演化轨迹', 'B  追加量上的柱＋线', 'C  共同阶段的对齐横条'],
           ['成本与恢复率直接配对，点旁标追加量', '保留追加横轴，两种图元分工', '逐行比较质量与开销，无曲线交叉'])):
        ox = 20+j*(w+20)
        draw.text((ox+10, 52), title, font=font, fill='#263746')
        draw.text((ox+10, 89), sub, font=small, fill='#606A74')
        t = t.resize((w, round(t.height*w/t.width)), Image.Resampling.LANCZOS)
        sheet.paste(t, (ox, 135))
    sheet.save(args.out/'comparison.png')
    print(args.out/'comparison.png')


if __name__ == '__main__':
    main()
