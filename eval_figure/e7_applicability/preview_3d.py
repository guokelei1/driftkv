"""SIMULATED cost-model layout comparison, not a new measured result.

Evaluate the existing formula on a joint history/read-load grid. This is NOT
interpolation of two one-dimensional slices and introduces no new coefficients.
Keep outputs outside manuscript assets while deciding whether 3D helps reading.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image, ImageDraw, ImageFont
from plot import costs, load_data


def main():
    d = load_data(__file__)
    # This existing configuration spans both positive and negative savings in
    # the declared ranges, making it useful for checking boundary readability.
    cfg = next(c for c in d['configurations']
               if c['dataset'] == 'RecFlow' and c['scale'] == 'Large')
    nh = np.linspace(7, 11, 17)
    nr = np.linspace(0, 8, 17)
    H, R = np.meshgrid(nh, nr)
    history, reads = 2**H, 2**R
    ledger = costs(cfg, d['scenario'], history, reads)
    saved = 100*(1-ledger['cachetide_total_flops']/ledger['rebuild_total_flops'])
    cmap = LinearSegmentedColormap.from_list('cost_saving',
                ['#B48749', '#F8F8F5', '#B5D0E3', '#1764AB'])
    norm = TwoSlopeNorm(vmin=-45, vcenter=0, vmax=90)
    plt.rcParams.update({'font.size': 7.2, 'axes.labelsize': 7.2,
                         'xtick.labelsize': 6.7, 'ytick.labelsize': 6.7})
    out = Path('/home/gkl/work/paper/build/e7_3d_comparison_20261020')
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in ['surface', 'contour']:
        fig = plt.figure(figsize=(3.35, 2.55))
        if kind == 'surface':
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(H, R, saved, cmap=cmap, norm=norm,
                            rstride=1, cstride=1, linewidth=.28,
                            edgecolor='#B0B7BC', antialiased=True, shade=False)
            ax.view_init(elev=25, azim=-130)
            ax.set_box_aspect((1.15, 1.05, .78))
            ax.set(xlim=(7, 11), ylim=(0, 8), zlim=(-45, 90))
            ax.set_xticks([7, 9, 11], ['128', '512', '2048'])
            ax.set_yticks([0, 4, 8], ['1', '16', '256'])
            ax.set_zticks([-40, 0, 40, 80])
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.pane.fill = False
                axis._axinfo['grid'].update({'color': '#DBE0E4', 'linewidth': .4})
            ax.tick_params(pad=0)
            # Figure-coordinate labels stay readable in a single-column PDF;
            # 3D auto-label placement can otherwise put them outside the canvas.
            fig.text(.70, .055, 'History length', ha='center', rotation=22,
                     fontsize=7.2)
            fig.text(.23, .09, 'Reads / behavior', ha='center', rotation=-31,
                     fontsize=7.2)
            fig.text(.025, .54, 'FLOPs saved (%)', va='center', rotation=90,
                     fontsize=7.2)
            fig.subplots_adjust(left=.13, right=.99, bottom=.21, top=.98)
        else:
            ax = fig.add_subplot()
            im = ax.pcolormesh(H, R, saved, cmap=cmap, norm=norm, shading='auto')
            levels = ax.contour(H, R, saved, levels=[0, 20, 40, 60, 80],
                                colors='#57636D', linewidths=.55)
            ax.clabel(levels, fmt=lambda v: f'{v:.0f}%', fontsize=6.7,
                      inline=True, inline_spacing=2)
            ax.set(xlim=(7, 11), ylim=(0, 8), xlabel='History length',
                   ylabel='Reads / behavior')
            ax.set_xticks([7, 9, 11], ['128', '512', '2048'])
            ax.set_yticks([0, 2, 4, 6, 8], ['1', '4', '16', '64', '256'])
            bar = fig.colorbar(im, ax=ax, ticks=[-40, 0, 40, 80], fraction=.06, pad=.05)
            bar.set_label('FLOPs saved (%)', labelpad=4)
            bar.outline.set_visible(False)
            ax.tick_params(length=2.5)
            fig.subplots_adjust(left=.16, right=.86, bottom=.20, top=.96)
        for ext in ['png', 'pdf']:
            meta = ({'Title': 'SIMULATED cost model: 3D versus 2D layout'}
                    if ext == 'pdf' else {'Description': 'SIMULATED formula-based layout comparison only.'})
            fig.savefig(out/f'{kind}.{ext}', dpi=300, bbox_inches='tight',
                        pad_inches=.03, metadata=meta)
        plt.close(fig)
        paths.append(out/f'{kind}.png')
    w = 850
    imgs = [Image.open(p).convert('RGB') for p in paths]
    height = max(round(im.height*w/im.width) for im in imgs)
    sheet = Image.new('RGB', (2*w+60, height+135), 'white')
    draw = ImageDraw.Draw(sheet)
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    f = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 21)
    draw.text((20, 8), 'E7 · RecFlow-Large · 同一联合网格（模拟成本模型，不是实测结果）',
              font=small, fill='#5C6974')
    for i, (im, title, note) in enumerate(zip(imgs,
            ['A  3D曲面', 'B  二维等高线＋颜色'],
            ['高度展示两种条件共同作用', '同样三个变量，直接读数与计算边界'])):
        x = 20+i*(w+20)
        draw.text((x+10, 48), title, font=f, fill='#283B49')
        draw.text((x+10, 84), note, font=small, fill='#5C6974')
        im = im.resize((w, round(im.height*w/im.width)), Image.Resampling.LANCZOS)
        sheet.paste(im, (x, 125))
    sheet.save(out/'comparison.png')
    (out/'grid.json').write_text(json.dumps({
        'synthetic': True, 'source': str(Path(__file__).with_name('data.json')),
        'configuration': 'RecFlow-Large',
        'grid_is_evaluated_from_existing_cost_formula': True,
        'history_length': history.tolist(), 'reads_per_behavior': reads.tolist(),
        'total_flops_saved_percent': saved.tolist()}, indent=2)+'\n')
    print(out/'comparison.png')


if __name__ == '__main__':
    main()
