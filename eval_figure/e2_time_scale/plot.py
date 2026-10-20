"""SIMULATED DATA: scale threshold and controlled-path table preview.

The paper uses a native LaTeX table. Thresholds derive from the retained
illustrative cost model. No measurement, seed or confidence interval is added.
The paper caption and file metadata identify the simulated data.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import finish, load_data
import matplotlib.pyplot as plt


def main():
    data = load_data(__file__)
    rows = []
    for name, cfg in zip(['Y-L', 'Y-Max', 'R-L', 'R-Max'], data['configurations']):
        fixed = sum(cfg['fixed_flops_percent_of_full_rebuild'].values())
        variable = sum(cfg['variable_flops_percent_of_full_rebuild'].values())
        crossing = 100 * fixed / (100 - variable)
        ratios = [cfg['path_cachetide_ms'][k] / cfg['path_reference_ms'][k]
                  for k in ['Prepare', 'Read', 'Append']]
        rows.append([name, f'{crossing:.1f}%', *[f'{r:.2f}' for r in ratios]])
    fig, ax = plt.subplots(figsize=(3.35, 1.15))
    ax.axis('off')
    table = ax.table(cellText=rows,
                     colLabels=['Config.', 'Break-even', 'Prepare', 'Score', 'Append'],
                     colWidths=[.23, .23, .18, .18, .18],
                     loc='center', cellLoc='right', colLoc='right')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.12)
    for (row, col), cell in table.get_celld().items():
        cell.visible_edges = ''
        cell.set_linewidth(.55)
        cell.PAD = .08
        if col == 0: cell.set_text_props(ha='left')
        if row == 0: cell.visible_edges = 'TB'
        if row == len(rows): cell.visible_edges = 'B'
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.02)
    finish(fig, __file__, '6-2', 'Scale and controlled-path table',
           paper_column=True, layout=False)


if __name__ == '__main__':
    main()
