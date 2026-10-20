"""Shared style for evaluation layout previews. All current data are synthetic."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
# TrueType Chinese outlines also render correctly after PDF composition.
FONT = Path('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
CJK = font_manager.FontProperties(fname=str(FONT)) if FONT.exists() else None
COLORS = {
    'CacheTide': '#1764AB', 'Rebuild': '#303840', 'Reuse': '#969DA4',
    'Static map': '#B65757', 'Partial rebuild': '#2D8B78',
    'Budget rebuild': '#B58735',
}
DATASET_COLORS = {'Yambda': '#1764AB', 'RecFlow': '#8964A2'}
SCALE_STYLES = {'Large': {'linestyle': '-', 'marker': 'o'},
                'Max': {'linestyle': '--', 'marker': 's'}}
MARKERS = {'CacheTide': 'o', 'Rebuild': 's', 'Reuse': 'o',
           'Static map': 'D', 'Partial rebuild': '^', 'Budget rebuild': 'v'}
STAGE_NAMES = ['Calibration', 'Shared tables', 'State preparation',
               'Behavior writes', 'Scoring', 'Transfers']
STAGE_COLORS = ['#B9A5D0', '#E2C779', '#80AACD', '#79B7A2', '#DC9691', '#BDC3C8']

def setup():
    plt.rcParams.update({
        'font.family': ['DejaVu Sans', 'WenQuanYi Micro Hei'],
        'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 7.7, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'lines.linewidth': 1.65,
        'lines.markersize': 4, 'axes.axisbelow': True,
        'grid.color': '#E4E8EB', 'grid.linewidth': .55,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })

def load_data(script):
    path = Path(script).resolve().with_name('data.json')
    data = json.loads(path.read_text())
    if data.get('meta', {}).get('synthetic') is not True:
        raise ValueError('This preview expects explicitly labeled synthetic data.')
    return data

def panel(ax, letter, title, ylabel=None, xlabel=None):
    ax.set_title(f'({letter}) {title}', loc='left', pad=8)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)
    ax.grid(axis='y')

def finish(fig, script, number, title, note='', top=.76, bottom=.21, right=.985,
           layout=True, paper_column=False):
    """Export clean artwork; the manuscript caption labels simulated values.

    By user request, no theme title, provenance watermark, or explanatory
    footnote is drawn inside the chart. Provenance remains in metadata/files.
    """
    if layout:
        fig.subplots_adjust(left=.09, right=right, bottom=bottom, top=top, wspace=.37)
    folder = Path(script).resolve().parent
    stem = f'fig_{number.replace("-", "_")}_simulated'
    for suffix in ['png', 'pdf']:
        metadata = ({'Title': title, 'Subject': 'Simulated data: layout preview only'}
                    if suffix == 'pdf' else {'Description': 'Simulated data: layout preview only'})
        fig.savefig(folder / f'{stem}.{suffix}', dpi=300, bbox_inches='tight',
                    pad_inches=.015, metadata=metadata)
    plt.close(fig)
    print(folder / f'{stem}.png')

setup()
