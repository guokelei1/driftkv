"""Render five current synthetic figures, then rebuild the overview.

Data files are read as-is: this command never regenerates or overwrites data.
CPU only; no model, experiment result, or paper file is modified.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIGURES = [
    ('e1_quality_cost', '6_1', 'Quality versus complete added cost across four configurations.'),
    ('e10_releases', '6_2_releases', 'Consecutive releases (a, b) and users returning after skipped releases (c, d).'),
    ('e11_design', '6_3_design', 'State capacity versus read width (a) and candidate conditioning (b).'),
    ('e12_cost', '6_4_cost', 'Release processing (a), steady-state request latency (b) and first-read latency (c).'),
    ('e13_boundaries', '6_5_boundaries', 'Holding a release fixed (a) and read count per release (b).'),
]

if __name__ == '__main__':
    for folder, _, _ in FIGURES:
        subprocess.run([sys.executable, str(ROOT / folder / 'plot.py')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'make_overview.py')], check=True)
