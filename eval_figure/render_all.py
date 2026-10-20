"""Render eight synthetic assets, then rebuild the overview.

Data files are read as-is: this command never regenerates or overwrites data.
CPU only; no model, experiment result, or paper file is modified.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIGURES = [
    ('e1_quality_cost', '6_1', 'Twelve per-release quality versus complete-cost frontiers.'),
    ('e2_time_breakdown', '6_5_time', 'Complete processing-time attribution on matched resources.'),
    ('e3_lifecycle', '6_3', 'Default-point recovery with per-release quality gaps and costs.'),
    ('e4_read_structure', '6_4', 'Table: candidate conditioning and code-width tradeoffs.'),
    ('e5_calibration', '6_5', 'Discrete calibration quality--cost tradeoffs.'),
    ('e6_implementation', '6_6', 'Defined cumulative projection and fusion optimizations.'),
    ('e7_applicability', '6_7', 'Cost-model boundaries by history length and read density.'),
    ('e3_release_hold', '6_3_hold', 'One fixed release: quality and cumulative cost through two window turnovers.'),
]

if __name__ == '__main__':
    for folder, _, _ in FIGURES:
        subprocess.run([sys.executable, str(ROOT / folder / 'plot.py')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'e2_time_breakdown/storage_table.py')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'make_overview.py')], check=True)
