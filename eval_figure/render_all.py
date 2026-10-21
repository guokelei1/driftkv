"""Render six current synthetic figures, then rebuild the overview.

Data files are read as-is: this command never regenerates or overwrites data.
CPU only; no model, experiment result, or paper file is modified.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIGURES = [
    ('e1_quality_cost', '6_1', 'Four configuration quality versus complete-cost frontiers.'),
    ('e2_time_breakdown', '6_5_time', 'Complete processing-time attribution on matched resources.'),
    ('e3_lifecycle', '6_3', 'Default-point recovery with per-release quality gaps and costs.'),
    ('e8_mechanism', '6_3_design', 'Read structure and calibration mechanisms.'),
    ('e6_implementation', '6_6', 'Preparation reuse and fused-reading optimizations.'),
    ('e9_boundaries', '6_5_boundaries', 'Fixed-release continuation and read-count cost boundary.'),
]

if __name__ == '__main__':
    for folder, _, _ in FIGURES:
        subprocess.run([sys.executable, str(ROOT / folder / 'plot.py')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'e2_time_breakdown/storage_table.py')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'make_overview.py')], check=True)
