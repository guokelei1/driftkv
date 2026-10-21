"""Copy explicitly simulated evaluation figures into the paper asset tree.

Run render_all.py first. This script copies PDF/PNG only; it neither compiles
the manuscript nor launches experiments. Data and plotting sources stay here.
"""
from pathlib import Path
import argparse
import json
import shutil

ROOT = Path(__file__).resolve().parent
EXPORTS = [
    ('e1_quality_cost', '6_1', 'quality_cost'),
    ('e2_time_breakdown', '6_5_time', 'time_breakdown'),
    ('e3_lifecycle', '6_3', 'lifecycle'),
    ('e8_mechanism', '6_3_design', 'design_analysis'),
    ('e6_implementation', '6_6', 'implementation'),
    ('e9_boundaries', '6_5_boundaries', 'boundaries'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path,
                        default=ROOT.parent.parent / 'paper')
    paper = parser.parse_args().paper_dir.resolve()
    for folder, number, name in EXPORTS:
        data = json.loads((ROOT / folder / 'data.json').read_text())
        if data.get('meta', {}).get('synthetic') is not True:
            raise ValueError(f'{folder}: expected explicitly simulated data')
        for extension in ('pdf', 'png'):
            source = ROOT / folder / f'fig_{number}_simulated.{extension}'
            target = paper / 'pic/evaluation' / extension / (
                f'fig_eval_{name}_simulated.{extension}')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f'Exported {len(EXPORTS)} simulated figures (PDF and PNG) to {paper}/pic/evaluation/')


if __name__ == '__main__':
    main()
