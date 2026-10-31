#!/usr/bin/env python3
"""Check full panel membership, request coverage and frozen checkpoint bindings."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import pyarrow.parquet as pq
from experiments.archive.code_structure.evaluate import PANEL, REFERENCE
from hstu_kvcache.evaluation.panels import load_panel
from hstu_kvcache.common.artifacts import sha256, write_json


def run(args):
    _, groups, _ = load_panel(PANEL)
    root = ROOT / 'results/code_structure_2026_10'
    for layout in args.layouts:
        directory = root / layout
        summary = json.loads((directory / 'quality4096/summary.json').read_text())
        if set(summary['selected_uids']) != set(groups['evaluation']) or len(summary['selected_uids']) != 4096:
            raise RuntimeError('final user panel differs')
        scores = pq.read_table(directory / 'quality4096/scores.parquet')
        reference = pq.read_table(REFERENCE, columns=['request_id', 'uid', 'version', 'label'])
        keys = ['request_id', 'uid', 'version', 'label']
        if scores.num_rows != 51731 or not scores.select(keys).sort_by('request_id').equals(reference.sort_by('request_id')):
            raise RuntimeError('final request panel differs')
        frozen = json.loads((root / 'frozen_candidates.json').read_text())['candidates'][layout]
        weights = {v: sha256(directory / 'calibration' / v / 'calibration.pt') for v in ['v3', 'v4', 'v5']}
        if any(weights[v] != frozen[v]['weights_sha256'] for v in weights):
            raise RuntimeError('candidate weights changed after calibration')
        for v in weights:
            meta = json.loads((directory / 'calibration' / v / 'calibration.json').read_text())
            if meta['weights_sha256'] != weights[v] or (set(meta['fit_uids']) | set(meta['validation_uids'])) & set(groups['evaluation']):
                raise RuntimeError('weight metadata or user roles differ')
        write_json(directory / 'quality4096/final_checks.json', dict(status='passed',
            users=4096, requests=51731, releases=['v3', 'v4', 'v5'], weights_sha256=weights,
            exact_user_membership=True, exact_request_identities=True, weights_unchanged=True,
            causal_calibration_roles_disjoint=True))
        print(json.dumps(dict(layout=layout, status='passed', users=4096, requests=51731)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layouts', nargs='+', required=True)
    run(p.parse_args())
