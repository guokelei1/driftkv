#!/usr/bin/env python3
"""Paired CPU-only analysis of completed Max calibration counterfactuals."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_auc import OUT, ROOT, auc_plan, paired_cluster_bootstrap, sha, weighted_auc


def run(edge):
    path = ROOT / 'results/read_correction_2026_09/v5/max_diagnostics/fitting_evaluation/max' / edge / 'summary.json'
    summary = json.loads(path.read_text())
    assert summary['status'] == 'complete'
    sources = [{'path': str(path), 'sha256': sha(path)}]
    points, frame = [], None
    for item in summary['points']:
        descriptor = item['scores']
        score_path = Path(descriptor['path'])
        assert sha(score_path) == descriptor['sha256']
        sources.append({'path': str(score_path), 'sha256': descriptor['sha256']})
        data = pd.read_parquet(score_path).sort_values('request_id').reset_index(drop=True)
        columns = ['request_id', 'uid', 'label', 'full_logit', 'reuse_logit']
        if frame is None:
            frame = data[columns].copy()
        else:
            assert frame[columns].equals(data[columns])
        tag = item['tag']
        frame[tag] = data.hstu_logit
        error = data.hstu_logit - data.full_logit
        reuse_error = data.reuse_logit - data.full_logit
        user_mean_error = error.groupby(data.uid).transform('mean')
        reuse_user_mean_error = reuse_error.groupby(data.uid).transform('mean')
        mse = float(np.mean(error ** 2))
        assert abs(mse - item['logit_mse_to_full']) < 1e-9
        computed_auc = weighted_auc(auc_plan(data.label.to_numpy(), data.hstu_logit.to_numpy()), np.ones(len(data)))
        assert abs(computed_auc - item['auc']) < 1e-12
        points.append({'edge': edge, 'method': tag, 'requests': len(data), 'uids': data.uid.nunique(),
            **{name: item[name] for name in ('full_auc', 'reuse_auc', 'auc', 'delta_auc_vs_reuse', 'recovery_percent')},
            'method_logit_mse_to_full': mse, 'reuse_logit_mse_to_full': float(np.mean(reuse_error ** 2)),
            'method_error_mean': float(error.mean()), 'reuse_error_mean': float(reuse_error.mean()),
            'method_between_uid_error_mse': float(np.mean(user_mean_error ** 2)),
            'reuse_between_uid_error_mse': float(np.mean(reuse_user_mean_error ** 2)),
            'method_within_uid_error_mse': float(np.mean((error - user_mean_error) ** 2)),
            'reuse_within_uid_error_mse': float(np.mean((reuse_error - reuse_user_mean_error) ** 2))})
    results, draw_rows = [], []
    for budget in (128, 512):
        methods = [f'{kind}_c{budget}_joint' for kind in ('weighted', 'mixed')]
        for baseline in (f'original_c{budget}', 'reuse_logit'):
            records, draws = paired_cluster_bootstrap(frame, methods, baseline)
            results.extend({'edge': edge, 'budget': budget, **row} for row in records)
            for method, values in draws.items():
                draw_rows.extend({'edge': edge, 'method': method, 'baseline': baseline,
                    'replicate': i, 'delta_auc': float(v)} for i, v in enumerate(values))
    prefix = f'fitting_{edge}'
    pd.DataFrame(points).to_csv(OUT / f'{prefix}_all_points.csv', index=False)
    pd.DataFrame(results).to_csv(OUT / f'{prefix}_bootstrap.csv', index=False)
    pd.DataFrame(draw_rows).to_csv(OUT / f'{prefix}_bootstrap_draws.csv', index=False)
    report = {'status': 'complete', 'edge': edge, 'points': points, 'paired_bootstrap': results,
        'source_rows_exactly_paired': True, 'sources': sources, 'script_sha256': sha(__file__),
        'helper_sha256': sha(OUT / 'bootstrap_auc.py'), 'replicates': 400, 'seed': 17,
        'error_type': 'final model logit error relative to Full; not hidden-layer read-vector error',
        'role': 'diagnostic UID-cluster uncertainty conditional on fixed corrections and user panel',
        'limitations': ['All raw/joint/original/targeted-layer-removal points retained; bootstrap focuses on two prospective joint candidates.',
                        'No training-seed repetition and no multiple-comparison adjustment.']}
    (OUT / f'{prefix}_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame(results)[['edge', 'method', 'baseline', 'delta_auc', 'ci95_low', 'ci95_high']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edge', required=True, choices=['v0_to_v1', 'v3_to_v4'])
    run(parser.parse_args().edge)
