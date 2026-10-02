#!/usr/bin/env python3
"""Paired AUC uncertainty on the larger unselected Max first-edge diagnostic sample."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_auc import OUT, ROOT, paired_cluster_bootstrap, sha


def main():
    path = ROOT / 'results/read_correction_2026_09/v5/max_diagnostics/random_evaluation_2048/max/v0_to_v1/summary.json'
    summary = json.loads(path.read_text())
    assert summary['status'] == 'complete' and summary['users'] == 2048
    frame, points = None, []
    sources = [{'path': str(path), 'sha256': sha(path)}]
    for point in summary['points']:
        descriptor = point['scores']
        score_path = Path(descriptor['path'])
        assert sha(score_path) == descriptor['sha256']
        sources.append({'path': str(score_path), 'sha256': descriptor['sha256']})
        data = pd.read_parquet(score_path).sort_values('request_id').reset_index(drop=True)
        columns = ['request_id', 'uid', 'label', 'full_logit', 'reuse_logit']
        if frame is None:
            frame = data[columns].copy()
        else:
            assert frame[columns].equals(data[columns])
        frame[point['tag']] = data.hstu_logit
        points.append({name: point[name] for name in ('tag', 'auc', 'full_auc', 'reuse_auc', 'delta_auc_vs_reuse', 'logit_mse_to_full')})
    methods = ['c128_original', 'c512_original']
    assert set(summary_point['tag'] for summary_point in summary['points']) == set(methods)
    records, draws = paired_cluster_bootstrap(frame, methods, 'reuse_logit')
    pd.DataFrame(records).to_csv(OUT / 'random2048_cluster_bootstrap.csv', index=False)
    pd.DataFrame([{'method': name, 'replicate': i, 'delta_auc': float(value)}
                  for name, values in draws.items() for i, value in enumerate(values)]).to_csv(
                      OUT / 'random2048_cluster_bootstrap_draws.csv', index=False)
    report = {'status': 'complete', 'scale': 'max', 'edge': 'v0_to_v1', 'users': 2048,
        'requests': len(frame), 'points': points, 'paired_bootstrap': records, 'sources': sources,
        'replicates': 400, 'seed': 17, 'script_sha256': sha(__file__), 'helper_sha256': sha(OUT / 'bootstrap_auc.py'),
        'reuse_logit_mse_to_full': float(np.mean((frame.reuse_logit - frame.full_logit) ** 2)),
        'role': 'conditional UID-cluster uncertainty on an unselected diagnostic sample; not training-seed replication'}
    (OUT / 'random2048_cluster_bootstrap.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame(records)[['method', 'delta_auc', 'ci95_low', 'ci95_high']].to_string(index=False))


if __name__ == '__main__':
    main()
