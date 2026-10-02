#!/usr/bin/env python3
"""UID-cluster uncertainty for completed small Max interventions and random-user controls."""
from pathlib import Path
import json

import pandas as pd

from bootstrap_auc import OUT, ROOT, paired_cluster_bootstrap, sha


def main():
    records, sources, all_draws = [], [], []
    for branch in ('interventions', 'random_evaluation'):
        for edge in ('v0_to_v1', 'v3_to_v4'):
            path = ROOT / 'results/read_correction_2026_09/v5/max_diagnostics' / branch / 'max' / edge / 'summary.json'
            summary = json.loads(path.read_text())
            assert summary['status'] == 'complete'
            sources.append({'path': str(path), 'sha256': sha(path)})
            by_tag = {point['tag']: point['scores'] for point in summary['points']}
            for budget in (128, 512):
                original = f'c{budget}_original'
                methods = [f'c{budget}_{name}' for name in ('drop_constant', 'late_only')] if branch == 'interventions' else [original]
                tags = list(dict.fromkeys([original, *methods]))
                frame = None
                for tag in tags:
                    descriptor = by_tag[tag]
                    score_path = Path(descriptor['path'])
                    assert sha(score_path) == descriptor['sha256']
                    sources.append({'path': str(score_path), 'sha256': descriptor['sha256']})
                    data = pd.read_parquet(score_path).sort_values('request_id').reset_index(drop=True)
                    paired = ['request_id', 'uid', 'label', 'reuse_logit']
                    if frame is None:
                        frame = data[paired].copy()
                    else:
                        assert frame[paired].equals(data[paired])
                    frame[tag] = data.hstu_logit
                baseline = original if branch == 'interventions' else 'reuse_logit'
                estimates, draws = paired_cluster_bootstrap(frame, methods, baseline)
                for row in estimates:
                    records.append({'branch': branch, 'scale': 'max', 'edge': edge, 'budget': budget, **row})
                for method, values in draws.items():
                    all_draws.extend({'branch': branch, 'edge': edge, 'method': method, 'baseline': baseline,
                                      'replicate': i, 'delta_auc': float(value)} for i, value in enumerate(values))
    pd.DataFrame(records).to_csv(OUT / 'intervention_cluster_bootstrap.csv', index=False)
    pd.DataFrame(all_draws).to_csv(OUT / 'intervention_cluster_bootstrap_draws.csv', index=False)
    result = {'status': 'complete', 'records': records, 'sources': sources, 'script_sha256': sha(__file__),
        'helper_sha256': sha(OUT / 'bootstrap_auc.py'), 'replicates': 400, 'seed': 17,
        'convention': 'paired UID-cluster percentile 95% CI, retaining each sampled user\'s entire request set',
        'limitations': ['Conditional on the completed 256-UID diagnostic samples and frozen corrections.',
                        'Not a training-seed replication or an independent population qualification.',
                        'No adjustment for multiple descriptive intervention comparisons.']}
    (OUT / 'intervention_cluster_bootstrap.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame(records)[['branch', 'edge', 'method', 'baseline', 'delta_auc', 'ci95_low', 'ci95_high']].to_string(index=False))


if __name__ == '__main__':
    main()
