#!/usr/bin/env python3
"""Paired UID-cluster bootstrap of retained request AUCs; CPU only."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc_plan(labels, scores):
    order = np.argsort(scores, kind='stable')
    ordered = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    return order, starts, labels[order].astype(np.float64)


def weighted_auc(plan, weights):
    order, starts, y = plan
    weights = weights[order]
    positive = np.add.reduceat(weights * y, starts)
    negative = np.add.reduceat(weights * (1 - y), starts)
    total_pairs = positive.sum() * negative.sum()
    if not total_pairs:
        return np.nan
    return float(np.sum(positive * (np.cumsum(negative) - .5 * negative)) / total_pairs)


def paired_cluster_bootstrap(frame, columns, baseline, *, repeats=400, seed=17):
    """Each sampled UID contributes all its rows, with multiplicity; same draw for every score."""
    labels = frame.label.to_numpy()
    _, inverse = np.unique(frame.uid.to_numpy(), return_inverse=True)
    users = int(inverse.max()) + 1
    plans = {name: auc_plan(labels, frame[name].to_numpy()) for name in [baseline, *columns]}
    point = {name: weighted_auc(plan, np.ones(len(frame))) for name, plan in plans.items()}
    generator = np.random.default_rng(seed)
    draws = {name: np.empty(repeats) for name in columns}
    for index in range(repeats):
        counts = np.bincount(generator.integers(users, size=users), minlength=users)
        weights = counts[inverse]
        base = weighted_auc(plans[baseline], weights)
        for name in columns:
            draws[name][index] = weighted_auc(plans[name], weights) - base
    results = []
    for name, values in draws.items():
        assert np.isfinite(values).all()
        lo, hi = np.quantile(values, [.025, .975])
        results.append({'method': name, 'baseline': baseline, 'requests': len(frame), 'uids': users,
            'method_auc': point[name], 'baseline_auc': point[baseline], 'delta_auc': point[name] - point[baseline],
            'ci95_low': float(lo), 'ci95_high': float(hi), 'bootstrap_mean_delta_auc': float(values.mean()),
            'bootstrap_std_delta_auc': float(values.std(ddof=1)),
            'bootstrap_fraction_negative': float(np.mean(values < 0)), 'replicates': repeats, 'seed': seed})
    return results, draws


def check_formula():
    y = np.array([1, 0, 1, 0, 0, 1])
    score = np.array([.2, .2, .5, .1, .5, .5])
    weights = np.array([2, 1, 3, 2, 1, 4])
    yy, zz = np.repeat(y, weights), np.repeat(score, weights)
    pos, neg = int(yy.sum()), int((1 - yy).sum())
    explicit = (rankdata(zz)[yy == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    assert abs(weighted_auc(auc_plan(y, score), weights) - explicit) < 1e-14


def main():
    check_formula()
    started, records, sources, draw_rows = time.perf_counter(), [], [], []
    for edge in ('v0_to_v1', 'v3_to_v4'):
        summary_path = ROOT / 'results/read_correction_2026_09/v5/population_run/evaluation/max' / edge / 'summary.json'
        summary = json.loads(summary_path.read_text())
        frame = None
        for budget in (128, 512):
            tag = f'query_pure_cross_phi_joint8_c{budget}'
            item = summary['scores'][tag]
            path = Path(item['path'])
            assert sha(path) == item['sha256']
            sources.append({'path': str(path), 'sha256': item['sha256']})
            data = pd.read_parquet(path).sort_values('request_id').reset_index(drop=True)
            if frame is None:
                frame = data[['request_id', 'uid', 'label', 'reuse_logit']].copy()
            else:
                paired = ['request_id', 'uid', 'label', 'reuse_logit']
                assert frame[paired].equals(data[paired])
            frame[f'q{budget}'] = data.hstu_logit
        results, draws = paired_cluster_bootstrap(frame, ['q128', 'q512'], 'reuse_logit')
        for item in results:
            records.append({'scale': 'max', 'edge': edge, **item})
        for method, values in draws.items():
            draw_rows.extend({'edge': edge, 'method': method, 'replicate': i, 'delta_auc': float(value)}
                             for i, value in enumerate(values))
    pd.DataFrame(records).to_csv(OUT / 'auc_cluster_bootstrap.csv', index=False)
    pd.DataFrame(draw_rows).to_csv(OUT / 'auc_cluster_bootstrap_draws.csv', index=False)
    report = {'status': 'complete', 'resampling_unit': 'UID, retaining all requests with multiplicity',
        'paired_resampling': True, 'confidence_interval': 'percentile 2.5%..97.5%, 400 draws, seed17',
        'role': 'conditional request-panel uncertainty; not model-training-seed replication or population qualification',
        'records': records, 'sources': sources, 'script_sha256': sha(__file__),
        'formula_check': 'weighted tie-aware AUC matches explicit integer-replicated sample',
        'elapsed_seconds': time.perf_counter() - started}
    (OUT / 'auc_cluster_bootstrap.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['records'], indent=2))


if __name__ == '__main__':
    main()
