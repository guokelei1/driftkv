#!/usr/bin/env python3
"""CPU-only descriptive diagnosis of retained request scores; never tunes or reruns models."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[5]
RESULTS = ROOT / 'results/read_correction_2026_09'
OUT = Path(__file__).resolve().parent
SCALES = ('medium', 'large', 'max')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(y, z):
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        return None
    return float((rankdata(z)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def metrics(frame, score):
    y = frame.label.to_numpy()
    uid = frame.uid.to_numpy()
    _, inv, counts = np.unique(uid, return_inverse=True, return_counts=True)
    full = frame.full_logit.to_numpy()
    reuse = frame.reuse_logit.to_numpy()
    pred = frame[score].to_numpy()
    err, reuse_err = pred - full, reuse - full
    delta, target = pred - reuse, full - reuse
    means = np.bincount(inv, weights=err) / counts
    reuse_means = np.bincount(inv, weights=reuse_err) / counts
    delta_means = np.bincount(inv, weights=delta) / counts
    target_means = np.bincount(inv, weights=target) / counts
    full_mean = np.bincount(inv, weights=full) / counts
    pred_mean = np.bincount(inv, weights=pred) / counts
    within_err = err - means[inv]
    within_reuse_err = reuse_err - reuse_means[inv]
    full_auc, reuse_auc, pred_auc = (auc(y, z) for z in (full, reuse, pred))
    output = {'requests': len(frame), 'uids': len(counts), 'positives': int(y.sum()),
        'full_auc': full_auc, 'reuse_auc': reuse_auc, 'method_auc': pred_auc,
        'auc_gain_over_reuse': None if pred_auc is None else pred_auc - reuse_auc,
        'recovery_percent': None if pred_auc is None or full_auc == reuse_auc else
                            100 * (pred_auc - reuse_auc) / (full_auc - reuse_auc),
        'mean_history_length': float(frame.history_length.mean()),
        'mean_inherited_fraction': float(frame.inherited_fraction.mean()),
        'mean_appends': float(frame.append_count_since_cutover.mean()),
        'method_error_mean': float(err.mean()), 'reuse_error_mean': float(reuse_err.mean()),
        'method_error_mse': float(np.mean(err ** 2)), 'reuse_error_mse': float(np.mean(reuse_err ** 2)),
        'method_between_uid_error_mse': float(np.mean(means[inv] ** 2)),
        'reuse_between_uid_error_mse': float(np.mean(reuse_means[inv] ** 2)),
        'method_within_uid_error_mse': float(np.mean(within_err ** 2)),
        'reuse_within_uid_error_mse': float(np.mean(within_reuse_err ** 2)),
        'correction_mean': float(delta.mean()), 'target_correction_mean': float(target.mean()),
        'correction_rms': float(np.sqrt(np.mean(delta ** 2))),
        'target_correction_rms': float(np.sqrt(np.mean(target ** 2))),
        'correction_target_correlation': corr(delta, target),
        'correction_uid_mean_correlation': corr(delta_means, target_means),
        'correction_within_uid_correlation': corr(delta - delta_means[inv], target - target_means[inv]),
        'method_full_uid_mean_correlation': corr(pred_mean, full_mean),
        'correction_positive_fraction': float(np.mean(delta > 0)),
        'method_error_abs_p95': float(np.quantile(np.abs(err), .95)),
        'method_mse_better_request_fraction': float(np.mean(err ** 2 < reuse_err ** 2))}
    # Equal-UID and pair-weighted AUC, restricted to users with both labels in this bin.
    # No training/evaluation labels are used to choose the bins or algorithm.
    slim = pd.DataFrame({'uid': uid, 'y': y, 'full': full, 'reuse': reuse, 'method': pred})
    sizes = slim.groupby('uid', sort=False).y.agg(['sum', 'count'])
    pairs = sizes['sum'] * (sizes['count'] - sizes['sum'])
    eligible = pairs > 0
    output['uids_with_both_labels'] = int(eligible.sum())
    for name in ('full', 'reuse', 'method'):
        ranks = slim.groupby('uid', sort=False)[name].rank(method='average')
        positive_ranks = (ranks * slim.y).groupby(slim.uid, sort=False).sum()
        numerator = positive_ranks - sizes['sum'] * (sizes['sum'] + 1) / 2
        values = numerator[eligible] / pairs[eligible]
        output[f'{name}_macro_uid_auc'] = float(values.mean()) if len(values) else None
        output[f'{name}_within_uid_pair_auc'] = float(numerator[eligible].sum() / pairs[eligible].sum()) if len(values) else None
    return output


def groups(frame):
    yield 'overall', 'all', np.ones(len(frame), dtype=bool)
    frac = frame.inherited_fraction.to_numpy()
    for lo, hi, label in [(0, 0, 'zero'), (0, .25, '(0,.25]'), (.25, .5, '(.25,.5]'),
                          (.5, .75, '(.5,.75]'), (.75, 1, '(.75,1)'), (1, 1, 'one')]:
        mask = frac == lo if lo == hi else (frac > lo) & (frac <= hi)
        if label == '(.75,1)':
            mask &= frac < 1
        yield 'inherited_fraction', label, mask
    for field, bins in [
        ('append_count_since_cutover', [(0, 0, '0'), (1, 255, '1-255'), (256, 511, '256-511'),
                                        (512, 1023, '512-1023'), (1024, np.inf, '1024+')]),
        ('history_length', [(0, 128, '<=128'), (129, 511, '129-511'), (512, 1023, '512-1023'),
                            (1024, 1024, '1024')]),
        ('days_since_cutover', [(0, 3, 'days0-3'), (3, 7, 'days3-7'), (7, 10, 'days7-10'),
                                (10, np.inf, 'days10+')]),
        ('evaluation_requests_per_uid', [(1, 4, '1-4'), (5, 8, '5-8'), (9, 16, '9-16'),
                                          (17, np.inf, '17+')])]:
        x = frame[field].to_numpy()
        for lo, hi, label in bins:
            mask = (x >= lo) & (x < hi if field == 'days_since_cutover' else x <= hi)
            yield field, label, mask
    # Joint split distinguishes inherited producer mix from short-history extrapolation.
    full = frame.history_length.to_numpy() == 1024
    for history_label, history_mask in [('short', ~full), ('full1024', full)]:
        for producer_label, producer_mask in [('all_parent', frac == 1), ('mixed', (frac > 0) & (frac < 1)),
                                               ('no_parent', frac == 0)]:
            yield 'history_x_producer', f'{history_label}/{producer_label}', history_mask & producer_mask
    # Restrict to the same UIDs observed both before and after an append. This
    # checks user-composition confounding; it is still not a causal intervention.
    for history_label, history_mask in [('anyhistory', np.ones(len(frame), dtype=bool)), ('full1024', full)]:
        pure = history_mask & (frac == 1)
        mixed = history_mask & (frac > 0) & (frac < 1)
        shared = np.intersect1d(frame.loc[pure, 'uid'], frame.loc[mixed, 'uid'])
        common = frame.uid.isin(shared).to_numpy()
        yield 'paired_transition_uid', f'{history_label}/all_parent', pure & common
        yield 'paired_transition_uid', f'{history_label}/mixed', mixed & common


def main():
    started = time.perf_counter()
    output, source_records, overlaps = [], [], {}
    for edge_num in range(1, 6):
        edge = f'v{edge_num - 1}_to_v{edge_num}'
        scale_uids = {}
        for scale in SCALES:
            panel_path = ROOT / 'results/selective_recompute_2026_09/panels' / scale / edge / 'binding.json'
            panel = json.loads(panel_path.read_text())
            frame, methods = None, []
            for version in ('v5', 'v4'):
                summary_path = RESULTS / version / 'population_run/evaluation' / scale / edge / 'summary.json'
                summary = json.loads(summary_path.read_text())
                assert summary['status'] == 'complete' and summary['users'] == 3000 and not summary['probe_only']
                source_records.append({'path': str(summary_path), 'sha256': sha(summary_path)})
                for tag, descriptor in summary['scores'].items():
                    name = f'q{tag.rsplit("c", 1)[-1]}' if version == 'v5' else f'h_{tag}'
                    path = Path(descriptor['path'])
                    assert sha(path) == descriptor['sha256']
                    source_records.append({'path': str(path), 'sha256': descriptor['sha256']})
                    scores = pd.read_parquet(path).sort_values('request_id').reset_index(drop=True)
                    if frame is None:
                        frame = scores.drop(columns=['hstu_logit', 'correction_flops']).copy()
                    else:
                        for column in frame.columns.intersection(scores.columns):
                            assert np.array_equal(frame[column].to_numpy(), scores[column].to_numpy()), (scale, edge, name, column)
                    frame[name] = scores.hstu_logit.to_numpy()
                    methods.append(name)
            assert frame.uid.nunique() == 3000
            scale_uids[scale] = set(frame.uid)
            frame['inherited_fraction'] = frame.inherited_count / frame.history_length
            frame['days_since_cutover'] = (frame.query_timestamp - int(panel['cutover'])) / 86400
            frame['evaluation_requests_per_uid'] = frame.groupby('uid').uid.transform('count')
            for axis, bin_label, mask in groups(frame):
                if not mask.any():
                    continue
                subset = frame.loc[mask]
                for method in methods:
                    output.append({'scale': scale, 'edge': edge, 'method': method, 'axis': axis,
                        'bin': bin_label, 'panel_requests': len(frame), **metrics(subset, method)})
            print(f'{scale}/{edge}: {len(frame)} requests, {len(output)} diagnostic rows', flush=True)
        overlaps[edge] = {f'{a}/{b}': len(scale_uids[a] & scale_uids[b])
                          for a, b in [('medium', 'large'), ('medium', 'max'), ('large', 'max')]}
    table = pd.DataFrame(output)
    table.to_csv(OUT / 'bucket_metrics.csv', index=False)
    table.loc[table.axis == 'overall'].to_csv(OUT / 'overall_metrics.csv', index=False)
    manifest = {'role': 'descriptive offline diagnosis only; no model, calibration, or evaluation rerun',
        'source_script': {'path': str(Path(__file__)), 'sha256': sha(__file__)}, 'sources': source_records,
        'source_parquet_hashes_verified': True, 'paired_request_fields_exact': True,
        'scales': SCALES, 'edges': 15, 'users_per_edge': 3000, 'methods': methods,
        'metric_definitions': {
            'recovery_percent': '100*(method AUC - Reuse AUC)/(Full AUC - Reuse AUC), descriptive only; raw AUC also retained',
            'error': 'method logit minus current Full logit, in raw logit units',
            'between_uid_error_mse': 'request-weighted squared per-UID mean error within the selected bucket',
            'within_uid_error_mse': 'request-weighted squared error after subtracting per-UID mean error in this bucket',
            'decomposition': 'between_uid_error_mse + within_uid_error_mse = error_mse exactly',
            'correction': 'method logit minus Reuse logit; target correction = Full minus Reuse',
            'macro_uid_auc': 'equal-UID AUC among users with both labels in this bucket; eligible UID count retained',
            'within_uid_pair_auc': 'AUC restricted to same-UID positive/negative request pairs, pair-weighted',
            'activity': 'number of evaluated requests for each UID on this fixed panel; descriptive, not causal scheduling',
            'paired_transition_uid': 'same UIDs observed in both pure inherited and mixed states; query/time differences remain uncontrolled',
            'time_bins': 'days since recorded release cutover, disjoint half-open bins',
            'cross_scale_comparison': 'model-specific fixed panels, not a paired same-user architecture intervention'},
        'cross_scale_uid_overlap_counts': overlaps,
        'elapsed_seconds': time.perf_counter() - started,
        'outputs': {name: sha(OUT / name) for name in ('bucket_metrics.csv', 'overall_metrics.csv')}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'complete': True, 'diagnostic_rows': len(table), 'elapsed_seconds': manifest['elapsed_seconds']}))


if __name__ == '__main__':
    main()
