#!/usr/bin/env python3
"""Pair query-gap controls and retain the teacher candidate-signal scale."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat(value, draws):
    draws = np.asarray(draws)
    valid = np.isfinite(draws)
    return dict(mean=float(value) if np.isfinite(value) else None,
        ci95=np.percentile(draws[valid], [2.5, 97.5]).tolist() if valid.any() else None,
        invalid_bootstrap_draws=int((~valid).sum()))


def normalized_change(now_error, now_signal, old_error, old_signal, w):
    """Ratio of paired-user means, not mean of unstable per-UID ratios."""
    point = np.mean(now_error)/np.mean(now_signal) - np.mean(old_error)/np.mean(old_signal)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = (w @ now_error)/(w @ now_signal) - (w @ old_error)/(w @ old_signal)
    return stat(point, draws)


def analyze(root):
    root = Path(root)
    config = json.loads((root/'configuration.json').read_text())
    runner = json.loads((root/'summary.json').read_text())
    if (config['format'] != 'kv_query_gap_control_v1' or runner['status'] != 'complete'
            or runner['users'] != 2000 or config['diagnostic']):
        raise ValueError('requires complete fixed2000 non-diagnostic query-gap run')
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()
    rows = []
    for artifact in runner['artifacts']:
        p = root/artifact['path']
        block = json.loads(p.read_text())
        if (sha(p) != artifact['sha256'] or block['identity_sha256'] != identity
                or block['uids'] != config['users'][len(rows):len(rows)+len(block['users'])]
                or [u['uid'] for u in block['users']] != block['uids']):
            raise ValueError('query-gap block identity or members changed')
        rows.extend(block['users'])
    if [u['uid'] for u in rows] != config['users']:
        raise ValueError('query-gap population incomplete')
    if any(sha(p) != h for p,h in config['sources'].items()):
        raise ValueError('query-gap bound sources changed')
    snapshots = [{p['checkpoint']:p for p in u['snapshots']} for u in rows]
    points = []
    for name in ['cutover','append_8','append_32','append_128']:
        selected = [s for s in snapshots if name in s]
        n = len(selected);rng = np.random.default_rng(17)
        w = np.array([np.bincount(rng.integers(n,size=n), minlength=n) for _ in range(500)], dtype=np.uint16)
        point = dict(checkpoint=name, unique_users=n, mechanism_use_permitted=n>=512, gaps={})
        for gap in ['actual_gap','cutover_gap']:
            now = [s[name]['metrics'][gap]['f'] for s in selected]
            old = [s['cutover']['metrics'][gap]['f'] for s in selected]
            error = np.array([m['centered_mse'] for m in now])
            old_error = np.array([m['centered_mse'] for m in old])
            signal = np.array([m['teacher_candidate_variance'] for m in now])
            old_signal = np.array([m['teacher_candidate_variance'] for m in old])
            forced = np.array([s[name]['metrics'][gap]['teacher_forced']['centered_mse'] for s in selected])
            measured = dict(centered_error=stat(error.mean(), w@error/n),
                paired_absolute_change=stat((error-old_error).mean(), w@(error-old_error)/n),
                teacher_candidate_variance=stat(signal.mean(), w@signal/n),
                matched_cutover_teacher_variance=stat(old_signal.mean(), w@old_signal/n),
                centered_error_over_teacher_variance=stat(error.mean()/signal.mean(), (w@error)/(w@signal)),
                matched_cutover_normalized_error=stat(old_error.mean()/old_signal.mean(), (w@old_error)/(w@old_signal)),
                paired_normalized_change=normalized_change(error,signal,old_error,old_signal,w),
                teacher_forcing_centered_improvement=stat((error-forced).mean(), w@(error-forced)/n))
            point['gaps'][gap] = measured
        point['actual_gap_mean'] = float(np.mean([s[name]['actual_gap'] for s in selected]))
        point['cutover_gap_mean'] = float(np.mean([s[name]['cutover_gap'] for s in selected]))
        points.append(point)
    return dict(edge=config['edge'], users=2000, physical_gpu=1, checkpoints=points,
        confidence='500 paired UID-cluster bootstrap, conditional on this fixed F and edge',
        scope='fixed Q16 teacher agreement; candidate-centered error is not AUC or recommendation quality',
        limitations=['Actual and fixed-gap paths use identical evolving caches and candidate IDs, but different counterfactual query inputs.',
            'Teacher candidate variance is a scale reference; absolute centered-error growth alone need not mean worsened relative discrimination.',
            'History, FIFO eviction and propagated query state still change; fixing gap does not isolate one write operation.',
            'Pointwise exploratory intervals, not five-edge or training-seed confirmation.'],
        provenance=dict(configuration_sha256=sha(root/'configuration.json'), runner_summary_sha256=sha(root/'summary.json'),
                        analysis_source_sha256=sha(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();a = analyze(args.input);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(a,indent=2,allow_nan=False)+'\n')
    lines=[f"# Query-gap控制：edge{a['edge']}", '',
        '固定2000UID，各追加点与自身cutover配对。误差先去用户panel公共偏移；同时报告teacher候选方差尺度。', '',
        '| checkpoint | UID | gap | centered MSE | absolute Δ | normalized MSE | normalized Δ [95%] |',
        '|---|---:|---|---:|---:|---:|---:|']
    for p in a['checkpoints']:
        for gap,m in p['gaps'].items():
            d=m['paired_normalized_change']
            lines.append(f"| {p['checkpoint']} | {p['unique_users']} | {gap} | {m['centered_error']['mean']:.6g} | {m['paired_absolute_change']['mean']:.6g} | {m['centered_error_over_teacher_variance']['mean']:.6g} | {d['mean']:.6g} {d['ci95']} |")
    lines += ['',*[f'- {s}' for s in a['limitations']]]
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',edge=a['edge'],users=a['users'])))
