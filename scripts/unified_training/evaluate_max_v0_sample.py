#!/usr/bin/env python3
"""Frozen 10% UID diagnostic of the completed Max V0, using existing Full scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/unified_training_2026_09/max/seed17/v0_e14_users10pct'
DATA = ROOT / 'data/processed/yambda5b_max_200k_v1/scales/max'
MANIFEST = ROOT / 'data/manifests/yambda5b_max_200k_hstu_native_v1'
CHECKPOINT = ROOT / 'results/unified_training_2026_09/max/seed17/v0_1epoch_4gpu_b80_cpu14/checkpoint/checkpoint_100.pt'
NAMESPACE = 'evokv:max:v0:e14:uid10pct:seed17'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    uids = pq.read_table(DATA / 'users.parquet', columns=['uid'])['uid'].to_pylist()
    assert len(uids) == len(set(uids)) == 200000
    selected = sorted(uids, key=lambda u: (hashlib.sha256(f'{NAMESPACE}:{u}'.encode()).digest(), u))[:20000]
    pq.write_table(pa.table({'uid': sorted(selected)}), OUT / 'users.parquet')
    fidelity = pq.read_table(MANIFEST / 'requests_fidelity.parquet',
                            filters=[('query_timestamp', '>=', 217 * 86400),
                                     ('query_timestamp', '<', 231 * 86400)],
                            columns=['request_id', 'uid', 'target_known'])
    sampled = fidelity.filter(pc.is_in(fidelity['uid'], value_set=pa.array(selected)))
    known = sampled.filter(sampled['target_known'])
    assert len(set(known['request_id'].to_pylist())) == known.num_rows
    files = [CHECKPOINT, DATA / 'dataset.json', DATA / 'users.parquet', DATA / 'item_mapping.parquet',
             MANIFEST / 'requests_fidelity.parquet', MANIFEST / 'requests_quality.parquet',
             OUT / 'users.parquet', ROOT / 'scripts/evaluate_yambda500m_release_candidates_raw.py',
             ROOT / 'scripts/adjudicate_yambda500m_release_candidates.py', Path(__file__)]
    assert sha(CHECKPOINT) == json.loads(CHECKPOINT.with_name('checkpoint.seal.json').read_text())['checkpoint_sha256']
    dump(OUT / 'configuration.json', {
        'purpose': 'user_requested_10pct_baseline_quality_diagnostic_not_release_admission',
        'authorization': 'user explicitly requested evaluating a 10% user sample first',
        'sample_namespace': NAMESPACE, 'selection': 'lowest SHA256(namespace:raw_uid), before label access',
        'population_users': 200000, 'sampled_users': 20000,
        'active_known_target_users': len(set(known['uid'].to_pylist())),
        'known_requests': known.num_rows, 'all_sample_requests': sampled.num_rows,
        'oov_target_requests_excluded': sampled.num_rows - known.num_rows,
        'window_days_half_open': [217, 231], 'training_days_half_open': [0, 217],
        'primary_metric': 'existing request-pooled ROC_AUC, actual like/dislike labels',
        'max_history': 1024, 'world_size': 4, 'eval_batch_per_rank': 64,
        'larger_evaluation_automatically_authorized': False,
        'frozen_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'canary', 'evaluate'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
        print((OUT / 'configuration.json').read_text())
        return
    config = json.loads((OUT / 'configuration.json').read_text())
    for p, h in config['frozen_sha256'].items():
        assert sha(ROOT / p) == h, p
    if args.phase == 'evaluate':
        assert json.loads((OUT / 'canary.pass.json').read_text())['status'] == 'passed'
    work = OUT / args.phase
    work.mkdir(exist_ok=False)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
               'scripts/evaluate_yambda500m_release_candidates_raw.py', '--stage', 'max_v0_e14_users10pct_' + args.phase,
               '--block', 'matrix_horizon', '--training-block', 'foundation',
               '--manifest-dir', str(MANIFEST), '--dataset-manifest', str(DATA / 'dataset.json'),
               '--users-path', str(OUT / 'users.parquet'), '--parent', 'v0=' + str(CHECKPOINT),
               '--start-day', '217', '--end-day', '231', '--training-start-day', '0', '--training-end-day', '217',
               '--batch-size', '64', '--history-threads', '14', '--arrow-cpu-threads', '14',
               '--arrow-io-threads', '4', '--torch-cpu-threads', '4', '--cpu-affinity-by-rank',
               ';'.join(','.join(str(i) for i in range(r * 14, (r + 1) * 14)) for r in range(4)),
               '--output', str(work / 'raw')]
    if args.phase == 'canary':
        command += ['--max-users', '16']
    dump(work / 'command.json', command)
    started = time.perf_counter()
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'OMP_NUM_THREADS': '4'}
    with (work / 'run.log').open('x') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (work / 'exit_status.txt').write_text(str(result.returncode) + '\n')
    if result.returncode:
        raise RuntimeError(f'Inspect {work}/run.log')
    raw_path = work / 'raw/raw.parquet'
    seal = json.loads((work / 'raw/raw.seal.json').read_text())
    assert sha(raw_path) == seal['raw_sha256']
    raw = pq.read_table(raw_path)
    assert 'label' not in raw.column_names
    assert np.isfinite(raw['hstu_logit'].to_numpy()).all()
    assert len(set(raw['request_id'].to_pylist())) == raw.num_rows
    assert set(raw['uid'].to_pylist()) <= set(pq.read_table(OUT / 'users.parquet')['uid'].to_pylist())
    metrics = seal['execution_runtime']['peak_memory_by_rank']
    if args.phase == 'canary':
        assert max(raw['history_length'].to_pylist()) == 1024
        rate = raw.num_rows / max(r['evaluation_seconds'] for r in metrics)
        peak = max(r['peak_reserved_mib'] for r in metrics)
        assert peak <= 45490 * .85
        report = {'status': 'passed', 'requests': raw.num_rows, 'users': len(set(raw['uid'].to_pylist())),
                  'wall_seconds': time.perf_counter() - started, 'requests_per_second': rate,
                  'peak_reserved_mib': peak, 'quality_metrics_read': False,
                  'estimated_sample_inference_seconds': config['known_requests'] / rate}
        dump(OUT / 'canary.pass.json', report)
        print(json.dumps(report), flush=True)
        return
    assert raw.num_rows == config['known_requests']
    assert len(set(raw['uid'].to_pylist())) == config['active_known_target_users']
    # Join labels only after the complete raw prediction seal exists.
    labels = pq.read_table(MANIFEST / 'requests_quality.parquet',
                           filters=[('query_timestamp', '>=', 217 * 86400), ('query_timestamp', '<', 231 * 86400)],
                           columns=['request_id', 'label'])
    labels = labels.filter(pc.is_in(labels['request_id'], value_set=raw['request_id']))
    assert labels.num_rows == raw.num_rows
    pq.write_table(labels, work / 'labels.parquet')
    subprocess.run([sys.executable, 'scripts/adjudicate_yambda500m_release_candidates.py',
                    '--raw', str(raw_path), '--seal', str(work / 'raw/raw.seal.json'),
                    '--labels', str(work / 'labels.parquet'), '--output', str(work / 'adjudication.json')],
                   cwd=ROOT, env=env, check=True)
    report = json.loads((work / 'adjudication.json').read_text())
    dump(OUT / 'summary.json', {
        'status': 'completed_10pct_user_baseline_diagnostic', 'model': 'Max V0',
        'sampled_users': 20000, 'evaluated_users': config['active_known_target_users'],
        'requests': raw.num_rows, 'positive_labels': int(pc.sum(labels['label']).as_py()),
        'negative_labels': raw.num_rows - int(pc.sum(labels['label']).as_py()),
        'window_days_half_open': [217, 231], 'metrics': report['parent_absolute']['hstu_native'],
        'wall_seconds': time.perf_counter() - started, 'raw_sha256': seal['raw_sha256'],
        'checkpoint_sha256': config['frozen_sha256'][str(CHECKPOINT.relative_to(ROOT))],
        'sample_users_sha256': sha(OUT / 'users.parquet'), 'full_population_result': False,
        'peak_reserved_mib': max(r['peak_reserved_mib'] for r in metrics),
    })
    print((OUT / 'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
