#!/usr/bin/env python3
"""Bounded Max warm-start and Full evaluation launch canary."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXECUTION = ROOT / 'configs/unified_training_2026_09/max_v1_epochs12_4gpu_b80_cpu14_execution.yaml'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def main():
    global EXECUTION
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-config", type=Path, default=EXECUTION)
    EXECUTION = parser.parse_args().execution_config.resolve()
    e = yaml.safe_load(EXECUTION.read_text())
    cp = ROOT / e['frozen_parent']['contract']
    c = yaml.safe_load(cp.read_text())
    assert sha(cp) == e['frozen_parent']['contract_sha256']
    for k, p in c['frozen_inputs'].items():
        if not k.endswith('_sha256'):
            assert sha(ROOT / p) == c['frozen_inputs'][k + '_sha256'], k
    version = c['scope']['versions'][0]
    parent_version = c['scope']['expected_parent_version']
    start, end = c['scope']['training_days_half_open']
    epochs = c['training']['passes']
    initial_epochs = int(c['training'].get('initial_window_epochs', 0))
    evaluation_parent_version = c['evaluation'].get('parent_version', parent_version)
    evaluation_parent = ROOT / c['frozen_inputs'][f'parent_{evaluation_parent_version}_checkpoint']
    out = ROOT / c['outputs']['root']
    out.mkdir(parents=True, exist_ok=True)
    parity_path = out / 'backend_parity.json'
    backend_scope = e.get('backend_validation_scope', 'all_reported_modes')
    all_backend_checks_passed = None
    if e.get('attention_backend') == 'auto':
        parity = json.loads(parity_path.read_text())
        all_backend_checks_passed = parity['passed']
        assert parity['checkpoint_sha256'] == c['frozen_inputs'][f'parent_{parent_version}_checkpoint_sha256']
        if backend_scope == 'bf16_train_fp32_inference':
            assert c['execution_backend']['numerical_gate'] == backend_scope
            assert c['training']['compute_dtype'] == 'bf16'
            assert parity['precisions']['torch.bfloat16']['passed']
            fp32 = parity['precisions']['torch.float32']
            assert fp32['triton_calls'] == c['model']['num_layers']
            assert all(fp32[key]['finite'] and fp32[key]['relative_l2'] < fp32['relative_l2_tolerance']
                       for key in ['hidden', 'logits', 'loss'])
        else:
            assert backend_scope == 'all_reported_modes' and parity['passed']
        write(out / 'backend_runtime_gate.json', {
            'passed': True, 'scope': backend_scope,
            'parity_report_sha256': sha(parity_path),
            'checkpoint_sha256': parity['checkpoint_sha256'],
            'all_reported_modes_passed': parity['passed'],
            'required_paths': (['BF16_forward_backward', 'FP32_forward_inference']
                               if backend_scope == 'bf16_train_fp32_inference' else ['all_reported_modes']),
            'unused_FP32_backward_passed': parity['precisions']['torch.float32']['passed'],
            'tolerances_unchanged': True,
            'limitation': 'A failed unused FP32 backward comparison remains in the original report; this gate does not qualify FP32 training.',
        })
    probe = out / 'canary'
    probe.mkdir(exist_ok=False)
    manifest = (ROOT / c['frozen_inputs']['request_manifest']).parent
    parent = ROOT / c['frozen_inputs'][f'parent_{parent_version}_checkpoint']
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CUDA_VISIBLE_DEVICES': '0,1,2,3',
           'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1',
           'EVOKV_ATTENTION_BACKEND': e.get('attention_backend', 'auto')}
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    runtime = []
    for flag, key in [('history-threads', 'history_threads'), ('arrow-cpu-threads', 'arrow_cpu_threads'),
                      ('arrow-io-threads', 'arrow_io_threads'), ('torch-cpu-threads', 'torch_cpu_threads'),
                      ('cpu-affinity-by-rank', 'cpu_affinity_by_rank')]:
        runtime += ['--' + flag, str(e['training_runtime'][key])]

    def run(name, command):
        write(probe / (name + '.command.json'), command)
        with (probe / (name + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (probe / (name + '.exit_status.txt')).write_text(str(result.returncode) + '\n')
        if result.returncode:
            raise RuntimeError(f'{name} canary failed; see {probe}')

    train = [*distributed, 'scripts/train_yambda500m_foundation_fsdp.py', '--version', version, '--branch', 'D14',
             '--launch-contract', str(cp), '--execution-contract', str(EXECUTION), '--manifest-dir', str(manifest),
             '--training-block', 'matrix_horizon', '--parent', str(parent), '--output', str(probe / 'train'),
             '--oov-buckets', '256', '--passes', str(epochs), '--global-batch-size', '80',
             '--train-start-day', str(start), '--train-end-day', str(end), '--canary-steps', '12', *runtime]
    run('train', train)
    torch.set_num_threads(4)
    checkpoint = probe / 'train/checkpoint_100.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['version'] == version and payload['parent_checkpoint_sha256'] == sha(parent)
    assert payload['training_day_range'] == [start, end] and payload['learning_rate'] == 5e-5
    assert payload['initial_window_epochs'] == initial_epochs
    assert payload['training_epoch_target'] == initial_epochs + epochs
    assert payload['local_training_epochs_completed'] == 0
    assert payload['optimizer_reset_at_start'] == c['training'].get('optimizer_reset_at_start', False)
    assert all(torch.isfinite(t).all().item() for t in payload['model'].values())
    del payload
    r = json.loads((probe / 'train/train_result.json').read_text())
    assert r['steps'] == 12 and r['local_batch_sizes_by_rank'] == [20] * 4
    assert all(x['canary_batch_width_min'] == 1024 for x in r['rank_metrics'])
    train_peak = max(x['peak_reserved_mib'] for x in r['rank_metrics'])
    assert train_peak <= 45490 * .85, train_peak
    run('evaluate', [*distributed, 'scripts/evaluate_yambda500m_release_candidates_raw.py',
                    '--stage', f'max_{version}_resource_canary', '--block', 'matrix_horizon',
                    '--training-block', 'matrix_horizon', '--manifest-dir', str(manifest),
                    '--dataset-manifest', str(ROOT / c['frozen_inputs']['dataset_manifest']),
                    '--parent', evaluation_parent_version + '=' + str(evaluation_parent), '--current', 'probe_e1=' + str(checkpoint),
                    *(['--current', 'probe_e2=' + str(checkpoint)] if epochs == 2 else []),
                    '--start-day', str(end), '--end-day', str(end + 14),
                    '--training-start-day', str(start), '--training-end-day', str(end), '--batch-size', '64',
                    '--max-users', '32', '--allow-canary-checkpoints', '--output', str(probe / 'raw'), *runtime])
    seal = json.loads((probe / 'raw/raw.seal.json').read_text())
    assert sha(probe / 'raw/raw.parquet') == seal['raw_sha256']
    raw = pq.read_table(probe / 'raw/raw.parquet').to_pandas()
    assert np.isfinite(raw.hstu_logit).all() and 'label' not in raw.columns
    assert not raw.duplicated(['request_id', 'model_name']).any()
    scores = raw.pivot(index='request_id', columns='model_name', values='hstu_logit')
    assert scores.notna().all().all()
    if epochs == 2:
        assert np.array_equal(scores.probe_e1, scores.probe_e2)
    eval_metrics = seal['execution_runtime']['peak_memory_by_rank']
    eval_peak = max(x['peak_reserved_mib'] for x in eval_metrics)
    assert eval_peak <= 45490 * .85, eval_peak
    spec = importlib.util.spec_from_file_location('trainer', ROOT / 'scripts/train_yambda500m_foundation_fsdp.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    q = manifest / 'requests_fidelity.parquet'
    u = pq.read_table(q, filters=[('query_timestamp', '>=', start * 86400), ('query_timestamp', '<', end * 86400),
                                 ('target_known', '=', True)], columns=['uid'])['uid'].to_numpy()
    uids, counts = np.unique(u, return_counts=True)
    assignment = trainer.balanced_uid_assignment(uids, counts, 4)
    loads = [0] * 4
    for uid, n in zip(uids, counts):
        loads[assignment[int(uid)]] += int(n)
    steps = max(math.ceil(n / 20) for n in loads)
    eval_rows = pq.read_table(q, filters=[('query_timestamp', '>=', end * 86400), ('query_timestamp', '<', (end + 14) * 86400),
                                         ('target_known', '=', True)], columns=['uid'])
    eval_rate = len(scores) / max(x['evaluation_seconds'] for x in eval_metrics)
    train_seconds = r['median_synchronized_step_seconds'] * 1.15
    pure_train = epochs * steps * train_seconds / 3600
    pure_eval = eval_rows.num_rows / eval_rate / 3600
    write(out / 'resource_estimate.json', {
        'formal_training_steps': epochs * steps, 'steps_per_pass': steps, 'unique_training_requests': len(u),
        'training_users': len(uids), 'rank_training_requests': loads, 'global_batch_size': 80,
        'probe_step_seconds': r['median_synchronized_step_seconds'], 'planning_step_seconds': train_seconds,
        'pure_training_hours': pure_train, 'full_E14_requests': eval_rows.num_rows,
        'eval_canary_requests_per_second': eval_rate, 'pure_evaluation_hours_estimate': pure_eval,
        'planning_total_hours': [round(pure_train + pure_eval + .5, 1), round(pure_train + pure_eval + 2, 1)],
        'basis': 'exact balanced step count; measured current-backend median plus 15%; measured Full evaluator; loading/saves/adjudication allowance; short-probe estimate only',
        'peak_canary_reserved_mib': train_peak, 'peak_eval_canary_reserved_mib': eval_peak,
        'retained_formal_checkpoints': [f'checkpoint_epoch_{initial_epochs + n}.pt' for n in c['training']['checkpoint_epochs']],
        'initial_window_epochs': initial_epochs,
        'additional_complete_epochs': epochs,
        'retain_final_optimizer_and_rng': bool(c['training'].get('recovery', {}).get('retain_after_completion')),
    })
    code = ['scripts/train_yambda500m_foundation_fsdp.py', 'scripts/evaluate_yambda500m_release_candidates_raw.py',
            'scripts/evaluate_yambda500m_foundation_raw.py', 'scripts/adjudicate_yambda500m_release_candidates.py',
            'scripts/unified_training/run_medium_v2_2epoch.py', 'scripts/unified_training/prepare_max_v1.py',
            'src/hstu_kvcache/training/foundation.py', 'src/hstu_kvcache/training/recovery.py',
            'src/hstu_kvcache/data/yambda_history.py', 'src/hstu_kvcache/data/oov.py']
    code += [str(p.relative_to(ROOT)) for p in (ROOT / 'src/hstu_kvcache/models').glob('*.py')]
    code += [str(p.relative_to(ROOT)) for p in (ROOT / 'src/hstu_kvcache/evaluation').glob('*.py')]
    if parity_path.exists():
        code += [str(parity_path.relative_to(ROOT)), 'scripts/unified_training/check_max_backend.py',
                 'scripts/benchmark_hstu_backends.py']
        code += [str((out / 'backend_runtime_gate.json').relative_to(ROOT))]
    write(out / 'canary.pass.json', {
        'passed': True, 'contract_sha256': sha(cp), 'execution_contract_sha256': sha(EXECUTION),
        'training_steps': 12, 'training_all_batch_widths': 1024, 'parent_lineage_verified': True,
        'evaluation_requests': len(scores), 'duplicate_current_logits_equal': True if epochs == 2 else None,
        'quality_metrics_read': False, 'probe_checkpoint_sha256': sha(checkpoint),
        'initial_window_epochs': initial_epochs,
        'cumulative_epoch_target': initial_epochs + epochs,
        'backend_validation_scope': backend_scope,
        'all_reported_backend_checks_passed': all_backend_checks_passed,
        'required_execution_paths_passed': True,
        'final_recovery_scope': 'Formal endpoint only; no optimizer recovery payload retained by this short canary',
        'code_sha256': {p: sha(ROOT / p) for p in code},
        'created_at_unix': time.time(),
    })
    checkpoint.unlink()
    print((out / 'resource_estimate.json').read_text(), flush=True)
    print('Canary passed; disposable probe weights removed. Formal launch not performed by this script.', flush=True)


if __name__ == '__main__':
    main()
