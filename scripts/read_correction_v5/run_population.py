#!/usr/bin/env python3
"""Four-GPU, resumable v5 calibration/evaluation queue inside tmux."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path('/home/gkl/work/evokv')
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from read_correction_v5.common import PANEL_ROOT, configuration, edge_name, sha256, sources, write_json
from read_correction_v5.calibrate import directory

ADMISSION_ERRORS = {'RuntimeError: GPU lacks initial free-memory reserve',
                    'RuntimeError: GPU lacks the configured initial free-memory reserve'}
EXPECTED_EDGES = {(s, e) for s in ('medium', 'large', 'max') for e in range(1, 6)}


def now():
    return datetime.now(timezone.utc).isoformat()


def candidates(plan, root, scale, edge):
    rows = []
    for method, key in (('query_only', 'query'), ('history_conditioned', 'history')):
        setting = plan[key]
        if key == 'history' and setting.get('reuse_v4'):
            continue
        feature = setting['feature_mode'] if key == 'query' else 'nonlinear'
        state = setting['state_mode']
        if setting.get('mix_profile', 'balanced') == 'old_heavy':
            state += '_old_heavy'
        if key == 'query' and setting.get('joint_epochs', 0):
            feature += f'_joint{setting["joint_epochs"]}'
        for budget in setting['budgets']:
            folder = directory(root / 'calibration', method, state, feature, budget, scale, edge)
            variants = [setting.get('variant', feature)] if key == 'query' else ['map_all', 'map_old_prefix']
            for variant in variants:
                rows.append({'tag': f'{key}_{state}_{variant}_c{budget}', 'method': method,
                    'variant': variant, 'calibration_dir': str(folder)})
    return rows


def finished(root, scale, edge, entries, source_hashes):
    path = root / 'evaluation' / scale / edge_name(edge) / 'summary.json'
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if (report['status'] != 'complete' or report['probe_only'] or report['users'] != 3000
            or report['partition'] != 'evaluation' or report['tags'] != [entry['tag'] for entry in entries]
            or report['inputs']['execution_sources'] != source_hashes):
        raise RuntimeError(f'completed edge differs from the frozen plan: {path}')
    panel = PANEL_ROOT / scale / edge_name(edge) / 'binding.json'
    if report['inputs']['panel_sha256'] != sha256(panel):
        raise RuntimeError('completed edge panel changed')
    for entry in entries:
        evidence = report['inputs']['policies'][entry['tag']]['calibration']
        if sha256(Path(evidence['path'])) != evidence['sha256']:
            raise RuntimeError('completed edge calibration changed')
        if sha256(Path(entry['calibration_dir']) / 'calibration.pt') != evidence['weights_sha256']:
            raise RuntimeError('completed edge correction weights changed')
    for output in report['scores'].values():
        if sha256(Path(output['path'])) != output['sha256']:
            raise RuntimeError('completed edge scores changed')
    return {'scale': scale, 'edge': edge_name(edge), 'users': report['users'],
        'requests': report['requests'], 'points': report['points'],
        'summary': {'path': str(path), 'sha256': sha256(path)}}


def run(root, figure_dir):
    if not os.environ.get('TMUX'):
        raise RuntimeError('run the authorized population queue inside tmux')
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    cfg, source_hashes, plan_hash = configuration(), sources(), sha256(plan_path)
    if plan['execution_sources'] != source_hashes or plan['settings'] != cfg:
        raise RuntimeError('execution sources or common settings differ from the fixed plan')
    if len(plan['edges']) != 15 or {tuple(e) for e in plan['edges']} != EXPECTED_EDGES:
        raise RuntimeError('plan must cover all 15 adjacent edges exactly once')
    if sorted(plan['gpus']) != [0, 1, 2, 3] or len(plan['gpus']) != 4:
        raise RuntimeError('plan must use GPU 0/1/2/3 once each')
    if cfg['memory_fraction'] > .70 or cfg['initial_free_fraction'] < .75:
        raise RuntimeError('memory limits differ from the authorized 70% cap / 75% admission')
    methods = ['query']
    if plan['history'].get('reuse_v4'):
        if plan['history']['budgets']:
            raise ValueError('reuse_v4 retains completed H results and must not request new H budgets')
    else:
        methods.append('history')
    if any(plan[m]['state_mode'] not in ('pure', 'mixed') for m in methods):
        raise ValueError('expected pure or genuinely mixed calibration state')
    if plan['query']['feature_mode'] not in ('head_phi', 'cross_phi'):
        raise ValueError('one prospective Q feature must be fixed')
    for method in methods:
        profile = plan[method].get('mix_profile', 'balanced')
        if profile not in ('balanced', 'old_heavy') or (profile == 'old_heavy' and plan[method]['state_mode'] != 'mixed'):
            raise ValueError('old_heavy profile requires mixed calibration')
        values = plan[method]['budgets']
        if not values or values != sorted(set(values)) or any(not isinstance(n, int) or n <= 0 for n in values):
            raise ValueError('calibration budgets must be positive, distinct and ascending')
    unit_users = int(plan.get('unit_users', 256))
    began = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + f'_{os.getpid()}'
    launch = {'status': 'running', 'started_at': now(), 'run_id': run_id,
        'tmux': os.environ['TMUX'], 'tmux_tag': plan.get('tmux_tag'),
        'plan': {'path': str(plan_path), 'sha256': plan_hash}, 'execution_sources': source_hashes,
        'query': plan['query'], 'history': plan['history'], 'gpus': plan['gpus'], 'quality_stop_threshold': None}
    write_json(root / 'launches' / f'{run_id}.json', launch)
    write_json(root / 'launch.json', launch)
    jobs, events, stopped = queue.Queue(), queue.Queue(), threading.Event()
    complete, active, failures = [], {}, []
    children, lock = {}, threading.Lock()
    env = {**os.environ, 'OMP_NUM_THREADS': str(cfg['torch_threads']),
        'MKL_NUM_THREADS': str(cfg['torch_threads']), 'PYTHONUNBUFFERED': '1'}

    def sealed():
        if sha256(plan_path) != plan_hash or sources() != source_hashes:
            raise RuntimeError('frozen plan or execution sources changed during the queue')

    def command(gpu, scale, edge, stage, argv):
        attempt = 0
        while not stopped.is_set():
            sealed()
            announced = False
            while not stopped.is_set():
                info = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
                    '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True)
                free, total = map(float, info.strip().split(','))
                if free / total >= cfg['initial_free_fraction']:
                    break
                if not announced:
                    events.put({'stage': 'waiting_for_free_gpu', 'gpu': gpu, 'scale': scale,
                        'edge': edge_name(edge), 'next_stage': stage, 'free_fraction': free / total, 'at': now()})
                    announced = True
                stopped.wait(15)
            if stopped.is_set():
                raise RuntimeError('queue stopped before stage')
            attempt += 1
            log = root / 'runtime' / scale / edge_name(edge) / f'{stage}_{run_id}_{attempt:02d}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            started, start = now(), time.perf_counter()
            events.put({'stage': stage, 'gpu': gpu, 'scale': scale, 'edge': edge_name(edge), 'log': str(log), 'at': started})
            with log.open('w') as handle:
                with lock:
                    if stopped.is_set():
                        raise RuntimeError('queue stopped before child launch')
                    child = subprocess.Popen([sys.executable, *argv], cwd=ROOT, env=env,
                        stdout=handle, stderr=subprocess.STDOUT)
                    children[gpu] = child
                code = child.wait()
                with lock:
                    children.pop(gpu, None)
            with log.open('rb') as handle:
                handle.seek(max(0, log.stat().st_size - 65536))
                lines = [line.strip() for line in handle.read().decode(errors='replace').splitlines() if line.strip()]
            retry = code == 1 and bool(lines) and lines[-1] in ADMISSION_ERRORS
            write_json(log.with_suffix('.exit.json'), {'exit_code': code, 'admission_retry': retry,
                'started_at': started, 'finished_at': now(), 'elapsed_seconds': time.perf_counter() - start,
                'argv': [sys.executable, *argv], 'log': {'path': str(log), 'sha256': sha256(log)}})
            if code == 0:
                return
            if retry:
                events.put({'stage': 'admission_retry', 'gpu': gpu, 'scale': scale,
                    'edge': edge_name(edge), 'next_stage': stage, 'log': str(log), 'at': now()})
                stopped.wait(15)
                continue
            raise RuntimeError(f'{scale}/{edge_name(edge)} {stage} exit {code}; {log}')
        raise RuntimeError('queue stopped before stage')

    def worker(gpu):
        while not stopped.is_set():
            try:
                scale, edge = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                common = ['--scale', scale, '--edge', str(edge), '--gpu', str(gpu)]
                for method, key in (('query_only', 'query'), ('history_conditioned', 'history')):
                    setting = plan[key]
                    if key == 'history' and setting.get('reuse_v4'):
                        continue
                    feature = setting['feature_mode'] if key == 'query' else None
                    groups = [setting['budgets']] if setting['state_mode'] == 'pure' else [[n] for n in setting['budgets']]
                    for budgets in groups:
                        arguments = ['scripts/read_correction_v5/calibrate.py', *common, '--method', method,
                            '--state-mode', setting['state_mode'], '--budgets', *map(str, budgets),
                            '--output-root', str(root / 'calibration')]
                        if feature:
                            arguments.extend(['--feature-modes', feature])
                            arguments.extend(['--query-joint-epochs', str(setting.get('joint_epochs', 0))])
                        arguments.extend(['--mix-profile', setting.get('mix_profile', 'balanced')])
                        stage = f'fit_{key}_{setting["state_mode"]}_c' + '_'.join(map(str, budgets))
                        command(gpu, scale, edge, stage, arguments)
                entries = candidates(plan, root, scale, edge)
                candidate_path = root / 'candidates' / scale / edge_name(edge) / 'calibration_list.json'
                if candidate_path.exists() and json.loads(candidate_path.read_text()) != entries:
                    raise RuntimeError('existing calibration list differs from the fixed plan')
                if not candidate_path.exists():
                    write_json(candidate_path, entries)
                command(gpu, scale, edge, 'evaluate', ['scripts/read_correction_v5/evaluate_full.py', *common,
                    '--calibration-list', str(candidate_path), '--output', str(root / 'evaluation' / scale / edge_name(edge)),
                    '--unit-users', str(unit_users), '--limit-users', '0'])
                result = finished(root, scale, edge, entries, source_hashes)
                if result is None:
                    raise RuntimeError('evaluation exited without a complete edge summary')
                events.put({'stage': 'edge_complete', 'gpu': gpu, 'result': result, 'at': now()})
            except Exception as error:
                stopped.set()
                events.put({'stage': 'failure', 'gpu': gpu, 'scale': scale, 'edge': edge_name(edge), 'error': str(error), 'at': now()})
                return

    def snapshot():
        return {'updated_at': now(), 'run_id': run_id, 'completed_edges': len(complete), 'expected_edges': 15,
            'pending_edges': jobs.qsize(), 'active': active, 'completed': complete, 'failures': failures,
            'elapsed_seconds': time.perf_counter() - began}

    for scale, edge in plan['edges']:
        previous = finished(root, scale, edge, candidates(plan, root, scale, edge), source_hashes)
        if previous is not None:
            complete.append(previous)
            print(json.dumps({'status': 'edge_already_complete', 'scale': scale, 'edge': edge_name(edge)}), flush=True)
        else:
            jobs.put((scale, edge))
    next_report = time.monotonic() + 7200
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, gpu) for gpu in plan['gpus']]
        while any(not future.done() for future in futures) or not events.empty():
            try:
                event = events.get(timeout=5)
            except queue.Empty:
                event = None
            if event:
                print(json.dumps(event), flush=True)
                if event['stage'] == 'edge_complete':
                    complete.append(event['result']); active.pop(event['gpu'], None)
                elif event['stage'] == 'failure':
                    failures.append(event); active.pop(event['gpu'], None)
                else:
                    active[event['gpu']] = event
                write_json(root / 'progress.json', snapshot())
            if time.monotonic() >= next_report:
                report = snapshot()
                write_json(root / 'reports' / f'{run_id}_{int(report["elapsed_seconds"]):06d}.json', report)
                next_report += 7200
        for future in futures:
            future.result()
    if not failures and len(complete) != 15:
        failures.append({'error': 'queue ended before all 15 edges completed'})
    result = {**snapshot(), 'status': 'failed' if failures else 'complete', 'finished_at': now()}
    write_json(root / 'complete.json', result)
    if failures:
        return 1
    sealed()
    log = root / 'figure_generation.log'
    with log.open('w') as handle:
        code = subprocess.run([sys.executable, 'figures/src/read_correction_v5_2026_09.py',
            '--input-root', str(root), '--output-dir', str(figure_dir), '--require-complete'],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    write_json(root / 'figure_generation.exit.json', {'exit_code': code, 'finished_at': now(), 'log_sha256': sha256(log)})
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--figure-dir', type=Path, default=ROOT / 'figures/out/read_correction_2026_09/v5')
    args = parser.parse_args()
    raise SystemExit(run(args.output_root.resolve(), args.figure_dir.resolve()))

