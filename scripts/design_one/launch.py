#!/usr/bin/env python3
"""Run the frozen Design 1 benchmark on four GPU UID shards, then aggregate.

Invoke this small coordinator inside tmux for long runs. Completed evaluation
units resume under matching source/input signatures in benchmark.py.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    design = parser.add_mutually_exclusive_group()
    design.add_argument('--design-calibration', type=Path)
    design.add_argument('--design-calibration-list', type=Path)
    parser.add_argument('--history-packs', type=Path)
    parser.add_argument('--cohort-size', type=int, default=128)
    parser.add_argument('--query-batch', type=int, default=128)
    parser.add_argument('--unit-users', type=int, default=256)
    parser.add_argument('--max-users', type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    script = ROOT / 'scripts/design_one/benchmark.py'
    common = ['--output', str(args.output), '--num-shards', '4',
              '--cohort-size', str(args.cohort_size), '--query-batch', str(args.query_batch),
              '--unit-users', str(args.unit_users), '--max-users', str(args.max_users)]
    if args.design_calibration:
        common += ['--design-calibration', str(args.design_calibration.resolve())]
    if args.design_calibration_list:
        common += ['--design-calibration-list', str(args.design_calibration_list.resolve())]
    if args.history_packs:
        common += ['--history-packs', str(args.history_packs.resolve())]
    state = {'status': 'running', 'started': stamp(), 'commands': [], 'workers': []}
    status = args.output / 'launch.json'

    def save():
        status.write_text(json.dumps(state, indent=2) + '\n')

    handles, workers = [], []
    try:
        for gpu in range(4):
            command = [sys.executable, '-u', str(script), 'score', *common,
                       '--gpu', str(gpu), '--shard-index', str(gpu)]
            handle = (args.output / f'worker_{gpu}.log').open('a', buffering=1)
            handles.append(handle)
            worker = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            workers.append(worker)
            state['commands'].append(command)
            state['workers'].append({'gpu': gpu, 'pid': worker.pid, 'exit_code': None})
            save()
        for gpu, worker in enumerate(workers):
            state['workers'][gpu]['exit_code'] = worker.wait()
            save()
        if any(record['exit_code'] for record in state['workers']):
            state.update(status='failed', finished=stamp())
            save()
            raise SystemExit(1)
        command = [sys.executable, '-u', str(script), 'aggregate', *common]
        state['aggregate_command'] = command
        save()
        with (args.output / 'aggregate.log').open('a', buffering=1) as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        state.update(status='complete' if result.returncode == 0 else 'failed',
                     aggregate_exit_code=result.returncode, finished=stamp())
        save()
        raise SystemExit(result.returncode)
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
