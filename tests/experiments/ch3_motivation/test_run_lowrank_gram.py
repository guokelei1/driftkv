"""Execution binding and resume behavior, independent of GPU math oracles."""
import argparse
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import run_lowrank_gram as wrapper


@pytest.mark.parametrize('protocol', ['release', 'timestamp'])
def test_original_dispatch_and_bound_resume(tmp_path, monkeypatch, protocol):
    monkeypatch.setattr(wrapper, 'ROOT', tmp_path)
    driver = tmp_path / 'experiments/ch3_motivation/code' / (
        'run_lowrank_release.py' if protocol == 'release' else 'run_lowrank_timestamp_guarded.py')
    driver.parent.mkdir(parents=True)
    driver.write_text('# CPU dispatch fixture\n')
    evidence = tmp_path / 'comparison.json'
    evidence.write_text('{}\n')
    readiness = tmp_path / 'readiness.json'
    readiness.write_text(json.dumps(dict(status='passed_actual_both_protocols_all8',
        execution_binding=wrapper.binding(), records=[dict(comparison=str(evidence),
        comparison_sha256=wrapper.sha256(evidence))])))
    output = tmp_path / 'output'
    values = ['--method', 'randomized', '--oversample', '16', '--power-iterations', '2',
              '--candidates', '230', '--output', str(output)]
    args = argparse.Namespace(protocol=protocol, readiness=readiness, driver_arguments=['--', *values])
    calls = []
    original_svd = torch.linalg.svd

    def original_run(path, run_name):
        assert path == str(driver) and run_name == '__main__'
        assert sys.argv == [str(driver), *values]
        assert (output / 'execution_optimization_configuration.json').is_file()
        assert torch.linalg.svd is not original_svd
        (output / 'configuration.json').write_text('{}\n')
        (output / 'summary.json').write_text('{"status":"complete"}\n')
        calls.append(path)

    monkeypatch.setattr(wrapper.runpy, 'run_path', original_run)
    wrapper.run(args)
    wrapper.run(args)
    assert len(calls) == 2 and torch.linalg.svd is original_svd
    report = json.loads((output / 'execution_optimization.json').read_text())
    assert report['enabled'] and report['original_result_sha256'] == wrapper.sha256(output / 'summary.json')
    # A changed proof can never attach a different execution to existing blocks.
    readiness.write_text(readiness.read_text() + '\n')
    with pytest.raises(ValueError, match='execution changed'):
        wrapper.run(args)
    assert len(calls) == 2
