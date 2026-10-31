"""Execution provenance precedes scores, remains immutable and restores context."""
import argparse
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import run_quality_batched as wrapper, probe_projected_fast as engine


def fixture(tmp_path):
    proof=tmp_path/'readiness/summary.json';proof.parent.mkdir()
    proof.write_text(json.dumps(dict(status='passed_actual_fg_and_nonzero_fgh',execution_binding=wrapper.optimized.binding())))
    (proof.parent/'exit.json').write_text('{"exit_code":0}')
    plan=tmp_path/'plan.json';plan.write_text('{}')
    return argparse.Namespace(readiness=proof,plan=plan,output=tmp_path/'output',gpu=1)


def test_dispatch_binding_and_exact_resume(tmp_path,monkeypatch):
    args=fixture(tmp_path)
    baseline=engine.ProjectedFastProbeTrajectory
    def original(values):
        assert (values.output/'execution_optimization_configuration.json').exists()
        assert engine.ProjectedFastProbeTrajectory is wrapper.optimized.BatchedAppendInputsTrajectory
        (values.output/'configuration.json').write_text('{}')
        (values.output/'summary.json').write_text('{}')
    monkeypatch.setattr(wrapper.original,'main',original)
    wrapper.run(args)
    assert engine.ProjectedFastProbeTrajectory is baseline
    assert json.loads((args.output/'execution_optimization.json').read_text())['enabled'] is True
    wrapper.run(args)
    proof=json.loads(args.readiness.read_text());proof['changed']=True;args.readiness.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='changed on resume'):wrapper.run(args)


def test_unbound_existing_scores_rejected(tmp_path):
    args=fixture(tmp_path);args.output.mkdir()
    (args.output/'configuration.json').write_text('{}')
    with pytest.raises(ValueError,match='silently adopt'):wrapper.run(args)
