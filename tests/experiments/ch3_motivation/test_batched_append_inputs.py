"""Exact input values and per-event state/score parity, including FIFO clearance."""
from pathlib import Path
import sys
import numpy as np
import torch
import pytest
ROOT=Path(__file__).resolve().parents[3]
from tests.experiments.ch3_motivation.test_probe_projected_fast import fixture, nonzero, cpu_backend
from experiments.ch3_motivation.code.probe_projected_fast import (
    ProjectedFastProbeTrajectory, ProjectedGraphPool,
)
from experiments.ch3_motivation.code.batched_append_inputs import (
    BatchedAppendInputsTrajectory, batched_append_inputs,
)
from experiments.ch3_motivation.code import probe_projected_fast as original

@pytest.mark.parametrize('stage',['f','fg','fgh'])
@pytest.mark.parametrize('initial',[2,5])
def test_all_event_inputs_scores_and_cache_state(stage,initial):
    parent,model,sample,probe=fixture(stage=stage,initial=initial)
    nonzero(probe)
    times=np.r_[np.arange(86400-initial,86400),np.arange(86400,86409)]
    events=times,np.arange(1,len(times)+1),np.ones(len(times),dtype=np.int64)
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=5,append_band_size=2,query_batch=3)
    old=ProjectedFastProbeTrajectory(parent,model,probe,events,cfg,pool=ProjectedGraphPool(model,probe,window=5,capture=False))
    new=BatchedAppendInputsTrajectory(parent,model,probe,events,cfg,pool=ProjectedGraphPool(model,probe,window=5,capture=False))
    for timestamp in range(86400,86409):
        old.advance(timestamp);new.advance(timestamp)
        torch.testing.assert_close(new.score(sample['candidates'],sample['query_delta']),old.score(sample['candidates'],sample['query_delta']),rtol=0,atol=0)
        torch.testing.assert_close(new.cache.k,old.cache.k,rtol=0,atol=0)
        torch.testing.assert_close(new.cache.v,old.cache.v,rtol=0,atol=0)
        assert new.source.seq_len==old.source.seq_len
        assert new.native.cursor==old.native.cursor
        assert new.native.ledger==old.native.ledger
        assert new.report()['counts']==old.report()['counts']
    assert new.source.seq_len==0
    assert new.append_items.shape==(1,9)
    assert new.report()['append_input_execution']['input_tensor_bytes']==9*20


def test_context_restores_on_failure():
    with pytest.raises(RuntimeError):
        with batched_append_inputs():
            assert original.ProjectedFastProbeTrajectory is BatchedAppendInputsTrajectory
            raise RuntimeError('test')
    assert original.ProjectedFastProbeTrajectory is ProjectedFastProbeTrajectory
