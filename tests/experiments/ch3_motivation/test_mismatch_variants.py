"""Definitions of the additional E4 window/head-scale comparisons."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.analysis import mismatch_rows
from experiments.ch3_motivation.code.run_mismatch_variants import aggregate_records, cache_diagnostics
from hstu_kvcache.models import HSTUKVCache


def test_joint_relative_definition_matches_primary_e4_and_splits_heads():
    generator = torch.Generator().manual_seed(41)
    old = HSTUKVCache(torch.randn(3, 1, 12, 8, generator=generator),
                      torch.randn(3, 1, 12, 8, generator=generator), 12)
    new = HSTUKVCache(torch.randn(3, 1, 12, 8, generator=generator),
                      torch.randn(3, 1, 12, 8, generator=generator), 12)
    scales = dict(key_rms=np.ones((3, 2)).tolist(), value_rms=np.ones((3, 2)).tolist())
    record = cache_diagnostics(old, new, uid=9, window=256, heads=2, fit_statistics=scales)
    primary = mismatch_rows(old, new, uids=[9])[0]
    assert record["absolute_top10_row_share"] == pytest.approx(primary["absolute_top10_row_share"], abs=1e-14)
    assert record["per_row_relative_top10_row_share"] == pytest.approx(primary["relative_top10_row_share"], abs=1e-14)
    assert record["absolute_key_share"] + record["absolute_value_share"] == pytest.approx(1)
    assert np.asarray(record["absolute_key_layer_head_mass"]).shape == (3, 2)


def test_fixed_fitting_rms_removes_layer_scale_without_eval_fitting():
    new_k = torch.ones(2, 1, 10, 2, dtype=torch.float64)
    new_k[1] *= 10
    zero = torch.zeros_like(new_k)
    old = HSTUKVCache(zero, zero, 10)
    new = HSTUKVCache(new_k, zero, 10)
    record = cache_diagnostics(old, new, uid=1, window=256, heads=2,
        fit_statistics=dict(key_rms=[[1, 1], [10, 10]], value_rms=[[1, 1], [1, 1]]))
    assert record["absolute_layer_evenness"] == pytest.approx([2 / 101, 200 / 101])
    assert record["fit_rms_layer_evenness"] == pytest.approx([1, 1])
    np.testing.assert_allclose(record["new_key_norm_rms"], [[1, 1], [10, 10]])


def test_sub_512_group_has_counts_only_and_512_has_quantiles():
    records = [dict(uid=i, window=256, history_rows=100,
                    absolute_top10_row_share=.2, absolute_layer_evenness=[1., 1.]) for i in range(512)]
    assert aggregate_records(records[:511]) == dict(users=511, eligible=False)
    result = aggregate_records(records)
    assert result["eligible"]
    assert result["metrics"]["absolute_top10_row_share"]["median"] == .2
    assert result["metrics"]["absolute_layer_evenness"]["median"] == [1., 1.]
