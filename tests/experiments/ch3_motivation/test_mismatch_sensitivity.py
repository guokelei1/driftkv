"""Known-distribution checks for E4 explanation and the 512-user floor."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.analyze_mismatch_sensitivity import (
    item_concentration, metric_summary, row_diagnostics,
)


def test_uniform_row_mass_has_uniform_decile_shares():
    result, _ = row_diagnostics(np.ones(100), np.arange(1, 101), extension_start=1000)
    assert result["top10_row_share"] == .1
    assert result["head10_mass_share"] == .1
    assert result["tail10_mass_share"] == .1
    assert result["mass_weighted_position"] == pytest.approx(.5)
    for decile in range(1, 11):
        assert result[f"position_decile_{decile}_mass_share"] == .1


def test_concentrated_head_and_repeated_item_are_identified():
    values = np.ones(100)
    values[:10] = 100
    ids = np.arange(1, 101)
    ids[:10] = 7
    result, top = row_diagnostics(values, ids, extension_start=1000)
    assert set(top) == set(range(10))
    assert result["head10_mass_share"] == pytest.approx(1000 / 1090)
    assert result["top10_in_head10_fraction"] == 1
    assert result["top10_in_tail10_fraction"] == 0
    assert result["top10_dominant_item_row_fraction"] == 1
    assert result["top10_position_mean"] == pytest.approx(.05)


def test_sub_512_subgroup_quantiles_are_suppressed():
    small = metric_summary([.5] * 511)
    assert small["users"] == 511 and not small["eligible"]
    assert small["median"] is None
    permitted = metric_summary([.5] * 512)
    assert permitted["eligible"] and permitted["median"] == .5


def test_item_pooling_uses_mass_and_marks_small_item_support():
    result, audit = item_concentration(np.array([1, 1, 2, 3]), np.array([7, 8, 7, 8]),
        dict(absolute=np.array([5., 4., 1., 0.]), relative=np.array([.5, .5, 1., 0.])))
    assert result["absolute"]["top1_item_mass_share"] == .9
    assert result["relative"]["top1_item_mass_share"] == .5
    assert all(row["role"] == "diagnostic_item_count_only" for row in audit)
