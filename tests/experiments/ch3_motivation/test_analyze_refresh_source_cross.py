"""Paired source matching differs from one-map dominance and common offsets."""
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.ch3_motivation.code.analyze_refresh_source_cross import METHODS, aggregate_users


def population(*, actual_wins_both=False):
    pattern = np.linspace(-1., 1., 16)+.7
    coefficients = dict(zip(METHODS, [5., 6., 2. if actual_wins_both else 1., 4.,
                                     1. if actual_wins_both else 2., 3., 0.], strict=True))
    scores = {name: (scale*pattern).tolist() for name, scale in coefficients.items()}
    mse = {name: float(np.mean(np.asarray(value)**2)) for name, value in scores.items()}
    return [dict(uid=uid, scores=scores, score_mse=mse, panel_candidates=list(range(16))) for uid in range(8)]


def test_matching_contrasts_and_centered_error_are_paired_with_one_teacher():
    result = aggregate_users(population(), replicates=32, seed=17)
    raw, centered = result["contrasts"]
    assert raw["actual_input_matching_advantage"]["mean"] > 0
    assert raw["native_input_matching_advantage"]["mean"] > 0
    assert raw["matching_interaction"]["mean"] == pytest.approx(
        raw["actual_input_matching_advantage"]["mean"]+raw["native_input_matching_advantage"]["mean"])
    assert raw["point_estimate_pattern"].startswith("matched_source_favored")
    assert centered["matching_interaction"]["mean"] < raw["matching_interaction"]["mean"]


def test_positive_interaction_does_not_mask_one_actual_map_winning_both_inputs():
    result = aggregate_users(population(actual_wins_both=True), replicates=32, seed=17)
    for contrast in result["contrasts"]:
        assert contrast["matching_interaction"]["mean"] > 0
        assert contrast["native_input_matching_advantage"]["mean"] < 0
        assert contrast["point_estimate_pattern"].startswith("actual_trained_map_wins_both")
