import sys
from pathlib import Path

import numpy as np

from experiments.ch3_motivation.code.run_candidate_residual_probe import centered_panels, panel_indices


def test_panels_use_teacher_only_and_keep_fixed_broad_positions():
    scores = np.arange(64, dtype=float)
    scores[32:48] = 100 + np.arange(16) * .0001
    indices = panel_indices(scores)
    np.testing.assert_array_equal(indices[0], np.r_[0:8, 56:64])
    np.testing.assert_array_equal(indices[1], np.arange(32, 48))
    assert len(set(indices[0])) == len(set(indices[1])) == 16
    np.testing.assert_array_equal(panel_indices(np.zeros(64))[1], np.arange(16))


def test_panel_centering_keeps_actual_candidate_targets_and_total_error():
    features = np.arange(64 * 16).reshape(64, 16).astype(np.float32)
    residual = np.arange(64 * 2 * 3 * 4).reshape(64, 2, 3, 4) / 100.
    teacher = np.linspace(-2, 2, 64)
    actual = teacher + np.linspace(.1, .2, 64)
    indices = panel_indices(teacher)
    out = centered_panels(features, residual, teacher, actual, indices)
    np.testing.assert_allclose(out["read_target"].mean(1), 0, atol=2e-6)
    np.testing.assert_allclose(out["logit_target"].mean(1), 0, atol=2e-9)
    np.testing.assert_array_equal(out["features"], features[indices])
    np.testing.assert_allclose(out["read_total_mse"], (residual[indices] ** 2).mean((1, 4)))
    np.testing.assert_allclose(out["logit_total_mse"], ((teacher-actual)[indices] ** 2).mean(1))
