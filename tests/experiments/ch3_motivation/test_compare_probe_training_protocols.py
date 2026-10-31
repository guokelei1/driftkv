"""Numerical difference-of-increments oracle; do not subtract independent CIs."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.compare_probe_training_protocols import contrast_draws


def test_h_contrast_preserves_shared_draw_cancellation():
    # Shared population variation dominates each method's marginal uncertainty.
    # Correct direct differences cancel it, leaving a constant h advantage.
    noise=np.array([0.,-.08,.11,-.12,.07])
    values={('joint','f'):.6+noise,('joint','fg'):.61+noise,('joint','fgh'):.63+noise,
            ('new_term_only','f'):.6+noise,('new_term_only','fg'):.615+noise,
            ('new_term_only','fgh'):.645+noise}
    result=contrast_draws(values)
    np.testing.assert_allclose(result['new_minus_joint_h_increment'],.01,atol=1e-15)
    np.testing.assert_allclose(result['new_minus_joint_g_increment'],.005,atol=1e-15)
    np.testing.assert_allclose(result['new_minus_joint_f'],0.,atol=0.)
    np.testing.assert_allclose(result['new_minus_joint_fgh'],.015,atol=1e-15)
