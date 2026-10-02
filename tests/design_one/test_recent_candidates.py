"""History-only recent queries retain seeded fill and the original default."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from read_correction_2026_09.v2.calibrate import mixed_candidates


def test_recent_sixteen_and_short_fill_preserve_original_default():
    history = np.r_[np.arange(1, 26), 25, 0, 31, 99, -1, 3, 5]
    original = [5, 3, 31, 25, 24, 23, 22, 21, 6, 27, 2, 30, 15, 18, 12, 19]
    np.testing.assert_array_equal(mixed_candidates(123, history, 32), original)
    recent = mixed_candidates(123, history, 32, recent_budget=16)
    np.testing.assert_array_equal(recent, [5, 3, 31, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13])
    assert len(set(recent)) == 16 and set(recent).issubset(history)
    assert np.all((recent > 0) & (recent < 32))

    short = np.array([-5, 0, 99, 2, 2, 4, 1, 4])
    original_short = [4, 1, 2, 25, 3, 31, 5, 6, 27, 30, 15, 18, 12, 19, 28, 26]
    np.testing.assert_array_equal(mixed_candidates(123, short, 32), original_short)
    filled = mixed_candidates(123, short, 32, recent_budget=16)
    np.testing.assert_array_equal(filled, original_short)
    np.testing.assert_array_equal(filled, mixed_candidates(123, short, 32, recent_budget=16))
    assert len(set(filled)) == 16 and np.all((filled > 0) & (filled < 32))
