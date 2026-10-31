"""Different additive norm definitions must not be mislabeled or selected."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.analyze_mismatch_norms import norm_shares, summarize_records


def test_sqrt_changes_contribution_but_preserves_highest_rows():
    d = np.ones(100)
    d[:10] = 100
    result = norm_shares(d)
    assert result["squared_top10_share"] == pytest.approx(1000 / 1090)
    assert result["norm_top10_share"] == pytest.approx(100 / 190)
    assert result["squared_top10_share"] > result["norm_top10_share"]
    uniform = norm_shares(np.ones(100))
    assert uniform["squared_top10_share"] == uniform["norm_top10_share"] == .1


def test_small_group_is_counts_only():
    assert summarize_records([{}] * 511) == dict(users=511, eligible=False)
