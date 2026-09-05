from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "insight_two"))

from run_paired_functional_delta_canary_v2 import _DiagnosticNumpyProxy


def test_v2_finite_guard_only_accepts_nonnumeric_metadata() -> None:
    proxy = _DiagnosticNumpyProxy()
    assert proxy.isfinite("paired_closure") is True
    assert proxy.isfinite(b"metadata") is True
    assert bool(proxy.isfinite(3.0)) is True
    assert bool(proxy.isfinite(float("inf"))) is False
    values = np.asarray([1.0, np.nan])
    assert proxy.isfinite(values).tolist() == [True, False]


def test_v2_proxy_forwards_other_numpy_operations() -> None:
    proxy = _DiagnosticNumpyProxy()
    assert proxy.arange(4).tolist() == [0, 1, 2, 3]
