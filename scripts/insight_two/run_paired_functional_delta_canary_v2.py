#!/usr/bin/env python3
"""Execution-only v2 amendment for the frozen paired-delta canary.

The v1 runner stopped before producing a metric row because its diagnostic
finite check sent a string field to ``numpy.isfinite``.  This wrapper keeps the
entire frozen computation and method grid unchanged, redirects it to a new
contract/output root, and narrows that one scalar check: strings and bytes are
metadata, while every numeric value still uses NumPy's original finite test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "insight_two"))

import run_paired_functional_delta_canary as frozen_v1  # noqa: E402


CONTRACT = (
    ROOT
    / "configs/contracts/"
    "yambda500m_medium_legacy_pointwise_insight2_paired_functional_delta_v2.yaml"
)
OUTPUT_ROOT = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/"
    "diagnostic_paired_functional_delta_v2"
)


class _DiagnosticNumpyProxy:
    """Forward NumPy unchanged except for nonnumeric diagnostic metadata."""

    def __getattr__(self, name: str):
        return getattr(np, name)

    @staticmethod
    def isfinite(value):
        if isinstance(value, (str, bytes)):
            return True
        return np.isfinite(value)


def main() -> None:
    frozen_v1.CONTRACT = CONTRACT
    frozen_v1.OUTPUT_ROOT = OUTPUT_ROOT
    frozen_v1.np = _DiagnosticNumpyProxy()
    frozen_v1.main()


if __name__ == "__main__":
    main()
