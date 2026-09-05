#!/usr/bin/env python3
"""Adjudicate the v2 execution amendment with the frozen v1 logic."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "insight_two"))

import adjudicate_paired_functional_delta as frozen_v1  # noqa: E402


def main() -> None:
    frozen_v1.CONTRACT = (
        ROOT
        / "configs/contracts/"
        "yambda500m_medium_legacy_pointwise_insight2_paired_functional_delta_v2.yaml"
    )
    frozen_v1.RESULT_ROOT = (
        ROOT
        / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/"
        "diagnostic_paired_functional_delta_v2"
    )
    frozen_v1.main()


if __name__ == "__main__":
    main()
