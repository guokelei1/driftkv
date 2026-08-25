#!/usr/bin/env python3
"""Launch the v3 data-bounded rolling recipe matrix via the shared runner."""
from pathlib import Path

import run_yambda500m_hstu_native_rolling_recipe_matrix_v2 as runner


ROOT = Path(__file__).resolve().parents[1]
runner.CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_rolling_recipe_matrix_v3.yaml"
runner.MANIFESTS = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3"
runner.OUTPUT = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"


if __name__ == "__main__":
    runner.main()
