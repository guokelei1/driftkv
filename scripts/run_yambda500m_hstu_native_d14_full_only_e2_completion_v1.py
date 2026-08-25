#!/usr/bin/env python3
"""Use the existing matrix Full-only evaluator to fill only D=14, E=2."""
from __future__ import annotations

from pathlib import Path

import run_yambda500m_hstu_native_rolling_recipe_matrix_v2 as matrix


ROOT = Path(__file__).resolve().parents[1]
matrix.CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_d14_full_only_e2_completion_v1.yaml"
matrix.MANIFESTS = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3"
matrix.OUTPUT = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/d14_full_only_e2_completion_v1"


def main() -> None:
    env = {**matrix.os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    for edge in range(1, 6):
        parent = matrix.ROOT / "results/yambda500m_small_seed17/hstu_native_release_chain_v1/v0/checkpoint_100.pt" if edge == 1 else (
            ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/train_14d/checkpoints" / f"v{edge - 1}" / "checkpoint_100.pt"
        )
        current = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/train_14d/checkpoints" / f"v{edge}" / "checkpoint_100.pt"
        matrix.evaluate(duration=14, horizon=2, edge=edge, parent=parent, current=current, env=env)


if __name__ == "__main__":
    main()
