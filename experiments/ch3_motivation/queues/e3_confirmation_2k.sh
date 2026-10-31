#!/usr/bin/env bash
# Prepared queue only. Launch separately after numerical/performance readiness.
set -euo pipefail
cd /home/gkl/work/evokv
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
ch3_gpu=${CH3_GPU:-0}
ch3_attempt=${CH3_ATTEMPT:-attempt_1}
ch3_scenario_root=${CH3_SCENARIO_ROOT:-experiments/ch3_motivation/runs/scenario_qualification/attempt_1}
for edge in 1 2 3 4 5; do
    output="experiments/ch3_motivation/runs/e3/edge_${edge}/confirmation/${ch3_attempt}"
    mkdir -p "$output"
    .venv/bin/python scripts/ch3_motivation/run_selective_population.py \
        --edge "$edge" --population-file "$ch3_scenario_root/edge_${edge}/confirmation/scenario_manifest.json" --run --gpu "$ch3_gpu" \
        --output "$output" >"$output/run.log" 2>&1
done
