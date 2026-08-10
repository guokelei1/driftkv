#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
export CUDA_VISIBLE_DEVICES=0

for seed in 14929 53117; do
  config="configs/evokv_root_cause/kuairand_query_full_interpolation_seed${seed}_20260808_v2.json"
  output="results/opportunity_discovery/evokv_kuairand_query_20260808_v0/full_interpolation_seed${seed}"
  mkdir -p "$output"
  python scripts/evaluate_evokv_kuairand_query_interpolation.py --config "$config" 2>&1 | tee "$output/run.log"
done
