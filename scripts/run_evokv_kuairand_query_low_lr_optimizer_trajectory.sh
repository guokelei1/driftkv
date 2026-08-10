#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/kuairand_query_low_lr_optimizer_trajectory_20260808_v2.json
output=results/opportunity_discovery/evokv_kuairand_query_20260808_v0/low_lr_optimizer_trajectory
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0
python scripts/evaluate_evokv_kuairand_query_update_scope.py --config "$config" 2>&1 | tee "$output/run.log"
