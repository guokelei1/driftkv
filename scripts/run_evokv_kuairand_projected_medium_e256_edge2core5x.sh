#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/kuairand_projected_medium_e256_edge2core5x_20260808_v3.json
output=results/opportunity_discovery/evokv_kuairand_projected_scale_20260808_v0/medium_e256_edge2core5x
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0
python scripts/run_evokv_kuairand_projected_scale.py --config "$config" 2>&1 | tee "$output/run.log"
