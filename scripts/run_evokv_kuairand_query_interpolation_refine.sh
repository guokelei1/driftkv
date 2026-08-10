#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/kuairand_query_publish_interpolation_refine_20260808_v1.json
output=results/opportunity_discovery/evokv_kuairand_query_20260808_v0/publish_interpolation_refine_seed4217
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0
python scripts/evaluate_evokv_kuairand_query_interpolation.py --config "$config" 2>&1 | tee "$output/run.log"
