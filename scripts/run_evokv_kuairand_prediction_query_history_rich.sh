#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/kuairand_prediction_query_history_rich_20260808_v1.json
output=results/opportunity_discovery/evokv_kuairand_query_20260808_v0/history_rich_seed4217
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0
python scripts/run_evokv_kuairand_prediction_query.py --config "$config" 2>&1 | tee "$output/run.log"
python scripts/validate_evokv_kuairand_prediction_query.py --config "$config" --result "$output/summary.json"
