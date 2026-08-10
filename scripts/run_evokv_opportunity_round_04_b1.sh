#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/ml1m_prediction_query_objective_20260807_v0.json
output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_b1_ml1m_prediction_query
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0
python scripts/train_evokv_ml1m_prediction_query.py --config "$config" 2>&1 | tee "$output/run.log"
python scripts/validate_evokv_ml1m_prediction_query.py --config "$config" --result "$output/summary.json"
