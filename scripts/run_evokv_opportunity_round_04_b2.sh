#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/ml1m_locked_popular50_confirmation_20260807_v0.json
output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_b2_ml1m_locked_popular50_confirmation
test ! -f "$output/summary.json"
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0
python scripts/run_evokv_ml1m_locked_confirmation.py --config "$config" 2>&1 | tee "$output/run.log"
python scripts/validate_evokv_ml1m_locked_confirmation.py --config "$config" --result "$output/summary.json"
