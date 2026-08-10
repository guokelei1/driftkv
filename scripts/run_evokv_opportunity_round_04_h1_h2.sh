#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
base_config=configs/evokv_root_cause/kuairand_candidate_aware_h128_l4_20260807_v0.json
base_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_h1_kuairand_h128_l4_base
intraday_config=configs/evokv_root_cause/kuairand_intraday_h128_l4_20260807_v0.json
intraday_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_h2_kuairand_intraday_h128_l4
mkdir -p "$base_output" "$intraday_output"
export CUDA_VISIBLE_DEVICES=0
python scripts/run_evokv_kuairand_engagement.py --config "$base_config" 2>&1 | tee "$base_output/run.log"
python scripts/validate_evokv_kuairand_engagement.py --config "$base_config" --result "$base_output/evaluation.json"
python scripts/run_evokv_kuairand_intraday_update.py --config "$intraday_config" 2>&1 | tee "$intraday_output/run.log"
python scripts/validate_evokv_kuairand_intraday_update.py --config "$intraday_config" --result "$intraday_output/result.json"
