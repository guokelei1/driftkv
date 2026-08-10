#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
base_config=configs/evokv_root_cause/kuairand_dense_hstu_small_base_20260807_v0.json
base_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_m1_dense_small_base
intraday_config=configs/evokv_root_cause/kuairand_dense_hstu_small_intraday_20260807_v0.json
intraday_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_m2_dense_small_intraday
chain_config=configs/evokv_root_cause/kuairand_dense_hstu_small_chain_20260807_v0.json
chain_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_m3_dense_small_chain
multiversion_config=configs/evokv_root_cause/kuairand_dense_hstu_small_multiversion_20260807_v0.json
multiversion_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_m4_dense_small_multiversion
mkdir -p "$base_output" "$intraday_output" "$chain_output" "$multiversion_output"
export CUDA_VISIBLE_DEVICES=0
python scripts/run_evokv_kuairand_engagement.py --config "$base_config" --training-only 2>&1 | tee -a "$base_output/run.log"
python scripts/validate_evokv_kuairand_engagement_training.py --config "$base_config" --result "$base_output/training.json"
python scripts/run_evokv_kuairand_intraday_update.py --config "$intraday_config" 2>&1 | tee "$intraday_output/run.log"
python scripts/validate_evokv_kuairand_intraday_update.py --config "$intraday_config" --result "$intraday_output/result.json"
python scripts/run_evokv_kuairand_intraday_chain_extension.py --config "$chain_config" 2>&1 | tee "$chain_output/run.log"
python scripts/validate_evokv_kuairand_intraday_chain_extension.py --config "$chain_config" --result "$chain_output/result.json"
python scripts/run_evokv_kuairand_multiversion_staleness.py --config "$multiversion_config" 2>&1 | tee "$multiversion_output/run.log"
python scripts/validate_evokv_kuairand_multiversion_staleness.py --config "$multiversion_config" --result "$multiversion_output/result.json"
