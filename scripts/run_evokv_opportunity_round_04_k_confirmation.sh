#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
base_config=configs/evokv_root_cause/kuairand_candidate_aware_seed14929_20260807_v0.json
base_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_k1_base_seed14929
intraday_config=configs/evokv_root_cause/kuairand_intraday_seed14929_20260807_v0.json
intraday_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_k2_intraday_seed14929
chain_config=configs/evokv_root_cause/kuairand_chain_extension_seed14929_20260807_v0.json
chain_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_k3_chain_seed14929
confirmation_config=configs/evokv_root_cause/kuairand_multiversion_confirmation_seed14929_20260807_v0.json
confirmation_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_k4_multiversion_confirmation
mkdir -p "$base_output" "$intraday_output" "$chain_output" "$confirmation_output"
export CUDA_VISIBLE_DEVICES=0
python scripts/run_evokv_kuairand_engagement.py --config "$base_config" 2>&1 | tee "$base_output/run.log"
python scripts/validate_evokv_kuairand_engagement.py --config "$base_config" --result "$base_output/evaluation.json"
python scripts/run_evokv_kuairand_intraday_update.py --config "$intraday_config" 2>&1 | tee "$intraday_output/run.log"
python scripts/validate_evokv_kuairand_intraday_update.py --config "$intraday_config" --result "$intraday_output/result.json"
python scripts/run_evokv_kuairand_intraday_chain_extension.py --config "$chain_config" 2>&1 | tee "$chain_output/run.log"
python scripts/validate_evokv_kuairand_intraday_chain_extension.py --config "$chain_config" --result "$chain_output/result.json"
python scripts/run_evokv_kuairand_multiversion_confirmation.py --config "$confirmation_config" 2>&1 | tee "$confirmation_output/run.log"
python scripts/validate_evokv_kuairand_multiversion_confirmation.py --config "$confirmation_config" --result "$confirmation_output/result.json"
