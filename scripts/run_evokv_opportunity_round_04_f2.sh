#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/kuairand_cumulative_staleness_20260807_v0.json
output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_f2_kuairand_cumulative_staleness
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0
python scripts/run_evokv_kuairand_cumulative_staleness.py --config "$config" 2>&1 | tee "$output/run.log"
python scripts/validate_evokv_kuairand_cumulative_staleness.py --config "$config" --result "$output/result.json"
