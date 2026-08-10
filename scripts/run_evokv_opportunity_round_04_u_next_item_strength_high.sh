#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/kuairand_next_item_update_strength_high_20260807_v0.json
output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_u1_next_item_strength_high
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0
python scripts/train_evokv_kuairand_next_item_strength.py --config "$config" 2>&1 | tee "$output/run.log"
python scripts/validate_evokv_kuairand_next_item_strength.py --config "$config" --result "$output/result.json"
