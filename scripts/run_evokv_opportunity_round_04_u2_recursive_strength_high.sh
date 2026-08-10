#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/kuairand_next_item_recursive_rollout_kv_focused_high_e3_20260807_v0.json
output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_u2_recursive_kv_focused_high_e3
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 scripts/evaluate_evokv_kuairand_next_item_recursive_rollout.py --config "$config" 2>&1 | tee "$output/run.log"
python scripts/validate_evokv_kuairand_next_item_recursive_rollout.py --config "$config" --result "$output/result.json"
