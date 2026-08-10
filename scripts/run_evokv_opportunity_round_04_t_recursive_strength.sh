#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
export CUDA_VISIBLE_DEVICES=0,1

balanced_config=configs/evokv_root_cause/kuairand_next_item_recursive_rollout_balanced_2x_e3_20260807_v0.json
balanced_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_t1_recursive_balanced_2x_e3
mkdir -p "$balanced_output"
torchrun --standalone --nproc_per_node=2 scripts/evaluate_evokv_kuairand_next_item_recursive_rollout.py --config "$balanced_config" 2>&1 | tee "$balanced_output/run.log"
python scripts/validate_evokv_kuairand_next_item_recursive_rollout.py --config "$balanced_config" --result "$balanced_output/result.json"

focused_config=configs/evokv_root_cause/kuairand_next_item_recursive_rollout_kv_focused_e3_20260807_v0.json
focused_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_t2_recursive_kv_focused_e3
mkdir -p "$focused_output"
torchrun --standalone --nproc_per_node=2 scripts/evaluate_evokv_kuairand_next_item_recursive_rollout.py --config "$focused_config" 2>&1 | tee "$focused_output/run.log"
python scripts/validate_evokv_kuairand_next_item_recursive_rollout.py --config "$focused_config" --result "$focused_output/result.json"
