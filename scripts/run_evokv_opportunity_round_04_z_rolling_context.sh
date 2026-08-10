#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
train_config=configs/evokv_root_cause/kuairand_next_item_rolling_context_chain_20260807_v0.json
screen_config=configs/evokv_root_cause/kuairand_next_item_rolling_context_screen_20260807_v0.json
train_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_z1_rolling_context_factorial
screen_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_z2_rolling_context_screen
mkdir -p "$train_output" "$screen_output"
export CUDA_VISIBLE_DEVICES=0
python scripts/train_evokv_kuairand_next_item_rolling_context.py --config "$train_config" 2>&1 | tee -a "$train_output/run.log"
python scripts/validate_evokv_kuairand_next_item_rolling_context.py --result "$train_output/training.json"
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 scripts/evaluate_evokv_kuairand_next_item_rolling_context.py --config "$screen_config" 2>&1 | tee -a "$screen_output/run.log"
python scripts/validate_evokv_kuairand_next_item_rolling_context_screen.py --result "$screen_output/result.json"
