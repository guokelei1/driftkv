#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/kuairand_next_item_chain_extension_20260807_v0.json
train_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_o1_next_item_chain
eval_output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_o2_next_item_multiversion
mkdir -p "$train_output" "$eval_output"
export CUDA_VISIBLE_DEVICES=0
python scripts/run_evokv_kuairand_next_item_chain.py --config "$config" --mode train 2>&1 | tee "$train_output/run.log"
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 scripts/run_evokv_kuairand_next_item_chain.py --config "$config" --mode evaluate 2>&1 | tee "$eval_output/run.log"
python scripts/validate_evokv_kuairand_next_item_chain.py --config "$config" --result "$eval_output/result.json"
