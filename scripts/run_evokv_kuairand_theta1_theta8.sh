#!/usr/bin/env bash
set -euo pipefail
config=configs/evokv_root_cause/kuairand_projected_theta1_theta8_seed53117_20260808_v0.json
output=results/opportunity_discovery/evokv_kuairand_projected_theta1_theta8_seed53117_v0
mkdir -p "$output"
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only | tee -a "$output/preflight.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee -a "$output/hardware.log"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" 2>&1 | tee -a "$output/run.log"
