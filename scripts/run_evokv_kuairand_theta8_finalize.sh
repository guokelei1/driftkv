#!/usr/bin/env bash
set -euo pipefail
config=configs/evokv_root_cause/kuairand_projected_theta1_theta8_seed53117_20260808_v0.json
output=results/opportunity_discovery/evokv_kuairand_projected_theta1_theta8_seed53117_v0
checkpoint=checkpoints/evokv_kuairand_projected_theta1_theta8_seed53117_v0
mkdir -p "$output"
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only | tee -a "$output/finalize_preflight.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee -a "$output/finalize_hardware.log"
if [[ -f "$checkpoint/theta_8/manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" 2>&1 | tee -a "$output/finalize.log"
else
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --stop-after-version 8 --candidate-priority standard_n8192_e2_l004 2>&1 | tee -a "$output/finalize.log"
fi
