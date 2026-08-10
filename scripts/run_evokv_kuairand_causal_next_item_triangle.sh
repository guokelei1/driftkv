#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1

config=configs/evokv_root_cause/kuairand_causal_next_item_triangle_theta1_theta8_20260808_v0.json
root=results/opportunity_discovery/evokv_kuairand_causal_next_item_triangle_20260808_v0
mkdir -p "$root"

sha256sum "$config" > "$root/config.sha256"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"

CUDA_VISIBLE_DEVICES=0 python scripts/run_evokv_kuairand_causal_next_item_triangle.py \
  --config "$config" \
  --phase train \
  2>&1 | tee "$root/train.log"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/run_evokv_kuairand_causal_next_item_triangle.py \
  --config "$config" \
  --phase evaluate \
  2>&1 | tee "$root/evaluate.log"
