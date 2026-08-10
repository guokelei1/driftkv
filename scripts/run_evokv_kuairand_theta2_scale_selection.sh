#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1

config=configs/evokv_root_cause/kuairand_projected_theta1_theta8_seed53117_20260808_v0.json
root=results/opportunity_discovery/evokv_kuairand_theta2_scale_selection_20260808_v0
mkdir -p "$root"

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"

for candidate in standard_n8192_e2_l004 kv2x_n8192_e2 kv4x_n8192_e2; do
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/probe_evokv_kuairand_candidate.py \
    --config "$config" \
    --version 2 \
    --candidate "$candidate" \
    --output "$root/$candidate.json" \
    2>&1 | tee "$root/$candidate.log"
done
