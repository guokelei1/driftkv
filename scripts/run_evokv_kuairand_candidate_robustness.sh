#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1

config=configs/evokv_root_cause/kuairand_projected_large_imported_anchor_kv2_seed53117_theta1_theta8_20260808_v2.json
root=results/opportunity_discovery/evokv_kuairand_imported_anchor_kv2_seed53117_v2/candidate_robustness
mkdir -p "$root"

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/evaluate_evokv_kuairand_candidate_robustness.py \
  --config "$config" \
  --output "$root/result.json" \
  2>&1 | tee "$root/run.log"
