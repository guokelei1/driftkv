#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/evokv_root_cause/kuairand_kv_only_candidate_triangle_20260808_v0.json}"
LOG="results/root_cause_campaign/kuairand_kv_only_candidate_triangle_20260808_v0/run.log"
mkdir -p "$(dirname "$LOG")"
test -f "$CONFIG"
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/evaluate_evokv_kuairand_kv_only_candidate_triangle.py \
  --config "$CONFIG" 2>&1 | tee -a "$LOG"
