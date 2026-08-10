#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/evokv_root_cause/kuairand_untied_cache_path_screen_20260808_v0.json}"
ROOT="results/root_cause_campaign/kuairand_untied_cache_path_screen_20260808_v0"
mkdir -p "$ROOT"
test -f "$CONFIG"
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2
CUDA_VISIBLE_DEVICES=0 python scripts/train_evokv_kuairand_untied_cache_path_screen.py \
  --config "$CONFIG" 2>&1 | tee -a "$ROOT/training.log"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/evaluate_evokv_kuairand_untied_cache_path_screen.py \
  --config "$CONFIG" 2>&1 | tee -a "$ROOT/evaluation.log"
