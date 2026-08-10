#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/evokv_root_cause/kuairand_cache_compatible_round_03_c2_two_gpu_v0.json}"
ROUND_ROOT="$(python -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["outputs"]["evaluation_result"]).parent)' "$CONFIG")"
mkdir -p "$ROUND_ROOT"

python scripts/preflight_evokv_root_cause_kuairand_cache_compatible.py \
  --config "$CONFIG" | tee "$ROUND_ROOT/preflight.json"

CUDA_VISIBLE_DEVICES=0 python \
  scripts/train_evokv_root_cause_kuairand_cache_compatible.py \
  --config "$CONFIG" 2>&1 | tee -a "$ROUND_ROOT/training.log"

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  scripts/evaluate_evokv_root_cause_kuairand_cache_compatible.py \
  --config "$CONFIG" 2>&1 | tee -a "$ROUND_ROOT/evaluation.log"

python scripts/validate_evokv_root_cause_kuairand_cache_compatible.py \
  --config "$CONFIG" | tee "$ROUND_ROOT/validation.json"
