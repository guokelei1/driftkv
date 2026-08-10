#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/evokv_root_cause/kuairand_natural_day_round_03_b_two_gpu_v0.json}"
ROUND_ROOT="$(python -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["outputs"]["training_result"]).parent)' "$CONFIG")"
mkdir -p "$ROUND_ROOT"

python scripts/preflight_evokv_root_cause_kuairand.py \
  --config "$CONFIG" \
  --phase training | tee "$ROUND_ROOT/preflight_training.json"

CUDA_VISIBLE_DEVICES=0 python scripts/train_evokv_root_cause_kuairand.py \
  --config "$CONFIG" 2>&1 | tee -a "$ROUND_ROOT/training.log"

python scripts/validate_evokv_root_cause_kuairand.py \
  --config "$CONFIG" \
  --artifact training | tee "$ROUND_ROOT/training_validation.json"

python scripts/preflight_evokv_root_cause_kuairand.py \
  --config "$CONFIG" \
  --phase evaluation | tee "$ROUND_ROOT/preflight_evaluation.json"

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  scripts/evaluate_evokv_root_cause_kuairand.py \
  --config "$CONFIG" 2>&1 | tee -a "$ROUND_ROOT/evaluation.log"

python scripts/validate_evokv_root_cause_kuairand.py \
  --config "$CONFIG" \
  --artifact evaluation | tee "$ROUND_ROOT/evaluation_validation.json"
