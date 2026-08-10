#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/evokv_root_cause/kuairand_path_attribution_round_03_c1_two_gpu_v0.json}"
ROUND_ROOT="$(python -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["outputs"]["result"]).parent)' "$CONFIG")"
mkdir -p "$ROUND_ROOT"

python scripts/preflight_evokv_root_cause_kuairand_path_attribution.py \
  --config "$CONFIG" | tee "$ROUND_ROOT/preflight.json"

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  scripts/evaluate_evokv_root_cause_kuairand_path_attribution.py \
  --config "$CONFIG" 2>&1 | tee -a "$ROUND_ROOT/evaluation.log"

python scripts/validate_evokv_root_cause_kuairand_path_attribution.py \
  --config "$CONFIG" | tee "$ROUND_ROOT/validation.json"
