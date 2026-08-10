#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-configs/evokv_root_cause/qk_attribution_round_03_a_two_gpu_v0.json}"
ROUND_ROOT="results/root_cause_campaign/evokv_root_cause_20260807_v0/round_03_a_qk_attribution"
RESULT="$ROUND_ROOT/result.json"
LOG="$ROUND_ROOT/run.log"

mkdir -p "$ROUND_ROOT"

if [[ -f "$RESULT" ]]; then
  python scripts/validate_evokv_root_cause_qk_attribution.py --config "$CONFIG"
  exit 0
fi

CUDA_VISIBLE_DEVICES=0,1 python scripts/preflight_evokv_root_cause_qk_attribution.py --config "$CONFIG"

set +e
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/evaluate_evokv_root_cause_qk_attribution.py --config "$CONFIG" \
  2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "$STATUS" -ne 0 ]]; then
  exit "$STATUS"
fi

python scripts/validate_evokv_root_cause_qk_attribution.py --config "$CONFIG"
