#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_stationary_coordinate_control_theta5_v1_v8_20260810_v1.json"
expected_sha256="0d5c1e820cfcfe3331f823071a281c9bdcb4a1eab977e159f38133e68d11874a"
output_root="results/root_cause_campaign/kuairand_stationary_coordinate_control_theta5_v1_v8_20260810_v1"
log="$output_root/run.log"

test "$(sha256sum "$config" | awk '{print $1}')" = "$expected_sha256"
python -c "from hstu_kvcache.streaming.kuairand_stationary_coordinate_control import load_stationary_coordinate_control_config; load_stationary_coordinate_control_config('$config')"
mkdir -p "$output_root"

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/evaluate_evokv_kuairand_stationary_coordinate_control.py \
  --config "$config" 2>&1 | tee "$log"

jq -e '.status == "complete_development_causal_control" and .fresh_function_invariance.passed' "$output_root/result.json" >/dev/null
