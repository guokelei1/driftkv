#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_natural_path_attribution_theta4_theta5_20260810_v0.json"
expected_sha256="63ba326c989bb79bacafa123266ac28d88d6ad3e0e5a66a276200c3378b3e9f3"
output_root="results/root_cause_campaign/kuairand_natural_path_attribution_theta4_theta5_20260810_v0"
log="$output_root/run.log"

test "$(sha256sum "$config" | awk '{print $1}')" = "$expected_sha256"
python -c "from hstu_kvcache.streaming.kuairand_natural_path_attribution import load_natural_path_attribution_config; load_natural_path_attribution_config('$config')"
mkdir -p "$output_root"

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/evaluate_evokv_kuairand_natural_path_attribution.py \
  --config "$config" 2>&1 | tee "$log"

jq -e '.status == "complete_development_attribution" and (.variants | length == 9)' "$output_root/result.json" >/dev/null
