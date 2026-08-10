#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

configs=(
  "configs/evokv_root_cause/kuairand_natural_path_attribution_theta1_theta2_20260810_v0.json"
  "configs/evokv_root_cause/kuairand_natural_path_attribution_theta2_theta3_20260810_v0.json"
  "configs/evokv_root_cause/kuairand_natural_path_attribution_theta3_theta4_20260810_v0.json"
)
hashes=(
  "20b8856933a6360cdd0af8ed4cb271a0fd0fe7758da124ca60c244a6ec333676"
  "c49f0b89687513dc6e6d3cfe282bf8bac3821fa65458ca532ecb460d01f70a25"
  "ea33c7d5e16ca3939308a504d2c14cdbf0ad41a1142e1d03efafb0b177d3b5f6"
)

for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  test "$(sha256sum "$config" | awk '{print $1}')" = "${hashes[$index]}"
  python -c "from hstu_kvcache.streaming.kuairand_natural_path_attribution import load_natural_path_attribution_config; load_natural_path_attribution_config('$config')"
done

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

for config in "${configs[@]}"; do
  output="$(jq -r '.output' "$config")"
  output_root="$(dirname "$output")"
  mkdir -p "$output_root"
  if test -f "$output" && jq -e '.status == "complete_development_attribution" and (.variants | length == 9)' "$output" >/dev/null; then
    continue
  fi
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
    scripts/evaluate_evokv_kuairand_natural_path_attribution.py \
    --config "$config" 2>&1 | tee "$output_root/run.log"
  jq -e '.status == "complete_development_attribution" and (.variants | length == 9)' "$output" >/dev/null
done
