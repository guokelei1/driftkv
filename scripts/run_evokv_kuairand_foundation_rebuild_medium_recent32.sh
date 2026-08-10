#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_foundation_rebuild_medium_recent32_theta12_20260809_v4.json"
output_root="results/root_cause_campaign/kuairand_foundation_rebuild_medium_recent32_theta12_20260809_v4"
result="$output_root/result.json"
mkdir -p "$output_root"

if [[ ! -f "$output_root/config_sha256.txt" ]]; then
  sha256sum "$config" > "$output_root/config_sha256.txt"
fi

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" 2>&1 | tee -a "$output_root/run.log"

jq -e '.status == "complete" and .checkpoint_count == 12' "$result" >/dev/null
jq '.decision' "$result"
echo "$result"
