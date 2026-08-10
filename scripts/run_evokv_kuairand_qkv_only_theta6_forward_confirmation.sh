#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_config="configs/evokv_root_cause/kuairand_foundation_rebuild_large_e4160_theta12_20260809_v7.json"
candidate_config="configs/evokv_root_cause/kuairand_qkv_only_theta6_forward_confirmation_20260810_v0.json"
candidate_sha256="6f2c48d3a48f9fb2cfcbddc9cfe1008dda44fe9032e743c778250c27578c6bce"
output_root="results/root_cause_campaign/kuairand_qkv_only_theta6_forward_confirmation_20260810_v0"
output="$output_root/result.json"
log="$output_root/run.log"

test "$(sha256sum "$candidate_config" | awk '{print $1}')" = "$candidate_sha256"
python -c "from hstu_kvcache.streaming.kuairand_projected_persistent import load_candidate_probe_config; load_candidate_probe_config('$candidate_config', '$source_config')"
mkdir -p "$output_root"

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

if ! test -f "$output"; then
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
    scripts/probe_evokv_kuairand_candidate.py \
    --config "$source_config" \
    --candidate-config "$candidate_config" \
    --version 6 \
    --candidate latest1_qkv_only_kv10x_e2 \
    --output "$output" 2>&1 | tee "$log"
fi

jq -e '.status == "complete" and .candidate.dense_update_scope == "qkv_only" and .version == 6' "$output" >/dev/null
