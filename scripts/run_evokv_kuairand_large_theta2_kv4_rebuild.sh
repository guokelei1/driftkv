#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20.json"
results="results/root_cause_campaign/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20"

test "$(sha256sum "$config" | awk '{print $1}')" = "44d23ebbf58ee037bae0aa9e28cbf0b397f036142988b0928e4dfa9f8f807c84"
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$results"
exec 8>"$results/theta2_kv4_rebuild.lock"
flock -n 8
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only > "$results/theta2_kv4_preflight.json"
jq -e '.completed_versions == 1 and .remaining_versions == 8' "$results/theta2_kv4_preflight.json" > /dev/null

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" \
  --stop-after-version 2 \
  2>&1 | tee "$results/theta2_kv4_rebuild.log"

jq -e '
  .status == "accepted"
  and .candidate.candidate.name == "large_projectionlow_kv4_e4"
  and .candidate.summary.sanity.passed == true
  and .candidate.summary.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent >= 2.0
  and .candidate.summary.comparisons.recompute_over_reuse.mrr.relative_percent > 0
' "$results/edges/theta_2/accepted.json" > /dev/null
