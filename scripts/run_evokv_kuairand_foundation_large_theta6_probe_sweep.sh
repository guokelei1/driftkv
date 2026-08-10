#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_config="configs/evokv_root_cause/kuairand_foundation_rebuild_large_e4160_theta12_20260809_v7.json"
source_sha256="f254fb5efcbee1e4756aa21cfb11c668b6dbf326851e48dd24d80b49001d10eb"
probe_config="configs/evokv_root_cause/kuairand_foundation_rebuild_large_theta6_probes_20260810_v1.json"
probe_sha256="bf152b797e63a80efa51d4a27ae84e8c0ba47f1dcd307afa2182d0debb9a0875"
output_root="results/root_cause_campaign/kuairand_foundation_rebuild_large_theta6_probes_20260810_v1"
lock_path="$output_root/sweep.lock"
candidates=(
  recent32_e3_kv015
  recent32_e3_kv030
  recent16_e2_kv005
  recent16_e3_kv010
  global_standard_n8192_e2_kv001
  global_kv_isolated_n16384_e2_kv002
  dense_interp_n8192_e2
  projection_dominant_n8192_e2
)

mkdir -p "$output_root"
test "$(sha256sum "$source_config" | awk '{print $1}')" = "$source_sha256"
test "$(sha256sum "$probe_config" | awk '{print $1}')" = "$probe_sha256"

exec 9>"$lock_path"
if ! flock -n 9; then
  echo "another large theta6 probe sweep holds $lock_path" >&2
  exit 1
fi

for candidate in "${candidates[@]}"; do
  output="$output_root/${candidate}.json"
  log="$output_root/${candidate}.log"
  if [[ -f "$output" ]]; then
    jq -e --arg candidate "$candidate" '.status == "complete" and .candidate.name == $candidate' "$output" >/dev/null
    continue
  fi
  for gpu in 0 1; do
    used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
    if (( used > 512 )); then
      echo "GPU${gpu} is not available: ${used} MiB used by compute processes" >&2
      exit 1
    fi
  done
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
    scripts/probe_evokv_kuairand_candidate.py \
    --config "$source_config" \
    --candidate-config "$probe_config" \
    --version 6 \
    --candidate "$candidate" \
    --output "$output" 2>&1 | tee "$log"
  jq -e --arg candidate "$candidate" '.status == "complete" and .candidate.name == $candidate' "$output" >/dev/null
done

for candidate in "${candidates[@]}"; do
  jq -c '{candidate:.candidate.name,all_ndcg:.summary.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent,tuning_ndcg:.partition_summaries.tuning.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent,holdout_ndcg:.partition_summaries.holdout.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent,holdout_mrr:.partition_summaries.holdout.comparisons.recompute_over_reuse.mrr.relative_percent,holdout_hr5:.partition_summaries.holdout.comparisons.recompute_over_reuse.hit_rate_at_5.relative_percent,holdout_fresh:.partition_summaries.holdout.endpoints.recompute.ndcg_at_5,holdout_reuse:.partition_summaries.holdout.endpoints.reuse.ndcg_at_5}' "$output_root/${candidate}.json"
done
