#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 CANDIDATE_CONFIG OUTPUT_ROOT [VERSION]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

candidate_config="$1"
output_root="$2"
version="${3:-6}"
source_config="$(jq -r '.source_config.path' "$candidate_config")"
expected_source_sha256="$(jq -r '.source_config.sha256' "$candidate_config")"
candidate_config_sha256="$(sha256sum "$candidate_config" | awk '{print $1}')"
lock_path="$output_root/sweep.lock"
mapfile -t candidates < <(jq -r '.candidates[].name' "$candidate_config")

test "${#candidates[@]}" -gt 0
test "$(sha256sum "$source_config" | awk '{print $1}')" = "$expected_source_sha256"
mkdir -p "$output_root"

if [[ -f "$output_root/config_sha256.txt" ]]; then
  test "$(awk '{print $1}' "$output_root/config_sha256.txt")" = "$candidate_config_sha256"
else
  sha256sum "$candidate_config" > "$output_root/config_sha256.txt"
fi

exec 9>"$lock_path"
if ! flock -n 9; then
  echo "another large candidate probe sweep holds $lock_path" >&2
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
    --candidate-config "$candidate_config" \
    --version "$version" \
    --candidate "$candidate" \
    --output "$output" 2>&1 | tee "$log"
  jq -e --arg candidate "$candidate" '.status == "complete" and .candidate.name == $candidate' "$output" >/dev/null
done

for candidate in "${candidates[@]}"; do
  jq -c '{candidate:.candidate.name,all_ndcg:.summary.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent,tuning_ndcg:.partition_summaries.tuning.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent,holdout_ndcg:.partition_summaries.holdout.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent,holdout_mrr:.partition_summaries.holdout.comparisons.recompute_over_reuse.mrr.relative_percent,holdout_hr5:.partition_summaries.holdout.comparisons.recompute_over_reuse.hit_rate_at_5.relative_percent,holdout_fresh:.partition_summaries.holdout.endpoints.recompute.ndcg_at_5,holdout_reuse:.partition_summaries.holdout.endpoints.reuse.ndcg_at_5}' "$output_root/${candidate}.json"
done
