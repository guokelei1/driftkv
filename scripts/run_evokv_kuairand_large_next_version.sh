#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 VERSION" >&2
  exit 2
fi

version=$1
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20.json"
results="results/root_cause_campaign/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20"

test "$version" -ge 2
test "$version" -le 9
test "$(sha256sum "$config" | awk '{print $1}')" = "e6be04fd67803b17ef3d778ee5fd65182a854fdfb0f97ad0761e687c71231d64"
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$results"
exec 8>"$results/theta${version}_step.lock"
flock -n 8
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only > "$results/theta${version}_preflight.json"
jq -e --argjson completed "$((version - 1))" '.completed_versions == $completed' "$results/theta${version}_preflight.json" > /dev/null

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" \
  --stop-after-version "$version" \
  2>&1 | tee "$results/theta${version}_step.log"

jq -e '
  .status == "accepted"
  and .candidate.summary.sanity.passed == true
  and .candidate.summary.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent >= 1.0
' "$results/edges/theta_${version}/accepted.json" > /dev/null
