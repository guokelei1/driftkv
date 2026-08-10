#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 VERSION CANDIDATE" >&2
  exit 2
fi

version=$1
candidate=$2
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20.json"
root="results/root_cause_campaign/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20/scheduled_probes"

test "$version" -ge 2
test "$version" -le 9
test "$(sha256sum "$config" | awk '{print $1}')" = "e6be04fd67803b17ef3d778ee5fd65182a854fdfb0f97ad0761e687c71231d64"
jq -e --arg version "$version" --arg candidate "$candidate" '.training.candidate_schedule[$version] == $candidate' "$config" >/dev/null
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$root"
exec 8>"$root/theta_${version}_${candidate}.lock"
flock -n 8
nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$root/theta_${version}_${candidate}_gpu_preflight.csv"

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/probe_evokv_kuairand_candidate.py \
  --config "$config" \
  --version "$version" \
  --candidate "$candidate" \
  --output "$root/theta_${version}_${candidate}.json" \
  2>&1 | tee "$root/theta_${version}_${candidate}.log"
