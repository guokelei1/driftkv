#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20.json"
root="results/root_cause_campaign/kuairand_latest_query_large_e4160_theta1_theta3_lineage_20260810_v27"

test "$(sha256sum "$config" | awk '{print $1}')" = "e6be04fd67803b17ef3d778ee5fd65182a854fdfb0f97ad0761e687c71231d64"
for version in 1 2 3; do
  test -f "checkpoints/evokv_kuairand_latest_query_large_e4160_theta1_theta9_frozen_v20/theta_${version}/manifest.json"
  test -f "results/root_cause_campaign/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20/edges/theta_${version}/accepted.json"
done
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$root"
exec 8>"$root/round.lock"
flock -n 8
nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/evaluate_evokv_kuairand_persistent_prefix_lineage.py \
  --config "$config" \
  --final-version 3 \
  --output "$root" \
  2>&1 | tee "$root/run.log"
