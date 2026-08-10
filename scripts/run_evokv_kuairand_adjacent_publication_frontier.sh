#!/usr/bin/env bash
set -euo pipefail

config=configs/evokv_root_cause/kuairand_adjacent_publication_frontier_20260809_v0.json
output_root=results/root_cause_campaign/kuairand_adjacent_publication_frontier_20260809_v0
checkpoint_root=checkpoints/evokv_kuairand_theta6_dense_interpolation_seed53117_v0
ledger=configs/evokv_root_cause/kuairand_checkpoint_retention_theta0_theta10_20260809_v1.json

mkdir -p "$output_root/logs"
python scripts/evaluate_evokv_kuairand_adjacent_publication_frontier.py \
  --config "$config" \
  --preflight-only > "$output_root/logs/code_preflight.json"

for version in 1 2 3 4 5 6 7 8 9 10; do
  expected=$(jq -r ".retained.theta1_theta10.manifest_sha256.theta$version" "$ledger")
  actual=$(sha256sum "$checkpoint_root/theta_$version/manifest.json" | awk '{print $1}')
  test "$expected" = "$actual"
done

mapfile -t gpu_free_mib < <(
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
)
test "${#gpu_free_mib[@]}" -ge 2
test "${gpu_free_mib[0]}" -ge 43000
test "${gpu_free_mib[1]}" -ge 43000

{
  date --iso-8601=seconds
  sha256sum "$config" "$ledger"
  df -B1 checkpoints
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
} > "$output_root/logs/resource_preflight.log"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/evaluate_evokv_kuairand_adjacent_publication_frontier.py \
  --config "$config" > "$output_root/logs/evaluation.log" 2>&1

jq '.summaries, .decision' "$output_root/result.json"
