#!/usr/bin/env bash
set -euo pipefail

config=configs/evokv_root_cause/kuairand_lineage_distribution_audit_20260809_v6.json
output=results/root_cause_campaign/kuairand_lineage_distribution_audit_seed53117_v6
source_output=results/root_cause_campaign/kuairand_lineage_retrained_seed53117_v5

test -d checkpoints/evokv_kuairand_lineage_retrained_seed53117_v5
test ! -f "$output/result.json" || exit 2
mkdir -p "$output"

for version in 1 2 3 4 5 6 7 8; do
  source_accepted="$source_output/edges/theta_${version}/accepted.json"
  target_accepted="$output/edges/theta_${version}/accepted.json"
  test -f "$source_accepted"
  if test ! -f "$target_accepted"; then
    mkdir -p "$(dirname "$target_accepted")"
    cp "$source_accepted" "$target_accepted"
  fi
done

python scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" \
  --preflight-only \
  > "$output/preflight.json"

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1

torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" \
  2>&1 | tee "$output/audit.log"
