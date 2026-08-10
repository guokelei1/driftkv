#!/usr/bin/env bash
set -euo pipefail

config=configs/evokv_root_cause/kuairand_projected_lineage_retrain_theta8_20260808_v5.json
target_checkpoints=checkpoints/evokv_kuairand_lineage_retrained_seed53117_v5
target_results=results/root_cause_campaign/kuairand_lineage_retrained_seed53117_v5

for version in 1 2 3 4 5 6 7; do
  test -f "$target_checkpoints/theta_${version}/manifest.json" || exit 2
done
test -d "$target_results" || exit 2

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1

torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_lineage_retrain.py \
  --config "$config" \
  2>&1 | tee "$target_results/train.log"

torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" \
  2>&1 | tee "$target_results/full_triangle.log"

python scripts/render_evokv_kuairand_reuse_loss_table.py \
  --result "$target_results/result.json"
