#!/usr/bin/env bash
set -euo pipefail

config=configs/evokv_root_cause/kuairand_amplified_theta5_theta10_lineage_20260809_v0.json
protocol=configs/evokv_root_cause/kuairand_extended_triangle_acceptance_20260809_v0.json
selection_root=results/root_cause_campaign/kuairand_amplified_theta5_theta10_lineage_seed53117_v0
main_root=results/root_cause_campaign/kuairand_amplified_fixed_seed53117_v0
checkpoint_root=checkpoints/evokv_kuairand_amplified_fixed_seed53117_v0
assessment="$selection_root/theta10_extended_acceptance.json"

python -c 'from hstu_kvcache.streaming.kuairand_lineage_retrain import load_lineage_retrain_config; load_lineage_retrain_config("configs/evokv_root_cause/kuairand_amplified_theta5_theta10_lineage_20260809_v0.json")'
for version in 5 6 7 8 9 10; do
  test ! -e "$checkpoint_root/theta_${version}"
done
mkdir -p "$selection_root/logs"
for version in 1 2 3 4; do
  source="$main_root/edges/theta_${version}/accepted.json"
  target="$selection_root/edges/theta_${version}/accepted.json"
  mkdir -p "$(dirname "$target")"
  cp --archive "$source" "$target"
done

{
  date --iso-8601=seconds
  sha256sum "$config" "$protocol"
  df -B1 checkpoints
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
} > "$selection_root/logs/preflight.log"

python scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" \
  --preflight-only > "$selection_root/logs/code_preflight.json"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_lineage_retrain.py \
  --config "$config" > "$selection_root/logs/train.log" 2>&1

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" > "$selection_root/logs/direct_lineage.log" 2>&1

python scripts/audit_evokv_kuairand_extended_triangle.py \
  --protocol "$protocol" \
  --result "$selection_root/result.json" \
  --output "$assessment" > "$selection_root/logs/assessment.log"
jq -e '.decision.passed == true and .decision.preferred_size_reached == true' "$assessment" > /dev/null

for version in 5 6 7 8 9 10; do
  source="$selection_root/edges/theta_${version}/accepted.json"
  target="$main_root/edges/theta_${version}/accepted.json"
  mkdir -p "$(dirname "$target")"
  cp --archive "$source" "$target"
done
cp --archive "$selection_root/result.json" "$main_root/extended_theta10_result.json"
cp --archive "$selection_root/reuse_loss_table.md" "$main_root/extended_theta10_reuse_loss_table.md"
cp --archive "$assessment" "$main_root/extended_theta10_acceptance.json"

jq '.metrics, .negative_entries, .cumulative, .decision' "$assessment"
