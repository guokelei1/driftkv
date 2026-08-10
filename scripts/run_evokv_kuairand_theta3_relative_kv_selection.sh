#!/usr/bin/env bash
set -euo pipefail

config=configs/evokv_root_cause/kuairand_amplified_theta3_relative_kv_selection_20260809_v2.json
selection_root=results/root_cause_campaign/kuairand_amplified_theta3_relative_kv_selection_seed53117_v2
main_root=results/root_cause_campaign/kuairand_amplified_fixed_seed53117_v0
checkpoint_root=checkpoints/evokv_kuairand_amplified_fixed_seed53117_v0
assessment="$selection_root/theta3_assessment.json"

python -c 'from hstu_kvcache.streaming.kuairand_lineage_retrain import load_lineage_retrain_config; load_lineage_retrain_config("configs/evokv_root_cause/kuairand_amplified_theta3_relative_kv_selection_20260809_v2.json")'
test ! -e "$checkpoint_root/theta_3"
mkdir -p "$selection_root/logs"
for version in 1 2; do
  source="$main_root/edges/theta_${version}/accepted.json"
  target="$selection_root/edges/theta_${version}/accepted.json"
  mkdir -p "$(dirname "$target")"
  if [[ ! -f "$target" ]]; then
    cp --archive "$source" "$target"
  fi
  cmp "$source" "$target"
done

{
  date --iso-8601=seconds
  sha256sum "$config"
  df -B1 checkpoints
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
} > "$selection_root/logs/preflight.log"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_lineage_retrain.py \
  --config "$config" > "$selection_root/logs/train.log" 2>&1

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" > "$selection_root/logs/direct_lineage.log" 2>&1

python scripts/validate_evokv_kuairand_theta3_lineage.py \
  --config "$config" \
  --result "$selection_root/result.json" \
  --output "$assessment" > "$selection_root/logs/assessment.log"
jq -e '.decision.passed == true' "$assessment" > /dev/null

mkdir -p "$main_root/edges/theta_3"
cp --archive "$selection_root/edges/theta_3/accepted.json" "$main_root/edges/theta_3/accepted.json"
jq '.decision, .selected_candidate, [.ordinary_holdout_cells[] | {target_version,source_version,cache_age,ranking_relative_percent}]' "$assessment"
