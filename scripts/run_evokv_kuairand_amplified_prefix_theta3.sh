#!/usr/bin/env bash
set -euo pipefail

config=configs/evokv_root_cause/kuairand_amplified_fixed_theta1_theta10_20260809_v0.json
root=results/root_cause_campaign/kuairand_amplified_fixed_seed53117_v0
checkpoint_root=checkpoints/evokv_kuairand_amplified_fixed_seed53117_v0
source_checkpoint=checkpoints/evokv_kuairand_lineage_retrained_seed53117_v5/theta_1
source_accepted=results/root_cause_campaign/kuairand_lineage_retrained_seed53117_v5/edges/theta_1/accepted.json
target_checkpoint="$checkpoint_root/theta_1"
target_accepted="$root/edges/theta_1/accepted.json"
assessment="$root/prefix_theta3_assessment.json"

mkdir -p "$checkpoint_root" "$root/logs" "$(dirname "$target_accepted")"
python -c 'from hstu_kvcache.streaming.kuairand_projected_persistent import load_persistent_config; load_persistent_config("configs/evokv_root_cause/kuairand_amplified_fixed_theta1_theta10_20260809_v0.json")'
sha256sum -c <(jq -r '"\(.selection_source.sha256)  \(.selection_source.path)"' "$config")
jq -e '.selected.name == "uniform2x_kv4x_n16384_e3" and .selected.holdout_opened_after_selection.ranking_relative_percent.ndcg_at_5 > 5' "$(jq -r '.selection_source.path' "$config")" > /dev/null

if [[ ! -e "$target_checkpoint" ]]; then
  cp --archive --link "$source_checkpoint" "$target_checkpoint"
fi
cmp "$source_checkpoint/manifest.json" "$target_checkpoint/manifest.json"
if [[ ! -e "$target_accepted" ]]; then
  temporary="$target_accepted.tmp.$$"
  jq --arg path "$target_checkpoint/manifest.json" '.checkpoint.path = $path' "$source_accepted" > "$temporary"
  mv "$temporary" "$target_accepted"
fi

{
  date --iso-8601=seconds
  sha256sum "$config"
  df -B1 checkpoints
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
} > "$root/logs/prefix_preflight.log"
python scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" \
  --preflight-only > "$root/logs/prefix_code_preflight.log"

if [[ ! -f "$checkpoint_root/theta_3/manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    scripts/train_evokv_kuairand_theta1_theta8.py \
    --config "$config" \
    --stop-after-version 3 > "$root/logs/prefix_theta3_train.log" 2>&1
fi

python scripts/validate_evokv_kuairand_amplified_prefix.py \
  --config "$config" \
  --output "$assessment" > "$root/logs/prefix_theta3_assessment.log"
jq '.decision, [.edges[] | {version,tuning,holdout,holdout_fresh}]' "$assessment"
