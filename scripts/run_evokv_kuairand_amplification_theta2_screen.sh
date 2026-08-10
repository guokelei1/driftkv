#!/usr/bin/env bash
set -euo pipefail

config=configs/evokv_root_cause/kuairand_amplification_theta2_screen_20260809_v1.json
root=results/root_cause_campaign/kuairand_amplification_theta2_screen_seed53117_v1
source_accepted=results/root_cause_campaign/kuairand_lineage_retrained_seed53117_v5/edges/theta_1/accepted.json
seeded_accepted="$root/edges/theta_1/accepted.json"
summary="$root/summary.json"

mkdir -p "$root/candidates" "$root/logs" "$(dirname "$seeded_accepted")"
python -c 'from hstu_kvcache.streaming.kuairand_projected_persistent import load_persistent_config; load_persistent_config("configs/evokv_root_cause/kuairand_amplification_theta2_screen_20260809_v1.json")'
test -f checkpoints/evokv_kuairand_lineage_retrained_seed53117_v5/theta_1/manifest.json
test -f "$source_accepted"
cp -n "$source_accepted" "$seeded_accepted" || true
cmp "$source_accepted" "$seeded_accepted"
{
  date --iso-8601=seconds
  sha256sum "$config"
  df -B1 checkpoints
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
} > "$root/logs/preflight.log"

mapfile -t candidates < <(jq -r '.training.candidate_ladder[].name' "$config")
for candidate in "${candidates[@]}"; do
  output="$root/candidates/$candidate.json"
  log="$root/logs/$candidate.log"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    scripts/probe_evokv_kuairand_candidate.py \
    --config "$config" \
    --version 2 \
    --candidate "$candidate" \
    --output "$output" > "$log" 2>&1
done

python scripts/summarize_evokv_kuairand_amplification_screen.py \
  --config "$config" \
  --root "$root" \
  --output "$summary" > "$root/logs/summary.log"
jq '{status,next,selected:{name:.selected.name,two_x_target_pass:.selected.two_x_target_pass,tuning:.selected.tuning.ranking_relative_percent,amplification:.selected.amplification_over_baseline,holdout:.selected.holdout_opened_after_selection.ranking_relative_percent}}' "$summary"
