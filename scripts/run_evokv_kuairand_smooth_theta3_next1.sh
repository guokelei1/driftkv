#!/usr/bin/env bash
set -euo pipefail

config=configs/evokv_root_cause/kuairand_smooth_theta3_next1_20260809_v0.json
root=results/root_cause_campaign/kuairand_stationary_prefix_low_seed53117_v0

python -c 'from hstu_kvcache.streaming.kuairand_lineage_retrain import load_lineage_retrain_config; load_lineage_retrain_config("configs/evokv_root_cause/kuairand_smooth_theta3_next1_20260809_v0.json")'
python -c 'from hstu_kvcache.streaming.kuairand_projected_persistent import preflight_persistent_chain; import json; print(json.dumps(preflight_persistent_chain("configs/evokv_root_cause/kuairand_smooth_theta3_next1_20260809_v0.json"), sort_keys=True))' > "$root/logs/smooth_theta3_preflight.json"

mapfile -t gpu_free_mib < <(
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
)
test "${gpu_free_mib[0]}" -ge 43000
test "${gpu_free_mib[1]}" -ge 43000

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_lineage_retrain.py \
  --config "$config" > "$root/logs/smooth_theta3_lineage.log" 2>&1

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" > "$root/logs/smooth_theta3_direct.log" 2>&1

jq '{round_id,decision}' "$root/result.json"
