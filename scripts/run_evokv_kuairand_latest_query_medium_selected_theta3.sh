#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_medium_selected_theta1_theta3_20260810_v15.json"
source_config="configs/evokv_root_cause/kuairand_latest_query_medium_projectionlow_theta1_theta3_20260810_v13.json"
source_checkpoints="checkpoints/evokv_kuairand_latest_query_medium_projectionlow_theta1_theta3_v13"
source_results="results/root_cause_campaign/kuairand_latest_query_medium_projectionlow_theta1_theta3_20260810_v13"
checkpoints="checkpoints/evokv_kuairand_latest_query_medium_selected_theta1_theta3_v15"
results="results/root_cause_campaign/kuairand_latest_query_medium_selected_theta1_theta3_20260810_v15"

test "$(sha256sum "$source_config" | awk '{print $1}')" = "a63d107e2fa622cfac34f8b1daa27b58d0ad67b961317ab94ee519bca09a4cb9"
test -f "$source_checkpoints/theta_1/manifest.json"
test -f "$source_checkpoints/theta_2/manifest.json"
test -f "$source_results/edges/theta_1/accepted.json"
test -f "$source_results/edges/theta_2/accepted.json"

used="$(nvidia-smi --id=0 --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
test "$used" -le 512

mkdir -p "$checkpoints" "$results/edges/theta_1" "$results/edges/theta_2"
if test ! -e "$checkpoints/theta_1"; then
  cp -al "$source_checkpoints/theta_1" "$checkpoints/theta_1"
fi
if test ! -e "$checkpoints/theta_2"; then
  cp -al "$source_checkpoints/theta_2" "$checkpoints/theta_2"
fi
cp -a "$source_results/edges/theta_1/accepted.json" "$results/edges/theta_1/accepted.json"
cp -a "$source_results/edges/theta_2/accepted.json" "$results/edges/theta_2/accepted.json"

exec 8>"$results/round.lock"
flock -n 8
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$results/gpu_preflight.csv"

python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only > "$results/preflight.log"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" 2>&1 | tee "$results/run.log"
python scripts/render_evokv_kuairand_reuse_loss_table.py --result "$results/result.json" > "$results/render.log"
