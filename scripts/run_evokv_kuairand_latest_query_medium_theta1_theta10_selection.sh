#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_medium_theta1_theta10_selection_20260810_v17.json"
source_config="configs/evokv_root_cause/kuairand_latest_query_medium_selected_theta1_theta3_20260810_v15.json"
source_checkpoints="checkpoints/evokv_kuairand_latest_query_medium_selected_theta1_theta3_v15"
source_results="results/root_cause_campaign/kuairand_latest_query_medium_selected_theta1_theta3_20260810_v15"
checkpoints="checkpoints/evokv_kuairand_latest_query_medium_theta1_theta10_selection_v17"
results="results/root_cause_campaign/kuairand_latest_query_medium_theta1_theta10_selection_20260810_v17"

test "$(sha256sum "$source_config" | awk '{print $1}')" = "5562675a96c2ed705b881809e23a18317879c154f10d8735d3f7c6abea406cc0"
test -f "$source_results/result.json"
for version in 1 2 3; do
  test -f "$source_checkpoints/theta_$version/manifest.json"
  test -f "$source_results/edges/theta_$version/accepted.json"
done

used="$(nvidia-smi --id=0 --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
test "$used" -le 512

mkdir -p "$checkpoints" "$results"
for version in 1 2 3; do
  if test ! -e "$checkpoints/theta_$version"; then
    cp -al "$source_checkpoints/theta_$version" "$checkpoints/theta_$version"
  fi
  mkdir -p "$results/edges/theta_$version"
  cp -a "$source_results/edges/theta_$version/accepted.json" "$results/edges/theta_$version/accepted.json"
done

exec 8>"$results/round.lock"
flock -n 8
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$results/gpu_preflight.csv"
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only > "$results/preflight.log"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" 2>&1 | tee "$results/run.log"
python scripts/render_evokv_kuairand_reuse_loss_table.py --result "$results/result.json" > "$results/render.log"
