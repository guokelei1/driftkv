#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_medium_theta1_theta9_frozen_20260810_v18.json"
source_config="configs/evokv_root_cause/kuairand_latest_query_medium_theta1_theta10_selection_20260810_v17.json"
source_checkpoints="checkpoints/evokv_kuairand_latest_query_medium_theta1_theta10_selection_v17"
source_results="results/root_cause_campaign/kuairand_latest_query_medium_theta1_theta10_selection_20260810_v17"
checkpoints="checkpoints/evokv_kuairand_latest_query_medium_theta1_theta9_frozen_v18"
results="results/root_cause_campaign/kuairand_latest_query_medium_theta1_theta9_frozen_20260810_v18"

test "$(sha256sum "$source_config" | awk '{print $1}')" = "f49669e2f16d6f3244fd45f34357bb341ccbd1769372ba5c63fb3d16418eb631"

used="$(nvidia-smi --id=0 --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
test "$used" -le 512

mkdir -p "$checkpoints" "$results"
for version in $(seq 1 9); do
  if test ! -e "$checkpoints/theta_$version"; then
    test -f "$source_checkpoints/theta_$version/manifest.json"
    cp -al "$source_checkpoints/theta_$version" "$checkpoints/theta_$version"
  fi
  if test ! -f "$results/edges/theta_$version/accepted.json"; then
    test -f "$source_results/edges/theta_$version/accepted.json"
    mkdir -p "$results/edges/theta_$version"
    cp -a "$source_results/edges/theta_$version/accepted.json" "$results/edges/theta_$version/accepted.json"
  fi
done

exec 8>"$results/round.lock"
flock -n 8
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$results/gpu_preflight.csv"
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only > "$results/preflight.log"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" 2>&1 | tee "$results/run.log"
python scripts/render_evokv_kuairand_reuse_loss_table.py --result "$results/result.json" > "$results/render.log"
