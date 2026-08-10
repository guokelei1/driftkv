#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_low="configs/evokv_root_cause/kuairand_latest_query_medium_projectionlow_theta1_theta3_20260810_v13.json"
config_train="configs/evokv_root_cause/kuairand_latest_query_medium_projectiontrain_theta1_theta3_20260810_v13.json"
root_low="results/root_cause_campaign/kuairand_latest_query_medium_projectionlow_theta1_theta3_20260810_v13"
root_train="results/root_cause_campaign/kuairand_latest_query_medium_projectiontrain_theta1_theta3_20260810_v13"

test "$(sha256sum "$config_low" | awk '{print $1}')" = "a63d107e2fa622cfac34f8b1daa27b58d0ad67b961317ab94ee519bca09a4cb9"
test "$(sha256sum "$config_train" | awk '{print $1}')" = "e288b1c0592875c95f3743b0ff7417fae9384bcb94e58c4abbd862ffdb2ffad8"

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$root_low" "$root_train"
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config_low" --preflight-only > "$root_low/preflight.log"
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config_train" --preflight-only > "$root_train/preflight.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$root_low/gpu_preflight.csv"
cp "$root_low/gpu_preflight.csv" "$root_train/gpu_preflight.csv"

exec 8>"$root_low/round.lock"
exec 9>"$root_train/round.lock"
flock -n 8
flock -n 9

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config_low" >> "$root_low/run.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config_train" >> "$root_train/run.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
pids=()
test "$status" -eq 0

python scripts/render_evokv_kuairand_reuse_loss_table.py --result "$root_low/result.json" > "$root_low/render.log"
python scripts/render_evokv_kuairand_reuse_loss_table.py --result "$root_train/result.json" > "$root_train/render.log"

trap - INT TERM EXIT
