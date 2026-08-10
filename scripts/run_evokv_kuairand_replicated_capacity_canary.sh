#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e512_r8_theta1_theta8_20260810_v30.json"
root="results/root_cause_campaign/kuairand_replicated_capacity_canary_20260810_v30"
result="$root/result.json"

test "$(sha256sum "$config" | awk '{print $1}')" = "554ddac865c5f299a97dc4343f6704cbd0c622fcd367930d07ed33a8b21c4f6b"
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$root"
exec 8>"$root/round.lock"
flock -n 8
nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/canary_evokv_kuairand_replicated_capacity.py \
  --config "$config" \
  --output "$result" \
  2>&1 | tee "$root/run.log"
jq -e '.status == "complete" and .passed == true' "$result" >/dev/null
