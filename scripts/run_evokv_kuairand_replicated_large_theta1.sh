#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e512_r8_theta1_theta8_20260810_v30.json"
config_sha256="554ddac865c5f299a97dc4343f6704cbd0c622fcd367930d07ed33a8b21c4f6b"
checkpoint_root="checkpoints/evokv_kuairand_latest_query_large_e512_r8_theta1_theta8_v30"
result_root="results/root_cause_campaign/kuairand_latest_query_large_e512_r8_theta1_theta8_20260810_v30"

test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha256"
mkdir -p "$result_root/logs"
exec 8>"$result_root/theta1.lock"
flock -n 8

if [[ -f "$checkpoint_root/theta_1/manifest.json" && -f "$result_root/edges/theta_1/accepted.json" ]]; then
  jq -e '.status == "accepted" and .version == 1' "$result_root/edges/theta_1/accepted.json" >/dev/null
  exit 0
fi

test ! -e "$checkpoint_root/theta_1/manifest.json"
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$result_root/logs/theta1_gpu_preflight.csv"
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only > "$result_root/logs/theta1_preflight.json"
jq -e '.status == "ready" and .completed_versions == 0 and .remaining_versions == 8' "$result_root/logs/theta1_preflight.json" >/dev/null

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" \
  --stop-after-version 1 \
  2>&1 | tee "$result_root/logs/theta1_train.log"

jq -e '
  .status == "accepted"
  and .version == 1
  and .candidate.summary.sanity.passed == true
' "$result_root/edges/theta_1/accepted.json" >/dev/null
test -f "$checkpoint_root/theta_1/manifest.json"
