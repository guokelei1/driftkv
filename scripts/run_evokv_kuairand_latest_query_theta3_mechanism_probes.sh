#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_single="configs/evokv_root_cause/kuairand_latest_query_medium_projectionlow_theta1_theta3_20260810_v13.json"
source_replay="configs/evokv_root_cause/kuairand_latest_query_medium_projectionlow_replay2_probe_source_20260810_v14.json"
probes_single="configs/evokv_root_cause/kuairand_latest_query_theta3_single_day_probes_20260810_v14.json"
probes_replay="configs/evokv_root_cause/kuairand_latest_query_theta3_replay2_probes_20260810_v14.json"
root="results/root_cause_campaign/kuairand_latest_query_theta3_mechanism_probes_20260810_v14"

test "$(sha256sum "$source_single" | awk '{print $1}')" = "a63d107e2fa622cfac34f8b1daa27b58d0ad67b961317ab94ee519bca09a4cb9"
test "$(sha256sum "$source_replay" | awk '{print $1}')" = "fa19ec491127dbd06c45d116bcdb1eaadd0f1bd0f5deaeab64b51a31b82b847b"
test "$(sha256sum "$probes_single" | awk '{print $1}')" = "4599ce749f5693a5421eb6267d1efc7f89ca0fcf621f2933172e529d2ab0e873"
test "$(sha256sum "$probes_replay" | awk '{print $1}')" = "4961cbaf44c013d6bf1a7768e6a39e021b86b179f0645ce9e5ee44ba88aeec32"

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$root"
exec 8>"$root/round.lock"
flock -n 8
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"

run_probe() {
  local gpu="$1"
  local source="$2"
  local probes="$3"
  local candidate="$4"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python scripts/probe_evokv_kuairand_candidate.py \
    --config "$source" \
    --version 3 \
    --candidate "$candidate" \
    --candidate-config "$probes" \
    --output "$root/$candidate.json" \
    > "$root/$candidate.log" 2>&1
}

(
  run_probe 0 "$source_single" "$probes_single" single_full_e2
  run_probe 0 "$source_single" "$probes_single" single_half_e2
  run_probe 0 "$source_replay" "$probes_replay" replay2_pooled_full_e1
  run_probe 0 "$source_replay" "$probes_replay" replay2_sequential_quarter_e1
) &
pid0="$!"

(
  run_probe 1 "$source_single" "$probes_single" single_full_e3
  run_probe 1 "$source_single" "$probes_single" single_half_e4
  run_probe 1 "$source_replay" "$probes_replay" replay2_pooled_half_e2
  run_probe 1 "$source_replay" "$probes_replay" replay2_sequential_half_e1
) &
pid1="$!"

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
test "$status" -eq 0

for result in "$root"/*.json; do
  jq -e '.status == "complete" and .summary.sanity.passed == true' "$result" > /dev/null
done
