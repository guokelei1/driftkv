#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

profile="${1:-full}"
case "$profile" in
  canary)
    expected_edges=1
    ;;
  full)
    expected_edges=7
    ;;
  *)
    echo "usage: $0 {canary|full}" >&2
    exit 2
    ;;
esac

config="configs/evokv_d1/development/kuairand_large_hotkv_adjacent_timing_20260811_v0.json"
config_sha256="898a243a33d77e5d55eef5e3b759cd44c60b4dae6efa14ebecda7b0d5602f9db"
output="results/design1/kuairand_large_hotkv_adjacent_timing_20260811_v0/$profile"
logs="$output/logs"

test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha256"
scripts/run_evokv_kuairand_large_baseline_rebuild.sh verify > /dev/null

mkdir -p "$logs"
exec 9>"$output/round.lock"
flock -n 9

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done
nvidia-smi --id=0,1 --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader \
  > "$logs/gpu_preflight.csv"

python scripts/benchmark_evokv_kuairand_large_hotkv.py \
  --config "$config" --profile "$profile" --preflight-only \
  | tee "$logs/preflight.json"

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/benchmark_evokv_kuairand_large_hotkv.py \
  --config "$config" --profile "$profile" \
  2>&1 | tee "$logs/run.log"

jq -e --arg profile "$profile" --argjson edges "$expected_edges" '
  .status == "complete"
  and .profile == $profile
  and .timing_scope.primary_scope == "hot_hbm_hstu_core_only"
  and (.edges | length) == $edges
  and all(.edges[]; .reuse_median_pass_ms > 0 and .recompute_median_pass_ms > 0)
' "$output/result.json" > /dev/null

jq '{profile, aggregate, edges}' "$output/result.json"
