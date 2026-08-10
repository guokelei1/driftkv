#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  printf 'usage: %s VERSION\n' "$0" >&2
  exit 2
fi

version="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e512_r8_theta1_theta8_20260811_v34.json"
config_sha256="907034295b508f64e259a92c22832f27188e66902133c01aa79adab687773282"
checkpoint_root="checkpoints/evokv_kuairand_latest_query_large_e512_r8_theta1_theta8_v30"
result_root="results/root_cause_campaign/kuairand_latest_query_large_e512_r8_theta1_theta8_20260810_v30"
lineage_root="$result_root/prefix_lineage_v34/theta1_theta${version}"

test "$version" -ge 2
test "$version" -le 8
test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha256"
for prior in $(seq 1 "$((version - 1))"); do
  test -f "$checkpoint_root/theta_${prior}/manifest.json"
  test -f "$result_root/edges/theta_${prior}/accepted.json"
done

mkdir -p "$result_root/logs" "$lineage_root"
exec 8>"$result_root/theta${version}.lock"
flock -n 8
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$result_root/logs/theta${version}_gpu_preflight.csv"
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only > "$result_root/logs/theta${version}_preflight.json"
jq -e --argjson expected "$((version - 1))" '.status == "ready" and .completed_versions == $expected' "$result_root/logs/theta${version}_preflight.json" >/dev/null

if [[ ! -f "$checkpoint_root/theta_${version}/manifest.json" || ! -f "$result_root/edges/theta_${version}/accepted.json" ]]; then
  test ! -e "$checkpoint_root/theta_${version}/manifest.json"
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
    scripts/train_evokv_kuairand_theta1_theta8.py \
    --config "$config" \
    --stop-after-version "$version" \
    2>&1 | tee "$result_root/logs/theta${version}_train.log"
fi

jq -e --argjson expected "$version" '
  .status == "accepted"
  and .version == $expected
  and .candidate.summary.sanity.passed == true
' "$result_root/edges/theta_${version}/accepted.json" >/dev/null
test -f "$checkpoint_root/theta_${version}/manifest.json"

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/evaluate_evokv_kuairand_persistent_prefix_lineage.py \
  --config "$config" \
  --final-version "$version" \
  --output "$lineage_root" \
  2>&1 | tee "$result_root/logs/theta${version}_lineage.log"

jq -e --argjson expected "$version" '
  .status == "complete"
  and .checkpoint_count == $expected
  and (.targets | length) == $expected
  and .prefix_lineage.evaluated_final_version == $expected
' "$lineage_root/result.json" >/dev/null
jq --argjson expected "$version" '{
  target_version: $expected,
  row: [.targets[$expected - 1].lineage[] | select(.source_version >= 1) | {
    source_version,
    mrr_relative_percent: .summary.comparisons.recompute_over_reuse.mrr.relative_percent,
    ndcg_at_5_relative_percent: .summary.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent,
    hit_rate_at_5_relative_percent: .summary.comparisons.recompute_over_reuse.hit_rate_at_5.relative_percent,
    reuse_ndcg_at_5: .summary.endpoints.reuse.ndcg_at_5,
    recompute_ndcg_at_5: .summary.endpoints.recompute.ndcg_at_5
  }]
}' "$lineage_root/result.json" > "$lineage_root/latest_row.json"
