#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

phase="${1:-prefix3}"
case "$phase" in
  prefix3)
    final_version=3
    ;;
  full8)
    final_version=8
    ;;
  *)
    exit 2
    ;;
esac

config="configs/evokv_root_cause/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37.json"
config_sha256="0d32b7d6cbbd327957b792056bbea79a62806037e66cb5f23d097f6fa5bcc649"
results="results/root_cause_campaign/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37"
lineage="$results/lineage_theta1_theta${final_version}"

test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha256"
mkdir -p "$results/logs"
exec 8>"$results/round.lock"
flock -n 8

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done
nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$results/logs/${phase}_gpu_preflight.csv"

python scripts/lift_evokv_kuairand_medium_capacity.py \
  --config "$config" \
  --stop-after-version "$final_version" \
  --preflight-only \
  > "$results/logs/${phase}_preflight.json"
jq -e --argjson final "$final_version" '.status == "ready" and .final_version == $final and .world_size == 2' "$results/logs/${phase}_preflight.json" >/dev/null

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/lift_evokv_kuairand_medium_capacity.py \
  --config "$config" \
  --stop-after-version "$final_version" \
  2>&1 | tee "$results/logs/${phase}_lift.log"

if [[ ! -f "$lineage/result.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
    scripts/evaluate_evokv_kuairand_persistent_prefix_lineage.py \
    --config "$config" \
    --final-version "$final_version" \
    --output "$lineage" \
    2>&1 | tee "$results/logs/${phase}_lineage.log"
fi

jq -e --argjson final "$final_version" '
  .status == "complete"
  and .geometry.global_model_parameter_bytes == 47960055552
  and (.targets | length) == $final
' "$lineage/result.json" >/dev/null

python scripts/render_evokv_kuairand_capacity_matrix.py \
  --result "$lineage/result.json" \
  --output "$results/theta1_theta${final_version}_matrix.json" \
  --first-version 1 \
  --final-version "$final_version" \
  | tee "$results/logs/${phase}_matrix.log"
