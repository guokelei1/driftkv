#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:-verify}"
registry="configs/evokv_root_cause/kuairand_large_baseline_registry_20260811_v0.json"
base_config="configs/evokv_root_cause/kuairand_latest_item_query_fullusers_h512_l8_20260810_v10.json"
medium_config="configs/evokv_root_cause/kuairand_latest_query_medium_theta9_half_e3_lineage_20260810_v22.json"
large_config="configs/evokv_root_cause/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37.json"
base_results="results/root_cause_campaign/kuairand_latest_item_query_fullusers_h512_l8_20260810_v10"
theta0="$base_results/checkpoints/seed_53117/theta0.pt"
medium_checkpoints="checkpoints/evokv_kuairand_latest_query_medium_theta9_half_e3_lineage_v22"
medium_results="results/root_cause_campaign/kuairand_latest_query_medium_theta9_half_e3_lineage_20260810_v22"
large_checkpoints="checkpoints/evokv_kuairand_latest_query_large_capacity_lift_theta1_theta8_v37"
large_results="results/root_cause_campaign/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37"
lineage="$large_results/lineage_theta1_theta8"
logs="results/baseline_rebuild_logs/kuairand_large_theta1_theta8_20260811_v0"

case "$mode" in
  verify)
    exec python scripts/verify_evokv_kuairand_large_baseline.py --registry "$registry" --scope current
    ;;
  verify-full)
    exec python scripts/verify_evokv_kuairand_large_baseline.py --registry "$registry" --scope full
    ;;
  resume|fresh)
    ;;
  *)
    echo "usage: $0 {verify|verify-full|resume|fresh}" >&2
    exit 2
    ;;
esac

mkdir -p "$logs" "results/root_cause_campaign/.locks"
exec 9>"results/root_cause_campaign/.locks/kuairand_large_baseline_rebuild.lock"
flock -n 9

python scripts/verify_evokv_kuairand_large_baseline.py --registry "$registry" --scope static \
  | tee "$logs/static_preflight.json"

if [[ "$mode" == "fresh" ]]; then
  test "${EVO_KV_CONFIRM_FRESH:-}" = "delete-kuairand-large-baseline-v0"
  targets=(
    "$base_results"
    "$medium_checkpoints"
    "$medium_results"
    "$large_checkpoints"
    "$large_results"
  )
  python - "$repo_root" "${targets[@]}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
allowed = {
    (root / value).resolve()
    for value in (
        "results/root_cause_campaign/kuairand_latest_item_query_fullusers_h512_l8_20260810_v10",
        "checkpoints/evokv_kuairand_latest_query_medium_theta9_half_e3_lineage_v22",
        "results/root_cause_campaign/kuairand_latest_query_medium_theta9_half_e3_lineage_20260810_v22",
        "checkpoints/evokv_kuairand_latest_query_large_capacity_lift_theta1_theta8_v37",
        "results/root_cause_campaign/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37",
    )
}
observed = {(root / value).resolve() for value in sys.argv[2:]}
if observed != allowed or any(path == root or root not in path.parents for path in observed):
    raise SystemExit("fresh cleanup target differs")
PY
  for target in "${targets[@]}"; do
    rm -rf -- "$target"
  done
  printf '%s\n' "fresh reset completed at $(date --iso-8601=seconds)" | tee "$logs/fresh_reset.log"
fi

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done
nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader \
  > "$logs/gpu_preflight.csv"

expected_theta0="$(jq -r '.bootstrap.theta0.sha256' "$registry")"
if [[ -f "$theta0" ]] && [[ "$(sha256sum "$theta0" | awk '{print $1}')" == "$expected_theta0" ]]; then
  printf '%s\n' "theta0 already valid" | tee "$logs/theta0_stage.log"
else
  if [[ -e "$base_results" ]]; then
    echo "theta0 boundary is incomplete; use fresh mode to rebuild without overwriting it" >&2
    exit 1
  fi
  scripts/run_evokv_kuairand_latest_item_query_fullusers.sh \
    2>&1 | tee "$logs/theta0_stage.log"
  test "$(sha256sum "$theta0" | awk '{print $1}')" = "$expected_theta0"
fi

medium_ready=1
for version in $(seq 1 9); do
  test -f "$medium_checkpoints/theta_$version/manifest.json" || medium_ready=0
  test -f "$medium_results/edges/theta_$version/accepted.json" || medium_ready=0
done
test -f "$medium_results/result.json" || medium_ready=0
if [[ "$medium_ready" == 0 ]]; then
  python scripts/train_evokv_kuairand_theta1_theta8.py \
    --config "$medium_config" --preflight-only | tee "$logs/medium_preflight.json"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python \
    scripts/train_evokv_kuairand_theta1_theta8.py --config "$medium_config" \
    2>&1 | tee "$logs/medium_train.log"
else
  printf '%s\n' "medium theta1-theta9 chain already complete" | tee "$logs/medium_train.log"
fi

large_ready=1
large_rebuilt=0
for version in $(seq 1 8); do
  test -f "$large_checkpoints/theta_$version/manifest.json" || large_ready=0
  test -f "$large_results/edges/theta_$version/accepted.json" || large_ready=0
done
test -f "$large_results/capacity_lift_theta1_theta8.json" || large_ready=0
if [[ "$large_ready" == 0 ]]; then
  python scripts/lift_evokv_kuairand_medium_capacity.py \
    --config "$large_config" --stop-after-version 8 --preflight-only \
    | tee "$logs/large_lift_preflight.json"
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
    scripts/lift_evokv_kuairand_medium_capacity.py \
    --config "$large_config" --stop-after-version 8 \
    2>&1 | tee "$logs/large_lift.log"
  large_rebuilt=1
else
  printf '%s\n' "large theta1-theta8 checkpoints already complete" | tee "$logs/large_lift.log"
fi

if [[ "$large_rebuilt" == 1 ]] && [[ -e "$lineage" ]]; then
  test "$lineage" = "results/root_cause_campaign/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37/lineage_theta1_theta8"
  rm -rf -- "$lineage"
fi

if [[ ! -f "$lineage/result.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
    scripts/evaluate_evokv_kuairand_persistent_prefix_lineage.py \
    --config "$large_config" --final-version 8 --output "$lineage" \
    2>&1 | tee "$logs/large_lineage.log"
else
  printf '%s\n' "large theta1-theta8 lineage already complete" | tee "$logs/large_lineage.log"
fi

python scripts/render_evokv_kuairand_capacity_matrix.py \
  --result "$lineage/result.json" \
  --output "$large_results/theta1_theta8_matrix.json" \
  --first-version 1 --final-version 8 \
  | tee "$logs/matrix_render.log"

python scripts/verify_evokv_kuairand_large_baseline.py \
  --registry "$registry" --scope current | tee "$logs/final_verification.json"
