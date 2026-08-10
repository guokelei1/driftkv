#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20.json"
results="results/root_cause_campaign/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20"
checkpoints="checkpoints/evokv_kuairand_latest_query_large_e4160_theta1_theta9_frozen_v20"
matrix="$results/selected_theta2_theta9_ndcg5_8x8.json"

test "$(sha256sum "$config" | awk '{print $1}')" = "e6be04fd67803b17ef3d778ee5fd65182a854fdfb0f97ad0761e687c71231d64"
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$results" "$checkpoints"
exec 8>"$results/round.lock"
flock -n 8
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free --format=csv,noheader > "$results/gpu_preflight.csv"
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only > "$results/preflight.log"

completed="$(jq -r '.completed_versions' "$results/preflight.log")"
if test "$completed" -eq 0; then
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
    scripts/train_evokv_kuairand_theta1_theta8.py \
    --config "$config" \
    --stop-after-version 1 \
    2>&1 | tee "$results/theta1_canary.log"
fi

jq -e '
  .status == "accepted"
  and .candidate.summary.sanity.passed == true
  and .candidate.summary.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent > 1.0
' "$results/edges/theta_1/accepted.json" > /dev/null

if test ! -f "$results/result.json"; then
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
    scripts/train_evokv_kuairand_theta1_theta8.py \
    --config "$config" \
    2>&1 | tee -a "$results/run.log"
fi

python scripts/render_evokv_kuairand_reuse_loss_table.py --result "$results/result.json" > "$results/render.log"
python scripts/render_evokv_kuairand_selected_8x8.py \
  --result "$results/result.json" \
  --output "$matrix" \
  --first-target 2 \
  --versions 8 \
  --metric ndcg_at_5 \
  > "$results/selected_matrix.log"

jq -e '
  .status == "complete"
  and .summary.adjacent_cells == 8
  and .summary.positive_adjacent_cells == 8
  and .summary.cells == 36
' "$matrix" > /dev/null
