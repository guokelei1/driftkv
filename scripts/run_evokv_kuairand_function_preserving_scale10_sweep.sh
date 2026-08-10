#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/evokv_root_cause/kuairand_function_preserving_scale10_k005_v015_20260809_v0.json
  configs/evokv_root_cause/kuairand_function_preserving_scale10_k010_v020_20260809_v0.json
)

mapfile -t gpu_free_mib < <(
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
)
test "${gpu_free_mib[0]}" -ge 43000
test "${gpu_free_mib[1]}" -ge 43000

for config in "${configs[@]}"; do
  python -c "from hstu_kvcache.streaming.kuairand_projected_gauge_triangle import load_projected_gauge_triangle_config; load_projected_gauge_triangle_config('$config')"
  output=$(python -c "import json; print(json.load(open('$config'))['outputs']['result'])")
  log="${output%/result.json}/run.log"
  mkdir -p "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    scripts/evaluate_evokv_kuairand_projected_gauge_triangle.py \
    --config "$config" > "$log" 2>&1
  jq '{transform,decision,fresh_function_invariance}' "$output"
done
