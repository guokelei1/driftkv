#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1

root=results/opportunity_discovery/evokv_kuairand_uniform_medium_selection_20260808_v0
mkdir -p "$root/kv003" "$root/kv004"

python - <<'PY'
from hstu_kvcache.streaming.kuairand_projected_scale import load_projected_config

for path in (
    "configs/evokv_root_cause/kuairand_projected_medium_uniform_kv003_selection_theta1_theta3_20260808_v0.json",
    "configs/evokv_root_cause/kuairand_projected_medium_uniform_kv004_selection_theta1_theta3_20260808_v0.json",
):
    load_projected_config(path)
PY

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"

for candidate in kv003 kv004; do
  CUDA_VISIBLE_DEVICES=0 python scripts/run_evokv_kuairand_projected_scale.py \
    --config "configs/evokv_root_cause/kuairand_projected_medium_uniform_${candidate}_selection_theta1_theta3_20260808_v0.json" \
    2>&1 | tee "$root/$candidate/run.log"
done
