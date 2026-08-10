#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1

root=results/opportunity_discovery/evokv_kuairand_uniform_medium_audit_20260808_v0
mkdir -p "$root/standard" "$root/kv_focused"

python - <<'PY'
from hstu_kvcache.streaming.kuairand_projected_scale import load_projected_config

paths = (
    "configs/evokv_root_cause/kuairand_projected_medium_uniform_standard_theta1_theta8_20260808_v0.json",
    "configs/evokv_root_cause/kuairand_projected_medium_uniform_kv_focused_theta1_theta8_20260808_v0.json",
)
for path in paths:
    load_projected_config(path)
PY

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"

CUDA_VISIBLE_DEVICES=0 python scripts/run_evokv_kuairand_projected_scale.py \
  --config configs/evokv_root_cause/kuairand_projected_medium_uniform_standard_theta1_theta8_20260808_v0.json \
  2>&1 | tee "$root/standard/run.log"

CUDA_VISIBLE_DEVICES=0 python scripts/run_evokv_kuairand_projected_scale.py \
  --config configs/evokv_root_cause/kuairand_projected_medium_uniform_kv_focused_theta1_theta8_20260808_v0.json \
  2>&1 | tee "$root/kv_focused/run.log"
