#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1

config=configs/evokv_root_cause/kuairand_projected_medium_uniform_kv002_continuous_theta1_theta8_20260808_v1.json
root=results/opportunity_discovery/evokv_kuairand_uniform_medium_continuous_20260808_v1/kv002
mkdir -p "$root"

python - <<'PY'
from hstu_kvcache.streaming.kuairand_projected_scale import load_projected_config

load_projected_config("configs/evokv_root_cause/kuairand_projected_medium_uniform_kv002_continuous_theta1_theta8_20260808_v1.json")
PY

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"
CUDA_VISIBLE_DEVICES=0 python scripts/run_evokv_kuairand_projected_scale.py --config "$config" 2>&1 | tee "$root/run.log"
