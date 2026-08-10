#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1

config=configs/evokv_root_cause/kuairand_projected_large_anchor_stationary_kv002_seed53117_theta1_theta3_20260808_v0.json
root=results/opportunity_discovery/evokv_kuairand_large_anchor_stationary_probe_20260808_v0/seed53117
mkdir -p "$root"

python - <<'PY'
from hstu_kvcache.streaming.kuairand_projected_scale import load_projected_config

load_projected_config("configs/evokv_root_cause/kuairand_projected_large_anchor_stationary_kv002_seed53117_theta1_theta3_20260808_v0.json")
PY

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/run_evokv_kuairand_projected_scale.py --config "$config" 2>&1 | tee "$root/run.log"
