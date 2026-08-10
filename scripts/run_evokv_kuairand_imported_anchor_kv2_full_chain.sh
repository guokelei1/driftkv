#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1

config=configs/evokv_root_cause/kuairand_projected_large_imported_anchor_kv2_seed53117_theta1_theta8_20260808_v2.json
root=results/opportunity_discovery/evokv_kuairand_imported_anchor_kv2_seed53117_v2
anchor=results/opportunity_discovery/evokv_kuairand_projected_theta1_theta8_seed53117_v0/edges/theta_1/accepted.json
imported=$root/edges/theta_1/accepted.json
mkdir -p "$root/edges/theta_1"

if [[ ! -f "$imported" ]]; then
  cp "$anchor" "$imported"
fi

cmp -s "$anchor" "$imported"

python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only | tee "$root/preflight.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" 2>&1 | tee "$root/run.log"
