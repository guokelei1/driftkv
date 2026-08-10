#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/kuairand_projected_large_e4160_two_gpu_20260808_v0.json
output=results/opportunity_discovery/evokv_kuairand_projected_scale_20260808_v0/large_e4160_two_gpu
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc-per-node=2 scripts/run_evokv_kuairand_projected_scale.py --config "$config" 2>&1 | tee "$output/run.log"
