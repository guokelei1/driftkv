#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/evokv_root_cause/kuairand_kv_only_chain_h256_l6_theta1_theta8_20260808_v0.json}"
root="results/root_cause_campaign/kuairand_kv_only_chain_h256_l6_20260808_v0"
mkdir -p "$root"

python -c 'from hstu_kvcache.streaming.kuairand_kv_only_chain import load_kv_only_chain_config; import sys; load_kv_only_chain_config(sys.argv[1])' "$config"
CUDA_VISIBLE_DEVICES=0 python scripts/train_evokv_kuairand_kv_only_chain.py --config "$config" 2>&1 | tee "$root/training.log"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/evaluate_evokv_kuairand_kv_only_chain.py --config "$config" 2>&1 | tee "$root/evaluation.log"

