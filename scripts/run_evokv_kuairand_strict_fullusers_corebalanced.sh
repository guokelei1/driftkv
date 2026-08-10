#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_true_next_item_strict_fullusers_h512_l8_corebalanced_20260810_v7.json"
output="results/root_cause_campaign/kuairand_true_next_item_strict_fullusers_h512_l8_corebalanced_20260810_v7"

test "$(sha256sum "$config" | awk '{print $1}')" = "5ff5fa4343277d08fc5bc6cdb397b47d027e0856a8cf402f6f3c30b1f799928c"

python - "$config" <<'PY'
import sys
from hstu_kvcache.streaming.kuairand_query_transition import load_config

document = load_config(sys.argv[1])
if document["data"]["user_limit"] is not None:
    raise SystemExit("full-user screen must not limit users")
if document["model"]["causal_diagonal"] != "exclusive":
    raise SystemExit("strict causal diagonal differs")
if document["evaluation"]["candidate_count"] != 100:
    raise SystemExit("candidate_count must remain 100")
PY

used="$(nvidia-smi --id=0 --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
test "$used" -le 512

free_bytes="$(df --output=avail -B1 "$repo_root" | tail -n 1 | tr -d ' ')"
test "$free_bytes" -ge 25769803776

mkdir -p "$output"
exec 8>"$output/round.lock"
flock -n 8

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config" 2>&1 | tee -a "$output/run.log"
python scripts/validate_evokv_kuairand_prediction_query.py --config "$config" --result "$output/summary.json"
