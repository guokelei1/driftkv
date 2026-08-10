#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_item_query_fullusers_h512_l8_20260810_v10.json"
output="results/root_cause_campaign/kuairand_latest_item_query_fullusers_h512_l8_20260810_v10"

test "$(sha256sum "$config" | awk '{print $1}')" = "a7757310eab0d7d332253292e633c7edb1bad8c8fc2707da11280bc1cb99f1ef"

python - "$config" <<'PY'
import sys
from hstu_kvcache.streaming.kuairand_query_transition import load_config

document = load_config(sys.argv[1])
if document["data"]["user_limit"] is not None:
    raise SystemExit("full-user run must not limit users")
if document["model"]["query_mode"] != "latest_item_query":
    raise SystemExit("latest-item query mode differs")
if document["model"]["num_layers"] != 8:
    raise SystemExit("HSTU layer count differs")
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
