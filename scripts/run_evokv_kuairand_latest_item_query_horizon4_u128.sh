#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_item_query_horizon4_h512_l8_u128_20260810_v11.json"
output="results/root_cause_campaign/kuairand_latest_item_query_horizon4_h512_l8_u128_20260810_v11"

test "$(sha256sum "$config" | awk '{print $1}')" = "9883397bebf5f0e54d695b8a6d5008b3af75425440ad6fba2e7d363172b5850f"

python - "$config" <<'PY'
import sys
from hstu_kvcache.streaming.kuairand_query_transition import build_workload, load_config

document = load_config(sys.argv[1])
if document["data"]["training_target_horizon"] != 4:
    raise SystemExit("training target horizon differs")
if document["model"]["query_mode"] != "latest_item_query":
    raise SystemExit("latest-item query mode differs")
if document["evaluation"]["candidate_count"] != 100:
    raise SystemExit("candidate_count must remain 100")
workload = build_workload(document)
if workload["metadata"]["base_effective_targets"] != 4 * workload["metadata"]["base_examples"]:
    raise SystemExit("base effective target count differs")
if workload["metadata"]["update_effective_targets"] != 4 * workload["metadata"]["update_examples"]:
    raise SystemExit("update effective target count differs")
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
