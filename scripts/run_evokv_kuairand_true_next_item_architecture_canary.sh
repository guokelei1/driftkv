#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_l8="configs/evokv_root_cause/kuairand_true_next_item_h512_l8_t512_canary_20260810_v1.json"
config_l16="configs/evokv_root_cause/kuairand_true_next_item_h512_l16_t512_canary_20260810_v1.json"
output_l8="results/root_cause_campaign/kuairand_true_next_item_h512_l8_t512_canary_20260810_v1"
output_l16="results/root_cause_campaign/kuairand_true_next_item_h512_l16_t512_canary_20260810_v1"

test "$(sha256sum "$config_l8" | awk '{print $1}')" = "780f4e9f18e041a7b81d85acb9a9dd6b30b89b4aab447b9ed14d5b94f920266b"
test "$(sha256sum "$config_l16" | awk '{print $1}')" = "3143f8d021bc2b606854e100a2b598bbe5034c58220f3bbb659a974f8a98a185"

python - "$config_l8" "$config_l16" <<'PY'
import sys
from hstu_kvcache.streaming.kuairand_query_transition import load_config

for path in sys.argv[1:]:
    load_config(path)
PY

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$output_l8" "$output_l16"
exec 8>"$output_l8/round.lock"
exec 9>"$output_l16/round.lock"
flock -n 8
flock -n 9

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_l8" >"$output_l8/run.log" 2>&1 &
pid_l8=$!
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_l16" >"$output_l16/run.log" 2>&1 &
pid_l16=$!

status=0
wait "$pid_l8" || status=1
wait "$pid_l16" || status=1
test "$status" -eq 0

python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_l8" --result "$output_l8/summary.json"
python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_l16" --result "$output_l16/summary.json"
