#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_rab="configs/evokv_root_cause/kuairand_true_next_item_reference_h512_l8_rab_u128_20260810_v3.json"
config_norab="configs/evokv_root_cause/kuairand_true_next_item_reference_h512_l8_norab_u128_20260810_v3.json"
output_rab="results/root_cause_campaign/kuairand_true_next_item_reference_h512_l8_rab_u128_20260810_v3"
output_norab="results/root_cause_campaign/kuairand_true_next_item_reference_h512_l8_norab_u128_20260810_v3"

test "$(sha256sum "$config_rab" | awk '{print $1}')" = "6bbe7a0bd4a79755f0d5094b3af2f62ead7b7e65d5eb9cb2fd1aa1bee1491ac7"
test "$(sha256sum "$config_norab" | awk '{print $1}')" = "f1eb1cc7c316b44fca02b2f92789df0c739487a3ca83a28e6543f820376a4833"

python - "$config_rab" "$config_norab" <<'PY'
import sys
from hstu_kvcache.streaming.kuairand_query_transition import load_config

for path in sys.argv[1:]:
    document = load_config(path)
    if document["model"]["block_variant"] != "hstu_reference":
        raise SystemExit("reference HSTU architecture differs")
    if document["evaluation"]["candidate_count"] != 100:
        raise SystemExit("candidate_count must remain 100")
PY

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

free_bytes="$(df --output=avail -B1 "$repo_root" | tail -n 1 | tr -d ' ')"
test "$free_bytes" -ge 85899345920

mkdir -p "$output_rab" "$output_norab"
exec 8>"$output_rab/round.lock"
exec 9>"$output_norab/round.lock"
flock -n 8
flock -n 9

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_rab" >>"$output_rab/run.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_norab" >>"$output_norab/run.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
pids=()
test "$status" -eq 0

python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_rab" --result "$output_rab/summary.json"
python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_norab" --result "$output_norab/summary.json"

trap - INT TERM EXIT
