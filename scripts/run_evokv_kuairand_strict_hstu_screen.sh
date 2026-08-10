#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_standard="configs/evokv_root_cause/kuairand_true_next_item_strict_hstu_h512_l8_u128_20260810_v4.json"
config_core="configs/evokv_root_cause/kuairand_true_next_item_strict_hstu_core_h512_l8_u128_20260810_v4.json"
output_standard="results/root_cause_campaign/kuairand_true_next_item_strict_hstu_h512_l8_u128_20260810_v4"
output_core="results/root_cause_campaign/kuairand_true_next_item_strict_hstu_core_h512_l8_u128_20260810_v4"

test "$(sha256sum "$config_standard" | awk '{print $1}')" = "1dcc7ac029d4843e5e4f3e4c6f44ec49eddf7671fd3360ea84a072934f35354d"
test "$(sha256sum "$config_core" | awk '{print $1}')" = "e97f3fab96d2891d6f3f581c972e47d499c957cdb2ab04a14539aa7c2e27679a"

python - "$config_standard" "$config_core" <<'PY'
import sys
from hstu_kvcache.streaming.kuairand_query_transition import load_config

for path in sys.argv[1:]:
    document = load_config(path)
    if document["model"]["block_variant"] != "hstu_reference":
        raise SystemExit("reference HSTU architecture differs")
    if document["model"]["causal_diagonal"] != "exclusive":
        raise SystemExit("strict causal diagonal differs")
    if document["evaluation"]["candidate_count"] != 100:
        raise SystemExit("candidate_count must remain 100")
PY

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

free_bytes="$(df --output=avail -B1 "$repo_root" | tail -n 1 | tr -d ' ')"
test "$free_bytes" -ge 85899345920

mkdir -p "$output_standard" "$output_core"
exec 8>"$output_standard/round.lock"
exec 9>"$output_core/round.lock"
flock -n 8
flock -n 9

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_standard" >>"$output_standard/run.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_core" >>"$output_core/run.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
pids=()
test "$status" -eq 0

python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_standard" --result "$output_standard/summary.json"
python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_core" --result "$output_core/summary.json"

trap - INT TERM EXIT
