#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_dense="configs/evokv_root_cause/kuairand_true_next_item_strict_theta0_branch_dense200_20260810_v6.json"
config_kv="configs/evokv_root_cause/kuairand_true_next_item_strict_theta0_branch_kv300_20260810_v6.json"
output_dense="results/root_cause_campaign/kuairand_true_next_item_strict_theta0_branch_dense200_20260810_v6"
output_kv="results/root_cause_campaign/kuairand_true_next_item_strict_theta0_branch_kv300_20260810_v6"

test "$(sha256sum "$config_dense" | awk '{print $1}')" = "eed7a379fe799798b124e0f6768d1974ed44dd5ec2597133f3f162f9661effae"
test "$(sha256sum "$config_kv" | awk '{print $1}')" = "722199774152b11177e9f30d70611f69c04524d58d5cdfa213e97b2d281f54ee"

python - "$config_dense" "$config_kv" <<'PY'
import sys
from hstu_kvcache.streaming.kuairand_query_transition import load_config

parents = set()
for path in sys.argv[1:]:
    document = load_config(path)
    parents.add(document["training"]["parent_theta0"]["sha256"])
    if document["model"]["causal_diagonal"] != "exclusive":
        raise SystemExit("strict causal diagonal differs")
    if document["evaluation"]["candidate_count"] != 100:
        raise SystemExit("candidate_count must remain 100")
if len(parents) != 1:
    raise SystemExit("update branches do not share one theta0")
PY

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

free_bytes="$(df --output=avail -B1 "$repo_root" | tail -n 1 | tr -d ' ')"
test "$free_bytes" -ge 64424509440

mkdir -p "$output_dense" "$output_kv"
exec 8>"$output_dense/round.lock"
exec 9>"$output_kv/round.lock"
flock -n 8
flock -n 9

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_dense" >>"$output_dense/run.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_kv" >>"$output_kv/run.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
pids=()
test "$status" -eq 0

python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_dense" --result "$output_dense/summary.json"
python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_kv" --result "$output_kv/summary.json"

trap - INT TERM EXIT
