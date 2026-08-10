#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_a="configs/evokv_root_cause/kuairand_latest_item_query_fullusers_coreonly300_20260810_v12.json"
config_b="configs/evokv_root_cause/kuairand_latest_item_query_fullusers_coreonly500_20260810_v12.json"
output_a="results/root_cause_campaign/kuairand_latest_item_query_fullusers_coreonly300_20260810_v12"
output_b="results/root_cause_campaign/kuairand_latest_item_query_fullusers_coreonly500_20260810_v12"

test "$(sha256sum "$config_a" | awk '{print $1}')" = "4018cc0f916fdac66e9ae42d2f8343ac2fb4c0092680aa4392b313a1240f8909"
test "$(sha256sum "$config_b" | awk '{print $1}')" = "cb791de4fbfedb591fd9a74c48fbbc3a35f91d1ba3f12c2ef1ad97ccd0aefbfd"

python - "$config_a" "$config_b" <<'PY'
import sys
from hstu_kvcache.streaming.kuairand_query_transition import load_config

parents = set()
for path in sys.argv[1:]:
    document = load_config(path)
    parents.add(document["training"]["parent_theta0"]["sha256"])
    if document["data"]["user_limit"] is not None:
        raise SystemExit("core-only branch must use all eligible users")
    if document["model"]["query_mode"] != "latest_item_query":
        raise SystemExit("latest-item query mode differs")
    if document["training"]["update_embedding_lr"] > 0.00001:
        raise SystemExit("embedding learning rate is not near-frozen")
    if document["evaluation"]["candidate_count"] != 100:
        raise SystemExit("candidate_count must remain 100")
if len(parents) != 1:
    raise SystemExit("core-only branches do not share theta0")
PY

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

free_bytes="$(df --output=avail -B1 "$repo_root" | tail -n 1 | tr -d ' ')"
test "$free_bytes" -ge 25769803776

mkdir -p "$output_a" "$output_b"
exec 8>"$output_a/round.lock"
exec 9>"$output_b/round.lock"
flock -n 8
flock -n 9

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_a" >>"$output_a/run.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_b" >>"$output_b/run.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
pids=()
test "$status" -eq 0

python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_a" --result "$output_a/summary.json"
python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_b" --result "$output_b/summary.json"

trap - INT TERM EXIT
