#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_embed005="configs/evokv_root_cause/kuairand_true_next_item_strict_theta0_branch_embed005_20260810_v5.json"
config_embed001="configs/evokv_root_cause/kuairand_true_next_item_strict_theta0_branch_embed001_20260810_v5.json"
output_embed005="results/root_cause_campaign/kuairand_true_next_item_strict_theta0_branch_embed005_20260810_v5"
output_embed001="results/root_cause_campaign/kuairand_true_next_item_strict_theta0_branch_embed001_20260810_v5"

test "$(sha256sum "$config_embed005" | awk '{print $1}')" = "b97145b2f1507b91b77342e3484204acb62fb44679cda38cba77cf2c67631751"
test "$(sha256sum "$config_embed001" | awk '{print $1}')" = "97f8c96f31372d160782180e9239869d7f3487e9afd17cb567b0d2b24d69452a"

python - "$config_embed005" "$config_embed001" <<'PY'
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

mkdir -p "$output_embed005" "$output_embed001"
exec 8>"$output_embed005/round.lock"
exec 9>"$output_embed001/round.lock"
flock -n 8
flock -n 9

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_embed005" >>"$output_embed005/run.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_embed001" >>"$output_embed001/run.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
pids=()
test "$status" -eq 0

python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_embed005" --result "$output_embed005/summary.json"
python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_embed001" --result "$output_embed001/summary.json"

trap - INT TERM EXIT
