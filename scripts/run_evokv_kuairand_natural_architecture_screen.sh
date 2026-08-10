#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_l8="configs/evokv_root_cause/kuairand_natural_architecture_h512_l8_t512_20260810_v0.json"
config_l16="configs/evokv_root_cause/kuairand_natural_architecture_h512_l16_t512_20260810_v0.json"
output_l8="results/root_cause_campaign/kuairand_natural_architecture_h512_l8_t512_20260810_v0"
output_l16="results/root_cause_campaign/kuairand_natural_architecture_h512_l16_t512_20260810_v0"

test "$(sha256sum "$config_l8" | awk '{print $1}')" = "44fd1683010abc3b1d8c94234c632e5b24f80e9f8230459e77766472b38a7dfc"
test "$(sha256sum "$config_l16" | awk '{print $1}')" = "2a747e2118ddc9c003ceadd90ce076c879aa15fd03fdc12a0e41380a1777e98d"

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
