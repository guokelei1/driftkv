#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

pooled_config="configs/evokv_root_cause/kuairand_foundation_rebuild_medium_pooled4_20260809_v0.json"
sequential_config="configs/evokv_root_cause/kuairand_foundation_rebuild_medium_sequential4_20260809_v0.json"
log_root="results/root_cause_campaign/kuairand_foundation_rebuild_medium_screen_20260809_v0"
mkdir -p "$log_root"

if [[ ! -f "$log_root/config_sha256.txt" ]]; then
  sha256sum "$pooled_config" "$sequential_config" > "$log_root/config_sha256.txt"
fi

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$pooled_config" 2>&1 | tee -a "$log_root/pooled.log" &
pooled_pid=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$sequential_config" 2>&1 | tee -a "$log_root/sequential.log" &
sequential_pid=$!

pooled_status=0
sequential_status=0
wait "$pooled_pid" || pooled_status=$?
wait "$sequential_pid" || sequential_status=$?

if [[ "$pooled_status" -ne 0 || "$sequential_status" -ne 0 ]]; then
  printf 'pooled_status=%s sequential_status=%s\n' "$pooled_status" "$sequential_status" >&2
  exit 1
fi

python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("results/root_cause_campaign/kuairand_foundation_rebuild_medium_pooled4_20260809_v0/result.json"),
    Path("results/root_cause_campaign/kuairand_foundation_rebuild_medium_sequential4_20260809_v0/result.json"),
]
for path in paths:
    result = json.loads(path.read_text())
    if result.get("status") != "complete" or result.get("checkpoint_count") != 4:
        raise SystemExit(f"incomplete result: {path}")
    print(path)
PY
