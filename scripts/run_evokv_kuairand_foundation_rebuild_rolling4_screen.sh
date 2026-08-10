#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

kv003_config="configs/evokv_root_cause/kuairand_foundation_rebuild_medium_rolling4_kv003_20260809_v1.json"
kv005_config="configs/evokv_root_cause/kuairand_foundation_rebuild_medium_rolling4_kv005_20260809_v1.json"
log_root="results/root_cause_campaign/kuairand_foundation_rebuild_rolling4_screen_20260809_v1"
mkdir -p "$log_root"

if [[ ! -f "$log_root/config_sha256.txt" ]]; then
  sha256sum "$kv003_config" "$kv005_config" > "$log_root/config_sha256.txt"
fi

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$kv003_config" 2>&1 | tee -a "$log_root/kv003.log" &
kv003_pid=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$kv005_config" 2>&1 | tee -a "$log_root/kv005.log" &
kv005_pid=$!

kv003_status=0
kv005_status=0
wait "$kv003_pid" || kv003_status=$?
wait "$kv005_pid" || kv005_status=$?

if [[ "$kv003_status" -ne 0 || "$kv005_status" -ne 0 ]]; then
  printf 'kv003_status=%s kv005_status=%s\n' "$kv003_status" "$kv005_status" >&2
  exit 1
fi

python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("results/root_cause_campaign/kuairand_foundation_rebuild_medium_rolling4_kv003_20260809_v1/result.json"),
    Path("results/root_cause_campaign/kuairand_foundation_rebuild_medium_rolling4_kv005_20260809_v1/result.json"),
]
for path in paths:
    result = json.loads(path.read_text())
    if result.get("status") != "complete" or result.get("checkpoint_count") != 4:
        raise SystemExit(f"incomplete result: {path}")
    print(path)
PY
