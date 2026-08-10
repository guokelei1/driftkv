#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_foundation_rebuild_medium_rolling4_theta12_kv005_20260809_v2.json"
log_root="results/root_cause_campaign/kuairand_foundation_rebuild_medium_rolling4_theta12_kv005_20260809_v2"
mkdir -p "$log_root"

if [[ ! -f "$log_root/config_sha256.txt" ]]; then
  sha256sum "$config" > "$log_root/config_sha256.txt"
fi

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" 2>&1 | tee -a "$log_root/run.log"

python - <<'PY'
import json
from pathlib import Path

path = Path("results/root_cause_campaign/kuairand_foundation_rebuild_medium_rolling4_theta12_kv005_20260809_v2/result.json")
result = json.loads(path.read_text())
if result.get("status") != "complete" or result.get("checkpoint_count") != 12:
    raise SystemExit(f"incomplete result: {path}")
print(path)
PY
