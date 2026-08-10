#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_medium_theta1_theta9_frozen_20260810_v18.json"
root="results/root_cause_campaign/kuairand_latest_query_medium_theta9_adjacent_probes_20260810_v21"

test "$(sha256sum "$config" | awk '{print $1}')" = "9dcc1d5c8df59592daf6141d1d073b3e48a237338e6299f2e95947f0894a8d11"
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$root"
exec 8>"$root/round.lock"
flock -n 8
nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"

run_probe() {
  local gpu=$1
  local candidate=$2
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python scripts/probe_evokv_kuairand_candidate.py \
    --config "$config" \
    --version 9 \
    --candidate "$candidate" \
    --output "$root/${candidate}.json" \
    > "$root/${candidate}.log" 2>&1
}

run_probe 0 balanced_half_e3 &
pid_a=$!
run_probe 1 balanced_quarter_e2 &
pid_b=$!
status_a=0
status_b=0
wait "$pid_a" || status_a=$?
wait "$pid_b" || status_b=$?
test "$status_a" -eq 0
test "$status_b" -eq 0
run_probe 0 balanced_quarter_e3

python - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("balanced_*.json")):
    document = json.loads(path.read_text())
    comparison = document["summary"]["comparisons"]["recompute_over_reuse"]
    rows.append(
        {
            "candidate": document["candidate"]["name"],
            "mrr_relative_percent": comparison["mrr"]["relative_percent"],
            "ndcg_at_5_relative_percent": comparison["ndcg_at_5"]["relative_percent"],
            "hit_rate_at_5_relative_percent": comparison["hit_rate_at_5"]["relative_percent"],
            "elapsed_seconds": document["elapsed_seconds"],
        }
    )
(root / "summary.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
print(json.dumps(rows, indent=2, sort_keys=True))
PY
