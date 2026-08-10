#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e4160_theta1_theta9_frozen_20260810_v20.json"
probes="configs/evokv_root_cause/kuairand_large_theta4_edge_search_20260810_v28.json"
root="results/root_cause_campaign/kuairand_large_theta4_edge_search_20260810_v28"

test "$(sha256sum "$config" | awk '{print $1}')" = "e6be04fd67803b17ef3d778ee5fd65182a854fdfb0f97ad0761e687c71231d64"
test "$(sha256sum "$probes" | awk '{print $1}')" = "1931241585290f6aacb959e12c73933a881e400f0fe2644cf25664e77c29f6d9"
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

mkdir -p "$root"
exec 8>"$root/round.lock"
flock -n 8
nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$root/gpu_preflight.csv"

for candidate in \
  large_theta4_native_half_e2 \
  large_theta4_native_half_e3 \
  large_theta4_native_half_e4 \
  large_theta4_kv2_e2 \
  large_theta4_kv2_e3 \
  large_theta4_kv4_e2; do
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
    scripts/probe_evokv_kuairand_candidate.py \
    --config "$config" \
    --version 4 \
    --candidate "$candidate" \
    --candidate-config "$probes" \
    --output "$root/${candidate}.json" \
    2>&1 | tee "$root/${candidate}.log"
done

python - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("large_theta4_*.json")):
    document = json.loads(path.read_text())
    comparison = document["summary"]["comparisons"]["recompute_over_reuse"]
    rows.append(
        {
            "candidate": document["candidate"]["name"],
            "mrr_relative_percent": comparison["mrr"]["relative_percent"],
            "ndcg_at_5_relative_percent": comparison["ndcg_at_5"]["relative_percent"],
            "hit_rate_at_5_relative_percent": comparison["hit_rate_at_5"]["relative_percent"],
            "fresh_ndcg_at_5": document["summary"]["endpoints"]["recompute"]["ndcg_at_5"],
            "reuse_ndcg_at_5": document["summary"]["endpoints"]["reuse"]["ndcg_at_5"],
            "elapsed_seconds": document["elapsed_seconds"],
        }
    )
(root / "summary.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
print(json.dumps(rows, indent=2, sort_keys=True))
PY
