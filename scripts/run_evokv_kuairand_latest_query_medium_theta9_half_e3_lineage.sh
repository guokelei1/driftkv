#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_medium_theta9_half_e3_lineage_20260810_v22.json"
source_config="configs/evokv_root_cause/kuairand_latest_query_medium_theta1_theta9_frozen_20260810_v18.json"
source_checkpoints="checkpoints/evokv_kuairand_latest_query_medium_theta1_theta9_frozen_v18"
source_results="results/root_cause_campaign/kuairand_latest_query_medium_theta1_theta9_frozen_20260810_v18"
checkpoints="checkpoints/evokv_kuairand_latest_query_medium_theta9_half_e3_lineage_v22"
results="results/root_cause_campaign/kuairand_latest_query_medium_theta9_half_e3_lineage_20260810_v22"

test "$(sha256sum "$source_config" | awk '{print $1}')" = "9dcc1d5c8df59592daf6141d1d073b3e48a237338e6299f2e95947f0894a8d11"
used="$(nvidia-smi --id=0 --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
test "$used" -le 512

mkdir -p "$checkpoints" "$results/lineage"
for version in $(seq 1 8); do
  if test ! -e "$checkpoints/theta_$version"; then
    test -f "$source_checkpoints/theta_$version/manifest.json"
    cp -al "$source_checkpoints/theta_$version" "$checkpoints/theta_$version"
  fi
  if test ! -f "$results/edges/theta_$version/accepted.json"; then
    test -f "$source_results/edges/theta_$version/accepted.json"
    mkdir -p "$results/edges/theta_$version"
    cp -a "$source_results/edges/theta_$version/accepted.json" "$results/edges/theta_$version/accepted.json"
  fi
  if test ! -f "$results/lineage/theta_$version.json"; then
    test -f "$source_results/lineage/theta_$version.json"
    cp -a "$source_results/lineage/theta_$version.json" "$results/lineage/theta_$version.json"
  fi
done

python - "$source_config" "$source_results/result.json" "$results/prefix_import.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

config = Path(sys.argv[1])
result = Path(sys.argv[2])
output = Path(sys.argv[3])
document = {
    "source_config": {"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
    "source_result": {"path": str(result), "sha256": hashlib.sha256(result.read_bytes()).hexdigest()},
    "imported_checkpoint_versions": list(range(1, 9)),
    "imported_lineage_targets": list(range(1, 9)),
}
output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
PY

exec 8>"$results/round.lock"
flock -n 8
nvidia-smi --id=0 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$results/gpu_preflight.csv"
python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" --preflight-only > "$results/preflight.log"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_evokv_kuairand_theta1_theta8.py --config "$config" 2>&1 | tee "$results/run.log"
python scripts/render_evokv_kuairand_selected_8x8.py \
  --result "$results/result.json" \
  --output "$results/selected_theta2_theta9_ndcg5_8x8.json" \
  --first-target 2 \
  --versions 8 \
  --metric ndcg_at_5
