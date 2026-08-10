#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_qkv_chain_theta5_theta12_20260810_v0.json"
expected_config_sha256="ee2b74c87c5c127ae2e4ce85a4df1ad50c86e78466b74b449ab5acb676b83ae7"
source_output_root="results/root_cause_campaign/kuairand_foundation_rebuild_large_e4160_theta12_20260809_v7"
checkpoint_root="checkpoints/evokv_kuairand_coordinate_aligned_chain_m0_m7_20260810_v2"
output_root="results/root_cause_campaign/kuairand_coordinate_aligned_chain_m0_m7_20260810_v2"
log_root="$output_root/logs"

test "$(sha256sum "$config" | awk '{print $1}')" = "$expected_config_sha256"
python -c "from hstu_kvcache.streaming.kuairand_qkv_chain_triangle import load_qkv_chain_config; load_qkv_chain_config('$config')"
mkdir -p "$checkpoint_root" "$output_root/edges/theta_5" "$log_root"

exec 9>"$output_root/round.lock"
flock -n 9

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

target_manifest="$checkpoint_root/theta_5/manifest.json"
test "$(sha256sum "$target_manifest" | awk '{print $1}')" = "9ab6e7ad60a76721c581e9db7f15cef2a44e78a80b3bba0463a1ff1f41d58646"

source_accepted="$source_output_root/edges/theta_5/accepted.json"
target_accepted="$output_root/edges/theta_5/accepted.json"
if ! test -f "$target_accepted"; then
  cp --archive "$source_accepted" "$target_accepted"
fi
cmp -s "$source_accepted" "$target_accepted"

python - "$config" "$output_root/preflight.json" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from hstu_kvcache.streaming.kuairand_qkv_chain_triangle import load_qkv_chain_config
from hstu_kvcache.streaming.kuairand_query_transition import file_sha256

config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
document = load_qkv_chain_config(config_path)
free = shutil.disk_usage(Path(document["outputs"]["checkpoint_root"]).parent).free
required = 7 * int(document["checkpoint"]["expected_sparse_checkpoint_bytes_per_version"])
required += int(document["checkpoint"]["write_reserve_bytes"])
record = {
    "status": "passed" if free >= required else "failed",
    "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
    "gpu_indices": [0, 1],
    "free_bytes": free,
    "required_bytes": required,
    "checkpoint_versions": list(range(5, 13)),
}
output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
if record["status"] != "passed":
    raise SystemExit("QKV chain disk preflight failed")
PY

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_evokv_kuairand_lineage_retrain.py \
  --config "$config" \
  --stop-after-version 12 2>&1 | tee "$log_root/train_theta6_theta12.log"

for version in $(seq 5 12); do
  test -f "$checkpoint_root/theta_${version}/manifest.json"
  test -f "$output_root/edges/theta_${version}/accepted.json"
done
jq -e '.status == "complete_selected_lineage_versions" and (.selected | length == 7)' "$output_root/lineage_retrain.json" >/dev/null

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/evaluate_evokv_kuairand_qkv_chain_triangle.py \
  --config "$config" 2>&1 | tee "$log_root/evaluate_triangle.log"

jq -e '.status == "complete_development_qkv_chain_triangle" and (.checkpoints | length == 8) and (.cells | length == 28)' "$output_root/qkv_chain_triangle.json" >/dev/null
