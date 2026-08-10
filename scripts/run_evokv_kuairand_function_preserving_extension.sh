#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(11|12|13)$ ]]; then
  echo "usage: $0 {11|12|13}" >&2
  exit 2
fi

target_version="$1"
case "$target_version" in
  11)
    config="configs/evokv_root_cause/kuairand_function_preserving_scale11_k005_v015_20260809_v0.json"
    expected_config_sha256="63784f876b737315ed417bea3545392b58f717a0322b04606f9a8c6b7d118739"
    ;;
  12)
    config="configs/evokv_root_cause/kuairand_function_preserving_scale12_k005_v015_20260809_v0.json"
    expected_config_sha256="169abcfcdc30b1ef4ffe3f4efd42baef8d530a692f4c79e0aaaf4476ba9b9ed2"
    ;;
  13)
    config="configs/evokv_root_cause/kuairand_function_preserving_scale13_k005_v015_20260809_v0.json"
    expected_config_sha256="a9fe9f73191da983c5872ecd58e664e9def0e496df632cf65d871dc48b68ae79"
    ;;
esac

output_root="results/root_cause_campaign/kuairand_function_preserving_extension_k005_v015_20260809_v0"
result_path="${output_root}/theta${target_version}_result.json"
bounded_path="${output_root}/theta${target_version}_bounded_from_theta3.json"
table_path="${output_root}/theta${target_version}_bounded_from_theta3.md"
log_path="${output_root}/theta${target_version}_evaluation.log"
mkdir -p "$output_root"
exec 9>"${output_root}/evaluation.lock"
if ! flock -n 9; then
  echo "another KuaiRand extension evaluation holds the lock" >&2
  exit 1
fi

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  if (( used > 512 )); then
    echo "GPU${gpu} is not available: ${used} MiB used by compute processes" >&2
    exit 1
  fi
done

python - "$config" "$expected_config_sha256" <<'PY'
import sys

from hstu_kvcache.streaming.kuairand_projected_gauge_triangle import (
    load_projected_gauge_triangle_config,
)
from hstu_kvcache.streaming.kuairand_query_transition import file_sha256

path, expected = sys.argv[1:]
load_projected_gauge_triangle_config(path)
if file_sha256(path) != expected:
    raise SystemExit("extension evaluation config hash differs")
PY

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
torchrun --standalone --nproc-per-node=2 \
  scripts/evaluate_evokv_kuairand_projected_gauge_triangle.py \
  --config "$config" \
  2>&1 | tee "$log_path"

python scripts/render_evokv_kuairand_bounded_loss_matrix.py \
  --result "$result_path" \
  --output-json "$bounded_path" \
  --output-table "$table_path" \
  --split holdout \
  --metric ndcg_at_5 \
  --first-version 3

python - "$result_path" "$bounded_path" "$target_version" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
bounded_path = Path(sys.argv[2])
target = int(sys.argv[3])
result = json.loads(result_path.read_text())
bounded = json.loads(bounded_path.read_text())
expected_cells = (target - 3) * (target - 2) // 2
if (
    result.get("status") != "complete_development_control"
    or not result.get("fresh_function_invariance", {}).get("passed")
    or bounded.get("final_version") != target
    or bounded.get("report_summary", {}).get("ordinary_cells") != expected_cells
):
    raise SystemExit("extension evaluation durable result differs")
print(json.dumps(bounded["report_summary"], indent=2, sort_keys=True))
PY
