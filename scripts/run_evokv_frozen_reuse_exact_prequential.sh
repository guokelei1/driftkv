#!/usr/bin/env bash
set -euo pipefail

round_label="${1:-frozen_reuse_exact_round1}"
visible_devices="${EVOKV_CUDA_VISIBLE_DEVICES:-0,1}"
resume="${EVOKV_RESUME:-0}"

if ! [[ "$round_label" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "invalid round label" >&2
  exit 2
fi
if [[ "$resume" != "0" && "$resume" != "1" ]]; then
  echo "EVOKV_RESUME must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$visible_devices" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "EVOKV_CUDA_VISIBLE_DEVICES must name two GPUs" >&2
  exit 2
fi

IFS=',' read -r -a devices <<< "$visible_devices"
if [[ "${devices[0]}" == "${devices[1]}" ]]; then
  echo "two distinct GPUs are required" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
result_root="results/baseline_rounds/frozen_reuse_exact/${round_label}"
log_root="logs/baseline_rounds/frozen_reuse_exact/${round_label}"
cell_root="${result_root}/cells"
config="configs/evokv_baselines/x_qk_xp_multiversion_two_gpu_baseline_v1.json"
base_root="checkpoints/evokv_xp_qk_e4096_h1536/seed0"
target_root="checkpoints/evokv_xp_qk_e4096_h1536/rounds/baseline_round3"
lock_path="results/baseline_rounds/frozen_reuse_exact/.${round_label}.lock"

mkdir -p "$(dirname "$lock_path")"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "round is already running" >&2
  exit 3
fi
if [[ -e "$result_root" && "$resume" == "0" ]]; then
  echo "result root exists; choose a new label or set EVOKV_RESUME=1" >&2
  exit 3
fi

inputs=(
  "$config"
  "configs/evokv_foundation/qk_xp_fixed_edge_inputs_summary.json"
  "data/processed/evokv_foundation/qk_xp_fixed_edge_inputs.npz"
  "${base_root}/theta_0/manifest.json"
  "${target_root}/theta_1/manifest.json"
  "${target_root}/theta_2/manifest.json"
  "${target_root}/theta_3/manifest.json"
  "scripts/evaluate_evokv_xp_d1_quality.py"
  "scripts/run_evokv_frozen_reuse_exact_prequential.sh"
  "src/hstu_kvcache/migration/xp_d1_quality.py"
  "src/hstu_kvcache/streaming/xp_multiversion.py"
  "src/hstu_kvcache/streaming/xp_version_training.py"
)
for path in "${inputs[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing input: $path" >&2
    exit 4
  fi
done

busy_pids="$({
  nvidia-smi -i "$visible_devices" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
} | sed '/^[[:space:]]*$/d' | sort -u)"
if [[ -n "$busy_pids" ]]; then
  echo "selected GPUs are busy: $busy_pids" >&2
  exit 4
fi
if (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) < 64 * 1024 * 1024 )); then
  echo "at least 64 GiB available DRAM is required" >&2
  exit 4
fi

mkdir -p "$result_root" "$log_root" "$cell_root"
hashes="${result_root}/input_hashes.tsv"
candidate="$(mktemp)"
trap 'rm -f "${candidate:-}"' EXIT
sha256sum "${inputs[@]}" > "$candidate"
if [[ -f "$hashes" ]]; then
  if ! cmp -s "$candidate" "$hashes"; then
    echo "round inputs changed; use a new label" >&2
    exit 5
  fi
else
  mv "$candidate" "$hashes"
  candidate=""
fi

python - "$result_root/preflight.json" "$round_label" "$visible_devices" "$config" <<'PY'
import json
import os
import pathlib
import platform
import subprocess
import sys

import torch

path = pathlib.Path(sys.argv[1])
value = {
    "schema": "evokv_frozen_reuse_exact_prequential_preflight_v0",
    "status": "pass",
    "round_label": sys.argv[2],
    "visible_devices": sys.argv[3],
    "config": sys.argv[4],
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "methods": ["all_frozen", "all_reuse", "all_exact"],
    "negative_count": 999,
    "cache_storage_dtype": "torch.float16",
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if path.exists():
    if path.read_text() != encoded:
        raise RuntimeError("preflight binding changed")
else:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)
PY

validate_cell() {
  python - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
edge = value.get("edge", {})
frozen = value.get("frozen_quality_by_negative_count")
target = value.get("quality_by_negative_count")
if (
    value.get("status") != "complete"
    or value.get("world_size") != 2
    or value.get("evaluation_kind") != "prequential"
    or edge.get("source_version") != int(sys.argv[2])
    or edge.get("target_version") != int(sys.argv[3])
    or edge.get("history_end") != int(sys.argv[4])
    or edge.get("update_end") != int(sys.argv[5])
    or not isinstance(frozen, dict)
    or set(frozen) != {"999"}
    or set(target) != {"999"}
    or set(frozen["999"].get("methods", {})) != {"all_reuse", "all_exact"}
    or set(target["999"].get("methods", {})) != {"all_reuse", "all_exact"}
    or value.get("args", {}).get("include_frozen_control") is not True
):
    raise RuntimeError(f"frozen control cell differs: {path}")
PY
}

for source_version in 0 1 2; do
  target_version=$((source_version + 1))
  history_end=$((72 + source_version * 8))
  update_end=$((history_end + 8))
  training_history_end=$((64 + source_version * 8))
  training_update_end=$((72 + source_version * 8))
  cell_id="theta${source_version}_to_theta${target_version}"
  output="${cell_root}/${cell_id}.json"
  log="${log_root}/${cell_id}.log"
  if [[ "$source_version" == "0" ]]; then
    source_root="$base_root"
  else
    source_root="$target_root"
  fi
  if [[ -f "$output" ]]; then
    if [[ "$resume" != "1" ]]; then
      echo "cell already exists: $output" >&2
      exit 6
    fi
    validate_cell "$output" "$source_version" "$target_version" "$history_end" "$update_end"
    echo "resume: validated $cell_id"
    continue
  fi
  echo "starting cell: $cell_id"
  CUDA_VISIBLE_DEVICES="$visible_devices" OMP_NUM_THREADS=1 torchrun \
    --standalone --nproc-per-node=2 scripts/evaluate_evokv_xp_d1_quality.py \
    --config "$config" \
    --source-checkpoint-root "$source_root" \
    --target-checkpoint-root "$target_root" \
    --source-version "$source_version" \
    --target-version "$target_version" \
    --history-end "$history_end" \
    --update-end "$update_end" \
    --training-history-end "$training_history_end" \
    --training-update-end "$training_update_end" \
    --qualification-role qualification \
    --capacity 288 \
    --batch-size-per-rank 4 \
    --qualification-batches 0 \
    --reuse-exact-suffix-offsets \
    --include-frozen-control \
    --diagnostic-negative-counts 999 \
    --diagnostic-evaluation-kind prequential \
    --candidate-seed 20260801 \
    --timing-repeats 1 \
    --output "$output" 2>&1 | tee "$log"
  validate_cell "$output" "$source_version" "$target_version" "$history_end" "$update_end"
  echo "completed cell: $cell_id"
done

python - "$result_root/round_manifest.json" "$cell_root" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

cells = []
for path in sorted(root.glob("*.json")):
    cells.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
if len(cells) != 3:
    raise RuntimeError("frozen control matrix is incomplete")
value = {"schema": "evokv_frozen_reuse_exact_prequential_round_v0", "status": "complete", "cells": cells}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if output.exists():
    if output.read_text() != encoded:
        raise RuntimeError("round manifest changed")
else:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, output)
PY

echo "round complete: $result_root/round_manifest.json"
echo "full K/V payloads retained: 0"
