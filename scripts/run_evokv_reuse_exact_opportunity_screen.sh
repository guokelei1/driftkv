#!/usr/bin/env bash
set -euo pipefail

round_label="${1:-reuse_exact_screen_round1}"
visible_devices="${EVOKV_CUDA_VISIBLE_DEVICES:-0,1}"
resume="${EVOKV_RESUME:-0}"
preflight_only="${EVOKV_PREFLIGHT_ONLY:-0}"

if ! [[ "$round_label" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "round label must contain only letters, digits, underscore, or dash" >&2
  exit 2
fi
if [[ "$resume" != "0" && "$resume" != "1" ]]; then
  echo "EVOKV_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "$preflight_only" != "0" && "$preflight_only" != "1" ]]; then
  echo "EVOKV_PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$visible_devices" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "EVOKV_CUDA_VISIBLE_DEVICES must name exactly two GPUs" >&2
  exit 2
fi

IFS=',' read -r -a device_list <<< "$visible_devices"
if [[ "${device_list[0]}" == "${device_list[1]}" ]]; then
  echo "two distinct GPUs are required" >&2
  exit 2
fi

for command in python torchrun jq sha256sum nvidia-smi git flock tee cmp mktemp; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 2
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_root="results/baseline_rounds/reuse_exact_opportunity/${round_label}"
log_root="logs/baseline_rounds/reuse_exact_opportunity/${round_label}"
cell_root="${result_root}/cells"
preflight="${result_root}/preflight.json"
matrix_path="${result_root}/matrix.json"
summary_json="${result_root}/summary.json"
summary_tsv="${result_root}/summary.tsv"
input_hashes="${result_root}/input_hashes.tsv"
benchmark_config="configs/evokv_baselines/x_qk_xp_multiversion_two_gpu_baseline_v1.json"
base_checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/seed0"
target_checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/rounds/baseline_round3"
lock_path="results/baseline_rounds/reuse_exact_opportunity/.${round_label}.lock"

mkdir -p "$(dirname "$lock_path")"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "another process owns round label: $round_label" >&2
  exit 3
fi
if [[ -e "$result_root" && "$resume" == "0" ]]; then
  echo "result root exists; use a new label or EVOKV_RESUME=1: $result_root" >&2
  exit 3
fi

required_inputs=(
  "$benchmark_config"
  "configs/evokv_foundation/qk_xp_fixed_edge_inputs_summary.json"
  "data/processed/evokv_foundation/qk_xp_fixed_edge_inputs.npz"
  "${base_checkpoint_root}/theta_0/manifest.json"
  "${target_checkpoint_root}/theta_1/manifest.json"
  "${target_checkpoint_root}/theta_2/manifest.json"
  "${target_checkpoint_root}/theta_3/manifest.json"
  "scripts/evaluate_evokv_xp_d1_quality.py"
  "scripts/summarize_evokv_reuse_exact_opportunity.py"
  "scripts/run_evokv_reuse_exact_opportunity_screen.sh"
  "src/hstu_kvcache/migration/xp_d1_quality.py"
  "src/hstu_kvcache/migration/xp_exact_baseline.py"
  "src/hstu_kvcache/streaming/xp_multiversion.py"
  "src/hstu_kvcache/streaming/xp_projected_edge.py"
  "src/hstu_kvcache/streaming/xp_version_training.py"
  "src/hstu_kvcache/streaming/sharded_edge.py"
  "src/hstu_kvcache/streaming/trainer.py"
)
for path in "${required_inputs[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "required input is absent: $path" >&2
    exit 4
  fi
done

gpu_uuids=()
for device in "${device_list[@]}"; do
  gpu_uuid="$(nvidia-smi -i "$device" --query-gpu=uuid --format=csv,noheader,nounits)"
  gpu_uuids+=("$gpu_uuid")
done
gpu_uuid_csv="${gpu_uuids[0]},${gpu_uuids[1]}"
busy_pids="$({
  nvidia-smi -i "$visible_devices" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
} | sed '/^[[:space:]]*$/d' | sort -u)"
if [[ -n "$busy_pids" ]]; then
  echo "selected GPUs already have compute processes: $busy_pids" >&2
  exit 4
fi

gib=$((1024 * 1024 * 1024))
disk_free_bytes="$(df -PB1 "$repo_root" | awk 'NR==2 {print $4}')"
memory_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if (( disk_free_bytes < 10 * gib )); then
  echo "at least 10 GiB free disk is required" >&2
  exit 4
fi
if (( memory_available_kib < 64 * 1024 * 1024 )); then
  echo "at least 64 GiB available DRAM is required" >&2
  exit 4
fi

mkdir -p "$result_root" "$log_root" "$cell_root"
hash_candidate="$(mktemp)"
trap 'if [[ -n "${hash_candidate:-}" ]]; then rm -f "$hash_candidate"; fi' EXIT
sha256sum "${required_inputs[@]}" > "$hash_candidate"
if [[ -f "$input_hashes" ]]; then
  if ! cmp -s "$hash_candidate" "$input_hashes"; then
    echo "screen inputs changed; use a new round label" >&2
    exit 5
  fi
else
  mv "$hash_candidate" "$input_hashes"
  hash_candidate=""
fi

python - "$preflight" "$round_label" "$visible_devices" "$gpu_uuid_csv" \
  "$disk_free_bytes" "$memory_available_kib" "$benchmark_config" \
  "$base_checkpoint_root" "$target_checkpoint_root" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

import torch

output = pathlib.Path(sys.argv[1])
base_root = pathlib.Path(sys.argv[8])
target_root = pathlib.Path(sys.argv[9])
manifests = [
    base_root / "theta_0" / "manifest.json",
    *(target_root / f"theta_{version}" / "manifest.json" for version in (1, 2, 3)),
]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

descriptors = []
for expected_version, manifest_path in enumerate(manifests):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != expected_version or manifest.get("world_size") != 2:
        raise RuntimeError(f"checkpoint manifest differs: {manifest_path}")
    artifacts = [manifest["dense"], manifest["projection"], *manifest["embedding_shards"]]
    for artifact in artifacts:
        path = manifest_path.parent / artifact["path"]
        if not path.is_file() or path.stat().st_size != int(artifact["bytes"]):
            raise RuntimeError(f"checkpoint artifact differs: {path}")
    descriptors.append(
        {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "version": expected_version,
            "artifacts_verified": True,
            "verification": "existence_and_declared_bytes",
        }
    )

value = {
    "schema": "evokv_reuse_exact_opportunity_screen_preflight_v0",
    "status": "pass",
    "round_label": sys.argv[2],
    "visible_devices": sys.argv[3],
    "gpu_uuids": sys.argv[4].split(","),
    "disk_free_bytes_at_start": int(sys.argv[5]),
    "memory_available_kib_at_start": int(sys.argv[6]),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    "benchmark_config": {
        "path": sys.argv[7],
        "sha256": sha256(pathlib.Path(sys.argv[7])),
    },
    "checkpoint_manifests": descriptors,
    "methods": ["all_reuse", "all_exact"],
    "negative_count": 999,
    "common_cache_endpoint": {
        "storage_dtype": "torch.float16",
        "consumption_dtype": "torch.float32",
    },
    "retention": {
        "keep": ["compact JSON metrics", "paired target contributions", "logs", "hashes"],
        "discard": ["full K/V payloads", "programs", "action plans", "GPU intermediates"],
    },
}
if output.exists():
    prior = json.loads(output.read_text())
    stable = (
        "schema",
        "round_label",
        "visible_devices",
        "gpu_uuids",
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "benchmark_config",
        "checkpoint_manifests",
        "methods",
        "negative_count",
        "common_cache_endpoint",
    )
    if any(prior.get(field) != value.get(field) for field in stable):
        raise RuntimeError("opportunity preflight binding changed")
else:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
PY

python - "$matrix_path" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
cells = []
for source in range(3):
    target = source + 1
    history = 72 + source * 8
    cells.append(
        {
            "cell_id": f"preq_theta{source}_to_theta{target}_h{history}",
            "category": "prequential",
            "source_version": source,
            "target_version": target,
            "history_end": history,
            "update_end": history + 8,
            "training_history_end": 64 + source * 8,
            "training_update_end": 72 + source * 8,
            "qualification_role": "qualification",
        }
    )
    for anchor in (145, 396):
        cells.append(
            {
                "cell_id": f"long_h{anchor}_theta{source}_to_theta{target}",
                "category": "long_context_characterization",
                "source_version": source,
                "target_version": target,
                "history_end": anchor,
                "update_end": anchor + 8,
                "training_history_end": None,
                "training_update_end": None,
                "qualification_role": "theta12",
            }
        )
value = {
    "schema": "evokv_reuse_exact_opportunity_screen_matrix_v0",
    "selection_policy": "all_nine_cells_reported_without_selection",
    "cells": cells,
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if path.exists():
    if path.read_text() != encoded:
        raise RuntimeError("opportunity matrix changed")
else:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)
PY

echo "preflight passed: GPUs=${visible_devices}, disk_free=$((disk_free_bytes / gib)) GiB"
if [[ "$preflight_only" == "1" ]]; then
  echo "preflight-only complete: $preflight"
  exit 0
fi

validate_cell() {
  local output="$1"
  local source_version="$2"
  local target_version="$3"
  local history_end="$4"
  local update_end="$5"
  local role="$6"
  local evaluation_kind="$7"
  python - "$output" "$source_version" "$target_version" "$history_end" \
    "$update_end" "$role" "$evaluation_kind" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
edge = value.get("edge", {})
quality = value.get("quality_by_negative_count", {})
expected_semantics = (
    "next_unseen_window"
    if sys.argv[7] == "prequential"
    else "nonprequential_long_context_characterization"
)
if (
    value.get("protocol") != "evokv_xp_reuse_exact_suffix_diagnostic_development_v0"
    or value.get("status") != "complete"
    or value.get("scientific_result") is not False
    or value.get("world_size") != 2
    or value.get("evaluation_kind") != sys.argv[7]
    or edge.get("source_version") != int(sys.argv[2])
    or edge.get("target_version") != int(sys.argv[3])
    or edge.get("history_end") != int(sys.argv[4])
    or edge.get("update_end") != int(sys.argv[5])
    or edge.get("evaluation_window", {}).get("semantics") != expected_semantics
    or value.get("role", {}).get("source_role") != sys.argv[6]
    or set(quality) != {"999"}
    or set(quality["999"].get("methods", {})) != {"all_reuse", "all_exact"}
    or value.get("recommendation_contract", {}).get("negative_candidates") != [999]
    or value.get("recommendation_contract", {}).get("common_cache_endpoint", {}).get("storage_dtype") != "torch.float16"
):
    raise RuntimeError(f"opportunity cell validation failed: {path}")
PY
}

run_cell() {
  local cell_id="$1"
  local source_version="$2"
  local target_version="$3"
  local history_end="$4"
  local update_end="$5"
  local training_history_end="$6"
  local training_update_end="$7"
  local role="$8"
  local evaluation_kind="$9"
  local source_root
  local output="${cell_root}/${cell_id}.json"
  local log="${log_root}/${cell_id}.log"
  if [[ "$source_version" == "0" ]]; then
    source_root="$base_checkpoint_root"
  else
    source_root="$target_checkpoint_root"
  fi
  if [[ -f "$output" ]]; then
    if [[ "$resume" != "1" ]]; then
      echo "cell output already exists: $output" >&2
      exit 6
    fi
    validate_cell "$output" "$source_version" "$target_version" "$history_end" "$update_end" "$role" "$evaluation_kind"
    echo "resume: validated completed cell $cell_id"
    return
  fi
  extra_args=()
  if [[ "$training_history_end" != "none" ]]; then
    extra_args+=(
      --training-history-end "$training_history_end"
      --training-update-end "$training_update_end"
    )
  fi
  echo "starting cell: $cell_id"
  CUDA_VISIBLE_DEVICES="$visible_devices" OMP_NUM_THREADS=1 torchrun \
    --standalone --nproc-per-node=2 scripts/evaluate_evokv_xp_d1_quality.py \
    --config "$benchmark_config" \
    --source-checkpoint-root "$source_root" \
    --target-checkpoint-root "$target_checkpoint_root" \
    --source-version "$source_version" \
    --target-version "$target_version" \
    --history-end "$history_end" \
    --update-end "$update_end" \
    --qualification-role "$role" \
    --capacity 288 \
    --batch-size-per-rank 4 \
    --qualification-batches 0 \
    --reuse-exact-suffix-offsets \
    --diagnostic-negative-counts 999 \
    --diagnostic-evaluation-kind "$evaluation_kind" \
    --candidate-seed 20260801 \
    --timing-repeats 1 \
    --output "$output" \
    "${extra_args[@]}" 2>&1 | tee "$log"
  validate_cell "$output" "$source_version" "$target_version" "$history_end" "$update_end" "$role" "$evaluation_kind"
  echo "completed cell: $cell_id"
}

for source_version in 0 1 2; do
  target_version=$((source_version + 1))
  history_end=$((72 + source_version * 8))
  training_history_end=$((64 + source_version * 8))
  training_update_end=$((72 + source_version * 8))
  run_cell \
    "preq_theta${source_version}_to_theta${target_version}_h${history_end}" \
    "$source_version" "$target_version" "$history_end" "$((history_end + 8))" \
    "$training_history_end" "$training_update_end" qualification prequential
  for anchor in 145 396; do
    run_cell \
      "long_h${anchor}_theta${source_version}_to_theta${target_version}" \
      "$source_version" "$target_version" "$anchor" "$((anchor + 8))" \
      none none theta12 long_context_characterization
  done
done

if [[ -e "$summary_json" || -e "$summary_tsv" ]]; then
  if [[ "$resume" == "1" && -f "$summary_json" && -f "$summary_tsv" ]]; then
    jq -e '.schema == "evokv_reuse_exact_opportunity_screen_summary_v0" and .status == "complete" and .matrix_complete == true and .cells_reported == 9' "$summary_json" >/dev/null
    echo "resume: validated completed summary"
  else
    echo "summary artifact state is incomplete or conflicts: $summary_json $summary_tsv" >&2
    exit 7
  fi
else
  python scripts/summarize_evokv_reuse_exact_opportunity.py \
    --round-label "$round_label" \
    --result-root "$result_root" \
    --preflight "$preflight" \
    --benchmark-config "$benchmark_config" \
    --base-checkpoint-root "$base_checkpoint_root" \
    --target-checkpoint-root "$target_checkpoint_root" \
    --output-json "$summary_json" \
    --output-tsv "$summary_tsv"
fi

echo "round complete: $summary_json"
echo "full K/V payloads retained: 0"
