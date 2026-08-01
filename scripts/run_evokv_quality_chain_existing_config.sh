#!/usr/bin/env bash
set -euo pipefail

round_label="${1:?round label is required}"
schedule="${2:?schedule is required}"
benchmark="${3:?benchmark is required}"
visible_devices="${EVOKV_CUDA_VISIBLE_DEVICES:-0,1}"
resume="${EVOKV_RESUME:-0}"
preflight_only="${EVOKV_PREFLIGHT_ONLY:-0}"

if ! [[ "$round_label" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "invalid round label" >&2
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
  echo "EVOKV_CUDA_VISIBLE_DEVICES must name two GPUs" >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "$visible_devices"
if [[ "${devices[0]}" == "${devices[1]}" ]]; then
  echo "two distinct GPUs are required" >&2
  exit 2
fi

for command in python torchrun sha256sum nvidia-smi flock cmp awk df du tee; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 2
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ ! -f "$schedule" || ! -f "$benchmark" ]]; then
  echo "schedule or benchmark is absent" >&2
  exit 3
fi
result_root="results/baseline_rounds/quality_chain/${round_label}"
log_root="logs/baseline_rounds/quality_chain/${round_label}"
cell_root="${result_root}/cells"
checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/${round_label}"
base_checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/seed0"
training_result="${result_root}/training.json"
ledger_root="${result_root}/version_ledgers"
summary_json="${result_root}/summary.json"
summary_tsv="${result_root}/summary.tsv"
lock_path="results/baseline_rounds/quality_chain/.${round_label}.lock"

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
if [[ -e "$checkpoint_root" && "$resume" == "0" ]]; then
  echo "checkpoint root exists; choose a new label or set EVOKV_RESUME=1" >&2
  exit 3
fi

readarray -t generated_inputs < <(python - "$schedule" "$benchmark" <<'PY'
import json
import pathlib
import sys

schedule = pathlib.Path(sys.argv[1]).resolve()
benchmark = pathlib.Path(sys.argv[2]).resolve()
schedule_value = json.loads(schedule.read_text())
benchmark_value = json.loads(benchmark.read_text())
if (
    schedule_value.get("protocol") != "evokv_xp_prequential_stream_training_development_v1"
    or len(schedule_value.get("updates", [])) != 4
    or schedule_value.get("training", {}).get("epochs_per_update") != 1
    or benchmark_value.get("quality_chain", {}).get("evaluated_edges") != [[1, 2], [2, 3], [3, 4]]
    or benchmark_value.get("model", {}).get("embedding_width") != 4096
    or benchmark_value.get("model", {}).get("hidden_size") != 1536
    or benchmark_value.get("model", {}).get("layers") != 24
):
    raise RuntimeError("stream-aligned existing configuration differs")
for field in ("edge_inputs", "edge_summary"):
    print((schedule.parent / schedule_value[field]).resolve())
quality_roles = benchmark_value["data"]["quality_roles"]["path"]
print(pathlib.Path(quality_roles).resolve())
PY
)

inputs=(
  "$schedule"
  "$benchmark"
  "${generated_inputs[0]}"
  "${generated_inputs[1]}"
  "${generated_inputs[2]}"
  "checkpoints/evokv_xp_qk_e4096_h1536/seed0/theta_0/manifest.json"
  "scripts/train_evokv_xp_multiversion.py"
  "scripts/evaluate_evokv_xp_d1_quality.py"
  "scripts/summarize_evokv_quality_chain.py"
  "src/hstu_kvcache/migration/xp_d1_quality.py"
  "src/hstu_kvcache/migration/xp_exact_baseline.py"
  "src/hstu_kvcache/streaming/xp_multiversion.py"
  "src/hstu_kvcache/streaming/xp_projected_edge.py"
  "src/hstu_kvcache/streaming/xp_version_training.py"
  "scripts/run_evokv_quality_chain_existing_config.sh"
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
gib=$((1024 * 1024 * 1024))
disk_free_bytes="$(df -PB1 "$repo_root" | awk 'NR==2 {print $4}')"
memory_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if (( disk_free_bytes < 280 * gib )); then
  echo "need 180 GiB candidate assets plus a 100 GiB free floor" >&2
  exit 4
fi
if (( memory_available_kib < 64 * 1024 * 1024 )); then
  echo "at least 64 GiB available DRAM is required" >&2
  exit 4
fi

mkdir -p "$result_root" "$log_root" "$cell_root"
input_hashes="${result_root}/input_hashes.tsv"
candidate="$(mktemp)"
trap 'rm -f "${candidate:-}"' EXIT
sha256sum "${inputs[@]}" > "$candidate"
if [[ -f "$input_hashes" ]]; then
  if ! cmp -s "$candidate" "$input_hashes"; then
    echo "round inputs changed; choose a new round label" >&2
    exit 5
  fi
else
  mv "$candidate" "$input_hashes"
  candidate=""
fi

python - "$result_root/preflight.json" "$round_label" "$visible_devices" "$schedule" "$benchmark" "$disk_free_bytes" "$memory_available_kib" <<'PY'
import json
import os
import pathlib
import platform
import subprocess
import sys

import torch

path = pathlib.Path(sys.argv[1])
value = {
    "schema": "evokv_qk_stream_aligned_existing_config_preflight_v0",
    "status": "pass",
    "round_label": sys.argv[2],
    "visible_devices": sys.argv[3],
    "schedule": sys.argv[4],
    "benchmark": sys.argv[5],
    "disk_free_bytes_at_start": int(sys.argv[6]),
    "memory_available_kib_at_start": int(sys.argv[7]),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "full_kv_payloads_retained": 0,
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if path.exists():
    original = json.loads(path.read_text())
    for field in ("schema", "round_label", "visible_devices", "schedule", "benchmark", "full_kv_payloads_retained"):
        if original.get(field) != value.get(field):
            raise RuntimeError("preflight binding changed")
else:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)
PY

if [[ "$preflight_only" == "1" ]]; then
  echo "preflight complete: $result_root/preflight.json"
  exit 0
fi

validate_training() {
  python - "$training_result" "$checkpoint_root" <<'PY'
import json
import pathlib
import sys

result = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
if (
    result.get("status") != "complete"
    or result.get("downstream_d1_d2_gate_passed") is not True
    or result.get("execution", {}).get("world_size") != 2
    or len(result.get("updates", [])) != 4
    or any(not (root / f"theta_{version}" / "manifest.json").is_file() for version in (1, 2, 3, 4))
):
    raise RuntimeError("stream-aligned training result differs")
PY
}

if [[ -f "$training_result" ]]; then
  if [[ "$resume" != "1" ]]; then
    echo "training result exists without resume" >&2
    exit 6
  fi
  validate_training
else
  if [[ -e "$checkpoint_root" || -e "$ledger_root" ]]; then
    echo "partial training state exists; use a fresh round label" >&2
    exit 6
  fi
  CUDA_VISIBLE_DEVICES="$visible_devices" OMP_NUM_THREADS=1 torchrun \
    --standalone --nproc-per-node=2 scripts/train_evokv_xp_multiversion.py \
    --schedule "$schedule" \
    --base-checkpoint-root "$base_checkpoint_root" \
    --checkpoint-root "$checkpoint_root" \
    --output "$training_result" \
    --ledger-dir "$ledger_root" \
    --batch-size-per-rank 1 \
    --progress-every 500 2>&1 | tee "$log_root/training.log"
  validate_training
fi

for source_version in 1 2 3; do
  target_version=$((source_version + 1))
  history_end=$((72 + source_version * 8))
  update_end=$((history_end + 8))
  training_history_end=$((64 + source_version * 8))
  training_update_end=$((72 + source_version * 8))
  output="${cell_root}/theta${source_version}_to_theta${target_version}.json"
  log="${log_root}/theta${source_version}_to_theta${target_version}.log"
  if [[ -f "$output" ]]; then
    if [[ "$resume" != "1" ]]; then
      echo "diagnostic cell exists without resume" >&2
      exit 7
    fi
    continue
  fi
  CUDA_VISIBLE_DEVICES="$visible_devices" OMP_NUM_THREADS=1 torchrun \
    --standalone --nproc-per-node=2 scripts/evaluate_evokv_xp_d1_quality.py \
    --config "$benchmark" \
    --source-checkpoint-root "$checkpoint_root" \
    --target-checkpoint-root "$checkpoint_root" \
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
done

python scripts/summarize_evokv_quality_chain.py \
  --training-result "$training_result" \
  --cell-root "$cell_root" \
  --output "$summary_json" \
  --tsv "$summary_tsv" | tee "$log_root/summarize.log"

du -sh "$checkpoint_root" "$result_root"
echo "round complete: $summary_json"
echo "full K/V payloads retained: 0"
