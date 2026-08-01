#!/usr/bin/env bash
set -euo pipefail

round_label="${1:-quality_chain_stream_aligned_round1}"
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

for command in python torchrun sha256sum nvidia-smi flock tee cmp awk df; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 2
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
result_root="results/baseline_rounds/quality_chain/${round_label}"
log_root="logs/baseline_rounds/quality_chain/${round_label}"
cell_root="${result_root}/cells"
checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/${round_label}"
base_checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/seed0"
previous_roles="configs/evokv_quality/qk_quality_roles_train8192_qual4096_v0.json"
roles="configs/evokv_quality/qk_quality_roles_stream_aligned_train8192_qual4096_v1.json"
edge_input="data/processed/evokv_quality/qk_xp_quality_stream_aligned_train8192_qual4096_v1.npz"
edge_summary="configs/evokv_quality/qk_xp_quality_stream_aligned_train8192_qual4096_v1_summary.json"
schedule="configs/evokv_quality/xp_qk_stream_aligned_train8192_qual4096_e1_v1.json"
benchmark="configs/evokv_quality/x_qk_xp_quality_stream_aligned_train8192_qual4096_v1.json"
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

inputs=(
  "data/tenrec/Tenrec.zip"
  "data/processed/evokv_foundation/qk_full_user_lengths.npz"
  "data/processed/evokv_d3_m1_qk_entity_2560.npz"
  "data/processed/evokv_d3_m1_qk_entity_cache/entity_catalog_base64_top250000.npz"
  "configs/evokv_foundation/qk_post_base_roles.json"
  "$previous_roles"
  "configs/evokv_foundation/xp_qk_multiversion_prequential3_development_v1.json"
  "configs/evokv_baselines/x_qk_xp_multiversion_two_gpu_baseline_v1.json"
  "${base_checkpoint_root}/theta_0/manifest.json"
  "scripts/build_evokv_qk_quality_roles.py"
  "scripts/build_evokv_qk_xp_edge_inputs.py"
  "scripts/build_evokv_quality_development_configs.py"
  "scripts/train_evokv_xp_multiversion.py"
  "scripts/evaluate_evokv_xp_d1_quality.py"
  "scripts/summarize_evokv_quality_chain.py"
  "src/hstu_kvcache/data/qk_xp_edge_inputs.py"
  "src/hstu_kvcache/migration/foundation_workload.py"
  "src/hstu_kvcache/migration/xp_d1_quality.py"
  "src/hstu_kvcache/migration/xp_exact_baseline.py"
  "src/hstu_kvcache/streaming/xp_multiversion.py"
  "src/hstu_kvcache/streaming/xp_projected_edge.py"
  "src/hstu_kvcache/streaming/xp_version_training.py"
  "scripts/run_evokv_quality_chain_stream_aligned_round.sh"
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
input_hashes="${result_root}/static_input_hashes.tsv"
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

python - "$result_root/preflight.json" "$result_root/preflight_latest.json" "$round_label" "$visible_devices" "$disk_free_bytes" "$memory_available_kib" <<'PY'
import json
import os
import pathlib
import platform
import subprocess
import sys

import torch

path = pathlib.Path(sys.argv[1])
latest = pathlib.Path(sys.argv[2])
value = {
    "schema": "evokv_qk_stream_aligned_quality_chain_preflight_v0",
    "status": "pass",
    "round_label": sys.argv[3],
    "visible_devices": sys.argv[4],
    "disk_free_bytes_at_start": int(sys.argv[5]),
    "memory_available_kib_at_start": int(sys.argv[6]),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "candidate": {
        "bootstrap_edge": "theta0_to_theta1",
        "bootstrap_edge_used_as_d1_evidence": False,
        "evaluated_edges": ["theta1_to_theta2", "theta2_to_theta3", "theta3_to_theta4"],
        "training_users": 8192,
        "qualification_users": 4096,
        "epochs_per_update": 1,
        "negative_count": 999,
    },
    "retention": {
        "keep": ["theta1-theta4 checkpoints", "compact metrics", "bindings", "logs"],
        "discard": ["all K/V payloads", "candidate tensors", "GPU staging state"],
    },
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if path.exists():
    original = json.loads(path.read_text())
    for field in ("schema", "round_label", "visible_devices", "candidate", "retention"):
        if original.get(field) != value.get(field):
            raise RuntimeError("preflight binding changed")
else:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)
temporary = latest.with_name(f".{latest.name}.{os.getpid()}.tmp")
temporary.write_text(encoded)
os.replace(temporary, latest)
PY

if [[ "$preflight_only" == "1" ]]; then
  echo "preflight complete: $result_root/preflight.json"
  exit 0
fi

python scripts/build_evokv_qk_quality_roles.py \
  --output "$roles" \
  --hash-salt evokv-qk-quality-stream-aligned-train8192-qual4096-v1 \
  --theta01-users 8192 \
  --qualification-users 4096 \
  --minimum-events 104 \
  --additional-exclusion-roles "$previous_roles" | tee "$log_root/build_roles.log"

if [[ -f "$edge_input" || -f "$edge_summary" ]]; then
  if [[ ! -f "$edge_input" || ! -f "$edge_summary" ]]; then
    echo "quality edge corpus is partial" >&2
    exit 6
  fi
  python - "$edge_input" "$edge_summary" <<'PY'
import hashlib
import json
import pathlib
import sys

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

edge = pathlib.Path(sys.argv[1])
summary = json.loads(pathlib.Path(sys.argv[2]).read_text())
if (
    summary.get("status") != "pass"
    or summary.get("records", {}).get("theta01") != 8192
    or summary.get("records", {}).get("qualification") != 4096
    or summary.get("boundaries", {}).get("theta01", {}).get("update") != [64, 104]
    or summary.get("artifact", {}).get("file_sha256") != sha256(edge)
):
    raise RuntimeError("existing stream-aligned edge corpus differs")
PY
else
  python scripts/build_evokv_qk_xp_edge_inputs.py \
    --roles "$roles" \
    --output "$edge_input" \
    --summary "$edge_summary" \
    --hash-salt evokv-qk-quality-stream-aligned-train8192-qual4096-v1 \
    --theta01-update-end 104 \
    --qualification-update-end 104 \
    --theta01-users 8192 \
    --theta12-users 2048 \
    --qualification-users 4096 2>&1 | tee "$log_root/build_edge_inputs.log"
fi

python scripts/build_evokv_quality_development_configs.py \
  --mode warmup_plus_three \
  --edge-input "$edge_input" \
  --edge-summary "$edge_summary" \
  --quality-roles "$roles" \
  --schedule-output "$schedule" \
  --benchmark-output "$benchmark" | tee "$log_root/build_configs.log"

python - "$schedule" "$benchmark" <<'PY'
import sys

from hstu_kvcache.migration.xp_exact_baseline import load_fixed_inputs
from hstu_kvcache.streaming.xp_multiversion import load_xp_multiversion_schedule

schedule = load_xp_multiversion_schedule(sys.argv[1])
inputs = load_fixed_inputs(sys.argv[2], "288", world_size=2)
if schedule.epochs_per_update != 1 or len(schedule.updates) != 4:
    raise RuntimeError("stream-aligned schedule differs")
if inputs.benchmark["quality_chain"]["evaluated_edges"] != [[1, 2], [2, 3], [3, 4]]:
    raise RuntimeError("stream-aligned benchmark differs")
PY

generated_hashes="${result_root}/generated_input_hashes.tsv"
generated_candidate="$(mktemp)"
sha256sum "$roles" "$edge_input" "$edge_summary" "$schedule" "$benchmark" > "$generated_candidate"
if [[ -f "$generated_hashes" ]]; then
  if ! cmp -s "$generated_candidate" "$generated_hashes"; then
    echo "generated corpus or config changed; choose a new round label" >&2
    exit 7
  fi
else
  mv "$generated_candidate" "$generated_hashes"
  generated_candidate=""
fi
rm -f "${generated_candidate:-}"

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
    exit 8
  fi
  validate_training
else
  if [[ -e "$checkpoint_root" || -e "$ledger_root" ]]; then
    echo "partial training state exists; use a fresh round label" >&2
    exit 8
  fi
  CUDA_VISIBLE_DEVICES="$visible_devices" OMP_NUM_THREADS=1 torchrun \
    --standalone --nproc-per-node=2 scripts/train_evokv_xp_multiversion.py \
    --schedule "$schedule" \
    --base-checkpoint-root "$base_checkpoint_root" \
    --checkpoint-root "$checkpoint_root" \
    --output "$training_result" \
    --ledger-dir "$ledger_root" \
    --batch-size-per-rank 1 \
    --progress-every 250 2>&1 | tee "$log_root/training.log"
  validate_training
fi

validate_cell() {
  python - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text())
edge = value.get("edge", {})
if (
    value.get("status") != "complete"
    or value.get("world_size") != 2
    or value.get("evaluation_kind") != "prequential"
    or edge.get("source_version") != int(sys.argv[2])
    or edge.get("target_version") != int(sys.argv[3])
    or edge.get("history_end") != int(sys.argv[4])
    or edge.get("update_end") != int(sys.argv[5])
    or set(value.get("quality_by_negative_count", {})) != {"999"}
    or set(value.get("frozen_quality_by_negative_count", {})) != {"999"}
):
    raise RuntimeError("stream-aligned quality cell differs")
PY
}

for source_version in 1 2 3; do
  target_version=$((source_version + 1))
  history_end=$((72 + source_version * 8))
  update_end=$((history_end + 8))
  training_history_end=$((64 + source_version * 8))
  training_update_end=$((72 + source_version * 8))
  cell_id="theta${source_version}_to_theta${target_version}"
  output="${cell_root}/${cell_id}.json"
  log="${log_root}/${cell_id}.log"
  if [[ -f "$output" ]]; then
    if [[ "$resume" != "1" ]]; then
      echo "diagnostic cell exists without resume" >&2
      exit 9
    fi
    validate_cell "$output" "$source_version" "$target_version" "$history_end" "$update_end"
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
  validate_cell "$output" "$source_version" "$target_version" "$history_end" "$update_end"
done

python scripts/summarize_evokv_quality_chain.py \
  --training-result "$training_result" \
  --cell-root "$cell_root" \
  --output "$summary_json" \
  --tsv "$summary_tsv" | tee "$log_root/summarize.log"

du -sh "$checkpoint_root" "$result_root" "$edge_input" || true
echo "round complete: $summary_json"
echo "full K/V payloads retained: 0"
