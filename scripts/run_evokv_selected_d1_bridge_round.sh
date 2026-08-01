#!/usr/bin/env bash
set -euo pipefail

round_label="${1:-selected_d1_bridge_round1}"
visible_devices="${EVOKV_CUDA_VISIBLE_DEVICES:-0,1}"
resume="${EVOKV_RESUME:-0}"
preflight_only="${EVOKV_PREFLIGHT_ONLY:-0}"

if ! [[ "$round_label" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "invalid round label" >&2
  exit 2
fi
if [[ "$visible_devices" != "0,1" ]]; then
  echo "current repository availability permits only GPU0/GPU1" >&2
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

for command in python torchrun sha256sum nvidia-smi flock cmp awk df tee; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 2
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/quality_chain_stream_aligned_train16384_round1"
baseline_root="results/baseline_rounds/quality_chain/quality_chain_stream_aligned_train16384_round1"
config="configs/evokv_quality/x_qk_xp_quality_stream_aligned_train16384_qual4096_nested_v2.json"
result_root="results/baseline_rounds/quality_chain/${round_label}"
log_root="logs/baseline_rounds/quality_chain/${round_label}"
summary_json="${result_root}/summary.json"
summary_tsv="${result_root}/summary.tsv"
lock_path="results/baseline_rounds/quality_chain/.${round_label}.lock"

mkdir -p "$(dirname "$lock_path")"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "round is already running" >&2
  exit 3
fi

inputs=(
  "$config"
  "${baseline_root}/training.json"
  "${baseline_root}/summary.json"
  "${baseline_root}/cells/theta1_to_theta2.json"
  "${baseline_root}/cells/theta2_to_theta3.json"
  "${baseline_root}/cells/theta3_to_theta4.json"
  "${checkpoint_root}/theta_1/manifest.json"
  "${checkpoint_root}/theta_2/manifest.json"
  "${checkpoint_root}/theta_3/manifest.json"
  "${checkpoint_root}/theta_4/manifest.json"
  "data/processed/evokv_quality/qk_xp_quality_stream_aligned_train16384_qual4096_nested_v2.npz"
  "scripts/evaluate_evokv_xp_d1_quality.py"
  "scripts/summarize_evokv_selected_d1_bridge.py"
  "scripts/run_evokv_selected_d1_bridge_round.sh"
  "src/hstu_kvcache/migration/program.py"
  "src/hstu_kvcache/migration/stage45_oldkv.py"
  "src/hstu_kvcache/migration/xp_d1_quality.py"
  "src/hstu_kvcache/migration/xp_exact_baseline.py"
)
for path in "${inputs[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing input: $path" >&2
    exit 4
  fi
done

python - "$baseline_root" "$checkpoint_root" <<'PY'
import json
import pathlib
import sys

baseline = pathlib.Path(sys.argv[1])
checkpoint = pathlib.Path(sys.argv[2])
training = json.loads((baseline / "training.json").read_text())
summary = json.loads((baseline / "summary.json").read_text())
if (
    training.get("status") != "complete"
    or training.get("downstream_d1_d2_gate_passed") is not True
    or training.get("stack_identity")
    != "xp_qk_stream_aligned_warmup_train16384_qual4096_e1_fixed010_development_v1"
    or training.get("corpus_audit", {}).get("split_users", {}).get("train", {}).get("users") != 16384
    or training.get("corpus_audit", {}).get("split_users", {}).get("quality", {}).get("users") != 4096
    or len(training.get("updates", [])) != 4
    or summary.get("status") != "complete"
    or len(summary.get("edges", [])) != 3
    or any(
        edge.get("metrics", {}).get("sampled_cross_entropy", {}).get(
            "exact_over_reuse_gain", 0.0
        ) <= 0.0
        for edge in summary.get("edges", [])
    )
    or any(not (checkpoint / f"theta_{version}" / "manifest.json").is_file() for version in (1, 2, 3, 4))
):
    raise RuntimeError("selected quality chain differs")
PY

busy_pids="$({
  nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
} | sed '/^[[:space:]]*$/d' | sort -u)"
if [[ -n "$busy_pids" ]]; then
  echo "GPU0/GPU1 are busy: $busy_pids" >&2
  exit 4
fi

gib=$((1024 * 1024 * 1024))
disk_free_bytes="$(df -PB1 "$repo_root" | awk 'NR==2 {print $4}')"
memory_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if (( disk_free_bytes < 120 * gib )); then
  echo "at least 120 GiB free disk is required" >&2
  exit 4
fi
if (( memory_available_kib < 64 * 1024 * 1024 )); then
  echo "at least 64 GiB available DRAM is required" >&2
  exit 4
fi

mkdir -p "$result_root" "$log_root"
input_hashes="${result_root}/input_hashes.tsv"
candidate="$(mktemp)"
trap 'rm -f "${candidate:-}"' EXIT
sha256sum "${inputs[@]}" > "$candidate"
if [[ -f "$input_hashes" ]]; then
  if ! cmp -s "$candidate" "$input_hashes"; then
    echo "round inputs changed; use a new round label" >&2
    exit 5
  fi
else
  mv "$candidate" "$input_hashes"
  candidate=""
fi

python - "$result_root/preflight.json" "$round_label" "$disk_free_bytes" "$memory_available_kib" <<'PY'
import json
import os
import pathlib
import platform
import subprocess
import sys

import torch

path = pathlib.Path(sys.argv[1])
value = {
    "schema": "evokv_selected_d1_bridge_preflight_development_v0",
    "status": "pass",
    "round_label": sys.argv[2],
    "visible_devices": "0,1",
    "disk_free_bytes_at_start": int(sys.argv[3]),
    "memory_available_kib_at_start": int(sys.argv[4]),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "checkpoint_root": "checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/quality_chain_stream_aligned_train16384_round1",
    "evaluated_edges": ["theta1_to_theta2", "theta2_to_theta3", "theta3_to_theta4"],
    "negative_candidates": 999,
    "common_cache_storage": "torch.float16",
    "full_kv_payloads_retained": 0,
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if path.exists():
    original = json.loads(path.read_text())
    for field in (
        "schema",
        "round_label",
        "visible_devices",
        "checkpoint_root",
        "evaluated_edges",
        "negative_candidates",
        "common_cache_storage",
        "full_kv_payloads_retained",
    ):
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

validate_edge() {
  python - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import hashlib
import json
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
program_path = pathlib.Path(sys.argv[2])
plan_path = pathlib.Path(sys.argv[3])
baseline_path = pathlib.Path(sys.argv[4])
source = int(sys.argv[5])
target = int(sys.argv[6])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

if not result_path.is_file() or not program_path.is_file() or not plan_path.is_file():
    raise RuntimeError("D1 bridge artifacts are incomplete")
result = json.loads(result_path.read_text())
baseline = json.loads(baseline_path.read_text())
quality = result.get("quality", {}).get("qualification_test", {})
if (
    result.get("protocol") != "evokv_xp_d1_quality_development_v1"
    or result.get("status") != "complete"
    or result.get("world_size") != 2
    or result.get("edge", {}).get("source_version") != source
    or result.get("edge", {}).get("target_version") != target
    or result.get("recommendation_contract", {}).get("negative_candidates") != 999
    or result.get("recommendation_contract", {}).get("common_cache_endpoint", {}).get("storage_dtype") != "torch.float16"
    or set(quality.get("methods", {})) != {"all_reuse", "compiled_direct_oldkv", "mixed_fixed20", "all_exact"}
    or result.get("roles", {}).get("qualification_test", {}).get("candidate_sha256_per_rank")
    != baseline.get("role", {}).get("candidate_sha256_per_rank_by_negative_count", {}).get("999")
    or quality.get("record_ids_sha256")
    != baseline.get("quality_by_negative_count", {}).get("999", {}).get("record_ids_sha256")
    or result.get("bindings", {}).get("program", {}).get("sha256") != sha256(program_path)
    or result.get("bindings", {}).get("action_plan", {}).get("sha256") != sha256(plan_path)
):
    raise RuntimeError("D1 bridge artifact validation failed")
PY
}

for source_version in 1 2 3; do
  target_version=$((source_version + 1))
  history_end=$((72 + source_version * 8))
  update_end=$((history_end + 8))
  training_history_end=$((64 + source_version * 8))
  training_update_end=$((72 + source_version * 8))
  edge="theta${source_version}_to_theta${target_version}"
  output="${result_root}/${edge}.json"
  program="${result_root}/${edge}_direct_oldkv_fp16.pt"
  plan="${result_root}/${edge}_action_plan_v2.json"
  baseline="${baseline_root}/cells/${edge}.json"
  log="${log_root}/${edge}.log"
  if [[ -e "$output" || -e "$program" || -e "$plan" ]]; then
    if [[ "$resume" != "1" ]]; then
      echo "D1 bridge artifacts exist without EVOKV_RESUME=1: $edge" >&2
      exit 6
    fi
    validate_edge "$output" "$program" "$plan" "$baseline" "$source_version" "$target_version"
    echo "resume: validated $edge"
    continue
  fi
  CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun \
    --standalone --nproc-per-node=2 scripts/evaluate_evokv_xp_d1_quality.py \
    --config "$config" \
    --source-checkpoint-root "$checkpoint_root" \
    --target-checkpoint-root "$checkpoint_root" \
    --source-version "$source_version" \
    --target-version "$target_version" \
    --history-end "$history_end" \
    --update-end "$update_end" \
    --training-history-end "$training_history_end" \
    --training-update-end "$training_update_end" \
    --probe-role theta12 \
    --qualification-role qualification \
    --capacity 288 \
    --batch-size-per-rank 4 \
    --fit-batches 4 \
    --probe-batches 4 \
    --qualification-batches 0 \
    --negative-count 999 \
    --candidate-seed 20260801 \
    --timing-repeats 1 \
    --output "$output" \
    --program-output "$program" \
    --action-plan-output "$plan" 2>&1 | tee "$log"
  validate_edge "$output" "$program" "$plan" "$baseline" "$source_version" "$target_version"
  echo "completed and validated: $edge"
done

python scripts/summarize_evokv_selected_d1_bridge.py \
  --result-root "$result_root" \
  --baseline-root "$baseline_root" \
  --output "$summary_json" \
  --tsv "$summary_tsv"

du -sh "$result_root"
df -h "$repo_root"
echo "round complete: $summary_json"
echo "full K/V payloads retained: 0"
