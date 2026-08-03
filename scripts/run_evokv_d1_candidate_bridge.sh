#!/usr/bin/env bash
set -euo pipefail

round_label="${1:?round label is required}"
visible_devices="${EVOKV_CUDA_VISIBLE_DEVICES:-0,1}"
checkpoint_root="${EVOKV_D1_CHECKPOINT_ROOT:?EVOKV_D1_CHECKPOINT_ROOT is required}"
baseline_root="${EVOKV_D1_BASELINE_ROOT:?EVOKV_D1_BASELINE_ROOT is required}"
config="${EVOKV_D1_CONFIG:?EVOKV_D1_CONFIG is required}"
resume="${EVOKV_RESUME:-0}"
preflight_only="${EVOKV_PREFLIGHT_ONLY:-0}"
residual_rank="${EVOKV_D1_RESIDUAL_RANK:-}"
residual_ridge="${EVOKV_D1_RESIDUAL_RIDGE:-0.001}"
max_fit_tokens_per_rank="${EVOKV_D1_MAX_FIT_TOKENS_PER_RANK:-4096}"

if ! [[ "$round_label" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "invalid round label" >&2
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
if [[ "$resume" != "0" && "$resume" != "1" ]]; then
  echo "EVOKV_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "$preflight_only" != "0" && "$preflight_only" != "1" ]]; then
  echo "EVOKV_PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ -n "$residual_rank" ]] && ! [[ "$residual_rank" =~ ^[0-9]+$ ]]; then
  echo "EVOKV_D1_RESIDUAL_RANK must be a nonnegative integer" >&2
  exit 2
fi
if ! [[ "$max_fit_tokens_per_rank" =~ ^[0-9]+$ ]] || (( max_fit_tokens_per_rank < 2 )); then
  echo "EVOKV_D1_MAX_FIT_TOKENS_PER_RANK must be at least two" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
result_root="results/baseline_rounds/quality_chain/${round_label}"
log_root="logs/baseline_rounds/quality_chain/${round_label}"
summary_json="${result_root}/summary.json"
summary_tsv="${result_root}/summary.tsv"
lock_path="results/baseline_rounds/quality_chain/.${round_label}.lock"

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
  "scripts/evaluate_evokv_xp_d1_quality.py"
  "scripts/summarize_evokv_selected_d1_bridge.py"
  "scripts/run_evokv_d1_candidate_bridge.sh"
  "src/hstu_kvcache/migration/program.py"
  "src/hstu_kvcache/migration/stage45_oldkv.py"
  "src/hstu_kvcache/migration/xp_d1_quality.py"
  "src/hstu_kvcache/migration/xp_exact_baseline.py"
  "src/hstu_kvcache/migration/xp_residual_fit.py"
)
for path in "${inputs[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing input: $path" >&2
    exit 3
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
    or training.get("execution", {}).get("world_size") != 2
    or training.get("corpus_audit", {}).get("split_users", {}).get("train", {}).get("users") != 16384
    or training.get("corpus_audit", {}).get("split_users", {}).get("quality", {}).get("users") != 4096
    or len(training.get("updates", [])) != 4
    or summary.get("status") != "complete"
    or len(summary.get("edges", [])) != 3
    or any(
        edge.get("metrics", {}).get("sampled_cross_entropy", {}).get("exact_over_reuse_gain", 0.0) <= 0.0
        for edge in summary.get("edges", [])
    )
    or any(not (checkpoint / f"theta_{version}" / "manifest.json").is_file() for version in (1, 2, 3, 4))
):
    raise RuntimeError("D1 candidate quality chain differs")
PY

busy_pids="$({ nvidia-smi -i "$visible_devices" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true; } | sed '/^[[:space:]]*$/d' | sort -u)"
if [[ -n "$busy_pids" ]]; then
  echo "selected GPUs are busy: $busy_pids" >&2
  exit 4
fi
gib=$((1024 * 1024 * 1024))
disk_free_bytes="$(df -PB1 "$repo_root" | awk 'NR==2 {print $4}')"
memory_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if (( disk_free_bytes < 20 * gib )); then
  echo "at least 20 GiB free disk is required" >&2
  exit 4
fi
if (( memory_available_kib < 64 * 1024 * 1024 )); then
  echo "at least 64 GiB available DRAM is required" >&2
  exit 4
fi

mkdir -p "$(dirname "$lock_path")" "$result_root" "$log_root"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "D1 candidate round is already running" >&2
  exit 4
fi
input_hashes="${result_root}/input_hashes.tsv"
candidate="$(mktemp)"
trap 'rm -f "${candidate:-}"' EXIT
sha256sum "${inputs[@]}" > "$candidate"
if [[ -f "$input_hashes" ]]; then
  if ! cmp -s "$candidate" "$input_hashes"; then
    echo "D1 candidate inputs changed; choose a new round label" >&2
    exit 5
  fi
else
  mv "$candidate" "$input_hashes"
  candidate=""
fi

python - "$result_root/preflight.json" "$round_label" "$visible_devices" "$checkpoint_root" "$baseline_root" "$config" "$disk_free_bytes" "$memory_available_kib" "$residual_rank" "$residual_ridge" "$max_fit_tokens_per_rank" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = {
    "schema": "evokv_d1_candidate_bridge_preflight_v0",
    "status": "pass",
    "scientific_result": False,
    "formal_result": False,
    "round_label": sys.argv[2],
    "visible_devices": sys.argv[3],
    "checkpoint_root": sys.argv[4],
    "baseline_root": sys.argv[5],
    "config": sys.argv[6],
    "disk_free_bytes_at_start": int(sys.argv[7]),
    "memory_available_kib_at_start": int(sys.argv[8]),
    "evaluated_edges": ["theta1_to_theta2", "theta2_to_theta3", "theta3_to_theta4"],
    "negative_candidates": 999,
    "program_kind": (
        "analytic_direct_oldkv_control"
        if not sys.argv[9]
        else "label_free_shared_low_rank_residual"
    ),
    "residual_rank": None if not sys.argv[9] else int(sys.argv[9]),
    "residual_ridge": float(sys.argv[10]),
    "max_fit_tokens_per_rank": int(sys.argv[11]),
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
        "baseline_root",
        "config",
        "evaluated_edges",
        "negative_candidates",
        "program_kind",
        "residual_rank",
        "residual_ridge",
        "max_fit_tokens_per_rank",
        "full_kv_payloads_retained",
    ):
        if original.get(field) != value.get(field):
            raise RuntimeError("D1 candidate preflight changed")
else:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)
PY

if [[ "$preflight_only" == "1" ]]; then
  echo "preflight complete: $result_root/preflight.json"
  exit 0
fi

fit_arguments=(--fit-batches 4)
if [[ -n "$residual_rank" ]]; then
  fit_arguments=(
    --fit-batches 16
    --residual-rank "$residual_rank"
    --residual-ridge "$residual_ridge"
    --max-fit-tokens-per-rank "$max_fit_tokens_per_rank"
  )
fi

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
  if [[ -e "$output" || -e "$program" || -e "$plan" ]]; then
    if [[ "$resume" != "1" || ! -f "$output" || ! -f "$program" || ! -f "$plan" ]]; then
      echo "D1 candidate edge is partial or exists without resume: $edge" >&2
      exit 6
    fi
    echo "resume: reusing complete $edge"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$visible_devices" OMP_NUM_THREADS=1 torchrun \
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
    "${fit_arguments[@]}" \
    --probe-batches 4 \
    --qualification-batches 0 \
    --negative-count 999 \
    --candidate-seed 20260801 \
    --timing-repeats 1 \
    --output "$output" \
    --program-output "$program" \
    --action-plan-output "$plan" 2>&1 | tee "$log_root/${edge}.log"
done

python scripts/summarize_evokv_selected_d1_bridge.py \
  --result-root "$result_root" \
  --baseline-root "$baseline_root" \
  --output "$summary_json" \
  --tsv "$summary_tsv"
python - "$summary_json" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text())
if (
    value.get("status") != "complete"
    or value.get("endpoint_parity_with_selected_baseline") is not True
    or len(value.get("edges", [])) != 3
):
    raise RuntimeError("D1 candidate summary differs")
PY
du -sh "$result_root"
echo "round complete: $summary_json"
echo "full K/V payloads retained: 0"
