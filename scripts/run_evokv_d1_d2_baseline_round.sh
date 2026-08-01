#!/usr/bin/env bash
set -euo pipefail

round_label="${1:-baseline_round3}"
visible_devices="${EVOKV_CUDA_VISIBLE_DEVICES:-0,1}"
resume="${EVOKV_RESUME:-0}"
preflight_only="${EVOKV_PREFLIGHT_ONLY:-0}"
persistent_cap_gib="${EVOKV_PERSISTENT_CAP_GIB:-450}"
disk_floor_gib="${EVOKV_DISK_FLOOR_GIB:-100}"

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
if ! [[ "$persistent_cap_gib" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVOKV_PERSISTENT_CAP_GIB must be a positive integer" >&2
  exit 2
fi
if ! [[ "$disk_floor_gib" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVOKV_DISK_FLOOR_GIB must be a positive integer" >&2
  exit 2
fi
if (( persistent_cap_gib > 500 )); then
  echo "persistent cap must not exceed 500 GiB for this round" >&2
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

if [[ "$preflight_only" == "1" ]]; then
  result_root="results/baseline_rounds/.preflight/${round_label}"
  log_root="logs/baseline_rounds/.preflight/${round_label}"
else
  result_root="results/baseline_rounds/${round_label}"
  log_root="logs/baseline_rounds/${round_label}"
fi
checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/rounds/${round_label}"
base_checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/seed0"
schedule="configs/evokv_foundation/xp_qk_multiversion_prequential3_development_v1.json"
benchmark_config="configs/evokv_baselines/x_qk_xp_multiversion_two_gpu_baseline_v1.json"
training_result="${result_root}/xp_multiversion_training.json"
ledger_root="${result_root}/xp_multiversion_ledgers"
audit_result="${result_root}/semantic_evidence_audit.json"
table_root="${result_root}/semantic_tables"
m2_result="${result_root}/m2_append_aware_lookup.json"
round_summary="${result_root}/round_summary.json"
lock_path="results/baseline_rounds/.${round_label}.lock"

mkdir -p "$(dirname "$lock_path")"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "another process owns round label: $round_label" >&2
  exit 3
fi

if [[ "$preflight_only" == "0" && -e "$result_root" && "$resume" == "0" ]]; then
  echo "result root exists; use a new label or EVOKV_RESUME=1: $result_root" >&2
  exit 3
fi
if [[ "$preflight_only" == "0" && -e "$checkpoint_root" && "$resume" == "0" ]]; then
  echo "checkpoint root exists; use a new label or EVOKV_RESUME=1: $checkpoint_root" >&2
  exit 3
fi

mkdir -p "$result_root" "$log_root"

bound_inputs=(
  "$schedule" \
  "$benchmark_config" \
  "data/processed/evokv_foundation/qk_xp_fixed_edge_inputs.npz" \
  "data/processed/evokv_foundation/x_qk_het_foundation.npz" \
  "${base_checkpoint_root}/theta_0/manifest.json" \
  "scripts/audit_evokv_d1_baseline_evidence.py" \
  "scripts/export_evokv_semantic_baseline_tables.py" \
  "scripts/train_evokv_xp_multiversion.py" \
  "src/hstu_kvcache/streaming/xp_multiversion.py" \
  "src/hstu_kvcache/streaming/xp_version_training.py" \
  "src/hstu_kvcache/streaming/xp_projected_edge.py" \
  "src/hstu_kvcache/streaming/sharded_edge.py" \
  "src/hstu_kvcache/streaming/trainer.py" \
  "src/hstu_kvcache/data/qk_xp_edge_inputs.py" \
  "scripts/evaluate_evokv_xp_d1_quality.py" \
  "src/hstu_kvcache/migration/xp_d1_quality.py" \
  "src/hstu_kvcache/migration/xp_exact_baseline.py" \
  "src/hstu_kvcache/migration/program.py" \
  "src/hstu_kvcache/migration/capsule.py" \
  "src/hstu_kvcache/migration/low_rank.py" \
  "src/hstu_kvcache/migration/layerwise.py" \
  "src/hstu_kvcache/migration/stage45_oldkv.py" \
  "src/hstu_kvcache/migration/artifacts.py" \
  "src/hstu_kvcache/migration/cohort_jagged.py" \
  "src/hstu_kvcache/migration/destination.py" \
  "src/hstu_kvcache/migration/design3_store.py" \
  "src/hstu_kvcache/migration/stage4_engine.py" \
  "src/hstu_kvcache/migration/stage4_source.py" \
  "src/hstu_kvcache/migration/stage45_reclaim.py" \
  "src/hstu_kvcache/migration/stage45_resident.py" \
  "scripts/benchmark_evokv_xp_m2_lookup_communication.py" \
  "src/hstu_kvcache/migration/xp_m2_lookup_baseline.py" \
  "src/hstu_kvcache/models/__init__.py" \
  "src/hstu_kvcache/models/attention.py" \
  "src/hstu_kvcache/models/block.py" \
  "src/hstu_kvcache/models/embeddings.py" \
  "src/hstu_kvcache/models/hstu.py" \
  "src/hstu_kvcache/models/kv_cache.py" \
  "src/hstu_kvcache/models/rmsnorm.py" \
  "scripts/summarize_evokv_d1_d2_baseline_round.py" \
  "scripts/run_evokv_d1_d2_baseline_round.sh"
)
for required in "${bound_inputs[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "required input is absent: $required" >&2
    exit 4
  fi
done

gpu_uuids=()
for device in "${device_list[@]}"; do
  if ! gpu_uuid="$(nvidia-smi -i "$device" --query-gpu=uuid --format=csv,noheader,nounits)"; then
    echo "GPU is unavailable: $device" >&2
    exit 4
  fi
  gpu_uuids+=("$gpu_uuid")
done
gpu_uuid_csv="${gpu_uuids[0]},${gpu_uuids[1]}"

busy_pids="$({
  nvidia-smi -i "$visible_devices" \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true
} | sed '/^[[:space:]]*$/d' | sort -u)"
if [[ -n "$busy_pids" ]]; then
  echo "selected GPUs already have compute processes: $busy_pids" >&2
  exit 4
fi

gib=$((1024 * 1024 * 1024))
disk_free_bytes="$(df -PB1 "$repo_root" | awk 'NR==2 {print $4}')"
memory_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
expected_new_gib=230
if [[ "$resume" == "1" && -f "$training_result" ]] && jq -e '
  .protocol == "evokv_xp_prequential_stream_training_development_v1"
  and .status == "complete"
  and .downstream_d1_d2_gate_passed == true
' "$training_result" >/dev/null; then
  expected_new_gib=10
fi
minimum_start_bytes=$(((expected_new_gib + disk_floor_gib) * gib))
if (( disk_free_bytes < minimum_start_bytes )); then
  echo "insufficient disk: need ${expected_new_gib} GiB expected assets plus ${disk_floor_gib} GiB floor" >&2
  exit 4
fi
if (( memory_available_kib < 64 * 1024 * 1024 )); then
  echo "at least 64 GiB available DRAM is required" >&2
  exit 4
fi

input_hashes="$result_root/input_hashes.tsv"
hash_candidate="$(mktemp)"
trap 'if [[ -n "${hash_candidate:-}" ]]; then rm -f "$hash_candidate"; fi' EXIT
sha256sum "${bound_inputs[@]}" > "$hash_candidate"
if [[ -f "$input_hashes" ]]; then
  if ! cmp -s "$hash_candidate" "$input_hashes"; then
    echo "round inputs or source code changed; use a new round label" >&2
    exit 5
  fi
else
  mv "$hash_candidate" "$input_hashes"
  hash_candidate=""
fi

preflight="$result_root/preflight.json"
preflight_latest="$result_root/preflight_latest.json"
python - "$preflight" "$preflight_latest" "$round_label" "$visible_devices" \
  "$gpu_uuid_csv" "$disk_free_bytes" "$memory_available_kib" \
  "$persistent_cap_gib" "$disk_floor_gib" <<'PY'
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
    "schema": "evokv_d1_d2_baseline_round_preflight_v0",
    "status": "pass",
    "round_label": sys.argv[3],
    "visible_devices": sys.argv[4],
    "gpu_uuids": sys.argv[5].split(","),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "disk_free_bytes_at_start": int(sys.argv[6]),
    "memory_available_kib_at_start": int(sys.argv[7]),
    "persistent_cap_gib": int(sys.argv[8]),
    "disk_floor_gib": int(sys.argv[9]),
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "git_dirty": bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
    ),
    "retention": {
        "keep": [
            "inference checkpoints",
            "version ledgers",
            "D1 programs and ActionPlans",
            "compact metrics, hashes, and logs",
        ],
        "discard": [
            "full K/V payloads",
            "fit/probe intermediate tensors",
            "candidate model copies",
            "reconstructible derived layouts",
        ],
    },
}
def write(target):
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)

if path.exists():
    initial = json.loads(path.read_text())
    for field in (
        "round_label",
        "visible_devices",
        "gpu_uuids",
        "python_version",
        "torch_version",
        "torch_cuda_version",
    ):
        if initial[field] != value[field]:
            raise RuntimeError(f"preflight hardware binding changed: {field}")
else:
    write(path)
write(latest)
PY

echo "preflight passed: GPUs=${visible_devices}, disk_free=$((disk_free_bytes / gib)) GiB, persistent_cap=${persistent_cap_gib} GiB"
if [[ "$preflight_only" == "1" ]]; then
  echo "preflight-only complete: $preflight"
  exit 0
fi

run_logged() {
  local stage="$1"
  shift
  local log="${log_root}/${stage}.log"
  echo "starting stage: $stage"
  "$@" 2>&1 | tee "$log"
  echo "completed stage: $stage"
}

path_bytes() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sb "$path" | awk '{print $1}'
  else
    echo 0
  fi
}

enforce_storage() {
  local stage="$1"
  local checkpoint_bytes
  local result_bytes
  local log_bytes
  local total_bytes
  local free_bytes
  checkpoint_bytes="$(path_bytes "$checkpoint_root")"
  result_bytes="$(path_bytes "$result_root")"
  log_bytes="$(path_bytes "$log_root")"
  total_bytes=$((checkpoint_bytes + result_bytes + log_bytes))
  free_bytes="$(df -PB1 "$repo_root" | awk 'NR==2 {print $4}')"
  if (( total_bytes > persistent_cap_gib * gib )); then
    echo "${stage}: durable round assets exceed ${persistent_cap_gib} GiB" >&2
    exit 6
  fi
  if (( free_bytes < disk_floor_gib * gib )); then
    echo "${stage}: disk safety floor of ${disk_floor_gib} GiB was crossed" >&2
    exit 6
  fi
  echo "storage after ${stage}: $((total_bytes / gib)) GiB durable, $((free_bytes / gib)) GiB free"
}

valid_audit() {
  [[ -f "$audit_result" ]] && jq -e '
    .schema == "evokv_d1_baseline_evidence_audit_v0"
    and .status == "pass"
    and .checkpoints.chains == 36
    and .checkpoints.files == 432
  ' "$audit_result" >/dev/null
}

valid_tables() {
  local manifest="${table_root}/manifest.json"
  [[ -f "$manifest" ]] && jq -e '
    .schema == "evokv_semantic_baseline_tables_v0"
    and .status == "pass"
    and .tables["m1_streaming_versions.tsv"].rows == 216
    and .tables["m1_cache_age.tsv"].rows == 396
    and .tables["d1_cost_quality.tsv"].rows == 36
    and .tables["d1_same_sla_structural.tsv"].rows == 36
    and all(.tables[]; has("bytes") and has("sha256"))
  ' "$manifest" >/dev/null || return 1
  local name
  local path
  local expected_bytes
  local expected_sha
  local observed_sha
  for name in \
    m1_streaming_versions.tsv \
    m1_cache_age.tsv \
    d1_cost_quality.tsv \
    d1_same_sla_structural.tsv; do
    path="${table_root}/${name}"
    [[ -f "$path" ]] || return 1
    expected_bytes="$(jq -r --arg name "$name" '.tables[$name].bytes' "$manifest")"
    expected_sha="$(jq -r --arg name "$name" '.tables[$name].sha256' "$manifest")"
    [[ "$(stat -c %s "$path")" == "$expected_bytes" ]] || return 1
    observed_sha="$(sha256sum "$path" | awk '{print $1}')"
    [[ "$observed_sha" == "$expected_sha" ]] || return 1
  done
}

valid_training() {
  [[ -f "$training_result" ]] || return 1
  jq -e '
    .protocol == "evokv_xp_prequential_stream_training_development_v1"
    and .status == "complete"
    and .downstream_d1_d2_gate_passed == true
    and .checkpoint_admission.ranking_metrics_used == false
    and (.updates | length) == 3
    and (.prequential_evaluations | length) == 4
    and all(.updates[]; .target_checkpoint_committed == true)
  ' "$training_result" >/dev/null || return 1
  for version in 1 2 3; do
    [[ -f "${checkpoint_root}/theta_${version}/manifest.json" ]] || return 1
    [[ -f "${ledger_root}/version_$(printf '%05d' "$version").json" ]] || return 1
  done
}

valid_d1_edge() {
  local edge_output="$1"
  local program_output="$2"
  local action_plan_output="$3"
  local source_version="$4"
  local target_version="$5"
  local benchmark_sha
  local program_sha
  local action_sha
  local training_history_end=$((64 + source_version * 8))
  local training_update_end=$((72 + source_version * 8))
  local history_end=$((72 + source_version * 8))
  local update_end=$((80 + source_version * 8))
  benchmark_sha="$(sha256sum "$benchmark_config" | awk '{print $1}')"
  [[ -f "$edge_output" && -f "$program_output" && -f "$action_plan_output" ]] || return 1
  jq -e \
    --argjson source "$source_version" \
    --argjson target "$target_version" \
    --argjson history_end "$history_end" \
    --argjson update_end "$update_end" \
    --argjson training_history_end "$training_history_end" \
    --argjson training_update_end "$training_update_end" \
    --arg benchmark_sha "$benchmark_sha" \
    --arg program_path "$program_output" \
    --arg action_path "$action_plan_output" '
      .protocol == "evokv_xp_d1_quality_development_v0"
      and .status == "complete"
      and .edge.source_version == $source
      and .edge.target_version == $target
      and .edge.history_end == $history_end
      and .edge.update_end == $update_end
      and .edge.training_window.history_end == $training_history_end
      and .edge.training_window.update_end == $training_update_end
      and .edge.evaluation_window.history_end == $history_end
      and .edge.evaluation_window.evaluation_end == $update_end
      and .roles.probe.audit.history_end == $history_end
      and .roles.probe.audit.update_end == $update_end
      and .roles.qualification_test.audit.history_end == $history_end
      and .roles.qualification_test.audit.update_end == $update_end
      and .bindings.benchmark_config.sha256 == $benchmark_sha
      and .bindings.program.path == $program_path
      and .bindings.action_plan.path == $action_path
    ' "$edge_output" >/dev/null || return 1
  program_sha="$(sha256sum "$program_output" | awk '{print $1}')"
  action_sha="$(sha256sum "$action_plan_output" | awk '{print $1}')"
  [[ "$program_sha" == "$(jq -r '.bindings.program.sha256' "$edge_output")" ]] || return 1
  [[ "$action_sha" == "$(jq -r '.bindings.action_plan.sha256' "$edge_output")" ]] || return 1
  jq -e \
    --argjson source "$source_version" \
    --argjson target "$target_version" \
    --arg program_sha "$program_sha" \
    --arg records_sha "$(jq -r '.bindings.action_plan.records_sha256' "$edge_output")" '
      .protocol == "evokv_xp_d1_action_plan_v2_development_v0"
      and .source_version == $source
      and .target_version == $target
      and .extent_contract.append_tokens == 32
      and .selection.budget_basis == "retained_tokens"
      and .bindings.program_sha256 == $program_sha
      and .records_sha256 == $records_sha
    ' "$action_plan_output" >/dev/null
}

valid_m2() {
  local benchmark_sha
  local workload_sha
  benchmark_sha="$(sha256sum "$benchmark_config" | awk '{print $1}')"
  workload_sha="$(sha256sum data/processed/evokv_foundation/x_qk_het_foundation.npz | awk '{print $1}')"
  [[ -f "$m2_result" ]] && jq -e \
    --arg benchmark_sha "$benchmark_sha" \
    --arg workload_sha "$workload_sha" '
      .protocol == "evokv_xp_m2_append_aware_lookup_development_v0"
      and .world_size == 2
      and .append_tokens_per_record == 32
      and .checkpoint.version == 1
      and .bindings.benchmark_config.sha256 == $benchmark_sha
      and .bindings.het_workload.sha256 == $workload_sha
      and [.cells[].retained_budget.fraction_requested] == [0, 0.2, 0.5, 1]
    ' "$m2_result" >/dev/null
}

valid_summary() {
  [[ -f "$round_summary" ]] && jq -e \
    --arg label "$round_label" '
      .schema == "evokv_d1_d2_baseline_round_summary_v0"
      and .status == "complete"
      and .round_label == $label
      and .retention.full_kv_payloads_retained == 0
      and .retention.durable_layout.allowlist_validation == "pass"
    ' "$round_summary" >/dev/null
}

cleanup_d1_edge() {
  local edge_name="$1"
  local edge_output="$2"
  local program_output="$3"
  local action_plan_output="$4"
  rm -f -- "$edge_output" "$program_output" "$action_plan_output"
  if [[ -d "${result_root}/d1" ]]; then
    while IFS= read -r temporary; do
      rm -f -- "$temporary"
      echo "removed reconstructible D1 temporary: $temporary"
    done < <(find "${result_root}/d1" -maxdepth 1 -type f -name ".*${edge_name}*.tmp")
  fi
}

if ! valid_audit; then
  run_logged semantic_evidence_audit \
    python scripts/audit_evokv_d1_baseline_evidence.py \
      --output "$audit_result"
  valid_audit
else
  echo "resume: semantic evidence audit already validates"
fi
enforce_storage semantic_evidence_audit

if ! valid_tables; then
  run_logged semantic_table_export \
    python scripts/export_evokv_semantic_baseline_tables.py \
      --output-dir "$table_root"
  valid_tables
else
  echo "resume: semantic baseline tables already validate"
fi
enforce_storage semantic_table_export

if ! valid_training; then
  if [[ -e "$training_result" || -e "$checkpoint_root" || -e "$ledger_root" ]]; then
    echo "incomplete or failed multiversion state is preserved; use a new round label after inspection" >&2
    exit 20
  fi
  if ! run_logged xp_multiversion_training \
    env CUDA_VISIBLE_DEVICES="$visible_devices" OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc-per-node=2 \
      scripts/train_evokv_xp_multiversion.py \
      --device cuda \
      --schedule "$schedule" \
      --base-checkpoint-root "$base_checkpoint_root" \
      --checkpoint-root "$checkpoint_root" \
      --output "$training_result" \
      --ledger-dir "$ledger_root" \
      --batch-size-per-rank 1 \
      --progress-every 100; then
    if [[ -d "$checkpoint_root" ]]; then
      while IFS= read -r temporary; do
        rm -f -- "$temporary"
        echo "removed reconstructible checkpoint temporary: $temporary"
      done < <(find "$checkpoint_root" -type f -name ".*.tmp")
    fi
    rm -f -- "${training_result}.tmp"
    echo "XP multiversion training stopped; committed versions were preserved" >&2
    exit 20
  fi
  if ! valid_training; then
    echo "XP multiversion gate did not pass; downstream D1/D2 stopped" >&2
    exit 21
  fi
else
  echo "resume: XP multiversion training already validates"
fi
enforce_storage xp_multiversion_training

for edge_index in 0 1 2; do
  source_version="$edge_index"
  target_version=$((edge_index + 1))
  training_history_end=$((64 + edge_index * 8))
  training_update_end=$((72 + edge_index * 8))
  history_end=$((72 + edge_index * 8))
  edge_name="theta${source_version}_to_theta${target_version}"
  edge_output="${result_root}/d1/${edge_name}.json"
  program_output="${result_root}/d1/${edge_name}_direct_oldkv_fp16.pt"
  action_plan_output="${result_root}/d1/${edge_name}_action_plan_v2.json"
  if (( source_version == 0 )); then
    source_root="$base_checkpoint_root"
  else
    source_root="$checkpoint_root"
  fi
  if valid_d1_edge \
    "$edge_output" \
    "$program_output" \
    "$action_plan_output" \
    "$source_version" \
    "$target_version"; then
    echo "resume: D1 edge already validates: $edge_name"
    enforce_storage "d1_${edge_name}"
    continue
  fi
  if [[ -e "$edge_output" || -e "$program_output" || -e "$action_plan_output" ]]; then
    cleanup_d1_edge "$edge_name" "$edge_output" "$program_output" "$action_plan_output"
    echo "removed reconstructible partial D1 artifacts: $edge_name"
  fi
  mkdir -p "${result_root}/d1"
  cleanup_d1_edge "$edge_name" "$edge_output" "$program_output" "$action_plan_output"
  if ! run_logged "d1_${edge_name}" \
    env CUDA_VISIBLE_DEVICES="$visible_devices" OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc-per-node=2 \
      scripts/evaluate_evokv_xp_d1_quality.py \
      --config "$benchmark_config" \
      --source-checkpoint-root "$source_root" \
      --target-checkpoint-root "$checkpoint_root" \
      --source-version "$source_version" \
      --target-version "$target_version" \
      --history-end "$history_end" \
      --update-end $((history_end + 8)) \
      --training-history-end "$training_history_end" \
      --training-update-end "$training_update_end" \
      --probe-role theta12 \
      --qualification-role qualification \
      --capacity 288 \
      --batch-size-per-rank 4 \
      --fit-batches 4 \
      --probe-batches 4 \
      --qualification-batches 0 \
      --negative-count 99 \
      --candidate-seed $((20260801 + edge_index * 1000033)) \
      --timing-repeats 3 \
      --output "$edge_output" \
      --program-output "$program_output" \
      --action-plan-output "$action_plan_output"; then
    cleanup_d1_edge "$edge_name" "$edge_output" "$program_output" "$action_plan_output"
    echo "D1 edge failed; reconstructible partial artifacts were removed: $edge_name" >&2
    exit 23
  fi
  if ! valid_d1_edge \
    "$edge_output" \
    "$program_output" \
    "$action_plan_output" \
    "$source_version" \
    "$target_version"; then
    cleanup_d1_edge "$edge_name" "$edge_output" "$program_output" "$action_plan_output"
    echo "D1 edge validation failed; reconstructible artifacts were removed: $edge_name" >&2
    exit 23
  fi
  enforce_storage "d1_${edge_name}"
done

if valid_m2; then
  echo "resume: append-aware M2 lookup baseline already validates"
else
  if [[ -e "$m2_result" ]]; then
    rm -f -- "$m2_result"
    echo "removed reconstructible invalid M2 result"
  fi
  rm -f -- "${m2_result}.tmp"
  if ! run_logged m2_append_aware_lookup \
    env CUDA_VISIBLE_DEVICES="$visible_devices" OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc-per-node=2 \
      scripts/benchmark_evokv_xp_m2_lookup_communication.py \
      --config "$benchmark_config" \
      --checkpoint-root "$checkpoint_root" \
      --checkpoint-version 1 \
      --fractions 0,0.2,0.5,1.0 \
      --micro-batch-records 8 \
      --warmup 1 \
      --repeats 5 \
      --output "$m2_result"; then
    rm -f -- "$m2_result" "${m2_result}.tmp"
    echo "M2 lookup characterization failed; reconstructible result was removed" >&2
    exit 24
  fi
  if ! valid_m2; then
    rm -f -- "$m2_result" "${m2_result}.tmp"
    echo "M2 lookup validation failed; reconstructible result was removed" >&2
    exit 24
  fi
fi
enforce_storage m2_append_aware_lookup

if valid_summary; then
  echo "resume: round summary already validates"
else
  if [[ -e "$round_summary" ]]; then
    rm -f -- "$round_summary"
    echo "removed reconstructible invalid round summary"
  fi
  rm -f -- "${round_summary}.tmp"
  python scripts/summarize_evokv_d1_d2_baseline_round.py \
    --round-label "$round_label" \
    --result-root "$result_root" \
    --checkpoint-root "$checkpoint_root" \
    --log-root "$log_root" \
    --output "$round_summary"
  valid_summary
fi
enforce_storage round_summary

echo "round complete: $round_summary"
echo "full K/V payloads retained: 0"
