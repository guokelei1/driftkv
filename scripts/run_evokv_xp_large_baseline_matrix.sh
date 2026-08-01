#!/usr/bin/env bash
set -euo pipefail

xp_capacity="${1:-144}"
xp_repeats="${2:-1}"
xp_methods="${3:-s0,s1}"
xp_label="${4:-manual}"
xp_visible_devices="${EVOKV_CUDA_VISIBLE_DEVICES:-0,1}"
xp_keep_store="${EVOKV_KEEP_STORE:-0}"
xp_restart_failed="${EVOKV_RESTART_FAILED:-0}"
xp_preflight_only="${EVOKV_PREFLIGHT_ONLY:-0}"
xp_source_version="${EVOKV_SOURCE_VERSION:-0}"
xp_target_version="${EVOKV_TARGET_VERSION:-1}"
xp_group_gib="${EVOKV_GROUP_GIB:-4}"

if [[ "$xp_capacity" != "144" && "$xp_capacity" != "288" ]]; then
  echo "capacity must be 144 or 288" >&2
  exit 2
fi
if ! [[ "$xp_repeats" =~ ^[1-9][0-9]*$ ]]; then
  echo "repeats must be a positive integer" >&2
  exit 2
fi
if ! [[ "$xp_label" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "label must contain only letters, digits, underscore, or dash" >&2
  exit 2
fi
if ! [[ "$xp_visible_devices" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "EVOKV_CUDA_VISIBLE_DEVICES must name exactly two GPUs" >&2
  exit 2
fi
if [[ "$xp_keep_store" != "0" && "$xp_keep_store" != "1" ]]; then
  echo "EVOKV_KEEP_STORE must be 0 or 1" >&2
  exit 2
fi
if [[ "$xp_restart_failed" != "0" && "$xp_restart_failed" != "1" ]]; then
  echo "EVOKV_RESTART_FAILED must be 0 or 1" >&2
  exit 2
fi
if [[ "$xp_preflight_only" != "0" && "$xp_preflight_only" != "1" ]]; then
  echo "EVOKV_PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$xp_source_version" =~ ^[0-9]+$ && "$xp_target_version" =~ ^[0-9]+$ ]]; then
  echo "checkpoint versions must be nonnegative integers" >&2
  exit 2
fi
if [[ "$xp_source_version" == "$xp_target_version" ]]; then
  echo "source and target checkpoint versions must differ" >&2
  exit 2
fi
if ! [[ "$xp_group_gib" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVOKV_GROUP_GIB must be a positive integer" >&2
  exit 2
fi

IFS=',' read -r -a xp_device_list <<< "$xp_visible_devices"
if [[ "${xp_device_list[0]}" == "${xp_device_list[1]}" ]]; then
  echo "EVOKV_CUDA_VISIBLE_DEVICES must name two distinct GPUs" >&2
  exit 2
fi

IFS=',' read -r -a xp_method_list <<< "$xp_methods"
if (( ${#xp_method_list[@]} == 0 )) || [[ -z "$xp_methods" || "$xp_methods" == *, || "$xp_methods" == ,* ]]; then
  echo "methods must be a nonempty comma-separated subset of s0,s1" >&2
  exit 2
fi
xp_seen_s0=0
xp_seen_s1=0
for xp_method in "${xp_method_list[@]}"; do
  case "$xp_method" in
    s0)
      if (( xp_seen_s0 == 1 )); then
        echo "methods must not contain duplicates" >&2
        exit 2
      fi
      xp_seen_s0=1
      ;;
    s1)
      if (( xp_seen_s1 == 1 )); then
        echo "methods must not contain duplicates" >&2
        exit 2
      fi
      xp_seen_s1=1
      ;;
    *)
      echo "methods must be a comma-separated subset of s0,s1" >&2
      exit 2
      ;;
  esac
done
if [[ "$xp_keep_store" == "1" ]] && (( ${#xp_method_list[@]} * xp_repeats > 1 )); then
  echo "EVOKV_KEEP_STORE=1 supports only one method and one repeat per invocation" >&2
  exit 2
fi

for xp_command in jq sha256sum nvidia-smi torchrun tee; do
  if ! command -v "$xp_command" >/dev/null 2>&1; then
    echo "required command is unavailable: $xp_command" >&2
    exit 2
  fi
done

xp_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$xp_repo_root"

xp_config="configs/evokv_baselines/x_qk_xp_two_gpu_baseline_v0.json"
xp_checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/seed0"
xp_output_root="results/system/evokv_xp_baselines/manual_${xp_label}"
xp_store_root="/dev/shm/evokv_xp_baselines"
xp_summary_tsv="${xp_output_root}/summary.tsv"
mkdir -p "$xp_output_root" "$xp_store_root"

xp_benchmark_id="$(jq -r '.benchmark_id' "$xp_config")"
xp_config_sha256="$(sha256sum "$xp_config" | awk '{print $1}')"
xp_target_bytes="$(
  jq -r --arg capacity "$xp_capacity" \
    '.capacity_points.out_of_core_primary[]
     | select((.single_version_target_gib_nominal | tostring) == $capacity)
     | .target_valid_bytes' \
    "$xp_config"
)"
xp_expected_records="$(
  jq -r --arg capacity "$xp_capacity" \
    '.capacity_points.out_of_core_primary[]
     | select((.single_version_target_gib_nominal | tostring) == $capacity)
     | .prefix_records' \
    "$xp_config"
)"
if ! [[ "$xp_target_bytes" =~ ^[0-9]+$ && "$xp_expected_records" =~ ^[0-9]+$ ]]; then
  echo "capacity definition is absent from the frozen config" >&2
  exit 2
fi

xp_group_bytes=$((xp_group_gib * 1024 * 1024 * 1024))
xp_source_checkpoint_manifest="${xp_checkpoint_root}/theta_${xp_source_version}/manifest.json"
xp_target_checkpoint_manifest="${xp_checkpoint_root}/theta_${xp_target_version}/manifest.json"
if [[ ! -f "$xp_source_checkpoint_manifest" || ! -f "$xp_target_checkpoint_manifest" ]]; then
  echo "checkpoint manifest is absent" >&2
  exit 2
fi
xp_source_checkpoint_sha256="$(sha256sum "$xp_source_checkpoint_manifest" | awk '{print $1}')"
xp_target_checkpoint_sha256="$(sha256sum "$xp_target_checkpoint_manifest" | awk '{print $1}')"

for xp_device in "${xp_device_list[@]}"; do
  if ! nvidia-smi -i "$xp_device" --query-gpu=uuid --format=csv,noheader,nounits >/dev/null 2>&1; then
    echo "GPU is unavailable: $xp_device" >&2
    exit 4
  fi
done
if ! xp_busy="$(
  nvidia-smi -i "$xp_visible_devices" \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null
)"; then
  echo "failed to query selected GPU processes" >&2
  exit 4
fi
xp_busy="$(printf '%s\n' "$xp_busy" | sed '/^[[:space:]]*$/d')"
if [[ -n "$xp_busy" ]]; then
  echo "selected GPUs already have compute processes: $xp_busy" >&2
  exit 4
fi

xp_validate_target() {
  local xp_result="$1"
  local xp_expected_method="$2"
  jq -e \
    --arg benchmark_id "$xp_benchmark_id" \
    --arg capacity "$xp_capacity" \
    --arg method "$xp_expected_method" \
    --arg config_sha256 "$xp_config_sha256" \
    --arg checkpoint_sha256 "$xp_target_checkpoint_sha256" \
    --argjson checkpoint_version "$xp_target_version" \
    --argjson target_bytes "$xp_target_bytes" \
    --argjson expected_records "$xp_expected_records" \
    --argjson group_bytes "$xp_group_bytes" \
    '
      . as $root
      | .benchmark_id == $benchmark_id
        and .capacity_name == $capacity
        and .method == $method
        and .endpoint == "target"
        and .world_size == 2
        and .group_target_bytes == $group_bytes
        and .checkpoint.version == $checkpoint_version
        and .checkpoint.sha256 == $checkpoint_sha256
        and .bindings.benchmark_config.sha256 == $config_sha256
        and .records == $expected_records
        and .records_expected == $expected_records
        and .written_bytes == $target_bytes
        and .d2h_bytes == $target_bytes
        and .capacity.global_live_store_payload_bytes == $target_bytes
        and .transaction.store_mode == "open"
        and (.transaction.source_manifest_sha256 | type == "string" and length == 64)
        and (.record_ids_sha256 | type == "string" and length == 64)
        and (.group_plan_sha256 | type == "string" and length == 64)
        and (.output_hash.sha256 | type == "string" and length == 64)
        and (.rank_reports | length) == 2
        and ([.rank_reports[].rank] | sort) == [0, 1]
        and ([.rank_reports[].records] | add) == .records
        and ([.rank_reports[].written_bytes] | add) == .written_bytes
        and ([.rank_reports[].d2h_bytes] | add) == .d2h_bytes
        and all(
          .rank_reports[];
          .written_bytes == .d2h_bytes
          and .rolling_journal.groups_committed == $root.groups
          and .rolling_journal.records_committed == .records
          and (.rolling_journal.sha256 | type == "string" and length == 64)
        )
    ' \
    "$xp_result" >/dev/null || return 1
}

xp_validate_source() {
  local xp_result="$1"
  local xp_rank0_store="$2"
  local xp_rank1_store="$3"
  local xp_rank0_ledger="$4"
  local xp_rank1_ledger="$5"
  jq -e \
    --arg benchmark_id "$xp_benchmark_id" \
    --arg capacity "$xp_capacity" \
    --arg config_sha256 "$xp_config_sha256" \
    --arg checkpoint_sha256 "$xp_source_checkpoint_sha256" \
    --argjson checkpoint_version "$xp_source_version" \
    --argjson expected_records "$xp_expected_records" \
    --argjson group_bytes "$xp_group_bytes" \
    '
      .benchmark_id == $benchmark_id
      and .capacity_name == $capacity
      and .method == "s0"
      and .endpoint == "old"
      and .world_size == 2
      and .group_target_bytes == $group_bytes
      and .checkpoint.version == $checkpoint_version
      and .checkpoint.sha256 == $checkpoint_sha256
      and .bindings.benchmark_config.sha256 == $config_sha256
      and .records == $expected_records
      and .records_expected == $expected_records
      and .transaction.store_mode == "create"
      and (.rank_reports | length) == 2
      and ([.rank_reports[].rank] | sort) == [0, 1]
      and ([.rank_reports[].records] | add) == .records
      and ([.rank_reports[].written_bytes] | add) == .written_bytes
      and ([.rank_reports[].d2h_bytes] | add) == .d2h_bytes
    ' \
    "$xp_result" >/dev/null || return 1
  local xp_expected_rank0_size
  local xp_expected_rank1_size
  xp_expected_rank0_size="$(jq -r '.rank_reports[] | select(.rank == 0) | .store.mapped_nbytes' "$xp_result")"
  xp_expected_rank1_size="$(jq -r '.rank_reports[] | select(.rank == 1) | .store.mapped_nbytes' "$xp_result")"
  [[ "$(stat -c %s "$xp_rank0_store")" == "$xp_expected_rank0_size" ]] || return 1
  [[ "$(stat -c %s "$xp_rank1_store")" == "$xp_expected_rank1_size" ]] || return 1
  jq -e \
    --arg benchmark_id "$xp_benchmark_id" \
    --arg capacity "$xp_capacity" \
    --arg config_sha256 "$xp_config_sha256" \
    '.phase == "source_complete"
     and .binding.benchmark_id == $benchmark_id
     and .binding.capacity_name == $capacity
     and .binding.rank == 0
     and .binding.world_size == 2
     and .binding.bindings.benchmark_config.sha256 == $config_sha256' \
    "$xp_rank0_ledger" >/dev/null || return 1
  jq -e \
    --arg benchmark_id "$xp_benchmark_id" \
    --arg capacity "$xp_capacity" \
    --arg config_sha256 "$xp_config_sha256" \
    '.phase == "source_complete"
     and .binding.benchmark_id == $benchmark_id
     and .binding.capacity_name == $capacity
     and .binding.rank == 1
     and .binding.world_size == 2
     and .binding.bindings.benchmark_config.sha256 == $config_sha256' \
    "$xp_rank1_ledger" >/dev/null || return 1
}

xp_archive_state() {
  local xp_run_name="$1"
  local xp_source_result="$2"
  local xp_target_result="$3"
  local xp_source_log="$4"
  local xp_target_log="$5"
  local xp_rank0_ledger="$6"
  local xp_rank1_ledger="$7"
  local xp_archive_root="${xp_output_root}/failed_${xp_run_name}_$(date +%Y%m%dT%H%M%S)"
  mkdir -p "$xp_archive_root"
  for xp_artifact in \
    "$xp_source_result" \
    "$xp_target_result" \
    "$xp_source_log" \
    "$xp_target_log" \
    "$xp_rank0_ledger" \
    "$xp_rank1_ledger"; do
    if [[ -f "$xp_artifact" ]]; then
      cp -- "$xp_artifact" "$xp_archive_root/"
    fi
  done
  echo "archived failed state: $xp_archive_root"
}

xp_cleanup_store() {
  local xp_store_prefix="$1"
  local xp_run_name="$2"
  local xp_rank0_ledger="${xp_store_prefix}.rank00.dram.ledger.json"
  local xp_rank1_ledger="${xp_store_prefix}.rank01.dram.ledger.json"
  case "$xp_store_prefix" in
    /dev/shm/evokv_xp_baselines/xp144_*|/dev/shm/evokv_xp_baselines/xp288_*) ;;
    *)
      echo "refusing unexpected store path: $xp_store_prefix" >&2
      exit 5
      ;;
  esac
  if [[ -f "$xp_rank0_ledger" ]]; then
    cp -- "$xp_rank0_ledger" "${xp_output_root}/${xp_run_name}_rank00_final_ledger.json"
  fi
  if [[ -f "$xp_rank1_ledger" ]]; then
    cp -- "$xp_rank1_ledger" "${xp_output_root}/${xp_run_name}_rank01_final_ledger.json"
  fi
  rm -f \
    "${xp_store_prefix}.rank00.dram" \
    "$xp_rank0_ledger" \
    "${xp_store_prefix}.rank01.dram" \
    "$xp_rank1_ledger"
}

xp_require_resources() {
  local xp_source_ready="$1"
  local xp_shm_required
  local xp_memory_required
  local xp_runtime_headroom=$((64 * 1024 * 1024 * 1024))
  if (( xp_source_ready == 1 )); then
    xp_shm_required=$((1 * 1024 * 1024 * 1024))
    xp_memory_required=$xp_runtime_headroom
  else
    xp_shm_required=$((xp_target_bytes + 16 * 1024 * 1024 * 1024))
    xp_memory_required=$((xp_target_bytes + xp_runtime_headroom))
  fi
  local xp_shm_available
  local xp_memory_available
  xp_shm_available="$(df --output=avail -B1 "$xp_store_root" | tail -n 1 | tr -d ' ')"
  xp_memory_available="$(awk '/MemAvailable/ {printf "%.0f", $2 * 1024}' /proc/meminfo)"
  if (( xp_shm_available < xp_shm_required )); then
    echo "insufficient /dev/shm: need $xp_shm_required available bytes, have $xp_shm_available" >&2
    exit 3
  fi
  if (( xp_memory_available < xp_memory_required )); then
    echo "insufficient host memory: need $xp_memory_required available bytes, have $xp_memory_available" >&2
    exit 3
  fi
}

xp_validate_pair() {
  local xp_s0_result="$1"
  local xp_s1_result="$2"
  jq -e -n \
    --slurpfile s0 "$xp_s0_result" \
    --slurpfile s1 "$xp_s1_result" \
    '
      ($s0[0]) as $a
      | ($s1[0]) as $b
      | $a.method == "s0"
        and $b.method == "s1"
        and $a.benchmark_id == $b.benchmark_id
        and $a.capacity_name == $b.capacity_name
        and $a.world_size == $b.world_size
        and $a.group_target_bytes == $b.group_target_bytes
        and $a.records == $b.records
        and $a.groups == $b.groups
        and $a.bindings == $b.bindings
        and $a.checkpoint.version == $b.checkpoint.version
        and $a.checkpoint.sha256 == $b.checkpoint.sha256
        and $a.record_ids_sha256 == $b.record_ids_sha256
        and $a.group_plan_sha256 == $b.group_plan_sha256
        and $a.written_bytes == $b.written_bytes
        and $a.output_hash.sha256 == $b.output_hash.sha256
    ' >/dev/null
}

if [[ "$xp_capacity" == "144" ]]; then
  xp_estimate="about 10-12 minutes for s0,s1 once"
else
  xp_estimate="about 20-25 minutes for s0,s1 once"
fi
echo "capacity=${xp_capacity}GiB repeats=$xp_repeats methods=$xp_methods GPUs=$xp_visible_devices"
echo "estimated wall time: $xp_estimate"
echo "results: $xp_output_root"

if [[ "$xp_preflight_only" == "1" ]]; then
  xp_require_resources 0
  for xp_method in "${xp_method_list[@]}"; do
    for ((xp_repeat = 1; xp_repeat <= xp_repeats; xp_repeat++)); do
      echo "planned: materialize source -> run $xp_method target -> validate -> preserve ledger -> release transient store, repeat=$xp_repeat"
    done
  done
  echo "preflight passed"
  exit 0
fi

for xp_method in "${xp_method_list[@]}"; do
  for ((xp_repeat = 1; xp_repeat <= xp_repeats; xp_repeat++)); do
    xp_repeat_tag="$(printf "%02d" "$xp_repeat")"
    xp_run_name="xp${xp_capacity}_${xp_label}_${xp_method}_r${xp_repeat_tag}"
    xp_store_prefix="${xp_store_root}/${xp_run_name}"
    xp_source_result="${xp_output_root}/${xp_run_name}_theta${xp_source_version}_source.json"
    xp_target_result="${xp_output_root}/${xp_run_name}_theta${xp_target_version}_${xp_method}.json"
    xp_source_log="${xp_output_root}/${xp_run_name}_theta${xp_source_version}_source.log"
    xp_target_log="${xp_output_root}/${xp_run_name}_theta${xp_target_version}_${xp_method}.log"
    xp_rank0_store="${xp_store_prefix}.rank00.dram"
    xp_rank1_store="${xp_store_prefix}.rank01.dram"
    xp_rank0_ledger="${xp_rank0_store}.ledger.json"
    xp_rank1_ledger="${xp_rank1_store}.ledger.json"

    if [[ -f "$xp_target_result" ]]; then
      if xp_validate_target "$xp_target_result" "$xp_method"; then
        if [[ "$xp_keep_store" == "0" ]] && {
          [[ -e "$xp_rank0_store" ]] || [[ -e "$xp_rank1_store" ]] ||
          [[ -e "$xp_rank0_ledger" ]] || [[ -e "$xp_rank1_ledger" ]];
        }; then
          xp_cleanup_store "$xp_store_prefix" "$xp_run_name"
          echo "removed completed transient store: $xp_store_prefix"
        elif [[ "$xp_keep_store" == "1" ]] && {
          [[ ! -f "$xp_rank0_store" ]] || [[ ! -f "$xp_rank1_store" ]] ||
          [[ ! -f "$xp_rank0_ledger" ]] || [[ ! -f "$xp_rank1_ledger" ]];
        }; then
          echo "valid result exists but retained store is incomplete: $xp_run_name" >&2
          exit 6
        fi
        echo "skip complete result: $xp_target_result"
        continue
      fi
      if [[ "$xp_restart_failed" == "0" ]]; then
        echo "invalid or incomplete target result exists; inspect it or rerun with EVOKV_RESTART_FAILED=1" >&2
        exit 6
      fi
    fi

    xp_source_ready=0
    xp_all_source_artifacts=0
    if [[ -f "$xp_source_result" && -f "$xp_rank0_store" && -f "$xp_rank1_store" && -f "$xp_rank0_ledger" && -f "$xp_rank1_ledger" ]]; then
      xp_all_source_artifacts=1
      if xp_validate_source \
        "$xp_source_result" \
        "$xp_rank0_store" \
        "$xp_rank1_store" \
        "$xp_rank0_ledger" \
        "$xp_rank1_ledger"; then
        xp_source_ready=1
      fi
    fi

    if (( xp_source_ready == 0 )) && {
      [[ -e "$xp_source_result" ]] || [[ -e "$xp_target_result" ]] ||
      [[ -e "$xp_rank0_store" ]] || [[ -e "$xp_rank1_store" ]] ||
      [[ -e "$xp_rank0_ledger" ]] || [[ -e "$xp_rank1_ledger" ]];
    }; then
      if [[ "$xp_restart_failed" == "0" ]]; then
        if (( xp_all_source_artifacts == 1 )); then
          echo "source artifacts exist but fail binding or phase validation" >&2
        else
          echo "partial or target-mutated artifacts exist for $xp_run_name" >&2
        fi
        echo "inspect them or rerun with EVOKV_RESTART_FAILED=1 to archive and restart this exact run" >&2
        exit 6
      fi
      xp_archive_state \
        "$xp_run_name" \
        "$xp_source_result" \
        "$xp_target_result" \
        "$xp_source_log" \
        "$xp_target_log" \
        "$xp_rank0_ledger" \
        "$xp_rank1_ledger"
      xp_cleanup_store "$xp_store_prefix" "$xp_run_name"
      rm -f "$xp_source_result" "$xp_target_result"
      xp_source_ready=0
    fi

    xp_require_resources "$xp_source_ready"

    if (( xp_source_ready == 0 )); then
      echo "materialize source: capacity=$xp_capacity method=$xp_method repeat=$xp_repeat"
      CUDA_VISIBLE_DEVICES="$xp_visible_devices" \
        torchrun --standalone --nproc-per-node=2 \
        scripts/benchmark_evokv_xp_exact_baselines.py \
        --config "$xp_config" \
        --checkpoint-root "$xp_checkpoint_root" \
        --checkpoint-version "$xp_source_version" \
        --capacity "$xp_capacity" \
        --method s0 \
        --endpoint old \
        --group-target-gib "$xp_group_gib" \
        --micro-batch-records 1 \
        --hash-mode sampled \
        --store-path "$xp_store_prefix" \
        --store-mode create \
        --output "$xp_source_result" \
        --quiet 2>&1 | tee "$xp_source_log"
      xp_validate_source \
        "$xp_source_result" \
        "$xp_rank0_store" \
        "$xp_rank1_store" \
        "$xp_rank0_ledger" \
        "$xp_rank1_ledger"
    else
      echo "resume from complete source: $xp_source_result"
    fi

    echo "run target: capacity=$xp_capacity method=$xp_method repeat=$xp_repeat"
    CUDA_VISIBLE_DEVICES="$xp_visible_devices" \
      torchrun --standalone --nproc-per-node=2 \
      scripts/benchmark_evokv_xp_exact_baselines.py \
      --config "$xp_config" \
      --checkpoint-root "$xp_checkpoint_root" \
      --checkpoint-version "$xp_target_version" \
      --capacity "$xp_capacity" \
      --method "$xp_method" \
      --endpoint target \
      --group-target-gib "$xp_group_gib" \
      --micro-batch-records 1 \
      --hash-mode sampled \
      --store-path "$xp_store_prefix" \
      --store-mode open \
      --source-manifest "$xp_source_result" \
      --output "$xp_target_result" \
      --quiet 2>&1 | tee "$xp_target_log"

    xp_validate_target "$xp_target_result" "$xp_method"
    jq -r \
      '[.capacity_name,.method,.records,.groups,.max_rank_wall_seconds,
        .lookup.requested_tokens,.lookup.remote_tokens,.d2h_bytes,
        .written_bytes,.peak_hbm_allocated_bytes_max_rank,
        .output_hash.sha256] | @tsv' \
      "$xp_target_result"

    if [[ "$xp_keep_store" == "0" ]]; then
      xp_cleanup_store "$xp_store_prefix" "$xp_run_name"
      echo "removed completed transient store: $xp_store_prefix"
    fi
  done
done

for ((xp_repeat = 1; xp_repeat <= xp_repeats; xp_repeat++)); do
  xp_repeat_tag="$(printf "%02d" "$xp_repeat")"
  xp_s0_result="${xp_output_root}/xp${xp_capacity}_${xp_label}_s0_r${xp_repeat_tag}_theta${xp_target_version}_s0.json"
  xp_s1_result="${xp_output_root}/xp${xp_capacity}_${xp_label}_s1_r${xp_repeat_tag}_theta${xp_target_version}_s1.json"
  if [[ -f "$xp_s0_result" && -f "$xp_s1_result" ]]; then
    xp_validate_target "$xp_s0_result" s0
    xp_validate_target "$xp_s1_result" s1
    xp_validate_pair "$xp_s0_result" "$xp_s1_result"
    echo "paired S0/S1 bindings and output hashes match for repeat $xp_repeat"
  fi
done

{
  printf 'capacity\tmethod\trecords\tgroups\twall_seconds\trequested_tokens\tremote_tokens\td2h_bytes\twritten_bytes\tpeak_hbm_bytes\toutput_sha256\tresult_path\n'
  shopt -s nullglob
  xp_result_files=(
    "$xp_output_root"/xp"${xp_capacity}_${xp_label}"_*_r??_theta"${xp_target_version}"_*.json
  )
  for xp_result_file in "${xp_result_files[@]}"; do
    jq -r --arg path "$xp_result_file" \
      '[.capacity_name,.method,.records,.groups,.max_rank_wall_seconds,
        .lookup.requested_tokens,.lookup.remote_tokens,.d2h_bytes,
        .written_bytes,.peak_hbm_allocated_bytes_max_rank,
        .output_hash.sha256,$path] | @tsv' \
      "$xp_result_file"
  done | sort
} > "$xp_summary_tsv"

echo "summary: $xp_summary_tsv"
