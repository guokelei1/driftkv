#!/usr/bin/env bash
set -euo pipefail

round_label="${1:-large_variable_d1_score_20260805_round1}"
visible_devices="${EVOKV_CUDA_VISIBLE_DEVICES:-0,1}"
resume="${EVOKV_RESUME:-0}"
preflight_only="${EVOKV_PREFLIGHT_ONLY:-0}"

if ! [[ "$round_label" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "invalid round label: $round_label" >&2
  exit 2
fi
if [[ "$visible_devices" != "0,1" ]]; then
  echo "this round is frozen to GPU0/GPU1" >&2
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
for command in python torchrun sha256sum nvidia-smi flock tee cmp awk df rg git; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 2
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
config="configs/evokv_d1/development/large_variable_score_sweep_two_gpu_v0.json"
registry="configs/evokv_foundation/selected_checkpoint_registry_development_v0.json"
source_archive="data/tenrec/Tenrec.zip"
qk_het="data/processed/evokv_foundation/x_qk_het_foundation.npz"
qk_catalog="data/processed/evokv_d3_m1_qk_entity_cache/entity_catalog_base64_top250000.npz"
qb_source_corpus="data/processed/evokv_qb_large_multifield/mf9_e4096_corpus.npz"
qb_catalog="data/processed/evokv_qb_large_multifield/mf9_e4096_catalog.npz"
qk_checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/quality_lr_dual_20260802_round1_lr015"
qb_checkpoint_root="checkpoints/evokv_qb_large_mf9_e4096/qb_large_round1/u30_e3"
qk_method_root="results/baseline_rounds/quality_chain/recursive_d1_round_a/qk_recursive_d1_round_a_20260804_round1/methods/ract_kv_exact0"
qk_program_root="${qk_method_root}/programs"
result_parent="results/baseline_rounds/large_variable_d1_score"
log_parent="logs/baseline_rounds/large_variable_d1_score"
result_root="${result_parent}/${round_label}"
log_root="${log_parent}/${round_label}"
lock_root="${log_parent}/.locks"
lock_path="${lock_root}/${round_label}.lock"

read -r qk_corpus qk_summary qk_records qb_corpus qb_summary qb_fit qb_probe qb_qualification < <(
  python - "$config" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1]))
qk = value["datasets"]["qk"]
qb = value["datasets"]["qb"]
print(
    qk["corpus"],
    qk["corpus_summary"],
    qk["qualification_records"],
    qb["corpus"],
    qb["corpus_summary"],
    qb["fit_records"],
    qb["probe_records"],
    qb["qualification_records"],
)
PY
)

static_inputs=(
  "$config"
  "$registry"
  "$source_archive"
  "$qk_het"
  "$qk_catalog"
  "$qb_source_corpus"
  "$qb_catalog"
  "$qk_program_root/theta1_to_theta2_direct_oldkv_fp16.pt"
  "$qk_program_root/theta2_to_theta3_direct_oldkv_fp16.pt"
  "$qk_program_root/theta3_to_theta4_direct_oldkv_fp16.pt"
  "$qk_method_root/method_summary.json"
  "$qk_checkpoint_root/theta_1/manifest.json"
  "$qk_checkpoint_root/theta_2/manifest.json"
  "$qk_checkpoint_root/theta_3/manifest.json"
  "$qk_checkpoint_root/theta_4/manifest.json"
  "$qb_checkpoint_root/theta_1/manifest.json"
  "$qb_checkpoint_root/theta_2/manifest.json"
  "$qb_checkpoint_root/theta_3/manifest.json"
  "scripts/build_evokv_large_variable_inference.py"
  "scripts/evaluate_evokv_large_variable_d1.py"
  "scripts/summarize_evokv_large_variable_d1.py"
  "scripts/validate_evokv_large_variable_d1_result.py"
  "scripts/verify_evokv_large_variable_d1.py"
  "scripts/verify_evokv_selected_checkpoints.py"
  "scripts/run_evokv_large_variable_d1_score_sweep.sh"
)
mapfile -t source_inputs < <(LC_ALL=C rg --files src/hstu_kvcache -g '*.py' | LC_ALL=C sort)
static_inputs+=("${source_inputs[@]}")
for path in "${static_inputs[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing frozen input: $path" >&2
    exit 4
  fi
done

mkdir -p "$lock_root"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "large variable D1 round is already running: $round_label" >&2
  exit 3
fi

ensure_gpu_idle() {
  local busy_pids
  busy_pids="$({
    nvidia-smi -i "$visible_devices" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
  } | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {print $1}' | sort -u)"
  if [[ -n "$busy_pids" ]]; then
    echo "GPU0/GPU1 already have compute processes: $busy_pids" >&2
    return 1
  fi
}

python scripts/verify_evokv_large_variable_d1.py --config "$config" >/dev/null
ensure_gpu_idle
mapfile -t gpu_rows < <(
  nvidia-smi -i "$visible_devices" \
    --query-gpu=index,name,memory.total,memory.used,uuid \
    --format=csv,noheader,nounits
)
if (( ${#gpu_rows[@]} != 2 )); then
  echo "exactly GPU0/GPU1 must be available" >&2
  exit 4
fi
for row in "${gpu_rows[@]}"; do
  total="$(awk -F, '{gsub(/ /,"",$3); print $3}' <<<"$row")"
  used="$(awk -F, '{gsub(/ /,"",$4); print $4}' <<<"$row")"
  if (( total < 45000 || used > 512 )); then
    echo "GPU preflight failed: $row" >&2
    exit 4
  fi
done
gib=$((1024 * 1024 * 1024))
disk_free_bytes="$(df -PB1 "$repo_root" | awk 'NR==2 {print $4}')"
memory_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if (( disk_free_bytes < 16 * gib )); then
  echo "large variable D1 round requires at least 16 GiB free disk" >&2
  exit 4
fi
if (( memory_available_kib < 512 * 1024 * 1024 )); then
  echo "large variable D1 round requires at least 512 GiB available host DRAM" >&2
  exit 4
fi

static_candidate="$(mktemp)"
corpus_candidate=""
cleanup() {
  if [[ -n "${static_candidate:-}" ]]; then
    rm -f "$static_candidate"
  fi
  if [[ -n "${corpus_candidate:-}" ]]; then
    rm -f "$corpus_candidate"
  fi
}
trap cleanup EXIT
sha256sum "${static_inputs[@]}" >"$static_candidate"

echo "large variable D1 score sweep is ready"
echo "models: QK theta1-theta4 and QB theta1-theta3, seven large checkpoints"
echo "devices: GPU0/GPU1, one two-rank job at a time"
echo "numeric precision: high for evaluation, IEEE FP32 for QB ridge fitting"
echo "variable qualification records: QK=$qk_records, QB=$qb_qualification"
echo "estimated wall time: 4-8 hours"
echo "estimated peak host DRAM: 360-450 GiB"
echo "estimated peak HBM per rank: 30-38 GiB"
echo "estimated additional durable disk: 2-4 GiB"
echo "result root: $result_root"
echo "log root: $log_root"
if [[ "$preflight_only" == "1" ]]; then
  echo "read-only preflight complete"
  exit 0
fi

if [[ -e "$result_root" && "$resume" == "0" ]]; then
  echo "result root exists; set EVOKV_RESUME=1 or choose a new round label" >&2
  exit 3
fi
static_hashes="${result_root}/static_input_hashes.tsv"
if [[ -e "$result_root" ]]; then
  if [[ ! -f "$static_hashes" ]] || ! cmp -s "$static_candidate" "$static_hashes"; then
    if [[ -f "${result_root}/execution_complete.json" ]]; then
      echo "completed round is immutable and its inputs differ; choose a new label" >&2
      exit 5
    fi
    stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    mkdir -p "${result_parent}/incomplete_attempts" "${log_parent}/incomplete_attempts"
    archived_result="${result_parent}/incomplete_attempts/${round_label}_${stamp}"
    mv "$result_root" "$archived_result"
    if [[ -e "$log_root" ]]; then
      mv "$log_root" "${log_parent}/incomplete_attempts/${round_label}_${stamp}"
    fi
    echo "resume: archived the incompatible incomplete attempt at $archived_result"
  fi
fi

mkdir -p "$result_root" "$log_root" "${result_root}/incomplete_attempts"
if [[ -f "$static_hashes" ]]; then
  if ! cmp -s "$static_candidate" "$static_hashes"; then
    echo "frozen static inputs differ" >&2
    exit 5
  fi
else
  cp "$static_candidate" "$static_hashes"
fi

python scripts/verify_evokv_selected_checkpoints.py \
  --registry "$registry" \
  --output "${result_root}/selected_checkpoint_verification.json" \
  >"${log_root}/selected_checkpoint_verification.log"

python - \
  "${result_root}/preflight.json" \
  "${result_root}/preflight_latest.json" \
  "$round_label" \
  "$visible_devices" \
  "$disk_free_bytes" \
  "$memory_available_kib" \
  "$config" \
  "$static_hashes" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


path = pathlib.Path(sys.argv[1])
latest = pathlib.Path(sys.argv[2])
config_path = pathlib.Path(sys.argv[7])
hashes_path = pathlib.Path(sys.argv[8])
value = {
    "schema": "evokv_large_variable_d1_preflight_v0",
    "status": "pass",
    "scientific_result": False,
    "formal_result": False,
    "round_label": sys.argv[3],
    "visible_devices": sys.argv[4],
    "world_size": 2,
    "gpu_inventory": subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            sys.argv[4],
            "--query-gpu=index,name,memory.total,memory.used,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines(),
    "disk_free_bytes_at_start": int(sys.argv[5]),
    "memory_available_kib_at_start": int(sys.argv[6]),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "config": {"path": str(config_path), "sha256": sha256(config_path)},
    "static_input_hashes": {"path": str(hashes_path), "sha256": sha256(hashes_path)},
    "execution": {
        "datasets_in_order": ["qk", "qb"],
        "large_model_checkpoints": 7,
        "estimated_wall_time": "4-8 hours",
        "estimated_peak_host_dram": "360-450 GiB",
        "estimated_peak_hbm_per_rank": "30-38 GiB",
        "single_current_serving_model": True,
        "numeric_precision": {
            "evaluation_float32_matmul_precision": "high",
            "ridge_float32_matmul_precision": "highest",
            "nvidia_tf32_override": "unset",
            "ridge_gram_accumulation": "ieee_fp32",
        },
    },
    "retention": {
        "keep": ["corpus bindings", "compact metrics", "QB fitted programs", "logs"],
        "discard": ["all full K/V payloads", "GPU state", "regenerable exact caches"],
    },
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if path.exists():
    previous = json.loads(path.read_text())
    for field in (
        "schema",
        "round_label",
        "visible_devices",
        "world_size",
        "config",
        "static_input_hashes",
        "execution",
        "retention",
    ):
        if previous.get(field) != value.get(field):
            raise RuntimeError("large variable D1 preflight binding changed")
else:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    temporary.replace(path)
temporary = latest.with_name(f".{latest.name}.{os.getpid()}.tmp")
temporary.write_text(encoded)
temporary.replace(latest)
PY

set +e
python scripts/build_evokv_large_variable_inference.py \
  --dataset qk \
  --output "$qk_corpus" \
  --summary "$qk_summary" \
  --qk-records "$qk_records" \
  2>&1 | tee "${log_root}/build_qk_variable_corpus.log"
status=${PIPESTATUS[0]}
set -e
if (( status != 0 )); then
  echo "QK variable corpus build failed; rerun with EVOKV_RESUME=1" >&2
  exit "$status"
fi

set +e
python scripts/build_evokv_large_variable_inference.py \
  --dataset qb \
  --output "$qb_corpus" \
  --summary "$qb_summary" \
  --qb-fit-records "$qb_fit" \
  --qb-probe-records "$qb_probe" \
  --qb-qualification-records "$qb_qualification" \
  2>&1 | tee "${log_root}/build_qb_variable_corpus.log"
status=${PIPESTATUS[0]}
set -e
if (( status != 0 )); then
  echo "QB variable corpus build failed; rerun with EVOKV_RESUME=1" >&2
  exit "$status"
fi

corpus_hashes="${result_root}/corpus_input_hashes.tsv"
corpus_candidate="$(mktemp)"
sha256sum "$qk_corpus" "$qk_summary" "$qb_corpus" "$qb_summary" >"$corpus_candidate"
if [[ -f "$corpus_hashes" ]]; then
  if ! cmp -s "$corpus_candidate" "$corpus_hashes"; then
    echo "frozen variable corpora changed; choose a new round label" >&2
    exit 5
  fi
else
  cp "$corpus_candidate" "$corpus_hashes"
fi

python scripts/verify_evokv_large_variable_d1.py \
  --config "$config" \
  --require-corpora \
  --output "${result_root}/input_verification.json" \
  >"${log_root}/input_verification.log"

validate_dataset() {
  local dataset="$1"
  local output="${result_root}/${dataset}"
  python scripts/validate_evokv_large_variable_d1_result.py \
    --config "$config" \
    --dataset "$dataset" \
    --result "${output}/result.json"
}

run_dataset() {
  local dataset="$1"
  local corpus="$2"
  local output="${result_root}/${dataset}"
  if [[ -e "$output" ]]; then
    if [[ -f "${output}/result.json" ]] && validate_dataset "$dataset" >"${log_root}/validate_${dataset}.log" 2>&1; then
      echo "resume: validated and skipped $dataset"
      return
    fi
    stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    archive="${result_root}/incomplete_attempts/${dataset}_${stamp}"
    mv "$output" "$archive"
    echo "resume: archived incomplete $dataset stage at $archive"
  fi
  ensure_gpu_idle
  attempt_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  log="${log_root}/${dataset}_${attempt_id}.log"
  echo "starting large variable D1 dataset: $dataset"
  set +e
  env -u NVIDIA_TF32_OVERRIDE \
    CUDA_VISIBLE_DEVICES="$visible_devices" \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc-per-node=2 \
      scripts/evaluate_evokv_large_variable_d1.py \
      --config "$config" \
      --dataset "$dataset" \
      --corpus "$corpus" \
      --output-root "$output" \
      2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    echo "$dataset failed with status $status; partial state remains at $output" >&2
    echo "resume: EVOKV_RESUME=1 scripts/run_evokv_large_variable_d1_score_sweep.sh $round_label" >&2
    exit "$status"
  fi
  validate_dataset "$dataset" | tee "${log_root}/validate_${dataset}.log"
  echo "completed and validated: $dataset"
}

run_dataset qk "$qk_corpus"
run_dataset qb "$qb_corpus"

python scripts/summarize_evokv_large_variable_d1.py \
  --config "$config" \
  --qk-result "${result_root}/qk/result.json" \
  --qb-result "${result_root}/qb/result.json" \
  --output "${result_root}/round_summary.json" \
  --tsv "${result_root}/round_summary.tsv" \
  --return-manifest "${result_root}/return_manifest.json" \
  2>&1 | tee "${log_root}/summarize.log"

mapfile -t unexpected_payloads < <(find "$result_root" -type f \( -name '*.npy' -o -name '*.npz' \) -print)
if (( ${#unexpected_payloads[@]} > 0 )); then
  printf '%s\n' "${unexpected_payloads[@]}" >&2
  echo "unexpected persistent K/V-like payloads found" >&2
  exit 7
fi

python - \
  "${result_root}/round_summary.json" \
  "${result_root}/round_summary.tsv" \
  "${result_root}/return_manifest.json" \
  "${result_root}/input_verification.json" \
  "${result_root}/execution_complete.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


def artifact(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


summary, table, returned, inputs, output = map(pathlib.Path, sys.argv[1:])
value = {
    "schema": "evokv_large_variable_d1_execution_complete_v0",
    "status": "complete",
    "scientific_result": False,
    "formal_result": False,
    "large_model_checkpoints": 7,
    "single_current_serving_model": True,
    "full_kv_payloads_persisted": 0,
    "artifacts": [artifact(path) for path in (summary, table, returned, inputs)],
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if output.exists():
    if output.read_text() != encoded:
        raise RuntimeError("large variable D1 completion marker differs")
else:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    temporary.replace(output)
print(json.dumps(value, sort_keys=True))
PY

echo "large variable D1 score sweep complete"
echo "return: ${result_root}/round_summary.json"
echo "return: ${result_root}/round_summary.tsv"
echo "return: ${result_root}/qk/result.json"
echo "return: ${result_root}/qb/result.json"
echo "return: ${result_root}/return_manifest.json"
