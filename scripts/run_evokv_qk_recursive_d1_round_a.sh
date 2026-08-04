#!/usr/bin/env bash
set -euo pipefail

round_label="${1:-qk_recursive_d1_round_a_20260804_round1}"
visible_devices="${EVOKV_CUDA_VISIBLE_DEVICES:-0,1}"
resume="${EVOKV_RESUME:-0}"
preflight_only="${EVOKV_PREFLIGHT_ONLY:-0}"

if ! [[ "$round_label" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "invalid round label: $round_label" >&2
  exit 2
fi
if [[ "$visible_devices" != "0,1" ]]; then
  echo "this round is frozen to GPU0/GPU1; GPU2/GPU3 and four-rank execution are unavailable" >&2
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
for command in python torchrun sha256sum nvidia-smi flock tee cmp diff awk df rg git; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 2
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
config="configs/evokv_d1/development/qk_recursive_round_a_two_gpu_v0.json"
registry="configs/evokv_foundation/selected_checkpoint_registry_development_v0.json"
benchmark="configs/evokv_quality/quality_lr_dual_20260802_round1_lr015_benchmark.json"
edge_input="data/processed/evokv_quality/qk_xp_quality_stream_aligned_train16384_qual4096_nested_v2.npz"
edge_summary="configs/evokv_quality/qk_xp_quality_stream_aligned_train16384_qual4096_nested_v2_summary.json"
design_contract="configs/evokv_d1/development/recursive_large_chain_design_v0.json"
checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/quality_lr_dual_20260802_round1_lr015"
incumbent_root="results/baseline_rounds/quality_chain/quality_lr015_d1_residual_20260802_round1_rank16"
result_root="results/baseline_rounds/quality_chain/recursive_d1_round_a/${round_label}"
log_root="logs/baseline_rounds/quality_chain/recursive_d1_round_a/${round_label}"
method_root="${result_root}/methods"
attempt_root="${result_root}/incomplete_attempts"
lock_path="${log_root}/.round.lock"
methods=(
  reuse_exact_baselines
  incumbent_rank16_recursive
  rollout_only_exact0
  ract_kv_exact0
  ract_kv_exact10
  ract_kv_exact20
)

mkdir -p "$(dirname "$lock_path")"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "recursive D1 round is already running: $round_label" >&2
  exit 3
fi
if [[ -e "$result_root" && "$resume" == "0" ]]; then
  echo "result root exists; choose a new label or set EVOKV_RESUME=1" >&2
  exit 3
fi

static_inputs=(
  "$config"
  "$registry"
  "$benchmark"
  "$edge_input"
  "$edge_summary"
  "$design_contract"
  "$incumbent_root/summary.json"
  "$incumbent_root/theta1_to_theta2_direct_oldkv_fp16.pt"
  "$incumbent_root/theta2_to_theta3_direct_oldkv_fp16.pt"
  "$incumbent_root/theta3_to_theta4_direct_oldkv_fp16.pt"
  "$checkpoint_root/theta_1/manifest.json"
  "$checkpoint_root/theta_2/manifest.json"
  "$checkpoint_root/theta_3/manifest.json"
  "$checkpoint_root/theta_4/manifest.json"
  "scripts/evaluate_evokv_qk_recursive_d1.py"
  "scripts/summarize_evokv_qk_recursive_d1.py"
  "scripts/verify_evokv_selected_checkpoints.py"
  "scripts/run_evokv_qk_recursive_d1_round_a.sh"
  "src/hstu_kvcache/migration/recursive_d1.py"
  "src/hstu_kvcache/migration/low_rank.py"
  "src/hstu_kvcache/migration/program.py"
  "src/hstu_kvcache/migration/stage45_oldkv.py"
  "src/hstu_kvcache/migration/xp_d1_quality.py"
  "src/hstu_kvcache/migration/xp_exact_baseline.py"
  "src/hstu_kvcache/streaming/sharded_edge.py"
  "src/hstu_kvcache/streaming/trainer.py"
  "src/hstu_kvcache/streaming/xp_multiversion.py"
  "src/hstu_kvcache/streaming/xp_version_training.py"
)
mapfile -t model_sources < <(
  LC_ALL=C rg --files src/hstu_kvcache/models | LC_ALL=C sort
)
static_inputs+=("${model_sources[@]}")
for path in "${static_inputs[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing frozen input: $path" >&2
    exit 4
  fi
done

python - "$config" "$repo_root" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


config_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
value = json.loads(config_path.read_text())
expected_methods = [
    "reuse_exact_baselines",
    "incumbent_rank16_recursive",
    "rollout_only_exact0",
    "ract_kv_exact0",
    "ract_kv_exact10",
    "ract_kv_exact20",
]
if (
    value.get("schema")
    != "evokv_qk_recursive_d1_round_a_two_gpu_development_v0"
    or value.get("status") != "ready_for_user_execution"
    or value.get("world_size") != 2
    or value.get("methods") != expected_methods
    or value.get("scientific_result") is not False
    or value.get("formal_result") is not False
    or value.get("serving_model_invariant", {}).get(
        "concurrent_recommendation_models"
    )
    != 1
):
    raise SystemExit("recursive D1 round config differs")
for binding in value["bindings"].values():
    path = pathlib.Path(binding["path"])
    path = path if path.is_absolute() else root / path
    if not path.is_file() or sha256(path) != binding["sha256"]:
        raise SystemExit(f"recursive D1 binding differs: {path}")
for binding in value["incumbent_programs"].values():
    path = pathlib.Path(binding["path"])
    path = path if path.is_absolute() else root / path
    if not path.is_file() or sha256(path) != binding["sha256"]:
        raise SystemExit(f"incumbent program differs: {path}")
PY

busy_pids="$({
  nvidia-smi -i "$visible_devices" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
} | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {print $1}' | sort -u)"
if [[ -n "$busy_pids" ]]; then
  echo "GPU0/GPU1 already have compute processes: $busy_pids" >&2
  exit 4
fi
mapfile -t gpu_rows < <(
  nvidia-smi -i "$visible_devices" \
    --query-gpu=index,name,memory.total,memory.used,uuid \
    --format=csv,noheader,nounits
)
if (( ${#gpu_rows[@]} != 2 )); then
  echo "exactly GPU0/GPU1 must be visible" >&2
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
if (( disk_free_bytes < 120 * gib )); then
  echo "recursive D1 round requires at least 120 GiB free disk" >&2
  exit 4
fi
if (( memory_available_kib < 256 * 1024 * 1024 )); then
  echo "recursive D1 round requires at least 256 GiB available host DRAM" >&2
  exit 4
fi

mkdir -p "$result_root" "$log_root" "$method_root" "$attempt_root"
static_hashes="${result_root}/static_input_hashes.tsv"
hash_candidate="$(mktemp)"
trap 'rm -f "${hash_candidate:-}"' EXIT
sha256sum "${static_inputs[@]}" > "$hash_candidate"
if [[ -f "$static_hashes" ]]; then
  if ! cmp -s "$hash_candidate" "$static_hashes"; then
    echo "frozen round inputs changed; choose a new round label" >&2
    diff -u "$static_hashes" "$hash_candidate" >&2 || true
    exit 5
  fi
else
  mv "$hash_candidate" "$static_hashes"
  hash_candidate=""
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
config = json.loads(config_path.read_text())
gpu_rows = subprocess.check_output(
    [
        "nvidia-smi",
        "-i",
        sys.argv[4],
        "--query-gpu=index,name,memory.total,memory.used,uuid",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).splitlines()
value = {
    "schema": "evokv_qk_recursive_d1_round_a_preflight_v0",
    "status": "pass",
    "scientific_result": False,
    "formal_result": False,
    "round_label": sys.argv[3],
    "visible_devices": sys.argv[4],
    "world_size": 2,
    "gpu_inventory": gpu_rows,
    "disk_free_bytes_at_start": int(sys.argv[5]),
    "memory_available_kib_at_start": int(sys.argv[6]),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "config": {
        "path": str(config_path),
        "sha256": sha256(config_path),
    },
    "static_input_hashes": {
        "path": str(hashes_path),
        "sha256": sha256(hashes_path),
    },
    "methods": config["methods"],
    "edges": config["edges"],
    "execution": {
        "sequential_two_rank_jobs": 6,
        "estimated_wall_time": "2-4 hours",
        "estimated_peak_host_dram": "220-300 GiB",
        "estimated_additional_durable_disk": "6-10 GiB",
        "single_current_serving_model": True,
        "true_recursive_kv_handoff": True,
    },
    "retention": {
        "keep": [
            "programs",
            "action plans",
            "edge metrics",
            "summaries",
            "logs",
            "bindings",
        ],
        "discard": [
            "all full K/V payloads",
            "GPU state",
            "regenerable exact prefixes",
        ],
    },
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if path.exists():
    original = json.loads(path.read_text())
    for field in (
        "schema",
        "round_label",
        "visible_devices",
        "world_size",
        "config",
        "static_input_hashes",
        "methods",
        "edges",
        "execution",
        "retention",
    ):
        if original.get(field) != value.get(field):
            raise RuntimeError("recursive D1 preflight binding changed")
else:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)
temporary = latest.with_name(f".{latest.name}.{os.getpid()}.tmp")
temporary.write_text(encoded)
os.replace(temporary, latest)
PY

validate_method() {
  local root="$1"
  local method="$2"
  python - "$root" "$method" "$config" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


root = pathlib.Path(sys.argv[1])
method = sys.argv[2]
config_path = pathlib.Path(sys.argv[3])
config = json.loads(config_path.read_text())
summary_path = root / "method_summary.json"
if not summary_path.is_file():
    raise SystemExit(f"method completion marker is absent: {summary_path}")
summary = json.loads(summary_path.read_text())
expected_edges = [
    f"theta{edge['source_version']}_to_theta{edge['target_version']}"
    for edge in config["edges"]
]
if (
    summary.get("protocol")
    != "evokv_qk_recursive_d1_round_a_development_v0"
    or summary.get("status") != "complete"
    or summary.get("method") != method
    or summary.get("world_size") != 2
    or summary.get("single_current_serving_model") is not True
    or summary.get("true_recursive_handoff") is not True
    or summary.get("hidden_exact_reset") is not False
    or summary.get("admissible_full_round") is not True
    or summary.get("full_kv_payloads_persisted") != 0
    or summary.get("round_config", {}).get("sha256")
    != sha256(config_path)
    or [edge.get("edge") for edge in summary.get("edges", [])]
    != expected_edges
):
    raise SystemExit(f"method completion marker differs: {method}")
previous_lineage = None
previous_cache_state = None
for descriptor in summary["edges"]:
    edge_path = pathlib.Path(descriptor["path"])
    action_path = pathlib.Path(descriptor["action_plan_path"])
    if (
        not edge_path.is_file()
        or sha256(edge_path) != descriptor["sha256"]
        or not action_path.is_file()
        or sha256(action_path) != descriptor["action_plan_sha256"]
    ):
        raise SystemExit(f"method edge artifact differs: {method}")
    edge = json.loads(edge_path.read_text())
    action = json.loads(action_path.read_text())
    handoff = edge.get("recursive_handoff", {})
    if (
        edge.get("status") != "complete"
        or edge.get("method") != method
        or edge.get("full_kv_payloads_persisted") != 0
        or handoff.get("hidden_exact_reset") is not False
        or previous_lineage is not None
        and handoff.get("input_lineage_sha256") != previous_lineage
        or action.get("protocol")
        != "evokv_qk_recursive_d1_action_plan_development_v0"
        or action.get("method") != method
        or action.get("input_lineage_sha256")
        != handoff.get("input_lineage_sha256")
        or action.get("output_lineage_sha256")
        != handoff.get("output_lineage_sha256")
        or action.get("output_cache_state_sha256")
        != handoff.get("output_cache_state", {}).get("sha256")
        or previous_cache_state is not None
        and handoff.get("input_cache_state", {}).get("sha256")
        != previous_cache_state
    ):
        raise SystemExit(f"method recursive handoff differs: {method}")
    previous_lineage = handoff.get("output_lineage_sha256")
    previous_cache_state = handoff.get("output_cache_state", {}).get("sha256")
for payload in root.rglob("*"):
    if payload.is_file() and payload.suffix.lower() in {".npy", ".npz"}:
        raise SystemExit(f"unexpected persistent K/V-like payload: {payload}")
print(json.dumps({"method": method, "status": "pass"}, sort_keys=True))
PY
}

echo "QK recursive D1 Round-A is ready"
echo "devices: GPU0/GPU1, one two-rank job at a time"
echo "estimated wall time: 2-4 hours"
echo "estimated peak host DRAM: 220-300 GiB"
echo "estimated additional durable disk: 6-10 GiB"
echo "result root: $result_root"
echo "log root: $log_root"
if [[ "$preflight_only" == "1" ]]; then
  echo "preflight complete: ${result_root}/preflight.json"
  exit 0
fi

export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
for method in "${methods[@]}"; do
  output="${method_root}/${method}"
  if [[ -e "$output" ]]; then
    if [[ -f "$output/method_summary.json" ]]; then
      if validate_method "$output" "$method"; then
        if [[ "$resume" != "1" ]]; then
          echo "completed method already exists without resume: $method" >&2
          exit 6
        fi
        echo "resume: validated and skipped $method"
        continue
      fi
      echo "invalid completed method exists; choose a new label: $method" >&2
      exit 6
    fi
    if [[ "$resume" != "1" ]]; then
      echo "incomplete method exists without resume: $method" >&2
      exit 6
    fi
    archive="${attempt_root}/${method}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    mv "$output" "$archive"
    echo "resume: archived incomplete method at $archive"
  fi
  mkdir -p "$output"
  attempt_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  log="${log_root}/${method}_${attempt_id}.log"
  echo "starting method: $method"
  set +e
  CUDA_VISIBLE_DEVICES="$visible_devices" \
    torchrun --standalone --nproc-per-node=2 \
      scripts/evaluate_evokv_qk_recursive_d1.py \
      --config "$config" \
      --method "$method" \
      --output-root "$output" \
      2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    echo "method failed with status $status: $method" >&2
    echo "partial state preserved at: $output" >&2
    echo "resume command: EVOKV_RESUME=1 scripts/run_evokv_qk_recursive_d1_round_a.sh $round_label" >&2
    exit "$status"
  fi
  validate_method "$output" "$method"
  echo "completed and validated method: $method"
done

python scripts/summarize_evokv_qk_recursive_d1.py \
  --result-root "$result_root" \
  --config "$config" \
  --output "${result_root}/round_summary.json" \
  --tsv "${result_root}/round_summary.tsv" \
  --return-manifest "${result_root}/return_manifest.json" \
  2>&1 | tee "${log_root}/summarize.log"

python - \
  "${result_root}/round_summary.json" \
  "${result_root}/round_summary.tsv" \
  "${result_root}/return_manifest.json" \
  "${result_root}/execution_complete.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


summary_path, table_path, manifest_path, output = map(
    pathlib.Path, sys.argv[1:]
)
summary = json.loads(summary_path.read_text())
manifest = json.loads(manifest_path.read_text())
if (
    summary.get("schema")
    != "evokv_qk_recursive_d1_round_a_summary_development_v0"
    or summary.get("status")
    not in {"complete_selected_policy", "complete_no_admitted_policy"}
    or summary.get("world_size") != 2
    or summary.get("single_current_serving_model") is not True
    or summary.get("physical_gpu_speedup_claimed") is not False
    or summary.get("full_kv_payloads_persisted") != 0
    or manifest.get("status") != "complete"
    or manifest.get("full_kv_payloads_persisted") != 0
):
    raise SystemExit("recursive D1 round summary validation failed")
value = {
    "schema": "evokv_qk_recursive_d1_round_a_execution_complete_v0",
    "status": "complete",
    "selected_policy": summary["selection"]["selected_policy"],
    "artifacts": {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (summary_path, table_path, manifest_path)
    },
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if output.exists():
    if output.read_text() != encoded:
        raise SystemExit("recursive D1 completion marker differs")
else:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, output)
print(json.dumps(value, sort_keys=True))
PY

echo "recursive D1 Round-A complete"
echo "return first: ${result_root}/round_summary.json"
echo "return first: ${result_root}/round_summary.tsv"
echo "return first: ${result_root}/return_manifest.json"
