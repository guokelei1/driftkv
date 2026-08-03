#!/usr/bin/env bash
set -euo pipefail

round_label="${1:-quality_lr015_d1_residual_20260802_round1}"
pair_a="${EVOKV_GPU_PAIR_A:-0,1}"
pair_b="${EVOKV_GPU_PAIR_B:-2,3}"
rank_a="${EVOKV_D1_RANK_A:-16}"
rank_b="${EVOKV_D1_RANK_B:-64}"
ridge="${EVOKV_D1_RESIDUAL_RIDGE:-0.001}"
maximum_tokens="${EVOKV_D1_MAX_FIT_TOKENS_PER_RANK:-4096}"
resume="${EVOKV_RESUME:-0}"
preflight_only="${EVOKV_PREFLIGHT_ONLY:-0}"

if ! [[ "$round_label" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "invalid round label" >&2
  exit 2
fi
if ! [[ "$pair_a" =~ ^[0-9]+,[0-9]+$ && "$pair_b" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "GPU pairs must each name two GPUs" >&2
  exit 2
fi
if ! [[ "$rank_a" =~ ^[1-9][0-9]*$ && "$rank_b" =~ ^[1-9][0-9]*$ ]]; then
  echo "residual ranks must be positive integers" >&2
  exit 2
fi
if [[ "$rank_a" == "$rank_b" ]]; then
  echo "residual ranks must differ" >&2
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

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
checkpoint_root="checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/quality_lr_dual_20260802_round1_lr015"
baseline_root="results/baseline_rounds/quality_chain/quality_lr_dual_20260802_round1_lr015"
config="configs/evokv_quality/quality_lr_dual_20260802_round1_lr015_benchmark.json"
control="results/baseline_rounds/quality_chain/quality_lr015_analytic_d1_round1/summary.json"
root="results/baseline_rounds/quality_chain/explorations/${round_label}"
log_root="logs/baseline_rounds/quality_chain/explorations/${round_label}"
label_a="${round_label}_rank${rank_a}"
label_b="${round_label}_rank${rank_b}"
summary_a="results/baseline_rounds/quality_chain/${label_a}/summary.json"
summary_b="results/baseline_rounds/quality_chain/${label_b}/summary.json"

for path in "$config" "$control" "$baseline_root/summary.json" "$checkpoint_root/theta_1/manifest.json" scripts/run_evokv_d1_candidate_bridge.sh scripts/compare_evokv_d1_candidate_bridges.py; do
  if [[ ! -f "$path" ]]; then
    echo "missing input: $path" >&2
    exit 3
  fi
done

IFS=',' read -r -a a_devices <<< "$pair_a"
IFS=',' read -r -a b_devices <<< "$pair_b"
all_devices=("${a_devices[@]}" "${b_devices[@]}")
if [[ "${a_devices[0]}" == "${a_devices[1]}" || "${b_devices[0]}" == "${b_devices[1]}" ]]; then
  echo "each GPU pair must contain two distinct devices" >&2
  exit 3
fi
if [[ "$(printf '%s\n' "${all_devices[@]}" | sort -u | wc -l)" != "4" ]]; then
  echo "the two GPU pairs must be disjoint" >&2
  exit 3
fi

mkdir -p "$root" "$log_root"
exec 9>"${root}/round.lock"
if ! flock -n 9; then
  echo "exploration round is already running" >&2
  exit 3
fi
if [[ -f "$root/comparison.json" && "$resume" == "0" ]]; then
  echo "completed comparison exists; choose a new label or set EVOKV_RESUME=1" >&2
  exit 3
fi

busy_pids="$({ nvidia-smi -i "$(IFS=,; echo "${all_devices[*]}")" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true; } | sed '/^[[:space:]]*$/d' | sort -u)"
if [[ -n "$busy_pids" ]]; then
  echo "selected GPUs are busy: $busy_pids" >&2
  exit 4
fi
gib=$((1024 * 1024 * 1024))
disk_free_bytes="$(df -PB1 "$repo_root" | awk 'NR==2 {print $4}')"
memory_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if (( disk_free_bytes < 40 * gib )); then
  echo "dual D1 exploration needs at least 40 GiB free disk" >&2
  exit 4
fi
if (( memory_available_kib < 128 * 1024 * 1024 )); then
  echo "dual D1 exploration needs at least 128 GiB available DRAM" >&2
  exit 4
fi

inputs=(
  "$config"
  "$control"
  "$baseline_root/summary.json"
  "scripts/evaluate_evokv_xp_d1_quality.py"
  "scripts/run_evokv_d1_candidate_bridge.sh"
  "scripts/compare_evokv_d1_candidate_bridges.py"
  "scripts/run_evokv_d1_dual_residual_exploration.sh"
  "src/hstu_kvcache/migration/xp_residual_fit.py"
)
candidate_hashes="$(mktemp)"
trap 'rm -f "${candidate_hashes:-}"' EXIT
sha256sum "${inputs[@]}" > "$candidate_hashes"
if [[ -f "$root/input_hashes.tsv" ]]; then
  if ! cmp -s "$candidate_hashes" "$root/input_hashes.tsv"; then
    echo "exploration inputs changed; choose a new round label" >&2
    exit 5
  fi
else
  mv "$candidate_hashes" "$root/input_hashes.tsv"
  candidate_hashes=""
fi

python - "$root/preflight.json" "$round_label" "$pair_a" "$pair_b" "$rank_a" "$rank_b" "$ridge" "$maximum_tokens" "$disk_free_bytes" "$memory_available_kib" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = {
    "schema": "evokv_d1_dual_residual_exploration_preflight_v0",
    "status": "pass",
    "scientific_result": False,
    "formal_result": False,
    "round_label": sys.argv[2],
    "candidates": [
        {"gpu_pair": sys.argv[3], "rank": int(sys.argv[5])},
        {"gpu_pair": sys.argv[4], "rank": int(sys.argv[6])},
    ],
    "fixed_axes": {
        "quality_chain": "LR0.15 24L/H1536/E4096",
        "fit_source": "disjoint theta12 label-free source/current K/V pairs",
        "ridge": float(sys.argv[7]),
        "maximum_fit_tokens_per_rank": int(sys.argv[8]),
        "online_program_shape": "identical to analytic direct-old-K/V control",
        "qualification_edges": [
            "theta1_to_theta2",
            "theta2_to_theta3",
            "theta3_to_theta4",
        ],
    },
    "disk_free_bytes_at_start": int(sys.argv[9]),
    "memory_available_kib_at_start": int(sys.argv[10]),
    "retention": {
        "full_kv_payloads": 0,
        "candidate_programs": "retain through result-dependent comparison",
    },
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if path.exists():
    original = json.loads(path.read_text())
    for field in ("schema", "round_label", "candidates", "fixed_axes", "retention"):
        if original.get(field) != value.get(field):
            raise RuntimeError("dual residual preflight differs")
else:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)
PY

EVOKV_CUDA_VISIBLE_DEVICES="$pair_a" \
EVOKV_D1_CHECKPOINT_ROOT="$checkpoint_root" \
EVOKV_D1_BASELINE_ROOT="$baseline_root" \
EVOKV_D1_CONFIG="$config" \
EVOKV_D1_RESIDUAL_RANK="$rank_a" \
EVOKV_D1_RESIDUAL_RIDGE="$ridge" \
EVOKV_D1_MAX_FIT_TOKENS_PER_RANK="$maximum_tokens" \
EVOKV_PREFLIGHT_ONLY=1 EVOKV_RESUME="$resume" \
  scripts/run_evokv_d1_candidate_bridge.sh "$label_a"
EVOKV_CUDA_VISIBLE_DEVICES="$pair_b" \
EVOKV_D1_CHECKPOINT_ROOT="$checkpoint_root" \
EVOKV_D1_BASELINE_ROOT="$baseline_root" \
EVOKV_D1_CONFIG="$config" \
EVOKV_D1_RESIDUAL_RANK="$rank_b" \
EVOKV_D1_RESIDUAL_RIDGE="$ridge" \
EVOKV_D1_MAX_FIT_TOKENS_PER_RANK="$maximum_tokens" \
EVOKV_PREFLIGHT_ONLY=1 EVOKV_RESUME="$resume" \
  scripts/run_evokv_d1_candidate_bridge.sh "$label_b"

if [[ "$preflight_only" == "1" ]]; then
  echo "dual residual preflight complete: $root/preflight.json"
  exit 0
fi

pids=()
labels=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

EVOKV_CUDA_VISIBLE_DEVICES="$pair_a" \
EVOKV_D1_CHECKPOINT_ROOT="$checkpoint_root" \
EVOKV_D1_BASELINE_ROOT="$baseline_root" \
EVOKV_D1_CONFIG="$config" \
EVOKV_D1_RESIDUAL_RANK="$rank_a" \
EVOKV_D1_RESIDUAL_RIDGE="$ridge" \
EVOKV_D1_MAX_FIT_TOKENS_PER_RANK="$maximum_tokens" \
EVOKV_RESUME=1 scripts/run_evokv_d1_candidate_bridge.sh "$label_a" \
  >"$log_root/${label_a}_driver.log" 2>&1 &
pids+=("$!")
labels+=("$label_a")
EVOKV_CUDA_VISIBLE_DEVICES="$pair_b" \
EVOKV_D1_CHECKPOINT_ROOT="$checkpoint_root" \
EVOKV_D1_BASELINE_ROOT="$baseline_root" \
EVOKV_D1_CONFIG="$config" \
EVOKV_D1_RESIDUAL_RANK="$rank_b" \
EVOKV_D1_RESIDUAL_RIDGE="$ridge" \
EVOKV_D1_MAX_FIT_TOKENS_PER_RANK="$maximum_tokens" \
EVOKV_RESUME=1 scripts/run_evokv_d1_candidate_bridge.sh "$label_b" \
  >"$log_root/${label_b}_driver.log" 2>&1 &
pids+=("$!")
labels+=("$label_b")

statuses=()
set +e
for pid in "${pids[@]}"; do
  wait "$pid"
  statuses+=("$?")
done
set -e
python - "$root/job_status.json" "${labels[0]}" "${statuses[0]}" "${labels[1]}" "${statuses[1]}" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = {
    "schema": "evokv_d1_dual_residual_job_status_v0",
    "jobs": [
        {"label": sys.argv[2], "exit_status": int(sys.argv[3])},
        {"label": sys.argv[4], "exit_status": int(sys.argv[5])},
    ],
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if path.exists() and path.read_text() != encoded:
    raise FileExistsError("dual residual job status differs")
if not path.exists():
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)
PY
if (( statuses[0] != 0 || statuses[1] != 0 )); then
  echo "one or more residual candidates failed; inspect $root/job_status.json" >&2
  exit 7
fi

python scripts/compare_evokv_d1_candidate_bridges.py \
  --control "$control" \
  --candidate "rank${rank_a}=${summary_a}" \
  --candidate "rank${rank_b}=${summary_b}" \
  --output "$root/comparison.json" \
  --tsv "$root/comparison.tsv"
echo "round complete: $root/comparison.json"
echo "full K/V payloads retained: 0"
