#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONPATH="${repo_root}/src:${repo_root}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

h_gate="results/scale_8l_v1/pilot/s3_m0_f_seed17_h_adjudication.json"
python - "${h_gate}" <<'PY'
import json
import sys

path = sys.argv[1]
value = json.load(open(path))
if value.get("pilot_H_passed") is not True:
    raise SystemExit(f"8L seed17 H gate has not passed: {path}")
if value.get("qualification_or_theta3_read") is not False:
    raise SystemExit("H gate unexpectedly accessed qualification/theta3")
PY

run_release() {
  local release="$1"
  local output="results/scale_8l_v1/releases/${release}/m0_f_seed17"
  if [[ -f "${output}/train_result.json" ]]; then
    echo "[scale-8l] ${release}: sealed completion artifact exists; skipping"
    return 0
  fi
  if [[ -e "${output}" ]]; then
    echo "[scale-8l] ${release}: incomplete output exists; refusing overwrite: ${output}" >&2
    return 2
  fi
  echo "[scale-8l] ${release}: starting $(date --iso-8601=seconds)"
  torchrun --standalone --nproc_per_node=4 \
    scripts/train_scale_8l_fsdp_release.py \
    --release "${release}" \
    --model m0_f \
    --seed 17
  echo "[scale-8l] ${release}: completed $(date --iso-8601=seconds)"
}

run_release r1_edge1
run_release r1_edge2
run_release r2

echo "[scale-8l] M0-F seed17 R1/R2 training queue completed $(date --iso-8601=seconds)"
