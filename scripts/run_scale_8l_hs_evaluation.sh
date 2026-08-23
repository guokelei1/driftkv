#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}/src:${repo_root}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p results/scale_8l_v1/logs

releases=(r1_edge1 r1_edge2 r2)
devices=(cuda:0 cuda:1 cuda:2)
pids=()

for index in "${!releases[@]}"; do
  release="${releases[$index]}"
  device="${devices[$index]}"
  output="results/scale_8l_v1/hs_raw/${release}/m0_f_seed17"
  if [[ -e "${output}" ]]; then
    echo "[scale-8l-hs] refusing to overwrite ${output}" >&2
    exit 2
  fi
  echo "[scale-8l-hs] starting ${release} on ${device} $(date --iso-8601=seconds)"
  python scripts/eval_scale_8l_hs_raw.py \
    --release "${release}" \
    --device "${device}" \
    > "results/scale_8l_v1/logs/hs_${release}_m0_f_seed17.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[scale-8l-hs] ${releases[$index]} raw scoring completed $(date --iso-8601=seconds)"
  else
    echo "[scale-8l-hs] ${releases[$index]} raw scoring failed" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "[scale-8l-hs] stopping before raw sealing" >&2
  exit 3
fi

python scripts/seal_scale_8l_hs_raw.py
for release in "${releases[@]}"; do
  python scripts/adjudicate_scale_8l_hs.py --release "${release}"
done

echo "[scale-8l-hs] raw scoring, sealing, and adjudication completed $(date --iso-8601=seconds)"
