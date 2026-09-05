#!/usr/bin/env bash
set -uo pipefail

readonly REPO_ROOT="/home/gkl/work/evokv"
readonly RESULT_ROOT="${REPO_ROOT}/results/yambda500m_medium_seed17/insight1_locality_v1"
readonly LOG_DIR="${RESULT_ROOT}/logs"
readonly LOG_PATH="${LOG_DIR}/formal_tmux.log"
readonly EXIT_PATH="${LOG_DIR}/formal_tmux.exit_code"

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/scripts"
export PYTHONUNBUFFERED="1"

printf '[%s] starting Insight 1 formal run in tmux\n' "$(date --iso-8601=seconds)" | tee -a "${LOG_PATH}"

/home/gkl/miniconda3/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  scripts/insight_one_locality/run_distributed.py \
  --scope formal \
  --batch-size 32 \
  --candidate-chunk 32 \
  2>&1 | tee -a "${LOG_PATH}"
run_status=${PIPESTATUS[0]}

printf '%s\n' "${run_status}" > "${EXIT_PATH}"
printf '[%s] Insight 1 formal run exited with status %s\n' \
  "$(date --iso-8601=seconds)" "${run_status}" | tee -a "${LOG_PATH}"
exit "${run_status}"
