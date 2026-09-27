#!/usr/bin/env bash
# Run from detached tmux after the focused Max canary has passed.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="$REPOSITORY_ROOT/results/insight/max_unified_auc_10k_20260921"
cd "$REPOSITORY_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

phase=starting
finish() {
    local status=$?
    printf '%s\n' "$status" > "$RESULT_ROOT/exit_status.txt"
    printf '%s phase=%s exit=%s\n' "$(date -u +%FT%TZ)" "$phase" "$status" >> "$RESULT_ROOT/launcher.runtime.log"
}
trap finish EXIT

printf '%s start\n' "$(date -u +%FT%TZ)" >> "$RESULT_ROOT/launcher.runtime.log"
sha256sum "$0" >> "$RESULT_ROOT/launcher.runtime.log"
phase=diagnostic
printf '%s\n' "$phase" > "$RESULT_ROOT/phase.txt"
python -u scripts/design/run_max_auc.py --batch-size 64 --threads 4 \
    --initial-record "$RESULT_ROOT/initial_preflight" \
    --output "$RESULT_ROOT/diagnostic" > "$RESULT_ROOT/main.runtime.log" 2>&1

# Historical renderer retired; this launcher records diagnostic execution only.

phase=completed
printf '%s\n' "$phase" > "$RESULT_ROOT/phase.txt"
