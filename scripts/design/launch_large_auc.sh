#!/usr/bin/env bash
# Fixed ten-layer diagnostic; all measurements stay in the dated result root.
set -u
cd /home/gkl/work/evokv || exit 1
result_root=results/insight/large_unified_auc_10k_20260920
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
printf '%s\n' main_diagnostic > "$result_root/running_phase.txt"
python scripts/design/run_large_auc_parallel.py --output "$result_root/diagnostic" --batch-size 64 --threads 4 > "$result_root/diagnostic.log" 2>&1
experiment_exit=$?
# Historical renderer retired; this launcher records diagnostic execution only.
printf '%s\n' "$experiment_exit" > "$result_root/exit_status.txt"
if [ "$experiment_exit" -eq 0 ]; then
    printf '%s\n' completed > "$result_root/running_phase.txt"
else
    printf '%s\n' failed > "$result_root/running_phase.txt"
fi
exit "$experiment_exit"
