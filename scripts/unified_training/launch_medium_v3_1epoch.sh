#!/usr/bin/env bash
set -u
cd /home/gkl/work/evokv
run_dir=results/unified_training_2026_09/medium/seed17/v3_1epoch_4gpu_b32_cpu14
/home/gkl/miniconda3/bin/python scripts/unified_training/run_medium_v2_2epoch.py \
  --execution-config configs/unified_training_2026_09/medium_v3_4gpu_b32_cpu14_execution.yaml \
  > "$run_dir/logs/pipeline.log" 2>&1
run_status=$?
printf '%s\n' "$run_status" > "$run_dir/exit_status.txt"
exit "$run_status"
