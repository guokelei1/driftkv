#!/usr/bin/env bash
set -euo pipefail

low_config=configs/evokv_root_cause/kuairand_stationary_prefix_low_theta3_20260809_v0.json
medium_config=configs/evokv_root_cause/kuairand_stationary_prefix_medium_theta3_20260809_v0.json
source_theta1=checkpoints/evokv_kuairand_theta6_dense_interpolation_seed53117_v0/theta_1
source_accepted=results/root_cause_campaign/kuairand_amplified_theta10_sparse_negative_seed53117_v0/edges/theta_1/accepted.json
low_checkpoint_root=checkpoints/evokv_kuairand_stationary_prefix_low_seed53117_v0
medium_checkpoint_root=checkpoints/evokv_kuairand_stationary_prefix_medium_seed53117_v0
low_output_root=results/root_cause_campaign/kuairand_stationary_prefix_low_seed53117_v0
medium_output_root=results/root_cause_campaign/kuairand_stationary_prefix_medium_seed53117_v0
round_root=results/root_cause_campaign/kuairand_stationary_prefix_parallel_20260809_v0

prepare_branch() {
  checkpoint_root=$1
  output_root=$2
  mkdir -p "$checkpoint_root" "$output_root/edges/theta_1" "$output_root/logs"
  if test ! -e "$checkpoint_root/theta_1"; then
    ln -s "$(realpath "$source_theta1")" "$checkpoint_root/theta_1"
  fi
  test "$(readlink -e "$checkpoint_root/theta_1")" = "$(realpath "$source_theta1")"
  if test ! -e "$output_root/edges/theta_1/accepted.json"; then
    cp --archive "$source_accepted" "$output_root/edges/theta_1/accepted.json"
  fi
  cmp "$source_accepted" "$output_root/edges/theta_1/accepted.json"
}

prepare_branch "$low_checkpoint_root" "$low_output_root"
prepare_branch "$medium_checkpoint_root" "$medium_output_root"
mkdir -p "$round_root/logs"

python -c 'from hstu_kvcache.streaming.kuairand_lineage_retrain import load_lineage_retrain_config; load_lineage_retrain_config("configs/evokv_root_cause/kuairand_stationary_prefix_low_theta3_20260809_v0.json")'
python -c 'from hstu_kvcache.streaming.kuairand_lineage_retrain import load_lineage_retrain_config; load_lineage_retrain_config("configs/evokv_root_cause/kuairand_stationary_prefix_medium_theta3_20260809_v0.json")'
python -c 'from hstu_kvcache.streaming.kuairand_projected_persistent import preflight_persistent_chain; import json; print(json.dumps(preflight_persistent_chain("configs/evokv_root_cause/kuairand_stationary_prefix_low_theta3_20260809_v0.json"), sort_keys=True))' > "$low_output_root/logs/preflight.json"
python -c 'from hstu_kvcache.streaming.kuairand_projected_persistent import preflight_persistent_chain; import json; print(json.dumps(preflight_persistent_chain("configs/evokv_root_cause/kuairand_stationary_prefix_medium_theta3_20260809_v0.json"), sort_keys=True))' > "$medium_output_root/logs/preflight.json"

mapfile -t gpu_free_mib < <(
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
)
test "${#gpu_free_mib[@]}" -ge 4
for gpu in 0 1 2 3; do
  test "${gpu_free_mib[$gpu]}" -ge 43000
done

{
  date --iso-8601=seconds
  sha256sum "$low_config" "$medium_config" "$source_theta1/manifest.json" "$source_accepted"
  df -B1 checkpoints
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
} > "$round_root/logs/preflight.log"

set +e
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_lineage_retrain.py \
  --config "$low_config" > "$low_output_root/logs/lineage_retrain.log" 2>&1 &
low_pid=$!
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
  scripts/train_evokv_kuairand_lineage_retrain.py \
  --config "$medium_config" > "$medium_output_root/logs/lineage_retrain.log" 2>&1 &
medium_pid=$!
wait "$low_pid"
low_lineage_status=$?
wait "$medium_pid"
medium_lineage_status=$?
set -e

low_direct_status=99
medium_direct_status=99
set +e
if test "$low_lineage_status" -eq 0; then
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    scripts/train_evokv_kuairand_theta1_theta8.py \
    --config "$low_config" > "$low_output_root/logs/direct_lineage.log" 2>&1 &
  low_direct_pid=$!
fi
if test "$medium_lineage_status" -eq 0; then
  CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
    scripts/train_evokv_kuairand_theta1_theta8.py \
    --config "$medium_config" > "$medium_output_root/logs/direct_lineage.log" 2>&1 &
  medium_direct_pid=$!
fi
if test "$low_lineage_status" -eq 0; then
  wait "$low_direct_pid"
  low_direct_status=$?
fi
if test "$medium_lineage_status" -eq 0; then
  wait "$medium_direct_pid"
  medium_direct_status=$?
fi
set -e

jq -n \
  --argjson low_lineage "$low_lineage_status" \
  --argjson medium_lineage "$medium_lineage_status" \
  --argjson low_direct "$low_direct_status" \
  --argjson medium_direct "$medium_direct_status" \
  '{status:"complete",low:{lineage:$low_lineage,direct:$low_direct},medium:{lineage:$medium_lineage,direct:$medium_direct}}' \
  > "$round_root/status.json"

if test "$low_direct_status" -eq 0; then
  jq '{round_id,decision,targets}' "$low_output_root/result.json"
fi
if test "$medium_direct_status" -eq 0; then
  jq '{round_id,decision,targets}' "$medium_output_root/result.json"
fi
test "$low_direct_status" -eq 0 -o "$medium_direct_status" -eq 0
