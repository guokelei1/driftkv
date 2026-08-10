#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || "${1:-all}" != "bridge" && "${1:-all}" != "next" && "${1:-all}" != "complete" && "${1:-all}" != "all" ]]; then
  echo "usage: $0 [bridge|next|complete|all]" >&2
  exit 2
fi

mode="${1:-all}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_foundation_rebuild_large_e4160_theta12_20260809_v7.json"
expected_config_sha256="f254fb5efcbee1e4756aa21cfb11c668b6dbf326851e48dd24d80b49001d10eb"
output_root="results/root_cause_campaign/kuairand_foundation_rebuild_large_e4160_theta12_20260809_v7"
checkpoint_root="checkpoints/evokv_kuairand_foundation_rebuild_large_e4160_theta12_20260809_v7"
log_root="$output_root/logs"
lock_path="$output_root/run.lock"
result="$output_root/result.json"
matrix_json="$output_root/bounded_matrix_theta5_theta12_holdout.json"
matrix_table="$output_root/bounded_matrix_theta5_theta12_holdout.md"

mkdir -p "$log_root" "$checkpoint_root"
actual_config_sha256="$(sha256sum "$config" | awk '{print $1}')"
if [[ "$actual_config_sha256" != "$expected_config_sha256" ]]; then
  echo "large foundation config hash differs" >&2
  exit 1
fi

exec 9>"$lock_path"
if ! flock -n 9; then
  echo "another large KuaiRand foundation process holds $lock_path" >&2
  exit 1
fi

python scripts/train_evokv_kuairand_theta1_theta8.py \
  --config "$config" \
  --preflight-only > "$log_root/preflight.json"

completed="$(jq -r '.completed_versions' "$log_root/preflight.json")"
target=12
if [[ "$mode" == "bridge" ]]; then
  target=5
elif [[ "$mode" == "next" ]]; then
  target=$((completed + 1))
  if (( target > 12 )); then
    target=12
  fi
fi

if (( completed < target )); then
  if pgrep -af 'torchrun.*train_evokv_kuairand_theta1_theta8.py' | grep -v "$$"; then
    echo "another KuaiRand persistent training process is active" >&2
    exit 1
  fi
  for gpu in 0 1; do
    used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
    if (( used > 512 )); then
      echo "GPU${gpu} is not available: ${used} MiB used by compute processes" >&2
      exit 1
    fi
  done
  export CUDA_VISIBLE_DEVICES=0,1
  export PYTHONUNBUFFERED=1
  if [[ "$mode" == "bridge" || "$mode" == "next" ]]; then
    torchrun --standalone --nproc-per-node=2 \
      scripts/train_evokv_kuairand_theta1_theta8.py \
      --config "$config" \
      --stop-after-version "$target" 2>&1 | tee -a "$log_root/advance_theta${target}.log"
  else
    torchrun --standalone --nproc-per-node=2 \
      scripts/train_evokv_kuairand_theta1_theta8.py \
      --config "$config" 2>&1 | tee -a "$log_root/complete_theta6_theta12_and_lineage.log"
  fi
fi

for version in $(seq 1 "$target"); do
  test -f "$checkpoint_root/theta_${version}/manifest.json"
  test -f "$output_root/edges/theta_${version}/accepted.json"
done

for version in 1 $(seq 5 "$target"); do
  jq -e '.embedding_storage == "full" and .status == "complete"' \
    "$checkpoint_root/theta_${version}/manifest.json" >/dev/null
done
for version in 2 3 4; do
  if (( version <= target )); then
    jq -e '.embedding_storage == "sparse_delta" and .status == "complete"' \
      "$checkpoint_root/theta_${version}/manifest.json" >/dev/null
  fi
done

if [[ "$mode" == "all" ]]; then
  exec "$0" complete
fi

if [[ "$mode" == "complete" ]]; then
  jq -e '.status == "complete" and .checkpoint_count == 12' "$result" >/dev/null
  python scripts/render_evokv_kuairand_persistent_bounded_matrix.py \
    --result "$result" \
    --output-json "$matrix_json" \
    --output-table "$matrix_table" \
    --first-version 5
  jq '{decision,geometry,checkpoint_count,checkpoint_bytes}' "$result"
  jq '.summaries' "$matrix_json"
fi
