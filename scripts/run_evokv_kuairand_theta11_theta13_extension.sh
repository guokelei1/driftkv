#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(11|12|13)$ ]]; then
  echo "usage: $0 {11|12|13}" >&2
  exit 2
fi

target_version="$1"
if [[ "$target_version" == "13" ]]; then
  config="configs/evokv_root_cause/kuairand_theta13_holdout_candidate_20260809_v8.json"
  expected_config_sha256="b16cef13b11d7de86122e8d786e7a7775159fd3a79e828b0849da1c1c4469664"
  output_root="results/root_cause_campaign/kuairand_theta13_holdout_candidate_seed53117_v8"
else
  config="configs/evokv_root_cause/kuairand_theta12_theta13_extension_20260809_v5.json"
  expected_config_sha256="811054c97aa23d71a9772ca118c043a66518f098e28f112d0fb6022f37554e09"
  output_root="results/root_cause_campaign/kuairand_theta12_theta13_extension_seed53117_v5"
fi
checkpoint_root="checkpoints/evokv_kuairand_theta6_dense_interpolation_seed53117_v0"
source_root="results/root_cause_campaign/kuairand_amplified_theta10_sparse_negative_seed53117_v0"
prior_extension_root="results/root_cause_campaign/kuairand_theta11_theta13_extension_seed53117_v3"
theta12_extension_root="results/root_cause_campaign/kuairand_theta12_theta13_extension_seed53117_v5"
log_root="${output_root}/logs"
lock_path="${output_root}/extension.lock"

actual_config_sha256="$(sha256sum "$config" | awk '{print $1}')"
if [[ "$actual_config_sha256" != "$expected_config_sha256" ]]; then
  echo "extension config hash differs" >&2
  exit 1
fi

mkdir -p "$output_root" "$log_root"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "another KuaiRand extension process holds $lock_path" >&2
  exit 1
fi

if pgrep -af 'torchrun.*train_evokv_kuairand_lineage_retrain.py' | grep -v "$$"; then
  echo "another KuaiRand lineage training process is active" >&2
  exit 1
fi

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  if (( used > 512 )); then
    echo "GPU${gpu} is not available: ${used} MiB used by compute processes" >&2
    exit 1
  fi
done

if (( target_version >= 13 )); then
  theta12_source="${theta12_extension_root}/edges/theta_12/accepted.json"
  theta12_destination="${output_root}/edges/theta_12/accepted.json"
  if [[ ! -f "$theta12_source" ]]; then
    echo "missing imported theta12 accepted record" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$theta12_destination")"
  if [[ -f "$theta12_destination" ]]; then
    if ! cmp -s "$theta12_source" "$theta12_destination"; then
      echo "imported theta12 accepted record differs" >&2
      exit 1
    fi
  else
    cp --archive "$theta12_source" "$theta12_destination"
  fi
fi

theta11_source="${prior_extension_root}/edges/theta_11/accepted.json"
theta11_destination="${output_root}/edges/theta_11/accepted.json"
if [[ ! -f "$theta11_source" ]]; then
  echo "missing imported theta11 accepted record" >&2
  exit 1
fi
mkdir -p "$(dirname "$theta11_destination")"
if [[ -f "$theta11_destination" ]]; then
  if ! cmp -s "$theta11_source" "$theta11_destination"; then
    echo "imported theta11 accepted record differs" >&2
    exit 1
  fi
else
  cp --archive "$theta11_source" "$theta11_destination"
fi

for version in $(seq 3 10); do
  source="${source_root}/edges/theta_${version}/accepted.json"
  destination="${output_root}/edges/theta_${version}/accepted.json"
  if [[ ! -f "$source" ]]; then
    echo "missing imported accepted record: $source" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" ]]; then
    if ! cmp -s "$source" "$destination"; then
      echo "imported accepted record differs: $destination" >&2
      exit 1
    fi
  else
    cp --archive "$source" "$destination"
  fi
done

for version in $(seq 3 10); do
  if [[ ! -f "${checkpoint_root}/theta_${version}/manifest.json" ]]; then
    echo "missing imported checkpoint manifest: theta_${version}" >&2
    exit 1
  fi
done

previous_version=$((target_version - 1))
if [[ ! -f "${checkpoint_root}/theta_${previous_version}/manifest.json" || ! -f "${output_root}/edges/theta_${previous_version}/accepted.json" ]]; then
  echo "theta${previous_version} is not durably accepted" >&2
  exit 1
fi

python - "$config" "$target_version" "$output_root" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from hstu_kvcache.streaming.kuairand_lineage_retrain import load_lineage_retrain_config
from hstu_kvcache.streaming.kuairand_query_transition import file_sha256

config_path = Path(sys.argv[1])
target_version = int(sys.argv[2])
output_root = Path(sys.argv[3])
document = load_lineage_retrain_config(config_path)
free_bytes = shutil.disk_usage(Path(document["outputs"]["checkpoint_root"]).parent).free
completed = target_version - 1
remaining = int(document["checkpoint"]["versions"]) - completed
required = (
    remaining
    * int(
        document["checkpoint"].get(
            "expected_checkpoint_bytes_per_version",
            document["checkpoint"]["expected_global_parameter_bytes"],
        )
    )
    + int(document["checkpoint"]["write_reserve_bytes"])
)
record = {
    "status": "passed" if free_bytes >= required else "failed",
    "target_version": target_version,
    "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
    "gpu_indices": [0, 1],
    "world_size": 2,
    "free_bytes": free_bytes,
    "required_remaining_bytes": required,
    "remaining_versions_including_target": remaining,
    "checkpoint_root": document["outputs"]["checkpoint_root"],
    "output_root": str(output_root),
}
path = output_root / "logs" / f"theta{target_version}_preflight.json"
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
if record["status"] != "passed":
    raise SystemExit("extension disk preflight failed")
PY

log_path="${log_root}/theta${target_version}_train.log"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
torchrun --standalone --nproc-per-node=2 \
  scripts/train_evokv_kuairand_lineage_retrain.py \
  --config "$config" \
  --stop-after-version "$target_version" \
  2>&1 | tee "$log_path"

if [[ ! -f "${checkpoint_root}/theta_${target_version}/manifest.json" || ! -f "${output_root}/edges/theta_${target_version}/accepted.json" ]]; then
  echo "theta${target_version} did not produce durable accepted artifacts" >&2
  exit 1
fi

python - "$target_version" "$checkpoint_root" "$output_root" <<'PY'
import json
import sys
from pathlib import Path

from hstu_kvcache.streaming.kuairand_query_transition import file_sha256

version = int(sys.argv[1])
checkpoint_root = Path(sys.argv[2])
output_root = Path(sys.argv[3])
manifest_path = checkpoint_root / f"theta_{version}" / "manifest.json"
accepted_path = output_root / "edges" / f"theta_{version}" / "accepted.json"
manifest = json.loads(manifest_path.read_text())
accepted = json.loads(accepted_path.read_text())
if manifest.get("status") != "complete" or accepted.get("status") != "accepted":
    raise SystemExit("extension durable artifact status differs")
record = {
    "status": "accepted",
    "version": version,
    "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
    "accepted": {"path": str(accepted_path), "sha256": file_sha256(accepted_path)},
    "checkpoint_bytes": manifest["checkpoint_bytes"],
    "candidate": accepted["candidate"]["candidate"]["name"],
}
(output_root / "logs" / f"theta{version}_durable.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(record, indent=2, sort_keys=True))
PY
