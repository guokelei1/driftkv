#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?usage: scripts/run_evokv_selected_checkpoint_rebuild.sh qk\|qb ROUND_LABEL}
round_label=${2:?usage: scripts/run_evokv_selected_checkpoint_rebuild.sh qk\|qb ROUND_LABEL}
devices=${EVOKV_CUDA_VISIBLE_DEVICES:-0,1}

if [[ ! "$round_label" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "invalid round label: $round_label" >&2
  exit 2
fi
if ! [[ "$devices" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "selected checkpoint rebuild requires exactly two CUDA devices" >&2
  exit 2
fi
IFS=',' read -r device_a device_b <<< "$devices"
if [[ "$device_a" == "$device_b" ]]; then
  echo "selected checkpoint rebuild requires two distinct CUDA devices" >&2
  exit 2
fi

case "$dataset" in
  qk)
    schedule=configs/evokv_quality/quality_lr_dual_20260802_round1_lr015.json
    benchmark=configs/evokv_quality/quality_lr_dual_20260802_round1_lr015_benchmark.json
    base_checkpoint_root=checkpoints/evokv_xp_qk_e4096_h1536/seed0
    base_manifest=${base_checkpoint_root}/theta_0/manifest.json
    bootstrap=${EVOKV_REBUILD_QK_BOOTSTRAP:-0}
    if [[ "$bootstrap" != 0 && "$bootstrap" != 1 ]]; then
      echo "EVOKV_REBUILD_QK_BOOTSTRAP must be 0 or 1" >&2
      exit 2
    fi
    for path in "$schedule" "$benchmark" scripts/run_evokv_quality_chain_existing_config.sh; do
      if [[ ! -f "$path" ]]; then
        echo "missing QK rebuild input: $path" >&2
        exit 3
      fi
    done
    if [[ ! -f "$base_manifest" ]]; then
      if [[ "$bootstrap" != 1 ]]; then
        echo "QK theta0 is absent; rerun with EVOKV_REBUILD_QK_BOOTSTRAP=1" >&2
        exit 3
      fi
      prepared=data/processed/evokv_d3_m1_qk_entity_2560.npz
      cooccurrence=data/processed/evokv_foundation/qk_xp_base_row_cooccurrence.npz
      cooccurrence_summary=configs/evokv_foundation/qk_xp_base_row_cooccurrence_summary.json
      foundation_workload=data/processed/evokv_foundation/x_qk_het_foundation.npz
      bootstrap_result_root=results/baseline_rounds/selected_rebuild_bootstrap/${round_label}
      bootstrap_checkpoint_root=checkpoints/evokv_selected_rebuild_bootstrap/${round_label}/qk_h1536
      bootstrap_source_result=${bootstrap_result_root}/qk_h1536_source.json
      bootstrap_theta0_result=${bootstrap_result_root}/qk_xp_theta0.json
      bootstrap_log_root=${bootstrap_result_root}/logs
      for path in \
        "$prepared" \
        "$cooccurrence" \
        "$cooccurrence_summary" \
        "$foundation_workload" \
        scripts/train_evokv_design3_m1_qk_sharded_edge.py \
        scripts/build_evokv_xp_theta0.py; do
        if [[ ! -f "$path" ]]; then
          echo "missing QK cold-bootstrap input: $path" >&2
          exit 3
        fi
      done
      if [[ -e "$bootstrap_result_root" || -e "$bootstrap_checkpoint_root" ]]; then
        echo "refusing to overwrite QK cold-bootstrap round: $round_label" >&2
        exit 4
      fi
      if [[ -e "$base_checkpoint_root/theta_0" ]]; then
        echo "incomplete QK theta0 target exists: $base_checkpoint_root/theta_0" >&2
        exit 4
      fi
      available_kib=$(df --output=avail -k /data | tail -n 1 | tr -d ' ')
      if (( available_kib < 360 * 1024 * 1024 )); then
        echo "cold QK rebuild needs at least 360 GiB free on /data" >&2
        exit 5
      fi
      mapfile -t gpu_rows < <(
        nvidia-smi -i "$device_a,$device_b" --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits
      )
      if (( ${#gpu_rows[@]} != 2 )); then
        echo "QK cold bootstrap requires two visible A40 GPUs" >&2
        exit 6
      fi
      for row in "${gpu_rows[@]}"; do
        used=$(awk -F, '{gsub(/ /,"",$3); print $3}' <<<"$row")
        total=$(awk -F, '{gsub(/ /,"",$2); print $2}' <<<"$row")
        if (( total < 45000 || used > 512 )); then
          echo "GPU preflight failed: $row" >&2
          exit 7
        fi
      done
      echo "QK theta0 cold bootstrap: transient H1536 source, then persistent E4096 theta0"
      echo "cold-bootstrap wall time: about 1.5-2.5 hours before the selected update chain"
      echo "transient source payload: about 35 GiB; it is reclaimed after theta0 validation"
      if [[ ${EVOKV_PREFLIGHT_ONLY:-0} == 1 ]]; then
        echo "QK cold-bootstrap preflight complete"
        exit 0
      fi
      mkdir -p "$bootstrap_log_root"
      export CUDA_VISIBLE_DEVICES="$devices"
      export OMP_NUM_THREADS=1
      export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
      torchrun --standalone --nproc-per-node=2 \
        scripts/train_evokv_design3_m1_qk_sharded_edge.py \
        --prepared-data "$prepared" \
        --checkpoint-dir "$bootstrap_checkpoint_root" \
        --output "$bootstrap_source_result" \
        >"$bootstrap_log_root/qk_h1536_source.log" 2>&1
      torchrun --standalone --nproc-per-node=2 \
        scripts/build_evokv_xp_theta0.py \
        --source-checkpoint-root "$bootstrap_checkpoint_root" \
        --cooccurrence "$cooccurrence" \
        --cooccurrence-summary "$cooccurrence_summary" \
        --checkpoint-root "$base_checkpoint_root" \
        --foundation-workload "$foundation_workload" \
        --output "$bootstrap_theta0_result" \
        --device cuda \
        >"$bootstrap_log_root/qk_xp_theta0.log" 2>&1
      python - \
        "$bootstrap_source_result" \
        "$bootstrap_theta0_result" \
        "$base_manifest" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


source_path, theta0_path, manifest_path = map(pathlib.Path, sys.argv[1:])
source = json.loads(source_path.read_text())
theta0 = json.loads(theta0_path.read_text())
manifest = json.loads(manifest_path.read_text())
if (
    source.get("status") != "complete"
    or theta0.get("status") != "complete"
    or theta0.get("optimizer_active_gate", {}).get("passed") is not True
    or theta0.get("execution", {}).get("world_size") != 2
    or manifest.get("version") != 0
    or manifest.get("world_size") != 2
    or theta0.get("checkpoint", {}).get("manifest") != manifest
):
    raise SystemExit("QK cold-bootstrap validation failed")
binding = {
    "schema": "evokv_qk_theta0_cold_bootstrap_validation_v0",
    "status": "pass",
    "source_result_sha256": sha256(source_path),
    "theta0_result_sha256": sha256(theta0_path),
    "theta0_manifest_sha256": sha256(manifest_path),
    "transient_source_checkpoint_reclaim_required": True,
}
output = theta0_path.with_name("validation.json")
output.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
PY
      transient_parent=$(dirname "$bootstrap_checkpoint_root")
      rm -rf -- "$bootstrap_checkpoint_root"
      rmdir --ignore-fail-on-non-empty "$transient_parent" 2>/dev/null || true
      python - "$bootstrap_result_root/validation.json" <<'PY'
import json
import os
import pathlib
import sys


path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
value["transient_source_checkpoint_reclaimed"] = True
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
      echo "validated and reclaimed transient QK H1536 source: $bootstrap_checkpoint_root"
    fi
    echo "QK selected-chain rebuild: fixed LR0.15 theta1--theta4 from registered-compatible theta0"
    echo "devices: $devices"
    echo "expected wall time: 75-110 minutes; durable growth: about 179 GiB"
    echo "results: results/baseline_rounds/quality_chain/$round_label"
    echo "checkpoints: checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/$round_label"
    EVOKV_CUDA_VISIBLE_DEVICES="$devices" \
      scripts/run_evokv_quality_chain_existing_config.sh \
      "$round_label" "$schedule" "$benchmark"
    ;;
  qb)
    profile=mf9_e4096
    catalog=data/processed/evokv_qb_large_multifield/${profile}_catalog.npz
    corpus=data/processed/evokv_qb_large_multifield/${profile}_corpus.npz
    foundation=configs/evokv_foundation/qb_large_${profile}_summary_development_v0.json
    audit=configs/evokv_foundation/qb_large_multifield_audit_development_v0.json
    hbm=results/system/evokv_qb_foundation/${profile}_hbm_admission_canary.json
    result_root=results/baseline_rounds/qb_large/${round_label}
    checkpoint_parent=checkpoints/evokv_qb_large_${profile}/${round_label}
    checkpoint_root=${checkpoint_parent}/selected_u30_e3
    first_result=${result_root}/theta0_theta2.json
    second_result=${result_root}/theta2_theta3.json
    log_root=${result_root}/logs
    for path in "$catalog" "$corpus" "$foundation" "$audit" "$hbm" scripts/train_evokv_qb_large_screen.py; do
      if [[ ! -f "$path" ]]; then
        echo "missing QB rebuild input: $path" >&2
        exit 3
      fi
    done
    if [[ -e "$result_root" || -e "$checkpoint_parent" ]]; then
      echo "refusing to overwrite QB rebuild round: $round_label" >&2
      exit 4
    fi
    available_kib=$(df --output=avail -k /data | tail -n 1 | tr -d ' ')
    if (( available_kib < 250 * 1024 * 1024 )); then
      echo "QB selected-chain rebuild needs at least 250 GiB free on /data" >&2
      exit 5
    fi
    mapfile -t gpu_rows < <(
      nvidia-smi -i "$device_a,$device_b" --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits
    )
    if (( ${#gpu_rows[@]} != 2 )); then
      echo "QB selected-chain rebuild requires two visible A40 GPUs" >&2
      exit 6
    fi
    for row in "${gpu_rows[@]}"; do
      used=$(awk -F, '{gsub(/ /,"",$3); print $3}' <<<"$row")
      total=$(awk -F, '{gsub(/ /,"",$2); print $2}' <<<"$row")
      if (( total < 45000 || used > 512 )); then
        echo "GPU preflight failed: $row" >&2
        exit 7
      fi
    done
    echo "QB selected-chain rebuild: fresh theta0, then u30_e3 theta1--theta3"
    echo "devices: $devices"
    echo "expected wall time: 70-120 minutes; durable growth: about 195 GiB"
    echo "results: $result_root"
    echo "checkpoints: $checkpoint_root"
    if [[ ${EVOKV_PREFLIGHT_ONLY:-0} == 1 ]]; then
      exit 0
    fi
    mkdir -p "$log_root" "$checkpoint_root"
    python - "$result_root/frozen_round.json" "$round_label" "$devices" "$catalog" "$corpus" "$foundation" "$audit" "$hbm" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


output = pathlib.Path(sys.argv[1])
inputs = [pathlib.Path(value) for value in sys.argv[4:]]
value = {
    "schema": "evokv_qb_selected_chain_rebuild_v0",
    "scientific_result": False,
    "formal_result": False,
    "round_label": sys.argv[2],
    "cuda_visible_devices": sys.argv[3],
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "inputs": {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in inputs
    },
    "model": "24L/H1536/E4096, nine base-only feature namespaces",
    "training": {
        "base_epochs": 1,
        "update_epochs": 3,
        "dense_projection_lr": 3e-5,
        "embedding_lr": 3e-4,
        "versions": [0, 1, 2, 3],
    },
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
    export CUDA_VISIBLE_DEVICES="$devices"
    export OMP_NUM_THREADS=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    torchrun --standalone --nproc-per-node=2 scripts/train_evokv_qb_large_screen.py \
      --profile "$profile" \
      --catalog "$catalog" \
      --corpus "$corpus" \
      --checkpoint-root "$checkpoint_root" \
      --output "$first_result" \
      --base-epochs 1 \
      --update-epochs 3 \
      --update-dense-lr 3e-5 \
      --update-projection-lr 3e-5 \
      --update-embedding-lr 3e-4 \
      --stop-version 2 \
      --progress-every 100 \
      >"$log_root/theta0_theta2.log" 2>&1
    torchrun --standalone --nproc-per-node=2 scripts/train_evokv_qb_large_screen.py \
      --profile "$profile" \
      --catalog "$catalog" \
      --corpus "$corpus" \
      --checkpoint-root "$checkpoint_root" \
      --resume-chain-result "$first_result" \
      --resume-version 2 \
      --output "$second_result" \
      --base-epochs 1 \
      --update-epochs 3 \
      --update-dense-lr 3e-5 \
      --update-projection-lr 3e-5 \
      --update-embedding-lr 3e-4 \
      --stop-version 3 \
      --progress-every 100 \
      >"$log_root/theta2_theta3.log" 2>&1
    python - "$first_result" "$second_result" "$checkpoint_root" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


first = json.loads(pathlib.Path(sys.argv[1]).read_text())
second = json.loads(pathlib.Path(sys.argv[2]).read_text())
root = pathlib.Path(sys.argv[3])
if (
    first.get("status") != "complete"
    or second.get("status") != "complete"
    or first.get("capacity", {}).get("forced_sharding_gate_passed") is not True
    or second.get("capacity", {}).get("forced_sharding_gate_passed") is not True
    or [entry.get("version") for entry in first.get("checkpoints", [])] != [0, 1, 2]
    or [entry.get("version") for entry in second.get("checkpoints", [])] != [0, 1, 2, 3]
    or [
        (edge.get("source_version"), edge.get("target_version"))
        for edge in second.get("edges", [])
    ]
    != [(2, 3)]
):
    raise SystemExit("QB selected-chain rebuild result differs")
for entry in second["checkpoints"]:
    path = pathlib.Path(entry["manifest_path"])
    if not path.is_file() or sha256(path) != entry["manifest_sha256"]:
        raise SystemExit("QB selected-chain manifest binding differs")
if any("kv" in path.name.lower() or "cache" in path.name.lower() for path in root.rglob("*")):
    raise SystemExit("unexpected durable K/V payload in QB checkpoint root")
PY
    du -sb "$checkpoint_root" "$result_root" >"$result_root/storage.tsv"
    echo "QB selected-chain rebuild complete"
    echo "return $first_result, $second_result, and $result_root/storage.tsv"
    ;;
  *)
    echo "dataset must be qk or qb" >&2
    exit 2
    ;;
esac
