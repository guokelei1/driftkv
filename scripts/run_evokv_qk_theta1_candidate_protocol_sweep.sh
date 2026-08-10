#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_CONFIG="configs/evokv_foundation/qk_theta1_candidate_protocol_sweep_two_gpu_v0.json"
ROUND_ROOT="results/foundation_model/qk_theta1/qk_theta1_candidate_protocol_sweep_20260806_round1"
FROZEN_CONFIG="$ROUND_ROOT/frozen_config.json"
PREFLIGHT="$ROUND_ROOT/preflight.json"
INPUT_HASHES="$ROUND_ROOT/input_hashes.tsv"
EVALUATION_LOG="$ROUND_ROOT/evaluation.log"
VALIDATION_LOG="$ROUND_ROOT/validation.log"
RESULT="$ROUND_ROOT/result.json"

mkdir -p "$ROUND_ROOT"
if [[ -f "$FROZEN_CONFIG" ]]; then
    if ! cmp --silent "$SOURCE_CONFIG" "$FROZEN_CONFIG"; then
        if [[ -f "$RESULT" || -s "$EVALUATION_LOG" ]]; then
            echo "frozen QK protocol sweep config changed after execution started" >&2
            exit 1
        fi
        cp "$SOURCE_CONFIG" "$FROZEN_CONFIG"
    fi
else
    cp "$SOURCE_CONFIG" "$FROZEN_CONFIG"
fi

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ -f "$RESULT" ]]; then
    python scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py \
        --config "$FROZEN_CONFIG" 2>&1 | tee -a "$VALIDATION_LOG"
    echo "QK theta1 candidate protocol sweep is already complete and valid"
    exit 0
fi

python - "$FROZEN_CONFIG" "$PREFLIGHT" <<'PY'
import json
import shutil
import sys
from pathlib import Path

import torch

config = json.loads(Path(sys.argv[1]).read_text())
required = [
    Path(config["source_checkpoint"]["root"]) / "theta_0" / "manifest.json",
    Path(config["current_checkpoint"]["root"]) / "theta_1" / "manifest.json",
    Path(config["data"]["corpus"]),
    Path(config["data"]["summary"]),
    Path(config["data"]["roles"]),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"required QK protocol sweep inputs are absent: {missing}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("QK protocol sweep requires exactly GPU0/GPU1")
minimum = int(config["execution"]["minimum_free_hbm_bytes_per_rank"])
gpus = []
for index in range(2):
    free, total = torch.cuda.mem_get_info(index)
    if free < minimum:
        raise RuntimeError(f"GPU{index} free HBM {free} is below {minimum}")
    gpus.append({
        "visible_index": index,
        "name": torch.cuda.get_device_name(index),
        "free_bytes": free,
        "total_bytes": total,
    })
disk = shutil.disk_usage(Path.cwd())
minimum_disk = int(config["execution"]["minimum_free_disk_bytes"])
if disk.free < minimum_disk:
    raise RuntimeError(f"free disk {disk.free} is below {minimum_disk}")
report = {
    "status": "pass",
    "visible_devices": "0,1",
    "gpus": gpus,
    "disk_free_bytes": disk.free,
    "minimum_disk_bytes": minimum_disk,
    "training_started": False,
    "qualification_consumed": False,
    "final_consumed": False,
}
path = Path(sys.argv[2])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(report, indent=2, sort_keys=True))
PY

HASH_CANDIDATE="$ROUND_ROOT/.input_hashes.$$.tmp"
trap 'rm -f "$HASH_CANDIDATE"' EXIT
sha256sum \
    "$FROZEN_CONFIG" \
    "src/hstu_kvcache/streaming/qk_protocol_sweep_runner.py" \
    "src/hstu_kvcache/streaming/qk_stream_version.py" \
    "scripts/evaluate_evokv_qk_theta1_candidate_protocol_sweep.py" \
    "scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py" \
    "scripts/run_evokv_qk_theta1_candidate_protocol_sweep.sh" \
    "checkpoints/evokv_qk_next_item_e4096_h1536/seed0/theta_0/manifest.json" \
    "checkpoints/evokv_qk_next_item_e4096_h1536/seed0/.theta_1_reuse_recompute_full_catalog_20260805_round1_work/theta_1/manifest.json" \
    "data/processed/evokv_foundation/qk_theta0_theta7_stream_v0.npz" \
    > "$HASH_CANDIDATE"
if [[ -f "$INPUT_HASHES" && -s "$EVALUATION_LOG" ]]; then
    cmp --silent "$HASH_CANDIDATE" "$INPUT_HASHES" || {
        echo "QK protocol sweep inputs changed after execution started" >&2
        exit 1
    }
else
    mv "$HASH_CANDIDATE" "$INPUT_HASHES"
fi

python scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py \
    --config "$FROZEN_CONFIG" --inputs-only 2>&1 | tee -a "$VALIDATION_LOG"

if [[ "${EVOKV_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "QK theta1 candidate protocol sweep preflight completed"
    exit 0
fi

torchrun --standalone --nproc_per_node=2 \
    scripts/evaluate_evokv_qk_theta1_candidate_protocol_sweep.py \
    --config "$FROZEN_CONFIG" 2>&1 | tee -a "$EVALUATION_LOG"

python scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py \
    --config "$FROZEN_CONFIG" 2>&1 | tee -a "$VALIDATION_LOG"

python - "$FROZEN_CONFIG" "$ROUND_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
round_root = Path(sys.argv[2])
config = json.loads(config_path.read_text())
result_path = Path(config["outputs"]["result"])

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()

result = json.loads(result_path.read_text())
gate = result["quality"]["primary_update_local"]["stable_gap_gate"]
manifest = {
    "status": "complete_development_measurement",
    "round_id": config["round_id"],
    "training_performed": False,
    "qualification_consumed": False,
    "final_consumed": False,
    "stable_gap_gate": gate["status"],
    "admitted": gate["admitted"],
    "artifacts_to_return": {
        "result": {"path": str(result_path), "sha256": digest(result_path)},
        "frozen_config": {"path": str(config_path), "sha256": digest(config_path)},
        "evaluation_log": {
            "path": str(round_root / "evaluation.log"),
            "sha256": digest(round_root / "evaluation.log"),
        },
        "validation_log": {
            "path": str(round_root / "validation.log"),
            "sha256": digest(round_root / "validation.log"),
        },
    },
}
(round_root / "return_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
(round_root / "execution_complete.json").write_text(
    json.dumps({"status": "complete_development_measurement"}, indent=2, sort_keys=True)
    + "\n"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
