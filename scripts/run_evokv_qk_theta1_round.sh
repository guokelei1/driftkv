#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_CONFIG="configs/evokv_foundation/qk_theta1_candidate_a_two_gpu_v0.json"
DATA_CONFIG="configs/evokv_foundation/qk_theta0_theta7_stream_data_v0.json"
ROUND_ROOT="results/foundation_model/qk_theta1/qk_theta1_candidate_a_20260805_round1"
FROZEN_CONFIG="$ROUND_ROOT/frozen_config.json"
FROZEN_DATA_CONFIG="$ROUND_ROOT/frozen_data_config.json"
CORPUS_LOG="$ROUND_ROOT/corpus_build.log"
TRAINING_LOG="$ROUND_ROOT/training_quality.log"
VALIDATION_LOG="$ROUND_ROOT/validation.log"
PREFLIGHT="$ROUND_ROOT/preflight.json"
INPUT_HASHES="$ROUND_ROOT/input_hashes.tsv"
RESULT="$ROUND_ROOT/result.json"

mkdir -p "$ROUND_ROOT"
if [[ -f "$FROZEN_CONFIG" ]]; then
    cmp --silent "$SOURCE_CONFIG" "$FROZEN_CONFIG" || {
        echo "frozen theta1 config differs from source config" >&2
        exit 1
    }
else
    cp "$SOURCE_CONFIG" "$FROZEN_CONFIG"
fi
if [[ -f "$FROZEN_DATA_CONFIG" ]]; then
    cmp --silent "$DATA_CONFIG" "$FROZEN_DATA_CONFIG" || {
        echo "frozen stream data config differs from source config" >&2
        exit 1
    }
else
    cp "$DATA_CONFIG" "$FROZEN_DATA_CONFIG"
fi

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ -f "$RESULT" ]]; then
    python scripts/validate_evokv_qk_stream_edge.py \
        --config "$FROZEN_CONFIG" 2>&1 | tee -a "$VALIDATION_LOG"
    echo "QK theta1 round is already complete and valid"
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
    Path(config["source_checkpoint"]["root"]) / "theta_0" / "training_state.json",
    Path(config["data"]["config"]),
    Path("data/tenrec/Tenrec.zip"),
    Path("data/processed/evokv_d3_m1_qk_entity_cache/entity_catalog_base64_top250000.npz"),
    Path("data/processed/evokv_foundation/qk_full_user_lengths.npz"),
    Path("data/processed/evokv_foundation/qk_theta0_next_item_corpus_v0.npz"),
    Path("data/processed/evokv_foundation/x_qk_het_foundation.npz"),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"required QK theta1 inputs are absent: {missing}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("QK theta1 round requires exactly GPU0/GPU1 to be visible")
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
    "config": str(Path(sys.argv[1])),
    "visible_devices": "0,1",
    "gpus": gpus,
    "disk": {
        "free_bytes": disk.free,
        "minimum_free_bytes": minimum_disk,
    },
}
path = Path(sys.argv[2])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(report, indent=2, sort_keys=True))
PY

sha256sum \
    "$FROZEN_CONFIG" \
    "$FROZEN_DATA_CONFIG" \
    "checkpoints/evokv_qk_next_item_e4096_h1536/seed0/theta_0/manifest.json" \
    "checkpoints/evokv_qk_next_item_e4096_h1536/seed0/theta_0/training_state.json" \
    "data/tenrec/Tenrec.zip" \
    "data/processed/evokv_d3_m1_qk_entity_cache/entity_catalog_base64_top250000.npz" \
    "data/processed/evokv_foundation/qk_full_user_lengths.npz" \
    "data/processed/evokv_foundation/qk_theta0_next_item_corpus_v0.npz" \
    "data/processed/evokv_foundation/x_qk_het_foundation.npz" \
    "src/hstu_kvcache/data/qk_stream_chain.py" \
    "src/hstu_kvcache/streaming/qk_stream_version.py" \
    "src/hstu_kvcache/streaming/qk_stream_runner.py" \
    "src/hstu_kvcache/streaming/xp_projected_edge.py" \
    "src/hstu_kvcache/streaming/xp_version_training.py" \
    "scripts/build_evokv_qk_stream_chain_corpus.py" \
    "scripts/train_evokv_qk_theta1.py" \
    "scripts/validate_evokv_qk_stream_edge.py" \
    "scripts/run_evokv_qk_theta1_round.sh" > "$INPUT_HASHES"

if [[ "${EVOKV_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "QK theta1 handoff preflight completed"
    exit 0
fi

CORPUS="data/processed/evokv_foundation/qk_theta0_theta7_stream_v0.npz"
if [[ ! -f "$CORPUS" ]]; then
    python scripts/build_evokv_qk_stream_chain_corpus.py \
        --config "$FROZEN_DATA_CONFIG" 2>&1 | tee -a "$CORPUS_LOG"
fi
python scripts/validate_evokv_qk_stream_edge.py \
    --config "$FROZEN_CONFIG" --corpus-only 2>&1 | tee -a "$VALIDATION_LOG"

torchrun --standalone --nproc_per_node=2 \
    scripts/train_evokv_qk_theta1.py \
    --config "$FROZEN_CONFIG" 2>&1 | tee -a "$TRAINING_LOG"

python scripts/validate_evokv_qk_stream_edge.py \
    --config "$FROZEN_CONFIG" 2>&1 | tee -a "$VALIDATION_LOG"

python - "$FROZEN_CONFIG" "$ROUND_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
round_root = Path(sys.argv[2])
config = json.loads(config_path.read_text())

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()

result_path = Path(config["outputs"]["result"])
result = json.loads(result_path.read_text())
artifacts = {
    "result": {"path": str(result_path), "sha256": digest(result_path)},
    "stream_summary": {
        "path": config["data"]["summary"],
        "sha256": digest(config["data"]["summary"]),
    },
    "stream_roles": {
        "path": config["data"]["roles"],
        "sha256": digest(config["data"]["roles"]),
    },
    "frozen_config": {"path": str(config_path), "sha256": digest(config_path)},
    "training_quality_log": {
        "path": str(round_root / "training_quality.log"),
        "sha256": digest(round_root / "training_quality.log"),
    },
    "validation_log": {
        "path": str(round_root / "validation.log"),
        "sha256": digest(round_root / "validation.log"),
    },
}
if result["checkpoint"]["committed"]:
    checkpoint = (
        Path(config["outputs"]["checkpoint_root"])
        / "theta_1"
        / "manifest.json"
    )
    artifacts["checkpoint_manifest"] = {
        "path": str(checkpoint),
        "sha256": digest(checkpoint),
    }
return_manifest = {
    "status": result["status"],
    "round_id": config["round_id"],
    "artifacts_to_return": artifacts,
}
(round_root / "return_manifest.json").write_text(
    json.dumps(return_manifest, indent=2, sort_keys=True) + "\n"
)
(round_root / "execution_complete.json").write_text(
    json.dumps(
        {"status": result["status"], "round_id": config["round_id"]},
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(json.dumps(return_manifest, indent=2, sort_keys=True))
PY
