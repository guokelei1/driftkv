#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_CONFIG="configs/evokv_foundation/qk_theta1_branch_a_e3_lr100_two_gpu_v0.json"
ROUND_ROOT="results/foundation_model/qk_theta1/qk_theta1_branch_a_e3_lr100_20260806_round1"
TRAINING_ROOT="$ROUND_ROOT/training"
ALIGNMENT_ROOT="$ROUND_ROOT/alignment"
PROTOCOL_ROOT="$ROUND_ROOT/protocol_sweep"
FROZEN_CONFIG="$ROUND_ROOT/frozen_training_config.json"
ALIGNMENT_CONFIG="$ALIGNMENT_ROOT/frozen_config.json"
PROTOCOL_CONFIG="$PROTOCOL_ROOT/frozen_config.json"
PREFLIGHT="$ROUND_ROOT/preflight.json"
INPUT_HASHES="$ROUND_ROOT/input_hashes.tsv"
TRAINING_LOG="$TRAINING_ROOT/training.log"
TRAINING_VALIDATION_LOG="$TRAINING_ROOT/validation.log"
ALIGNMENT_LOG="$ALIGNMENT_ROOT/evaluation.log"
ALIGNMENT_VALIDATION_LOG="$ALIGNMENT_ROOT/validation.log"
PROTOCOL_LOG="$PROTOCOL_ROOT/evaluation.log"
PROTOCOL_VALIDATION_LOG="$PROTOCOL_ROOT/validation.log"
SUMMARY_LOG="$ROUND_ROOT/summary.log"
TRAINING_RESULT="$TRAINING_ROOT/result.json"
ALIGNMENT_RESULT="$ALIGNMENT_ROOT/result.json"
PROTOCOL_RESULT="$PROTOCOL_ROOT/result.json"
SUMMARY_RESULT="$ROUND_ROOT/summary.json"

mkdir -p "$TRAINING_ROOT" "$ALIGNMENT_ROOT" "$PROTOCOL_ROOT"
if [[ -f "$FROZEN_CONFIG" ]]; then
    cmp --silent "$SOURCE_CONFIG" "$FROZEN_CONFIG" || {
        echo "frozen branch-A training config differs from source" >&2
        exit 1
    }
else
    cp "$SOURCE_CONFIG" "$FROZEN_CONFIG"
fi

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

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
    Path(config["data"]["roles"]),
    Path(config["data"]["corpus"]),
    Path(config["data"]["summary"]),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"required branch-A inputs are absent: {missing}")
final_checkpoint = (
    Path(config["outputs"]["checkpoint_root"])
    / f"theta_{config['edge']['target_version']}"
)
if final_checkpoint.exists():
    raise FileExistsError(f"committed theta1 would shadow the candidate: {final_checkpoint}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("QK branch-A round requires exactly GPU0/GPU1")
minimum_hbm = int(config["execution"]["minimum_free_hbm_bytes_per_rank"])
gpus = []
for index in range(2):
    free, total = torch.cuda.mem_get_info(index)
    if free < minimum_hbm:
        raise RuntimeError(f"GPU{index} free HBM {free} is below {minimum_hbm}")
    gpus.append(
        {
            "visible_index": index,
            "name": torch.cuda.get_device_name(index),
            "free_bytes": free,
            "total_bytes": total,
        }
    )
disk = shutil.disk_usage(Path.cwd())
minimum_disk = int(config["execution"]["minimum_free_disk_bytes"])
checkpoint = (
    Path(config["outputs"]["work_checkpoint_root"])
    / "theta_1"
    / "manifest.json"
)
if not checkpoint.is_file() and disk.free < minimum_disk:
    raise RuntimeError(f"free disk {disk.free} is below {minimum_disk}")
report = {
    "status": "pass",
    "round_id": config["round_id"],
    "branch": "A",
    "visible_devices": "0,1",
    "gpus": gpus,
    "disk_free_bytes": disk.free,
    "minimum_disk_bytes_for_new_checkpoint": minimum_disk,
    "candidate_checkpoint_already_present": checkpoint.is_file(),
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
sha256sum \
    "$FROZEN_CONFIG" \
    "docs/01-1_qk_theta0_theta1_branch_exploration.md" \
    "configs/evokv_foundation/qk_theta0_theta7_stream_data_v0.json" \
    "configs/evokv_foundation/qk_theta0_theta7_stream_roles_v0.json" \
    "checkpoints/evokv_qk_next_item_e4096_h1536/seed0/theta_0/manifest.json" \
    "checkpoints/evokv_qk_next_item_e4096_h1536/seed0/theta_0/training_state.json" \
    "data/processed/evokv_foundation/qk_theta0_theta7_stream_v0.npz" \
    "src/hstu_kvcache/data/qk_stream_chain.py" \
    "src/hstu_kvcache/streaming/qk_stream_version.py" \
    "src/hstu_kvcache/streaming/qk_stream_runner.py" \
    "src/hstu_kvcache/streaming/qk_full_catalog_runner.py" \
    "src/hstu_kvcache/streaming/qk_alignment_runner.py" \
    "src/hstu_kvcache/streaming/qk_protocol_sweep_runner.py" \
    "scripts/train_evokv_qk_theta1_branch_a_e3_lr100.py" \
    "scripts/materialize_evokv_qk_theta1_branch_a_e3_evaluations.py" \
    "scripts/summarize_evokv_qk_theta1_branch_a_e3.py" \
    "scripts/validate_evokv_qk_full_catalog_tuning.py" \
    "scripts/validate_evokv_qk_theta1_stream_alignment.py" \
    "scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py" \
    "scripts/run_evokv_qk_theta1_branch_a_e3_round.sh" \
    > "$HASH_CANDIDATE"
if [[ -f "$INPUT_HASHES" ]]; then
    cmp --silent "$HASH_CANDIDATE" "$INPUT_HASHES" || {
        echo "branch-A inputs changed after the round was frozen" >&2
        exit 1
    }
    unlink "$HASH_CANDIDATE"
else
    mv "$HASH_CANDIDATE" "$INPUT_HASHES"
fi

python scripts/validate_evokv_qk_full_catalog_tuning.py \
    --config "$FROZEN_CONFIG" --corpus-only \
    2>&1 | tee -a "$TRAINING_VALIDATION_LOG"

if [[ "${EVOKV_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "QK branch-A e3 preflight completed"
    exit 0
fi

if [[ -f "$TRAINING_RESULT" ]]; then
    python scripts/validate_evokv_qk_full_catalog_tuning.py \
        --config "$FROZEN_CONFIG" \
        2>&1 | tee -a "$TRAINING_VALIDATION_LOG"
else
    torchrun --standalone --nproc_per_node=2 \
        scripts/train_evokv_qk_theta1_branch_a_e3_lr100.py \
        --config "$FROZEN_CONFIG" \
        2>&1 | tee -a "$TRAINING_LOG"
    python scripts/summarize_evokv_qk_reuse_recompute.py \
        --config "$FROZEN_CONFIG" \
        2>&1 | tee -a "$SUMMARY_LOG"
    python scripts/validate_evokv_qk_full_catalog_tuning.py \
        --config "$FROZEN_CONFIG" \
        2>&1 | tee -a "$TRAINING_VALIDATION_LOG"
fi

python scripts/materialize_evokv_qk_theta1_branch_a_e3_evaluations.py \
    --training-config "$FROZEN_CONFIG" \
    --alignment-config "$ALIGNMENT_CONFIG" \
    --protocol-config "$PROTOCOL_CONFIG"

python scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py \
    --config "$PROTOCOL_CONFIG" --inputs-only \
    2>&1 | tee -a "$PROTOCOL_VALIDATION_LOG"
if [[ -f "$PROTOCOL_RESULT" ]]; then
    python scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py \
        --config "$PROTOCOL_CONFIG" \
        2>&1 | tee -a "$PROTOCOL_VALIDATION_LOG"
else
    torchrun --standalone --nproc_per_node=2 \
        scripts/evaluate_evokv_qk_theta1_candidate_protocol_sweep.py \
        --config "$PROTOCOL_CONFIG" \
        2>&1 | tee -a "$PROTOCOL_LOG"
    python scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py \
        --config "$PROTOCOL_CONFIG" \
        2>&1 | tee -a "$PROTOCOL_VALIDATION_LOG"
fi

python scripts/validate_evokv_qk_theta1_stream_alignment.py \
    --config "$ALIGNMENT_CONFIG" --inputs-only \
    2>&1 | tee -a "$ALIGNMENT_VALIDATION_LOG"
if [[ -f "$ALIGNMENT_RESULT" ]]; then
    python scripts/validate_evokv_qk_theta1_stream_alignment.py \
        --config "$ALIGNMENT_CONFIG" \
        2>&1 | tee -a "$ALIGNMENT_VALIDATION_LOG"
else
    torchrun --standalone --nproc_per_node=2 \
        scripts/evaluate_evokv_qk_theta1_stream_alignment.py \
        --config "$ALIGNMENT_CONFIG" \
        2>&1 | tee -a "$ALIGNMENT_LOG"
    python scripts/validate_evokv_qk_theta1_stream_alignment.py \
        --config "$ALIGNMENT_CONFIG" \
        2>&1 | tee -a "$ALIGNMENT_VALIDATION_LOG"
fi

python scripts/summarize_evokv_qk_theta1_branch_a_e3.py \
    --config "$FROZEN_CONFIG" \
    --alignment-config "$ALIGNMENT_CONFIG" \
    --protocol-config "$PROTOCOL_CONFIG" \
    2>&1 | tee -a "$SUMMARY_LOG"

python - "$FROZEN_CONFIG" "$ALIGNMENT_CONFIG" "$PROTOCOL_CONFIG" "$ROUND_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

training_config = Path(sys.argv[1])
alignment_config = Path(sys.argv[2])
protocol_config = Path(sys.argv[3])
round_root = Path(sys.argv[4])
config = json.loads(training_config.read_text())

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()

summary = Path(config["outputs"]["summary_json"])
checkpoint = (
    Path(config["outputs"]["work_checkpoint_root"])
    / "theta_1"
    / "manifest.json"
)
artifacts = {
    "summary": {"path": str(summary), "sha256": digest(summary)},
    "summary_markdown": {
        "path": config["outputs"]["summary_markdown"],
        "sha256": digest(config["outputs"]["summary_markdown"]),
    },
    "training_result": {
        "path": config["outputs"]["result"],
        "sha256": digest(config["outputs"]["result"]),
    },
    "alignment_result": {
        "path": json.loads(alignment_config.read_text())["outputs"]["result"],
        "sha256": digest(
            json.loads(alignment_config.read_text())["outputs"]["result"]
        ),
    },
    "protocol_sweep_result": {
        "path": json.loads(protocol_config.read_text())["outputs"]["result"],
        "sha256": digest(
            json.loads(protocol_config.read_text())["outputs"]["result"]
        ),
    },
    "candidate_checkpoint_manifest": {
        "path": str(checkpoint),
        "sha256": digest(checkpoint),
    },
}
return_manifest = {
    "status": "complete_development_measurement",
    "round_id": config["round_id"],
    "branch": "A",
    "qualification_consumed": False,
    "final_consumed": False,
    "artifacts_to_return": artifacts,
}
(round_root / "return_manifest.json").write_text(
    json.dumps(return_manifest, indent=2, sort_keys=True) + "\n"
)
(round_root / "execution_complete.json").write_text(
    json.dumps(return_manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(return_manifest, indent=2, sort_keys=True))
PY

echo "QK branch-A e3 round completed: $SUMMARY_RESULT"
