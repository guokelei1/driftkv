#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PLAN="configs/evokv_foundation/qk_theta2_route_a_sweep_two_gpu_v0.json"
SWEEP_ROOT="results/foundation_model/qk_theta2/qk_theta2_route_a_sweep_20260806_round1"
PREFLIGHT="$SWEEP_ROOT/preflight.json"
INPUT_HASHES="$SWEEP_ROOT/input_hashes.tsv"
MATERIALIZE_LOG="$SWEEP_ROOT/materialize.log"
SUMMARY_LOG="$SWEEP_ROOT/summary.log"
RETURN_MANIFEST="$SWEEP_ROOT/return_manifest.json"

CANDIDATES=(
    "theta2_route_a_e3_lr100"
    "theta2_route_a_e4_lr100"
    "theta2_route_a_e3_lr150"
)
ROUND_IDS=(
    "qk_theta2_route_a_e3_lr100_20260806_round1"
    "qk_theta2_route_a_e4_lr100_20260806_round1"
    "qk_theta2_route_a_e3_lr150_20260806_round1"
)
ENTRIES=(
    "scripts/train_evokv_qk_theta2_route_a_e3_lr100.py"
    "scripts/train_evokv_qk_theta2_route_a_e4_lr100.py"
    "scripts/train_evokv_qk_theta2_route_a_e3_lr150.py"
)

mkdir -p "$SWEEP_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python scripts/materialize_evokv_qk_theta2_route_a_sweep.py \
    --plan "$PLAN" 2>&1 | tee -a "$MATERIALIZE_LOG"

python - "$PLAN" "$PREFLIGHT" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

plan_path = Path(sys.argv[1])
output = Path(sys.argv[2])
plan = json.loads(plan_path.read_text())
source = plan["source_checkpoint"]
source_root = Path(source["root"]) / "theta_1"

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()

required = [
    source_root / "manifest.json",
    source_root / "training_state.json",
    source_root / "optimizer_resume.pt",
    Path(plan["data"]["config"]),
    Path(plan["data"]["roles"]),
    Path(plan["data"]["corpus"]),
    Path(plan["data"]["summary"]),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"required theta2 inputs are absent: {missing}")
if (
    digest(source_root / "manifest.json") != source["manifest_sha256"]
    or digest(source_root / "training_state.json")
    != source["training_state_sha256"]
    or digest(source_root / "optimizer_resume.pt")
    != source["optimizer_resume_sha256"]
):
    raise RuntimeError("selected theta1 input differs")
final_checkpoint = Path(plan["outputs"]["checkpoint_parent"]) / "theta_2"
if final_checkpoint.exists():
    raise FileExistsError(f"committed theta2 would shadow the sweep: {final_checkpoint}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("QK theta2 sweep requires exactly GPU0/GPU1")
minimum_hbm = int(plan["execution"]["minimum_free_hbm_bytes_per_rank"])
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
minimum_disk = int(plan["execution"]["minimum_free_disk_bytes"])
if disk.free < minimum_disk:
    raise RuntimeError(f"free disk {disk.free} is below {minimum_disk}")
report = {
    "status": "pass",
    "round_id": plan["round_id"],
    "visible_devices": "0,1",
    "gpus": gpus,
    "disk_free_bytes": disk.free,
    "minimum_disk_bytes": minimum_disk,
    "candidate_count": len(plan["candidates"]),
    "estimated_wall_minutes": plan["execution"]["estimated_wall_minutes_total"],
    "qualification_consumed": False,
    "final_consumed": False,
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(report, indent=2, sort_keys=True))
PY

python - "$PLAN" "$INPUT_HASHES" <<'PY'
import hashlib
import sys
from pathlib import Path

plan = Path(sys.argv[1])
output = Path(sys.argv[2])
paths = [
    plan,
    Path("docs/01-1_qk_theta0_theta1_branch_exploration.md"),
    Path("configs/evokv_foundation/qk_theta1_branch_a_e3_freeze_20260806_v0.json"),
    Path("configs/evokv_foundation/qk_theta0_theta7_stream_data_v0.json"),
    Path("configs/evokv_foundation/qk_theta0_theta7_stream_roles_v0.json"),
    Path("data/processed/evokv_foundation/qk_theta0_theta7_stream_v0.npz"),
    Path("checkpoints/evokv_qk_next_item_e4096_h1536/seed0/.theta_1_branch_a_e3_lr100_20260806_round1_work/theta_1/manifest.json"),
    Path("checkpoints/evokv_qk_next_item_e4096_h1536/seed0/.theta_1_branch_a_e3_lr100_20260806_round1_work/theta_1/training_state.json"),
    Path("checkpoints/evokv_qk_next_item_e4096_h1536/seed0/.theta_1_branch_a_e3_lr100_20260806_round1_work/theta_1/optimizer_resume.pt"),
    Path("src/hstu_kvcache/streaming/qk_stream_version.py"),
    Path("src/hstu_kvcache/streaming/qk_stream_runner.py"),
    Path("src/hstu_kvcache/streaming/qk_full_catalog_runner.py"),
    Path("src/hstu_kvcache/streaming/qk_alignment_runner.py"),
    Path("src/hstu_kvcache/streaming/qk_protocol_sweep_runner.py"),
    Path("scripts/materialize_evokv_qk_theta2_route_a_sweep.py"),
    Path("scripts/materialize_evokv_qk_theta2_evaluations.py"),
    Path("scripts/train_evokv_qk_theta2_route_a_e3_lr100.py"),
    Path("scripts/train_evokv_qk_theta2_route_a_e4_lr100.py"),
    Path("scripts/train_evokv_qk_theta2_route_a_e3_lr150.py"),
    Path("scripts/evaluate_evokv_qk_theta1_stream_alignment.py"),
    Path("scripts/evaluate_evokv_qk_theta1_candidate_protocol_sweep.py"),
    Path("scripts/validate_evokv_qk_full_catalog_tuning.py"),
    Path("scripts/validate_evokv_qk_theta1_stream_alignment.py"),
    Path("scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py"),
    Path("scripts/summarize_evokv_qk_theta2_candidate.py"),
    Path("scripts/validate_evokv_qk_theta2_candidate_summary.py"),
    Path("scripts/summarize_evokv_qk_theta2_route_a_sweep.py"),
    Path("scripts/validate_evokv_qk_theta2_route_a_sweep.py"),
    Path("scripts/run_evokv_qk_theta2_route_a_sweep.sh"),
]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()

payload = "".join(f"{digest(path)}  {path}\n" for path in paths)
if output.is_file():
    if output.read_text() != payload:
        raise RuntimeError("theta2 sweep inputs changed after freeze")
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(output)
print(payload, end="")
PY

for index in "${!CANDIDATES[@]}"; do
    ROUND_ROOT="results/foundation_model/qk_theta2/${ROUND_IDS[$index]}"
    CONFIG="$ROUND_ROOT/frozen_training_config.json"
    mkdir -p "$ROUND_ROOT/training" "$ROUND_ROOT/alignment" "$ROUND_ROOT/protocol_sweep"
    python scripts/validate_evokv_qk_full_catalog_tuning.py \
        --config "$CONFIG" --corpus-only \
        2>&1 | tee -a "$ROUND_ROOT/training/validation.log"
done

if [[ "${EVOKV_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "QK theta2 route-A sweep preflight completed"
    exit 0
fi

for index in "${!CANDIDATES[@]}"; do
    CANDIDATE="${CANDIDATES[$index]}"
    ROUND_ROOT="results/foundation_model/qk_theta2/${ROUND_IDS[$index]}"
    CONFIG="$ROUND_ROOT/frozen_training_config.json"
    TRAINING_RESULT="$ROUND_ROOT/training/result.json"
    ALIGNMENT_CONFIG="$ROUND_ROOT/alignment/frozen_config.json"
    ALIGNMENT_RESULT="$ROUND_ROOT/alignment/result.json"
    PROTOCOL_CONFIG="$ROUND_ROOT/protocol_sweep/frozen_config.json"
    PROTOCOL_RESULT="$ROUND_ROOT/protocol_sweep/result.json"

    if [[ -f "$TRAINING_RESULT" ]]; then
        python scripts/validate_evokv_qk_full_catalog_tuning.py \
            --config "$CONFIG" \
            2>&1 | tee -a "$ROUND_ROOT/training/validation.log"
    else
        torchrun --standalone --nproc_per_node=2 \
            "${ENTRIES[$index]}" --config "$CONFIG" \
            2>&1 | tee -a "$ROUND_ROOT/training/training.log"
        python scripts/summarize_evokv_qk_reuse_recompute.py \
            --config "$CONFIG" \
            2>&1 | tee -a "$ROUND_ROOT/training/summary.log"
        python scripts/validate_evokv_qk_full_catalog_tuning.py \
            --config "$CONFIG" \
            2>&1 | tee -a "$ROUND_ROOT/training/validation.log"
    fi

    python scripts/materialize_evokv_qk_theta2_evaluations.py \
        --training-config "$CONFIG" \
        2>&1 | tee -a "$ROUND_ROOT/materialize_evaluations.log"

    python scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py \
        --config "$PROTOCOL_CONFIG" --inputs-only \
        2>&1 | tee -a "$ROUND_ROOT/protocol_sweep/validation.log"
    if [[ -f "$PROTOCOL_RESULT" ]]; then
        python scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py \
            --config "$PROTOCOL_CONFIG" \
            2>&1 | tee -a "$ROUND_ROOT/protocol_sweep/validation.log"
    else
        torchrun --standalone --nproc_per_node=2 \
            scripts/evaluate_evokv_qk_theta1_candidate_protocol_sweep.py \
            --config "$PROTOCOL_CONFIG" \
            2>&1 | tee -a "$ROUND_ROOT/protocol_sweep/evaluation.log"
        python scripts/validate_evokv_qk_theta1_candidate_protocol_sweep.py \
            --config "$PROTOCOL_CONFIG" \
            2>&1 | tee -a "$ROUND_ROOT/protocol_sweep/validation.log"
    fi

    python scripts/validate_evokv_qk_theta1_stream_alignment.py \
        --config "$ALIGNMENT_CONFIG" --inputs-only \
        2>&1 | tee -a "$ROUND_ROOT/alignment/validation.log"
    if [[ -f "$ALIGNMENT_RESULT" ]]; then
        python scripts/validate_evokv_qk_theta1_stream_alignment.py \
            --config "$ALIGNMENT_CONFIG" \
            2>&1 | tee -a "$ROUND_ROOT/alignment/validation.log"
    else
        torchrun --standalone --nproc_per_node=2 \
            scripts/evaluate_evokv_qk_theta1_stream_alignment.py \
            --config "$ALIGNMENT_CONFIG" \
            2>&1 | tee -a "$ROUND_ROOT/alignment/evaluation.log"
        python scripts/validate_evokv_qk_theta1_stream_alignment.py \
            --config "$ALIGNMENT_CONFIG" \
            2>&1 | tee -a "$ROUND_ROOT/alignment/validation.log"
    fi

    python scripts/summarize_evokv_qk_theta2_candidate.py \
        --config "$CONFIG" \
        2>&1 | tee -a "$ROUND_ROOT/summary.log"
    python scripts/validate_evokv_qk_theta2_candidate_summary.py \
        --config "$CONFIG" \
        2>&1 | tee -a "$ROUND_ROOT/validation.log"
    echo "QK theta2 candidate completed: $CANDIDATE"
done

python scripts/summarize_evokv_qk_theta2_route_a_sweep.py \
    --plan "$PLAN" 2>&1 | tee -a "$SUMMARY_LOG"
python scripts/validate_evokv_qk_theta2_route_a_sweep.py \
    --plan "$PLAN" 2>&1 | tee -a "$SUMMARY_LOG"

python - "$PLAN" "$RETURN_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
output = Path(sys.argv[2])
plan = json.loads(plan_path.read_text())
result_parent = Path(plan["outputs"]["result_parent"])

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()

candidates = []
for candidate in plan["candidates"]:
    round_root = result_parent / candidate["round_id"]
    config = json.loads((round_root / "frozen_training_config.json").read_text())
    checkpoint = Path(config["outputs"]["work_checkpoint_root"]) / "theta_2" / "manifest.json"
    summary = round_root / "summary.json"
    candidates.append(
        {
            "candidate": candidate["candidate_name"],
            "summary": {"path": str(summary), "sha256": digest(summary)},
            "checkpoint_manifest": {
                "path": str(checkpoint),
                "sha256": digest(checkpoint),
            },
        }
    )
summary = Path(plan["outputs"]["summary_json"])
summary_markdown = Path(plan["outputs"]["summary_markdown"])
manifest = {
    "status": "complete_development_measurement",
    "round_id": plan["round_id"],
    "selection_deferred": True,
    "qualification_consumed": False,
    "final_consumed": False,
    "artifacts_to_return": {
        "sweep_summary": {"path": str(summary), "sha256": digest(summary)},
        "sweep_summary_markdown": {
            "path": str(summary_markdown),
            "sha256": digest(summary_markdown),
        },
        "candidates": candidates,
    },
}
payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
if output.is_file() and output.read_text() != payload:
    raise FileExistsError("theta2 return manifest differs")
output.write_text(payload)
(output.parent / "execution_complete.json").write_text(payload)
print(payload, end="")
PY

echo "QK theta2 route-A sweep completed: $RETURN_MANIFEST"
