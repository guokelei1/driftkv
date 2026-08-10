#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PLAN="configs/evokv_foundation/qk_theta2_update_relevance_two_gpu_v0.json"
ROUND_ROOT="results/foundation_model/qk_theta2/qk_theta2_update_relevance_20260806_round1"
PREFLIGHT="$ROUND_ROOT/preflight.json"
INPUT_HASHES="$ROUND_ROOT/input_hashes.tsv"
MATERIALIZE_LOG="$ROUND_ROOT/materialize.log"
RETURN_MANIFEST="$ROUND_ROOT/return_manifest.json"

EXISTING=(
    "theta2_route_a_e3_lr100"
    "theta2_route_a_e4_lr100"
    "theta2_route_a_e3_lr150"
)
FALLBACK=(
    "theta2_relevance_e1_lr100_n8"
    "theta2_relevance_e2_lr075_n8"
    "theta2_relevance_e2_lr075_n32"
)
ENTRIES=(
    "scripts/train_evokv_qk_theta2_relevance_e1_lr100_n8.py"
    "scripts/train_evokv_qk_theta2_relevance_e2_lr075_n8.py"
    "scripts/train_evokv_qk_theta2_relevance_e2_lr075_n32.py"
)

mkdir -p "$ROUND_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python scripts/materialize_evokv_qk_theta2_update_relevance.py \
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
for candidate in plan["existing_candidates"]:
    required.extend(
        (
            Path(candidate["checkpoint_root"]) / "theta_2" / "manifest.json",
            Path(candidate["training_result"]),
            Path(candidate["previous_alignment_result"]),
        )
    )
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"QK update relevance inputs are absent: {missing}")
if (
    digest(source_root / "manifest.json") != source["manifest_sha256"]
    or digest(source_root / "training_state.json") != source["training_state_sha256"]
    or digest(source_root / "optimizer_resume.pt") != source["optimizer_resume_sha256"]
):
    raise RuntimeError("selected theta1 input differs")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("QK update relevance round requires exactly GPU0/GPU1")
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
    "existing_candidates": len(plan["existing_candidates"]),
    "fallback_candidates": len(plan["fallback_training"]["candidates"]),
    "estimated_wall_minutes_existing_only": plan["execution"]["estimated_wall_minutes_existing_only"],
    "estimated_wall_minutes_with_fallback": plan["execution"]["estimated_wall_minutes_with_fallback"],
    "qualification_consumed": False,
    "final_consumed": False,
}
output.parent.mkdir(parents=True, exist_ok=True)
payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(payload)
temporary.replace(output)
print(payload, end="")
PY

python - "$PLAN" "$INPUT_HASHES" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
output = Path(sys.argv[2])
plan = json.loads(plan_path.read_text())
paths = [
    plan_path,
    Path("docs/01-1_qk_theta0_theta1_branch_exploration.md"),
    Path(plan["data"]["config"]),
    Path(plan["data"]["roles"]),
    Path(plan["data"]["corpus"]),
    Path(plan["data"]["summary"]),
    Path(plan["source_checkpoint"]["root"]) / "theta_1" / "manifest.json",
    Path(plan["source_checkpoint"]["root"]) / "theta_1" / "training_state.json",
    Path(plan["source_checkpoint"]["root"]) / "theta_1" / "optimizer_resume.pt",
    Path("src/hstu_kvcache/streaming/qk_stream_version.py"),
    Path("src/hstu_kvcache/streaming/qk_stream_runner.py"),
    Path("src/hstu_kvcache/streaming/qk_full_catalog_runner.py"),
    Path("src/hstu_kvcache/streaming/qk_protocol_sweep_runner.py"),
    Path("src/hstu_kvcache/streaming/qk_update_relevance_runner.py"),
    Path("scripts/materialize_evokv_qk_theta2_update_relevance.py"),
    Path("scripts/evaluate_evokv_qk_theta2_update_relevance.py"),
    Path("scripts/validate_evokv_qk_theta2_update_relevance.py"),
    Path("scripts/summarize_evokv_qk_theta2_update_relevance.py"),
    Path("scripts/validate_evokv_qk_full_catalog_tuning.py"),
    Path("scripts/summarize_evokv_qk_reuse_recompute.py"),
    Path("scripts/run_evokv_qk_theta2_update_relevance.sh"),
]
for candidate in plan["existing_candidates"]:
    paths.extend(
        (
            Path(candidate["checkpoint_root"]) / "theta_2" / "manifest.json",
            Path(candidate["training_result"]),
            Path(candidate["previous_alignment_result"]),
        )
    )
for candidate in plan["fallback_training"]["candidates"]:
    paths.append(Path(candidate["entry"]))

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()

payload = "".join(f"{digest(path)}  {path}\n" for path in paths)
if output.is_file() and output.read_text() != payload:
    raise RuntimeError("QK update relevance round inputs changed after freeze")
if not output.is_file():
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(output)
print(payload, end="")
PY

if [[ "${EVOKV_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "QK theta2 update-relevance preflight completed"
    exit 0
fi

run_relevance() {
    local candidate="$1"
    local root="$ROUND_ROOT/candidates/$candidate/relevance"
    local config="$root/frozen_config.json"
    local result="$root/result.json"
    mkdir -p "$root"
    python scripts/validate_evokv_qk_theta2_update_relevance.py \
        --config "$config" --inputs-only 2>&1 | tee -a "$root/validation.log"
    if [[ -f "$result" ]]; then
        python scripts/validate_evokv_qk_theta2_update_relevance.py \
            --config "$config" 2>&1 | tee -a "$root/validation.log"
    else
        torchrun --standalone --nproc_per_node=2 \
            scripts/evaluate_evokv_qk_theta2_update_relevance.py \
            --config "$config" 2>&1 | tee -a "$root/evaluation.log"
        python scripts/validate_evokv_qk_theta2_update_relevance.py \
            --config "$config" 2>&1 | tee -a "$root/validation.log"
    fi
}

for candidate in "${EXISTING[@]}"; do
    run_relevance "$candidate"
done

python scripts/summarize_evokv_qk_theta2_update_relevance.py \
    --plan "$PLAN" --phase existing 2>&1 | tee -a "$ROUND_ROOT/summary.log"

RUN_FALLBACK="$(python - "$ROUND_ROOT/existing_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
print("0" if summary["admitted_candidates"] else "1")
PY
)"

FINAL_SUMMARY="$ROUND_ROOT/existing_summary.json"
FALLBACK_EXECUTED=false
if [[ "$RUN_FALLBACK" == "1" ]]; then
    FALLBACK_EXECUTED=true
    for index in "${!FALLBACK[@]}"; do
        candidate="${FALLBACK[$index]}"
        root="$ROUND_ROOT/candidates/$candidate"
        training_config="$root/training/frozen_config.json"
        training_result="$root/training/result.json"
        mkdir -p "$root/training"
        python scripts/validate_evokv_qk_full_catalog_tuning.py \
            --config "$training_config" --corpus-only \
            2>&1 | tee -a "$root/training/validation.log"
        if [[ -f "$training_result" ]]; then
            python scripts/validate_evokv_qk_full_catalog_tuning.py \
                --config "$training_config" \
                2>&1 | tee -a "$root/training/validation.log"
        else
            torchrun --standalone --nproc_per_node=2 \
                "${ENTRIES[$index]}" --config "$training_config" \
                2>&1 | tee -a "$root/training/training.log"
            python scripts/summarize_evokv_qk_reuse_recompute.py \
                --config "$training_config" \
                2>&1 | tee -a "$root/training/summary.log"
            python scripts/validate_evokv_qk_full_catalog_tuning.py \
                --config "$training_config" \
                2>&1 | tee -a "$root/training/validation.log"
        fi
        python scripts/materialize_evokv_qk_theta2_update_relevance.py \
            --plan "$PLAN" --candidate "$candidate" \
            2>&1 | tee -a "$root/materialize.log"
        run_relevance "$candidate"
    done
    python scripts/summarize_evokv_qk_theta2_update_relevance.py \
        --plan "$PLAN" --phase complete 2>&1 | tee -a "$ROUND_ROOT/summary.log"
    FINAL_SUMMARY="$ROUND_ROOT/summary.json"
fi

python - "$PLAN" "$FINAL_SUMMARY" "$FALLBACK_EXECUTED" "$RETURN_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
fallback_executed = sys.argv[3].lower() == "true"
output = Path(sys.argv[4])
plan = json.loads(plan_path.read_text())
summary = json.loads(summary_path.read_text())
round_root = Path(plan["outputs"]["round_root"])

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()

candidates = []
for value in summary["candidates"]:
    name = value["candidate"]
    result = round_root / "candidates" / name / "relevance" / "result.json"
    candidates.append(
        {
            "candidate": name,
            "relevance_result": {"path": str(result), "sha256": digest(result)},
            "admitted_relations": value["admitted_relations"],
        }
    )
manifest = {
    "status": "complete_development_measurement",
    "round_id": plan["round_id"],
    "fallback_executed": fallback_executed,
    "selection_deferred": True,
    "qualification_consumed": False,
    "final_consumed": False,
    "artifacts_to_return": {
        "summary": {"path": str(summary_path), "sha256": digest(summary_path)},
        "candidates": candidates,
    },
}
payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
if output.is_file() and output.read_text() != payload:
    raise FileExistsError("QK update relevance return manifest differs")
output.write_text(payload)
(output.parent / "execution_complete.json").write_text(payload)
print(payload, end="")
PY

echo "QK theta2 update-relevance round completed: $RETURN_MANIFEST"
