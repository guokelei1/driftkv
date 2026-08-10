#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PLAN="configs/evokv_foundation/qk_theta2_core_heavy_two_gpu_v0.json"
ROUND_ROOT="results/foundation_model/qk_theta2/qk_theta2_core_heavy_20260807_round1"
PREFLIGHT="$ROUND_ROOT/preflight.json"
INPUT_HASHES="$ROUND_ROOT/input_hashes.tsv"
MATERIALIZE_LOG="$ROUND_ROOT/materialize.log"
RETURN_MANIFEST="$ROUND_ROOT/return_manifest.json"

CANDIDATES=(
    "theta2_core_d150_p100_e025_n32"
    "theta2_core_d200_p100_e025_n32"
    "theta2_core_d150_p150_e025_n32"
    "theta2_core_d150_p100_e050_n32"
)
ENTRIES=(
    "scripts/train_evokv_qk_theta2_core_d150_p100_e025_n32.py"
    "scripts/train_evokv_qk_theta2_core_d200_p100_e025_n32.py"
    "scripts/train_evokv_qk_theta2_core_d150_p150_e025_n32.py"
    "scripts/train_evokv_qk_theta2_core_d150_p100_e050_n32.py"
)

mkdir -p "$ROUND_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python scripts/materialize_evokv_qk_theta2_core_heavy.py \
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
anchor = plan["anchor_candidate"]
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
    Path(anchor["checkpoint_root"]) / "theta_2" / "manifest.json",
    Path(anchor["training_result"]),
    Path(anchor["relevance_result"]),
    Path(plan["data"]["config"]),
    Path(plan["data"]["roles"]),
    Path(plan["data"]["corpus"]),
    Path(plan["data"]["summary"]),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"QK core-heavy inputs are absent: {missing}")
checks = (
    (source_root / "manifest.json", source["manifest_sha256"]),
    (source_root / "training_state.json", source["training_state_sha256"]),
    (source_root / "optimizer_resume.pt", source["optimizer_resume_sha256"]),
    (
        Path(anchor["checkpoint_root"]) / "theta_2" / "manifest.json",
        anchor["manifest_sha256"],
    ),
    (Path(anchor["training_result"]), anchor["training_result_sha256"]),
    (Path(anchor["relevance_result"]), anchor["relevance_result_sha256"]),
    (Path(plan["data"]["config"]), plan["data"]["config_sha256"]),
    (Path(plan["data"]["corpus"]), plan["data"]["corpus_sha256"]),
)
if any(digest(path) != expected for path, expected in checks):
    raise RuntimeError("QK core-heavy frozen input differs")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("QK core-heavy round requires exactly GPU0/GPU1")
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
    "anchor_candidate": anchor["candidate_name"],
    "search_candidates": len(plan["search_training"]["candidates"]),
    "estimated_wall_minutes": plan["execution"]["estimated_wall_minutes"],
    "estimated_new_checkpoint_bytes": plan["execution"][
        "estimated_new_checkpoint_bytes"
    ],
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
source = plan["source_checkpoint"]
anchor = plan["anchor_candidate"]
paths = [
    plan_path,
    Path("docs/01-1_qk_theta0_theta1_branch_exploration.md"),
    Path(plan["data"]["config"]),
    Path(plan["data"]["roles"]),
    Path(plan["data"]["corpus"]),
    Path(plan["data"]["summary"]),
    Path(source["root"]) / "theta_1" / "manifest.json",
    Path(source["root"]) / "theta_1" / "training_state.json",
    Path(source["root"]) / "theta_1" / "optimizer_resume.pt",
    Path(anchor["checkpoint_root"]) / "theta_2" / "manifest.json",
    Path(anchor["training_result"]),
    Path(anchor["relevance_result"]),
    Path("src/hstu_kvcache/streaming/qk_stream_version.py"),
    Path("src/hstu_kvcache/streaming/qk_stream_runner.py"),
    Path("src/hstu_kvcache/streaming/qk_full_catalog_runner.py"),
    Path("src/hstu_kvcache/streaming/qk_protocol_sweep_runner.py"),
    Path("src/hstu_kvcache/streaming/qk_update_relevance_runner.py"),
    Path("scripts/materialize_evokv_qk_theta2_core_heavy.py"),
    Path("scripts/evaluate_evokv_qk_theta2_update_relevance.py"),
    Path("scripts/validate_evokv_qk_theta2_update_relevance.py"),
    Path("scripts/summarize_evokv_qk_theta2_core_heavy.py"),
    Path("scripts/validate_evokv_qk_full_catalog_tuning.py"),
    Path("scripts/summarize_evokv_qk_reuse_recompute.py"),
    Path("scripts/run_evokv_qk_theta2_core_heavy.sh"),
]
paths.extend(Path(value["entry"]) for value in plan["search_training"]["candidates"])


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


payload = "".join(f"{digest(path)}  {path}\n" for path in paths)
if output.is_file() and output.read_text() != payload:
    raise RuntimeError("QK core-heavy inputs changed after freeze")
if not output.is_file():
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(output)
print(payload, end="")
PY

if [[ "${EVOKV_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "QK theta2 core-heavy preflight completed"
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

for index in "${!CANDIDATES[@]}"; do
    candidate="${CANDIDATES[$index]}"
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
    python scripts/materialize_evokv_qk_theta2_core_heavy.py \
        --plan "$PLAN" --candidate "$candidate" \
        2>&1 | tee -a "$root/materialize.log"
    run_relevance "$candidate"
done

python scripts/summarize_evokv_qk_theta2_core_heavy.py \
    --plan "$PLAN" 2>&1 | tee -a "$ROUND_ROOT/summary.log"

python - "$PLAN" "$RETURN_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
output = Path(sys.argv[2])
plan = json.loads(plan_path.read_text())
round_root = Path(plan["outputs"]["round_root"])
summary_path = Path(plan["outputs"]["summary_json"])
summary = json.loads(summary_path.read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


candidates = []
for value in plan["search_training"]["candidates"]:
    name = value["candidate_name"]
    training = round_root / "candidates" / name / "training" / "result.json"
    relevance = round_root / "candidates" / name / "relevance" / "result.json"
    candidates.append(
        {
            "candidate": name,
            "training_result": {"path": str(training), "sha256": digest(training)},
            "relevance_result": {
                "path": str(relevance),
                "sha256": digest(relevance),
            },
        }
    )
manifest = {
    "status": "complete_development_measurement",
    "round_id": plan["round_id"],
    "anchor_candidate": plan["anchor_candidate"]["candidate_name"],
    "preferred_candidates": summary["preferred_candidates"],
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
    raise FileExistsError("QK core-heavy return manifest differs")
output.write_text(payload)
(output.parent / "execution_complete.json").write_text(payload)
print(payload, end="")
PY

echo "QK theta2 core-heavy round completed: $RETURN_MANIFEST"
