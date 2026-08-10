#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_CONFIG="configs/evokv_foundation/qk_theta0_next_item_two_gpu_v0.json"
ROUND_ROOT="results/foundation_model/qk_theta0/qk_theta0_next_item_20260805_round1"
FROZEN_CONFIG="$ROUND_ROOT/frozen_config.json"
CORPUS_LOG="$ROUND_ROOT/corpus_build.log"
TRAINING_LOG="$ROUND_ROOT/training.log"
VALIDATION_LOG="$ROUND_ROOT/validation.log"
PREFLIGHT="$ROUND_ROOT/preflight.json"
INPUT_HASHES="$ROUND_ROOT/input_hashes.tsv"
RESULT="$ROUND_ROOT/result.json"

mkdir -p "$ROUND_ROOT"
if [[ -f "$FROZEN_CONFIG" ]]; then
    cmp --silent "$SOURCE_CONFIG" "$FROZEN_CONFIG" || {
        echo "frozen config differs from source config" >&2
        exit 1
    }
else
    cp "$SOURCE_CONFIG" "$FROZEN_CONFIG"
fi

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ -f "$RESULT" ]]; then
    python scripts/validate_evokv_qk_theta0.py --config "$FROZEN_CONFIG" | tee -a "$VALIDATION_LOG"
    echo "QK theta0 round is already complete and valid"
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
    Path(config["data"]["source"]),
    Path(config["data"]["catalog"]),
    Path(config["data"]["user_lengths"]),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"required QK theta0 inputs are absent: {missing}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("QK theta0 round requires exactly GPU0/GPU1 to be visible")
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
minimum_disk = 150 * (1 << 30)
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
    "data/tenrec/Tenrec.zip" \
    "data/processed/evokv_d3_m1_qk_entity_cache/entity_catalog_base64_top250000.npz" \
    "data/processed/evokv_foundation/qk_full_user_lengths.npz" \
    "src/hstu_kvcache/data/qk_theta0.py" \
    "src/hstu_kvcache/streaming/xp_projected_edge.py" \
    "src/hstu_kvcache/streaming/xp_version_training.py" \
    "scripts/build_evokv_qk_theta0_corpus.py" \
    "scripts/train_evokv_qk_theta0.py" \
    "scripts/validate_evokv_qk_theta0.py" \
    "scripts/run_evokv_qk_theta0_round.sh" > "$INPUT_HASHES"

if [[ "${EVOKV_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "QK theta0 handoff preflight completed"
    exit 0
fi

CORPUS="data/processed/evokv_foundation/qk_theta0_next_item_corpus_v0.npz"
if [[ ! -f "$CORPUS" ]]; then
    python scripts/build_evokv_qk_theta0_corpus.py --config "$FROZEN_CONFIG" 2>&1 | tee -a "$CORPUS_LOG"
fi
python scripts/validate_evokv_qk_theta0.py --config "$FROZEN_CONFIG" --corpus-only 2>&1 | tee -a "$VALIDATION_LOG"

python - "$FROZEN_CONFIG" <<'PY'
import json
import math
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
summary = json.loads(Path(config["data"]["corpus_summary"]).read_text())
records = int(summary["records"])
tokens = int(summary["tokens"])
maximum_users = int(config["data"]["maximum_selected_users_for_round"])
maximum_tokens = int(config["data"]["maximum_selected_tokens_for_round"])
if records > maximum_users or tokens > maximum_tokens:
    raise RuntimeError(
        f"prepared corpus exceeds frozen round budget: records={records}/{maximum_users}, "
        f"tokens={tokens}/{maximum_tokens}"
    )
world_size = int(config["execution"]["world_size"])
batch = int(config["execution"]["batch_size_per_rank"])
steps = math.ceil(records / (world_size * batch))
print(json.dumps({
    "status": "admitted",
    "records": records,
    "tokens": tokens,
    "training_steps": steps,
    "maximum_records": maximum_users,
    "maximum_tokens": maximum_tokens,
}, indent=2, sort_keys=True))
PY

torchrun --standalone --nproc_per_node=2 \
    scripts/train_evokv_qk_theta0.py \
    --config "$FROZEN_CONFIG" 2>&1 | tee -a "$TRAINING_LOG"

python scripts/validate_evokv_qk_theta0.py --config "$FROZEN_CONFIG" 2>&1 | tee -a "$VALIDATION_LOG"

python - "$FROZEN_CONFIG" "$ROUND_ROOT" <<'PY'
import hashlib
import json
import shutil
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

result = Path(config["outputs"]["result"])
checkpoint = Path(config["outputs"]["checkpoint_root"]) / "theta_0" / "manifest.json"
summary = Path(config["data"]["corpus_summary"])
return_manifest = {
    "status": "complete",
    "round_id": config["round_id"],
    "artifacts_to_return": {
        "result": {"path": str(result), "sha256": digest(result)},
        "checkpoint_manifest": {"path": str(checkpoint), "sha256": digest(checkpoint)},
        "corpus_summary": {"path": str(summary), "sha256": digest(summary)},
        "frozen_config": {"path": str(config_path), "sha256": digest(config_path)},
        "training_log": {"path": str(round_root / "training.log"), "sha256": digest(round_root / "training.log")},
        "validation_log": {"path": str(round_root / "validation.log"), "sha256": digest(round_root / "validation.log")},
    },
}
(round_root / "return_manifest.json").write_text(
    json.dumps(return_manifest, indent=2, sort_keys=True) + "\n"
)
(round_root / "execution_complete.json").write_text(
    json.dumps({"status": "complete", "round_id": config["round_id"]}, indent=2, sort_keys=True) + "\n"
)
cache = Path(config["data"]["builder_cache"])
expected = (Path.cwd() / "data/processed/evokv_foundation/qk_theta0_builder_cache_v0").resolve()
if cache.resolve() != expected:
    raise RuntimeError("QK theta0 builder cache cleanup target differs")
if cache.is_dir():
    shutil.rmtree(cache)
print(json.dumps(return_manifest, indent=2, sort_keys=True))
PY
