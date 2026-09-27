"""Shared paths and small records for the selected adjacent Reuse evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/unified_reuse_2026_09/adjacent_e14.json"
DEFAULT_OUTPUT = ROOT / "results/unified_reuse_2026_09"
DAY = 86_400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def binding(scale: str, edge: int) -> dict:
    if scale not in {"medium", "large", "max"} or edge not in range(1, 6):
        raise ValueError("scale must be medium/large/max and edge must be 1..5")
    plan = json.loads(PLAN.read_text())
    setting = plan["scales"][scale]
    start = 217 + 14 * edge
    checkpoint_dir = ROOT / "results/unified_training_2026_09" / scale / "seed17/checkpoints"
    return {
        "scale": scale, "edge": f"v{edge-1}_to_v{edge}", "edge_index": edge,
        "days": [start, start + 14], "cutover": start * DAY,
        "parent": checkpoint_dir / f"v{edge-1}/checkpoint_100.pt",
        "current": checkpoint_dir / f"v{edge}/checkpoint_100.pt",
        "full_raw": ROOT / setting["full_raw"][edge-1],
        "full_current_name": setting["full_current_names"][edge-1],
        "requests": ROOT / setting["request_manifest_dir"] / "requests_fidelity.parquet",
        "labels": ROOT / setting["request_manifest_dir"] / "requests_quality.parquet",
        "dataset": ROOT / setting["dataset_manifest"],
        "cohort_size": setting["cohort_size_per_gpu"],
        "query_chunk_size": setting["query_chunk_size"],
        "gpus": plan["gpus"], "report_interval_seconds": plan["report_interval_seconds"],
    }
