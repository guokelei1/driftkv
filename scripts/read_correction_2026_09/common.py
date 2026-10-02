"""Shared locations and compact provenance for the two correction experiments."""
from __future__ import annotations

import json
from pathlib import Path

from selective_recompute_2026_09.common import edge_name, sha256, signature, write_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/read_correction_2026_09"
PANEL_ROOT = ROOT / "results/selective_recompute_2026_09/panels"
PLAN = ROOT / "configs/read_correction_2026_09/plan.json"
METHODS = ("query_only", "history_conditioned")


def plan():
    value = json.loads(PLAN.read_text())
    for path in value["method_settings"].values():
        value.update(json.loads((ROOT / path).read_text()))
    return value


def sources():
    paths = [PLAN]
    paths.extend(ROOT / path for path in json.loads(PLAN.read_text())["method_settings"].values())
    for directory in (ROOT / "scripts/read_correction_2026_09",
                      ROOT / "src/hstu_kvcache/read_correction",
                      ROOT / "src/hstu_kvcache/models"):
        paths.extend(sorted(directory.rglob("*.py")))
    for name in (
        "scripts/evaluate_yambda500m_foundation_raw.py",
        "scripts/selective_recompute_2026_09/common.py",
        "scripts/selective_recompute_2026_09/calibrate.py",
        "scripts/selective_recompute_2026_09/evaluate.py",
        "scripts/selective_recompute_2026_09/scheduling.py",
        "scripts/selective_recompute_2026_09/cost.py",
        "src/hstu_kvcache/baselines/layer_recompute/core.py",
        "src/hstu_kvcache/baselines/serving.py",
        "src/hstu_kvcache/adaptation/reader.py",
        "src/hstu_kvcache/adaptation/summary.py",
        "src/hstu_kvcache/data/yambda_history.py",
        "src/hstu_kvcache/training/foundation.py",
    ):
        paths.append(ROOT / name)
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(paths))}


def method_directory(method, scale, edge, *, output_root=OUTPUT, revision=None):
    return Path(output_root) / method / "development" / (revision or plan()["revision"]) / scale / edge_name(edge)
