"""Inputs and records for the isolated v4 read-time history experiment."""
import json
from pathlib import Path

from read_correction_2026_09.v2.common import PANEL_ROOT, RESERVATIONS, edge_name, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/read_correction_2026_09/v4"


def configuration():
    return {
        "revision": "v4_affine_probe", "evaluation_role": "development_exploration",
        "authorization": "2026-09-29 user: continue one iteration; small Medium/Large probes then all15edges if supported; Max not used for quality tuning",
        "calibration_users": 128, "validation_users": 16, "tokens_per_user": 128,
        "calibration_queries_per_user": 16, "ridge": .01, "seed": 17,
        "history_length": 1024, "attention_backend": "triton",
        "capture_batches": {"medium": 32, "large": 16, "max": 8},
        "cohort_sizes": {"medium": 128, "large": 64, "max": 32},
        "query_batches": {"medium": 128, "large": 64, "max": 32},
        "torch_threads": 4, "history_threads": 4,
        "memory_fraction": .70, "initial_free_fraction": .75, "append_band_size": 32,
        "development_edges": [["medium", 1], ["medium", 4], ["large", 1], ["large", 2]],
        "quality_stop_threshold": None,
    }


def sources():
    from read_correction_2026_09.v2.common import sources as prior_sources
    result = prior_sources()
    for folder in (ROOT / "scripts/read_correction_v4", ROOT / "src/hstu_kvcache/read_correction_v4"):
        for path in sorted(folder.rglob("*.py")):
            result[str(path.relative_to(ROOT))] = sha256(path)
    # The fixed Full-minus-Reuse cost arithmetic is shared with the v3 evaluator.
    for name in ("evaluate.py", "common.py"):
        path = ROOT / "scripts/read_correction_v3" / name
        result[str(path.relative_to(ROOT))] = sha256(path)
    return result
