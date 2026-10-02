"""Isolated third exploration, reusing the unchanged v1/v2 numerical sources."""
import json
from pathlib import Path

from read_correction_2026_09.v2.common import (
    PANEL_ROOT, RESERVATIONS, edge_name, sha256, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/read_correction_2026_09/v3"
PLAN = ROOT / "configs/read_correction_2026_09/v3/plan.json"


def plan():
    return json.loads(PLAN.read_text())


def sources():
    from read_correction_2026_09.v2.common import sources as prior_sources
    result = prior_sources()
    for path in [PLAN, *sorted(Path(__file__).parent.rglob("*.py"))]:
        result[str(path.relative_to(ROOT))] = sha256(path)
    return result


def method_directory(root, scale, edge):
    return Path(root) / "history_conditioned" / scale / edge_name(edge)
