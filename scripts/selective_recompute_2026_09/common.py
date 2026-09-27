"""Small shared records; experiment state is separate from retained Reuse."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/selective_recompute_2026_09/plan.json"
OUTPUT = ROOT / "results/selective_recompute_2026_09"
METHODS = ("layer", "tail", "deviation", "query")
DAY = 86400


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def plan():
    return json.loads(PLAN.read_text())


def edge_name(edge):
    return f"v{edge - 1}_to_v{edge}"


def budgets(layers):
    config = plan()
    counts = sorted({min(layers - 1, max(1, math.ceil(layers * f)))
                     for f in config["layer_fractions"]})
    return {
        "layer": [{"name": f"layers_{n}", "layers": n} for n in counts],
        **{method: [{"name": f"fraction_{f:g}", "fraction": f}
                    for f in config["token_fractions"]]
           for method in ("tail", "deviation", "query")},
    }


def sources():
    paths = [PLAN]
    for directory in (ROOT / "scripts/selective_recompute_2026_09",
                      ROOT / "src/hstu_kvcache/baselines", ROOT / "src/hstu_kvcache/models"):
        paths.extend(sorted(directory.rglob("*.py")))
    paths += [ROOT / "scripts/evaluate_yambda500m_foundation_raw.py",
              ROOT / "src/hstu_kvcache/data/yambda_history.py",
              ROOT / "src/hstu_kvcache/training/foundation.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
