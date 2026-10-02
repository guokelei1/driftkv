"""Shared saved-result helpers for the current read-correction figures."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

SCALES = {"medium": "Medium (6 layers)", "large": "Large (10 layers)", "max": "Max (16 layers)"}
EDGES = tuple(f"v{i}_to_v{i + 1}" for i in range(5))
REQUIRED = ("method", "scale", "edge", "budget", "relative_flops_percent", "recovery_percent")
PAIRED_INTEGERS = ("users", "requests", "full_history_flops", "reuse_append_flops", "full_minus_reuse_flops")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paired(point, reference):
    for field in PAIRED_INTEGERS:
        if point[field] != reference[field]:
            raise ValueError(f"different panel or compute denominator in {field}: {point['scale']}/{point['edge']}")
    for field in ("full_auc", "reuse_auc"):
        if not math.isclose(point[field], reference[field], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"different frozen {field}: {point['scale']}/{point['edge']}")


def write_csv(records: list[dict], destination: Path) -> None:
    fields = list(REQUIRED) + sorted(set().union(*(set(row) for row in records)).difference(REQUIRED))
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
