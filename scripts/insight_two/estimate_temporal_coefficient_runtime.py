#!/usr/bin/env python3
"""Seal the 512-user runtime estimate for the temporal-coordinate diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_temporal_coefficient_v1"
)
CANARY = BASE / "canary/summary.json"
PERSISTENCE_ESTIMATE = BASE.parent / "diagnostic_temporal_persistence_v1/resource_estimate.json"
CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_temporal_coefficient_v1.yaml"
)
OUTPUT = BASE / "resource_estimate.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    persistence = json.loads(PERSISTENCE_ESTIMATE.read_text(encoding="utf-8"))
    ratio = max(
        float(persistence["user_ratio"]),
        float(persistence["global_request_ratio"]),
        float(persistence["maximum_rank_request_ratio"]),
    )
    seconds = float(canary["elapsed_seconds"]) * ratio
    payload = {
        "status": "temporal_coordinate_discovery_resource_estimate",
        "contract_sha256": sha256(CONTRACT),
        "canary_elapsed_seconds": canary["elapsed_seconds"],
        "conservative_scale_ratio": ratio,
        "estimated_512_user_seconds": seconds,
        "estimated_512_user_minutes": seconds / 60,
        "canary_peak_allocated_mib": canary["peak_allocated_mib"],
        "canary_peak_reserved_mib": canary["peak_reserved_mib"],
        "under_30_minutes": seconds <= 30 * 60,
        "request_count_source": str(PERSISTENCE_ESTIMATE.relative_to(ROOT)),
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
