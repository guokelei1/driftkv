#!/usr/bin/env python3
"""Seal a conservative 512-user runtime estimate from the cone canary."""

from __future__ import annotations

import json
from pathlib import Path

from insight_two.common import sha256_file


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_attention_cone_moments_v1"
)
CANARY = BASE / "canary/summary.json"
ANALYSIS = BASE / "canary/analysis/summary.json"
CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_attention_cone_moments_v1.yaml"
)
OUTPUT = BASE / "resource_estimate.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    run = json.loads(CANARY.read_text(encoding="utf-8"))
    gate = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    if not run.get("passed") or not gate.get("discovery_launch_gate_passed"):
        raise RuntimeError("attention-cone canary did not unlock discovery")
    ratio = 512 / 32
    seconds = float(run["elapsed_seconds"]) * ratio
    payload = {
        "status": "attention_cone_moment_discovery_resource_estimate",
        "contract_sha256": sha256_file(CONTRACT),
        "canary_elapsed_seconds": run["elapsed_seconds"],
        "conservative_user_ratio": ratio,
        "estimated_512_user_seconds": seconds,
        "estimated_512_user_minutes": seconds / 60,
        "canary_peak_allocated_mib": run["peak_allocated_mib"],
        "canary_peak_reserved_mib": run["peak_reserved_mib"],
        "under_30_minutes": seconds <= 30 * 60,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
