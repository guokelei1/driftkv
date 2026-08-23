#!/usr/bin/env python3
"""Seal the complete P9.5 rolling-lineage validation matrix."""

from __future__ import annotations

import json
from pathlib import Path

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_5_rolling_validation_matrix_v1.yaml"
RAW_ROOT = ROOT / "results/p9/rolling_validation_raw"
OUTPUT = ROOT / "results/p9/p9_5_rolling_validation_raw_seal_v1.json"


def main() -> None:
    artifacts = []
    failures = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            for seed in (17, 37, 71):
                root = RAW_ROOT / release / f"{model}_seed{seed}"
                result_path = root / "result.json"
                raw_path = root / "rolling_actions.parquet"
                if not result_path.exists() or not raw_path.exists():
                    failures.append(f"missing:{release}:{model}:{seed}")
                    continue
                result = json.loads(result_path.read_text())
                if result["status"] != "passed":
                    failures.append(f"failed:{release}:{model}:{seed}")
                if result["contract_hash"] != p7.sha256_file(CONTRACT):
                    failures.append(f"contract:{release}:{model}:{seed}")
                if result["raw_sha256"] != p7.sha256_file(raw_path):
                    failures.append(f"raw_hash:{release}:{model}:{seed}")
                artifacts.append({
                    "release": release, "model": model, "seed": seed,
                    "result": str(result_path.relative_to(ROOT)),
                    "result_sha256": p7.sha256_file(result_path),
                    "raw": str(raw_path.relative_to(ROOT)),
                    "raw_sha256": p7.sha256_file(raw_path),
                    "requests": sum(row["requests"] for row in result["audits"]),
                })
    if failures or len(artifacts) != 24:
        raise RuntimeError(f"P9.5 seal refused: {failures}; artifacts={len(artifacts)}")
    payload = {
        "status": "P9_5_all_24_materialized_rolling_lineage_cells_sealed",
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": p7.sha256_file(CONTRACT),
        "cells": len(artifacts),
        "total_unique_request_evaluations": sum(row["requests"] for row in artifacts),
        "artifacts": artifacts,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "cells", "total_unique_request_evaluations")}, indent=2))


if __name__ == "__main__":
    main()
