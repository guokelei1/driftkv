#!/usr/bin/env python3
"""Seal all P9.9 raw logits before any quality adjudication."""

import json
from pathlib import Path

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/p9/heldout_rolling_quality_raw/full"
CONTRACT = ROOT / "configs/contracts/p9_9_heldout_rolling_quality_contract_v1.yaml"
EVALUATOR = ROOT / "scripts/eval_p9_heldout_rolling_quality_raw.py"
OUTPUT = ROOT / "results/p9/p9_9_heldout_rolling_quality_raw_seal_v1.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    artifacts = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            for seed in (17, 37, 71):
                manifest_path = RAW / release / f"{model}_seed{seed}" / "raw_manifest.json"
                if not manifest_path.exists():
                    raise RuntimeError(f"missing P9.9 raw cell: {release}/{model}/{seed}")
                manifest = json.loads(manifest_path.read_text())
                if manifest["status"] != "passed" or manifest["scope"] != "full" or manifest["metrics_computed"]:
                    raise RuntimeError(f"invalid P9.9 raw cell: {manifest_path}")
                raw_path = ROOT / manifest["raw_path"]
                if p7.sha256_file(raw_path) != manifest["raw_sha256"]:
                    raise RuntimeError(f"P9.9 raw hash mismatch: {raw_path}")
                artifacts.append(manifest)
    payload = {
        "status": "P9_9_all_24_heldout_rolling_quality_raw_cells_sealed_before_metrics",
        "contract_sha256": p7.sha256_file(CONTRACT),
        "evaluator_sha256": p7.sha256_file(EVALUATOR),
        "cells": len(artifacts),
        "requests": sum(row["requests"] for row in artifacts),
        "request_action_rows": sum(row["requests"] * row["actions"] for row in artifacts),
        "artifacts": artifacts,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "cells", "requests", "request_action_rows")}, indent=2))


if __name__ == "__main__":
    main()
