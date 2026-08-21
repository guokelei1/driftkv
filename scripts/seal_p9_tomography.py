#!/usr/bin/env python3
"""Seal the complete 24-cell P9.2 diagnostic raw matrix before aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import train_p7_theta0 as p7

import run_p9_tomography as ledger

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_tomography_contract_v1.yaml"
EVIDENCE = ROOT / "results/p9/p8_evidence_seal_v1.json"
OUTPUT = ROOT / "results/p9/p9_2_tomography_raw_seal_v1.json"
EXPECTED_ACTIONS = [
    "layer_0", "layer_1", "layer_2", "layer_3", "oldest_half",
    "middle", "recent_128", "recent_32", "recent_8", "recent_1",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P9.2 raw seal: {args.output}")
    cells = []
    for job in ledger.jobs():
        manifest_path = job.output / "raw_manifest.json"
        raw_path = job.output / "F_fidelity_tomography.parquet"
        if not manifest_path.exists() or not raw_path.exists():
            raise FileNotFoundError(f"incomplete P9.2 cell: {job}")
        manifest = json.loads(manifest_path.read_text())
        identity = (manifest["release"], manifest["model"], int(manifest["seed"]))
        if identity != (job.release, job.model, job.seed):
            raise RuntimeError(f"P9.2 cell identity mismatch: {job} vs {identity}")
        if manifest["actions"] != EXPECTED_ACTIONS:
            raise RuntimeError(f"P9.2 action list changed: {job}")
        if manifest["diagnostic_not_executable_action"] is not True:
            raise RuntimeError(f"diagnostic splice lost warning: {job}")
        raw_hash = p7.sha256_file(raw_path)
        if raw_hash != manifest["raw_sha256"]:
            raise RuntimeError(f"P9.2 raw hash changed: {raw_path}")
        if manifest["max_full_baseline_abs_delta"] > 1e-5 or manifest["max_reuse_baseline_abs_delta"] > 1e-5:
            raise RuntimeError(f"P9.2 baseline invariant failed: {job}")
        metadata = pq.read_metadata(raw_path)
        expected_rows = int(manifest["requests"]) * len(EXPECTED_ACTIONS)
        if metadata.num_rows != expected_rows:
            raise RuntimeError(f"P9.2 row conservation failed: {job}")
        cells.append({
            "release": job.release, "model": job.model, "seed": job.seed,
            "requests": int(manifest["requests"]), "rows": metadata.num_rows,
            "manifest": str(manifest_path.relative_to(ROOT)), "manifest_sha256": p7.sha256_file(manifest_path),
            "raw": str(raw_path.relative_to(ROOT)), "raw_sha256": raw_hash,
            "source_p8_raw_sha256": manifest["source_p8_raw_hash"],
            "checkpoint_sha256": manifest["checkpoint_hash"], "parent_checkpoint_sha256": manifest["parent_checkpoint_hash"],
            "max_full_baseline_abs_delta": manifest["max_full_baseline_abs_delta"],
            "max_reuse_baseline_abs_delta": manifest["max_reuse_baseline_abs_delta"],
        })
    payload = {
        "status": "P9_2_all_24_diagnostic_raw_cells_sealed_before_aggregation",
        "contract_hash": p7.sha256_file(CONTRACT), "p8_evidence_seal_hash": p7.sha256_file(EVIDENCE),
        "cells": cells, "cell_count": len(cells), "actions": EXPECTED_ACTIONS,
        "diagnostic_not_executable_action": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
