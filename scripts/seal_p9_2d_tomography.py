#!/usr/bin/env python3
"""Seal all 12 P9.3 two-dimensional diagnostic raw cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

import eval_p9_2d_tomography_raw as evaluator
import run_p9_2d_tomography as ledger
import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_3_2d_tomography_contract_v1.yaml"
OUTPUT = ROOT / "results/p9/p9_3_2d_tomography_raw_seal_v1.json"
EXPECTED_ACTIONS = list(evaluator.action_names_2d(4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P9.3 seal: {args.output}")
    evaluator.validate_contract()
    cells = []
    for job in ledger.jobs():
        manifest_path = job.output / "raw_manifest.json"
        raw_path = job.output / "F_fidelity_2d_tomography.parquet"
        if not manifest_path.exists() or not raw_path.exists():
            raise FileNotFoundError(f"incomplete P9.3 cell: {job}")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["release"], manifest["model"], int(manifest["seed"])) != (job.release, job.model, job.seed):
            raise RuntimeError(f"P9.3 cell identity mismatch: {job}")
        if manifest["contract_hash"] != p7.sha256_file(CONTRACT):
            raise RuntimeError(f"P9.3 contract changed for cell: {job}")
        if manifest["actions"] != EXPECTED_ACTIONS:
            raise RuntimeError(f"P9.3 action order changed: {job}")
        if manifest["diagnostic_not_executable_action"] is not True:
            raise RuntimeError(f"P9.3 diagnostic warning missing: {job}")
        if p7.sha256_file(raw_path) != manifest["raw_sha256"]:
            raise RuntimeError(f"P9.3 raw changed after write: {job}")
        rows = pq.read_metadata(raw_path).num_rows
        if rows != int(manifest["requests"]) * len(EXPECTED_ACTIONS):
            raise RuntimeError(f"P9.3 row conservation failed: {job}")
        if manifest["max_full_baseline_abs_delta"] > 1e-5 or manifest["max_reuse_baseline_abs_delta"] > 1e-5:
            raise RuntimeError(f"P9.3 baseline invariant failed: {job}")
        cells.append({
            "release": job.release, "model": job.model, "seed": job.seed,
            "requests": int(manifest["requests"]), "rows": rows,
            "manifest": str(manifest_path.relative_to(ROOT)), "manifest_sha256": p7.sha256_file(manifest_path),
            "raw": str(raw_path.relative_to(ROOT)), "raw_sha256": manifest["raw_sha256"],
            "source_p8_raw_sha256": manifest["source_p8_raw_hash"],
            "checkpoint_sha256": manifest["checkpoint_hash"],
            "parent_checkpoint_sha256": manifest["parent_checkpoint_hash"],
            "max_full_baseline_abs_delta": manifest["max_full_baseline_abs_delta"],
            "max_reuse_baseline_abs_delta": manifest["max_reuse_baseline_abs_delta"],
        })
    payload = {
        "status": "P9_3_all_12_diagnostic_2d_cells_sealed_before_metrics",
        "contract_hash": p7.sha256_file(CONTRACT), "cell_count": len(cells),
        "actions": EXPECTED_ACTIONS, "diagnostic_not_executable_action": True, "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
