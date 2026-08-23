#!/usr/bin/env python3
"""Seal all 24 dependency-closed P9.4 executor raw cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

import eval_p9_executor_raw as evaluator
import run_p9_executor as ledger
import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_4_executor_contract_v1.yaml"
OUTPUT = ROOT / "results/p9/p9_4_executor_raw_seal_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P9.4 seal: {args.output}")
    evaluator.validate_contract()
    cells = []
    for job in ledger.jobs():
        manifest_path = job.output / "raw_manifest.json"
        raw_path = job.output / "F_fidelity_executor.parquet"
        if not manifest_path.exists() or not raw_path.exists():
            raise FileNotFoundError(f"incomplete P9.4 cell: {job}")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["release"], manifest["model"], int(manifest["seed"])) != (job.release, job.model, job.seed):
            raise RuntimeError(f"P9.4 identity mismatch: {job}")
        if manifest["contract_hash"] != p7.sha256_file(CONTRACT):
            raise RuntimeError(f"P9.4 contract changed: {job}")
        if tuple(manifest["actions"]) != evaluator.ACTIONS:
            raise RuntimeError(f"P9.4 action order changed: {job}")
        if p7.sha256_file(raw_path) != manifest["raw_sha256"]:
            raise RuntimeError(f"P9.4 raw changed: {job}")
        rows = pq.read_metadata(raw_path).num_rows
        if rows != int(manifest["requests"]) * len(evaluator.ACTIONS):
            raise RuntimeError(f"P9.4 row conservation failed: {job}")
        if max(manifest["invariants"].values()) > 1e-5:
            raise RuntimeError(f"P9.4 invariant failed: {job}")
        cells.append({
            "release": job.release, "model": job.model, "seed": job.seed,
            "requests": int(manifest["requests"]), "rows": rows,
            "manifest": str(manifest_path.relative_to(ROOT)), "manifest_sha256": p7.sha256_file(manifest_path),
            "raw": str(raw_path.relative_to(ROOT)), "raw_sha256": manifest["raw_sha256"],
            "checkpoint_sha256": manifest["checkpoint_hash"],
            "parent_checkpoint_sha256": manifest["parent_checkpoint_hash"],
            "source_p8_raw_sha256": manifest["source_p8_raw_hash"],
            "invariants": manifest["invariants"],
        })
    payload = {
        "status": "P9_4_all_24_dependency_closed_executor_cells_sealed_before_metrics",
        "contract_hash": p7.sha256_file(CONTRACT), "actions": list(evaluator.ACTIONS),
        "cell_count": len(cells), "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
