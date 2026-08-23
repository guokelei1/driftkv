#!/usr/bin/env python3
"""Seal P10 target-free policy assignments before any quality join."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p10_0_cheap_profiler_contract_v1.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    args = parser.parse_args()
    raw = ROOT / f"results/p10/p10_0_cheap_profiler_raw/{args.mode}"
    output = ROOT / f"results/p10/p10_0_cheap_profiler_{args.mode}_seal_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    contract = yaml.safe_load(CONTRACT.read_text())
    summary = json.loads((raw / "run_summary.json").read_text())
    expected = 24 if args.mode == "full" else len(contract["canary"]["semantic_cells"])
    if len(summary["cells"]) != expected or summary["quality_joined"]:
        raise RuntimeError("incomplete or contaminated P10 run summary")
    forbidden = tuple(value.lower() for value in contract["sealing"]["forbidden_assignment_columns"])
    artifacts = []
    for row in summary["cells"]:
        result_path = ROOT / row["result"]
        result = json.loads(result_path.read_text())
        assignment_path = ROOT / result["assignments_path"]
        columns = pq.read_schema(assignment_path).names
        contaminated = [column for column in columns if any(word in column.lower() for word in forbidden)]
        if contaminated:
            raise RuntimeError(f"forbidden assignment columns: {contaminated}")
        if result["quality_joined"] or result["assignments_sha256"] != p7.sha256_file(assignment_path):
            raise RuntimeError(f"contaminated or changed assignment: {row}")
        artifacts.append({
            **{key: row[key] for key in ("release", "model", "seed", "states")},
            "result": row["result"],
            "result_sha256": p7.sha256_file(result_path),
            "assignments": result["assignments_path"],
            "assignments_sha256": result["assignments_sha256"],
            "assignment_rows": pq.read_metadata(assignment_path).num_rows,
        })
    payload = {
        "status": f"P10_0_{args.mode}_target_free_policy_assignments_sealed_before_quality_join",
        "contract_sha256": p7.sha256_file(CONTRACT),
        "run_summary_sha256": p7.sha256_file(raw / "run_summary.json"),
        "cells": len(artifacts),
        "assignment_rows": sum(row["assignment_rows"] for row in artifacts),
        "artifacts": artifacts,
        "quality_joined": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "cells", "assignment_rows")}, indent=2))


if __name__ == "__main__":
    main()
