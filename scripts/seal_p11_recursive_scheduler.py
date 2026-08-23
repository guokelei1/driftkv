#!/usr/bin/env python3
"""Validate and seal P11.2 assignments before same-cost adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p11_2_recursive_scheduler_replay_v1.yaml"
RAW = ROOT / "results/p11/p11_2_recursive_scheduler_raw"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    args = parser.parse_args()
    output = ROOT / f"results/p11/p11_2_recursive_scheduler_{args.mode}_seal_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    contract = yaml.safe_load(CONTRACT.read_text())
    summary_path = RAW / args.mode / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    artifacts = []
    for cell in summary["cells"]:
        result_path, assignment_path = ROOT / cell["result"], ROOT / cell["assignments"]
        if p7.sha256_file(result_path) != cell["result_sha256"] or p7.sha256_file(assignment_path) != cell["assignments_sha256"]:
            raise RuntimeError("P11.2 artifact changed before seal")
        table = pq.read_table(assignment_path)
        forbidden = set(contract["sealing"]["forbidden_assignment_columns"])
        if forbidden & set(table.column_names):
            raise RuntimeError("P11.2 assignment contains forbidden columns")
        frame = table.to_pandas()
        if frame.duplicated(["uid", "sample_fraction", "budget_fraction"]).any():
            raise RuntimeError("duplicate P11.2 assignment")
        if not (frame.loc[frame["calibration_sample"], "action"] == "exact_all").all():
            raise RuntimeError("probe state does not terminate in Exact")
        artifacts.append(dict(cell, assignment_rows=table.num_rows))
    repeat_equal = None
    if args.mode == "canary":
        original = pq.read_table(ROOT / artifacts[0]["assignments"])
        repeat = pq.read_table(RAW / "canary_repeat/m1_seed17/assignments.parquet")
        repeat_equal = original.equals(repeat)
        if not repeat_equal:
            raise RuntimeError("P11.2 canary repeat changed assignments")
    payload = {
        "status": f"P11_2_{args.mode}_assignments_sealed",
        "contract_sha256": p7.sha256_file(CONTRACT), "run_summary_sha256": p7.sha256_file(summary_path),
        "artifacts": artifacts, "deterministic_repeat_equal": repeat_equal,
        "quality_joined": False, "same_cost_adjudication_performed": False,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(artifacts),
                      "deterministic_repeat_equal": repeat_equal}, indent=2))


if __name__ == "__main__":
    main()
