#!/usr/bin/env python3
"""Seal the frozen P8 inputs before any P9 tomography score is written."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_p7_theta0 as p7
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_tomography_contract_v1.yaml"
OUTPUT = ROOT / "results/p9/p8_evidence_seal_v1.json"
SOURCES = {
    "r0_raw_seal": "results/p8/r0_control/raw_score_seal_v1.json",
    "r0_adjudication": "results/p8/r0_control/adjudication_v1.json",
    "r1_edge1_raw_seal": "results/p8/r1_edge1/raw_score_seal_v1.json",
    "r1_edge1_adjudication": "results/p8/r1_edge1/hs_adjudication_v1.json",
    "r1_edge2_raw_seal": "results/p8/r1_edge2/raw_score_seal_v1.json",
    "r1_edge2_adjudication": "results/p8/r1_edge2/hs_adjudication_v1.json",
    "r2_raw_seal": "results/p8/r2/raw_score_seal_v1.json",
    "r2_adjudication": "results/p8/r2/hs_adjudication_v1.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite sealed evidence: {args.output}")
    contract = yaml.safe_load(CONTRACT.read_text())
    artifacts = []
    for key, relative in SOURCES.items():
        path = ROOT / relative
        actual = p7.sha256_file(path)
        expected = contract["input_hashes"][key]
        if actual != expected:
            raise RuntimeError(f"P8 evidence hash changed for {key}: {actual} != {expected}")
        payload = json.loads(path.read_text())
        artifacts.append({"name": key, "path": relative, "sha256": actual, "status": payload.get("status")})
    if not any(row["status"] == "R0_blocking_control_passed" for row in artifacts):
        raise RuntimeError("P8 R0 blocking control is not sealed as passed")
    if any("adjudicated" not in str(row["status"]) for row in artifacts if "adjudication" in row["name"] and row["name"] != "r0_adjudication"):
        raise RuntimeError("P8 H/S adjudication is incomplete")
    payload = {
        "status": "P9_P8_evidence_sealed_before_tomography",
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_hash": p7.sha256_file(CONTRACT),
        "artifacts": artifacts,
        "scope": contract["scope"],
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "artifacts": len(artifacts), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
