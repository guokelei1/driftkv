#!/usr/bin/env python3
"""Seal all four 8L frozen-action raw cells before metric computation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_method_v1.yaml"
RAW = ROOT / "results/scale_8l_v1/actions_raw/full"
OUTPUT = ROOT / "results/scale_8l_v1/actions_raw_seal_v1.json"


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = yaml.safe_load(CONTRACT.read_text()); artifacts = []
    for release in contract["scope"]["releases"]:
        manifest_path = RAW / release / "m0_f_seed17/raw_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "passed_raw_scores_unadjudicated" or manifest["metrics_computed"]:
            raise RuntimeError(f"raw cell is not sealable: {release}")
        raw = ROOT / manifest["raw"]
        if p7.sha256_file(raw) != manifest["raw_sha256"]: raise RuntimeError(f"raw hash mismatch: {release}")
        if manifest["contract_sha256"] != p7.sha256_file(CONTRACT): raise RuntimeError("contract mismatch")
        artifacts.append({"release": release, "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": p7.sha256_file(manifest_path), "raw": manifest["raw"],
            "raw_sha256": manifest["raw_sha256"]})
    payload = {"status": "scale_8l_all_action_raw_cells_sealed_before_metrics",
        "contract_sha256": p7.sha256_file(CONTRACT), "artifacts": artifacts,
        "qualification_or_theta3_read": False}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
