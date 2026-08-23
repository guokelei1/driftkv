#!/usr/bin/env python3
"""Seal all six P11.4 raw recursive-quality cells."""

from __future__ import annotations

import json
from pathlib import Path

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p11_4_recursive_policy_quality_v1.yaml"
RAW = ROOT / "results/p11/p11_4_recursive_policy_quality_raw/full"
OUTPUT = ROOT / "results/p11/p11_4_recursive_policy_quality_raw_seal_v1.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    artifacts = []
    for model in ("m0_f", "m1"):
        for seed in (17, 37, 71):
            manifest_path = RAW / f"{model}_seed{seed}/raw_manifest.json"
            if not manifest_path.exists():
                raise RuntimeError(f"missing P11.4 raw cell: {model} seed{seed}")
            manifest = json.loads(manifest_path.read_text())
            raw_path = ROOT / manifest["raw_path"]
            if manifest["status"] != "passed_recursive_quality_raw_unadjudicated":
                raise RuntimeError("failed P11.4 raw cell cannot be sealed")
            if manifest["contract_sha256"] != p7.sha256_file(CONTRACT) or manifest["raw_sha256"] != p7.sha256_file(raw_path):
                raise RuntimeError("P11.4 raw artifact changed")
            artifacts.append({"model": model, "seed": seed, "users": manifest["users"],
                              "requests": manifest["requests"], "raw_path": manifest["raw_path"],
                              "raw_sha256": manifest["raw_sha256"],
                              "manifest": str(manifest_path.relative_to(ROOT)),
                              "manifest_sha256": p7.sha256_file(manifest_path)})
    payload = {"status": "P11_4_all_recursive_quality_raw_cells_sealed",
               "contract_sha256": p7.sha256_file(CONTRACT), "artifacts": artifacts,
               "metrics_computed": False, "policy_changed": False}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
