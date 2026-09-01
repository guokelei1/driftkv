#!/usr/bin/env python3
"""Validate the single current Large D14 v0..v5 checkpoint lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_large_d14_canonical_v0_v5_v1.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(contract_path: Path, *, verify_checkpoints: bool) -> dict:
    contract_path = contract_path.resolve()
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    manifest_path = (ROOT / contract["outputs"]["chain_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract_hash = sha256_file(contract_path)
    if manifest["contract_sha256"] != contract_hash:
        raise RuntimeError("canonical chain manifest does not bind its contract")
    expected_versions = [f"v{index}" for index in range(6)]
    if list(contract["checkpoints"]) != expected_versions or list(manifest["versions"]) != expected_versions:
        raise RuntimeError("canonical chain must contain exactly ordered v0..v5")
    previous_hash = None
    rows = []
    for version in expected_versions:
        frozen = contract["checkpoints"][version]
        pointer = manifest["versions"][version]
        for key in ("path", "sha256", "parent_sha256", "training_days_half_open", "epochs"):
            if pointer[key] != frozen[key]:
                raise RuntimeError(f"canonical pointer differs from contract: {version}.{key}")
        if frozen["parent_sha256"] != previous_hash:
            raise RuntimeError(f"canonical parent hash is not direct: {version}")
        path = (ROOT / frozen["path"]).resolve()
        if not path.exists():
            raise RuntimeError(f"canonical checkpoint missing: {version}")
        row = {
            "version": version, "path": frozen["path"], "sha256": frozen["sha256"],
            "epochs": float(frozen["epochs"]), "training_days_half_open": frozen["training_days_half_open"],
        }
        if verify_checkpoints:
            actual_hash = sha256_file(path)
            if actual_hash != frozen["sha256"]:
                raise RuntimeError(f"canonical checkpoint hash mismatch: {version}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            completed = float(payload.get("training_epochs_completed", payload.get("passes", 1.0)))
            if payload["version"] != version or payload.get("parent_checkpoint_sha256") != previous_hash:
                raise RuntimeError(f"canonical checkpoint payload lineage mismatch: {version}")
            if completed != float(frozen["epochs"]) or payload.get("training_day_range") != frozen["training_days_half_open"]:
                raise RuntimeError(f"canonical checkpoint recipe mismatch: {version}")
            row["cache_producer_sha256"] = payload["cache_producer_sha256"]
            del payload
        rows.append(row)
        previous_hash = frozen["sha256"]
    for edge, values in contract["full_only_release_gains"]["edges"].items():
        source = (ROOT / values["source"]).resolve()
        if not source.exists() or sha256_file(source) != values["source_sha256"]:
            raise RuntimeError(f"canonical quality source mismatch: {edge}")
        if float(values["AUC"]) <= 0.0:
            raise RuntimeError(f"canonical development AUC gate failed: {edge}")
    return {
        "status": "canonical_large_D14_v0_v5_chain_valid",
        "contract_sha256": contract_hash,
        "checkpoint_bytes_verified": verify_checkpoints,
        "current_version": manifest["current_version"],
        "versions": rows,
        "all_canonical_edges_AUC_positive": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.contract, verify_checkpoints=args.verify_checkpoints), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
