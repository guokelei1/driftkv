#!/usr/bin/env python3
"""Seal the three 8L rolling-lineage H/S raw cells before metric reveal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_hs_v1.yaml"
RAW_ROOT = ROOT / "results/scale_8l_v1/hs_raw"
OUTPUT = ROOT / "results/scale_8l_v1/hs_raw_seal_v1.json"
RELEASES = ("r1_edge1", "r1_edge2", "r2")
FIDELITY_FORBIDDEN = {
    "label", "is_organic", "prior_30m_same_item", "latest_item",
    "long_gap_at_least_3d", "feedback_history_stratum_v2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["sealed_inputs"]["raw_sealer_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("raw sealer changed after H/S contract freeze")
    artifacts = []
    runs = []
    for release in RELEASES:
        manifest_path = RAW_ROOT / release / "m0_f_seed17/raw_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "raw_scores_written_before_metrics":
            raise RuntimeError(f"raw cell failed: {release}")
        if manifest["scope"] != "full_development_edge":
            raise RuntimeError(f"refusing to seal canary scope: {release}")
        if manifest["contract_sha256"] != sha256_file(CONTRACT):
            raise RuntimeError("raw cell contract hash differs")
        if manifest["metrics_computed"] is not False or manifest["qualification_or_theta3_read"] is not False:
            raise RuntimeError("raw cell violated reveal/data-access boundary")
        for row in manifest["artifacts"]:
            path = ROOT / row["path"]
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"raw score hash differs: {path}")
            schema = set(pq.read_schema(path).names)
            if row["view"] == "fidelity" and schema & FIDELITY_FORBIDDEN:
                raise RuntimeError(f"fidelity schema leaked quality fields: {path}")
            artifacts.append({"release": release, **row})
        runs.append({
            "release": release,
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "parent_checkpoint_sha256": manifest["parent_checkpoint_sha256"],
            "population": manifest["population"],
            "lineage": manifest["lineage"],
            "base_full_recent_max_abs_delta": manifest["base_full_recent_max_abs_delta"],
            "request_local_full_companion_max_abs_logit": manifest[
                "exact_rolling_vs_request_local_full_max_abs_logit_companion"
            ],
        })
    value = {
        "status": "sealed_all_scale_8l_seed17_HS_raw_scores_before_metrics",
        "evidence_level": "development_scale_pilot_single_seed",
        "contract_sha256": sha256_file(CONTRACT),
        "qualification_or_theta3_read": False,
        "metrics_computed": False,
        "runs": runs,
        "artifacts": artifacts,
    }
    OUTPUT.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"status": value["status"], "artifacts": len(artifacts), "sha256": sha256_file(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
