#!/usr/bin/env python3
"""Seal all six R0 raw-path runs before the blocking-control metrics."""

from __future__ import annotations

import json
from pathlib import Path

import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/p8/staleness_raw/r0"
OUTPUT = ROOT / "results/p8/r0_control/raw_score_seal_v1.json"
CONTRACT = ROOT / "configs/contracts/f_release_chain_contract_v1.yaml"


def main() -> None:
    artifacts = []
    for model in ("m0_f", "m1"):
        for seed in (17, 37, 71):
            root = RAW / f"{model}_seed{seed}"
            manifest_path = root / "raw_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest["metrics_computed"] is not False:
                raise RuntimeError("R0 raw manifest claims metrics were computed")
            if manifest["cache_path_max_abs_delta"] != 0.0:
                raise RuntimeError("R0 changed the cache-producing path")
            for artifact in manifest["artifacts"]:
                path = ROOT / artifact["path"]
                if p7.sha256_file(path) != artifact["sha256"]:
                    raise RuntimeError(f"raw score hash changed: {path}")
                if artifact["view"].startswith("fidelity"):
                    forbidden = {"label", "target_index", "is_target", "feedback_history_stratum_v2"}
                    if forbidden & set(artifact["schema"]):
                        raise RuntimeError("fidelity raw schema exposes quality fields")
                artifacts.append({"model": model, "seed": seed, **artifact})
    payload = {
        "status": "sealed_all_R0_raw_scores_before_blocking_metrics",
        "contract_hash": p7.sha256_file(CONTRACT), "runs": 6,
        "raw_files": len(artifacts), "metrics_computed": False, "artifacts": artifacts,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "raw_files": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
