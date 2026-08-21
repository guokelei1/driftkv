#!/usr/bin/env python3
"""Seal one complete P8 R1/R2 raw-score matrix before H/S metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "results/p8/staleness_raw"
CONTRACT = ROOT / "configs/contracts/f_release_chain_contract_v1.yaml"
MODELS = {"m0_f": ("F",), "m1": ("N", "R", "F")}
VIEWS = {
    "N": ("quality", "fidelity"),
    "R": ("quality_rankable", "fidelity_all_eligible"),
    "F": ("quality", "fidelity"),
}
RELEASES = ("r1_edge1", "r1_edge2", "r2")
SEEDS = (17, 37, 71)


def seal_release(release: str, output: Path) -> dict:
    artifacts = []
    runs = []
    for model, workloads in MODELS.items():
        for seed in SEEDS:
            root = RAW_ROOT / release / f"{model}_seed{seed}"
            manifest_path = root / "raw_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"missing raw manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            if manifest["metrics_computed"] is not False:
                raise RuntimeError(f"raw manifest already claims metrics: {manifest_path}")
            if manifest["release"] != release or manifest["model_name"] != model or manifest["seed"] != seed:
                raise RuntimeError(f"raw manifest identity mismatch: {manifest_path}")
            expected = {(workload, view) for workload in workloads for view in VIEWS[workload]}
            actual = {(row["workload"], row["view"]) for row in manifest["artifacts"]}
            if actual != expected:
                raise RuntimeError(f"raw view matrix mismatch for {model}/seed{seed}: {actual} != {expected}")
            for artifact in manifest["artifacts"]:
                path = ROOT / artifact["path"]
                if p7.sha256_file(path) != artifact["sha256"]:
                    raise RuntimeError(f"raw score hash changed: {path}")
                schema = set(artifact["schema"])
                if artifact["view"].startswith("fidelity"):
                    forbidden = {
                        "label", "target_index", "is_target", "is_organic",
                        "prior_30m_same_item", "latest_item", "feedback_history_stratum_v2",
                    }
                    if forbidden & schema:
                        raise RuntimeError(f"fidelity schema exposes quality fields: {path}")
                artifacts.append({"model": model, "seed": seed, **artifact})
            training = ROOT / f"results/p8/release_training/{release}/{model}_seed{seed}/train_result.json"
            result = json.loads(training.read_text())
            if result["staleness_scored"] is not False:
                raise RuntimeError(f"training result claims prior staleness scoring: {training}")
            if result["checkpoint_hash"] != manifest["checkpoint_hash"]:
                raise RuntimeError(f"checkpoint/raw lineage mismatch: {manifest_path}")
            runs.append({
                "model": model,
                "seed": seed,
                "admitted": bool(result["admitted"]),
                "training_result": str(training.relative_to(ROOT)),
                "training_result_hash": p7.sha256_file(training),
                "raw_manifest": str(manifest_path.relative_to(ROOT)),
                "raw_manifest_hash": p7.sha256_file(manifest_path),
                "checkpoint_hash": manifest["checkpoint_hash"],
                "parent_hash": manifest["parent_hash"],
            })
    payload = {
        "status": f"sealed_all_{release}_raw_scores_before_H_S_metrics",
        "release": release,
        "contract_hash": p7.sha256_file(CONTRACT),
        "runs": runs,
        "run_count": len(runs),
        "raw_files": len(artifacts),
        "metrics_computed": False,
        "artifacts": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=RELEASES, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"results/p8/{args.release}/raw_score_seal_v1.json"
    payload = seal_release(args.release, output)
    print(json.dumps({
        "status": payload["status"], "runs": payload["run_count"],
        "raw_files": payload["raw_files"], "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
