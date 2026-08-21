#!/usr/bin/env python3
"""Seal every P7.8 raw-score file before any metric computation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "results/p7/h_qualification/raw"
OUTPUT = ROOT / "results/p7/h_qualification/raw_score_seal_v1.json"
RUN_PLAN = ROOT / "configs/contracts/p7_8_qualification_run_plan_v1.json"
MODELS = {
    "m0_n": {"N": ("quality", "fidelity")},
    "m0_r": {
        "R": ("quality_rankable", "fidelity_all_eligible", "fidelity_rankable_companion")
    },
    "m0_f": {"F": ("quality", "fidelity")},
    "m1": {
        "N": ("quality", "fidelity"),
        "R": ("quality_rankable", "fidelity_all_eligible", "fidelity_rankable_companion"),
        "F": ("quality", "fidelity"),
    },
}
FIDELITY_FORBIDDEN = {
    "target_index",
    "label",
    "is_target",
    "is_organic",
    "prior_30m_same_item",
    "latest_item",
    "long_gap_at_least_3d",
    "history_position_cohort",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(not OUTPUT.exists(), f"refusing to overwrite raw seal: {OUTPUT}")
    plan = json.loads(RUN_PLAN.read_text())
    artifacts: list[dict[str, Any]] = []
    for model, workloads in MODELS.items():
        for seed in (17, 37, 71):
            directory = RAW_ROOT / f"{model}_seed{seed}"
            summary_path = directory / "raw_run_summary.json"
            require(summary_path.is_file(), f"missing raw run summary: {summary_path}")
            summary = json.loads(summary_path.read_text())
            require(summary["metrics_computed"] is False, f"metrics already computed: {model}/{seed}")
            require(summary["run_plan_sha256"] == sha256_file(RUN_PLAN), "run-plan hash mismatch")
            indexed = {(row["workload"], row["view"]): row for row in summary["outputs"]}
            expected = {(workload, view) for workload, views in workloads.items() for view in views}
            require(set(indexed) == expected, f"raw view set differs: {model}/{seed}")
            for key in sorted(expected):
                row = indexed[key]
                path = ROOT / row["path"]
                require(path.is_file(), f"missing raw score file: {path}")
                digest = sha256_file(path)
                require(digest == row["sha256"], f"raw score hash differs: {path}")
                schema = set(pq.read_schema(path).names)
                if "fidelity" in key[1]:
                    require(not schema & FIDELITY_FORBIDDEN, f"fidelity leaked quality fields: {path}")
                require(row["base_full_recent_max_abs_delta"] == 0.0, f"Base path differs: {path}")
                require(row["fidelity_schema_has_forbidden_fields"] is False, f"fidelity schema audit failed: {path}")
                artifacts.append(
                    {
                        "model_condition": model,
                        "seed": seed,
                        "workload": key[0],
                        "view": key[1],
                        "path": row["path"],
                        "sha256": digest,
                        "requests": row["requests"],
                        "candidate_rows": row["candidate_rows"],
                        "schema": sorted(schema),
                    }
                )
    payload = {
        "status": "sealed_all_raw_scores_before_metrics",
        "run_plan_sha256": sha256_file(RUN_PLAN),
        "qualification_index_sha256": plan["qualification_index_sha256"],
        "raw_files": len(artifacts),
        "metrics_computed": False,
        "artifacts": artifacts,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "raw_files": len(artifacts), "sha256": sha256_file(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
