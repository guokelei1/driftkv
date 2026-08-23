#!/usr/bin/env python3
"""Seal the three P9.10 full-population runtime conditions."""

import json
from pathlib import Path

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_10_full_population_runtime_contract_v1.yaml"
EVALUATOR = ROOT / "scripts/eval_p9_full_population_runtime.py"
RAW = ROOT / "results/p9/full_population_runtime/full"
OUTPUT = ROOT / "results/p9/p9_10_full_population_runtime_raw_seal_v1.json"
CONDITIONS = (
    "edge1_m0_r2_seed17", "edge1_m1_r2_seed17", "edge2_m1_r1_edge2_seed17"
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    artifacts = []
    for name in CONDITIONS:
        path = RAW / name / "result.json"
        result = json.loads(path.read_text())
        if result["status"] != "P9_10_full_population_migration_runtime_measured" or result["scope"] != "full":
            raise RuntimeError(f"invalid P9.10 condition {name}")
        if result["condition"]["name"] != name or result["scheduler_authorized"]:
            raise RuntimeError(f"P9.10 condition metadata differs: {name}")
        artifacts.append({
            "condition": name, "path": str(path.relative_to(ROOT)),
            "sha256": p7.sha256_file(path), "states": result["states"],
        })
    payload = {
        "status": "P9_10_three_full_population_runtime_conditions_sealed",
        "contract_sha256": p7.sha256_file(CONTRACT),
        "evaluator_sha256": p7.sha256_file(EVALUATOR),
        "artifacts": artifacts,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "conditions": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
