#!/usr/bin/env python3
"""Re-account P9 transition work over every P9.7 cutover state."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

import analyze_p9_transition_costs as costs
import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/p9_full_population_v1"
AUDIT = ROOT / "results/p9/p9_7_full_population_audit_v1.json"
CANARY = ROOT / "results/p9/p9_7_uid_executor_canary_v1.json"
OUTPUT = ROOT / "results/p9/p9_7_full_population_costs_v1.json"
ACTIONS = ["noop", "layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128", "exact_all"]


def main() -> None:
    audit = json.loads(AUDIT.read_text())
    canary = json.loads(CANARY.read_text())
    if audit["status"] != "P9_7_full_population_and_probe_audit_passed":
        raise RuntimeError("P9.7 population audit has not passed")
    if not canary["full_uid_keyed_executor_authorized"]:
        raise RuntimeError("P9.7 uid-keyed executor canary has not passed")
    edges = []
    for edge in ("edge1", "edge2"):
        table = pq.read_table(MANIFEST / edge / "states.parquet", columns=["uid", "effective_prefix_length"])
        lengths = [
            (int(uid), int(length))
            for uid, length in zip(table["uid"].to_pylist(), table["effective_prefix_length"].to_pylist(), strict=True)
        ]
        edges.append({
            "edge": edge,
            "states": len(lengths),
            "logical_costs": costs.aggregate_logical(lengths, ACTIONS),
        })
    payload = {
        "status": "P9_7_all_materialized_state_costs_accounted",
        "population_audit_sha256": p7.sha256_file(AUDIT),
        "uid_executor_canary_sha256": p7.sha256_file(CANARY),
        "edges": edges,
        "served_subset_cost_artifact_role": "P9_6_runtime_and_companion_only",
        "formal_frontier_authorized": False,
        "scheduler_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "populations": {row["edge"]: row["states"] for row in edges},
    }, indent=2))


if __name__ == "__main__":
    main()
