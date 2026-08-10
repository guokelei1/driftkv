from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hstu_kvcache.streaming.kuairand_query_transition import _atomic_json, file_sha256

METRICS = ("mrr", "ndcg_at_5", "hit_rate_at_5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    result_path = Path(args.result)
    document = json.loads(config_path.read_text())
    result = json.loads(result_path.read_text())
    root = Path(document["outputs"]["root"])
    accepted = json.loads((root / "edges/theta_3/accepted.json").read_text())
    candidate_path = Path(accepted["candidate"]["path"])
    candidate = json.loads(candidate_path.read_text())
    if (
        result.get("status") != "complete"
        or result.get("checkpoint_count") != 3
        or result.get("config", {}).get("sha256") != file_sha256(config_path)
        or candidate.get("tuning_lineage_gate", {}).get("passed") is not True
        or file_sha256(candidate_path) != accepted["candidate"]["sha256"]
    ):
        raise RuntimeError("KuaiRand theta3 lineage result differs")
    ordinary = []
    for target in result["targets"][1:]:
        for lineage in target["lineage"]:
            if int(lineage["source_version"]) < 1:
                continue
            summary = lineage["holdout"]
            comparison = summary["comparisons"]["recompute_over_reuse"]
            ordinary.append(
                {
                    "target_version": int(target["target_version"]),
                    "source_version": int(lineage["source_version"]),
                    "cache_age": int(lineage["cache_age"]),
                    "ranking_relative_percent": {
                        metric: float(comparison[metric]["relative_percent"]) for metric in METRICS
                    },
                    "fresh": {
                        metric: float(summary["endpoints"]["recompute"][metric])
                        for metric in METRICS
                    },
                    "sanity": summary["sanity"],
                }
            )
    latest = [row for row in ordinary if row["target_version"] == 3]
    all_primary_positive = all(
        row["ranking_relative_percent"][metric] > 0
        for row in ordinary
        for metric in ("ndcg_at_5", "hit_rate_at_5")
    )
    negative_mrr_cells = sum(row["ranking_relative_percent"]["mrr"] <= 0 for row in ordinary)
    all_sanity = all(row["sanity"]["passed"] for row in ordinary)
    latest_ndcg = [row["ranking_relative_percent"]["ndcg_at_5"] for row in latest]
    adjacent = next(row for row in latest if row["source_version"] == 2)
    fresh_floor = all(
        row["fresh"]["mrr"] >= 0.09
        and row["fresh"]["ndcg_at_5"] >= 0.08
        and row["fresh"]["hit_rate_at_5"] >= 0.12
        for row in ordinary
    )
    passed = bool(
        len(ordinary) == 3
        and len(latest) == 2
        and all_primary_positive
        and negative_mrr_cells <= 1
        and all_sanity
        and adjacent["ranking_relative_percent"]["ndcg_at_5"] >= 0.5
        and float(np.mean(latest_ndcg)) >= 3.0
        and fresh_floor
    )
    output = {
        "protocol": "evokv_kuairand_theta3_lineage_validation_v0",
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "result": {"path": str(result_path), "sha256": file_sha256(result_path)},
        "selected_candidate": accepted["candidate"]["candidate"],
        "tuning_lineage_gate": candidate["tuning_lineage_gate"],
        "ordinary_holdout_cells": ordinary,
        "decision": {
            "all_ordinary_ndcg_and_hr_positive": all_primary_positive,
            "negative_mrr_cells": negative_mrr_cells,
            "all_same_model_sanity_passed": all_sanity,
            "theta3_adjacent_ndcg_relative_percent": adjacent["ranking_relative_percent"][
                "ndcg_at_5"
            ],
            "theta3_lineage_mean_ndcg_relative_percent": float(np.mean(latest_ndcg)),
            "fresh_absolute_floor_passed": fresh_floor,
            "passed": passed,
            "next": "promote_theta3_and_continue" if passed else "revise_theta3_search",
        },
    }
    _atomic_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
