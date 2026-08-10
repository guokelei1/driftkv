from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hstu_kvcache.streaming.kuairand_query_transition import _atomic_json, file_sha256

METRICS = ("mrr", "ndcg_at_5", "hit_rate_at_5")


def _ranking(summary: dict[str, Any]) -> dict[str, float]:
    comparison = summary["comparisons"]["recompute_over_reuse"]
    return {metric: float(comparison[metric]["relative_percent"]) for metric in METRICS}


def _fresh(summary: dict[str, Any]) -> dict[str, float]:
    endpoint = summary["endpoints"]["recompute"]
    return {metric: float(endpoint[metric]) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    document = json.loads(config_path.read_text())
    root = Path(document["outputs"]["root"])
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    screen_path = Path(document["selection_source"]["path"])
    if file_sha256(screen_path) != document["selection_source"]["sha256"]:
        raise RuntimeError("KuaiRand amplification screen binding differs")
    screen = json.loads(screen_path.read_text())
    selected_name = document["selection_source"]["selected_candidate"]
    if screen.get("selected", {}).get("name") != selected_name:
        raise RuntimeError("KuaiRand amplification selection differs")
    candidate = document["training"]["candidate_ladder"][0]
    if candidate["name"] != selected_name:
        raise RuntimeError("KuaiRand amplification fixed candidate differs")
    edges = []
    for version in (2, 3):
        accepted_path = root / "edges" / f"theta_{version}" / "accepted.json"
        accepted = json.loads(accepted_path.read_text())
        manifest_path = checkpoint_root / f"theta_{version}" / "manifest.json"
        if (
            accepted.get("status") != "accepted"
            or accepted.get("version") != version
            or accepted.get("checkpoint_policy") != "fixed_schedule"
            or accepted.get("candidate", {}).get("candidate") != candidate
            or set(accepted.get("candidate", {}).get("partition_summaries", {}))
            != {"tuning", "holdout"}
            or not manifest_path.is_file()
            or file_sha256(manifest_path) != accepted.get("checkpoint", {}).get("sha256")
        ):
            raise RuntimeError("KuaiRand amplification prefix artifact differs")
        partitions = accepted["candidate"]["partition_summaries"]
        edges.append(
            {
                "version": version,
                "update_date": document["transitions"][version - 1]["update_date"],
                "evaluation_date": document["transitions"][version - 1]["evaluation_date"],
                "full": _ranking(accepted["candidate"]["summary"]),
                "tuning": _ranking(partitions["tuning"]),
                "holdout": _ranking(partitions["holdout"]),
                "holdout_fresh": _fresh(partitions["holdout"]),
                "tuning_users": int(partitions["tuning"]["partition"]["users"]),
                "holdout_users": int(partitions["holdout"]["partition"]["users"]),
                "sanity": {
                    partition: partitions[partition]["sanity"]
                    for partition in ("tuning", "holdout")
                },
                "checkpoint": {
                    "path": str(manifest_path),
                    "sha256": file_sha256(manifest_path),
                    "bytes": int(accepted["checkpoint"]["bytes"]),
                },
            }
        )
    screen_candidate_path = Path(
        next(
            value["path"] for value in screen["tuning_candidates"] if value["name"] == selected_name
        )
    )
    screen_candidate = json.loads(screen_candidate_path.read_text())
    theta2_reproduced = all(
        np.isclose(
            edges[0]["full"][metric],
            _ranking(screen_candidate["summary"])[metric],
            rtol=0,
            atol=1e-9,
        )
        for metric in METRICS
    )
    all_positive = all(
        edge[partition][metric] > 0
        for edge in edges
        for partition in ("tuning", "holdout")
        for metric in METRICS
    )
    all_sanity = all(
        edge["sanity"][partition]["passed"] for edge in edges for partition in ("tuning", "holdout")
    )
    holdout_ndcg = [edge["holdout"]["ndcg_at_5"] for edge in edges]
    fresh_floor = all(
        edge["holdout_fresh"]["mrr"] >= 0.09
        and edge["holdout_fresh"]["ndcg_at_5"] >= 0.08
        and edge["holdout_fresh"]["hit_rate_at_5"] >= 0.12
        for edge in edges
    )
    passed = bool(
        theta2_reproduced
        and all_positive
        and all_sanity
        and min(holdout_ndcg) >= 1.0
        and float(np.mean(holdout_ndcg)) >= 5.0
        and fresh_floor
    )
    result = {
        "protocol": "evokv_kuairand_amplified_fixed_prefix_v0",
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "selected_candidate": candidate,
        "edges": edges,
        "decision": {
            "theta2_exactly_reproduced": theta2_reproduced,
            "all_tuning_and_holdout_ranking_positive": all_positive,
            "all_same_model_sanity_passed": all_sanity,
            "minimum_holdout_ndcg_relative_percent": min(holdout_ndcg),
            "mean_holdout_ndcg_relative_percent": float(np.mean(holdout_ndcg)),
            "fresh_absolute_floor_passed": fresh_floor,
            "passed": passed,
            "next": "resume_unchanged_theta4_theta10" if passed else "stop_fixed_schedule",
        },
    }
    _atomic_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
