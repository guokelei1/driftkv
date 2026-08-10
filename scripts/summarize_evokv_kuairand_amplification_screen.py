from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from hstu_kvcache.streaming.kuairand_query_transition import (
    _atomic_json,
    file_sha256,
)

METRICS = ("mrr", "ndcg_at_5", "hit_rate_at_5")


def _ranking(summary: dict[str, Any]) -> dict[str, float]:
    comparison = summary["comparisons"]["recompute_over_reuse"]
    return {metric: float(comparison[metric]["relative_percent"]) for metric in METRICS}


def _fresh(summary: dict[str, Any]) -> dict[str, float]:
    endpoint = summary["endpoints"]["recompute"]
    return {metric: float(endpoint[metric]) for metric in METRICS}


def _fresh_update(summary: dict[str, Any]) -> dict[str, float]:
    comparison = summary["comparisons"]["fresh_update_value"]
    return {metric: float(comparison[metric]["relative_percent"]) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    root = Path(args.root)
    document = json.loads(config_path.read_text())
    config_sha256 = file_sha256(config_path)
    candidates = document["training"]["candidate_ladder"]
    results = []
    for candidate in candidates:
        path = root / "candidates" / f"{candidate['name']}.json"
        result = json.loads(path.read_text())
        if (
            result.get("status") != "complete"
            or result.get("version") != 2
            or result.get("candidate") != candidate
            or result.get("config", {}).get("sha256") != config_sha256
            or set(result.get("partition_summaries", {})) != {"tuning", "holdout"}
        ):
            raise RuntimeError("KuaiRand amplification candidate result differs")
        tuning = result["partition_summaries"]["tuning"]
        results.append(
            {
                "name": candidate["name"],
                "path": str(path),
                "sha256": file_sha256(path),
                "training": result["training"],
                "tuning": {
                    "ranking_relative_percent": _ranking(tuning),
                    "fresh": _fresh(tuning),
                    "fresh_update_relative_percent": _fresh_update(tuning),
                    "records": int(tuning["partition"]["records"]),
                    "users": int(tuning["partition"]["users"]),
                    "sanity": tuning["sanity"],
                },
            }
        )
    baseline = results[0]
    if baseline["name"] != "baseline_stationary_kv2x_n8192_e2":
        raise RuntimeError("KuaiRand amplification baseline differs")
    baseline_ranking = baseline["tuning"]["ranking_relative_percent"]
    baseline_fresh = baseline["tuning"]["fresh"]
    for result in results:
        tuning = result["tuning"]
        ranking = tuning["ranking_relative_percent"]
        amplification = {
            metric: ranking[metric] / baseline_ranking[metric]
            if baseline_ranking[metric] > 0
            else None
            for metric in METRICS
        }
        retention = {metric: tuning["fresh"][metric] / baseline_fresh[metric] for metric in METRICS}
        quality_pass = bool(
            tuning["sanity"]["passed"]
            and all(value > 0 for value in ranking.values())
            and all(value > 0 for value in tuning["fresh_update_relative_percent"].values())
            and min(retention.values()) >= 0.95
        )
        target_pass = bool(
            quality_pass
            and 6.0 <= ranking["ndcg_at_5"] <= 12.0
            and ranking["mrr"] >= 3.0
            and ranking["hit_rate_at_5"] >= 3.0
        )
        result["amplification_over_baseline"] = amplification
        result["fresh_retention_over_baseline"] = retention
        result["quality_pass"] = quality_pass
        result["two_x_target_pass"] = target_pass
    eligible = [result for result in results[1:] if result["quality_pass"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda result: (
                result["two_x_target_pass"],
                -abs(
                    math.log(
                        max(
                            result["tuning"]["ranking_relative_percent"]["ndcg_at_5"],
                            1e-12,
                        )
                        / 8.0
                    )
                ),
                min(result["tuning"]["ranking_relative_percent"].values()),
                min(result["fresh_retention_over_baseline"].values()),
            ),
        )
        selected_raw = json.loads(Path(selected["path"]).read_text())
        selected_holdout = selected_raw["partition_summaries"]["holdout"]
        selection = {
            "name": selected["name"],
            "two_x_target_pass": selected["two_x_target_pass"],
            "tuning": selected["tuning"],
            "amplification_over_baseline": selected["amplification_over_baseline"],
            "fresh_retention_over_baseline": selected["fresh_retention_over_baseline"],
            "holdout_opened_after_selection": {
                "ranking_relative_percent": _ranking(selected_holdout),
                "fresh": _fresh(selected_holdout),
                "fresh_update_relative_percent": _fresh_update(selected_holdout),
                "records": int(selected_holdout["partition"]["records"]),
                "users": int(selected_holdout["partition"]["users"]),
                "sanity": selected_holdout["sanity"],
            },
        }
    else:
        selection = None
    output = {
        "protocol": "evokv_kuairand_amplification_theta2_screen_v0",
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": config_sha256},
        "selection_rule": {
            "selection_partition": "tuning_users_only",
            "holdout_policy": "open_only_for_selected_candidate",
            "fresh_retention_floor": 0.95,
            "primary_relative_percent_range": [6.0, 12.0],
            "minimum_supporting_relative_percent": 3.0,
        },
        "baseline": baseline,
        "tuning_candidates": results[1:],
        "selected": selection,
        "next": "freeze_selected_schedule_for_theta2_theta10"
        if selection is not None and selection["two_x_target_pass"]
        else "refine_training_strength_without_opening_other_holdouts",
    }
    _atomic_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
