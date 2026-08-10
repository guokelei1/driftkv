from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

from hstu_kvcache.streaming.ml1m_opportunity import _atomic_json, file_sha256

PROTOCOL = "evokv_ml1m_positive_opportunity_synthesis_v0"


def load_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("candidate_protocol") != "popular_unseen_50"
        or document.get("metric_family")
        != ["hit_rate_at_5", "hit_rate_at_10", "mrr", "ndcg_at_5", "ndcg_at_10"]
        or document.get("minimum_relative_percent") != 5.0
    ):
        raise ValueError("ML1m positive-opportunity synthesis config differs")
    for source in document.get("sources", {}).values():
        if file_sha256(source.get("path", "")) != source.get("sha256"):
            raise ValueError("ML1m positive-opportunity source binding differs")
    return document


def _seed_summaries(document: dict[str, Any]) -> list[dict[str, Any]]:
    discovery = json.loads(Path(document["sources"]["discovery"]["path"]).read_text())
    variant = next(value for value in discovery["variants"] if value["id"] == "q1_legacy_normalized_query")
    output = []
    for seed in variant["seed_results"]:
        result = next(
            value
            for value in seed["strategy_results"]
            if value["strategy"]["id"] == document["candidate_protocol"]
        )
        output.append({"role": "discovery", "seed": seed["seed"], "summary": result["summary"]})
    for role, key in (("gate_audit", "gate_audit"), ("final_confirmation", "final_confirmation")):
        source = json.loads(Path(document["sources"][key]["path"]).read_text())
        output.append({"role": role, "seed": source["seed"], "summary": source["summary"]})
    return output


def _seed_interval(values: np.ndarray) -> list[float]:
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    mean = float(values.mean())
    return [mean - half, mean + half]


def _aggregate(seeds: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [value["summary"]["comparisons"]["recompute_over_reuse"][metric] for value in seeds]
    absolute = np.asarray([value["absolute"] for value in values], dtype=np.float64)
    relative = np.asarray([value["relative_percent"] for value in values], dtype=np.float64)
    return {
        "absolute_by_seed": absolute.tolist(),
        "absolute_mean": float(absolute.mean()),
        "diagnostic_training_seed_95_interval": _seed_interval(absolute),
        "positive_user_cluster_ci_count": sum(value["positive_direction_with_ci"] for value in values),
        "relative_percent_by_seed": relative.tolist(),
        "relative_percent_mean": float(relative.mean()),
        "relative_percent_range": [float(relative.min()), float(relative.max())],
    }


def _seed_record(value: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    summary = value["summary"]
    stale = summary["comparisons"]["recompute_over_reuse"]
    passing = [
        metric
        for metric in document["metric_family"]
        if stale[metric]["positive_direction_with_ci"]
        and stale[metric]["relative_percent"] >= document["minimum_relative_percent"]
    ]
    return {
        "role": value["role"],
        "seed": value["seed"],
        "users": summary["users"],
        "passing_ranking_metrics": passing,
        "fresh_update_mrr_positive": summary["comparisons"]["fresh_update_value"]["mrr"][
            "positive_direction_with_ci"
        ],
        "fresh_update_ndcg10_positive": summary["comparisons"]["fresh_update_value"][
            "ndcg_at_10"
        ]["positive_direction_with_ci"],
        "history_mrr_positive": summary["comparisons"]["history_value"]["mrr"][
            "positive_direction_with_ci"
        ],
        "history_ndcg10_positive": summary["comparisons"]["history_value"]["ndcg_at_10"][
            "positive_direction_with_ci"
        ],
        "stale_cross_entropy_positive": stale["candidate_cross_entropy"][
            "positive_direction_with_ci"
        ],
        "endpoints": summary["endpoints"],
        "comparisons": summary["comparisons"],
        "representation_drift": summary["representation_drift"],
    }


def run(config_path: str | Path) -> dict[str, Any]:
    document = load_config(config_path)
    result_path = Path(document["outputs"]["result"])
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        validate_result(result, document)
        return result
    seeds = _seed_summaries(document)
    records = [_seed_record(value, document) for value in seeds]
    attribution = json.loads(
        Path(document["sources"]["cache_path_attribution"]["path"]).read_text()
    )
    all_names = set(attribution["variants"][-1]["current_parameter_names"])
    cache_names = set(attribution["variants"][-2]["current_parameter_names"])
    cache_safe_names = sorted(all_names - cache_names)
    aggregate = {
        metric: _aggregate(seeds, metric)
        for metric in ["candidate_cross_entropy", *document["metric_family"]]
    }
    final = next(value for value in records if value["role"] == "final_confirmation")
    result = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete_positive_development_opportunity",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "sources": document["sources"],
        "workload": {
            "dataset": "ML-1M hard_v5",
            "users": 5923,
            "model": "legacy HSTU H64 L2 normalized query objective",
            "transition": "train theta0; ingest per-user dev target into theta1; evaluate untouched test target",
            "candidate_protocol": "one positive plus 49 train-popular items unseen by that user",
        },
        "seed_records": records,
        "aggregate_diagnostic": aggregate,
        "mechanism": {
            "cache_producing_parameter_tensors": sorted(cache_names),
            "cache_safe_parameter_tensors": cache_safe_names,
            "cache_producing_tensor_count": len(cache_names),
            "cache_safe_tensor_count": len(cache_safe_names),
            "cumulative_path_recovery": attribution["recovery"],
            "exact_cache_path_boundary_passed": attribution["decision"][
                "cache_path_is_exact_boundary"
            ],
        },
        "decision": {
            "development_positive_opportunity": (
                len(final["passing_ranking_metrics"]) >= 2
                and final["fresh_update_mrr_positive"]
                and final["fresh_update_ndcg10_positive"]
                and final["history_mrr_positive"]
                and final["history_ndcg10_positive"]
                and final["stale_cross_entropy_positive"]
                and attribution["decision"]["cache_path_is_exact_boundary"]
            ),
            "final_confirmation_seed": final["seed"],
            "final_confirmation_passing_metrics": final["passing_ranking_metrics"],
            "formal_promotion": False,
            "formal_blockers": [
                "candidate and metric-family selection used earlier development seeds",
                "only one seed was untouched after the final metric-family gate was frozen",
                "candidate cross-entropy calibration worsens after the update despite ranking gains",
                "the positive workload is small fixed-catalog ML-1M rather than primary QK/QB",
            ],
        },
    }
    validate_result(result, document)
    _atomic_json(result_path, result)
    return result


def validate_result(result: dict[str, Any], document: dict[str, Any]) -> None:
    if (
        result.get("protocol") != PROTOCOL
        or result.get("round_id") != document["round_id"]
        or result.get("status") != "complete_positive_development_opportunity"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or result.get("config", {}).get("sha256") != file_sha256(result["config"]["path"])
        or [value.get("seed") for value in result.get("seed_records", [])]
        != [4217, 14929, 23711, 53117]
        or result.get("decision", {}).get("development_positive_opportunity") is not True
        or result.get("decision", {}).get("formal_promotion") is not False
    ):
        raise ValueError("ML1m positive-opportunity synthesis differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
