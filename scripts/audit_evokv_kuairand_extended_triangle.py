from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from hstu_kvcache.streaming.kuairand_query_transition import _atomic_json, file_sha256

RANKING_METRICS = ("mrr", "ndcg_at_5", "hit_rate_at_5")


def _fraction(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--promote-accepted-from")
    parser.add_argument("--promote-accepted-to")
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    result_path = Path(args.result)
    protocol = json.loads(protocol_path.read_text())
    result = json.loads(result_path.read_text())
    if protocol.get("protocol") != "evokv_kuairand_extended_triangle_acceptance_v0":
        raise RuntimeError("KuaiRand extended triangle protocol differs")
    if result.get("status") != "complete":
        raise RuntimeError("KuaiRand extended triangle result is incomplete")
    evaluation = protocol["evaluation"]
    acceptance = protocol["acceptance"]
    primary = evaluation["primary_metric"]
    minimum_source = int(evaluation["ordinary_minimum_source_version"])
    rows = []
    for target in result["targets"]:
        target_version = int(target["target_version"])
        for lineage in target["lineage"]:
            source_version = int(lineage["source_version"])
            if source_version < minimum_source:
                continue
            summary = lineage["holdout"]
            comparison = summary["comparisons"]["recompute_over_reuse"]
            rows.append(
                {
                    "target_version": target_version,
                    "source_version": source_version,
                    "cache_age": int(lineage["cache_age"]),
                    "relative_percent": {
                        metric: float(comparison[metric]["relative_percent"])
                        for metric in RANKING_METRICS
                    },
                    "fresh": {
                        metric: float(summary["endpoints"]["recompute"][metric])
                        for metric in RANKING_METRICS
                    },
                    "sanity": bool(summary["sanity"]["passed"]),
                }
            )
    versions = int(result["checkpoint_count"])
    expected_cells = versions * (versions - 1) // 2
    if len(rows) != expected_cells:
        raise RuntimeError("KuaiRand extended triangle ordinary cell count differs")
    adjacent = [row for row in rows if row["source_version"] == row["target_version"] - 1]
    cumulative = []
    for target_version in range(3, versions + 1):
        target_rows = [row for row in rows if row["target_version"] == target_version]
        adjacent_row = next(
            row for row in target_rows if row["source_version"] == target_version - 1
        )
        older = [row for row in target_rows if row["source_version"] < target_version - 1]
        oldest_max = max(older, key=lambda row: row["relative_percent"][primary])
        margin = oldest_max["relative_percent"][primary] - adjacent_row["relative_percent"][primary]
        cumulative.append(
            {
                "target_version": target_version,
                "adjacent_relative_percent": adjacent_row["relative_percent"][primary],
                "strongest_older_source_version": oldest_max["source_version"],
                "strongest_older_relative_percent": oldest_max["relative_percent"][primary],
                "margin_relative_percent": margin,
                "observed": margin
                >= float(acceptance["cumulative_primary_margin_relative_percent"]),
            }
        )
    primary_positive_fraction = _fraction([row["relative_percent"][primary] > 0 for row in rows])
    ranking_positive_fraction = _fraction(
        [row["relative_percent"][metric] > 0 for row in rows for metric in RANKING_METRICS]
    )
    adjacent_positive_fraction = _fraction(
        [row["relative_percent"][primary] > 0 for row in adjacent]
    )
    adjacent_mean = float(np.mean([row["relative_percent"][primary] for row in adjacent]))
    cumulative_fraction = _fraction([row["observed"] for row in cumulative])
    all_sanity = all(row["sanity"] for row in rows)
    fresh_floor = all(
        row["fresh"][metric] >= float(minimum)
        for row in rows
        for metric, minimum in acceptance["minimum_fresh_metrics"].items()
    )
    passed = bool(
        versions >= int(acceptance["minimum_versions"])
        and primary_positive_fraction >= float(acceptance["minimum_primary_positive_fraction"])
        and ranking_positive_fraction >= float(acceptance["minimum_all_ranking_positive_fraction"])
        and adjacent_positive_fraction
        >= float(acceptance["minimum_adjacent_primary_positive_fraction"])
        and adjacent_mean >= float(acceptance["minimum_adjacent_primary_mean_relative_percent"])
        and cumulative_fraction >= float(acceptance["minimum_cumulative_row_fraction"])
        and fresh_floor
        and (all_sanity or not acceptance["require_all_same_model_sanity"])
    )
    output = {
        "protocol": protocol["protocol"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "protocol_artifact": {
            "path": str(protocol_path),
            "sha256": file_sha256(protocol_path),
        },
        "result_artifact": {
            "path": str(result_path),
            "sha256": file_sha256(result_path),
        },
        "versions": versions,
        "ordinary_cells": len(rows),
        "metrics": {
            "primary_positive_fraction": primary_positive_fraction,
            "all_ranking_positive_fraction": ranking_positive_fraction,
            "adjacent_primary_positive_fraction": adjacent_positive_fraction,
            "adjacent_primary_mean_relative_percent": adjacent_mean,
            "cumulative_row_fraction": cumulative_fraction,
            "all_same_model_sanity": all_sanity,
            "fresh_absolute_floor": fresh_floor,
        },
        "negative_entries": [
            {
                "target_version": row["target_version"],
                "source_version": row["source_version"],
                "metric": metric,
                "relative_percent": row["relative_percent"][metric],
            }
            for row in rows
            for metric in RANKING_METRICS
            if row["relative_percent"][metric] <= 0
        ],
        "adjacent": adjacent,
        "cumulative": cumulative,
        "decision": {
            "passed": passed,
            "preferred_size_reached": versions >= int(acceptance["preferred_versions"]),
            "next": "extend_or_freeze" if passed else "revise_without_promotion",
        },
    }
    _atomic_json(Path(args.output), output)
    promote_from = args.promote_accepted_from
    promote_to = args.promote_accepted_to
    if bool(promote_from) != bool(promote_to):
        raise ValueError("KuaiRand promotion arguments differ")
    if promote_from:
        if not passed:
            raise RuntimeError("KuaiRand extended triangle failed promotion")
        source = Path(promote_from)
        target = Path(promote_to)
        if target.is_file() and file_sha256(target) != file_sha256(source):
            raise RuntimeError("KuaiRand promoted accepted artifact differs")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            shutil.copy2(source, target)
        output["promotion"] = {
            "source": str(source),
            "target": str(target),
            "sha256": file_sha256(target),
        }
        _atomic_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
