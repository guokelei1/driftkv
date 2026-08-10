from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from hstu_kvcache.streaming.kuairand_query_transition import file_sha256


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_selected_matrix(
    result_path: str | Path,
    first_target: int,
    versions: int,
    metric: str,
) -> dict[str, Any]:
    path = Path(result_path)
    result = json.loads(path.read_text())
    last_target = first_target + versions - 1
    targets = {
        int(target["target_version"]): target for target in result.get("targets", [])
    }
    if (
        result.get("status") != "complete"
        or versions != 8
        or metric not in ("mrr", "ndcg_at_5", "hit_rate_at_5")
        or set(range(first_target, last_target + 1)) - set(targets)
    ):
        raise ValueError("KuaiRand selected matrix result differs")
    cache_versions = list(range(first_target - 1, last_target))
    target_versions = list(range(first_target, last_target + 1))
    matrix: list[list[float | None]] = []
    cells = []
    adjacent = []
    for target_version in target_versions:
        by_source = {
            int(lineage["source_version"]): lineage
            for lineage in targets[target_version]["lineage"]
        }
        row: list[float | None] = []
        for source_version in cache_versions:
            if source_version >= target_version:
                row.append(None)
                continue
            lineage = by_source.get(source_version)
            if lineage is None:
                raise ValueError("KuaiRand selected matrix lineage differs")
            summary = lineage.get("holdout", lineage["summary"])
            value = float(
                summary["comparisons"]["recompute_over_reuse"][metric][
                    "relative_percent"
                ]
            )
            endpoints = summary["endpoints"]
            row.append(value)
            cell = {
                "target_version": target_version,
                "source_version": source_version,
                "cache_age": target_version - source_version,
                "relative_percent": value,
                "reuse_absolute": float(endpoints["reuse"][metric]),
                "recompute_absolute": float(endpoints["recompute"][metric]),
            }
            cells.append(cell)
            if source_version == target_version - 1:
                adjacent.append(cell)
        matrix.append(row)
    values = np.asarray([cell["relative_percent"] for cell in cells], dtype=np.float64)
    adjacent_values = np.asarray(
        [cell["relative_percent"] for cell in adjacent], dtype=np.float64
    )
    negative = [cell for cell in cells if cell["relative_percent"] < 0]
    rows_with_older_peak = 0
    rows_with_age_gain = 0
    for target_version in target_versions[1:]:
        row_cells = [cell for cell in cells if cell["target_version"] == target_version]
        adjacent_value = next(
            cell["relative_percent"]
            for cell in row_cells
            if cell["source_version"] == target_version - 1
        )
        older_values = [
            cell["relative_percent"]
            for cell in row_cells
            if cell["source_version"] < target_version - 1
        ]
        rows_with_older_peak += int(max(older_values) > adjacent_value)
        rows_with_age_gain += int(max(older_values) >= adjacent_value + 1.0)
    return {
        "status": "complete",
        "protocol": "evokv_kuairand_selected_eight_version_matrix_v0",
        "scientific_result": False,
        "formal_result": False,
        "source_result": {
            "path": str(path),
            "sha256": file_sha256(path),
            "round_id": result["round_id"],
        },
        "metric": metric,
        "positive_direction": "recompute_better_than_reuse",
        "target_versions": target_versions,
        "cache_versions": cache_versions,
        "matrix_relative_percent": matrix,
        "cells": cells,
        "summary": {
            "cells": len(cells),
            "positive_cells": int(np.count_nonzero(values > 0)),
            "negative_cells": len(negative),
            "mean_cell_relative_percent": float(values.mean()),
            "median_cell_relative_percent": float(np.median(values)),
            "minimum_cell_relative_percent": float(values.min()),
            "maximum_cell_relative_percent": float(values.max()),
            "adjacent_cells": len(adjacent),
            "positive_adjacent_cells": int(np.count_nonzero(adjacent_values > 0)),
            "mean_adjacent_relative_percent": float(adjacent_values.mean()),
            "minimum_adjacent_relative_percent": float(adjacent_values.min()),
            "maximum_adjacent_relative_percent": float(adjacent_values.max()),
            "rows_with_older_cache_worse_than_adjacent": rows_with_older_peak,
            "rows_with_at_least_one_point_age_gain": rows_with_age_gain,
        },
        "negative_cells": negative,
    }


def render_markdown(document: dict[str, Any]) -> str:
    metric = document["metric"].replace("ndcg_at_5", "NDCG@5").replace(
        "hit_rate_at_5", "HR@5"
    ).upper()
    cache_versions = document["cache_versions"]
    lines = [
        f"# KuaiRand selected 8×8 {metric} Recompute-over-Reuse matrix",
        "",
        "Positive means Recompute is more accurate than Reuse. Values are relative percentages.",
        "",
        "| current \\ cache | "
        + " | ".join(f"theta{version}" for version in cache_versions)
        + " |",
        "|---|" + "---:|" * len(cache_versions),
    ]
    for target_version, row in zip(
        document["target_versions"], document["matrix_relative_percent"], strict=True
    ):
        cells = ["—" if value is None else f"{value:+.3f}%" for value in row]
        lines.append(f"| theta{target_version} | " + " | ".join(cells) + " |")
    summary = document["summary"]
    lines.extend(
        [
            "",
            f"Adjacent: {summary['positive_adjacent_cells']}/{summary['adjacent_cells']} positive, mean {summary['mean_adjacent_relative_percent']:+.3f}%, minimum {summary['minimum_adjacent_relative_percent']:+.3f}%.",
            f"All cells: {summary['positive_cells']}/{summary['cells']} positive, mean {summary['mean_cell_relative_percent']:+.3f}%, median {summary['median_cell_relative_percent']:+.3f}%.",
            f"Age accumulation: {summary['rows_with_older_cache_worse_than_adjacent']}/7 later rows contain an older cache with larger loss than the adjacent cache; {summary['rows_with_at_least_one_point_age_gain']}/7 exceed it by at least one percentage point.",
            "",
            "Development evidence only; no score scaling or K/V coordinate perturbation is applied.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--first-target", type=int, default=2)
    parser.add_argument("--versions", type=int, default=8)
    parser.add_argument("--metric", default="ndcg_at_5")
    args = parser.parse_args()
    output = Path(args.output)
    document = build_selected_matrix(
        args.result, args.first_target, args.versions, args.metric
    )
    markdown_path = output.with_suffix(".md")
    _atomic_text(markdown_path, render_markdown(document))
    document["markdown"] = {
        "path": str(markdown_path),
        "sha256": file_sha256(markdown_path),
    }
    _atomic_json(output, document)
    print(json.dumps(document["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
