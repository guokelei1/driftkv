from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

RANKING_METRICS = ("mrr", "ndcg_at_5", "hit_rate_at_5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_loss(cell: dict[str, Any], split: str, metric: str) -> float:
    endpoints = cell[split]["endpoints"]
    fresh = float(endpoints["recompute"][metric])
    reuse = float(endpoints["reuse"][metric])
    if fresh <= 0.0:
        raise ValueError(f"non-positive Fresh endpoint for {metric}")
    return 100.0 * (fresh - reuse) / fresh


def _split_summary(
    cells: list[dict[str, Any]],
    split: str,
    metric: str,
    first_version: int,
    final_version: int,
) -> dict[str, Any]:
    ordinary = [
        cell
        for cell in cells
        if first_version + 1 <= int(cell["target_version"]) <= final_version
        and first_version
        <= int(cell["source_version"])
        < int(cell["target_version"])
    ]
    versions = final_version - first_version + 1
    expected = versions * (versions - 1) // 2
    if len(ordinary) != expected:
        raise ValueError("ordinary triangle cell count differs")
    values = [_bounded_loss(cell, split, metric) for cell in ordinary]
    adjacent = [
        cell
        for cell in ordinary
        if int(cell["source_version"]) == int(cell["target_version"]) - 1
    ]
    adjacent_values = [_bounded_loss(cell, split, metric) for cell in adjacent]
    by_age: dict[str, list[float]] = {}
    for cell in ordinary:
        age = str(int(cell["target_version"]) - int(cell["source_version"]))
        by_age.setdefault(age, []).append(_bounded_loss(cell, split, metric))
    accumulation = []
    for target in range(first_version + 2, final_version + 1):
        row = [cell for cell in ordinary if int(cell["target_version"]) == target]
        adjacent_cell = next(
            cell for cell in row if int(cell["source_version"]) == target - 1
        )
        older = [cell for cell in row if int(cell["source_version"]) < target - 1]
        strongest = max(older, key=lambda cell: _bounded_loss(cell, split, metric))
        adjacent_loss = _bounded_loss(adjacent_cell, split, metric)
        strongest_loss = _bounded_loss(strongest, split, metric)
        accumulation.append(
            {
                "target_version": target,
                "adjacent_loss_percent": adjacent_loss,
                "strongest_older_source_version": int(strongest["source_version"]),
                "strongest_older_loss_percent": strongest_loss,
                "increase_over_adjacent_percent_points": strongest_loss - adjacent_loss,
            }
        )
    return {
        "ordinary_cells": len(values),
        "positive_cells": sum(value > 0.0 for value in values),
        "positive_fraction": sum(value > 0.0 for value in values) / len(values),
        "mean_percent": statistics.fmean(values),
        "median_percent": statistics.median(values),
        "minimum_percent": min(values),
        "maximum_percent": max(values),
        "adjacent": {
            "cells": len(adjacent_values),
            "positive_cells": sum(value > 0.0 for value in adjacent_values),
            "mean_percent": statistics.fmean(adjacent_values),
            "minimum_percent": min(adjacent_values),
            "maximum_percent": max(adjacent_values),
            "values_percent": adjacent_values,
        },
        "mean_by_cache_age_percent": {
            age: statistics.fmean(age_values)
            for age, age_values in sorted(by_age.items(), key=lambda value: int(value[0]))
        },
        "accumulation": accumulation,
        "negative_cells": [
            {
                "target_version": int(cell["target_version"]),
                "source_version": int(cell["source_version"]),
                "loss_percent": _bounded_loss(cell, split, metric),
            }
            for cell in ordinary
            if _bounded_loss(cell, split, metric) <= 0.0
        ],
    }


def _matrix(
    cells: list[dict[str, Any]],
    split: str,
    metric: str,
    first_version: int,
    final_version: int,
) -> list[list[float | None]]:
    by_pair = {
        (int(cell["target_version"]), int(cell["source_version"])): _bounded_loss(
            cell, split, metric
        )
        for cell in cells
        if int(cell["source_version"]) >= first_version
    }
    return [
        [
            0.0
            if source == target
            else by_pair.get((target, source))
            if source < target
            else None
            for source in range(first_version, final_version + 1)
        ]
        for target in range(first_version, final_version + 1)
    ]


def _fresh_endpoints(
    cells: list[dict[str, Any]], split: str, first_version: int, final_version: int
) -> list[dict[str, Any]]:
    output = []
    for target in range(first_version, final_version + 1):
        cell = next(cell for cell in cells if int(cell["target_version"]) == target)
        output.append(
            {
                "target_version": target,
                **{
                    metric: float(cell[split]["endpoints"]["recompute"][metric])
                    for metric in RANKING_METRICS
                },
            }
        )
    return output


def _render(result: dict[str, Any]) -> str:
    split = result["selected_split"]
    metric = result["primary_metric"]
    matrix = result["matrix_percent"]
    first_version = result["first_version"]
    final_version = result["final_version"]
    versions = final_version - first_version + 1
    lines = [
        f"# KuaiRand {versions}-version bounded Reuse loss matrix",
        "",
        f"Split: `{split}`. Metric: `{metric}`. Each cell is `100 × (Fresh − Reuse) / Fresh`; positive means recomputation is better.",
        "",
        "| current \\ cache | "
        + " | ".join(
            f"theta{source}" for source in range(first_version, final_version + 1)
        )
        + " |",
        "|---|" + "---:|" * versions,
    ]
    for target, row in enumerate(matrix, start=first_version):
        rendered = ["—" if value is None else f"{value:+.2f}%" for value in row]
        lines.append(f"| theta{target} | " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "## Fresh absolute endpoints",
            "",
            "| model | MRR | NDCG@5 | HR@5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for endpoint in result["fresh_endpoints"]:
        lines.append(
            f"| theta{endpoint['target_version']} | {endpoint['mrr']:.6f} | "
            f"{endpoint['ndcg_at_5']:.6f} | {endpoint['hit_rate_at_5']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--split", choices=("tuning", "holdout"), default="holdout")
    parser.add_argument("--metric", choices=RANKING_METRICS, default="ndcg_at_5")
    parser.add_argument("--first-version", type=int, default=1)
    args = parser.parse_args()
    source_path = Path(args.result)
    source = json.loads(source_path.read_text())
    if source.get("status") != "complete_development_control":
        raise ValueError("KuaiRand gauge triangle result is incomplete")
    if not source.get("fresh_function_invariance", {}).get("passed"):
        raise ValueError("Fresh function invariance did not pass")
    cells = source["cells"]
    final_version = max(int(cell["target_version"]) for cell in cells)
    first_version = int(args.first_version)
    if not 1 <= first_version < final_version:
        raise ValueError("first version differs")
    transform = source.get("transform")
    if transform is None:
        transform = {
            "mode": "orthogonal_rotation",
            "step_radians": source["step_radians"],
        }
    output = {
        "protocol": "evokv_kuairand_bounded_reuse_loss_matrix_v0",
        "status": "complete_development_control",
        "scientific_result": False,
        "formal_result": False,
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "transform": transform,
        "fresh_function_invariance": source["fresh_function_invariance"],
        "first_version": first_version,
        "final_version": final_version,
        "selected_split": args.split,
        "primary_metric": args.metric,
        "definition": "100 * (Fresh - Reuse) / Fresh",
        "selection_summary": _split_summary(
            cells, "tuning", args.metric, first_version, final_version
        ),
        "report_summary": _split_summary(
            cells, args.split, args.metric, first_version, final_version
        ),
        "matrix_percent": _matrix(
            cells, args.split, args.metric, first_version, final_version
        ),
        "fresh_endpoints": _fresh_endpoints(
            cells, args.split, first_version, final_version
        ),
    }
    output_path = Path(args.output_json)
    table_path = Path(args.output_table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    table_path.write_text(_render(output))
    print(json.dumps(output["report_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
