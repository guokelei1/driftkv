from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = ("ndcg_at_5", "mrr", "hit_rate_at_5")
LABELS = {"ndcg_at_5": "NDCG@5", "mrr": "MRR", "hit_rate_at_5": "HR@5"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cells(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [cell for target in source["targets"] for cell in target["lineage"]]


def _loss(cell: dict[str, Any], split: str, metric: str) -> float:
    endpoints = cell[split]["endpoints"]
    fresh = float(endpoints["recompute"][metric])
    reuse = float(endpoints["reuse"][metric])
    if fresh <= 0.0:
        raise ValueError("non-positive Fresh endpoint")
    return 100.0 * (fresh - reuse) / fresh


def _ordinary(
    cells: list[dict[str, Any]], first_version: int, final_version: int
) -> list[dict[str, Any]]:
    selected = [
        cell
        for cell in cells
        if first_version + 1 <= int(cell["target_version"]) <= final_version
        and first_version <= int(cell["source_version"]) < int(cell["target_version"])
    ]
    versions = final_version - first_version + 1
    if len(selected) != versions * (versions - 1) // 2:
        raise ValueError("persistent triangle cell count differs")
    return selected


def _summary(
    cells: list[dict[str, Any]],
    split: str,
    metric: str,
    first_version: int,
    final_version: int,
) -> dict[str, Any]:
    selected = _ordinary(cells, first_version, final_version)
    values = [_loss(cell, split, metric) for cell in selected]
    adjacent = [
        cell
        for cell in selected
        if int(cell["source_version"]) == int(cell["target_version"]) - 1
    ]
    adjacent_values = [_loss(cell, split, metric) for cell in adjacent]
    by_age: dict[str, list[float]] = {}
    for cell in selected:
        age = str(int(cell["target_version"]) - int(cell["source_version"]))
        by_age.setdefault(age, []).append(_loss(cell, split, metric))
    return {
        "ordinary_cells": len(values),
        "positive_cells": sum(value > 0.0 for value in values),
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
        "negative_cells": [
            {
                "target_version": int(cell["target_version"]),
                "source_version": int(cell["source_version"]),
                "loss_percent": _loss(cell, split, metric),
            }
            for cell in selected
            if _loss(cell, split, metric) <= 0.0
        ],
    }


def _matrix(
    cells: list[dict[str, Any]],
    split: str,
    metric: str,
    first_version: int,
    final_version: int,
) -> list[list[float | None]]:
    selected = {
        (int(cell["target_version"]), int(cell["source_version"])): _loss(
            cell, split, metric
        )
        for cell in _ordinary(cells, first_version, final_version)
    }
    return [
        [
            0.0
            if source == target
            else selected[(target, source)]
            if source < target
            else None
            for source in range(first_version, final_version + 1)
        ]
        for target in range(first_version, final_version + 1)
    ]


def _fresh(
    cells: list[dict[str, Any]], split: str, first_version: int, final_version: int
) -> list[dict[str, Any]]:
    output = []
    for version in range(first_version, final_version + 1):
        cell = next(cell for cell in cells if int(cell["target_version"]) == version)
        output.append(
            {
                "version": version,
                **{
                    metric: float(cell[split]["endpoints"]["recompute"][metric])
                    for metric in METRICS
                },
            }
        )
    return output


def _render_matrix(
    matrix: list[list[float | None]], first_version: int, metric: str
) -> list[str]:
    versions = len(matrix)
    lines = [
        f"## {LABELS[metric]}",
        "",
        "| current \\ cache | "
        + " | ".join(f"theta{version}" for version in range(first_version, first_version + versions))
        + " |",
        "|---|" + "---:|" * versions,
    ]
    for target, row in enumerate(matrix, start=first_version):
        values = ["—" if value is None else f"{value:+.2f}%" for value in row]
        lines.append(f"| theta{target} | " + " | ".join(values) + " |")
    return lines


def _render(output: dict[str, Any]) -> str:
    lines = [
        "# KuaiRand persistent bounded Reuse-loss matrices",
        "",
        f"Split: `{output['split']}`. Each cell is `100 × (Fresh − Reuse) / Fresh`; positive means recomputation is better.",
        "",
    ]
    for metric in METRICS:
        lines.extend(
            _render_matrix(output["matrices_percent"][metric], output["first_version"], metric)
        )
        lines.append("")
    lines.extend(
        [
            "## Fresh absolute endpoints",
            "",
            "| model | NDCG@5 | MRR | HR@5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for endpoint in output["fresh_endpoints"]:
        lines.append(
            f"| theta{endpoint['version']} | {endpoint['ndcg_at_5']:.6f} | "
            f"{endpoint['mrr']:.6f} | {endpoint['hit_rate_at_5']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--split", choices=("tuning", "holdout"), default="holdout")
    parser.add_argument("--first-version", type=int, default=5)
    args = parser.parse_args()
    source_path = Path(args.result)
    source = json.loads(source_path.read_text())
    if source.get("status") != "complete":
        raise ValueError("persistent result is incomplete")
    cells = _cells(source)
    final_version = max(int(target["target_version"]) for target in source["targets"])
    if not 1 <= args.first_version < final_version:
        raise ValueError("first version differs")
    output = {
        "protocol": "evokv_kuairand_persistent_bounded_matrix_v0",
        "status": "complete_development_evidence",
        "scientific_result": False,
        "formal_result": False,
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "split": args.split,
        "definition": "100 * (Fresh - Reuse) / Fresh",
        "first_version": args.first_version,
        "final_version": final_version,
        "summaries": {
            metric: _summary(cells, args.split, metric, args.first_version, final_version)
            for metric in METRICS
        },
        "matrices_percent": {
            metric: _matrix(cells, args.split, metric, args.first_version, final_version)
            for metric in METRICS
        },
        "fresh_endpoints": _fresh(cells, args.split, args.first_version, final_version),
    }
    output_path = Path(args.output_json)
    table_path = Path(args.output_table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    table_path.write_text(_render(output))
    print(json.dumps(output["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
