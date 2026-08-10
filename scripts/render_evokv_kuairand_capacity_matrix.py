from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from hstu_kvcache.streaming.kuairand_query_transition import file_sha256

METRICS = ("ndcg_at_5", "mrr", "hit_rate_at_5")
LABELS = {"ndcg_at_5": "NDCG@5", "mrr": "MRR", "hit_rate_at_5": "HR@5"}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def build_matrix(
    result_path: str | Path,
    first_version: int,
    final_version: int,
) -> dict[str, Any]:
    path = Path(result_path)
    source = json.loads(path.read_text())
    if source.get("status") != "complete" or not 1 <= first_version < final_version:
        raise ValueError("KuaiRand capacity matrix source differs")
    cells = {}
    for target in source.get("targets", []):
        target_version = int(target["target_version"])
        if not first_version < target_version <= final_version:
            continue
        for lineage in target["lineage"]:
            source_version = int(lineage["source_version"])
            if first_version <= source_version < target_version:
                cells[(target_version, source_version)] = lineage.get(
                    "holdout", lineage["summary"]
                )
    versions = final_version - first_version + 1
    if len(cells) != versions * (versions - 1) // 2:
        raise ValueError("KuaiRand capacity matrix triangle differs")
    matrices = {}
    summaries = {}
    absolute_endpoints = {}
    for metric in METRICS:
        matrix = []
        values = []
        adjacent = []
        negatives = []
        for target_version in range(first_version, final_version + 1):
            row = []
            for source_version in range(first_version, final_version + 1):
                if source_version == target_version:
                    row.append(0.0)
                elif source_version > target_version:
                    row.append(None)
                else:
                    summary = cells[(target_version, source_version)]
                    value = float(
                        summary["comparisons"]["recompute_over_reuse"][metric][
                            "relative_percent"
                        ]
                    )
                    row.append(value)
                    values.append(value)
                    if source_version == target_version - 1:
                        adjacent.append(value)
                    if value < 0:
                        negatives.append(
                            {
                                "target_version": target_version,
                                "source_version": source_version,
                                "relative_percent": value,
                            }
                        )
                    absolute_endpoints[f"{target_version}:{source_version}"] = {
                        "target_version": target_version,
                        "source_version": source_version,
                        "reuse": {
                            name: float(summary["endpoints"]["reuse"][name])
                            for name in METRICS
                        },
                        "recompute": {
                            name: float(summary["endpoints"]["recompute"][name])
                            for name in METRICS
                        },
                    }
            matrix.append(row)
        matrices[metric] = matrix
        summaries[metric] = {
            "cells": len(values),
            "positive_cells": sum(value > 0 for value in values),
            "negative_cells": len(negatives),
            "mean_relative_percent": statistics.fmean(values),
            "median_relative_percent": statistics.median(values),
            "minimum_relative_percent": min(values),
            "maximum_relative_percent": max(values),
            "adjacent_cells": len(adjacent),
            "positive_adjacent_cells": sum(value > 0 for value in adjacent),
            "mean_adjacent_relative_percent": statistics.fmean(adjacent),
            "minimum_adjacent_relative_percent": min(adjacent),
            "negative_cell_records": negatives,
        }
    return {
        "protocol": "evokv_kuairand_capacity_lift_matrix_v0",
        "status": "complete_development_evidence",
        "scientific_result": False,
        "formal_result": False,
        "definition": "100 * (Recompute - Reuse) / Reuse",
        "source": {"path": str(path), "sha256": file_sha256(path)},
        "first_version": first_version,
        "final_version": final_version,
        "matrices_relative_percent": matrices,
        "summaries": summaries,
        "absolute_endpoints": list(absolute_endpoints.values()),
    }


def render_markdown(document: dict[str, Any]) -> str:
    first_version = int(document["first_version"])
    final_version = int(document["final_version"])
    versions = list(range(first_version, final_version + 1))
    lines = [
        "# KuaiRand large capacity-lift Recompute-over-Reuse matrices",
        "",
        "Positive means Recompute is more accurate than Reuse. Values are unscaled relative percentages reported by the evaluator.",
        "",
    ]
    for metric in METRICS:
        lines.extend(
            [
                f"## {LABELS[metric]}",
                "",
                "| current \\ cache | "
                + " | ".join(f"theta{version}" for version in versions)
                + " |",
                "|---|" + "---:|" * len(versions),
            ]
        )
        for target_version, row in zip(
            versions,
            document["matrices_relative_percent"][metric],
            strict=True,
        ):
            cells = ["—" if value is None else f"{value:+.3f}%" for value in row]
            lines.append(f"| theta{target_version} | " + " | ".join(cells) + " |")
        summary = document["summaries"][metric]
        lines.extend(
            [
                "",
                f"All cells: {summary['positive_cells']}/{summary['cells']} positive; mean {summary['mean_relative_percent']:+.3f}%; median {summary['median_relative_percent']:+.3f}%.",
                f"Adjacent cells: {summary['positive_adjacent_cells']}/{summary['adjacent_cells']} positive; mean {summary['mean_adjacent_relative_percent']:+.3f}%; minimum {summary['minimum_adjacent_relative_percent']:+.3f}%.",
                "",
            ]
        )
    lines.append("Development evidence only; no K/V perturbation or metric scaling is applied.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--first-version", type=int, default=1)
    parser.add_argument("--final-version", type=int, required=True)
    args = parser.parse_args()
    output_path = Path(args.output)
    document = build_matrix(
        args.result,
        args.first_version,
        args.final_version,
    )
    markdown_path = output_path.with_suffix(".md")
    _atomic_text(markdown_path, render_markdown(document))
    document["markdown"] = {
        "path": str(markdown_path),
        "sha256": file_sha256(markdown_path),
    }
    _atomic_text(output_path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
