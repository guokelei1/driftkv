from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

METRICS = (
    "sampled_cross_entropy",
    "hit_rate_at_10",
    "ndcg_at_10",
    "mean_reciprocal_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != value:
            raise FileExistsError(f"quality chain summary differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    training = json.loads(args.training_result.read_text())
    if (
        training.get("status") != "complete"
        or training.get("downstream_d1_d2_gate_passed") is not True
        or len(training.get("updates", [])) not in {3, 4}
    ):
        raise ValueError("quality chain training result is incomplete")
    cells = sorted(args.cell_root.glob("theta*_to_theta*.json"))
    if len(cells) != 3:
        raise ValueError("quality chain requires three diagnostic cells")
    rows = []
    bindings = []
    for path in cells:
        value = json.loads(path.read_text())
        if (
            value.get("status") != "complete"
            or value.get("world_size") != 2
            or value.get("evaluation_kind") != "prequential"
            or value.get("args", {}).get("include_frozen_control") is not True
        ):
            raise ValueError(f"quality chain cell differs: {path}")
        target = value["quality_by_negative_count"]["999"]["methods"]
        frozen = value["frozen_quality_by_negative_count"]["999"][
            "methods"
        ]
        frozen_metrics = frozen["all_exact"]["recommendation"]
        reuse_metrics = target["all_reuse"]["recommendation"]
        exact_metrics = target["all_exact"]["recommendation"]
        edge = value["edge"]
        row = {
            "edge": (
                f"theta{edge['source_version']}_to_"
                f"theta{edge['target_version']}"
            ),
            "positive_targets": int(exact_metrics["positive_targets"]),
            "cache_relative_error": float(
                target["all_reuse"]["cache_fidelity"][
                    "relative_error_mean"
                ]
            ),
            "reuse_score_cosine_to_exact": float(
                reuse_metrics["score_cosine_to_exact"]
            ),
            "reuse_top10_overlap_with_exact": float(
                reuse_metrics["top10_overlap_with_exact"]
            ),
            "metrics": {},
        }
        for metric in METRICS:
            frozen_value = float(frozen_metrics[metric])
            reuse_value = float(reuse_metrics[metric])
            exact_value = float(exact_metrics[metric])
            if metric == "sampled_cross_entropy":
                streaming_utility = frozen_value - exact_value
                maintenance_gain = reuse_value - exact_value
            else:
                streaming_utility = exact_value - frozen_value
                maintenance_gain = exact_value - reuse_value
            row["metrics"][metric] = {
                "frozen": frozen_value,
                "reuse": reuse_value,
                "exact": exact_value,
                "streaming_update_utility": streaming_utility,
                "exact_over_reuse_gain": maintenance_gain,
            }
        rows.append(row)
        bindings.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    ce_update = [
        row["metrics"]["sampled_cross_entropy"][
            "streaming_update_utility"
        ]
        for row in rows
    ]
    ce_maintenance = [
        row["metrics"]["sampled_cross_entropy"][
            "exact_over_reuse_gain"
        ]
        for row in rows
    ]
    summary = {
        "schema": "evokv_qk_quality_chain_summary_v0",
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "interpretation": {
            "all_edges_target_exact_ce_better_than_frozen": all(
                value > 0 for value in ce_update
            ),
            "all_edges_target_exact_ce_better_than_reuse": all(
                value > 0 for value in ce_maintenance
            ),
            "minimum_streaming_update_ce_utility": min(ce_update),
            "minimum_exact_over_reuse_ce_gain": min(ce_maintenance),
            "decision_boundary": (
                "inspect this baseline-only result before fitting or "
                "selecting any EvoKV migration policy"
            ),
        },
        "training": {
            "path": str(args.training_result),
            "sha256": sha256(args.training_result),
            "stack_identity": training["stack_identity"],
            "total_wall_seconds": training["execution"][
                "total_wall_seconds"
            ],
        },
        "cells": bindings,
        "edges": rows,
        "full_kv_payloads_retained": 0,
    }
    atomic_text(
        args.output,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    columns = [
        "edge",
        "positive_targets",
        "frozen_ce",
        "reuse_ce",
        "exact_ce",
        "streaming_update_ce_utility",
        "exact_over_reuse_ce_gain",
        "frozen_ndcg",
        "reuse_ndcg",
        "exact_ndcg",
        "cache_relative_error",
        "reuse_top10_overlap_with_exact",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        ce = row["metrics"]["sampled_cross_entropy"]
        ndcg = row["metrics"]["ndcg_at_10"]
        values = [
            row["edge"],
            row["positive_targets"],
            ce["frozen"],
            ce["reuse"],
            ce["exact"],
            ce["streaming_update_utility"],
            ce["exact_over_reuse_gain"],
            ndcg["frozen"],
            ndcg["reuse"],
            ndcg["exact"],
            row["cache_relative_error"],
            row["reuse_top10_overlap_with_exact"],
        ]
        lines.append("\t".join(str(value) for value in values))
    atomic_text(args.tsv, "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tsv": str(args.tsv),
                **summary["interpretation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
