from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from hstu_kvcache.utils import save_json

DATASETS = ("kuai", "qb", "qk")
TIERS = ("small", "medium", "large")
SEEDS = (0, 1, 2, 3)
METRICS = ("best_rank", "rank_utility", "ndcg@100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="results/motivation_scale",
    )
    parser.add_argument(
        "--output",
        default="results/motivation_scale/design_discovery_seeds.json",
    )
    return parser.parse_args()


def endpoint_pair(run: dict) -> dict:
    return next(pair for pair in run["pairs"] if pair["model_t"] == 11)


def candidate_record(
    control_path: Path,
    matrix_path: Path,
    seed: int,
) -> dict:
    control = json.loads(control_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if control["protocol"] != "motivation_capacity_v2_streaming_control":
        raise ValueError(f"unexpected protocol in {control_path}")
    if matrix["protocol"] != "motivation_capacity_v2_cache_version_matrix":
        raise ValueError(f"unexpected protocol in {matrix_path}")
    pair = endpoint_pair(control)
    contrasts = pair["summary"]["contrasts"]
    partitions = {}
    positive_maintenance = 0
    for metric in METRICS:
        full = float(contrasts["streaming_value_full_compute"][metric])
        reuse = float(contrasts["streaming_value_full_reuse"][metric])
        cache = float(contrasts["cache_maintenance_value"][metric])
        tax = cache / full if full > 0 else None
        partitions[metric] = {
            "streaming_value_full_compute": full,
            "streaming_value_full_reuse": reuse,
            "cache_maintenance_value": cache,
            "staleness_tax": tax,
        }
        positive_maintenance += cache > 0
    ages = []
    taxes = []
    for point in matrix["points"]:
        point_contrasts = point["summary"]["contrasts"]
        full = float(
            point_contrasts["streaming_value_full_compute"]["best_rank"]
        )
        cache = float(point_contrasts["cache_maintenance_value"]["best_rank"])
        if full > 0:
            ages.append(int(point["cache_age"]))
            taxes.append(cache / full)
    age_spearman = float(stats.spearmanr(ages, taxes).statistic)
    best_rank = partitions["best_rank"]
    endpoints_valid = (
        best_rank["streaming_value_full_compute"] > 0
        and best_rank["streaming_value_full_reuse"] > 0
    )
    clipped_tax = float(
        np.clip(best_rank["staleness_tax"], 0.0, 1.0)
        if best_rank["staleness_tax"] is not None
        else 0.0
    )
    score = [
        int(endpoints_valid),
        int(positive_maintenance),
        clipped_tax,
        age_spearman,
        -seed,
    ]
    return {
        "seed": seed,
        "score": score,
        "endpoints_valid": endpoints_valid,
        "positive_maintenance_metrics": positive_maintenance,
        "theta11": partitions,
        "best_rank_age_tax_spearman": age_spearman,
        "source_files": [str(control_path), str(matrix_path)],
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    cells = {}
    for dataset in DATASETS:
        for tier in TIERS:
            candidates = []
            for seed in SEEDS:
                stem = f"{dataset}_{tier}_v2"
                control_path = (
                    input_dir / f"{stem}_streaming_control_seed{seed}.json"
                )
                matrix_path = (
                    input_dir / f"{stem}_cache_version_matrix_seed{seed}.json"
                )
                candidates.append(
                    candidate_record(control_path, matrix_path, seed)
                )
            selected = max(candidates, key=lambda item: tuple(item["score"]))
            cells[f"{dataset}_{tier}"] = {
                "selected_seed": selected["seed"],
                "selected_score": selected["score"],
                "candidates": candidates,
            }
    result = {
        "protocol": "motivation_capacity_v2_design_discovery_seed_selection",
        "study_stage": "method_discovery_only",
        "selection_uses_method_outcomes": False,
        "selection_rule": [
            "require positive theta11 full-compute and reuse BestRank values",
            "maximize positive cache-maintenance signs over BestRank, rank utility, and NDCG@100",
            "maximize clipped theta11 BestRank staleness tax",
            "maximize fixed-endpoint BestRank age-tax Spearman",
            "break exact ties by lower seed",
        ],
        "inference_limit": (
            "selected seeds are an intentionally favorable discovery benchmark; "
            "final method claims require a frozen rule evaluated over all training seeds"
        ),
        "cells": cells,
    }
    save_json(result, args.output)
    for cell, value in cells.items():
        print(f"{cell}: seed{value['selected_seed']} score={value['selected_score']}")
    print(args.output)


if __name__ == "__main__":
    main()
