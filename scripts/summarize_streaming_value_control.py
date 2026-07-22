"""Aggregate frozen, full-reuse, and full-compute controls over training seeds."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from hstu_kvcache.utils import save_json

METRICS = (
    "mrr",
    "ndcg@10",
    "ndcg@100",
    "hit@10",
    "hit@100",
    "best_rank",
    "mean_rank",
    "rank_utility",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        default="results/validity/streaming_control6l_seed[0-3].json",
    )
    parser.add_argument(
        "--output",
        default="results/validity/streaming_control6l_summary.json",
    )
    return parser.parse_args()


def mean_ci95(values: list[float]) -> dict[str, float | int | list[float]]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return {"mean": mean, "std": 0.0, "ci95": [mean, mean], "n": len(array)}
    std = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * std / math.sqrt(len(array)))
    return {
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
        "n": len(array),
    }


def find_pair(run: dict, model_t: int) -> dict:
    return next(pair for pair in run["pairs"] if pair["model_t"] == model_t)


def aggregate_cell(runs: list[dict], model_t: int) -> dict:
    pairs = [find_pair(run, model_t) for run in runs]
    condition_names = pairs[0]["summary"]["conditions"]
    contrast_names = pairs[0]["summary"]["contrasts"]
    conditions = {
        condition: {
            metric: mean_ci95(
                [pair["summary"]["conditions"][condition][metric] for pair in pairs]
            )
            for metric in METRICS
        }
        for condition in condition_names
    }
    contrasts = {
        contrast: {
            metric: mean_ci95(
                [pair["summary"]["contrasts"][contrast][metric] for pair in pairs]
            )
            for metric in METRICS
        }
        for contrast in contrast_names
    }
    retention = {}
    for metric in ("best_rank", "ndcg@100"):
        total = [
            pair["summary"]["contrasts"]["streaming_value_full_compute"][metric]
            for pair in pairs
        ]
        reused = [
            pair["summary"]["contrasts"]["streaming_value_full_reuse"][metric]
            for pair in pairs
        ]
        maintained = [
            pair["summary"]["contrasts"]["cache_maintenance_value"][metric]
            for pair in pairs
        ]
        retention[metric] = {
            "reuse_fraction_of_streaming_value": mean_ci95(
                [numerator / denominator for numerator, denominator in zip(reused, total, strict=True)]
            ),
            "cache_fraction_of_streaming_value": mean_ci95(
                [numerator / denominator for numerator, denominator in zip(maintained, total, strict=True)]
            ),
        }
    return {
        "model_t": model_t,
        "eval_date": pairs[0]["eval_date"],
        "n_users_per_seed": pairs[0]["n_users"],
        "dtheta_rel": mean_ci95([pair["dtheta_rel"] for pair in pairs]),
        "conditions": conditions,
        "contrasts": contrasts,
        "streaming_value_partition": retention,
        "current_incremental_parity_max_abs": max(
            pair["summary"]["current_incremental_parity_max_abs"] for pair in pairs
        ),
        "frozen_incremental_parity_max_abs": max(
            pair["summary"]["frozen_incremental_parity_max_abs"] for pair in pairs
        ),
    }


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.inputs))
    if not files:
        raise FileNotFoundError(args.inputs)
    runs = [json.loads(Path(path).read_text()) for path in files]
    model_ts = runs[0]["model_ts"]
    result = {
        "protocol": "streaming_value_control_v1_seed_summary",
        "statistical_unit": "training_seed",
        "source_files": files,
        "num_seeds": len(runs),
        "conditions": runs[0]["conditions"],
        "cells": [aggregate_cell(runs, model_t) for model_t in model_ts],
    }
    save_json(result, args.output)
    for cell in result["cells"]:
        print(f"theta=0->{cell['model_t']}")
        for name, value in cell["contrasts"].items():
            print(
                f"  {name:>28} "
                f"rank={value['best_rank']['mean']:.2f} "
                f"rank_ci={value['best_rank']['ci95']} "
                f"ndcg100={value['ndcg@100']['mean']:.5f} "
                f"ndcg_ci={value['ndcg@100']['ci95']}"
            )
    print(args.output)


if __name__ == "__main__":
    main()
