from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
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
    summaries = [find_pair(run, model_t)["summary"] for run in runs]
    cells = [summary["configs"] for summary in summaries]
    names = set.intersection(*(set(cell) for cell in cells))
    full_rank = [cell["recompute"]["gain_over_reuse"]["best_rank"] for cell in cells]
    full_ndcg = [cell["recompute"]["gain_over_reuse"]["ndcg@100"] for cell in cells]
    full_rank_summary = mean_ci95(full_rank)
    full_ndcg_summary = mean_ci95(full_ndcg)
    configs = {}
    for name in names:
        values = [cell[name] for cell in cells]
        rank = [value["gain_over_reuse"]["best_rank"] for value in values]
        ndcg = [value["gain_over_reuse"]["ndcg@100"] for value in values]
        rank_summary = mean_ci95(rank)
        ndcg_summary = mean_ci95(ndcg)
        configs[name] = {
            "migration_ratio_to_recompute": mean_ci95(
                [value["migration_ratio_to_recompute"] for value in values]
            ),
            "best_rank_gain": rank_summary,
            "ndcg100_gain": ndcg_summary,
            "best_rank_minus_full": mean_ci95(
                [value - full for value, full in zip(rank, full_rank, strict=True)]
            ),
            "ndcg100_minus_full": mean_ci95(
                [value - full for value, full in zip(ndcg, full_ndcg, strict=True)]
            ),
            "best_rank_recovery_ratio_of_means": (
                rank_summary["mean"] / full_rank_summary["mean"]
                if full_rank_summary["mean"] != 0.0
                else None
            ),
            "ndcg100_recovery_ratio_of_means": (
                ndcg_summary["mean"] / full_ndcg_summary["mean"]
                if full_ndcg_summary["mean"] != 0.0
                else None
            ),
        }
    return {
        "model_t": model_t,
        "num_seeds": len(runs),
        "per_layer_stale_cache_error_rel": [
            mean_ci95(
                [
                    summary["per_layer_stale_cache_error_rel"][layer]
                    for summary in summaries
                ]
            )
            for layer in range(len(summaries[0]["per_layer_stale_cache_error_rel"]))
        ],
        "full_best_rank_denominator_identifiable": (
            full_rank_summary["ci95"][0] > 0.0
            or full_rank_summary["ci95"][1] < 0.0
        ),
        "full_ndcg100_denominator_identifiable": (
            full_ndcg_summary["ci95"][0] > 0.0
            or full_ndcg_summary["ci95"][1] < 0.0
        ),
        "configs": dict(
            sorted(
                configs.items(),
                key=lambda item: item[1]["migration_ratio_to_recompute"]["mean"],
            )
        ),
    }


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.inputs))
    if not files:
        raise FileNotFoundError(args.inputs)
    runs = [json.loads(Path(path).read_text()) for path in files]
    model_ts = sorted(set.intersection(*(set(run["model_ts"]) for run in runs)))
    result = {
        "protocol": "ordered_exposure_method_summary_v1",
        "statistical_unit": "training_seed",
        "source_files": files,
        "seeds": [run["seed"] for run in runs],
        "cells": [aggregate_cell(runs, model_t) for model_t in model_ts],
    }
    save_json(result, args.output)
    for cell in result["cells"]:
        print(f"theta=0->{cell['model_t']} seeds={cell['num_seeds']}")
        for name, value in cell["configs"].items():
            print(
                f"  {name:>18} "
                f"time={value['migration_ratio_to_recompute']['mean']:.3f} "
                f"rank={value['best_rank_gain']['mean']:.2f} "
                f"ndcg100={value['ndcg100_gain']['mean']:.5f}"
            )
    print(args.output)


if __name__ == "__main__":
    main()
