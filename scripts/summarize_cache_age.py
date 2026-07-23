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
    parser.add_argument(
        "--kuai",
        default="results/exposure/kuai_allages_streaming_control_seed[0-3].json",
    )
    parser.add_argument(
        "--qb",
        default=(
            "results/exposure/"
            "qb_fixed_horizon_allages_streaming_control_seed[0-3].json"
        ),
    )
    parser.add_argument(
        "--qk",
        default="results/exposure/qk_top5k_allages_streaming_control_seed[0-3].json",
    )
    parser.add_argument(
        "--output",
        default="results/exposure/cache_age_cross_dataset_summary.json",
    )
    return parser.parse_args()


def mean_ci95(values: np.ndarray) -> dict:
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "std": 0.0, "ci95": [mean, mean], "n": len(values)}
    std = float(values.std(ddof=1))
    half = float(t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    return {
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
        "n": len(values),
    }


def load_runs(pattern: str) -> tuple[list[str], list[dict]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths, [json.loads(Path(path).read_text()) for path in paths]


def contrast(pair: dict, name: str, metric: str) -> float:
    return float(pair["summary"]["contrasts"][name][metric])


def summarize_dataset(pattern: str) -> dict:
    paths, runs = load_runs(pattern)
    ages = [int(pair["model_t"]) for pair in runs[0]["pairs"]]
    if any([int(pair["model_t"]) for pair in run["pairs"]] != ages for run in runs):
        raise ValueError("cache ages differ across seeds")
    cache_rank = np.asarray(
        [
            [contrast(pair, "cache_maintenance_value", "best_rank") for pair in run["pairs"]]
            for run in runs
        ]
    )
    stream_rank = np.asarray(
        [
            [
                contrast(pair, "streaming_value_full_compute", "best_rank")
                for pair in run["pairs"]
            ]
            for run in runs
        ]
    )
    reuse_rank = np.asarray(
        [
            [
                contrast(pair, "streaming_value_full_reuse", "best_rank")
                for pair in run["pairs"]
            ]
            for run in runs
        ]
    )
    cache_ndcg = np.asarray(
        [
            [contrast(pair, "cache_maintenance_value", "ndcg@100") for pair in run["pairs"]]
            for run in runs
        ]
    )
    stream_ndcg = np.asarray(
        [
            [
                contrast(pair, "streaming_value_full_compute", "ndcg@100")
                for pair in run["pairs"]
            ]
            for run in runs
        ]
    )
    cache_utility = np.asarray(
        [
            [
                contrast(pair, "cache_maintenance_value", "rank_utility")
                for pair in run["pairs"]
            ]
            for run in runs
        ]
    )
    peak = np.maximum(cache_rank.max(axis=1), np.finfo(float).eps)
    normalized = cache_rank / peak[:, None]
    increments = np.diff(normalized, axis=1)
    transition_means = increments.mean(axis=0)
    selected = int(np.argmax(transition_means))
    endpoint_rank_tax = cache_rank[:, -1] / stream_rank[:, -1]
    endpoint_rank_retention = reuse_rank[:, -1] / stream_rank[:, -1]
    endpoint_ndcg_tax = cache_ndcg[:, -1] / stream_ndcg[:, -1]
    endpoint_rank_inflation = np.expm1(cache_utility[:, -1])
    peak_locations = np.argmax(increments, axis=1)
    location_counts = {
        f"{ages[index]}->{ages[index + 1]}": int(np.count_nonzero(peak_locations == index))
        for index in range(len(ages) - 1)
    }
    return {
        "source_files": paths,
        "num_seeds": len(runs),
        "ages": ages,
        "endpoint": {
            "best_rank_staleness_tax": mean_ci95(endpoint_rank_tax),
            "best_rank_reuse_retention": mean_ci95(endpoint_rank_retention),
            "ndcg100_staleness_tax": mean_ci95(endpoint_ndcg_tax),
            "multiplicative_rank_inflation": mean_ci95(endpoint_rank_inflation),
        },
        "trajectory": [
            {
                "age": age,
                "best_rank_maintenance": mean_ci95(cache_rank[:, index]),
                "normalized_to_seed_peak": mean_ci95(normalized[:, index]),
                "rank_utility_maintenance": mean_ci95(cache_utility[:, index]),
            }
            for index, age in enumerate(ages)
        ],
        "transitions": [
            {
                "from_age": ages[index],
                "to_age": ages[index + 1],
                "normalized_increment": mean_ci95(increments[:, index]),
                "positive_seeds": int(np.count_nonzero(increments[:, index] > 0)),
            }
            for index in range(len(ages) - 1)
        ],
        "exploratory_change_point": {
            "from_age": ages[selected],
            "to_age": ages[selected + 1],
            "normalized_increment": mean_ci95(increments[:, selected]),
            "positive_seeds": int(np.count_nonzero(increments[:, selected] > 0)),
            "largest_transition_location_counts": location_counts,
            "per_seed_abruptness": mean_ci95(increments.max(axis=1)),
            "uniform_four_transition_reference": 1.0 / (len(ages) - 1),
        },
    }


def main() -> None:
    args = parse_args()
    datasets = {
        "kuairand": summarize_dataset(args.kuai),
        "tenrec_qb": summarize_dataset(args.qb),
        "tenrec_qk": summarize_dataset(args.qk),
    }
    rank_taxes = [
        value["endpoint"]["best_rank_staleness_tax"]["mean"]
        for value in datasets.values()
    ]
    ndcg_taxes = [
        value["endpoint"]["ndcg100_staleness_tax"]["mean"]
        for value in datasets.values()
    ]
    result = {
        "protocol": "cache_age_cross_dataset_v1",
        "statistical_unit": "training_seed",
        "metric_definitions": {
            "best_rank_staleness_tax": (
                "(full_compute - full_reuse) / (full_compute - frozen)"
            ),
            "best_rank_reuse_retention": (
                "(full_reuse - frozen) / (full_compute - frozen)"
            ),
            "ndcg100_staleness_tax": (
                "(full_compute - full_reuse) / (full_compute - frozen)"
            ),
            "multiplicative_rank_inflation": (
                "exp(mean(log((stale_rank + 1) / (fresh_rank + 1)))) - 1"
            ),
            "normalized_to_seed_peak": (
                "BestRank maintenance at age / maximum positive maintenance in that seed"
            ),
        },
        "datasets": datasets,
        "cross_dataset_endpoint_range": {
            "best_rank_staleness_tax": [min(rank_taxes), max(rank_taxes)],
            "best_rank_max_over_min": max(rank_taxes) / min(rank_taxes),
            "ndcg100_staleness_tax": [min(ndcg_taxes), max(ndcg_taxes)],
            "ndcg100_max_over_min": max(ndcg_taxes) / min(ndcg_taxes),
        },
        "interpretation": {
            "change_points_are_exploratory": True,
            "fixed_age_is_insufficient": (
                "The largest mean transition occurs at a different cache age per dataset, "
                "and the largest per-seed transition is not fixed within a dataset."
            ),
        },
    }
    save_json(result, args.output)
    for name, value in datasets.items():
        endpoint = value["endpoint"]
        change = value["exploratory_change_point"]
        print(
            f"{name} rank_tax={endpoint['best_rank_staleness_tax']['mean']:.3f} "
            f"ndcg_tax={endpoint['ndcg100_staleness_tax']['mean']:.3f} "
            f"change={change['from_age']}->{change['to_age']} "
            f"jump={change['normalized_increment']['mean']:.3f} "
            f"positive={change['positive_seeds']}/{value['num_seeds']}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
