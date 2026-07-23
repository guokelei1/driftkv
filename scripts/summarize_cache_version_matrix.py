from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from hstu_kvcache.utils import save_json

METRICS = ("best_rank", "mean_rank", "rank_utility", "ndcg@100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset name and quoted glob as NAME=PATTERN",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    result = {
        "values": array.tolist(),
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }
    if array.size > 1:
        half_width = float(
            stats.t.ppf(0.975, array.size - 1)
            * array.std(ddof=1)
            / math.sqrt(array.size)
        )
        result["ci95"] = [result["mean"] - half_width, result["mean"] + half_width]
    else:
        result["ci95"] = [result["mean"], result["mean"]]
    return result


def parse_dataset(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"dataset must use NAME=PATTERN: {spec}")
    name, pattern = spec.split("=", 1)
    if not name or not pattern:
        raise ValueError(f"dataset must use NAME=PATTERN: {spec}")
    return name, pattern


def point_values(point: dict, metric: str) -> tuple[float, float]:
    contrasts = point["summary"]["contrasts"]
    stream = contrasts["streaming_value_full_compute"][metric]
    cache = contrasts["cache_maintenance_value"][metric]
    return float(stream), float(cache)


def summarize_metric(runs: list[dict], ages: list[int], metric: str) -> dict:
    by_age = {}
    seed_tax = {}
    for age in ages:
        streams = []
        cache_gaps = []
        taxes = []
        for run in runs:
            point = next(point for point in run["points"] if point["cache_age"] == age)
            stream, cache = point_values(point, metric)
            streams.append(stream)
            cache_gaps.append(cache)
            taxes.append(cache / stream if stream > 0 else float("nan"))
        valid_taxes = [value for value in taxes if np.isfinite(value)]
        seed_tax[age] = taxes
        by_age[str(age)] = {
            "streaming_value": describe(streams),
            "cache_maintenance_gap": describe(cache_gaps),
            "staleness_tax": describe(valid_taxes),
            "staleness_tax_ratio_of_means": (
                float(np.mean(cache_gaps) / np.mean(streams))
                if np.mean(streams) > 0
                else None
            ),
        }
    transitions = []
    for younger, older in zip(ages, ages[1:], strict=False):
        changes = [
            older_value - younger_value
            for younger_value, older_value in zip(
                seed_tax[younger],
                seed_tax[older],
                strict=True,
            )
            if np.isfinite(younger_value) and np.isfinite(older_value)
        ]
        transitions.append(
            {
                "from_age": younger,
                "to_age": older,
                "tax_change": describe(changes),
                "positive_seeds": int(sum(value > 0 for value in changes)),
            }
        )
    largest = max(transitions, key=lambda item: item["tax_change"]["mean"])
    endpoint = by_age[str(ages[-1])]
    endpoint_mean = endpoint["staleness_tax"]["mean"]
    first_crossings = {}
    for threshold in (0.1, 0.2, 0.3):
        crossings = []
        for seed_index in range(len(runs)):
            crossing = next(
                (
                    age
                    for age in ages
                    if np.isfinite(seed_tax[age][seed_index])
                    and seed_tax[age][seed_index] >= threshold
                ),
                None,
            )
            crossings.append(crossing)
        first_crossings[str(threshold)] = crossings
    monotonic_violations = [
        sum(
            seed_tax[older][seed_index] < seed_tax[younger][seed_index]
            for younger, older in zip(ages, ages[1:], strict=False)
        )
        for seed_index in range(len(runs))
    ]
    per_seed_largest_transitions = []
    for seed_index, run in enumerate(runs):
        candidates = [
            {
                "from_age": younger,
                "to_age": older,
                "tax_change": seed_tax[older][seed_index]
                - seed_tax[younger][seed_index],
            }
            for younger, older in zip(ages, ages[1:], strict=False)
            if np.isfinite(seed_tax[younger][seed_index])
            and np.isfinite(seed_tax[older][seed_index])
        ]
        selected = max(candidates, key=lambda item: item["tax_change"])
        selected["seed"] = run["seed"]
        endpoint_tax = seed_tax[ages[-1]][seed_index]
        selected["share_of_endpoint_tax"] = (
            selected["tax_change"] / endpoint_tax
            if np.isfinite(endpoint_tax) and endpoint_tax > 0
            else None
        )
        per_seed_largest_transitions.append(selected)
    return {
        "by_age": by_age,
        "endpoint": endpoint,
        "transitions": transitions,
        "largest_mean_transition": largest,
        "largest_step_share_of_endpoint": (
            largest["tax_change"]["mean"] / endpoint_mean
            if endpoint_mean > 0
            else None
        ),
        "first_crossing_age": first_crossings,
        "monotonic_violations_per_seed": monotonic_violations,
        "per_seed_largest_transition": per_seed_largest_transitions,
    }


def summarize_dataset(name: str, pattern: str) -> dict:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    runs = [json.loads(Path(path).read_text()) for path in paths]
    protocol = {run["protocol"] for run in runs}
    current_t = {run["current_t"] for run in runs}
    if len(protocol) != 1 or len(current_t) != 1:
        raise ValueError(f"{name} mixes protocols or current versions")
    age_sets = [
        tuple(sorted(point["cache_age"] for point in run["points"])) for run in runs
    ]
    if len(set(age_sets)) != 1:
        raise ValueError(f"{name} has inconsistent cache ages")
    ages = list(age_sets[0])
    result = {
        "protocol": next(iter(protocol)),
        "paths": paths,
        "seeds": [run["seed"] for run in runs],
        "current_t": next(iter(current_t)),
        "n_users": [run["n_users"] for run in runs],
        "ages": ages,
        "metrics": {
            metric: summarize_metric(runs, ages, metric) for metric in METRICS
        },
    }
    endpoint_points = [
        next(point for point in run["points"] if point["cache_age"] == ages[-1])
        for run in runs
    ]
    rank_regrets = []
    log_rank_regrets = []
    harm_rates = []
    for point in endpoint_points:
        regrets = np.asarray(
            [
                record["conditions"]["reuse"]["best_rank"]
                - record["conditions"]["full_compute"]["best_rank"]
                for record in point["per_user"]
            ],
            dtype=np.float64,
        )
        log_regrets = np.asarray(
            [
                math.log1p(record["conditions"]["reuse"]["best_rank"])
                - math.log1p(record["conditions"]["full_compute"]["best_rank"])
                for record in point["per_user"]
            ],
            dtype=np.float64,
        )
        rank_regrets.append(
            {
                "median": float(np.median(regrets)),
                "p75": float(np.quantile(regrets, 0.75)),
                "p90": float(np.quantile(regrets, 0.9)),
            }
        )
        log_rank_regrets.append(
            {
                "median": float(np.median(log_regrets)),
                "p75": float(np.quantile(log_regrets, 0.75)),
                "p90": float(np.quantile(log_regrets, 0.9)),
            }
        )
        harm_rates.append(float(np.mean(regrets > 0)))
    result["endpoint_user_diagnostics"] = {
        "rank_regret_per_seed": rank_regrets,
        "log_rank_regret_per_seed": log_rank_regrets,
        "harm_rate": describe(harm_rates),
    }
    return result


def main() -> None:
    args = parse_args()
    datasets = {
        name: summarize_dataset(name, pattern)
        for name, pattern in map(parse_dataset, args.dataset)
    }
    cross_dataset = {}
    for metric in METRICS:
        endpoint_taxes = {
            name: result["metrics"][metric]["endpoint"]["staleness_tax"]["mean"]
            for name, result in datasets.items()
        }
        all_positive = all(value > 0 for value in endpoint_taxes.values())
        cross_dataset[metric] = {
            "endpoint_staleness_tax": endpoint_taxes,
            "all_positive": all_positive,
            "max_over_min": (
                max(endpoint_taxes.values()) / min(endpoint_taxes.values())
                if all_positive
                else None
            ),
        }
    result = {
        "protocol": "cache_version_matrix_cross_dataset_summary_v1",
        "primary_metric": {
            "name": "best_rank_staleness_tax",
            "definition": "(full_compute - reuse) / (full_compute - frozen), with rank direction corrected",
            "interpretation": "fraction of streaming-training value lost to a model-version-stale cache",
        },
        "datasets": datasets,
        "cross_dataset": cross_dataset,
    }
    save_json(result, args.output)
    for name, dataset in datasets.items():
        metric = dataset["metrics"]["best_rank"]
        endpoint = metric["endpoint"]["staleness_tax"]
        transition = metric["largest_mean_transition"]
        print(
            f"{name}: endpoint={endpoint['mean']:.3f} "
            f"ci=[{endpoint['ci95'][0]:.3f},{endpoint['ci95'][1]:.3f}] "
            f"jump={transition['from_age']}->{transition['to_age']} "
            f"delta={transition['tax_change']['mean']:.3f} "
            f"positive={transition['positive_seeds']}/{transition['tax_change']['n']}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
