from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from hstu_kvcache.utils import save_json

DATASETS = ("kuai", "qb", "qk")
TIERS = ("small", "medium", "large")
SEEDS = (0, 1, 2, 3)
METRICS = ("best_rank", "rank_utility", "ndcg@100")
PROTOCOLS = {
    "core": "motivation_capacity_v2_training",
    "control": "motivation_capacity_v2_streaming_control",
    "matrix": "motivation_capacity_v2_cache_version_matrix",
    "cost": "motivation_capacity_v2_operator_cost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="results/motivation_scale",
    )
    parser.add_argument(
        "--output",
        default="results/motivation_scale/capacity_v2_summary.json",
    )
    return parser.parse_args()


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    if array.size > 1:
        half_width = float(
            stats.t.ppf(0.975, array.size - 1)
            * std
            / math.sqrt(array.size)
        )
    else:
        half_width = 0.0
    return {
        "values": array.tolist(),
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "ci95": [mean - half_width, mean + half_width],
        "positive_seeds": int(np.sum(array > 0)),
        "negative_seeds": int(np.sum(array < 0)),
    }


def read_result(path: Path, protocol: str) -> dict:
    result = json.loads(path.read_text())
    if result.get("protocol") != protocol:
        raise ValueError(
            f"{path} has protocol {result.get('protocol')!r}, expected {protocol!r}"
        )
    return result


def result_path(
    input_dir: Path,
    dataset: str,
    tier: str,
    stage: str,
    seed: int,
) -> Path:
    suffix = {
        "core": "core",
        "control": "streaming_control",
        "matrix": "cache_version_matrix",
        "cost": "operator_cost",
    }[stage]
    return input_dir / f"{dataset}_{tier}_v2_{suffix}_seed{seed}.json"


def endpoint_pair(run: dict) -> dict:
    return next(pair for pair in run["pairs"] if pair["model_t"] == 11)


def metric_partition(pairs: list[dict], metric: str) -> dict:
    full = [
        float(
            pair["summary"]["contrasts"]["streaming_value_full_compute"][metric]
        )
        for pair in pairs
    ]
    reuse = [
        float(pair["summary"]["contrasts"]["streaming_value_full_reuse"][metric])
        for pair in pairs
    ]
    cache = [
        float(pair["summary"]["contrasts"]["cache_maintenance_value"][metric])
        for pair in pairs
    ]
    valid = [
        index for index, value in enumerate(full) if np.isfinite(value) and value > 0
    ]
    return {
        "streaming_value_full_compute": describe(full),
        "streaming_value_full_reuse": describe(reuse),
        "cache_maintenance_value": describe(cache),
        "reuse_fraction_of_streaming_value": describe(
            [reuse[index] / full[index] for index in valid]
        ),
        "staleness_tax": describe(
            [cache[index] / full[index] for index in valid]
        ),
        "staleness_tax_denominator_positive_seeds": len(valid),
    }


def point_tax(point: dict, metric: str) -> float:
    contrasts = point["summary"]["contrasts"]
    full = float(contrasts["streaming_value_full_compute"][metric])
    cache = float(contrasts["cache_maintenance_value"][metric])
    return cache / full if full > 0 else float("nan")


def matrix_metric(runs: list[dict], metric: str) -> dict:
    ages = [
        int(point["cache_age"])
        for point in sorted(runs[0]["points"], key=lambda item: item["cache_age"])
    ]
    taxes = {
        run["seed"]: {
            int(point["cache_age"]): point_tax(point, metric)
            for point in run["points"]
        }
        for run in runs
    }
    by_age = {
        str(age): describe(
            [
                taxes[run["seed"]][age]
                for run in runs
                if np.isfinite(taxes[run["seed"]][age])
            ]
        )
        for age in ages
    }
    correlations = []
    violations = []
    largest_per_seed = []
    for run in runs:
        seed = run["seed"]
        seed_taxes = np.asarray([taxes[seed][age] for age in ages])
        valid = np.isfinite(seed_taxes)
        correlations.append(
            float(stats.spearmanr(np.asarray(ages)[valid], seed_taxes[valid]).statistic)
        )
        violations.append(
            int(
                sum(
                    taxes[seed][older] < taxes[seed][younger]
                    for younger, older in zip(ages, ages[1:], strict=False)
                )
            )
        )
        transitions = [
            {
                "from_age": younger,
                "to_age": older,
                "tax_change": taxes[seed][older] - taxes[seed][younger],
            }
            for younger, older in zip(ages, ages[1:], strict=False)
            if np.isfinite(taxes[seed][younger])
            and np.isfinite(taxes[seed][older])
        ]
        selected = max(transitions, key=lambda item: item["tax_change"])
        largest_per_seed.append({"seed": seed, **selected})
    discovery = next(item for item in largest_per_seed if item["seed"] == 0)
    replication_changes = [
        taxes[seed][discovery["to_age"]] - taxes[seed][discovery["from_age"]]
        for seed in SEEDS[1:]
    ]
    return {
        "ages": ages,
        "by_age": by_age,
        "endpoint_staleness_tax": by_age[str(ages[-1])],
        "age_tax_spearman": describe(correlations),
        "monotonic_violations_per_seed": violations,
        "largest_transition_per_seed": largest_per_seed,
        "seed0_selected_transition": {
            **discovery,
            "replication_seeds_1_to_3": describe(replication_changes),
        },
    }


def training_coverage(core: dict) -> dict:
    base = core["base_training_coverage"][0]
    stream_targets = sum(
        record["eligible_targets"]
        for window in core["windows"]
        for record in window["training_coverage"]
    )
    stream_tokens = sum(
        record["tokens"]
        for window in core["windows"]
        for record in window["training_coverage"]
    )
    return {
        "base_sequences_per_epoch": int(base["sequences"]),
        "base_tokens_per_epoch": int(base["tokens"]),
        "base_eligible_targets_per_epoch": int(base["eligible_targets"]),
        "stream_tokens": int(stream_tokens),
        "stream_eligible_targets": int(stream_targets),
    }


def cost_summary(run: dict) -> dict:
    point = next(
        point
        for point in run["points"]
        if point["axis"] == "sequence_length"
        and point["batch_size"] == 32
        and point["seq_len"] == 128
    )
    cheap = point["configs"]["cheap_all"]
    full = point["configs"]["recompute"]
    return {
        "measurement_seed": int(run["seed"]),
        "batch_size": 32,
        "sequence_length": 128,
        "full_recompute_ms": float(full["latency_ms"]),
        "cheap_refresh_ms": float(cheap["latency_ms"]),
        "cheap_ratio_to_full": float(cheap["ratio_to_recompute"]),
        "full_users_per_second": float(full["users_per_second"]),
        "cheap_users_per_second": float(cheap["users_per_second"]),
    }


def summarize_cell(input_dir: Path, dataset: str, tier: str) -> dict:
    paths = {
        stage: [
            result_path(input_dir, dataset, tier, stage, seed)
            for seed in (SEEDS if stage != "cost" else (0,))
        ]
        for stage in PROTOCOLS
    }
    missing = [
        str(path)
        for stage_paths in paths.values()
        for path in stage_paths
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    results = {
        stage: [
            read_result(path, PROTOCOLS[stage]) for path in stage_paths
        ]
        for stage, stage_paths in paths.items()
    }
    for stage in ("core", "control", "matrix"):
        seeds = sorted(int(run["seed"] if "seed" in run else run["args"]["seed"]) for run in results[stage])
        if seeds != list(SEEDS):
            raise ValueError(f"{dataset}/{tier}/{stage} has seeds {seeds}")
    if any(len(run["windows"]) != 11 for run in results["core"]):
        raise ValueError(f"{dataset}/{tier} core result does not have 11 windows")
    if any(len(run["pairs"]) != 6 for run in results["control"]):
        raise ValueError(f"{dataset}/{tier} control result does not have 6 pairs")
    if any(len(run["points"]) != 11 for run in results["matrix"]):
        raise ValueError(f"{dataset}/{tier} matrix result does not have 11 points")
    core = results["core"][0]
    coverage = training_coverage(core)
    control_pairs = [endpoint_pair(run) for run in results["control"]]
    return {
        "dataset": dataset,
        "tier": tier,
        "seeds": list(SEEDS),
        "data_and_model": {
            "num_users": int(core["data"]["num_users"]),
            "num_items": int(core["data"]["num_items"]),
            "num_parameters": int(core["num_parameters"]),
            "num_layers": int(core["args"]["num_layers"]),
            "hidden_size": int(core["args"]["hidden_size"]),
            **coverage,
        },
        "theta11_value_partition": {
            metric: metric_partition(control_pairs, metric) for metric in METRICS
        },
        "fixed_theta11_cache_age": {
            metric: matrix_metric(results["matrix"], metric) for metric in METRICS
        },
        "resident_gpu_cost_seed0": cost_summary(results["cost"][0]),
        "source_files": {
            stage: [str(path) for path in stage_paths]
            for stage, stage_paths in paths.items()
        },
    }


def scale_contrasts(cells: dict[str, dict]) -> dict:
    output = {}
    for dataset in DATASETS:
        small = cells[f"{dataset}_small"]["theta11_value_partition"]
        large = cells[f"{dataset}_large"]["theta11_value_partition"]
        output[dataset] = {}
        for metric in METRICS:
            small_values = small[metric]["staleness_tax"]["values"]
            large_values = large[metric]["staleness_tax"]["values"]
            output[dataset][metric] = {
                "large_minus_small_staleness_tax": describe(
                    [
                        large_value - small_value
                        for small_value, large_value in zip(
                            small_values,
                            large_values,
                            strict=True,
                        )
                    ]
                )
            }
    return output


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    cells = {
        f"{dataset}_{tier}": summarize_cell(input_dir, dataset, tier)
        for dataset in DATASETS
        for tier in TIERS
    }
    result = {
        "protocol": "motivation_capacity_v2_seed_summary",
        "statistical_unit": "independent_training_seed",
        "selection_rule": (
            "seed 0 froze the complete 3x3 matrix; seeds 1-3 replicate every cell"
        ),
        "num_training_runs": len(DATASETS) * len(TIERS) * len(SEEDS),
        "num_core_control_matrix_artifacts": (
            len(DATASETS) * len(TIERS) * len(SEEDS) * 3
        ),
        "metrics": {
            "best_rank_staleness_tax": (
                "(full_compute - reuse) / (full_compute - frozen), "
                "computed within seed with rank direction corrected"
            ),
            "full_compute_reference": (
                "cache-fidelity reference, not a guaranteed ranking upper bound"
            ),
        },
        "cells": cells,
        "paired_scale_contrasts": scale_contrasts(cells),
    }
    save_json(result, args.output)
    for key, cell in cells.items():
        model = cell["data_and_model"]
        best_rank = cell["theta11_value_partition"]["best_rank"]
        cost = cell["resident_gpu_cost_seed0"]
        print(
            f"{key:12s} users={model['num_users']:5d} "
            f"params={model['num_parameters']:8d} "
            f"full={best_rank['streaming_value_full_compute']['mean']:8.2f} "
            f"reuse={best_rank['streaming_value_full_reuse']['mean']:8.2f} "
            f"cache={best_rank['cache_maintenance_value']['mean']:8.2f} "
            f"tax={best_rank['staleness_tax']['mean']:7.3f} "
            f"cost={cost['full_recompute_ms']:6.3f}ms"
        )
    print(args.output)


if __name__ == "__main__":
    main()
