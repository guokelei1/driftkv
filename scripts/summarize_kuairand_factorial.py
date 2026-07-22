from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from hstu_kvcache.utils import save_json

CELL_SPECS = {
    "baseline": {
        "label": "top5k_seq128_6l_h96",
        "method_pattern": "results/scaling/sequence_length_seed{seed}.json",
        "core_pattern": "results/validity/core6l_seed{seed}.json",
        "axis_value": 128,
    },
    "more_data": {
        "label": "top20k_seq256_6l_h96",
        "method_pattern": "results/scaling/factorial_more_data_method_seed{seed}.json",
        "core_pattern": "results/scaling/factorial_more_data_core_seed{seed}.json",
    },
    "larger_model": {
        "label": "top5k_seq128_12l_h192",
        "method_pattern": "results/scaling/factorial_larger_model_method_seed{seed}.json",
        "core_pattern": "results/scaling/factorial_larger_model_core_seed{seed}.json",
    },
    "both": {
        "label": "top20k_seq256_12l_h192",
        "method_pattern": "results/scaling/factorial_both_method_seed{seed}.json",
        "core_pattern": "results/scaling/factorial_both_core_seed{seed}.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output", default="results/scaling/kuairand_factorial_summary.json")
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def mean_interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        interval = [mean, mean]
    else:
        half = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array)))
        interval = [mean - half, mean + half]
    return {"mean": mean, "ci95": interval, "seed_values": array.tolist()}


def get_point(run: dict, axis_value: int | None) -> dict:
    if axis_value is None:
        if len(run["points"]) != 1:
            raise ValueError("Expected exactly one factorial point")
        return run["points"][0]
    return next(point for point in run["points"] if point["axis_value"] == axis_value)


def canonical_names(num_layers: int) -> dict[str, str]:
    suffix_depths = [
        max(1, round(num_layers / 3)),
        max(1, round(2 * num_layers / 3)),
        max(1, num_layers - 1),
    ]
    intervals = [f"interval_l{num_layers - depth + 1}_l{num_layers}" for depth in suffix_depths]
    return {
        "reuse": "reuse",
        "cheap": "cheap_all",
        "suffix_one_third": intervals[0],
        "suffix_two_thirds": intervals[1],
        "suffix_all_but_first": intervals[2],
        "full": "recompute",
    }


def config_summary(points: list[dict], num_items: int, names: dict[str, str]) -> dict:
    raw = [point["summary"]["configs"] for point in points]
    full_rank = [configs["recompute"]["gain_over_reuse"]["best_rank"] for configs in raw]
    full_ndcg = [configs["recompute"]["gain_over_reuse"]["ndcg@100"] for configs in raw]
    output = {}
    for canonical, source in names.items():
        configs = [value[source] for value in raw]
        rank = [value["gain_over_reuse"]["best_rank"] for value in configs]
        ndcg = [value["gain_over_reuse"]["ndcg@100"] for value in configs]
        recovery = [
            numerator / denominator if abs(denominator) > 1e-12 else float("nan")
            for numerator, denominator in zip(rank, full_rank, strict=True)
        ]
        rank_difference = [
            numerator - denominator
            for numerator, denominator in zip(rank, full_rank, strict=True)
        ]
        ndcg_difference = [
            numerator - denominator
            for numerator, denominator in zip(ndcg, full_ndcg, strict=True)
        ]
        output[canonical] = {
            "source_name": source,
            "cost_ratio": mean_interval(
                [value["migration_ratio_to_recompute"] for value in configs]
            ),
            "latency_ms_per_user": mean_interval(
                [value["migration_ms_per_user"] for value in configs]
            ),
            "best_rank_gain": mean_interval(rank),
            "best_rank_gain_catalog_fraction": mean_interval(
                [value / num_items for value in rank]
            ),
            "ndcg100_gain": mean_interval(ndcg),
            "paired_difference_from_full": {
                "best_rank_gain": mean_interval(rank_difference),
                "ndcg100_gain": mean_interval(ndcg_difference),
            },
            "cache_error_rel": mean_interval([value["cache_error_rel"] for value in configs]),
            "best_rank_recovery": mean_interval(recovery),
            "best_rank_recovery_ratio_of_means": float(np.mean(rank) / np.mean(full_rank)),
            "extra_state_ratio_to_kv": mean_interval(
                [value["extra_state_ratio_to_kv"] for value in configs]
            ),
        }
    return output


def validate_constant(values: list[object], name: str) -> object:
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{name} differs across seeds: {values}")
    return values[0]


def cell_summary(spec: dict, seeds: list[int]) -> dict:
    method_paths = [spec["method_pattern"].format(seed=seed) for seed in seeds]
    core_paths = [spec["core_pattern"].format(seed=seed) for seed in seeds]
    methods = [load_json(path) for path in method_paths]
    cores = [load_json(path) for path in core_paths]
    points = [get_point(run, spec.get("axis_value")) for run in methods]
    args = [run["args"] for run in cores]
    num_layers = int(validate_constant([value["num_layers"] for value in args], "num_layers"))
    num_items = int(validate_constant([run["data"]["num_items"] for run in cores], "num_items"))
    num_users = int(validate_constant([run["data"]["num_users"] for run in cores], "num_users"))
    num_parameters = int(
        validate_constant([run["num_parameters"] for run in cores], "num_parameters")
    )
    names = canonical_names(num_layers)
    configs = config_summary(points, num_items, names)
    return {
        "label": spec["label"],
        "data": {
            "num_users": num_users,
            "num_items": num_items,
            "max_items": int(validate_constant([value["max_items"] for value in args], "max_items")),
            "sequence_length": int(
                validate_constant([value["seq_len"] for value in args], "seq_len")
            ),
            "base_days": int(
                validate_constant([value["base_days"] for value in args], "base_days")
            ),
            "stream_days": len(cores[0]["data"]["stream_dates"]),
        },
        "model": {
            "num_layers": num_layers,
            "hidden_size": int(
                validate_constant([value["hidden_size"] for value in args], "hidden_size")
            ),
            "num_heads": int(
                validate_constant([value["num_heads"] for value in args], "num_heads")
            ),
            "head_dim": int(
                validate_constant([value["head_dim"] for value in args], "head_dim")
            ),
            "num_parameters": num_parameters,
        },
        "training": {
            "base_epochs": int(
                validate_constant([value["base_epochs"] for value in args], "base_epochs")
            ),
            "stream_epochs": int(
                validate_constant([value["stream_epochs"] for value in args], "stream_epochs")
            ),
            "final_base_loss": mean_interval([run["base_losses"][-1] for run in cores]),
            "runtime_seconds": mean_interval([run["runtime_seconds"] for run in cores]),
            "cumulative_dtheta_rel": mean_interval([point["dtheta_rel"] for point in points]),
        },
        "evaluation": {
            "n_users_per_seed": int(
                validate_constant([point["n_users"] for point in points], "n_users")
            ),
            "fresh_incremental_parity_max_abs": max(
                point["summary"]["fresh_incremental_parity_max_abs"] for point in points
            ),
            "optimized_full_cache_error_rel_max": max(
                point["summary"]["configs"]["recompute"]["cache_error_rel"] for point in points
            ),
        },
        "configs": configs,
        "source_files": {"core": core_paths, "method": method_paths},
    }


def main() -> None:
    args = parse_args()
    result = {
        "protocol": "kuairand_factorial_v1_seed_level_summary",
        "seeds": args.seeds,
        "factor_definition": {
            "data_scale": "top-5k/length-128 versus top-20k/length-256 bundle",
            "model_scale": "6-layer hidden-96 versus 12-layer hidden-192",
            "warning": "The data-scale factor changes retained catalog and active context together.",
        },
        "cells": {
            name: cell_summary(spec, args.seeds)
            for name, spec in CELL_SPECS.items()
        },
    }
    save_json(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
