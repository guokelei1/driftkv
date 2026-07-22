from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output", default="results/scaling/multiaxis_summary.json")
    return parser.parse_args()


def mean_interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        interval = [mean, mean]
    else:
        half = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array)))
        interval = [mean - half, mean + half]
    return {"mean": mean, "ci95": interval, "seed_values": array.tolist()}


def config_stats(configs: list[dict]) -> dict:
    full_rank = [value["recompute"]["gain_over_reuse"]["best_rank"] for value in configs]
    output = {}
    for name in configs[0]:
        rank = [value[name]["gain_over_reuse"]["best_rank"] for value in configs]
        ndcg = [value[name]["gain_over_reuse"]["ndcg@100"] for value in configs]
        cost = [value[name]["migration_ratio_to_recompute"] for value in configs]
        error = [value[name]["cache_error_rel"] for value in configs]
        recovery = [
            numerator / denominator if abs(denominator) > 1e-12 else float("nan")
            for numerator, denominator in zip(rank, full_rank, strict=True)
        ]
        output[name] = {
            "cost_ratio": mean_interval(cost),
            "best_rank_gain": mean_interval(rank),
            "ndcg100_gain": mean_interval(ndcg),
            "cache_error_rel": mean_interval(error),
            "best_rank_recovery": mean_interval(recovery),
            "best_rank_recovery_ratio_of_means": float(np.mean(rank) / np.mean(full_rank)),
        }
    return output


def load_axis(seeds: list[int], name: str) -> list[dict]:
    return [json.loads(Path(f"results/scaling/{name}_seed{seed}.json").read_text()) for seed in seeds]


def sequence_summary(seeds: list[int]) -> dict:
    runs = load_axis(seeds, "sequence_length")
    lengths = [point["axis_value"] for point in runs[0]["points"]]
    output = {}
    for length in lengths:
        points = [
            next(point for point in run["points"] if point["axis_value"] == length)
            for run in runs
        ]
        output[str(length)] = {
            "dtheta_rel": mean_interval([point["dtheta_rel"] for point in points]),
            "configs": config_stats([point["summary"]["configs"] for point in points]),
        }
    return output


def update_summary(seeds: list[int]) -> dict:
    runs = load_axis(seeds, "update_magnitude")
    alphas = [point["axis_value"] for point in runs[0]["points"]]
    output = {}
    for alpha in alphas:
        points = [
            next(point for point in run["points"] if point["axis_value"] == alpha)
            for run in runs
        ]
        output[str(alpha)] = {
            "dtheta_rel": mean_interval([point["dtheta_rel"] for point in points]),
            "configs": config_stats([point["summary"]["configs"] for point in points]),
        }
    return output


def depth_config_names(depth: int) -> dict[str, str]:
    one_third = max(1, round(depth / 3))
    two_thirds = max(1, round(2 * depth / 3))
    all_but_first = max(1, depth - 1)
    interval = lambda suffix: f"interval_l{depth - suffix + 1}_l{depth}"
    return {
        "cheap_all": "cheap_all",
        "suffix_one_third": interval(one_third),
        "suffix_two_thirds": interval(two_thirds),
        "suffix_all_but_first": interval(all_but_first),
        "recompute": "recompute",
    }


def depth_runs(seeds: list[int], depth: int) -> list[dict]:
    if depth == 6:
        runs = load_axis(seeds, "sequence_length")
        return [
            next(point for point in run["points"] if point["axis_value"] == 128)
            for run in runs
        ]
    return [
        json.loads(Path(f"results/scaling/depth{depth}_method_seed{seed}.json").read_text())["points"][0]
        for seed in seeds
    ]


def depth_summary(seeds: list[int]) -> dict:
    output = {}
    for depth in (3, 6, 9):
        points = depth_runs(seeds, depth)
        names = depth_config_names(depth)
        raw_configs = [point["summary"]["configs"] for point in points]
        canonical = [
            {canonical_name: configs[source_name] for canonical_name, source_name in names.items()}
            for configs in raw_configs
        ]
        normalized = []
        for configs in canonical:
            normalized.append({
                name: value
                for name, value in configs.items()
            })
            normalized[-1]["recompute"] = configs["recompute"]
        output[str(depth)] = {
            "dtheta_rel": mean_interval([point["dtheta_rel"] for point in points]),
            "configs": config_stats(normalized),
        }
    return output


def movielens_summary(seeds: list[int]) -> dict:
    runs = load_axis(seeds, "movielens")
    output = {}
    for version in (1, 2):
        points = [next(value for value in run["versions"] if value["version"] == version) for run in runs]
        configs = [point["summary"]["configs"] for point in points]
        candidate = [point["summary"]["candidate_20"]["configs"] for point in points]
        candidate_output = {}
        for name in candidate[0]:
            candidate_output[name] = {
                "best_rank_gain": mean_interval(
                    [value[name]["gain_over_reuse"]["best_rank"] for value in candidate]
                ),
                "ndcg10_gain": mean_interval(
                    [value[name]["gain_over_reuse"]["ndcg@10"] for value in candidate]
                ),
            }
        output[str(version)] = {
            "dtheta_rel": mean_interval([point["dtheta_rel"] for point in points]),
            "configs": config_stats(configs),
            "candidate_20": candidate_output,
            "parity_max_abs": max(
                point["summary"]["fresh_incremental_parity_max_abs"] for point in points
            ),
            "optimized_full_kv_max_abs": max(
                point["summary"]["optimized_full_kv_max_abs"] for point in points
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    result = {
        "protocol": "scaling_v1_seed_level_summary",
        "seeds": args.seeds,
        "sequence_length": sequence_summary(args.seeds),
        "update_magnitude": update_summary(args.seeds),
        "model_depth": depth_summary(args.seeds),
        "movielens": movielens_summary(args.seeds),
    }
    save_json(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
