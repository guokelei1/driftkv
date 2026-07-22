"""Aggregate held-out interval validation with training seed as the unit."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from hstu_kvcache.utils import save_json

COMPARISONS = (
    ("same_cost_one_block", "interval_l3_l4", "interval_l5_l6"),
    ("same_cost_two_blocks", "interval_l3_l5", "interval_l4_l6"),
    ("near_cost_early_three", "interval_l1_l3", "interval_l3_l6"),
    ("near_cost_early_four", "interval_l1_l4", "interval_l2_l6"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discovery",
        default="results/validity/interval_oracle_seed0.json",
    )
    parser.add_argument(
        "--validation-inputs",
        default="results/validity/interval_validation_seed[1-3].json",
    )
    parser.add_argument(
        "--output",
        default="results/validity/interval_validation_summary.json",
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


def aggregate_configs(cells: list[dict]) -> dict:
    names = set.intersection(*(set(cell["configs"]) for cell in cells))
    output = {}
    for name in names:
        values = [cell["configs"][name] for cell in cells]
        output[name] = {
            "migration_ratio_to_recompute": mean_ci95(
                [value["migration_ratio_to_recompute"] for value in values]
            ),
            "best_rank_gain": mean_ci95(
                [value["gain_over_reuse"]["best_rank"] for value in values]
            ),
            "ndcg100_gain": mean_ci95(
                [value["gain_over_reuse"]["ndcg@100"] for value in values]
            ),
            "best_rank_recovery": mean_ci95(
                [value["quality_recovery"]["best_rank"] for value in values]
            ),
            "ndcg100_recovery": mean_ci95(
                [value["quality_recovery"]["ndcg@100"] for value in values]
            ),
            "pareto_frequency": {
                "best_rank": sum(name in cell["pareto_best_rank"] for cell in cells),
                "ndcg100": sum(name in cell["pareto_ndcg100"] for cell in cells),
                "joint": sum(name in cell["pareto_joint"] for cell in cells),
                "n": len(cells),
            },
        }
    return dict(
        sorted(
            output.items(),
            key=lambda item: item[1]["migration_ratio_to_recompute"]["mean"],
        )
    )


def paired_comparisons(cells: list[dict]) -> dict:
    output = {}
    for label, non_suffix, suffix in COMPARISONS:
        record = {
            "non_suffix": non_suffix,
            "suffix": suffix,
        }
        for metric in ("best_rank", "ndcg@100"):
            values = [
                cell["configs"][non_suffix]["gain_over_reuse"][metric]
                - cell["configs"][suffix]["gain_over_reuse"][metric]
                for cell in cells
            ]
            record[f"{metric}_non_suffix_minus_suffix"] = mean_ci95(values)
            record[f"{metric}_direction"] = [
                1 if value > 0 else -1 if value < 0 else 0
                for value in values
            ]
        output[label] = record
    return output


def aggregate_runs(runs: list[dict], model_t: int) -> dict:
    cells = [find_pair(run, model_t)["summary"] for run in runs]
    return {
        "model_t": model_t,
        "num_seeds": len(runs),
        "configs": aggregate_configs(cells),
        "paired_comparisons": paired_comparisons(cells),
    }


def terminal_comparison(discovery: dict) -> dict:
    output = {}
    for top_n in range(7):
        values = [
            pair["summary"]["legacy_suffix_comparison"][f"suffix_{top_n}"]
            for pair in discovery["pairs"]
        ]
        output[f"suffix_{top_n}"] = {
            "optimized_over_legacy_mean": float(
                np.mean([value["optimized_over_legacy"] for value in values])
            ),
            "optimized_ratio_to_legacy_recompute_mean": float(
                np.mean(
                    [
                        value["optimized_ratio_to_legacy_recompute"]
                        for value in values
                    ]
                )
            ),
            "kv_max_abs": max(value["kv_max_abs"] for value in values),
        }
    return output


def main() -> None:
    args = parse_args()
    discovery = json.loads(Path(args.discovery).read_text())
    validation_files = sorted(glob.glob(args.validation_inputs))
    if not validation_files:
        raise FileNotFoundError(args.validation_inputs)
    validation = [json.loads(Path(path).read_text()) for path in validation_files]
    all_runs = [discovery, *validation]
    result = {
        "protocol": "interval_validation_summary_v1",
        "discovery_seed": discovery["seed"],
        "validation_seeds": [run["seed"] for run in validation],
        "validation_source_files": validation_files,
        "terminal_projection": terminal_comparison(discovery),
        "heldout_validation": [
            aggregate_runs(validation, model_t)
            for model_t in discovery["model_ts"]
        ],
        "all_seed_descriptive": [
            aggregate_runs(all_runs, model_t)
            for model_t in discovery["model_ts"]
        ],
    }
    save_json(result, args.output)
    for cell in result["heldout_validation"]:
        print(f"theta=0->{cell['model_t']} heldout_seeds={cell['num_seeds']}")
        for name, value in cell["configs"].items():
            print(
                f"  {name:>18} "
                f"time={value['migration_ratio_to_recompute']['mean']:.3f} "
                f"rank={value['best_rank_gain']['mean']:.2f} "
                f"ndcg100={value['ndcg100_gain']['mean']:.5f} "
                f"pareto={value['pareto_frequency']['joint']}/{cell['num_seeds']}"
            )
    print(args.output)


if __name__ == "__main__":
    main()
