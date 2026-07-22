"""Aggregate repaired motivation runs with seed, rather than user, as the unit."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t

from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="results/validity/core_seed[0-3].json")
    parser.add_argument("--output", default="results/validity/multiseed_summary.json")
    return parser.parse_args()


def mean_ci95(
    values: list[float],
    bounds: tuple[float, float] | None = None,
) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return {"mean": mean, "std": 0.0, "ci95": [mean, mean], "n": len(array)}
    std = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * std / math.sqrt(len(array)))
    interval = [mean - half, mean + half]
    if bounds is not None:
        interval = [max(bounds[0], interval[0]), min(bounds[1], interval[1])]
    return {"mean": mean, "std": std, "ci95": interval, "n": len(array)}


def mode_window(run: dict, window: int, mode: str) -> dict:
    value = run["windows"][window]
    if mode == "cumulative_theta0":
        return value["cumulative_theta0"]
    return value


def aggregate_mode(runs: list[dict], mode: str) -> dict:
    num_windows = min(len(run["windows"]) for run in runs)
    metrics = ("best_rank", "mean_rank", "mrr", "ndcg@10", "ndcg@100", "hit@10", "hit@100")
    output: dict[str, object] = {"windows": []}
    for window in range(num_windows):
        cells = [mode_window(run, window, mode)["summary"] for run in runs]
        record: dict[str, object] = {
            "window": window + 1,
            "eval_date": runs[0]["windows"][window]["eval_date"],
            "step_dtheta_rel": mean_ci95(
                [run["windows"][window]["step_dtheta_rel"] for run in runs]
            ),
            "cumulative_dtheta_rel": mean_ci95(
                [run["windows"][window]["cumulative_dtheta_rel"] for run in runs]
            ),
        }
        for metric in metrics:
            record[metric] = {
                "fresh": mean_ci95([cell[metric]["fresh"] for cell in cells]),
                "stale": mean_ci95([cell[metric]["stale"] for cell in cells]),
                "gain": mean_ci95([cell[metric]["gain"] for cell in cells]),
            }
        record["kv_drift_rel"] = mean_ci95(
            [cell["fidelity"]["kv_drift_rel"] for cell in cells]
        )
        record["drift_rank_gain_rho"] = mean_ci95(
            [cell["drift_quality_correlation"]["rank_utility_gain"]["rho"] for cell in cells],
            bounds=(-1.0, 1.0),
        )
        record["selection"] = {}
        for budget_index, budget in enumerate((0.1, 0.2, 0.5)):
            selection = [cell["selection"][budget_index] for cell in cells]
            record["selection"][str(budget)] = {
                "drift_minus_random": mean_ci95(
                    [
                        item["drift_select"] - item["random_select_mean"]
                        for item in selection
                    ]
                ),
                "oracle_minus_random": mean_ci95(
                    [
                        item["oracle_select"] - item["random_select_mean"]
                        for item in selection
                    ]
                ),
            }
        output["windows"].append(record)

    output["seed_level"] = {}
    for metric in metrics:
        seed_means = [
            float(
                np.mean(
                    [
                        mode_window(run, window, mode)["summary"][metric]["gain"]
                        for window in range(num_windows)
                    ]
                )
            )
            for run in runs
        ]
        output["seed_level"][metric] = {
            "gain": mean_ci95(seed_means),
            "per_seed": seed_means,
        }
    seed_rhos = []
    for run in runs:
        gains = [
            mode_window(run, window, mode)["summary"]["best_rank"]["gain"]
            for window in range(num_windows)
        ]
        ages = [run["windows"][window]["cumulative_dtheta_rel"] for window in range(num_windows)]
        rho, _ = spearmanr(ages, gains)
        seed_rhos.append(float(rho))
    output["seed_level"]["age_best_rank_gain_spearman"] = {
        "rho": mean_ci95(seed_rhos, bounds=(-1.0, 1.0)),
        "per_seed": seed_rhos,
    }
    all_cell_rhos = [
        mode_window(run, window, mode)["summary"]["drift_quality_correlation"][
            "rank_utility_gain"
        ]["rho"]
        for run in runs
        for window in range(num_windows)
    ]
    output["cell_level_descriptive"] = {
        "drift_rank_gain_rho": mean_ci95(all_cell_rhos, bounds=(-1.0, 1.0)),
    }
    return output


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.inputs))
    if not files:
        raise FileNotFoundError(args.inputs)
    runs = [json.loads(Path(path).read_text()) for path in files]
    result = {
        "statistical_unit": "training_seed",
        "source_files": files,
        "num_seeds": len(runs),
        "one_step": aggregate_mode(runs, "one_step"),
        "cumulative_theta0": aggregate_mode(runs, "cumulative_theta0"),
    }
    save_json(result, args.output)
    print(json.dumps(result["one_step"]["seed_level"], indent=2))
    print(json.dumps(result["cumulative_theta0"]["seed_level"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
