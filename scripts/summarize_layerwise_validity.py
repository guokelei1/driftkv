"""Seed-level aggregation for validity-v1 layerwise cache migration runs."""

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
    parser.add_argument("--inputs", default="results/validity/layerwise_seed[0-3].json")
    parser.add_argument("--output", default="results/validity/layerwise_multiseed_summary.json")
    return parser.parse_args()


def mean_ci95(values: list[float]) -> dict[str, float | int | list[float]]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return {"mean": mean, "std": 0.0, "ci95": [mean, mean], "n": len(array)}
    std = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * std / math.sqrt(len(array)))
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half], "n": len(array)}


def find_pair(run: dict, mode: str, model_t: int) -> dict:
    return next(
        pair
        for pair in run["pairs"]
        if pair["mode"] == mode and pair["model_t"] == model_t
    )


def pareto_configs(configs: dict, gain_metric: str) -> list[str]:
    names = list(configs)
    output = []
    for name in names:
        cost = configs[name]["migration_ratio_to_recompute"]["mean"]
        gain = configs[name]["gain_over_reuse"][gain_metric]["mean"]
        dominated = any(
            other != name
            and configs[other]["migration_ratio_to_recompute"]["mean"] <= cost
            and configs[other]["gain_over_reuse"][gain_metric]["mean"] >= gain
            and (
                configs[other]["migration_ratio_to_recompute"]["mean"] < cost
                or configs[other]["gain_over_reuse"][gain_metric]["mean"] > gain
            )
            for other in names
        )
        if not dominated:
            output.append(name)
    return output


def aggregate_cell(runs: list[dict], mode: str, model_t: int) -> dict:
    pairs = [find_pair(run, mode, model_t) for run in runs]
    config_names = list(pairs[0]["summary"]["configs"])
    metrics = (
        "best_rank",
        "mean_rank",
        "mrr",
        "ndcg@10",
        "ndcg@100",
        "hit@10",
        "hit@100",
        "rank_utility",
    )
    configs = {}
    for name in config_names:
        values = [pair["summary"]["configs"][name] for pair in pairs]
        cheap_values = [pair["summary"]["configs"]["cheap_all"] for pair in pairs]
        full_values = [pair["summary"]["configs"]["recompute"] for pair in pairs]
        gains = {
            metric: mean_ci95([value["gain_over_reuse"][metric] for value in values])
            for metric in metrics
        }
        configs[name] = {
            "gain_over_reuse": gains,
            "incremental_gain_over_cheap": {
                metric: mean_ci95(
                    [
                        value["gain_over_reuse"][metric]
                        - cheap["gain_over_reuse"][metric]
                        for value, cheap in zip(values, cheap_values, strict=True)
                    ]
                )
                for metric in metrics
            },
            "remaining_gap_to_recompute": {
                metric: mean_ci95(
                    [
                        full_value["gain_over_reuse"][metric]
                        - value["gain_over_reuse"][metric]
                        for value, full_value in zip(values, full_values, strict=True)
                    ]
                )
                for metric in metrics
            },
            "migration_ratio_to_recompute": mean_ci95(
                [value["migration_ratio_to_recompute"] for value in values]
            ),
            "migration_ms_per_user": mean_ci95(
                [value["migration_ms_per_user"] for value in values]
            ),
            "cache_error_rel": mean_ci95([value["cache_error_rel"] for value in values]),
            "extra_state_ratio_to_kv": mean_ci95(
                [value["extra_state_ratio_to_kv"] for value in values]
            ),
            "extra_state_fp16_bytes_per_user": mean_ci95(
                [value["extra_state_fp16_bytes_per_user"] for value in values]
            ),
        }
    full = configs["recompute"]["gain_over_reuse"]
    for value in configs.values():
        value["quality_recovery"] = {}
        for metric in metrics:
            denominator = full[metric]["mean"]
            value["quality_recovery"][metric] = (
                value["gain_over_reuse"][metric]["mean"] / denominator
                if abs(denominator) > 1e-12
                else float("nan")
            )
    output = {
        "mode": mode,
        "model_t": model_t,
        "stale_t": pairs[0]["stale_t"],
        "dtheta_rel": mean_ci95([pair["dtheta_rel"] for pair in pairs]),
        "n_users_per_seed": pairs[0]["n_users"],
        "per_layer_stale_cache_error_rel": [
            mean_ci95(
                [pair["summary"]["per_layer_stale_cache_error_rel"][layer] for pair in pairs]
            )
            for layer in range(
                len(pairs[0]["summary"]["per_layer_stale_cache_error_rel"])
            )
        ],
        "configs": configs,
    }
    output["pareto_best_rank"] = pareto_configs(configs, "best_rank")
    output["pareto_ndcg100"] = pareto_configs(configs, "ndcg@100")
    return output


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.inputs))
    if not files:
        raise FileNotFoundError(args.inputs)
    runs = [json.loads(Path(path).read_text()) for path in files]
    modes = runs[0]["stale_modes"]
    model_ts = runs[0]["model_ts"]
    result = {
        "statistical_unit": "training_seed",
        "source_files": files,
        "num_seeds": len(runs),
        "operator": runs[0]["operator"],
        "cells": [
            aggregate_cell(runs, mode, model_t)
            for mode in modes
            for model_t in model_ts
        ],
    }
    save_json(result, args.output)
    for cell in result["cells"]:
        if cell["mode"] != "cumulative_theta0":
            continue
        print(
            f"{cell['mode']} theta={cell['stale_t']}->{cell['model_t']} "
            f"dtheta={cell['dtheta_rel']['mean']:.4f}"
        )
        for name, value in cell["configs"].items():
            print(
                f"  {name:>24} time={value['migration_ratio_to_recompute']['mean']:.3f} "
                f"rank_gain={value['gain_over_reuse']['best_rank']['mean']:.2f} "
                f"rank_recovery={value['quality_recovery']['best_rank']:.1%} "
                f"ndcg100_recovery={value['quality_recovery']['ndcg@100']:.1%}"
            )
    print(args.output)


if __name__ == "__main__":
    main()
