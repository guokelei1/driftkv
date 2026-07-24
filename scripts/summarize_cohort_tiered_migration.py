from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

from hstu_kvcache.utils import save_json

DATASETS = ("kuai", "qb", "qk")
TIERS = ("small", "medium", "large")
SEEDS = (0, 1, 2, 3)
TARGETS = ("0.5", "0.75", "0.9")
METRICS = ("best_rank", "rank_utility", "ndcg@100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="results/motivation_scale",
    )
    parser.add_argument(
        "--selection",
        default="results/motivation_scale/design_discovery_seeds.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "results/motivation_scale/"
            "cohort_tiered_migration_v1_summary.json"
        ),
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


def action_family(name: str) -> str:
    if name == "cheap_prepacked" or name.startswith("adapter_rank_"):
        return "compiled_projection"
    if name.startswith("residual_p"):
        return "residual_transport"
    if name == "recompute":
        return "full_recompute"
    raise ValueError(f"unexpected production action: {name}")


def load_run(
    path: Path,
    protocol: str,
    study_stage: str,
) -> dict:
    result = json.loads(path.read_text())
    if result.get("protocol") != protocol:
        raise ValueError(f"unexpected protocol in {path}")
    if result.get("study_stage") != study_stage:
        raise ValueError(f"unexpected study stage in {path}")
    return result


def target_record(run: dict, target: str) -> dict:
    selection = run["selection"]["fidelity_targets"][target]
    selected = selection["test"]
    full = run["test"]["configs"]["recompute"]
    gains = selected["gain_over_reuse"]
    full_gains = full["gain_over_reuse"]
    return {
        "seed": int(run["seed"]),
        "selected_action": selection["selected"],
        "selected_family": action_family(selection["selected"]),
        "test_users": int(run["split"]["test_users"]),
        "cost_ratio_to_full": float(
            selected["migration_ratio_to_recompute"]
        ),
        "cache_fidelity_recovery": float(
            selected["cache_fidelity_recovery"]
        ),
        "probe_cache_fidelity_recovery": float(
            selection["probe"]["cache_fidelity_recovery"]
        ),
        "extra_state_ratio_to_kv": float(
            selected["extra_state_ratio_to_kv"]
        ),
        "selected_gain_over_reuse": {
            metric: float(gains[metric]) for metric in METRICS
        },
        "full_gain_over_reuse": {
            metric: float(full_gains[metric]) for metric in METRICS
        },
        "selected_minus_full": {
            metric: float(gains[metric] - full_gains[metric])
            for metric in METRICS
        },
        "selected_recovery_of_full_gain": {
            metric: (
                float(gains[metric] / full_gains[metric])
                if full_gains[metric] > 0
                else None
            )
            for metric in METRICS
        },
    }


def run_record(run: dict) -> dict:
    return {
        "seed": int(run["seed"]),
        "fit_users": int(run["split"]["fit_users"]),
        "probe_users": int(run["split"]["probe_users"]),
        "test_users": int(run["split"]["test_users"]),
        "fit_elapsed_ms": float(run["fit"]["elapsed_ms"]),
        "targets": {
            target: target_record(run, target)
            for target in TARGETS
        },
    }


def aggregate_target(records: list[dict], target: str) -> dict:
    values = [record["targets"][target] for record in records]
    output = {
        "selected_actions": [
            value["selected_action"] for value in values
        ],
        "selected_action_counts": dict(
            Counter(value["selected_action"] for value in values)
        ),
        "selected_family_counts": dict(
            Counter(value["selected_family"] for value in values)
        ),
        "cost_ratio_to_full": describe(
            [value["cost_ratio_to_full"] for value in values]
        ),
        "cache_fidelity_recovery": describe(
            [value["cache_fidelity_recovery"] for value in values]
        ),
        "probe_cache_fidelity_recovery": describe(
            [
                value["probe_cache_fidelity_recovery"]
                for value in values
            ]
        ),
        "extra_state_ratio_to_kv": describe(
            [value["extra_state_ratio_to_kv"] for value in values]
        ),
        "test_fidelity_target_met": int(
            sum(
                value["cache_fidelity_recovery"] >= float(target)
                for value in values
            )
        ),
        "metrics": {},
    }
    for metric in METRICS:
        recoveries = [
            value["selected_recovery_of_full_gain"][metric]
            for value in values
            if value["selected_recovery_of_full_gain"][metric] is not None
        ]
        output["metrics"][metric] = {
            "selected_gain_over_reuse": describe(
                [
                    value["selected_gain_over_reuse"][metric]
                    for value in values
                ]
            ),
            "full_gain_over_reuse": describe(
                [
                    value["full_gain_over_reuse"][metric]
                    for value in values
                ]
            ),
            "selected_minus_full": describe(
                [
                    value["selected_minus_full"][metric]
                    for value in values
                ]
            ),
            "selected_recovery_of_full_gain": (
                describe(recoveries) if recoveries else None
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    selection = json.loads(Path(args.selection).read_text())
    cells = {}
    validation_records = []
    for dataset in DATASETS:
        for tier in TIERS:
            cell = f"{dataset}_{tier}"
            discovery_seed = int(
                selection["cells"][cell]["selected_seed"]
            )
            discovery_path = (
                input_dir
                / (
                    f"{cell}_v2_cohort_tiered_discovery_"
                    f"seed{discovery_seed}.json"
                )
            )
            discovery_run = load_run(
                discovery_path,
                "cohort_tiered_migration_discovery_v1",
                "motivation_selected_seed_discovery",
            )
            validation_paths = [
                input_dir / f"{cell}_v2_cohort_tiered_seed{seed}.json"
                for seed in SEEDS
                if seed != discovery_seed
            ]
            validation_runs = [
                load_run(
                    path,
                    "cohort_tiered_migration_v1",
                    "frozen_rule_replication",
                )
                for path in validation_paths
            ]
            discovery_record = run_record(discovery_run)
            records = [run_record(run) for run in validation_runs]
            validation_records.extend(records)
            primary = aggregate_target(records, "0.5")
            primary["passes_frozen_mean_gate"] = {
                "cost_below_full": (
                    primary["cost_ratio_to_full"]["mean"] < 1.0
                ),
                "best_rank_positive": (
                    primary["metrics"]["best_rank"][
                        "selected_gain_over_reuse"
                    ]["mean"]
                    > 0
                ),
                "rank_utility_positive": (
                    primary["metrics"]["rank_utility"][
                        "selected_gain_over_reuse"
                    ]["mean"]
                    > 0
                ),
            }
            primary["passes_all_frozen_mean_gates"] = all(
                primary["passes_frozen_mean_gate"].values()
            )
            cells[cell] = {
                "discovery_seed": discovery_seed,
                "discovery": discovery_record,
                "heldout_seed_replication": {
                    "seeds": [record["seed"] for record in records],
                    "fit_elapsed_ms": describe(
                        [
                            record["fit_elapsed_ms"]
                            for record in records
                        ]
                    ),
                    "targets": {
                        "0.5": primary,
                        "0.75": aggregate_target(records, "0.75"),
                        "0.9": aggregate_target(records, "0.9"),
                    },
                },
                "source_files": {
                    "discovery": str(discovery_path),
                    "heldout_seed_replication": [
                        str(path) for path in validation_paths
                    ],
                },
            }
    primary_records = [
        record["targets"]["0.5"] for record in validation_records
    ]
    endpoint_tracking = {}
    for metric in METRICS:
        selected_values = np.asarray(
            [
                record["selected_gain_over_reuse"][metric]
                for record in primary_records
            ],
            dtype=np.float64,
        )
        full_values = np.asarray(
            [
                record["full_gain_over_reuse"][metric]
                for record in primary_records
            ],
            dtype=np.float64,
        )
        positive_full = full_values > 0
        recoveries = selected_values[positive_full] / full_values[
            positive_full
        ]
        endpoint_tracking[metric] = {
            "descriptive_cross_cell_only": True,
            "full_positive_seeds": int(positive_full.sum()),
            "selected_positive_among_full_positive": int(
                np.sum(selected_values[positive_full] > 0)
            ),
            "same_direction_signs": int(
                np.sum(np.sign(selected_values) == np.sign(full_values))
            ),
            "spearman_selected_vs_full": float(
                stats.spearmanr(selected_values, full_values).statistic
            ),
            "median_recovery_when_full_positive": float(
                np.median(recoveries)
            ),
        }
    gate_cells = {
        cell: value["heldout_seed_replication"]["targets"]["0.5"][
            "passes_all_frozen_mean_gates"
        ]
        for cell, value in cells.items()
    }
    result = {
        "protocol": "cohort_tiered_migration_v1_seed_summary",
        "statistical_unit": "independent_training_seed",
        "method_selection": {
            "discovery_cells": 9,
            "frozen_rule_replication_cells": 27,
            "fit_users_per_run": 40,
            "probe_users_per_run": 60,
            "primary_fidelity_target": 0.5,
            "secondary_fidelity_targets": [0.75, 0.9],
            "task_labels_used_for_selection": False,
        },
        "cells": cells,
        "cross_cell_primary_validation": {
            "cells_passing_all_frozen_mean_gates": int(
                sum(gate_cells.values())
            ),
            "gate_by_cell": gate_cells,
            "selected_family_counts": dict(
                Counter(
                    record["selected_family"]
                    for record in primary_records
                )
            ),
            "positive_seed_signs": {
                metric: int(
                    sum(
                        record["selected_gain_over_reuse"][metric] > 0
                        for record in primary_records
                    )
                )
                for metric in METRICS
            },
            "num_validation_seeds": len(primary_records),
            "cost_ratio_to_full": describe(
                [
                    record["cost_ratio_to_full"]
                    for record in primary_records
                ]
            ),
            "cache_fidelity_recovery": describe(
                [
                    record["cache_fidelity_recovery"]
                    for record in primary_records
                ]
            ),
            "test_fidelity_target_met": int(
                sum(
                    record["cache_fidelity_recovery"] >= 0.5
                    for record in primary_records
                )
            ),
            "full_endpoint_tracking": endpoint_tracking,
        },
    }
    save_json(result, args.output)
    for cell, value in cells.items():
        primary = value["heldout_seed_replication"]["targets"]["0.5"]
        best_rank = primary["metrics"]["best_rank"][
            "selected_gain_over_reuse"
        ]
        rank_utility = primary["metrics"]["rank_utility"][
            "selected_gain_over_reuse"
        ]
        ndcg = primary["metrics"]["ndcg@100"][
            "selected_gain_over_reuse"
        ]
        print(
            f"{cell:12s} "
            f"cost={primary['cost_ratio_to_full']['mean']:.3f} "
            f"fidelity={primary['cache_fidelity_recovery']['mean']:.3f} "
            f"rank={best_rank['mean']:+.2f} "
            f"({best_rank['positive_seeds']}/3) "
            f"utility={rank_utility['mean']:+.5f} "
            f"ndcg100={ndcg['mean']:+.5f} "
            f"pass={primary['passes_all_frozen_mean_gates']}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
