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
            "progressive_prefix_replay_v1_summary.json"
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


def load_run(path: Path, expected_stage: str) -> dict:
    result = json.loads(path.read_text())
    if result.get("protocol") != "progressive_prefix_replay_v1":
        raise ValueError(f"unexpected protocol in {path}")
    if result.get("study_stage") != expected_stage:
        raise ValueError(f"unexpected study stage in {path}")
    if result.get("selection_rule", {}).get("target") != 0.2:
        raise ValueError(f"unexpected fidelity target in {path}")
    return result


def run_record(run: dict) -> dict:
    selected_name = run["selected"]["selected"]
    selected = run["selected"]["test"]
    configs = run["test"]["summary"]["configs"]
    full = configs["recompute"]
    cheap = configs["cheap_all"]
    gains = selected["gain_over_reuse"]
    full_gains = full["gain_over_reuse"]
    return {
        "seed": int(run["seed"]),
        "selected_action": selected_name,
        "selected_action_spec": selected["action"],
        "probe_users": int(run["split"]["probe_users"]),
        "test_users": int(run["split"]["test_users"]),
        "cost_ratio_to_full": float(
            selected["migration_ratio_to_recompute"]
        ),
        "cache_fidelity_recovery": float(
            selected["cache_fidelity_recovery"]
        ),
        "probe_cache_fidelity_recovery": float(
            run["selected"]["probe"]["cache_fidelity_recovery"]
        ),
        "extra_state_ratio_to_kv": float(
            selected["extra_state_ratio_to_kv"]
        ),
        "selected_gain_over_reuse": {
            metric: float(gains[metric]) for metric in METRICS
        },
        "cheap_gain_over_reuse": {
            metric: float(cheap["gain_over_reuse"][metric])
            for metric in METRICS
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


def aggregate_records(records: list[dict]) -> dict:
    output = {
        "seeds": [record["seed"] for record in records],
        "selected_actions": [
            record["selected_action"] for record in records
        ],
        "selected_action_counts": dict(
            Counter(record["selected_action"] for record in records)
        ),
        "probe_users": [record["probe_users"] for record in records],
        "test_users": [record["test_users"] for record in records],
        "cost_ratio_to_full": describe(
            [record["cost_ratio_to_full"] for record in records]
        ),
        "cache_fidelity_recovery": describe(
            [record["cache_fidelity_recovery"] for record in records]
        ),
        "probe_cache_fidelity_recovery": describe(
            [
                record["probe_cache_fidelity_recovery"]
                for record in records
            ]
        ),
        "extra_state_ratio_to_kv": describe(
            [record["extra_state_ratio_to_kv"] for record in records]
        ),
        "metrics": {},
    }
    for metric in METRICS:
        recoveries = [
            record["selected_recovery_of_full_gain"][metric]
            for record in records
            if record["selected_recovery_of_full_gain"][metric] is not None
        ]
        output["metrics"][metric] = {
            "selected_gain_over_reuse": describe(
                [
                    record["selected_gain_over_reuse"][metric]
                    for record in records
                ]
            ),
            "cheap_gain_over_reuse": describe(
                [
                    record["cheap_gain_over_reuse"][metric]
                    for record in records
                ]
            ),
            "full_gain_over_reuse": describe(
                [
                    record["full_gain_over_reuse"][metric]
                    for record in records
                ]
            ),
            "selected_minus_full": describe(
                [
                    record["selected_minus_full"][metric]
                    for record in records
                ]
            ),
            "selected_recovery_of_full_gain": (
                describe(recoveries) if recoveries else None
            ),
        }
    output["passes_frozen_mean_gate"] = {
        "cost_below_full": (
            output["cost_ratio_to_full"]["mean"] < 1.0
        ),
        "best_rank_positive": (
            output["metrics"]["best_rank"]["selected_gain_over_reuse"][
                "mean"
            ]
            > 0
        ),
        "rank_utility_positive": (
            output["metrics"]["rank_utility"]["selected_gain_over_reuse"][
                "mean"
            ]
            > 0
        ),
    }
    output["passes_all_frozen_mean_gates"] = all(
        output["passes_frozen_mean_gate"].values()
    )
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
                / f"{cell}_v2_prefix_replay_seed{discovery_seed}.json"
            )
            discovery_run = load_run(
                discovery_path,
                "discovery_replay",
            )
            validation_paths = [
                input_dir / f"{cell}_v2_prefix_replay_seed{seed}.json"
                for seed in SEEDS
                if seed != discovery_seed
            ]
            validation_runs = [
                load_run(path, "frozen_rule_replication")
                for path in validation_paths
            ]
            discovery_record = run_record(discovery_run)
            records = [run_record(run) for run in validation_runs]
            validation_records.extend(records)
            cells[cell] = {
                "discovery_seed": discovery_seed,
                "discovery": discovery_record,
                "heldout_seed_replication": aggregate_records(records),
                "source_files": {
                    "discovery": str(discovery_path),
                    "heldout_seed_replication": [
                        str(path) for path in validation_paths
                    ],
                },
            }
    gate_cells = {
        cell: value["heldout_seed_replication"][
            "passes_all_frozen_mean_gates"
        ]
        for cell, value in cells.items()
    }
    result = {
        "protocol": "progressive_prefix_replay_v1_seed_summary",
        "statistical_unit": "independent_training_seed",
        "method_selection": {
            "discovery_cells": 9,
            "frozen_rule_replication_cells": 27,
            "action_family": "cheap plus O(L) current-model prefix replay",
            "fidelity_target": 0.2,
            "probe_users_per_run": 60,
            "task_labels_used_for_action_selection": False,
        },
        "cells": cells,
        "cross_cell_validation": {
            "cells_passing_all_frozen_mean_gates": int(
                sum(gate_cells.values())
            ),
            "gate_by_cell": gate_cells,
            "positive_seed_signs": {
                metric: int(
                    sum(
                        record["selected_gain_over_reuse"][metric] > 0
                        for record in validation_records
                    )
                )
                for metric in METRICS
            },
            "num_validation_seeds": len(validation_records),
            "cost_ratio_to_full": describe(
                [
                    record["cost_ratio_to_full"]
                    for record in validation_records
                ]
            ),
            "cache_fidelity_recovery": describe(
                [
                    record["cache_fidelity_recovery"]
                    for record in validation_records
                ]
            ),
            "test_fidelity_target_met": int(
                sum(
                    record["cache_fidelity_recovery"] >= 0.2
                    for record in validation_records
                )
            ),
        },
    }
    save_json(result, args.output)
    for cell, value in cells.items():
        validation = value["heldout_seed_replication"]
        best_rank = validation["metrics"]["best_rank"][
            "selected_gain_over_reuse"
        ]
        rank_utility = validation["metrics"]["rank_utility"][
            "selected_gain_over_reuse"
        ]
        ndcg = validation["metrics"]["ndcg@100"][
            "selected_gain_over_reuse"
        ]
        print(
            f"{cell:12s} "
            f"cost={validation['cost_ratio_to_full']['mean']:.3f} "
            f"fidelity={validation['cache_fidelity_recovery']['mean']:.3f} "
            f"rank={best_rank['mean']:+.2f} ({best_rank['positive_seeds']}/3) "
            f"utility={rank_utility['mean']:+.5f} "
            f"ndcg100={ndcg['mean']:+.5f} "
            f"pass={validation['passes_all_frozen_mean_gates']}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
