from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from hstu_kvcache.utils import save_json

DATASETS = ("kuai", "qb", "qk")
TIERS = ("small", "medium", "large")
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
            "structural_design_discovery_summary.json"
        ),
    )
    return parser.parse_args()


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "n": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_selected(
    runs: dict[str, dict],
    target: str,
) -> dict:
    rows = []
    for cell, run in runs.items():
        selected = run["selection"]["fidelity_targets"][target]
        test = selected["test"]
        rows.append(
            {
                "cell": cell,
                "selected": selected["selected"],
                "cost_ratio_to_full": float(
                    test["migration_ratio_to_recompute"]
                ),
                "cache_fidelity_recovery": float(
                    test["cache_fidelity_recovery"]
                ),
                "gain_over_reuse": {
                    metric: float(test["gain_over_reuse"][metric])
                    for metric in METRICS
                },
            }
        )
    return {
        "target": float(target),
        "selected_action_counts": dict(
            Counter(row["selected"] for row in rows)
        ),
        "cost_ratio_to_full": describe(
            [row["cost_ratio_to_full"] for row in rows]
        ),
        "cache_fidelity_recovery": describe(
            [row["cache_fidelity_recovery"] for row in rows]
        ),
        "positive_cell_signs": {
            metric: int(
                sum(
                    row["gain_over_reuse"][metric] > 0
                    for row in rows
                )
            )
            for metric in METRICS
        },
        "cells": rows,
    }


def load_runs(
    input_dir: Path,
    selected_seeds: dict[str, int],
    suffix: str,
    protocol: str,
) -> dict[str, dict]:
    output = {}
    for cell, seed in selected_seeds.items():
        path = input_dir / f"{cell}_v2_{suffix}_seed{seed}.json"
        run = json.loads(path.read_text())
        if run.get("protocol") != protocol:
            raise ValueError(f"unexpected protocol in {path}")
        output[cell] = run
    return output


def recent_fragmentation(runs: dict[str, dict]) -> dict:
    comparisons = []
    for cell, run in runs.items():
        actions = run["action_space"]
        configs = run["test"]["summary"]["configs"]
        full_span = {
            action["top_n_full"]: name
            for name, action in actions.items()
            if action.get("kind", "recent_suffix") == "recent_suffix"
            and action["recent_fraction"] == 1.0
        }
        for name, action in actions.items():
            if (
                action.get("kind", "recent_suffix") != "recent_suffix"
                or action["recent_fraction"] >= 1.0
                or action["top_n_full"] == 0
            ):
                continue
            reference = full_span.get(action["top_n_full"])
            if reference is None:
                continue
            partial_cost = float(
                configs[name]["migration_ratio_to_recompute"]
            )
            full_span_cost = float(
                configs[reference]["migration_ratio_to_recompute"]
            )
            comparisons.append(
                {
                    "cell": cell,
                    "partial_action": name,
                    "full_span_action": reference,
                    "partial_cost_ratio": partial_cost,
                    "full_span_cost_ratio": full_span_cost,
                    "partial_slower": partial_cost > full_span_cost,
                }
            )
    return {
        "comparisons": len(comparisons),
        "partial_slower_count": int(
            sum(value["partial_slower"] for value in comparisons)
        ),
        "records": comparisons,
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    selection = json.loads(Path(args.selection).read_text())
    selected_seeds = {
        f"{dataset}_{tier}": int(
            selection["cells"][f"{dataset}_{tier}"]["selected_seed"]
        )
        for dataset in DATASETS
        for tier in TIERS
    }
    recent = load_runs(
        input_dir,
        selected_seeds,
        "structural_replay_discovery",
        "structural_replay_discovery_v1",
    )
    intervals = load_runs(
        input_dir,
        selected_seeds,
        "structural_replay_interval_discovery",
        "structural_replay_discovery_v1",
    )
    residual = load_runs(
        input_dir,
        selected_seeds,
        "residual_transport_discovery",
        "residual_transport_discovery_v1",
    )
    prefix = load_runs(
        input_dir,
        selected_seeds,
        "prefix_replay",
        "progressive_prefix_replay_v1",
    )
    for run in prefix.values():
        run["selection"] = {
            "fidelity_targets": {
                "0.2": run["selected"],
            }
        }
    cohort = load_runs(
        input_dir,
        selected_seeds,
        "cohort_tiered_discovery",
        "cohort_tiered_migration_discovery_v1",
    )
    unified_selected = [
        run["selection"]["fidelity_targets"][target]["selected"]
        for run in cohort.values()
        for target in ("0.5", "0.75", "0.9")
    ]
    result = {
        "protocol": "structural_design_discovery_summary_v1",
        "study_stage": "descriptive_motivation_selected_seed_discovery",
        "num_cells": len(selected_seeds),
        "statistical_boundary": (
            "one motivation-selected checkpoint per cell; these are "
            "architecture discovery units, not independent confirmation"
        ),
        "screens": {
            "recent_suffix_target_0.2": aggregate_selected(
                recent,
                "0.2",
            ),
            "all_intervals_target_0.2": aggregate_selected(
                intervals,
                "0.2",
            ),
            "prefix_target_0.2": aggregate_selected(prefix, "0.2"),
            "residual_target_0.2": aggregate_selected(
                residual,
                "0.2",
            ),
            "residual_target_0.5": aggregate_selected(
                residual,
                "0.5",
            ),
            "cohort_tiered_target_0.5": aggregate_selected(
                cohort,
                "0.5",
            ),
            "cohort_tiered_target_0.75": aggregate_selected(
                cohort,
                "0.75",
            ),
        },
        "recent_token_fragmentation": recent_fragmentation(recent),
        "unified_production_selection": {
            "targets_per_cell": [0.5, 0.75, 0.9],
            "selected_action_counts": dict(
                Counter(unified_selected)
            ),
            "plain_prefix_selected": int(
                sum(name.startswith("prefix_p") for name in unified_selected)
            ),
        },
    }
    save_json(result, args.output)
    for name, value in result["screens"].items():
        print(
            f"{name:31s} "
            f"cost={value['cost_ratio_to_full']['mean']:.3f} "
            f"fidelity={value['cache_fidelity_recovery']['mean']:.3f} "
            f"signs={value['positive_cell_signs']}"
        )
    fragmentation = result["recent_token_fragmentation"]
    print(
        "recent_partial_slower="
        f"{fragmentation['partial_slower_count']}/"
        f"{fragmentation['comparisons']}"
    )
    print(args.output)


if __name__ == "__main__":
    main()
