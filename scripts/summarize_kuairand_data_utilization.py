from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_kuairand_factorial import (
    canonical_names,
    config_summary,
    mean_interval,
    validate_constant,
)

from hstu_kvcache.utils import save_json

MODE_SPECS = {
    "latest": {
        "core_pattern": "results/scaling/top50k_latest_core_seed{seed}.json",
        "method_pattern": "results/scaling/top50k_latest_method_seed{seed}.json",
        "control_pattern": "results/scaling/top50k_latest_streaming_control_seed{seed}.json",
    },
    "all_chunks": {
        "core_pattern": "results/scaling/top50k_all_chunks_core_seed{seed}.json",
        "method_pattern": "results/scaling/top50k_all_chunks_method_seed{seed}.json",
        "control_pattern": "results/scaling/top50k_all_chunks_streaming_control_seed{seed}.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--output",
        default="results/scaling/kuairand_data_utilization_summary.json",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def coverage_summary(cores: list[dict]) -> dict:
    base = [run["base_training_coverage"][0] for run in cores]
    stream_targets = [
        sum(
            day["eligible_targets"]
            for window in run["windows"]
            for day in window["training_coverage"]
        )
        for run in cores
    ]
    return {
        "base_sequences_per_epoch": int(
            validate_constant([value["sequences"] for value in base], "base sequences")
        ),
        "base_context_tokens_per_epoch": int(
            validate_constant([value["tokens"] for value in base], "base tokens")
        ),
        "base_eligible_targets_per_epoch": int(
            validate_constant([value["eligible_targets"] for value in base], "base targets")
        ),
        "stream_eligible_targets": int(
            validate_constant(stream_targets, "stream targets")
        ),
    }


def mode_summary(spec: dict, seeds: list[int]) -> dict:
    core_paths = [spec["core_pattern"].format(seed=seed) for seed in seeds]
    method_paths = [spec["method_pattern"].format(seed=seed) for seed in seeds]
    control_paths = [spec["control_pattern"].format(seed=seed) for seed in seeds]
    cores = [load_json(path) for path in core_paths]
    methods = [load_json(path) for path in method_paths]
    controls = [load_json(path) for path in control_paths]
    points = [run["points"][0] for run in methods]
    metadata = [run["args"] for run in cores]
    num_layers = int(
        validate_constant([value["num_layers"] for value in metadata], "num_layers")
    )
    num_items = int(
        validate_constant([run["data"]["num_items"] for run in cores], "num_items")
    )
    pooled = [run["pooled_cumulative_theta0_descriptive"] for run in cores]
    control_pairs = [run["pairs"][0] for run in controls]
    contrasts = {}
    for name in (
        "streaming_value_full_compute",
        "streaming_value_full_reuse",
        "cache_maintenance_value",
    ):
        contrasts[name] = {
            "best_rank": mean_interval(
                [pair["summary"]["contrasts"][name]["best_rank"] for pair in control_pairs]
            ),
            "ndcg100": mean_interval(
                [pair["summary"]["contrasts"][name]["ndcg@100"] for pair in control_pairs]
            ),
        }
    return {
        "training_sequences": validate_constant(
            [value["training_sequences"] for value in metadata],
            "training_sequences",
        ),
        "data": {
            "num_items": num_items,
            "num_users": int(
                validate_constant([run["data"]["num_users"] for run in cores], "num_users")
            ),
            "sequence_length": int(
                validate_constant([value["seq_len"] for value in metadata], "seq_len")
            ),
        },
        "model": {
            "num_layers": num_layers,
            "hidden_size": int(
                validate_constant([value["hidden_size"] for value in metadata], "hidden_size")
            ),
            "num_parameters": int(
                validate_constant([run["num_parameters"] for run in cores], "num_parameters")
            ),
        },
        "coverage": coverage_summary(cores),
        "training": {
            "final_base_loss": mean_interval([run["base_losses"][-1] for run in cores]),
            "cumulative_dtheta_rel": mean_interval([point["dtheta_rel"] for point in points]),
        },
        "pooled_cumulative_gap": {
            "best_rank": mean_interval([value["best_rank"]["gain"] for value in pooled]),
            "ndcg100": mean_interval([value["ndcg@100"]["gain"] for value in pooled]),
        },
        "streaming_value_control_theta5": {
            "contrasts": contrasts,
            "current_incremental_parity_max_abs": max(
                pair["summary"]["current_incremental_parity_max_abs"]
                for pair in control_pairs
            ),
            "frozen_incremental_parity_max_abs": max(
                pair["summary"]["frozen_incremental_parity_max_abs"]
                for pair in control_pairs
            ),
        },
        "configs": config_summary(points, num_items, canonical_names(num_layers)),
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
        "source_files": {
            "core": core_paths,
            "method": method_paths,
            "streaming_control": control_paths,
        },
    }


def main() -> None:
    args = parse_args()
    result = {
        "protocol": "kuairand_data_utilization_v1_seed_level_summary",
        "seeds": args.seeds,
        "scope": "top-50k, sequence length 512, six-layer hidden-96 model",
        "timing_scope": "Only resident-GPU method timing is comparable; training wall time is omitted.",
        "modes": {
            name: mode_summary(spec, args.seeds)
            for name, spec in MODE_SPECS.items()
        },
    }
    save_json(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
