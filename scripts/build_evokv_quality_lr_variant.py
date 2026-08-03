from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-schedule", type=Path, required=True)
    parser.add_argument("--base-benchmark", type=Path, required=True)
    parser.add_argument("--schedule-output", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--learning-rate-scale", type=float, required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"generated config differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if (
        not args.base_schedule.is_file()
        or not args.base_benchmark.is_file()
        or not args.tag.replace("_", "").isalnum()
        or not 0.01 <= args.learning_rate_scale <= 1.0
    ):
        raise ValueError("quality LR variant arguments differ")
    schedule = json.loads(args.base_schedule.read_text())
    benchmark = copy.deepcopy(json.loads(args.base_benchmark.read_text()))
    if (
        schedule.get("protocol")
        != "evokv_xp_prequential_stream_training_development_v1"
        or len(schedule.get("updates", [])) != 4
        or schedule.get("training", {}).get("epochs_per_update") != 1
        or benchmark.get("quality_chain", {}).get("evaluated_edges")
        != [[1, 2], [2, 3], [3, 4]]
        or benchmark.get("model", {}).get("embedding_width") != 4096
        or benchmark.get("model", {}).get("hidden_size") != 1536
        or benchmark.get("model", {}).get("layers") != 24
    ):
        raise ValueError("quality LR base configuration differs")
    training_users = benchmark["quality_chain"].get("training_users")
    qualification_users = benchmark["quality_chain"].get(
        "qualification_users"
    )
    epochs_per_update = schedule["training"].get("epochs_per_update")
    if (
        not isinstance(training_users, int)
        or training_users < 1
        or not isinstance(qualification_users, int)
        or qualification_users < 1
        or epochs_per_update != benchmark["quality_chain"].get(
            "epochs_per_update"
        )
    ):
        raise ValueError("quality LR population binding differs")
    scale = args.learning_rate_scale
    scale_name = f"{round(scale * 100):03d}x"
    candidate_name = f"lr_fixed_{scale_name}"
    schedule["learning_rate_policy"] = {
        "candidates": [
            {
                "dense": 1.0e-4 * scale,
                "embedding": 1.0e-3 * scale,
                "name": candidate_name,
                "projection": 1.0e-4 * scale,
            }
        ],
        "mode": "predeclared_fixed",
        "quality_observed_for_selection": False,
        "selected_candidate": candidate_name,
        "selection_role": "none",
    }
    schedule["stack_identity"] = (
        "xp_qk_stream_aligned_warmup_"
        f"train{training_users}_qual{qualification_users}_"
        f"e{epochs_per_update}_fixed{scale_name}_{args.tag}_development_v1"
    )
    schedule["development_variant"] = {
        "axis": "optimizer_step_size",
        "base_schedule": str(args.base_schedule),
        "learning_rate_scale": scale,
        "model_geometry_changed": False,
        "training_corpus_changed": False,
    }
    benchmark["benchmark_id"] = (
        "x_qk_xp_quality_stream_aligned_"
        f"train{training_users}_qual{qualification_users}_"
        f"e{epochs_per_update}_two_gpu_{args.tag}_development_v1"
    )
    benchmark["quality_chain"]["schedule_path"] = str(
        args.schedule_output
    )
    benchmark["quality_chain"]["learning_rate_scale"] = scale
    benchmark["quality_chain"]["development_variant_tag"] = args.tag
    benchmark["quality_chain"]["model_geometry_changed"] = False
    benchmark["quality_chain"]["training_corpus_changed"] = False
    benchmark["status"] = "stream_aligned_quality_lr_candidate_ready"
    atomic_json(args.schedule_output, schedule)
    atomic_json(args.benchmark_output, benchmark)
    print(
        json.dumps(
            {
                "benchmark": str(args.benchmark_output),
                "learning_rate_scale": scale,
                "schedule": str(args.schedule_output),
                "status": "complete",
                "tag": args.tag,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
