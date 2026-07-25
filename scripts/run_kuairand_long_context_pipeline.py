"""Run data preparation, four-GPU training, and motivation evaluation in order."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from hstu_kvcache.streaming import (
    SUPPORTED_LONG_CONTEXT_BASE_DAYS,
    long_context_split_name,
    motivation_protocol_for_base_days,
    training_protocol_for_base_days,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-days",
        type=int,
        choices=SUPPORTED_LONG_CONTEXT_BASE_DAYS,
        default=4,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--reuse-training", action="store_true")
    parser.add_argument("--overwrite-motivation", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def pipeline_paths(base_days: int, seed: int) -> dict[str, Path]:
    split = long_context_split_name(base_days)
    exploration = "_exploration" if base_days != 8 else ""
    prepared = (
        Path("data/processed/kuairand_long_context_8plus8_v2.npz")
        if base_days == 8
        else Path(
            f"data/processed/kuairand_long_context_{split}_exploration_v1.npz"
        )
    )
    checkpoint_dir = Path(
        f"checkpoints/kuairand_long_context_{split}{exploration}/seed{seed}"
    )
    training_result = Path(
        f"results/motivation_scale/long_context_{split}_training"
        f"{exploration}_seed{seed}.json"
    )
    motivation_result = Path(
        f"results/motivation_scale/long_context_{split}"
        f"_motivation_all_pairs{exploration}_seed{seed}.json"
    )
    return {
        "prepared": prepared,
        "checkpoint_dir": checkpoint_dir,
        "training_result": training_result,
        "motivation_result": motivation_result,
    }


def execute(command: list[str], root: Path, print_only: bool) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    if not print_only:
        subprocess.run(command, cwd=root, check=True)


def validate_training(
    path: Path,
    checkpoint_dir: Path,
    base_days: int,
    seed: int,
) -> None:
    source = json.loads(path.read_text())
    expected_protocol = training_protocol_for_base_days(base_days)
    if source.get("protocol") != expected_protocol:
        raise ValueError("existing training result uses a different protocol")
    if source.get("status") != "complete":
        raise ValueError("existing training result is incomplete")
    if int(source["args"]["seed"]) != seed:
        raise ValueError("existing training result uses a different seed")
    online_days = 16 - base_days
    missing = [
        str(checkpoint_dir / f"theta_{version}.pt")
        for version in range(online_days + 1)
        if not (checkpoint_dir / f"theta_{version}.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"existing training is missing checkpoints: {missing}")


def validate_motivation(path: Path, base_days: int, seed: int) -> dict:
    result = json.loads(path.read_text())
    if result.get("protocol") != motivation_protocol_for_base_days(base_days):
        raise ValueError("motivation result uses a different protocol")
    if int(result["seed"]) != seed:
        raise ValueError("motivation result uses a different seed")
    evaluable_versions = 15 - base_days
    expected_pairs = evaluable_versions * (evaluable_versions + 1) // 2
    if int(result["comparison_count"]) != expected_pairs:
        raise ValueError("motivation result does not contain the complete matrix")
    return result


def main() -> None:
    args = parse_args()
    if args.nproc_per_node != 4:
        raise ValueError("the formal long-context pipeline requires exactly four workers")
    root = Path(__file__).resolve().parents[1]
    relative_paths = pipeline_paths(args.base_days, args.seed)
    paths = {
        name: root / path
        for name, path in relative_paths.items()
    }
    if not args.print_only:
        if (
            paths["motivation_result"].exists()
            and not args.overwrite_motivation
        ):
            raise FileExistsError(
                "motivation output already exists; inspect it or pass "
                "--overwrite-motivation"
            )
        if not args.reuse_training and (
            paths["training_result"].exists()
            or paths["checkpoint_dir"].exists()
        ):
            raise FileExistsError(
                "training artifacts already exist; pass --reuse-training only "
                "after a complete run"
            )
    python = sys.executable
    prepare = [
        python,
        "scripts/prepare_kuairand_long_context.py",
        "--base-days",
        str(args.base_days),
        "--output",
        str(relative_paths["prepared"]),
    ]
    if paths["prepared"].exists():
        prepare.append("--validate-existing")
    execute(prepare, root, args.print_only)
    train = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "scripts/train_kuairand_long_context.py",
        "--base-days",
        str(args.base_days),
        "--seed",
        str(args.seed),
        "--prepared-data",
        str(relative_paths["prepared"]),
        "--checkpoint-dir",
        str(relative_paths["checkpoint_dir"]),
        "--output",
        str(relative_paths["training_result"]),
    ]
    if args.reuse_training:
        if not args.print_only:
            validate_training(
                paths["training_result"],
                paths["checkpoint_dir"],
                args.base_days,
                args.seed,
            )
        print("$ reuse validated complete training artifacts", flush=True)
    else:
        execute(train, root, args.print_only)
        if not args.print_only:
            validate_training(
                paths["training_result"],
                paths["checkpoint_dir"],
                args.base_days,
                args.seed,
            )
    evaluate = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "scripts/evaluate_kuairand_long_context_motivation.py",
        "--base-days",
        str(args.base_days),
        "--seed",
        str(args.seed),
        "--prepared-data",
        str(relative_paths["prepared"]),
        "--training-result",
        str(relative_paths["training_result"]),
        "--checkpoint-dir",
        str(relative_paths["checkpoint_dir"]),
        "--output",
        str(relative_paths["motivation_result"]),
    ]
    execute(evaluate, root, args.print_only)
    if not args.print_only:
        result = validate_motivation(
            paths["motivation_result"],
            args.base_days,
            args.seed,
        )
        print(
            json.dumps(
                {
                    "training_result": str(relative_paths["training_result"]),
                    "motivation_result": str(relative_paths["motivation_result"]),
                    "comparison_count": result["comparison_count"],
                    "status": "complete",
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
