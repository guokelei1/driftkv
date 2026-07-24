from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TIERS = {
    "small": {
        "hidden_size": 64,
        "num_layers": 3,
        "num_heads": 4,
        "head_dim": 16,
    },
    "medium": {
        "hidden_size": 96,
        "num_layers": 6,
        "num_heads": 4,
        "head_dim": 24,
    },
    "large": {
        "hidden_size": 128,
        "num_layers": 9,
        "num_heads": 4,
        "head_dim": 32,
    },
}

DATASETS = {
    "kuai": {
        "max_items": 50000,
        "max_users": {"small": 250, "medium": 500, "large": 1000},
        "training_sequences": "all_chunks",
    },
    "qb": {
        "max_items": 50000,
        "prepared": "data/processed/motivation_capacity_v2_qb_{tier}.npz",
        "training_sequences": "latest",
    },
    "qk": {
        "max_items": 5000,
        "prepared": "data/processed/motivation_capacity_v2_qk_{tier}.npz",
        "training_sequences": "latest",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("core", "control", "matrix", "cost"),
        default=("core", "control", "matrix"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def paths(dataset: str, tier: str, seed: int) -> dict[str, Path]:
    stem = f"{dataset}_{tier}_v2"
    return {
        "core": Path(f"results/motivation_scale/{stem}_core_seed{seed}.json"),
        "control": Path(
            f"results/motivation_scale/{stem}_streaming_control_seed{seed}.json"
        ),
        "matrix": Path(
            f"results/motivation_scale/{stem}_cache_version_matrix_seed{seed}.json"
        ),
        "cost": Path(
            f"results/motivation_scale/{stem}_operator_cost_seed{seed}.json"
        ),
        "checkpoint": Path(
            f"checkpoints/motivation_capacity_v2/{dataset}_{tier}_seed{seed}"
        ),
        "log": Path(f"logs/motivation_capacity_v2/{dataset}_{tier}_seed{seed}.log"),
    }


def complete(path: Path, stage: str) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if stage == "core":
        return (
            result.get("protocol") == "motivation_capacity_v2_training"
            and len(result.get("windows", [])) == 11
        )
    if stage == "control":
        return (
            result.get("protocol") == "motivation_capacity_v2_streaming_control"
            and len(result.get("pairs", [])) == 6
        )
    if stage == "matrix":
        return (
            result.get("protocol")
            == "motivation_capacity_v2_cache_version_matrix"
            and len(result.get("points", [])) == 11
        )
    return result.get("protocol") == "motivation_capacity_v2_operator_cost"


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as output:
        subprocess.run(
            command,
            check=True,
            stdout=output,
            stderr=subprocess.STDOUT,
        )


def core_command(
    dataset: str,
    tier: str,
    seed: int,
    device: str,
    cell_paths: dict[str, Path],
) -> list[str]:
    model = TIERS[tier]
    data = DATASETS[dataset]
    command = [
        sys.executable,
        "scripts/motivation_validity.py",
        "--protocol",
        "motivation_capacity_v2_training",
        "--device",
        device,
        "--seed",
        str(seed),
        "--hidden-size",
        str(model["hidden_size"]),
        "--num-layers",
        str(model["num_layers"]),
        "--num-heads",
        str(model["num_heads"]),
        "--head-dim",
        str(model["head_dim"]),
        "--seq-len",
        "128",
        "--max-items",
        str(data["max_items"]),
        "--base-days",
        "14",
        "--base-epochs",
        "6",
        "--base-lr",
        "3e-4",
        "--stream-lr",
        "1e-4",
        "--stream-window-days",
        "1",
        "--stream-epochs",
        "2",
        "--max-windows",
        "11",
        "--max-eval-users",
        "1000",
        "--batch-size",
        "32",
        "--bootstrap-samples",
        "200",
        "--training-sequences",
        str(data["training_sequences"]),
        "--output",
        str(cell_paths["core"]),
        "--checkpoint-dir",
        str(cell_paths["checkpoint"]),
    ]
    if dataset == "kuai":
        command.extend(["--max-users", str(data["max_users"][tier])])
    else:
        command.extend(["--prepared-data", str(data["prepared"]).format(tier=tier)])
    return command


def control_command(
    seed: int,
    device: str,
    cell_paths: dict[str, Path],
) -> list[str]:
    return [
        sys.executable,
        "scripts/streaming_value_control.py",
        "--protocol",
        "motivation_capacity_v2_streaming_control",
        "--device",
        device,
        "--seed",
        str(seed),
        "--run-result",
        str(cell_paths["core"]),
        "--checkpoint-dir",
        str(cell_paths["checkpoint"]),
        "--model-ts",
        "1",
        "3",
        "5",
        "7",
        "9",
        "11",
        "--max-eval-users",
        "1000",
        "--output",
        str(cell_paths["control"]),
    ]


def matrix_command(
    seed: int,
    device: str,
    cell_paths: dict[str, Path],
) -> list[str]:
    return [
        sys.executable,
        "scripts/cache_version_matrix.py",
        "--protocol",
        "motivation_capacity_v2_cache_version_matrix",
        "--device",
        device,
        "--seed",
        str(seed),
        "--run-result",
        str(cell_paths["core"]),
        "--checkpoint-dir",
        str(cell_paths["checkpoint"]),
        "--current-t",
        "11",
        "--max-eval-users",
        "1000",
        "--output",
        str(cell_paths["matrix"]),
    ]


def cost_command(
    tier: str,
    seed: int,
    device: str,
    cell_paths: dict[str, Path],
) -> list[str]:
    return [
        sys.executable,
        "scripts/operator_cost_scaling.py",
        "--protocol",
        "motivation_capacity_v2_operator_cost",
        "--device",
        device,
        "--seed",
        str(seed),
        "--run-result",
        str(cell_paths["core"]),
        "--checkpoint-dir",
        str(cell_paths["checkpoint"]),
        "--model-t",
        "11",
        "--seq-lens",
        "128",
        "--batch-sizes",
        "32",
        "--fixed-batch-size",
        "32",
        "--fixed-seq-len",
        "128",
        "--suffix-depths",
        "0",
        str(TIERS[tier]["num_layers"]),
        "--timing-repeats",
        "15",
        "--output",
        str(cell_paths["cost"]),
    ]


def main() -> None:
    args = parse_args()
    Path("results/motivation_scale").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/motivation_capacity_v2").mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for tier in TIERS:
            cell_paths = paths(args.dataset, tier, seed)
            commands = {
                "core": core_command(
                    args.dataset,
                    tier,
                    seed,
                    args.device,
                    cell_paths,
                ),
                "control": control_command(seed, args.device, cell_paths),
                "matrix": matrix_command(seed, args.device, cell_paths),
                "cost": cost_command(tier, seed, args.device, cell_paths),
            }
            for stage in args.stages:
                output_path = cell_paths[stage]
                if not args.force and complete(output_path, stage):
                    print(f"skip dataset={args.dataset} tier={tier} seed={seed} stage={stage}")
                    continue
                print(f"run dataset={args.dataset} tier={tier} seed={seed} stage={stage}")
                run(commands[stage], cell_paths["log"])
                if not complete(output_path, stage):
                    raise RuntimeError(f"incomplete artifact: {output_path}")
                print(f"done dataset={args.dataset} tier={tier} seed={seed} stage={stage}")


if __name__ == "__main__":
    main()
