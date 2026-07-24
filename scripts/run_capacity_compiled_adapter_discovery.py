from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TIERS = ("small", "medium", "large")
PROTOCOL = "compiled_capacity_adapter_discovery_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("kuai", "qb", "qk"),
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--selection",
        default="results/motivation_scale/design_discovery_seeds.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        result.get("protocol") == PROTOCOL
        and result.get("study_stage")
        == "motivation_selected_seed_discovery"
        and result.get("selection", {}).get("recommended", {}).get("test")
    )


def main() -> None:
    args = parse_args()
    selection = json.loads(Path(args.selection).read_text())
    log_dir = Path("logs/capacity_compiled_adapter_discovery")
    log_dir.mkdir(parents=True, exist_ok=True)
    for tier in TIERS:
        cell = f"{args.dataset}_{tier}"
        seed = int(selection["cells"][cell]["selected_seed"])
        output = Path(
            "results/motivation_scale/"
            f"{cell}_v2_compiled_adapter_discovery_seed{seed}.json"
        )
        if not args.force and complete(output):
            print(f"skip {cell} seed{seed}", flush=True)
            continue
        command = [
            sys.executable,
            "scripts/low_rank_migration_search.py",
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--run-result",
            f"results/motivation_scale/{cell}_v2_core_seed{seed}.json",
            "--checkpoint-dir",
            f"checkpoints/motivation_capacity_v2/{cell}_seed{seed}",
            "--model-t",
            "11",
            "--max-users",
            "1000",
            "--fit-users",
            "40",
            "--probe-users",
            "60",
            "--timing-repeats",
            "3",
            "--protocol",
            PROTOCOL,
            "--study-stage",
            "motivation_selected_seed_discovery",
            "--output",
            str(output),
        ]
        log_path = log_dir / f"{cell}_seed{seed}.log"
        print(f"run {cell} seed{seed} on {args.device}", flush=True)
        with log_path.open("w") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if not complete(output):
            raise RuntimeError(f"incomplete result: {output}")
        print(f"done {cell} seed{seed}", flush=True)


if __name__ == "__main__":
    main()
