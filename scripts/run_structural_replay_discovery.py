from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TIERS = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("kuai", "qb", "qk"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--selection",
        default="results/motivation_scale/design_discovery_seeds.json",
    )
    parser.add_argument("--max-users", type=int, default=300)
    parser.add_argument("--probe-users", type=int, default=60)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--all-intervals", action="store_true")
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
        result.get("protocol") == "structural_replay_discovery_v1"
        and result.get("probe", {}).get("summary", {}).get("configs")
        and result.get("test", {}).get("summary", {}).get("configs")
    )


def main() -> None:
    args = parse_args()
    selection = json.loads(Path(args.selection).read_text())
    log_dir = Path("logs/structural_replay_discovery")
    log_dir.mkdir(parents=True, exist_ok=True)
    for tier in TIERS:
        cell = f"{args.dataset}_{tier}"
        seed = int(selection["cells"][cell]["selected_seed"])
        variant = "interval_discovery" if args.all_intervals else "discovery"
        output = Path(
            "results/motivation_scale/"
            f"{cell}_v2_structural_replay_{variant}_seed{seed}.json"
        )
        if not args.force and complete(output):
            print(f"skip {cell} seed{seed}", flush=True)
            continue
        command = [
            sys.executable,
            "scripts/structural_replay_search.py",
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
            str(args.max_users),
            "--probe-users",
            str(args.probe_users),
            "--timing-repeats",
            str(args.timing_repeats),
            "--output",
            str(output),
        ]
        if args.all_intervals:
            command.append("--all-intervals")
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
