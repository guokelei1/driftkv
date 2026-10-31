#!/usr/bin/env python3
"""Small launcher for maintained workflows; implementations live in src/experiments."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

COMMANDS = {
    "train-yambda": "hstu_kvcache.training.yambda_fsdp",
    "train-recflow": "hstu_kvcache.training.recflow.train_feedback",
    "evaluate-yambda": "hstu_kvcache.evaluation.yambda",
    "evaluate-recflow": "hstu_kvcache.evaluation.recflow.evaluate_feedback",
    "prepare-yambda": "hstu_kvcache.data.yambda.build_scales",
    "prepare-max": "hstu_kvcache.data.yambda.prepare_max",
    "download": "hstu_kvcache.data.download",
    "figure9a": "experiments.figure9a.run",
    "figure9max": "experiments.figure9max.run",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python scripts/evokv.py COMMAND [arguments]\n")
        for command, module in COMMANDS.items():
            print(f"  {command:20} {module}")
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command: {command}. Use --help to list commands.")
    sys.argv = [COMMANDS[command], *sys.argv[2:]]
    runpy.run_module(COMMANDS[command], run_name="__main__")


if __name__ == "__main__":
    main()
