#!/usr/bin/env python3
"""Deferred 202 job: train only V0/V1, compare on D20, then exit.

The current configuration deliberately rejects formal execution until the user
gives the next launch instruction. No background queue is created here.
"""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/foundation_recflow/configs/large_v0_v1_202.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not config.get("training_authorized", False):
        raise SystemExit("Waiting for the user's further instruction; no formal training started.")
    if config["machine"] != "g202" or config["stop_after"] != "v1":
        raise ValueError("This job is restricted to machine 202 and Large V0/V1")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, config["physical_gpus"])),
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH="src")
    base = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4"]
    train = base + ["src/hstu_kvcache/training/recflow/train_feedback.py", "--mode", "train", "--config", str(args.config), "--output", str(args.output)]
    if args.resume:
        train.append("--resume")
    subprocess.run(train, env=env, check=True)
    subprocess.run(base + ["src/hstu_kvcache/evaluation/recflow/evaluate_feedback.py", "--config", str(args.config),
        "--training", str(args.output), "--output", str(args.output / "paired_d20")], env=env, check=True)


if __name__ == "__main__":
    main()
