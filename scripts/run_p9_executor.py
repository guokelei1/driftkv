#!/usr/bin/env python3
"""Deterministic all-cell P9.4 executor replay ledger."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "results/p9/executor_raw/full"
RELEASES = ("r0", "r1_edge1", "r1_edge2", "r2")
MODELS = ("m0_f", "m1")
SEEDS = (17, 37, 71)


@dataclass(frozen=True)
class Job:
    release: str
    model: str
    seed: int

    @property
    def output(self) -> Path:
        return RAW_ROOT / self.release / f"{self.model}_seed{self.seed}"

    def command(self, gpu: int) -> str:
        if gpu not in (0, 1):
            raise ValueError("P9.4 GPU allowlist is physical GPU 0,1")
        return (
            f"PYTHONPATH=src CUDA_VISIBLE_DEVICES={gpu} python scripts/eval_p9_executor_raw.py "
            f"--release {self.release} --model {self.model} --seed {self.seed} --device cuda:0"
        )


def jobs() -> tuple[Job, ...]:
    return tuple(Job(release, model, seed) for release in RELEASES for model in MODELS for seed in SEEDS)


def complete(job: Job) -> bool:
    return (job.output / "raw_manifest.json").exists() and (job.output / "F_fidelity_executor.parquet").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--next", type=int, default=0)
    args = parser.parse_args()
    all_jobs = jobs(); done = [job for job in all_jobs if complete(job)]; pending = [job for job in all_jobs if not complete(job)]
    print(json.dumps({
        "status": "P9_4_pending" if pending else "P9_4_raw_complete",
        "total": len(all_jobs), "complete": len(done), "pending": len(pending),
        "completed_jobs": [asdict(job) for job in done], "next_jobs": [asdict(job) for job in pending[:args.next]],
    }, indent=2))


if __name__ == "__main__":
    main()
