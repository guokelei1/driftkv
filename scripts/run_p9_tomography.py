#!/usr/bin/env python3
"""Deterministic P9.2 coarse-scan job ledger; it never starts controller work."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "results/p9/tomography_raw/full"
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
            raise ValueError("P9 GPU allowlist is physical GPU 0,1")
        # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to cuda:0.
        # This both enforces the allowlist and prevents a child process from
        # touching physical GPU 2/3.
        return (
            f"PYTHONPATH=src CUDA_VISIBLE_DEVICES={gpu} python scripts/eval_p9_tomography_raw.py "
            f"--release {self.release} --model {self.model} --seed {self.seed} --device cuda:0"
        )


def jobs() -> tuple[Job, ...]:
    return tuple(Job(release, model, seed) for release in RELEASES for model in MODELS for seed in SEEDS)


def complete(job: Job) -> bool:
    return (job.output / "raw_manifest.json").exists() and (job.output / "F_fidelity_tomography.parquet").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--next", type=int, default=0, help="emit the next N deterministic jobs as JSON")
    args = parser.parse_args()
    all_jobs = jobs()
    done = [job for job in all_jobs if complete(job)]
    pending = [job for job in all_jobs if not complete(job)]
    if args.status or not args.next:
        print(json.dumps({
            "status": "P9_2_coarse_scan_pending" if pending else "P9_2_coarse_scan_raw_complete",
            "total": len(all_jobs), "complete": len(done), "pending": len(pending),
            "completed_jobs": [asdict(job) for job in done],
        }, indent=2))
    if args.next:
        print(json.dumps([asdict(job) for job in pending[:args.next]], indent=2))


if __name__ == "__main__":
    main()
