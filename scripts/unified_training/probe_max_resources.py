#!/usr/bin/env python3
"""Run one authorized, bounded Max resource probe (never formal training)."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=[64, 80, 96, 128], required=True)
    args = parser.parse_args()
    execution = ROOT / f"configs/unified_training_2026_09/max_probe_b{args.batch}_execution.yaml"
    e = yaml.safe_load(execution.read_text())
    contract = ROOT / e["frozen_parent"]["contract"]
    out = ROOT / e["output_amendment"]["root"]
    out.mkdir(exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "scripts/train_yambda500m_foundation_fsdp.py",
        "--version",
        "v0",
        "--branch",
        "shared",
        "--launch-contract",
        str(contract),
        "--execution-contract",
        str(execution),
        "--manifest-dir",
        "data/manifests/yambda5b_max_200k_hstu_native_v1",
        "--training-block",
        "foundation",
        "--train-start-day",
        "203",
        "--train-end-day",
        "217",
        "--output",
        str(out / "train"),
        "--oov-buckets",
        "256",
        "--passes",
        "1",
        "--global-batch-size",
        str(args.batch),
        "--canary-steps",
        "12",
    ]
    for flag, key in [
        ("history-threads", "history_threads"),
        ("arrow-cpu-threads", "arrow_cpu_threads"),
        ("arrow-io-threads", "arrow_io_threads"),
        ("torch-cpu-threads", "torch_cpu_threads"),
        ("cpu-affinity-by-rank", "cpu_affinity_by_rank"),
    ]:
        command += ["--" + flag, str(e["training_runtime"][key])]
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.time()
    with (out / "train.log").open("x") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "OMP_NUM_THREADS": "4",
                "PYTHONUNBUFFERED": "1",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    (out / "exit_status.txt").write_text(str(result.returncode) + "\n")
    if result.returncode:
        raise RuntimeError(f"probe failed, inspect {out}/train.log")
    import torch

    payload = torch.load(out / "train/checkpoint_100.pt", map_location="cpu", weights_only=False)
    assert (
        payload["status"] == "distributed_canary_checkpoint"
        and payload["parent_checkpoint_sha256"] is None
    )
    assert payload["config"]["num_layers"] == 16 and payload["config"]["hidden_size"] == 320
    assert all(torch.isfinite(t).all().item() for t in payload["model"].values())
    r = json.loads((out / "train/train_result.json").read_text())
    assert r["steps"] == 12 and r["local_batch_sizes_by_rank"] == [args.batch // 4] * 4
    assert all(m["canary_batch_width_min"] == 1024 for m in r["rank_metrics"])
    peak = max(m["peak_reserved_mib"] for m in r["rank_metrics"])
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    with (out / "train/checkpoint_100.pt").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "status": "completed_resource_probe",
        "global_batch": args.batch,
        "peak_reserved_mib": peak,
        "total_device_mib": total,
        "reserved_memory_fraction": peak / total,
        "passes_15pct_memory_reserve": peak <= 0.85 * total,
        "all_ranks_all_steps_width": 1024,
        "requests_per_second": r["global_requests_per_second"],
        "median_step_seconds": r["median_synchronized_step_seconds"],
        "wall_seconds": time.time() - started,
        "checkpoint_sha256": digest,
        "quality_metrics_read": False,
        "formal_training": False,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
