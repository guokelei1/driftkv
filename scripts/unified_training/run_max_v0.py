#!/usr/bin/env python3
"""Prepared Max foundation launch; requires a recorded explicit user launch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXECUTION = ROOT / "configs/unified_training_2026_09/max_v0_prepared_execution.yaml"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume the latest complete local recovery generation")
    args = parser.parse_args()
    execution = yaml.safe_load(EXECUTION.read_text())
    cp = ROOT / execution["frozen_parent"]["contract"]
    contract = yaml.safe_load(cp.read_text())
    assert sha(cp) == execution["frozen_parent"]["contract_sha256"]
    for key, path in contract["frozen_inputs"].items():
        if not key.endswith("_sha256"):
            assert sha(ROOT / path) == contract["frozen_inputs"][key + "_sha256"], key
    out = ROOT / execution["output_amendment"]["root"]
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
        str(cp),
        "--execution-contract",
        str(EXECUTION),
        "--manifest-dir",
        str((ROOT / contract["frozen_inputs"]["request_manifest"]).parent),
        "--training-block",
        "foundation",
        "--train-start-day",
        "0",
        "--train-end-day",
        "217",
        "--output",
        str(out / "checkpoint"),
        "--oov-buckets",
        "256",
        "--passes",
        "1",
        "--global-batch-size",
        "80",
        "--progress-interval",
        "500",
    ]
    for flag, key in [
        ("history-threads", "history_threads"),
        ("arrow-cpu-threads", "arrow_cpu_threads"),
        ("arrow-io-threads", "arrow_io_threads"),
        ("torch-cpu-threads", "torch_cpu_threads"),
        ("cpu-affinity-by-rank", "cpu_affinity_by_rank"),
    ]:
        command += ["--" + flag, str(execution["training_runtime"][key])]
    recovery = contract['training']['recovery']
    command += ['--recovery-interval-steps', str(recovery['interval_steps']),
                '--recovery-first-step', str(recovery['first_step'])]
    resume_path = None
    if args.resume:
        complete = sorted((out / 'checkpoint/recovery').glob('step_*/complete.json'))
        if not complete:
            raise RuntimeError('No complete recovery checkpoint is available')
        resume_path = complete[-1].parent
        command += ['--resume-recovery', str(resume_path)]
    if args.print_command:
        print(
            json.dumps(
                {
                    "command": command,
                    "launch_authorized": contract["authorization"]["explicit_user_launch_received"],
                },
                indent=2,
            )
        )
        return
    if not contract["authorization"]["explicit_user_launch_received"]:
        raise RuntimeError(
            "Preparation only: record the user’s explicit Max V0 launch before execution"
        )
    readiness_path = ROOT / "results/unified_training_2026_09/max/preparation/readiness.json"
    readiness = json.loads(readiness_path.read_text())
    assert readiness["status"] == "ready_for_explicit_training_launch"
    for path, digest in readiness["code_sha256"].items():
        assert sha(ROOT / path) == digest, path
    out.mkdir(parents=True, exist_ok=args.resume)
    attempt = f'resume_{resume_path.name}_{time.time_ns()}' if args.resume else 'train'
    configuration_name = f'{attempt}.configuration.json' if args.resume else 'configuration.json'
    (out / configuration_name).write_text(
        json.dumps(
            {
                "contract": str(cp.relative_to(ROOT)),
                "contract_sha256": sha(cp),
                "execution_sha256": sha(EXECUTION),
                "readiness_sha256": sha(readiness_path),
                "command": command,
            },
            indent=2,
        )
        + "\n"
    )
    started = time.time()
    (out / "progress.json").write_text(
        json.dumps({"stage": "train", "started_at_unix": started}) + "\n"
    )
    try:
        with (out / f"{attempt}.log").open("x") as log:
            subprocess.run(
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
                check=True,
            )
        import torch

        checkpoint = out / "checkpoint/checkpoint_100.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["version"] == "v0" and payload["parent_checkpoint_sha256"] is None
        assert (
            payload["training_day_range"] == [0, 217] and payload["training_epochs_completed"] == 1
        )
        assert all(torch.isfinite(t).all().item() for t in payload["model"].values())
        seal = {
            "status": "max_v0_checkpoint_sealed",
            "version": "v0",
            "seed": 17,
            "checkpoint_sha256": sha(checkpoint),
            "contract_sha256": sha(cp),
            "training_day_range": [0, 217],
            "epochs": 1,
            "parent_checkpoint_sha256": None,
        }
        (checkpoint.parent / "checkpoint.seal.json").write_text(json.dumps(seal, indent=2) + "\n")
        recovery_root = checkpoint.parent / 'recovery'
        if recovery_root.exists():
            retired = [json.loads(p.read_text()) for p in sorted(recovery_root.glob('step_*/complete.json'))]
            (out / 'recovery_retirement.json').write_text(json.dumps({
                'reason': 'final checkpoint validated and sealed', 'generations': retired,
            }, indent=2) + '\n')
            shutil.rmtree(recovery_root)
    except Exception as error:
        (out / "exit_status.txt").write_text("1\n")
        (out / "progress.json").write_text(
            json.dumps({"stage": "failed", "error": str(error)}) + "\n"
        )
        raise
    (out / "exit_status.txt").write_text("0\n")
    (out / "progress.json").write_text(
        json.dumps({"stage": "complete", "elapsed_seconds": time.time() - started}) + "\n"
    )


if __name__ == "__main__":
    main()
