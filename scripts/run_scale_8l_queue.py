#!/usr/bin/env python3
"""Resumable four-GPU launcher for the prospective 8L experiment chain.

The queue automates execution, not scientific authorization.  It stops at H/S
gates whose adjudication artifacts must be produced by frozen evaluators.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import scale_8l_common as scale

GPUS = (0, 1, 2, 3)


@dataclass(frozen=True)
class Job:
    stage: str
    model: str | None
    seed: int | None
    release: str | None
    artifact: Path

    def command(self, gpu: int) -> list[str]:
        if gpu not in GPUS:
            raise ValueError(f"GPU {gpu} is outside frozen allowlist {GPUS}")
        device = f"cuda:{gpu}"
        if self.stage == "s1_audit":
            return ["python", "scripts/audit_scale_8l_resources.py"]
        if self.stage == "s2_canary":
            return ["python", "scripts/eval_scale_8l_correctness_canary.py", "--device", device]
        if self.stage == "theta0":
            return ["torchrun", "--standalone", "--nproc_per_node=4", "scripts/train_scale_8l_fsdp_theta0.py", "--model", str(self.model), "--seed", str(self.seed)]
        if self.stage == "release":
            return ["torchrun", "--standalone", "--nproc_per_node=4", "scripts/train_scale_8l_fsdp_release.py", "--release", str(self.release), "--model", str(self.model), "--seed", str(self.seed)]
        raise ValueError(self.stage)


def model_jobs(stage: str, release: str | None = None) -> list[Job]:
    jobs = []
    for model in scale.MODELS:
        for seed in scale.SEEDS:
            if stage == "theta0":
                artifact = scale.OUTPUT / "theta0" / f"{model}_seed{seed}" / "train_result.json"
            else:
                assert release is not None
                artifact = scale.OUTPUT / "releases" / release / f"{model}_seed{seed}" / "train_result.json"
            jobs.append(Job(stage, model, seed, release, artifact))
    return jobs


def gate(path: Path, accepted: tuple[str, ...]) -> tuple[bool, str]:
    if not path.exists():
        return False, f"WAITING_FOR_FROZEN_ADJUDICATION:{path.relative_to(scale.ROOT)}"
    status = json.loads(path.read_text()).get("status")
    if status not in accepted:
        return False, f"BLOCKED_BY_GATE:{path.relative_to(scale.ROOT)}:{status}"
    return True, status


def next_wave() -> tuple[list[Job], str]:
    scale.contract()
    audit = Job("s1_audit", None, None, None, scale.OUTPUT / "s1_resource_audit.json")
    if not audit.artifact.exists():
        return [audit], "S1"
    audit_payload = json.loads(audit.artifact.read_text())
    if audit_payload.get("contract_sha256") != scale.sha256_file(scale.CONTRACT):
        return [], "BLOCKED:S1_ARTIFACT_CONTRACT_HASH_MISMATCH"
    audit_gate = audit_payload.get("gate", {})
    if not (
        audit_gate.get("f_has_history_beyond_512")
        and audit_gate.get("frozen_queries_candidates_and_base_features_reused")
        and audit_gate.get("long_training_launched") is False
    ):
        return [], "BLOCKED:S1_RESOURCE_OR_COVERAGE_GATE"
    canary = Job("s2_canary", None, 17, None, scale.OUTPUT / "s2_correctness_canary.json")
    if not canary.artifact.exists():
        return [canary], "S2"
    canary_payload = json.loads(canary.artifact.read_text())
    if canary_payload.get("contract_sha256") != scale.sha256_file(scale.CONTRACT):
        return [], "BLOCKED:S2_ARTIFACT_CONTRACT_HASH_MISMATCH"
    if canary_payload.get("status") != "passed":
        return [], "BLOCKED:S2_CORRECTNESS_CANARY"
    memory = Job("s2_fsdp_memory", None, 17, None, scale.OUTPUT / "s2_fsdp_training_preflight.json")
    if not memory.artifact.exists():
        return [], "WAITING_FOR_MANUAL_FSDP_PREFLIGHT: run torchrun --standalone --nproc_per_node=4 scripts/eval_scale_8l_fsdp_preflight.py"
    memory_payload = json.loads(memory.artifact.read_text())
    if memory_payload.get("contract_sha256") != scale.sha256_file(scale.CONTRACT):
        return [], "BLOCKED:S2_FSDP_ARTIFACT_CONTRACT_HASH_MISMATCH"
    if memory_payload.get("status") != "passed":
        return [], "BLOCKED:S2_TRAINING_MEMORY"
    trainer_canary = scale.OUTPUT / "trainer_canary/m0_f_seed17/train_result.json"
    if not trainer_canary.exists():
        return [], "WAITING_FOR_TRAINER_CANARY"
    trainer_payload = json.loads(trainer_canary.read_text())
    if (
        trainer_payload.get("contract_hash") != scale.sha256_file(scale.CONTRACT)
        or trainer_payload.get("status") != "scale_theta0_FSDP_trainer_canary_passed"
        or trainer_payload.get("checkpoint_retained") is not False
    ):
        return [], "BLOCKED:S2_TRAINER_CANARY"

    # Cost-aware prospective ordering: establish one complete M0-F seed17
    # scale chain before spending compute on replication seeds or M1.  The
    # frozen scope remains all three seeds; only execution order changes.
    pilot_theta = next(
        job for job in model_jobs("theta0")
        if job.model == "m0_f" and job.seed == 17
    )
    if not pilot_theta.artifact.exists():
        return [pilot_theta], "S3_PILOT_THETA0_M0_F_SEED17"
    passed, message = gate(
        scale.OUTPUT / "pilot/s3_m0_f_seed17_h_adjudication.json",
        ("passed", "scale_H_gate_passed"),
    )
    if not passed:
        return [], message

    for release, accepted in (
        ("r0", ("passed", "R0_identity_passed")),
        ("r1_edge1", ("passed", "scale_HS_gate_passed")),
        ("r1_edge2", ("passed", "scale_HS_gate_passed")),
        ("r2", ("passed", "scale_HS_gate_passed")),
    ):
        pilot_release = next(
            job for job in model_jobs("release", release)
            if job.model == "m0_f" and job.seed == 17
        )
        if not pilot_release.artifact.exists():
            return [pilot_release], f"S4_PILOT_{release}_M0_F_SEED17"
        passed, message = gate(
            scale.OUTPUT / f"pilot/s4_{release}_m0_f_seed17_adjudication.json",
            accepted,
        )
        if not passed:
            return [], message

    replication_gate = scale.OUTPUT / "pilot/replication_authorization.json"
    passed, message = gate(replication_gate, ("authorized", "pilot_chain_passed_replication_authorized"))
    if not passed:
        return [], "PILOT_CHAIN_COMPLETE_" + message

    pending = [job for job in model_jobs("theta0") if not job.artifact.exists()]
    if pending:
        return pending[:1], "S3_REPLICATION_THETA0_FSDP_ALL_GPUS"
    return [], "REPLICATION_THETA0_COMPLETE_PENDING_PER_SEED_H_AND_RELEASE_AUTHORIZATION"


def active() -> list[str]:
    output = subprocess.run(["ps", "-eo", "pid,etime,args"], text=True, capture_output=True, check=True).stdout
    names = ("train_scale_8l_", "eval_scale_8l_correctness_canary.py", "eval_scale_8l_fsdp_preflight.py", "audit_scale_8l_resources.py")
    return [line.strip() for line in output.splitlines() if any(name in line for name in names) and "run_scale_8l_queue.py" not in line]


def launch(wave: list[Job]) -> list[dict]:
    log_root = scale.OUTPUT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    launched = []
    for gpu, job in zip(GPUS, wave):
        suffix = "_".join(str(x) for x in (job.release, job.model, f"seed{job.seed}" if job.seed is not None else None) if x is not None)
        log = log_root / f"{job.stage}_{suffix or 'global'}.log"
        command = job.command(gpu)
        env = {**os.environ, "PYTHONPATH": "src:scripts", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        stream = log.open("ab")
        process = subprocess.Popen(command, cwd=scale.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        stream.close()
        launched.append({"pid": process.pid, "gpu": gpu, "stage": job.stage, "model": job.model, "seed": job.seed, "release": job.release, "log": str(log)})
    return launched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--commands", action="store_true")
    parser.add_argument("--launch-wave", action="store_true")
    parser.add_argument("--run-until-gate", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    running = active()
    wave, stage = next_wave()
    if args.commands:
        for gpu, job in zip(GPUS, wave):
            print(shlex.join(["env", "PYTHONPATH=src:scripts", "OMP_NUM_THREADS=8", *job.command(gpu)]))
        if not wave:
            print(stage)
        return
    if args.launch_wave:
        if running:
            raise RuntimeError("scale jobs already active")
        print(json.dumps({"stage": stage, "launched": launch(wave) if wave else []}, indent=2))
        return
    if args.run_until_gate:
        while True:
            running = active()
            if running:
                time.sleep(args.poll_seconds)
                continue
            wave, stage = next_wave()
            if not wave:
                print(stage)
                return
            print(json.dumps({"stage": stage, "launched": launch(wave)}, indent=2), flush=True)
            time.sleep(args.poll_seconds)
    else:
        print(json.dumps({
            "status": stage, "active_processes": running,
            "next_jobs": [{"stage": job.stage, "model": job.model, "seed": job.seed, "release": job.release, "artifact": str(job.artifact)} for job in wave],
            "GPU_allowlist": list(GPUS),
            "scientific_gates_are_not_bypassed": True,
        }, indent=2))


if __name__ == "__main__":
    main()
