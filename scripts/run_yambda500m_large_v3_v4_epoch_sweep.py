#!/usr/bin/env python3
"""Prepare, canary, run, and summarize the Large v3->v4 epoch sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_large_v3_v4_epoch_sweep_v1.yaml"
CANARY_ACK = "RUN_LARGE_V3_V4_EPOCH_SWEEP_CANARY"
FORMAL_ACK = "RUN_LARGE_V3_V4_EPOCH_SWEEP"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def epoch_label(epoch: float) -> str:
    return f"{epoch:.1f}".replace(".", "p")


class EpochSweep:
    def __init__(self, contract_path: Path = CONTRACT) -> None:
        self.contract_path = contract_path.resolve()
        self.contract = yaml.safe_load(self.contract_path.read_text(encoding="utf-8"))
        self.contract_hash = sha256_file(self.contract_path)
        self.output = (ROOT / self.contract["outputs"]["root"]).resolve()
        self.canary_dir = (ROOT / self.contract["outputs"]["canary"]).resolve()
        self.checkpoint_dir = (ROOT / self.contract["outputs"]["checkpoints"]).resolve()
        self.full_dir = (ROOT / self.contract["outputs"]["full_only"]).resolve()
        self.logs = self.output / "logs"
        self.state = self.output / "state.json"
        self.manifest = (ROOT / self.contract["frozen_inputs"]["manifest"]).resolve().parent
        self.dataset = (ROOT / self.contract["frozen_inputs"]["dataset_manifest"]).resolve()
        frozen_parent = self.contract["frozen_parent"]
        parent_key = next(
            (key for key in ("parent_checkpoint", "v3_checkpoint", "v4_e2_checkpoint") if key in frozen_parent),
            None,
        )
        if parent_key is None:
            raise RuntimeError("staged epoch sweep contract has no parent checkpoint")
        self.parent = (ROOT / frozen_parent[parent_key]).resolve()
        self.epochs = tuple(float(value) for value in self.contract["training"]["checkpoint_epochs"])
        self.gpus = list(map(int, self.contract["resource_plan"]["physical_gpus"]))
        self.world = int(self.contract["resource_plan"]["world_size"])
        self._validate_contract()

    def _validate_contract(self) -> None:
        if self.epochs != (0.5, 1.0, 1.5, 2.0):
            raise RuntimeError("Large v3->v4 epoch endpoints drifted")
        if self.contract["scope"]["branches"]["D14"]["training_days_half_open"] != [259, 273]:
            raise RuntimeError("Large v3->v4 training window drifted")
        if self.contract["evaluation"]["day_range_half_open"] != [273, 287]:
            raise RuntimeError("Large v3->v4 E14 window drifted")
        if self.world != 4 or self.gpus != [0, 1, 2, 3]:
            raise RuntimeError("epoch sweep is frozen to one four-rank GPU0/1/2/3 job")
        for section, pairs in (
            ("frozen_inputs", (
                "unified_scale_contract", "dataset_manifest", "item_mapping",
                "manifest", "requests_quality", "requests_fidelity",
            )),
            ("frozen_parent", (
                "base_large_contract", "base_execution_contract", "v3_checkpoint",
                "v3_checkpoint_seal", "original_one_epoch_v4_checkpoint",
            )),
        ):
            values = self.contract[section]
            for key in pairs:
                path = (ROOT / values[key]).resolve()
                if not path.exists() or sha256_file(path) != values[f"{key}_sha256"]:
                    raise RuntimeError(f"epoch sweep frozen input mismatch: {section}.{key}")
        for key in ("observed_full_only_adjudication", "observed_admission_seal"):
            path = (ROOT / self.contract["evidence_boundary"][key]).resolve()
            if sha256_file(path) != self.contract["evidence_boundary"][f"{key}_sha256"]:
                raise RuntimeError(f"epoch sweep observed failure input mismatch: {key}")

    @property
    def gpu_env(self) -> dict[str, str]:
        return {
            **os.environ,
            "PYTHONPATH": "src",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus)),
            "OMP_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
        }

    @property
    def cpu_args(self) -> list[str]:
        affinity = ";".join(
            ",".join(map(str, range(rank * 14, (rank + 1) * 14)))
            for rank in range(self.world)
        )
        return [
            "--history-threads", "14", "--arrow-cpu-threads", "14",
            "--arrow-io-threads", "4", "--torch-cpu-threads", "4",
            "--cpu-affinity-by-rank", affinity,
        ]

    @property
    def distributed(self) -> list[str]:
        return ["torchrun", "--standalone", f"--nproc_per_node={self.world}"]

    def checkpoint(self, epoch: float) -> Path:
        raw = f"{epoch:.6f}".rstrip("0").rstrip(".").replace(".", "p")
        return self.checkpoint_dir / f"checkpoint_epoch_{raw}.pt"

    def run(self, name: str, command: list[str], *, gpu: bool = False) -> dict:
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{name}.log"
        runtime_path = self.logs / f"{name}.runtime.json"
        if log_path.exists() or runtime_path.exists():
            raise RuntimeError(f"existing epoch-sweep step log requires audit: {name}")
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=self.gpu_env if gpu else {
                    **os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1",
                }, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            returncode = process.wait()
        runtime = {
            "name": name, "returncode": returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "command": command,
        }
        atomic_json(runtime_path, runtime)
        if returncode:
            raise RuntimeError(f"epoch-sweep step failed ({returncode}): {name}")
        return runtime

    def preflight(self) -> None:
        output = subprocess.check_output([
            "nvidia-smi", f"--id={','.join(map(str, self.gpus))}",
            "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",
        ], text=True)
        rows = [tuple(int(part.strip()) for part in line.split(",")) for line in output.splitlines()]
        if [row[0] for row in rows] != self.gpus or any(row[1] < 42_000 for row in rows):
            raise RuntimeError(f"epoch-sweep GPUs are not free enough: {rows}")
        free_gib = shutil.disk_usage(ROOT).free / 2**30
        if free_gib < float(self.contract["resource_plan"]["minimum_free_workspace_gib"]):
            raise RuntimeError(f"epoch sweep has only {free_gib:.1f} GiB free workspace")

    def train_command(self, output: Path, *, canary: bool) -> list[str]:
        training = self.contract["training"]
        command = [
            *self.distributed, "scripts/train_yambda500m_foundation_fsdp.py",
            "--version", "v4", "--branch", "D14",
            "--launch-contract", str(self.contract_path),
            "--manifest-dir", str(self.manifest), "--training-block", "matrix_horizon",
            "--parent", str(self.parent), "--output", str(output),
            "--oov-buckets", str(self.contract["model"]["oov_buckets"]),
            "--passes", str(training["passes"]),
            "--checkpoint-epochs", ",".join(map(str, self.epochs)),
            "--global-batch-size", str(training["global_batch_size"]),
            "--train-start-day", "259", "--train-end-day", "273",
            "--progress-interval", "100", *self.cpu_args,
        ]
        if canary:
            command.extend(["--canary-steps", str(self.contract["resource_plan"]["focused_canary_steps"])])
        return command

    def eval_command(self, output: Path, currents: dict[str, Path], *, canary: bool) -> list[str]:
        command = [
            *self.distributed, "scripts/evaluate_yambda500m_release_candidates_raw.py",
            "--stage", "large_D14_v3_to_v4_E14_epoch_sweep_canary" if canary else "large_D14_v3_to_v4_E14_epoch_sweep",
            "--block", "matrix_horizon", "--training-block", "matrix_horizon",
            "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
            "--parent", f"v3={self.parent}",
            "--start-day", "273", "--end-day", "287",
            "--training-start-day", "259", "--training-end-day", "273",
            "--batch-size", "64", "--output", str(output), *self.cpu_args,
        ]
        for name, path in currents.items():
            command.extend(["--current", f"{name}={path}"])
        if canary:
            command.extend([
                "--max-users", str(self.contract["resource_plan"]["focused_canary_eval_users_per_rank"]),
                "--allow-canary-checkpoints",
            ])
        return command

    def write_state(self, status: str, **values: object) -> None:
        atomic_json(self.state, {
            "status": status, "contract": str(self.contract_path.relative_to(ROOT)),
            "contract_sha256": self.contract_hash, **values,
        })

    def canary(self, acknowledgement: str | None) -> None:
        if acknowledgement != CANARY_ACK:
            raise RuntimeError(f"canary requires --acknowledge {CANARY_ACK}")
        self.preflight()
        train_dir = self.canary_dir / "train"
        eval_dir = self.canary_dir / "full_only"
        if self.canary_dir.exists():
            raise RuntimeError(f"existing canary directory requires audit: {self.canary_dir}")
        self.run("canary_train", self.train_command(train_dir, canary=True), gpu=True)
        checkpoint = train_dir / "checkpoint_100.pt"
        aliases = {f"v4_e{epoch_label(epoch)}": checkpoint for epoch in self.epochs}
        self.run("canary_full_only_raw", self.eval_command(eval_dir, aliases, canary=True), gpu=True)
        train_result = json.loads((train_dir / "train_result.json").read_text(encoding="utf-8"))
        seal = json.loads((eval_dir / "raw.seal.json").read_text(encoding="utf-8"))
        peak_limit = float(self.contract["resource_plan"]["focused_canary_peak_reserved_mib_limit"])
        training_peak = max(
            float(value["peak_reserved_mib"])
            for value in train_result["rank_metrics"]
        )
        evaluation_peak = max(
            float(value["peak_reserved_mib"])
            for value in seal["execution_runtime"]["peak_memory_by_rank"]
        )
        checks = {
            "finite_loss": bool(torch.isfinite(torch.tensor(train_result["mean_rank0_loss"]))),
            "world_size_four": train_result["world_size"] == self.world,
            "four_candidate_aliases": len(seal["currents"]) == 4,
            "raw_sealed": seal["status"] == "release_candidate_full_only_raw_sealed_before_label_join",
            "no_reuse": not seal["contains_reuse"],
            "training_peak_under_limit": training_peak < peak_limit,
            "five_model_evaluation_peak_under_limit": evaluation_peak < peak_limit,
        }
        payload = {
            "status": "large_v3_v4_epoch_sweep_canary_passed" if all(checks.values()) else "large_v3_v4_epoch_sweep_canary_failed",
            "contract_sha256": self.contract_hash, "checks": checks,
            "training_peak_reserved_mib": training_peak,
            "five_model_evaluation_peak_reserved_mib": evaluation_peak,
            "peak_reserved_mib_limit": peak_limit,
            "train_checkpoint_sha256": sha256_file(checkpoint),
            "raw_sha256": seal["raw_sha256"],
        }
        atomic_json(self.canary_dir / "summary.json", payload)
        self.write_state(payload["status"])
        if not all(checks.values()):
            raise RuntimeError(f"epoch-sweep canary failed: {checks}")

    def validate_formal_checkpoints(self) -> dict[str, dict]:
        result_path = self.checkpoint_dir / "train_result.json"
        paths = [self.checkpoint(epoch) for epoch in self.epochs]
        if not result_path.exists() or not all(path.exists() for path in paths):
            raise RuntimeError("formal staged training artifacts are incomplete")
        metadata: dict[str, dict] = {}
        for epoch, path in zip(self.epochs, paths, strict=True):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["status"] != "formal_staged_epoch_checkpoint":
                raise RuntimeError(f"unexpected staged checkpoint status: {path}")
            if float(payload["training_epochs_completed"]) != epoch or payload["parent_checkpoint_sha256"] != sha256_file(self.parent):
                raise RuntimeError(f"staged checkpoint lineage mismatch: {path}")
            metadata[epoch_label(epoch)] = {
                "epoch": epoch, "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "cache_producer_sha256": payload["cache_producer_sha256"],
            }
            del payload
        original = torch.load(
            ROOT / self.contract["frozen_parent"]["original_one_epoch_v4_checkpoint"],
            map_location="cpu", weights_only=False,
        )
        metadata["1p0"]["matches_original_v4_cache_producer"] = (
            metadata["1p0"]["cache_producer_sha256"] == original["cache_producer_sha256"]
        )
        del original
        atomic_json(self.checkpoint_dir / "checkpoints.seal.json", {
            "status": "large_v3_v4_staged_checkpoints_sealed",
            "contract_sha256": self.contract_hash, "checkpoints": metadata,
        })
        return metadata

    def summarize(self, metadata: dict[str, dict]) -> None:
        report_path = self.full_dir / "adjudication.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        old = report["parent_absolute"]["hstu_native"]
        rows = []
        for epoch in self.epochs:
            name = f"v4_e{epoch_label(epoch)}"
            candidate = report["candidates"][name]
            new = candidate["absolute"]["hstu_native"]
            paired = candidate["paired_release_gain"]["parent_minus_current_log_loss"]
            gates = {
                "AUC_positive": new["ROC_AUC"] > old["ROC_AUC"],
                "loss_positive": old["log_loss"] > new["log_loss"],
                "Brier_not_worse": new["Brier"] <= old["Brier"],
                "bootstrap_lower_positive": paired["user_cluster_bootstrap_95CI"]["p2_5"] > 0.0,
            }
            rows.append({
                "candidate": name, "training_epochs": epoch,
                "new_vs_old_AUC_relative_percent": 100 * (new["ROC_AUC"] - old["ROC_AUC"]) / old["ROC_AUC"],
                "new_loss_reduction_percent": 100 * (old["log_loss"] - new["log_loss"]) / old["log_loss"],
                "new_Brier_reduction_percent": 100 * (old["Brier"] - new["Brier"]) / old["Brier"],
                "gates": gates, "all_gates_pass": all(gates.values()),
                "checkpoint": metadata[epoch_label(epoch)],
            })
        payload = {
            "status": "large_v3_v4_epoch_sweep_full_only_complete",
            "contract_sha256": self.contract_hash,
            "evidence_boundary": self.contract["evidence_boundary"]["interpretation"],
            "raw_sha256": report["raw_sha256"], "rows": rows,
        }
        atomic_json(ROOT / self.contract["outputs"]["summary_json"], payload)
        lines = [
            "# Large D14 v3→v4 cumulative epoch sweep", "",
            "Status: **Full-only complete**. This is post-hoc endpoint-strength development, not independent Large qualification.", "",
            "| Epoch | New vs Old AUC | Loss reduction | Brier reduction | Strict gate | Original-v4 producer match |",
            "| ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for row in rows:
            match = row["checkpoint"].get("matches_original_v4_cache_producer")
            lines.append(
                f"| {row['training_epochs']:.1f} | {row['new_vs_old_AUC_relative_percent']:+.3f}% | "
                f"{row['new_loss_reduction_percent']:+.3f}% | {row['new_Brier_reduction_percent']:+.3f}% | "
                f"{'PASS' if row['all_gates_pass'] else 'FAIL'} | "
                f"{'N/A' if match is None else str(match)} |"
            )
        atomic_text(ROOT / self.contract["outputs"]["summary_markdown"], "\n".join(lines) + "\n")

    def formal(self, acknowledgement: str | None) -> None:
        if acknowledgement != FORMAL_ACK:
            raise RuntimeError(f"formal run requires --acknowledge {FORMAL_ACK}")
        canary = self.canary_dir / "summary.json"
        if not canary.exists():
            raise RuntimeError("formal epoch sweep requires the focused canary first")
        canary_payload = json.loads(canary.read_text(encoding="utf-8"))
        if canary_payload.get("status") != "large_v3_v4_epoch_sweep_canary_passed" or canary_payload.get("contract_sha256") != self.contract_hash:
            raise RuntimeError("formal epoch sweep is not bound to a passing current-contract canary")
        self.preflight()
        if not self.checkpoint_dir.exists():
            self.run("formal_train", self.train_command(self.checkpoint_dir, canary=False), gpu=True)
        metadata = self.validate_formal_checkpoints()
        raw, seal, report = self.full_dir / "raw.parquet", self.full_dir / "raw.seal.json", self.full_dir / "adjudication.json"
        if not report.exists():
            if self.full_dir.exists() and not (raw.exists() and seal.exists()):
                raise RuntimeError(f"partial epoch-sweep Full directory requires audit: {self.full_dir}")
            if not self.full_dir.exists():
                currents = {
                    f"v4_e{epoch_label(epoch)}": self.checkpoint(epoch)
                    for epoch in self.epochs
                }
                self.run("formal_full_only_raw", self.eval_command(self.full_dir, currents, canary=False), gpu=True)
            self.run("formal_full_only_adjudicate", [
                sys.executable, "scripts/adjudicate_yambda500m_release_candidates.py",
                "--raw", str(raw), "--seal", str(seal),
                "--labels", str(self.manifest / "requests_quality.parquet"),
                "--output", str(report),
            ])
        if json.loads(report.read_text(encoding="utf-8"))["raw_sha256"] != sha256_file(raw):
            raise RuntimeError("epoch-sweep adjudication/raw mismatch")
        self.summarize(metadata)
        self.write_state("large_v3_v4_epoch_sweep_full_only_complete")

    def plan(self) -> None:
        payload = {
            "contract": str(self.contract_path.relative_to(ROOT)),
            "contract_sha256": self.contract_hash,
            "parent": str(self.parent.relative_to(ROOT)),
            "training_window": [259, 273], "evaluation_window": [273, 287],
            "checkpoint_epochs": list(self.epochs),
            "expected_checkpoint_steps": self.contract["training"]["expected_checkpoint_steps"],
            "formal_training_wall_clock_estimate_hours": self.contract["resource_plan"]["formal_training_wall_clock_estimate_hours"],
            "formal_full_eval_wall_clock_estimate_hours": self.contract["resource_plan"]["formal_full_eval_wall_clock_estimate_hours"],
            "formal_launch_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    def status(self) -> None:
        payload = {
            "contract_sha256": self.contract_hash,
            "canary": json.loads((self.canary_dir / "summary.json").read_text(encoding="utf-8"))["status"] if (self.canary_dir / "summary.json").exists() else "not_run",
            "checkpoints": {
                epoch_label(epoch): self.checkpoint(epoch).exists()
                for epoch in self.epochs
            },
            "full_only_adjudicated": (self.full_dir / "adjudication.json").exists(),
            "state": json.loads(self.state.read_text(encoding="utf-8")) if self.state.exists() else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "status", "canary", "formal"), required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--acknowledge")
    args = parser.parse_args()
    sweep = EpochSweep(args.contract)
    if args.mode == "plan":
        sweep.plan()
    elif args.mode == "status":
        sweep.status()
    elif args.mode == "canary":
        sweep.canary(args.acknowledge)
    else:
        sweep.formal(args.acknowledge)


if __name__ == "__main__":
    main()
