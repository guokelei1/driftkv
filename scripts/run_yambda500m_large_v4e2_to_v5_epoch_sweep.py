#!/usr/bin/env python3
"""Run the frozen Large v4@2.0 -> v5 one/two-epoch Full-only sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from run_yambda500m_large_v3_v4_epoch_sweep import (
    EpochSweep,
    ROOT,
    atomic_json,
    atomic_text,
    epoch_label,
    sha256_file,
)


CONTRACT = ROOT / "configs/contracts/yambda500m_large_v4e2_to_v5_epoch_sweep_v1.yaml"
CANARY_ACK = "RUN_LARGE_V4E2_TO_V5_EPOCH_SWEEP_CANARY"
FORMAL_ACK = "RUN_LARGE_V4E2_TO_V5_EPOCH_SWEEP"


class V5EpochSweep(EpochSweep):
    def __init__(self, contract_path: Path = CONTRACT) -> None:
        super().__init__(contract_path)

    def _validate_contract(self) -> None:
        if self.epochs != (1.0, 2.0):
            raise RuntimeError("v5 epoch endpoints drifted")
        branch = self.contract["scope"]["branches"]["D14"]
        if branch["training_days_half_open"] != [273, 287]:
            raise RuntimeError("v5 training window drifted")
        if self.contract["evaluation"]["day_range_half_open"] != [287, 301]:
            raise RuntimeError("v5 E14_partial window drifted")
        if self.world != 4 or self.gpus != [0, 1, 2, 3]:
            raise RuntimeError("v5 epoch sweep is frozen to GPU0/1/2/3")
        for section, keys in (
            ("frozen_inputs", (
                "unified_scale_contract", "dataset_manifest", "item_mapping",
                "manifest", "requests_quality", "requests_fidelity",
            )),
            ("frozen_parent", ("v4_e2_checkpoint", "v4_e2_checkpoint_seal")),
            ("evidence_boundary", (
                "v4_epoch_sweep_summary", "legacy_v5_head_to_head_summary",
            )),
        ):
            values = self.contract[section]
            for key in keys:
                path = (ROOT / values[key]).resolve()
                if not path.exists() or sha256_file(path) != values[f"{key}_sha256"]:
                    raise RuntimeError(f"v5 sweep frozen input mismatch: {section}.{key}")

    def train_command(self, output: Path, *, canary: bool) -> list[str]:
        training = self.contract["training"]
        command = [
            *self.distributed, "scripts/train_yambda500m_foundation_fsdp.py",
            "--version", "v5", "--branch", "D14",
            "--launch-contract", str(self.contract_path),
            "--manifest-dir", str(self.manifest), "--training-block", "matrix_horizon",
            "--parent", str(self.parent), "--output", str(output),
            "--oov-buckets", str(self.contract["model"]["oov_buckets"]),
            "--passes", str(training["passes"]),
            "--checkpoint-epochs", ",".join(map(str, self.epochs)),
            "--global-batch-size", str(training["global_batch_size"]),
            "--train-start-day", "273", "--train-end-day", "287",
            "--progress-interval", "100", *self.cpu_args,
        ]
        if canary:
            command.extend(["--canary-steps", str(self.contract["resource_plan"]["focused_canary_steps"])])
        return command

    def eval_command(self, output: Path, currents: dict[str, Path], *, canary: bool) -> list[str]:
        command = [
            *self.distributed, "scripts/evaluate_yambda500m_release_candidates_raw.py",
            "--stage", "large_v4e2_to_v5_E14_partial_epoch_sweep_canary" if canary else "large_v4e2_to_v5_E14_partial_epoch_sweep",
            "--block", "matrix_horizon", "--training-block", "matrix_horizon",
            "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
            "--parent", f"v4_e2={self.parent}",
            "--start-day", "287", "--end-day", "301",
            "--training-start-day", "273", "--training-end-day", "287",
            "--batch-size", str(self.contract["resource_plan"]["full_eval_batch_size_per_rank"]),
            "--output", str(output), *self.cpu_args,
        ]
        for name, path in currents.items():
            command.extend(["--current", f"{name}={path}"])
        if canary:
            command.extend([
                "--max-users", str(self.contract["resource_plan"]["focused_canary_eval_users_per_rank"]),
                "--allow-canary-checkpoints",
            ])
        return command

    def canary(self, acknowledgement: str | None) -> None:
        if acknowledgement != CANARY_ACK:
            raise RuntimeError(f"canary requires --acknowledge {CANARY_ACK}")
        self.preflight()
        train_dir = self.canary_dir / "train"
        eval_dir = self.canary_dir / "full_only"
        if self.canary_dir.exists():
            raise RuntimeError(f"existing v5 sweep canary requires audit: {self.canary_dir}")
        self.run("canary_train", self.train_command(train_dir, canary=True), gpu=True)
        checkpoint = train_dir / "checkpoint_100.pt"
        aliases = {f"v5_e{epoch_label(epoch)}": checkpoint for epoch in self.epochs}
        self.run("canary_full_only_raw", self.eval_command(eval_dir, aliases, canary=True), gpu=True)
        train_result = json.loads((train_dir / "train_result.json").read_text(encoding="utf-8"))
        seal = json.loads((eval_dir / "raw.seal.json").read_text(encoding="utf-8"))
        limit = float(self.contract["resource_plan"]["focused_canary_peak_reserved_mib_limit"])
        train_peak = max(float(value["peak_reserved_mib"]) for value in train_result["rank_metrics"])
        eval_peak = max(float(value["peak_reserved_mib"]) for value in seal["execution_runtime"]["peak_memory_by_rank"])
        checks = {
            "finite_loss": bool(torch.isfinite(torch.tensor(train_result["mean_rank0_loss"]))),
            "world_size_four": train_result["world_size"] == self.world,
            "two_candidate_aliases": seal["currents"] == ["v5_e1p0", "v5_e2p0"],
            "raw_sealed": seal["status"] == "release_candidate_full_only_raw_sealed_before_label_join",
            "no_reuse": not seal["contains_reuse"],
            "training_peak_under_limit": train_peak < limit,
            "three_model_eval_peak_under_limit": eval_peak < limit,
        }
        payload = {
            "status": "large_v4e2_to_v5_epoch_sweep_canary_passed" if all(checks.values()) else "large_v4e2_to_v5_epoch_sweep_canary_failed",
            "contract_sha256": self.contract_hash, "checks": checks,
            "training_peak_reserved_mib": train_peak,
            "three_model_eval_peak_reserved_mib": eval_peak,
            "peak_reserved_mib_limit": limit,
            "checkpoint_sha256": sha256_file(checkpoint), "raw_sha256": seal["raw_sha256"],
        }
        atomic_json(self.canary_dir / "summary.json", payload)
        self.write_state(payload["status"])
        if not all(checks.values()):
            raise RuntimeError(f"v5 epoch-sweep canary failed: {checks}")

    def validate_formal_checkpoints(self) -> dict[str, dict]:
        result_path = self.checkpoint_dir / "train_result.json"
        paths = [self.checkpoint(epoch) for epoch in self.epochs]
        if not result_path.exists() or not all(path.exists() for path in paths):
            raise RuntimeError("v5 staged training artifacts are incomplete")
        metadata: dict[str, dict] = {}
        parent_hash = sha256_file(self.parent)
        for epoch, path in zip(self.epochs, paths, strict=True):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["status"] != "formal_staged_epoch_checkpoint":
                raise RuntimeError(f"unexpected v5 staged checkpoint status: {path}")
            if float(payload["training_epochs_completed"]) != epoch:
                raise RuntimeError(f"v5 staged epoch mismatch: {path}")
            if payload["version"] != "v5" or payload["parent_checkpoint_sha256"] != parent_hash:
                raise RuntimeError(f"v5 direct-parent lineage mismatch: {path}")
            metadata[epoch_label(epoch)] = {
                "epoch": epoch, "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "cache_producer_sha256": payload["cache_producer_sha256"],
            }
            del payload
        atomic_json(self.checkpoint_dir / "checkpoints.seal.json", {
            "status": "large_v4e2_to_v5_staged_checkpoints_sealed",
            "contract_sha256": self.contract_hash,
            "parent_checkpoint_sha256": parent_hash, "checkpoints": metadata,
        })
        return metadata

    def summarize(self, metadata: dict[str, dict]) -> None:
        report = json.loads((self.full_dir / "adjudication.json").read_text(encoding="utf-8"))
        parent = report["parent_absolute"]["hstu_native"]
        rows = []
        for epoch in self.epochs:
            name = f"v5_e{epoch_label(epoch)}"
            candidate = report["candidates"][name]
            current = candidate["absolute"]["hstu_native"]
            paired = candidate["paired_release_gain"]["parent_minus_current_log_loss"]
            gates = {
                "AUC_positive": current["ROC_AUC"] > parent["ROC_AUC"],
                "loss_positive": parent["log_loss"] > current["log_loss"],
                "Brier_not_worse": current["Brier"] <= parent["Brier"],
                "bootstrap_lower_positive": paired["user_cluster_bootstrap_95CI"]["p2_5"] > 0.0,
            }
            rows.append({
                "candidate": name, "training_epochs": epoch,
                "v5_vs_v4e2_AUC_relative_percent": 100 * (current["ROC_AUC"] - parent["ROC_AUC"]) / parent["ROC_AUC"],
                "v5_vs_v4e2_loss_reduction_percent": 100 * (parent["log_loss"] - current["log_loss"]) / parent["log_loss"],
                "v5_vs_v4e2_Brier_reduction_percent": 100 * (parent["Brier"] - current["Brier"]) / parent["Brier"],
                "gates": gates, "all_gates_pass": all(gates.values()),
                "checkpoint": metadata[epoch_label(epoch)],
            })
        payload = {
            "status": "large_v4e2_to_v5_epoch_sweep_E14_partial_complete",
            "contract_sha256": self.contract_hash,
            "completeness": self.contract["evaluation"]["completeness"],
            "evidence_boundary": self.contract["evidence_boundary"]["interpretation"],
            "raw_sha256": report["raw_sha256"], "rows": rows,
        }
        atomic_json(ROOT / self.contract["outputs"]["summary_json"], payload)
        lines = [
            "# Large V4@2.0 → V5 epoch sweep", "",
            "Status: **E14_partial Full-only complete**. Both V5 endpoints are direct children of V4@2.0.", "",
            "| V5 epoch | AUC vs V4@2 | Loss reduction | Brier reduction | Strict gate |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                f"| {row['training_epochs']:.1f} | {row['v5_vs_v4e2_AUC_relative_percent']:+.3f}% | "
                f"{row['v5_vs_v4e2_loss_reduction_percent']:+.3f}% | "
                f"{row['v5_vs_v4e2_Brier_reduction_percent']:+.3f}% | "
                f"{'PASS' if row['all_gates_pass'] else 'FAIL'} |"
            )
        atomic_text(ROOT / self.contract["outputs"]["summary_markdown"], "\n".join(lines) + "\n")

    def formal(self, acknowledgement: str | None) -> None:
        if acknowledgement != FORMAL_ACK:
            raise RuntimeError(f"formal run requires --acknowledge {FORMAL_ACK}")
        canary_path = self.canary_dir / "summary.json"
        if not canary_path.exists():
            raise RuntimeError("formal v5 epoch sweep requires the focused canary first")
        canary = json.loads(canary_path.read_text(encoding="utf-8"))
        if canary.get("status") != "large_v4e2_to_v5_epoch_sweep_canary_passed" or canary.get("contract_sha256") != self.contract_hash:
            raise RuntimeError("formal v5 epoch sweep lacks a passing current-contract canary")
        self.preflight()
        if not self.checkpoint_dir.exists():
            self.run("formal_train", self.train_command(self.checkpoint_dir, canary=False), gpu=True)
        metadata = self.validate_formal_checkpoints()
        raw, seal = self.full_dir / "raw.parquet", self.full_dir / "raw.seal.json"
        report = self.full_dir / "adjudication.json"
        if not report.exists():
            if self.full_dir.exists() and not (raw.exists() and seal.exists()):
                raise RuntimeError(f"partial v5 sweep Full directory requires audit: {self.full_dir}")
            if not self.full_dir.exists():
                currents = {f"v5_e{epoch_label(epoch)}": self.checkpoint(epoch) for epoch in self.epochs}
                self.run("formal_full_only_raw", self.eval_command(self.full_dir, currents, canary=False), gpu=True)
            self.run("formal_full_only_adjudicate", [
                sys.executable, "scripts/adjudicate_yambda500m_release_candidates.py",
                "--raw", str(raw), "--seal", str(seal),
                "--labels", str(self.manifest / "requests_quality.parquet"),
                "--output", str(report),
            ])
        if json.loads(report.read_text(encoding="utf-8"))["raw_sha256"] != sha256_file(raw):
            raise RuntimeError("v5 epoch sweep adjudication/raw mismatch")
        self.summarize(metadata)
        self.write_state("large_v4e2_to_v5_epoch_sweep_E14_partial_complete")

    def status(self) -> None:
        print(json.dumps({
            "contract_sha256": self.contract_hash,
            "canary": json.loads((self.canary_dir / "summary.json").read_text(encoding="utf-8"))["status"] if (self.canary_dir / "summary.json").exists() else "not_run",
            "checkpoints": {epoch_label(epoch): self.checkpoint(epoch).exists() for epoch in self.epochs},
            "E14_partial_raw": (self.full_dir / "raw.seal.json").exists(),
            "E14_partial_adjudicated": (self.full_dir / "adjudication.json").exists(),
            "state": json.loads(self.state.read_text(encoding="utf-8")) if self.state.exists() else None,
        }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("status", "canary", "formal"), required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--acknowledge")
    args = parser.parse_args()
    sweep = V5EpochSweep(args.contract)
    if args.mode == "status":
        sweep.status()
    elif args.mode == "canary":
        sweep.canary(args.acknowledge)
    else:
        sweep.formal(args.acknowledge)


if __name__ == "__main__":
    main()
