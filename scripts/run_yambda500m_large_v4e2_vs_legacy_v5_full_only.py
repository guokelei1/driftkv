#!/usr/bin/env python3
"""Run the frozen Large v4@2.0 versus legacy-v5 Full-only comparison."""

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

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_large_v4e2_vs_legacy_v5_full_only_v1.yaml"
SCOPE = ROOT / "configs/contracts/yambda500m_large_v4e2_vs_legacy_v5_e14_only_v1.yaml"
CANARY_ACK = "RUN_LARGE_V4E2_VS_LEGACY_V5_CANARY"
FORMAL_ACK = "RUN_LARGE_V4E2_VS_LEGACY_V5_FULL_ONLY"


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


class FullComparison:
    def __init__(self, contract_path: Path = CONTRACT, scope_path: Path = SCOPE) -> None:
        self.contract_path = contract_path.resolve()
        self.contract = yaml.safe_load(self.contract_path.read_text(encoding="utf-8"))
        self.contract_hash = sha256_file(self.contract_path)
        self.scope_path = scope_path.resolve()
        self.scope = yaml.safe_load(self.scope_path.read_text(encoding="utf-8"))
        self.scope_hash = sha256_file(self.scope_path)
        self.output = (ROOT / self.contract["outputs"]["root"]).resolve()
        self.canary_dir = (ROOT / self.contract["outputs"]["canary"]).resolve()
        self.logs = self.output / "logs"
        self.state = self.output / "state.json"
        self.manifest = (ROOT / self.contract["frozen_inputs"]["manifest"]).resolve().parent
        self.dataset = (ROOT / self.contract["frozen_inputs"]["dataset_manifest"]).resolve()
        self.reference = (ROOT / self.contract["models"]["reference_checkpoint"]).resolve()
        self.comparison = (ROOT / self.contract["models"]["comparison_checkpoint"]).resolve()
        self.gpus = list(map(int, self.contract["resource_plan"]["physical_gpus"]))
        self.world = int(self.contract["resource_plan"]["world_size"])
        self._validate_contract()
        self._validate_scope()

    def _validate_contract(self) -> None:
        if self.gpus != [0, 1, 2, 3] or self.world != 4:
            raise RuntimeError("comparison is frozen to one four-rank GPU0/1/2/3 job")
        horizons = self.contract["evaluation"]["horizons"]
        if horizons["E7"]["days_half_open"] != [287, 294]:
            raise RuntimeError("complete E7 window drifted")
        if horizons["E14_partial"]["days_half_open"] != [287, 301]:
            raise RuntimeError("E14_partial window drifted")
        for section, keys in (
            ("models", (
                "reference_checkpoint", "comparison_checkpoint", "comparison_checkpoint_seal",
            )),
            ("frozen_inputs", (
                "base_large_contract", "unified_scale_contract", "dataset_manifest",
                "item_mapping", "manifest", "requests_quality", "requests_fidelity",
            )),
            ("evidence_boundary", ("v4_epoch_sweep_summary",)),
        ):
            values = self.contract[section]
            for key in keys:
                path = (ROOT / values[key]).resolve()
                if not path.exists() or sha256_file(path) != values[f"{key}_sha256"]:
                    raise RuntimeError(f"frozen comparison input mismatch: {section}.{key}")

    def _validate_scope(self) -> None:
        parent = self.scope["frozen_parent"]
        if parent["base_contract_sha256"] != self.contract_hash:
            raise RuntimeError("E14-only scope does not bind the comparison contract")
        canary = (ROOT / parent["passing_e14_partial_canary"]).resolve()
        if not canary.exists() or sha256_file(canary) != parent["passing_e14_partial_canary_sha256"]:
            raise RuntimeError("E14-only scope does not bind the passing canary")
        change = self.scope["scope_change"]
        if change["formal_horizons"] != ["E14_partial"] or change["E7_formal"] != "excluded_before_raw_or_label_read":
            raise RuntimeError("formal comparison scope must be E14_partial only")

    @property
    def formal_horizons(self) -> tuple[str, ...]:
        return tuple(self.scope["scope_change"]["formal_horizons"])

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

    def horizon_dir(self, name: str) -> Path:
        return (ROOT / self.contract["outputs"][name]).resolve()

    def preflight(self) -> None:
        output = subprocess.check_output([
            "nvidia-smi", f"--id={','.join(map(str, self.gpus))}",
            "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",
        ], text=True)
        rows = [tuple(int(part.strip()) for part in line.split(",")) for line in output.splitlines()]
        if [row[0] for row in rows] != self.gpus or any(row[1] < 42_000 for row in rows):
            raise RuntimeError(f"comparison GPUs are not free enough: {rows}")
        free_gib = shutil.disk_usage(ROOT).free / 2**30
        if free_gib < float(self.contract["resource_plan"]["minimum_free_workspace_gib"]):
            raise RuntimeError(f"comparison has only {free_gib:.1f} GiB free workspace")

    def run(self, name: str, command: list[str], *, gpu: bool) -> dict:
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{name}.log"
        runtime_path = self.logs / f"{name}.runtime.json"
        if log_path.exists() or runtime_path.exists():
            raise RuntimeError(f"existing comparison step log requires audit: {name}")
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=self.gpu_env if gpu else {
                    **os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1",
                }, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line); log.write(line); log.flush()
            returncode = process.wait()
        runtime = {
            "name": name, "returncode": returncode,
            "elapsed_seconds": time.perf_counter() - started, "command": command,
        }
        atomic_json(runtime_path, runtime)
        if returncode:
            raise RuntimeError(f"comparison step failed ({returncode}): {name}")
        return runtime

    def eval_command(self, name: str, output: Path, *, canary: bool) -> list[str]:
        start, end = self.contract["evaluation"]["horizons"][name]["days_half_open"]
        command = [
            "torchrun", "--standalone", f"--nproc_per_node={self.world}",
            "scripts/evaluate_yambda500m_release_candidates_raw.py",
            "--stage", f"large_v4e2_vs_legacy_v5_{name}{'_canary' if canary else ''}",
            "--block", "matrix_horizon", "--training-block", "matrix_horizon",
            "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
            "--parent", f"v4_e2={self.reference}",
            "--current", f"v5_legacy={self.comparison}",
            "--start-day", str(start), "--end-day", str(end),
            "--training-start-day", "273", "--training-end-day", "287",
            "--batch-size", str(self.contract["resource_plan"]["batch_size_per_rank"]),
            "--output", str(output), *self.cpu_args,
        ]
        if canary:
            command.extend([
                "--max-users", str(self.contract["resource_plan"]["focused_canary_users_per_rank"]),
            ])
        return command

    def write_state(self, status: str) -> None:
        atomic_json(self.state, {
            "status": status,
            "contract": str(self.contract_path.relative_to(ROOT)),
            "contract_sha256": self.contract_hash,
            "scope_contract": str(self.scope_path.relative_to(ROOT)),
            "scope_contract_sha256": self.scope_hash,
        })

    def canary(self, acknowledgement: str | None) -> None:
        if acknowledgement != CANARY_ACK:
            raise RuntimeError(f"canary requires --acknowledge {CANARY_ACK}")
        self.preflight()
        if self.canary_dir.exists():
            raise RuntimeError(f"existing comparison canary requires audit: {self.canary_dir}")
        raw_dir = self.canary_dir / "E14_partial"
        self.run("canary_raw", self.eval_command("E14_partial", raw_dir, canary=True), gpu=True)
        seal = json.loads((raw_dir / "raw.seal.json").read_text(encoding="utf-8"))
        peaks = seal["execution_runtime"]["peak_memory_by_rank"]
        peak = max(float(value["peak_reserved_mib"]) for value in peaks)
        limit = float(self.contract["resource_plan"]["focused_canary_peak_reserved_mib_limit"])
        checks = {
            "raw_sealed": seal["status"] == "release_candidate_full_only_raw_sealed_before_label_join",
            "two_models": seal["parent"] == "v4_e2" and seal["currents"] == ["v5_legacy"],
            "no_reuse": not seal["contains_reuse"],
            "world_size_four": seal["execution_runtime"]["world_size"] == self.world,
            "peak_under_limit": peak < limit,
        }
        payload = {
            "status": "large_v4e2_vs_legacy_v5_canary_passed" if all(checks.values()) else "large_v4e2_vs_legacy_v5_canary_failed",
            "contract_sha256": self.contract_hash, "checks": checks,
            "peak_reserved_mib": peak, "peak_reserved_mib_limit": limit,
            "raw_sha256": seal["raw_sha256"],
        }
        atomic_json(self.canary_dir / "summary.json", payload)
        self.write_state(payload["status"])
        if not all(checks.values()):
            raise RuntimeError(f"comparison canary failed: {checks}")

    def validate_raw(self, name: str) -> tuple[Path, Path]:
        directory = self.horizon_dir(name)
        raw, seal_path = directory / "raw.parquet", directory / "raw.seal.json"
        if not raw.exists() or not seal_path.exists():
            raise RuntimeError(f"comparison raw is incomplete: {name}")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal["raw_sha256"] != sha256_file(raw) or seal["contains_reuse"]:
            raise RuntimeError(f"comparison raw seal mismatch: {name}")
        expected = int(self.contract["evaluation"]["horizons"][name]["expected_known_requests"])
        if int(seal["rows"]) != expected * 2:
            raise RuntimeError(f"comparison row conservation failed: {name}")
        return raw, seal_path

    def summarize(self) -> None:
        rows = []
        for name in self.formal_horizons:
            report = json.loads((self.horizon_dir(name) / "adjudication.json").read_text(encoding="utf-8"))
            reference = report["parent_absolute"]["hstu_native"]
            candidate = report["candidates"]["v5_legacy"]
            comparison = candidate["absolute"]["hstu_native"]
            paired = candidate["paired_release_gain"]["parent_minus_current_log_loss"]
            gates = {
                "AUC_positive": comparison["ROC_AUC"] > reference["ROC_AUC"],
                "loss_positive": reference["log_loss"] > comparison["log_loss"],
                "Brier_not_worse": comparison["Brier"] <= reference["Brier"],
                "bootstrap_lower_positive": paired["user_cluster_bootstrap_95CI"]["p2_5"] > 0.0,
            }
            rows.append({
                "horizon": name,
                "completeness": self.contract["evaluation"]["horizons"][name]["completeness"],
                "v5_vs_v4e2_AUC_relative_percent": 100 * (comparison["ROC_AUC"] - reference["ROC_AUC"]) / reference["ROC_AUC"],
                "v5_vs_v4e2_loss_reduction_percent": 100 * (reference["log_loss"] - comparison["log_loss"]) / reference["log_loss"],
                "v5_vs_v4e2_Brier_reduction_percent": 100 * (reference["Brier"] - comparison["Brier"]) / reference["Brier"],
                "gates": gates, "all_gates_pass": all(gates.values()),
                "raw_sha256": report["raw_sha256"],
            })
        payload = {
            "status": "large_v4e2_vs_legacy_v5_E14_partial_full_only_complete",
            "contract_sha256": self.contract_hash,
            "scope_contract_sha256": self.scope_hash,
            "lineage_warning": "legacy v5 was trained from original v4@1.0, not reference v4@2.0",
            "evidence_boundary": self.contract["evidence_boundary"]["interpretation"],
            "rows": rows,
        }
        atomic_json(ROOT / self.contract["outputs"]["summary_json"], payload)
        lines = [
            "# Large V4@2.0 vs legacy V5 Full-only", "",
            "Legacy V5 was trained from the original V4@1.0, not from this V4@2.0; this is a post-hoc head-to-head comparison.", "",
            "| Horizon | Completeness | V5 vs V4@2 AUC | Loss reduction | Brier reduction | Strict gate |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                f"| {row['horizon']} | {row['completeness']} | "
                f"{row['v5_vs_v4e2_AUC_relative_percent']:+.3f}% | "
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
            raise RuntimeError("formal comparison requires the focused canary first")
        canary = json.loads(canary_path.read_text(encoding="utf-8"))
        if canary.get("status") != "large_v4e2_vs_legacy_v5_canary_passed" or canary.get("contract_sha256") != self.contract_hash:
            raise RuntimeError("formal comparison lacks a passing current-contract canary")
        self.preflight()
        # Seal every scope-authorized label-free population before metrics are read.
        for name in self.formal_horizons:
            directory = self.horizon_dir(name)
            if not directory.exists():
                self.run(f"formal_{name}_raw", self.eval_command(name, directory, canary=False), gpu=True)
            self.validate_raw(name)
        self.write_state("both_label_free_raw_populations_sealed")
        for name in self.formal_horizons:
            directory = self.horizon_dir(name)
            raw, seal = self.validate_raw(name)
            report = directory / "adjudication.json"
            if not report.exists():
                self.run(f"formal_{name}_adjudicate", [
                    sys.executable, "scripts/adjudicate_yambda500m_release_candidates.py",
                    "--raw", str(raw), "--seal", str(seal),
                    "--labels", str(self.manifest / "requests_quality.parquet"),
                    "--output", str(report),
                ], gpu=False)
            if json.loads(report.read_text(encoding="utf-8"))["raw_sha256"] != sha256_file(raw):
                raise RuntimeError(f"comparison adjudication/raw mismatch: {name}")
        self.summarize()
        self.write_state("large_v4e2_vs_legacy_v5_E14_partial_full_only_complete")

    def status(self) -> None:
        print(json.dumps({
            "contract_sha256": self.contract_hash,
            "scope_contract_sha256": self.scope_hash,
            "formal_horizons": list(self.formal_horizons),
            "canary": json.loads((self.canary_dir / "summary.json").read_text(encoding="utf-8"))["status"] if (self.canary_dir / "summary.json").exists() else "not_run",
            "E7_raw": (self.horizon_dir("E7") / "raw.seal.json").exists(),
            "E7_adjudicated": (self.horizon_dir("E7") / "adjudication.json").exists(),
            "E14_partial_raw": (self.horizon_dir("E14_partial") / "raw.seal.json").exists(),
            "E14_partial_adjudicated": (self.horizon_dir("E14_partial") / "adjudication.json").exists(),
            "state": json.loads(self.state.read_text(encoding="utf-8")) if self.state.exists() else None,
        }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("status", "canary", "formal"), required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--scope-contract", type=Path, default=SCOPE)
    parser.add_argument("--acknowledge")
    args = parser.parse_args()
    comparison = FullComparison(args.contract, args.scope_contract)
    if args.mode == "status":
        comparison.status()
    elif args.mode == "canary":
        comparison.canary(args.acknowledge)
    else:
        comparison.formal(args.acknowledge)


if __name__ == "__main__":
    main()
