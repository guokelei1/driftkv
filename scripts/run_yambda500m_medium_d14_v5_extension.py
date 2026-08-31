#!/usr/bin/env python3
"""Run the separately contracted Medium D14 v4->v5 extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_v5_extension_v1.yaml"
EXECUTION = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_v5_execution_v1.yaml"
ACK = "RUN_MEDIUM_D14_V5_EXTENSION"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Runner:
    def __init__(self, contract_path: Path, execution_path: Path, *, threads: int) -> None:
        self.contract_path = contract_path.resolve()
        self.execution_path = execution_path.resolve()
        self.contract = yaml.safe_load(self.contract_path.read_text(encoding="utf-8"))
        self.execution = yaml.safe_load(self.execution_path.read_text(encoding="utf-8"))
        self.contract_hash = sha256(self.contract_path)
        self.execution_hash = sha256(self.execution_path)
        self.threads = int(threads)
        self.output = (ROOT / self.contract["outputs"]["root"]).resolve()
        self.logs = self.output / "logs"
        self.log_jsonl = self.logs / "pipeline.jsonl"
        self.state_path = self.output / "state.json"
        self.manifest = (ROOT / self.contract["manifest"]["output"]).resolve()
        self.dataset = (ROOT / self.contract["frozen_inputs"]["dataset_manifest"]).resolve()
        self.parent = (ROOT / self.contract["frozen_parent"]["v4_checkpoint"]).resolve()
        self.checkpoint_dir = self.output / "checkpoint"
        self.checkpoint = self.checkpoint_dir / "checkpoint_100.pt"
        self.world = int(self.execution["execution_amendment"]["world_size"])
        self.gpus = list(map(int, self.execution["execution_amendment"]["physical_gpus"]))
        self.global_batch = int(self.execution["execution_amendment"]["global_train_batch_size"])
        self._validate_contracts()

    def _validate_contracts(self) -> None:
        frozen = self.contract["frozen_inputs"]
        for key in ("unified_scale_contract", "dataset_manifest", "item_mapping"):
            if sha256(ROOT / frozen[key]) != frozen[f"{key}_sha256"]:
                raise RuntimeError(f"D14 v5 frozen input mismatch: {key}")
        parent = self.contract["frozen_parent"]
        for key in ("matrix_contract", "original_execution_contract", "v4_checkpoint", "v4_checkpoint_seal"):
            if sha256(ROOT / parent[key]) != parent[f"{key}_sha256"]:
                raise RuntimeError(f"D14 v5 frozen parent mismatch: {key}")
        execution_parent = self.execution["frozen_parent"]
        if execution_parent["contract_sha256"] != self.contract_hash:
            raise RuntimeError("D14 v5 execution contract does not bind launch contract")
        if sha256(ROOT / execution_parent["contract"]) != self.contract_hash:
            raise RuntimeError("D14 v5 execution parent path mismatch")
        if sha256(ROOT / execution_parent["proven_reuse_runtime_contract"]) != execution_parent["proven_reuse_runtime_contract_sha256"]:
            raise RuntimeError("D14 v5 proven Reuse runtime parent mismatch")
        if self.world != 4 or self.gpus != [0, 1, 2, 3]:
            raise RuntimeError("D14 v5 extension requires one four-rank GPU0/1/2/3 job")
        if self.global_batch != 32 or self.execution["execution_amendment"]["local_batch_sizes_by_rank"] != [8, 8, 8, 8]:
            raise RuntimeError("D14 v5 global train batch must remain 32 as 8/8/8/8")
        affinity = self.execution["cpu_affinity"]
        cpu_sets = [set(map(int, affinity[f"rank{rank}"])) for rank in range(self.world)]
        if any(len(values) != 14 for values in cpu_sets) or len(set().union(*cpu_sets)) != 56:
            raise RuntimeError("D14 v5 evaluation requires 56 disjoint physical CPU cores")
        d7_state = ROOT / "results/yambda500m_medium_seed17/full_reuse_matrix_v1/D7/forced_reuse_diagnostic_v1/state.json"
        if not d7_state.exists() or json.loads(d7_state.read_text()).get("status") != "forced_d7_reuse_complete":
            raise RuntimeError("D14 v5 cannot start until the serial D7 four-GPU queue is complete")

    @property
    def distributed_prefix(self) -> list[str]:
        return ["torchrun", "--standalone", f"--nproc_per_node={self.world}"]

    @property
    def gpu_env(self) -> dict[str, str]:
        omp = str(self.execution["full_only_runtime"]["omp_num_threads_per_rank"])
        return {
            **os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus)),
            "OMP_NUM_THREADS": omp, "PYTHONUNBUFFERED": "1",
        }

    def cpu_args(self, section: str) -> list[str]:
        runtime = self.execution[section]
        affinity = ";".join(
            ",".join(map(str, self.execution["cpu_affinity"][f"rank{rank}"]))
            for rank in range(self.world)
        )
        return [
            "--history-threads", str(runtime["history_threads_per_rank"]),
            "--arrow-cpu-threads", str(runtime["arrow_cpu_threads_per_rank"]),
            "--arrow-io-threads", str(runtime["arrow_io_threads_per_rank"]),
            "--torch-cpu-threads", str(runtime["torch_cpu_threads_per_rank"]),
            "--cpu-affinity-by-rank", affinity,
        ]

    def event(self, event: str, **values: object) -> None:
        self.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.log_jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": utc_now(), "event": event, **values}, sort_keys=True) + "\n")

    def state(self, status: str, **values: object) -> None:
        atomic_json(self.state_path, {
            "status": status, "updated_at_utc": utc_now(),
            "contract": str(self.contract_path.relative_to(ROOT)), "contract_sha256": self.contract_hash,
            "execution_contract": str(self.execution_path.relative_to(ROOT)),
            "execution_contract_sha256": self.execution_hash,
            "world_size": self.world, "physical_gpus": self.gpus, **values,
        })

    def run(self, name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        log = self.logs / f"{name}.log"
        retry = 1
        while log.exists():
            log = self.logs / f"{name}.retry{retry}.log"; retry += 1
        self.event("step_start", name=name, command=command, log=str(log))
        print("+", " ".join(command), flush=True)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line); stream.write(line)
            code = process.wait()
        self.event("step_end", name=name, returncode=code)
        if code:
            self.state("failed", failed_step=name, returncode=code)
            raise subprocess.CalledProcessError(code, command)

    def preflight(self) -> None:
        if shutil.disk_usage(ROOT).free / 2**30 < 30:
            raise RuntimeError("D14 v5 extension requires at least 30 GiB free workspace")
        rows = subprocess.check_output([
            "nvidia-smi", f"--id={','.join(map(str, self.gpus))}",
            "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()
        values = [tuple(map(int, (part.strip() for part in row.split(",")))) for row in rows]
        values.sort()
        if [row[0] for row in values] != self.gpus or any(row[1] < 40000 for row in values):
            raise RuntimeError(f"D14 v5 four-GPU preflight failed: {values}")
        self.event("preflight_pass", gpus=values)

    def validate_manifest(self) -> bool:
        descriptor = self.manifest / "manifest.json"
        if not descriptor.exists():
            if self.manifest.exists():
                raise RuntimeError(f"partial D14 v5 manifest requires audit: {self.manifest}")
            return False
        payload = json.loads(descriptor.read_text())
        if payload.get("contract_sha256") != self.contract_hash:
            raise RuntimeError("D14 v5 manifest belongs to another contract")
        if payload.get("window_days_half_open") != [217, 301]:
            raise RuntimeError("D14 v5 manifest does not include the frozen partial tail")
        for name, artifact in payload["artifacts"].items():
            path = self.manifest / name
            if not path.exists() or sha256(path) != artifact["sha256"]:
                raise RuntimeError(f"D14 v5 manifest artifact mismatch: {name}")
        return True

    def prepare(self) -> None:
        if self.validate_manifest():
            self.event("manifest_skip_valid")
            return
        self.run("prepare_manifest", [
            sys.executable, "scripts/build_yambda500m_hstu_native_matrix_manifest.py",
            "--contract", str(self.contract_path), "--output", str(self.manifest),
            "--threads", str(self.threads),
        ], env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"})
        if not self.validate_manifest():
            raise RuntimeError("D14 v5 manifest builder returned invalid output")

    def training_dir(self, *, canary: bool) -> Path:
        return (ROOT / self.execution["canary"]["training_output"]).resolve() if canary else self.checkpoint_dir

    def training_valid(self, *, canary: bool) -> bool:
        directory = self.training_dir(canary=canary)
        checkpoint, result, seal = directory / "checkpoint_100.pt", directory / "train_result.json", directory / "checkpoint.seal.json"
        existing = [path.exists() for path in (checkpoint, result, seal)]
        if not any(existing):
            if directory.exists():
                raise RuntimeError(f"partial D14 v5 training directory requires audit: {directory}")
            return False
        if not all(existing):
            raise RuntimeError(f"partial D14 v5 training artifacts require audit: {directory}")
        result_payload = json.loads(result.read_text())
        seal_payload = json.loads(seal.read_text())
        expected = {
            "contract_sha256": self.contract_hash,
            "execution_contract_sha256": self.execution_hash,
            "checkpoint_sha256": sha256(checkpoint),
            "parent_checkpoint_sha256": sha256(self.parent),
            "canary": canary,
        }
        if any(seal_payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"D14 v5 checkpoint seal mismatch: {directory}")
        if result_payload.get("contract_sha256") != self.contract_hash or result_payload.get("execution_contract_sha256") != self.execution_hash:
            raise RuntimeError(f"D14 v5 train result contract mismatch: {directory}")
        return True

    def train(self, *, canary: bool) -> Path:
        directory = self.training_dir(canary=canary)
        if self.training_valid(canary=canary):
            self.event("training_skip_valid", canary=canary)
            return directory / "checkpoint_100.pt"
        command = [
            *self.distributed_prefix, "scripts/train_yambda500m_foundation_fsdp.py",
            "--version", "v5", "--branch", "D14", "--launch-contract", str(self.contract_path),
            "--execution-contract", str(self.execution_path), "--manifest-dir", str(self.manifest),
            "--training-block", "matrix_horizon", "--output", str(directory),
            "--parent", str(self.parent), "--oov-buckets", "256", "--passes", "1",
            "--global-batch-size", str(self.global_batch), "--train-start-day", "273", "--train-end-day", "287",
        ]
        if canary:
            command.extend(["--canary-steps", str(self.execution["execution_amendment"]["focused_correctness_canary"]["training_steps"])])
        self.run(f"{'canary_' if canary else 'formal_'}train_D14_v5", command, env=self.gpu_env)
        checkpoint = directory / "checkpoint_100.pt"
        result = json.loads((directory / "train_result.json").read_text())
        atomic_json(directory / "checkpoint.seal.json", {
            "status": "medium_D14_v5_extension_checkpoint_sealed",
            "contract_sha256": self.contract_hash, "execution_contract_sha256": self.execution_hash,
            "checkpoint_sha256": sha256(checkpoint), "parent_checkpoint_sha256": sha256(self.parent),
            "world_size": self.world, "physical_gpus": self.gpus,
            "global_batch_size": self.global_batch, "steps": result["steps"], "canary": canary,
        })
        if not self.training_valid(canary=canary):
            raise RuntimeError("D14 v5 training did not seal correctly")
        return checkpoint

    @staticmethod
    def window(name: str) -> tuple[int, int, bool]:
        values = {"E3": (287, 290, False), "E7": (287, 294, False), "E14_partial": (287, 301, True)}
        return values[name]

    def full_dir(self, name: str, *, canary: bool) -> Path:
        prefix = self.output / "canary" if canary else self.output
        return prefix / "full_only" / name / "v4_to_v5"

    def reuse_dir(self, name: str, *, canary: bool) -> Path:
        prefix = self.output / "canary" if canary else self.output
        return prefix / "reuse" / name / "v4_to_v5"

    def evaluate_full(self, name: str, current: Path, *, canary: bool) -> None:
        start, end, partial = self.window(name)
        directory = self.full_dir(name, canary=canary)
        raw, seal, report = directory / "raw.parquet", directory / "raw.seal.json", directory / "adjudication.json"
        cell_seal = directory / "extension.seal.json"
        if report.exists():
            payload = json.loads(cell_seal.read_text()) if cell_seal.exists() else {}
            if payload.get("adjudication_sha256") != sha256(report) or payload.get("contract_sha256") != self.contract_hash:
                raise RuntimeError(f"D14 v5 Full cell seal mismatch: {directory}")
            return
        if directory.exists() and not (raw.exists() and seal.exists()):
            raise RuntimeError(f"partial D14 v5 Full directory requires audit: {directory}")
        if not directory.exists():
            runtime = self.execution["full_only_runtime"]
            command = [
                *self.distributed_prefix, "scripts/evaluate_yambda500m_release_candidates_raw.py",
                "--stage", f"medium_D14_v4_to_v5_{name}_full_only{'_canary' if canary else ''}",
                "--block", "matrix_horizon", "--training-block", "matrix_horizon",
                "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
                "--parent", f"v4={self.parent}", "--current", f"v5={current}",
                "--start-day", str(start), "--end-day", str(end),
                "--training-start-day", "273", "--training-end-day", "287",
                "--output", str(directory), "--batch-size", str(runtime["batch_size_per_rank"]),
                *self.cpu_args("full_only_runtime"),
            ]
            if canary:
                command.extend(["--max-users", str(self.execution["canary"]["maximum_users_per_rank"]), "--allow-canary-checkpoints"])
            self.run(f"{'canary_' if canary else 'formal_'}full_D14_v5_{name}_raw", command, env=self.gpu_env)
        if sha256(raw) != json.loads(seal.read_text())["raw_sha256"]:
            raise RuntimeError(f"D14 v5 Full raw differs from seal: {directory}")
        if canary:
            return
        self.run(f"formal_full_D14_v5_{name}_adjudicate", [
            sys.executable, "scripts/adjudicate_yambda500m_release_candidates.py",
            "--raw", str(raw), "--seal", str(seal),
            "--labels", str(self.manifest / "requests_quality.parquet"), "--output", str(report),
        ], env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"})
        atomic_json(cell_seal, {
            "status": "medium_D14_v5_full_only_cell_sealed", "contract_sha256": self.contract_hash,
            "execution_contract_sha256": self.execution_hash, "window": name,
            "partial_tail_diagnostic": partial, "raw_sha256": sha256(raw),
            "raw_seal_sha256": sha256(seal), "adjudication_sha256": sha256(report),
            "serving_admission": False,
        })

    def evaluate_reuse(self, name: str, current: Path, *, canary: bool) -> None:
        start, end, partial = self.window(name)
        directory = self.reuse_dir(name, canary=canary)
        raw, seal, report = directory / "raw.parquet", directory / "raw.seal.json", directory / "adjudication.json"
        cell_seal = directory / "extension.seal.json"
        if report.exists():
            payload = json.loads(cell_seal.read_text()) if cell_seal.exists() else {}
            if payload.get("adjudication_sha256") != sha256(report) or payload.get("contract_sha256") != self.contract_hash:
                raise RuntimeError(f"D14 v5 Reuse cell seal mismatch: {directory}")
            return
        if directory.exists() and not (raw.exists() and seal.exists()):
            raise RuntimeError(f"partial D14 v5 Reuse directory requires audit: {directory}")
        if not directory.exists():
            runtime = self.execution["reuse_runtime"]
            command = [
                *self.distributed_prefix, "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
                "--stage", f"medium_D14_v4_to_v5_{name}_reuse{'_canary' if canary else ''}",
                "--edge", "v4_to_v5", "--cutover-day", "287", "--start-day", str(start), "--end-day", str(end),
                "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
                "--parent", str(self.parent), "--current", str(current), "--output", str(directory),
                "--include-parent-exact", "--cohort-size", str(runtime["cohort_size_per_rank"]),
                "--query-chunk-size", str(runtime["query_chunk_size_per_rank"]),
                *self.cpu_args("reuse_runtime"),
            ]
            if canary:
                command.extend(["--max-users", str(self.execution["canary"]["maximum_users_per_rank"]), "--allow-canary-checkpoints"])
            self.run(f"{'canary_' if canary else 'formal_'}reuse_D14_v5_{name}_raw", command, env=self.gpu_env)
        seal_payload = json.loads(seal.read_text())
        if sha256(raw) != seal_payload["raw_sha256"] or seal_payload["rows"] != 3 * seal_payload["requests"]:
            raise RuntimeError(f"D14 v5 Reuse raw differs from seal: {directory}")
        if canary:
            return
        self.run(f"formal_reuse_D14_v5_{name}_adjudicate", [
            sys.executable, "scripts/adjudicate_yambda500m_hstu_native_onehop_reuse.py",
            "--raw", str(raw), "--seal", str(seal),
            "--labels", str(self.manifest / "requests_quality.parquet"), "--output", str(report),
        ], env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"})
        atomic_json(cell_seal, {
            "status": "medium_D14_v5_adjacent_reuse_cell_sealed", "contract_sha256": self.contract_hash,
            "execution_contract_sha256": self.execution_hash, "window": name,
            "partial_tail_diagnostic": partial, "raw_sha256": sha256(raw),
            "raw_seal_sha256": sha256(seal), "adjudication_sha256": sha256(report),
            "serving_admission": False,
        })

    def canary(self) -> None:
        marker = self.output / "canary" / "canary.pass.json"
        if marker.exists():
            payload = json.loads(marker.read_text())
            if payload.get("contract_sha256") != self.contract_hash or payload.get("execution_contract_sha256") != self.execution_hash:
                raise RuntimeError("D14 v5 canary marker contract mismatch")
            self.event("canary_skip_valid")
            return
        current = self.train(canary=True)
        name = str(self.execution["canary"]["full_only_window"])
        self.evaluate_full(name, current, canary=True)
        self.evaluate_reuse(name, current, canary=True)
        full_seal = json.loads((self.full_dir(name, canary=True) / "raw.seal.json").read_text())
        reuse_seal = json.loads((self.reuse_dir(name, canary=True) / "raw.seal.json").read_text())
        peaks = [
            *full_seal.get("execution_runtime", {}).get("peak_memory_by_rank", []),
            *reuse_seal.get("execution_runtime", {}).get("peak_memory_by_rank", []),
        ]
        limit = float(self.execution["canary"]["maximum_peak_reserved_mib_per_rank"])
        checks = {
            "full_world_size": full_seal.get("execution_runtime", {}).get("world_size") == self.world,
            "full_batch_size": full_seal.get("execution_runtime", {}).get("batch_size_per_rank") == self.execution["full_only_runtime"]["batch_size_per_rank"],
            "reuse_world_size": reuse_seal.get("execution_runtime", {}).get("world_size") == self.world,
            "reuse_three_paths": reuse_seal.get("rows") == 3 * reuse_seal.get("requests", -1),
            "eight_peak_records": len(peaks) == 2 * self.world,
            "peaks_below_limit": len(peaks) == 2 * self.world and all(float(value["peak_reserved_mib"]) < limit for value in peaks),
        }
        if not all(checks.values()):
            raise RuntimeError(f"D14 v5 canary failed: {checks}")
        atomic_json(marker, {
            "status": "medium_D14_v5_four_gpu_canary_passed",
            "contract_sha256": self.contract_hash, "execution_contract_sha256": self.execution_hash,
            "training_checkpoint_sha256": sha256(current),
            "full_raw_sha256": full_seal["raw_sha256"], "reuse_raw_sha256": reuse_seal["raw_sha256"],
            "peak_memory_records": peaks, "checks": checks, "quality_read": False,
        })
        self.event("canary_pass", checks=checks)

    def summarize(self, *, require_complete: bool) -> None:
        rows = []
        for name in ("E3", "E7", "E14_partial"):
            full_path = self.full_dir(name, canary=False) / "adjudication.json"
            reuse_path = self.reuse_dir(name, canary=False) / "adjudication.json"
            if not (full_path.exists() and reuse_path.exists()):
                continue
            report = json.loads(reuse_path.read_text())
            three = report["three_path_summary"]
            old, new, reuse = three["old_parent"], three["new_current"], three["adjacent_one_hop_reuse"]
            rows.append({
                "window": name, "day_range": self.window(name)[:2],
                "partial_tail_diagnostic": self.window(name)[2], "requests": three["requests"],
                "new_relative_to_old_AUC_percent": 100 * (new["ROC_AUC"] - old["ROC_AUC"]) / old["ROC_AUC"],
                "reuse_relative_to_old_AUC_percent": 100 * (reuse["ROC_AUC"] - old["ROC_AUC"]) / old["ROC_AUC"],
                "reuse_AUC_gain_retained_percent": three["reuse_AUC_gain_retained_percent"],
                "old_parent": old, "new_current": new, "adjacent_one_hop_reuse": reuse,
                "full_adjudication_sha256": sha256(full_path), "reuse_adjudication_sha256": sha256(reuse_path),
            })
        complete = len(rows) == 3
        if require_complete and not complete:
            raise RuntimeError(f"D14 v5 extension incomplete: {len(rows)}/3")
        payload = {
            "status": "medium_D14_v5_extension_complete" if complete else "medium_D14_v5_extension_partial",
            "contract_sha256": self.contract_hash, "execution_contract_sha256": self.execution_hash,
            "checkpoint_sha256": sha256(self.checkpoint) if self.checkpoint.exists() else None,
            "complete_windows": ["E3", "E7"], "partial_tail_diagnostic_windows": ["E14_partial"],
            "serving_admission": False, "rows": rows,
        }
        atomic_json(self.output / "summary.json", payload)
        lines = [
            "# Medium D14 v4 → v5 extension", "",
            f"Status: **{payload['status']}**. E3/E7 are complete windows; E14_partial contains incomplete source day300 and is diagnostic only.", "",
            "| Window | Requests | New vs Old AUC | Reuse retained | Old loss | New loss | Reuse loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            recovery = row["reuse_AUC_gain_retained_percent"]
            recovery_text = "N/A" if recovery is None else f"{recovery:+.2f}%"
            lines.append(
                f"| {row['window']} | {row['requests']:,} | {row['new_relative_to_old_AUC_percent']:+.2f}% | "
                f"{recovery_text} | {row['old_parent']['log_loss']:.6f} | {row['new_current']['log_loss']:.6f} | "
                f"{row['adjacent_one_hop_reuse']['log_loss']:.6f} |"
            )
        (self.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def formal(self, acknowledgement: str | None) -> None:
        if acknowledgement != ACK:
            raise RuntimeError(f"D14 v5 extension requires --acknowledge-long-run {ACK}")
        self.prepare(); self.preflight(); self.state("canary_started")
        self.canary(); self.state("canary_passed")
        current = self.train(canary=False); self.state("v5_training_complete", checkpoint=str(current.relative_to(ROOT)))
        for name in ("E3", "E7", "E14_partial"):
            self.evaluate_full(name, current, canary=False)
            self.evaluate_reuse(name, current, canary=False)
            self.summarize(require_complete=False)
            self.state("evaluation_running", last_completed_window=name)
        self.summarize(require_complete=True)
        self.state("complete", summary=str((self.output / "summary.json").relative_to(ROOT)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "prepare", "canary", "formal", "summarize"), default="plan")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--execution-contract", type=Path, default=EXECUTION)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--acknowledge-long-run")
    args = parser.parse_args()
    runner = Runner(args.contract, args.execution_contract, threads=args.threads)
    if args.mode == "plan":
        print(json.dumps({
            "contract_sha256": runner.contract_hash, "execution_contract_sha256": runner.execution_hash,
            "training": "D14 v4_to_v5 [273,287)", "world_size": runner.world,
            "physical_gpus": runner.gpus, "global_train_batch_size": runner.global_batch,
            "full_only_batch_size_per_rank": runner.execution["full_only_runtime"]["batch_size_per_rank"],
            "reuse_cohort_size_per_rank": runner.execution["reuse_runtime"]["cohort_size_per_rank"],
            "windows": ["E3", "E7", "E14_partial"], "formal_acknowledgement": ACK,
        }, indent=2))
    elif args.mode == "prepare":
        runner.prepare()
    elif args.mode == "canary":
        runner.prepare(); runner.preflight(); runner.canary()
    elif args.mode == "summarize":
        runner.summarize(require_complete=False)
    else:
        runner.formal(args.acknowledge_long_run)


if __name__ == "__main__":
    main()
