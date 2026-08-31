#!/usr/bin/env python3
"""Contract-driven Large D7/D14 training and prospective qualification.

The runner has four explicit phases:

* ``prepare`` builds and seals label-free request manifests on CPU;
* ``resource-canary`` tests the frozen 10L/H320 point without reading labels;
* ``formal`` runs the execution-sealed checkpoint, Full-only, admission, then
  adjacent Reuse/PRO matrix in that order;
* ``status`` reports resumable artifact counts without mutating the workspace.

Model/data logic remains in the existing trainer and evaluators.  This module
only validates contracts, queues one four-rank job at a time, monitors GPU
runtime, and seals orchestration artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from insight.pro_lazy_cost import architecture_pro_cost


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTRACT = ROOT / "configs/contracts/yambda500m_large_hstu_native_d7_d14_full_reuse_pro_v1.yaml"
EXECUTION_CONTRACT = ROOT / "configs/contracts/yambda500m_large_hstu_native_d7_d14_execution_v1.yaml"
D14_E14_SCOPE_AMENDMENT = ROOT / "configs/contracts/yambda500m_large_reuse_scope_d14_e14_only_v1.yaml"
REUSE_SCOPE_AMENDMENT = ROOT / "configs/contracts/yambda500m_large_full_only_stop_v1.yaml"
FORMAL_ACK = "RUN_LARGE_D7_D14_10L_H320"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LargePipeline:
    def __init__(self, contract_path: Path, execution_path: Path, threads: int,
                 reuse_scope_path: Path = REUSE_SCOPE_AMENDMENT) -> None:
        self.contract_path = contract_path.resolve()
        self.execution_path = execution_path.resolve()
        self.reuse_scope_path = reuse_scope_path.resolve()
        self.contract = yaml.safe_load(self.contract_path.read_text(encoding="utf-8"))
        self.contract_hash = sha256_file(self.contract_path)
        self.reuse_scope = yaml.safe_load(self.reuse_scope_path.read_text(encoding="utf-8"))
        self.reuse_scope_hash = sha256_file(self.reuse_scope_path)
        self.threads = int(threads)
        self.gpus = list(map(int, self.contract["resource_canary"]["physical_gpus"]))
        self.world = int(self.contract["resource_canary"]["world_size"])
        self.manifest = (ROOT / self.contract["manifest"]["output"]).resolve()
        self.dataset = (ROOT / self.contract["frozen_inputs"]["dataset_manifest"]).resolve()
        self.output = (ROOT / self.contract["outputs"]["root"]).resolve()
        self.logs = self.output / "logs"
        self.log_jsonl = self.logs / "pipeline.jsonl"
        self.state_path = (ROOT / self.contract["outputs"]["state"]).resolve()
        self.resource_root = (ROOT / self.contract["outputs"]["resource_canary"]).resolve()
        self.execution = None
        self.execution_hash = None
        self._validate_base()
        if self.execution_path.exists():
            self.execution = yaml.safe_load(self.execution_path.read_text(encoding="utf-8"))
            self.execution_hash = sha256_file(self.execution_path)
            self._validate_execution()
        self._validate_reuse_scope()

    def _validate_base(self) -> None:
        if self.contract["decision_basis"]["frozen_primary"] != "10L_H320_heads10_context1024":
            raise RuntimeError("Large primary architecture drifted")
        model = self.contract["model"]
        expected = (10, 320, 10, 1024)
        actual = tuple(int(model[key]) for key in ("num_layers", "hidden_size", "num_heads", "max_seq_len"))
        if actual != expected or int(model["hidden_size"]) // int(model["num_heads"]) != 32:
            raise RuntimeError(f"Large model shape drifted: {actual}")
        if self.world != 4 or self.gpus != [0, 1, 2, 3]:
            raise RuntimeError("Large qualification is frozen to one four-rank job on GPU0/1/2/3")
        frozen = self.contract["frozen_inputs"]
        for key in (
            "unified_scale_contract", "dataset_manifest", "item_mapping", "population",
            "small_PRO_quality_contract", "small_PRO_theoretical_compute",
        ):
            path = (ROOT / frozen[key]).resolve()
            if not path.exists() or sha256_file(path) != frozen[f"{key}_sha256"]:
                raise RuntimeError(f"frozen Large input mismatch: {key}")
        branches = self.contract["scope"]["branches"]
        if branches["D7"]["training_days"] != 7 or branches["D7"]["updates"] != 10 or branches["D7"]["evaluation_days"] != [7]:
            raise RuntimeError("D7 must remain ten updates with E7 only")
        if branches["D14"]["training_days"] != 14 or branches["D14"]["updates"] != 5 or branches["D14"]["evaluation_days"] != [7, 14]:
            raise RuntimeError("D14 must remain five updates with E7/E14")
        if branches["D14"]["v4_to_v5_E14_name"] != "E14_partial":
            raise RuntimeError("the incomplete fifth D14 E14 window must remain explicit")
        pro = self.contract["large_PRO"]
        cost = architecture_pro_cost(
            layers=int(model["num_layers"]), hidden=int(model["hidden_size"]),
            heads=int(model["num_heads"]), context=int(model["max_seq_len"]),
            repair_evidence=int(pro["repair_width"]), carriers=int(pro["carriers"]),
        )
        if abs(float(cost["over_full_fraction"]) - float(pro["theoretical_compute_fraction_of_Full"])) > 1e-12:
            raise RuntimeError("Large PRO theoretical compute no longer matches the frozen mapping")

    def _validate_execution(self) -> None:
        assert self.execution is not None
        parent = self.execution["frozen_parent"]
        if (ROOT / parent["contract"]).resolve() != self.contract_path or parent["contract_sha256"] != self.contract_hash:
            raise RuntimeError("Large execution seal does not bind this base contract")
        summary = self.resource_root / "summary.json"
        manifest_descriptor = self.manifest / "manifest.json"
        for key, path in (
            ("resource_canary_summary", summary),
            ("manifest", manifest_descriptor),
            ("requests_quality", self.manifest / "requests_quality.parquet"),
            ("requests_fidelity", self.manifest / "requests_fidelity.parquet"),
        ):
            if (ROOT / parent[key]).resolve() != path.resolve() or sha256_file(path) != parent[f"{key}_sha256"]:
                raise RuntimeError(f"Large execution parent mismatch: {key}")
        if json.loads(summary.read_text(encoding="utf-8"))["status"] != "large_resource_canary_passed":
            raise RuntimeError("formal execution is bound to a non-passing resource canary")
        runtime = self.execution["execution_amendment"]
        if int(runtime["world_size"]) != self.world or list(map(int, runtime["physical_gpus"])) != self.gpus:
            raise RuntimeError("Large execution world/GPU allowlist drifted")
        if runtime["architecture"] != "10L_H320_heads10_context1024":
            raise RuntimeError("only the frozen primary architecture may use this execution seal")

    def _validate_reuse_scope(self) -> None:
        parent = self.reuse_scope["frozen_parent"]
        if (ROOT / parent["base_contract"]).resolve() != self.contract_path:
            raise RuntimeError("Large Reuse scope amendment points to another base contract")
        if parent["base_contract_sha256"] != self.contract_hash:
            raise RuntimeError("Large Reuse scope amendment/base contract hash mismatch")
        if (ROOT / parent["execution_contract"]).resolve() != self.execution_path:
            raise RuntimeError("Large Reuse scope amendment points to another execution contract")
        if self.execution_hash is None or parent["execution_contract_sha256"] != self.execution_hash:
            raise RuntimeError("Large Reuse scope amendment/execution contract hash mismatch")
        if "superseded_reuse_scope" in parent:
            superseded = (ROOT / parent["superseded_reuse_scope"]).resolve()
            if not superseded.exists() or sha256_file(superseded) != parent["superseded_reuse_scope_sha256"]:
                raise RuntimeError("Large Full-only stop/superseded Reuse scope hash mismatch")
        scope = self.reuse_scope["reuse_scope"]
        if not bool(scope.get("formal_reuse_enabled", True)):
            if scope["branch"] != "none" or scope["horizon_days"] is not None:
                raise RuntimeError("disabled Large formal Reuse scope must not name a branch/horizon")
            if list(scope["edges"]) or int(scope["expected_cells"]) != 0:
                raise RuntimeError("disabled Large formal Reuse scope must contain zero cells")
            if bool(scope["include_frozen_PRO_in_same_raw_pass"]):
                raise RuntimeError("disabled Large formal Reuse scope cannot include PRO")
            return
        if scope["branch"] != "D14" or int(scope["horizon_days"]) != 14:
            raise RuntimeError("Large formal Reuse is restricted to D14/E14")
        if list(map(int, scope["edges"])) != [1, 2, 3, 4, 5] or int(scope["expected_cells"]) != 5:
            raise RuntimeError("Large D14/E14 Reuse scope must retain all five adjacent edges")
        if scope["edge5_name"] != "E14_partial" or not scope["include_frozen_PRO_in_same_raw_pass"]:
            raise RuntimeError("Large scoped Reuse must preserve the partial-tail marker and frozen PRO path")

    def reuse_tasks(self) -> list[tuple[str, int, int]]:
        scope = self.reuse_scope["reuse_scope"]
        if not bool(scope.get("formal_reuse_enabled", True)):
            return []
        return [
            (str(scope["branch"]), edge, int(scope["horizon_days"]))
            for edge in map(int, scope["edges"])
        ]

    def event(self, event: str, **values: object) -> None:
        self.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.log_jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": utc_now(), "event": event, **values}, ensure_ascii=False, sort_keys=True) + "\n")

    def write_state(self, status: str, **values: object) -> None:
        atomic_json(self.state_path, {
            "status": status,
            "updated_at_utc": utc_now(),
            "contract": str(self.contract_path.relative_to(ROOT)),
            "contract_sha256": self.contract_hash,
            "execution_contract": str(self.execution_path.relative_to(ROOT)) if self.execution else None,
            "execution_contract_sha256": self.execution_hash,
            "reuse_scope_amendment": str(self.reuse_scope_path.relative_to(ROOT)),
            "reuse_scope_amendment_sha256": self.reuse_scope_hash,
            "world_size": self.world,
            "physical_gpus": self.gpus,
            **values,
        })

    @property
    def distributed_prefix(self) -> list[str]:
        return ["torchrun", "--standalone", f"--nproc_per_node={self.world}"]

    @property
    def affinity(self) -> str:
        return ";".join(",".join(map(str, range(rank * 14, (rank + 1) * 14))) for rank in range(4))

    @property
    def cpu_args(self) -> list[str]:
        return [
            "--history-threads", "14", "--arrow-cpu-threads", "14",
            "--arrow-io-threads", "4", "--torch-cpu-threads", "4",
            "--cpu-affinity-by-rank", self.affinity,
        ]

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

    def _gpu_sample(self) -> list[dict]:
        output = subprocess.check_output([
            "nvidia-smi", f"--id={','.join(map(str, self.gpus))}",
            "--query-gpu=index,utilization.gpu,memory.used,memory.free,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5)
        rows = []
        for line in output.strip().splitlines():
            index, util, used, free, power = [value.strip() for value in line.split(",")]
            rows.append({
                "index": int(index), "utilization_percent": float(util),
                "memory_used_mib": float(used), "memory_free_mib": float(free),
                "power_watts": float(power),
            })
        return sorted(rows, key=lambda row: row["index"])

    def run(self, name: str, command: list[str], *, gpu: bool = False, env: dict[str, str] | None = None) -> dict:
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{name}.log"
        runtime_path = self.logs / f"{name}.runtime.json"
        if log_path.exists() or runtime_path.exists():
            raise RuntimeError(f"existing step log requires audit before rerun: {name}")
        self.event("step_start", name=name, command=command)
        print("+", " ".join(command), flush=True)
        started = time.perf_counter()
        samples: list[list[dict]] = []
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None

            def drain() -> None:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            try:
                while process.poll() is None:
                    if gpu:
                        try:
                            samples.append(self._gpu_sample())
                        except (subprocess.SubprocessError, ValueError):
                            pass
                    time.sleep(1.0)
                returncode = process.wait()
            except KeyboardInterrupt:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=15)
                reader.join(timeout=15)
                raise
            reader.join()
        elapsed = time.perf_counter() - started
        flattened = [row for sample in samples for row in sample]
        active_samples = [sample for sample in samples if any(row["utilization_percent"] >= 10 for row in sample)]
        active = [row for sample in active_samples for row in sample]
        runtime = {
            "status": "passed" if returncode == 0 else "failed",
            "name": name,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "gpu_monitoring": gpu,
            "samples": len(samples),
            "active_samples": len(active_samples),
            "active_duty_cycle_percent": None if not samples else 100.0 * len(active_samples) / len(samples),
            "overall_mean_gpu_utilization_percent": None if not flattened else sum(row["utilization_percent"] for row in flattened) / len(flattened),
            "active_mean_gpu_utilization_percent": None if not active else sum(row["utilization_percent"] for row in active) / len(active),
            "peak_memory_used_mib": None if not flattened else max(row["memory_used_mib"] for row in flattened),
            "peak_power_watts": None if not flattened else max(row["power_watts"] for row in flattened),
            "log": str(log_path.relative_to(ROOT)),
        }
        atomic_json(runtime_path, runtime)
        self.event("step_end", name=name, returncode=returncode, runtime=runtime)
        if returncode:
            self.write_state("failed", failed_step=name, returncode=returncode)
            raise subprocess.CalledProcessError(returncode, command)
        return runtime

    def validate_manifest(self) -> bool:
        descriptor = self.manifest / "manifest.json"
        if not descriptor.exists():
            if self.manifest.exists():
                raise RuntimeError(f"partial Large manifest requires audit: {self.manifest}")
            return False
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        if payload["contract_sha256"] != self.contract_hash:
            raise RuntimeError("Large request manifest belongs to another contract")
        for name, artifact in payload["artifacts"].items():
            path = self.manifest / name
            if not path.exists() or sha256_file(path) != artifact["sha256"]:
                raise RuntimeError(f"Large manifest artifact mismatch: {name}")
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
            raise RuntimeError("Large manifest builder returned without a valid seal")
        self.write_state("data_prepared", manifest=str(self.manifest.relative_to(ROOT)))

    def preflight(self) -> list[dict]:
        rows = subprocess.check_output([
            "nvidia-smi", f"--id={','.join(map(str, self.gpus))}",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()
        values = []
        for row in rows:
            index, name, total, free = [value.strip() for value in row.split(",", 3)]
            values.append({"index": int(index), "name": name, "memory_total_mib": int(total), "memory_free_mib": int(free)})
        values.sort(key=lambda value: value["index"])
        minimum = int(self.contract["resource_plan"]["minimum_free_memory_mib_per_gpu_before_job"])
        if [value["index"] for value in values] != self.gpus or any(value["memory_free_mib"] < minimum for value in values):
            raise RuntimeError(f"Large GPU preflight failed: {values}")
        free_gib = shutil.disk_usage(ROOT).free / 2**30
        if free_gib < float(self.contract["resource_plan"]["minimum_free_workspace_gib_before_formal"]):
            raise RuntimeError(f"workspace has only {free_gib:.1f} GiB free")
        self.event("preflight_pass", gpus=values, workspace_free_gib=free_gib)
        return values

    def train_command(self, branch: str, version: int, output: Path, batch: int,
                      *, parent: Path | None, canary_steps: int = 0, formal: bool = False) -> list[str]:
        if version == 0:
            start, end, block, branch_arg = 0, 217, "foundation", "shared"
        else:
            duration = int(self.contract["scope"]["branches"][branch]["training_days"])
            start, end, block, branch_arg = 217 + (version - 1) * duration, 217 + version * duration, "matrix_horizon", branch
        command = [
            *self.distributed_prefix, "scripts/train_yambda500m_foundation_fsdp.py",
            "--version", f"v{version}", "--branch", branch_arg,
            "--launch-contract", str(self.contract_path),
            "--manifest-dir", str(self.manifest), "--training-block", block,
            "--output", str(output), "--oov-buckets", str(self.contract["model"]["oov_buckets"]),
            "--passes", str(self.contract["training"]["passes"]),
            "--global-batch-size", str(batch), "--train-start-day", str(start), "--train-end-day", str(end),
            "--progress-interval", "100", *self.cpu_args,
        ]
        if formal:
            if self.execution is None:
                raise RuntimeError("formal training requires the post-canary execution seal")
            command.extend(["--execution-contract", str(self.execution_path)])
        if parent is not None:
            command.extend(["--parent", str(parent)])
        if canary_steps:
            command.extend(["--canary-steps", str(canary_steps)])
        return command

    def raw_full_command(self, branch: str, edge: int, horizon: int, output: Path,
                         parent: Path, current: Path, batch: int, max_users: int = 0,
                         allow_canary: bool = False) -> list[str]:
        duration = int(self.contract["scope"]["branches"][branch]["training_days"])
        cutover = 217 + edge * duration
        command = [
            *self.distributed_prefix, "scripts/evaluate_yambda500m_release_candidates_raw.py",
            "--stage", f"large_{branch}_{self.horizon_label(branch, edge, horizon)}_edge{edge}_full_only",
            "--block", "matrix_horizon", "--training-block", "matrix_horizon",
            "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
            "--parent", f"v{edge-1}={parent}", "--current", f"v{edge}={current}",
            "--start-day", str(cutover), "--end-day", str(cutover + horizon),
            "--training-start-day", str(cutover - duration), "--training-end-day", str(cutover),
            "--batch-size", str(batch), "--output", str(output), *self.cpu_args,
        ]
        if max_users:
            command.extend(["--max-users", str(max_users)])
        if allow_canary:
            command.append("--allow-canary-checkpoints")
        return command

    def raw_reuse_command(self, branch: str, edge: int, horizon: int, output: Path,
                          parent: Path, current: Path, cohort: int, query_chunk: int,
                          max_users: int = 0, allow_canary: bool = False) -> list[str]:
        duration = int(self.contract["scope"]["branches"][branch]["training_days"])
        cutover = 217 + edge * duration
        command = [
            *self.distributed_prefix, "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
            "--stage", f"large_{branch}_{self.horizon_label(branch, edge, horizon)}_edge{edge}_reuse",
            "--edge", f"v{edge-1}_to_v{edge}", "--cutover-day", str(cutover),
            "--start-day", str(cutover), "--end-day", str(cutover + horizon),
            "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
            "--parent", str(parent), "--current", str(current), "--output", str(output),
            "--cohort-size", str(cohort), "--query-chunk-size", str(query_chunk),
            "--include-parent-exact", *self.cpu_args,
        ]
        if branch == "D14":
            pro = self.contract["large_PRO"]
            command.extend([
                "--include-pro-lazy", "--pro-repair-width", str(pro["repair_width"]),
                "--pro-carriers", str(pro["carriers"]), "--pro-path", str(pro["path"]),
            ])
        if max_users:
            command.extend(["--max-users", str(max_users)])
        if allow_canary:
            command.append("--allow-canary-checkpoints")
        return command

    @staticmethod
    def safe_runtime(runtime: dict, *, minimum_util: float) -> bool:
        active = runtime.get("active_mean_gpu_utilization_percent")
        return runtime["returncode"] == 0 and active is not None and float(active) >= minimum_util

    def resource_canary(self) -> None:
        self.prepare()
        gpus = self.preflight()
        summary_path = self.resource_root / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if payload.get("contract_sha256") != self.contract_hash:
                raise RuntimeError("resource canary summary belongs to another contract")
            print(json.dumps(payload, indent=2))
            return
        self.resource_root.mkdir(parents=True, exist_ok=True)
        canary = self.contract["resource_canary"]
        minimum_util = float(canary["minimum_GPU_utilization_mean_percent_during_compute_phase"])
        training_candidates = []
        for batch in map(int, canary["train_candidates_in_order"]):
            directory = self.resource_root / f"train_b{batch}" / "v0"
            if (directory / "train_result.json").exists() and (directory / "checkpoint_100.pt").exists():
                runtime = json.loads((self.logs / f"canary_train_v0_b{batch}.runtime.json").read_text(encoding="utf-8"))
            else:
                try:
                    runtime = self.run(
                        f"canary_train_v0_b{batch}",
                        self.train_command("D7", 0, directory, batch, parent=None,
                                           canary_steps=int(canary["training_steps_per_candidate"])),
                        gpu=True, env=self.gpu_env,
                    )
                except subprocess.CalledProcessError:
                    training_candidates.append({"global_batch_size": batch, "status": "failed"})
                    break
            result = json.loads((directory / "train_result.json").read_text(encoding="utf-8"))
            peak = max(float(row["peak_reserved_mib"]) for row in result["rank_metrics"])
            entry = {
                "global_batch_size": batch, "status": "passed", "directory": str(directory.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(directory / "checkpoint_100.pt"),
                "peak_reserved_mib": peak, "throughput_requests_per_second": result["global_requests_per_second"],
                "finite_loss": bool(float("-inf") < float(result["mean_rank0_loss"]) < float("inf")),
                "runtime": runtime,
            }
            entry["safe"] = bool(
                peak < float(canary["formal_peak_reserved_mib_limit"])
                and entry["finite_loss"] and self.safe_runtime(runtime, minimum_util=minimum_util)
            )
            training_candidates.append(entry)
            if peak > float(canary["stop_escalation_if_peak_reserved_mib_above"]):
                break
        safe_train = [entry for entry in training_candidates if entry.get("safe")]
        if not safe_train:
            atomic_json(summary_path, {
                "status": "large_resource_canary_failed", "contract_sha256": self.contract_hash,
                "reason": "no_safe_training_batch", "training_candidates": training_candidates,
                "quality_labels_read": False,
            })
            raise RuntimeError("no Large training batch passed the physical canary")
        selected_train = safe_train[0]
        for entry in safe_train[1:]:
            if float(entry["throughput_requests_per_second"]) >= 0.95 * float(selected_train["throughput_requests_per_second"]):
                selected_train = entry
        train_batch = int(selected_train["global_batch_size"])
        v0 = ROOT / selected_train["directory"] / "checkpoint_100.pt"
        v1_dir = self.resource_root / "train_selected" / "v1"
        if (v1_dir / "train_result.json").exists() and (v1_dir / "checkpoint_100.pt").exists():
            v1_runtime = json.loads((self.logs / f"canary_train_v1_b{train_batch}.runtime.json").read_text(encoding="utf-8"))
        else:
            v1_runtime = self.run(
                f"canary_train_v1_b{train_batch}",
                self.train_command("D14", 1, v1_dir, train_batch, parent=v0,
                                   canary_steps=int(canary["training_steps_per_candidate"])),
                gpu=True, env=self.gpu_env,
            )
        v1_result = json.loads((v1_dir / "train_result.json").read_text(encoding="utf-8"))
        v1_peak = max(float(row["peak_reserved_mib"]) for row in v1_result["rank_metrics"])
        v1_safe = bool(
            v1_peak < float(canary["formal_peak_reserved_mib_limit"])
            and float("-inf") < float(v1_result["mean_rank0_loss"]) < float("inf")
            and self.safe_runtime(v1_runtime, minimum_util=minimum_util)
        )
        v1 = v1_dir / "checkpoint_100.pt"

        full_candidates = []
        for batch in map(int, canary["full_eval_batch_candidates_per_rank"]):
            directory = self.resource_root / f"full_b{batch}"
            if (directory / "raw.seal.json").exists() and (directory / "raw.parquet").exists():
                runtime = json.loads((self.logs / f"canary_full_b{batch}.runtime.json").read_text(encoding="utf-8"))
            else:
                runtime = self.run(
                    f"canary_full_b{batch}",
                    self.raw_full_command("D14", 1, 7, directory, v0, v1, batch,
                                          max_users=16, allow_canary=True),
                    gpu=True, env=self.gpu_env,
                )
            seal = json.loads((directory / "raw.seal.json").read_text(encoding="utf-8"))
            entry = {"batch_size_per_rank": batch, "runtime": runtime, "raw_sha256": seal["raw_sha256"]}
            entry["safe"] = self.safe_runtime(runtime, minimum_util=minimum_util) and (runtime["peak_memory_used_mib"] or 0) < float(canary["formal_peak_reserved_mib_limit"])
            full_candidates.append(entry)
        safe_full = [entry for entry in full_candidates if entry["safe"]]
        if not safe_full:
            raise RuntimeError("no Full evaluation batch passed the physical canary")
        selected_full = safe_full[0]
        for entry in safe_full[1:]:
            if float(entry["runtime"]["elapsed_seconds"]) <= 1.05 * float(selected_full["runtime"]["elapsed_seconds"]):
                selected_full = entry

        reuse_candidates = []
        query_chunk = int(canary["reuse_query_chunk_candidates_per_rank"][0])
        for cohort in map(int, canary["reuse_cohort_candidates_per_rank"]):
            directory = self.resource_root / f"reuse_c{cohort}_q{query_chunk}"
            if (directory / "raw.seal.json").exists() and (directory / "raw.parquet").exists():
                runtime = json.loads((self.logs / f"canary_reuse_c{cohort}_q{query_chunk}.runtime.json").read_text(encoding="utf-8"))
            else:
                runtime = self.run(
                    f"canary_reuse_c{cohort}_q{query_chunk}",
                    self.raw_reuse_command("D14", 1, 7, directory, v0, v1, cohort, query_chunk,
                                           max_users=12, allow_canary=True),
                    gpu=True, env=self.gpu_env,
                )
            seal = json.loads((directory / "raw.seal.json").read_text(encoding="utf-8"))
            entry = {"cohort_size_per_rank": cohort, "query_chunk_size_per_rank": query_chunk, "runtime": runtime, "raw_sha256": seal["raw_sha256"]}
            entry["safe"] = runtime["returncode"] == 0 and (runtime["peak_memory_used_mib"] or 0) < float(canary["formal_peak_reserved_mib_limit"])
            reuse_candidates.append(entry)
        safe_reuse = [entry for entry in reuse_candidates if entry["safe"]]
        if not safe_reuse:
            raise RuntimeError("no Reuse/PRO cohort passed the physical canary")
        selected_reuse = safe_reuse[0]
        for entry in safe_reuse[1:]:
            if float(entry["runtime"]["elapsed_seconds"]) <= 1.05 * float(selected_reuse["runtime"]["elapsed_seconds"]):
                selected_reuse = entry

        # The 12-user sizing runs are long enough for memory and relative
        # throughput, but too short for a stable one-second utilization sample.
        # Confirm the selected configuration on a longer label-free compute
        # segment without changing cohort or query-chunk parameters.
        selected_cohort = int(selected_reuse["cohort_size_per_rank"])
        utilization_output = self.resource_root / f"reuse_c{selected_cohort}_q{query_chunk}_utilization"
        utilization_name = f"canary_reuse_c{selected_cohort}_q{query_chunk}_utilization"
        if (utilization_output / "raw.seal.json").exists() and (utilization_output / "raw.parquet").exists():
            reuse_utilization_runtime = json.loads((self.logs / f"{utilization_name}.runtime.json").read_text(encoding="utf-8"))
        else:
            reuse_utilization_runtime = self.run(
                utilization_name,
                self.raw_reuse_command(
                    "D14", 1, 7, utilization_output, v0, v1,
                    selected_cohort, query_chunk, max_users=48, allow_canary=True,
                ), gpu=True, env=self.gpu_env,
            )
        reuse_compute_utilization_safe = self.safe_runtime(
            reuse_utilization_runtime, minimum_util=minimum_util
        )
        if not reuse_compute_utilization_safe:
            raise RuntimeError("selected Reuse/PRO runtime failed the extended GPU-utilization canary")

        state_output = self.resource_root / "state_io"
        if (state_output / "summary.json").exists():
            state_runtime = json.loads((self.logs / "canary_state_io.runtime.json").read_text(encoding="utf-8"))
        else:
            state_runtime = self.run(
                "canary_state_io",
                [
                    *self.distributed_prefix, "scripts/canary_yambda500m_large_state_io.py",
                    "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
                    "--parent", str(v0), "--current", str(v1), "--output", str(state_output),
                    "--cutover-day", "231", "--end-day", "238",
                    "--users-per-rank", str(canary["state_io_sample_users_per_rank"]),
                    "--allow-canary-checkpoints", *self.cpu_args,
                ], gpu=True, env=self.gpu_env,
            )
        state_summary = json.loads((state_output / "summary.json").read_text(encoding="utf-8"))
        passed = bool(
            v1_safe
            and state_summary["status"] == "large_state_io_canary_passed"
            and reuse_compute_utilization_safe
        )
        payload = {
            "status": "large_resource_canary_passed" if passed else "large_resource_canary_failed",
            "contract": str(self.contract_path.relative_to(ROOT)), "contract_sha256": self.contract_hash,
            "completed_at_utc": utc_now(), "quality_labels_read": False,
            "physical_gpus": self.gpus, "gpu_preflight": gpus,
            "architecture": self.contract["decision_basis"]["frozen_primary"],
            "training_candidates": training_candidates,
            "selected_train_global_batch_size": train_batch,
            "selected_v0_checkpoint": str(v0.relative_to(ROOT)), "selected_v0_checkpoint_sha256": sha256_file(v0),
            "selected_v1_checkpoint": str(v1.relative_to(ROOT)), "selected_v1_checkpoint_sha256": sha256_file(v1),
            "selected_v1_restore_and_training_safe": v1_safe,
            "full_candidates": full_candidates,
            "selected_full_batch_size_per_rank": int(selected_full["batch_size_per_rank"]),
            "reuse_candidates": reuse_candidates,
            "selected_reuse_cohort_size_per_rank": int(selected_reuse["cohort_size_per_rank"]),
            "selected_reuse_query_chunk_size_per_rank": int(selected_reuse["query_chunk_size_per_rank"]),
            "reuse_extended_compute_runtime": reuse_utilization_runtime,
            "state_io_summary": str((state_output / "summary.json").relative_to(ROOT)),
            "state_io_summary_sha256": sha256_file(state_output / "summary.json"),
            "state_io_runtime": state_runtime,
            "pro_path": self.contract["large_PRO"]["path"],
        }
        atomic_json(summary_path, payload)
        self.write_state(payload["status"], resource_canary_summary=str(summary_path.relative_to(ROOT)))
        if not passed:
            raise RuntimeError("Large focused resource canary failed")
        print(json.dumps(payload, indent=2))

    def checkpoint_dir(self, branch: str, version: int) -> Path:
        return self.output / "shared_v0" if version == 0 else self.output / branch / "checkpoints" / f"v{version}"

    def checkpoint(self, branch: str, version: int) -> Path:
        return self.checkpoint_dir(branch, version) / "checkpoint_100.pt"

    def validate_checkpoint(self, branch: str, version: int) -> bool:
        directory = self.checkpoint_dir(branch, version)
        checkpoint, result, seal = directory / "checkpoint_100.pt", directory / "train_result.json", directory / "checkpoint.seal.json"
        present = [path.exists() for path in (checkpoint, result, seal)]
        if not any(present):
            if directory.exists():
                raise RuntimeError(f"partial checkpoint directory requires audit: {directory}")
            return False
        if not all(present):
            raise RuntimeError(f"partial checkpoint artifacts require audit: {directory}")
        payload = json.loads(seal.read_text(encoding="utf-8"))
        expected = {
            "status": "large_checkpoint_sealed", "contract_sha256": self.contract_hash,
            "execution_contract_sha256": self.execution_hash, "world_size": self.world,
            "physical_gpus": self.gpus, "branch": "shared" if version == 0 else branch,
            "version": f"v{version}", "checkpoint_sha256": sha256_file(checkpoint),
        }
        if payload != expected:
            raise RuntimeError(f"checkpoint seal mismatch: {directory}")
        return True

    def train_formal(self, branch: str, version: int) -> None:
        if self.validate_checkpoint(branch, version):
            self.event("checkpoint_skip_valid", branch=branch, version=version)
            return
        assert self.execution is not None
        batch = int(self.execution["execution_amendment"]["global_train_batch_size"])
        parent = None if version == 0 else self.checkpoint(branch, version - 1)
        if parent is not None and not parent.exists():
            raise RuntimeError(f"direct parent checkpoint absent: {parent}")
        directory = self.checkpoint_dir(branch, version)
        runtime = self.run(
            f"train_{'shared' if version == 0 else branch}_v{version}",
            self.train_command(branch, version, directory, batch, parent=parent, formal=True),
            gpu=True, env=self.gpu_env,
        )
        result = json.loads((directory / "train_result.json").read_text(encoding="utf-8"))
        if result["status"] != "formal_training_complete" or result["execution_contract_sha256"] != self.execution_hash:
            raise RuntimeError("formal trainer output differs from execution seal")
        atomic_json(directory / "checkpoint.seal.json", {
            "status": "large_checkpoint_sealed", "contract_sha256": self.contract_hash,
            "execution_contract_sha256": self.execution_hash, "world_size": self.world,
            "physical_gpus": self.gpus, "branch": "shared" if version == 0 else branch,
            "version": f"v{version}", "checkpoint_sha256": sha256_file(directory / "checkpoint_100.pt"),
        })
        self.event("checkpoint_sealed", branch=branch, version=version, runtime=runtime)

    def horizon_label(self, branch: str, edge: int, horizon: int) -> str:
        if branch == "D14" and edge == 5 and horizon == 14:
            return "E14_partial"
        return f"E{horizon}"

    def full_dir(self, branch: str, edge: int, horizon: int) -> Path:
        return self.output / branch / "full_only" / self.horizon_label(branch, edge, horizon) / f"v{edge-1}_to_v{edge}"

    def reuse_dir(self, branch: str, edge: int, horizon: int) -> Path:
        return self.output / branch / "reuse" / self.horizon_label(branch, edge, horizon) / f"v{edge-1}_to_v{edge}"

    def evaluate_full(self, branch: str, edge: int, horizon: int) -> None:
        assert self.execution is not None
        directory = self.full_dir(branch, edge, horizon)
        raw, seal, report = directory / "raw.parquet", directory / "raw.seal.json", directory / "adjudication.json"
        if report.exists():
            if json.loads(report.read_text(encoding="utf-8"))["raw_sha256"] != sha256_file(raw):
                raise RuntimeError(f"Full report/raw mismatch: {directory}")
            return
        if directory.exists() and not (raw.exists() and seal.exists()):
            raise RuntimeError(f"partial Full directory requires audit: {directory}")
        if not directory.exists():
            batch = int(self.execution["execution_amendment"]["full_eval_batch_size_per_rank"])
            self.run(
                f"full_{branch}_{self.horizon_label(branch, edge, horizon)}_edge{edge}",
                self.raw_full_command(branch, edge, horizon, directory,
                                      self.checkpoint(branch, edge - 1), self.checkpoint(branch, edge), batch),
                gpu=True, env=self.gpu_env,
            )
        seal_payload = json.loads(seal.read_text(encoding="utf-8"))
        if sha256_file(raw) != seal_payload["raw_sha256"]:
            raise RuntimeError(f"Full raw seal mismatch: {directory}")
        self.run(
            f"adjudicate_full_{branch}_{self.horizon_label(branch, edge, horizon)}_edge{edge}",
            [sys.executable, "scripts/adjudicate_yambda500m_release_candidates.py",
             "--raw", str(raw), "--seal", str(seal),
             "--labels", str(self.manifest / "requests_quality.parquet"), "--output", str(report)],
            env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"},
        )

    def primary_horizon(self, branch: str, edge: int) -> int:
        return 7 if branch == "D7" or edge == 5 else 14

    def admission_path(self, branch: str, edge: int) -> Path:
        return self.output / branch / "admission" / f"v{edge-1}_to_v{edge}.seal.json"

    def seal_admission(self, branch: str, edge: int) -> dict:
        path = self.admission_path(branch, edge)
        horizon = self.primary_horizon(branch, edge)
        report_path = self.full_dir(branch, edge, horizon) / "adjudication.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["full_only_report_sha256"] != sha256_file(report_path):
                raise RuntimeError(f"admission seal/report mismatch: {path}")
            return payload
        report = json.loads(report_path.read_text(encoding="utf-8"))
        parent = report["parent_absolute"]["hstu_native"]
        candidate = report["candidates"][f"v{edge}"]
        current = candidate["absolute"]["hstu_native"]
        paired = candidate["paired_release_gain"]["parent_minus_current_log_loss"]
        gates = {
            "current_minus_parent_ROC_AUC_strictly_positive": current["ROC_AUC"] > parent["ROC_AUC"],
            "parent_minus_current_log_loss_strictly_positive": parent["log_loss"] > current["log_loss"],
            "current_brier_not_greater_than_parent": current["Brier"] <= parent["Brier"],
            "bootstrap_95CI_lower_strictly_positive": paired["user_cluster_bootstrap_95CI"]["p2_5"] > 0.0,
        }
        payload = {
            "status": "large_full_only_release_admission_sealed",
            "contract_sha256": self.contract_hash, "execution_contract_sha256": self.execution_hash,
            "branch": branch, "edge": f"v{edge-1}_to_v{edge}", "primary_horizon_days": horizon,
            "full_only_report_sha256": sha256_file(report_path), "gates": gates,
            "all_metric_gates_pass": all(gates.values()),
            "adjacent_reuse_PRO_diagnostic_unlocked_after_seal": True,
            "serving_lineage_promoted": False,
            "interpretation": "model admission is reported independently; all-edge compatibility diagnostics do not alter serving lineage",
        }
        atomic_json(path, payload)
        self.event("admission_sealed", branch=branch, edge=edge, all_metric_gates_pass=payload["all_metric_gates_pass"])
        return payload

    def evaluate_reuse(self, branch: str, edge: int, horizon: int) -> None:
        assert self.execution is not None
        admission = self.seal_admission(branch, edge)
        if not admission["adjacent_reuse_PRO_diagnostic_unlocked_after_seal"]:
            raise RuntimeError("Reuse attempted before Full-only admission seal")
        directory = self.reuse_dir(branch, edge, horizon)
        raw, seal, report = directory / "raw.parquet", directory / "raw.seal.json", directory / "adjudication.json"
        if report.exists():
            if json.loads(report.read_text(encoding="utf-8"))["raw_sha256"] != sha256_file(raw):
                raise RuntimeError(f"Reuse report/raw mismatch: {directory}")
            return
        if directory.exists() and not (raw.exists() and seal.exists()):
            raise RuntimeError(f"partial Reuse directory requires audit: {directory}")
        if not directory.exists():
            runtime = self.execution["execution_amendment"]
            self.run(
                f"reuse_{branch}_{self.horizon_label(branch, edge, horizon)}_edge{edge}",
                self.raw_reuse_command(
                    branch, edge, horizon, directory,
                    self.checkpoint(branch, edge - 1), self.checkpoint(branch, edge),
                    int(runtime["reuse_cohort_size_per_rank"]), int(runtime["reuse_query_chunk_size_per_rank"]),
                ), gpu=True, env=self.gpu_env,
            )
        seal_payload = json.loads(seal.read_text(encoding="utf-8"))
        if sha256_file(raw) != seal_payload["raw_sha256"]:
            raise RuntimeError(f"Reuse raw seal mismatch: {directory}")
        self.run(
            f"adjudicate_reuse_{branch}_{self.horizon_label(branch, edge, horizon)}_edge{edge}",
            [sys.executable, "scripts/adjudicate_yambda500m_hstu_native_onehop_reuse.py",
             "--raw", str(raw), "--seal", str(seal),
             "--labels", str(self.manifest / "requests_quality.parquet"), "--output", str(report)],
            env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"},
        )

    def train_all(self) -> None:
        self.train_formal("D7", 0)
        for branch in ("D7", "D14"):
            updates = int(self.contract["scope"]["branches"][branch]["updates"])
            for version in range(1, updates + 1):
                self.train_formal(branch, version)
        self.write_state("all_large_checkpoints_complete")

    def evaluate_all(self) -> None:
        # Preserve the protocol boundary: every Full-only cell and every
        # admission decision is sealed before the first Reuse/PRO label is read.
        for branch in ("D7", "D14"):
            values = self.contract["scope"]["branches"][branch]
            for edge in range(1, int(values["updates"]) + 1):
                for horizon in map(int, values["evaluation_days"]):
                    self.evaluate_full(branch, edge, horizon)
                self.seal_admission(branch, edge)
                self.summarize(require_complete=False)
        self.write_state("all_large_full_only_and_admission_complete")
        reuse_tasks = self.reuse_tasks()
        for branch, edge, horizon in reuse_tasks:
            self.evaluate_reuse(branch, edge, horizon)
            self.summarize(require_complete=False)
        self.write_state(
            "large_D14_E14_reuse_PRO_scope_complete"
            if reuse_tasks else "large_full_only_complete_formal_reuse_not_run"
        )

    def summary_rows(self) -> list[dict]:
        rows = []
        for branch, edge, horizon in self.reuse_tasks():
            admission_path = self.admission_path(branch, edge)
            admission = json.loads(admission_path.read_text(encoding="utf-8")) if admission_path.exists() else None
            report_path = self.reuse_dir(branch, edge, horizon) / "adjudication.json"
            if not report_path.exists():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = report["three_path_summary"]
            old, new, reuse = summary["old_parent"], summary["new_current"], summary["adjacent_one_hop_reuse"]
            old_auc = float(old["ROC_AUC"])
            row = {
                "branch": branch, "edge": f"v{edge-1}_to_v{edge}",
                "horizon": self.horizon_label(branch, edge, horizon),
                "horizon_days": horizon,
                "complete_horizon": not (branch == "D14" and edge == 5 and horizon == 14),
                "requests": summary["requests"],
                "model_admission_pass": None if admission is None else admission["all_metric_gates_pass"],
                "new_vs_old_AUC_relative_percent": 100.0 * (float(new["ROC_AUC"]) - old_auc) / old_auc,
                "reuse_AUC_gain_retained_percent": summary["reuse_AUC_gain_retained_percent"],
                "reuse_log_loss_gain_retained_percent": summary["reuse_log_loss_gain_retained_percent"],
                "old": old, "new": new, "reuse": reuse,
            }
            if "PRO" in summary:
                row["PRO"] = summary["PRO"]
            rows.append(row)
        return rows

    def summarize(self, *, require_complete: bool) -> None:
        rows = self.summary_rows()
        expected = int(self.reuse_scope["reuse_scope"]["expected_cells"])
        full_expected = sum(
            int(values["updates"]) * len(values["evaluation_days"])
            for values in self.contract["scope"]["branches"].values()
        )
        full_completed = sum(1 for _ in self.output.glob("D*/full_only/E*/v*_to_v*/adjudication.json"))
        reuse_enabled = bool(self.reuse_scope["reuse_scope"].get("formal_reuse_enabled", True))
        complete = (
            len(rows) == expected
            if reuse_enabled else full_completed == full_expected and len(rows) == 0
        )
        if require_complete and not complete:
            raise RuntimeError(
                f"Large summary is incomplete: Full={full_completed}/{full_expected}, "
                f"Reuse={len(rows)}/{expected}"
            )
        payload = {
            "status": (
                "large_D14_E14_reuse_PRO_scope_complete" if complete else "large_D14_E14_reuse_PRO_scope_in_progress"
            ) if reuse_enabled else (
                "large_full_only_complete_formal_reuse_not_run"
                if complete else "large_full_only_in_progress_formal_reuse_not_run"
            ),
            "contract_sha256": self.contract_hash, "execution_contract_sha256": self.execution_hash,
            "reuse_scope_amendment": str(self.reuse_scope_path.relative_to(ROOT)),
            "reuse_scope_amendment_sha256": self.reuse_scope_hash,
            "completed_cells": len(rows), "expected_cells": expected,
            "completed_full_only_cells": full_completed,
            "expected_full_only_cells": full_expected,
            "partial_horizon_policy": "D14 v4_to_v5 E14_partial is directional diagnostic only",
            "rows": rows,
        }
        atomic_json((ROOT / self.contract["outputs"]["summary_json"]).resolve(), payload)
        lines = [
            (
                "# Yambda-500M Large D14/E14 Reuse + PRO qualification"
                if reuse_enabled else "# Yambda-500M Large Full-only completion"
            ), "",
            f"Status: **{payload['status']}**. Full-only {full_completed}/{full_expected}; formal Reuse {len(rows)}/{expected}.", "",
            (
                "Formal Reuse is scoped to D14/E14 only; D7/E7 and D14/E7 Reuse are not run."
                if reuse_enabled else "Formal Reuse and PRO were cancelled before their first quality cell; the formal queue ends after Full-only."
            ), "",
            "`E14_partial` is never interpreted as a complete 14-day qualification horizon.", "",
            "| Branch | Edge | Horizon | Requests | New vs Old AUC | Reuse AUC gain retained | Reuse loss gain retained | PRO AUC gain retained | PRO loss gain retained |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            def pct(value):
                return "N/A" if value is None else f"{float(value):+.2f}%"
            pro = row.get("PRO", {})
            lines.append(
                f"| {row['branch']} | {row['edge'].replace('_to_', ' → ')} | {row['horizon']} | {row['requests']:,} | "
                f"{pct(row['new_vs_old_AUC_relative_percent'])} | {pct(row['reuse_AUC_gain_retained_percent'])} | "
                f"{pct(row['reuse_log_loss_gain_retained_percent'])} | {pct(pro.get('AUC_gain_retained_percent'))} | "
                f"{pct(pro.get('log_loss_gain_retained_percent'))} |"
            )
        atomic_text((ROOT / self.contract["outputs"]["summary_markdown"]).resolve(), "\n".join(lines) + "\n")

    def formal(self, acknowledgement: str | None) -> None:
        if acknowledgement != FORMAL_ACK:
            raise RuntimeError(f"formal Large run requires --acknowledge-long-run {FORMAL_ACK}")
        if self.execution is None:
            raise RuntimeError("formal Large run requires the post-canary execution contract")
        self.prepare()
        self.preflight()
        self.write_state("large_formal_started")
        self.train_all()
        self.evaluate_all()
        self.summarize(require_complete=True)
        self.write_state(
            "large_D14_E14_reuse_PRO_scope_complete"
            if self.reuse_tasks() else "large_full_only_complete_formal_reuse_not_run"
        )

    def status(self) -> None:
        checkpoints = sum(1 for path in self.output.glob("**/checkpoint.seal.json") if "resource_canary" not in path.parts)
        full = sum(1 for _ in self.output.glob("D*/full_only/E*/v*_to_v*/adjudication.json"))
        reuse = sum(1 for _ in self.output.glob("D*/reuse/E*/v*_to_v*/adjudication.json"))
        progress_paths = [
            path for path in self.output.glob("**/progress.json")
            if "resource_canary" not in path.parts
            and "implementation_canary" not in path.parts
            and "interruptions" not in path.parts
            and not any(part.startswith("aborted_") for part in path.parts)
        ]
        latest_progress_path = max(progress_paths, key=lambda path: path.stat().st_mtime) if progress_paths else None
        if latest_progress_path is not None:
            try:
                latest_progress_display = str(latest_progress_path.relative_to(ROOT))
            except ValueError:
                latest_progress_display = str(latest_progress_path)
        else:
            latest_progress_display = None
        payload = {
            "contract_sha256": self.contract_hash,
            "manifest_ready": self.validate_manifest() if self.manifest.exists() else False,
            "resource_canary": json.loads((self.resource_root / "summary.json").read_text(encoding="utf-8"))["status"] if (self.resource_root / "summary.json").exists() else "not_run",
            "execution_contract_present": self.execution is not None,
            "formal_checkpoints": {"complete": checkpoints, "expected": 16},
            "full_cells": {"complete": full, "expected": 20},
            "reuse_cells": {"complete": reuse, "expected": len(self.reuse_tasks())},
            "current_training_progress": (
                {
                    "path": latest_progress_display,
                    **json.loads(latest_progress_path.read_text(encoding="utf-8")),
                }
                if latest_progress_path is not None else None
            ),
            "pipeline_state": json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "resource-canary", "formal", "status"), required=True)
    parser.add_argument("--contract", type=Path, default=BASE_CONTRACT)
    parser.add_argument("--execution-contract", type=Path, default=EXECUTION_CONTRACT)
    parser.add_argument("--reuse-scope-contract", type=Path, default=REUSE_SCOPE_AMENDMENT)
    parser.add_argument("--threads", type=int, default=56)
    parser.add_argument("--acknowledge-long-run")
    args = parser.parse_args()
    pipeline = LargePipeline(
        args.contract, args.execution_contract, args.threads,
        reuse_scope_path=args.reuse_scope_contract,
    )
    if args.mode == "prepare":
        pipeline.prepare()
    elif args.mode == "resource-canary":
        pipeline.resource_canary()
    elif args.mode == "formal":
        pipeline.formal(args.acknowledge_long_run)
    else:
        pipeline.status()


if __name__ == "__main__":
    main()
