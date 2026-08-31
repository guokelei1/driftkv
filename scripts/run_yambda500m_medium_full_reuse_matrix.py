#!/usr/bin/env python3
"""Run the frozen Medium D7/D14 Full-only and adjacent-Reuse evaluations.

The formal matrix preserves Full-only admission before Reuse. A separately
contracted D7 forced-diagnostic mode can evaluate locked candidate-chain edges
without changing the formal admission or serving/cache lineage.
"""

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
CONTRACT = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d7_d14_full_reuse_v1.yaml"
EXECUTION = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d7_d14_execution_admission_v1.yaml"
CPU_RUNTIME = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_cpu_runtime_v2.yaml"
REUSE_RUNTIME = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_reuse_4gpu_runtime_v3.yaml"
FORCED_D7_REUSE = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d7_forced_reuse_diagnostic_v1.yaml"
FORMAL_ACK = "RUN_MEDIUM_D7_D14"
FORCED_D7_ACK = "RUN_MEDIUM_D7_FORCED_REUSE"


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


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Pipeline:
    def __init__(self, contract_path: Path, *, execution_path: Path = EXECUTION,
                 cpu_runtime_path: Path = CPU_RUNTIME,
                 reuse_runtime_path: Path = REUSE_RUNTIME,
                 forced_d7_reuse_path: Path = FORCED_D7_REUSE, threads: int) -> None:
        self.contract_path = contract_path.resolve()
        self.execution_path = execution_path.resolve()
        self.cpu_runtime_path = cpu_runtime_path.resolve()
        self.reuse_runtime_path = reuse_runtime_path.resolve()
        self.forced_d7_reuse_path = forced_d7_reuse_path.resolve()
        self.contract = yaml.safe_load(self.contract_path.read_text(encoding="utf-8"))
        self.execution = yaml.safe_load(self.execution_path.read_text(encoding="utf-8"))
        self.cpu_runtime = yaml.safe_load(self.cpu_runtime_path.read_text(encoding="utf-8"))
        self.reuse_runtime = yaml.safe_load(self.reuse_runtime_path.read_text(encoding="utf-8"))
        self.forced_d7_reuse = yaml.safe_load(self.forced_d7_reuse_path.read_text(encoding="utf-8"))
        self.contract_hash = sha256(self.contract_path)
        self.execution_hash = sha256(self.execution_path)
        self.cpu_runtime_hash = sha256(self.cpu_runtime_path)
        self.reuse_runtime_hash = sha256(self.reuse_runtime_path)
        self.forced_d7_reuse_hash = sha256(self.forced_d7_reuse_path)
        self.threads = int(threads)
        self.manifest = (ROOT / self.contract["manifest"]["output"]).resolve()
        self.output = (ROOT / self.contract["outputs"]["root"]).resolve()
        self.logs = self.output / "logs"
        self.log_jsonl = self.logs / "pipeline.jsonl"
        self.state_path = self.output / "pipeline_state.json"
        self.dataset = (ROOT / self.contract["frozen_inputs"]["dataset_manifest"]).resolve()
        amendment = self.execution["execution_amendment"]
        self.world = int(amendment["world_size"])
        self.physical_gpus = list(map(int, amendment["physical_gpus"]))
        self.global_batch = int(amendment["global_train_batch_size"])
        self.reuse_world = int(self.reuse_runtime["scope"]["world_size"])
        self.reuse_physical_gpus = list(map(int, self.reuse_runtime["scope"]["physical_gpus"]))
        self.forced_d7_output = (ROOT / self.forced_d7_reuse["scope"]["output"]).resolve()
        self.forced_d7_state_path = self.forced_d7_output / "state.json"
        self._validate_contracts()

    def _validate_contracts(self) -> None:
        frozen = self.contract["frozen_inputs"]
        for key in ("unified_scale_contract", "dataset_manifest", "item_mapping"):
            if sha256(ROOT / frozen[key]) != frozen[f"{key}_sha256"]:
                raise RuntimeError(f"frozen input hash mismatch: {key}")
        parent = self.execution["frozen_parent"]
        if parent["contract_sha256"] != self.contract_hash:
            raise RuntimeError("execution supplement does not bind this base contract")
        if (ROOT / parent["contract"]).resolve() != self.contract_path:
            raise RuntimeError("execution supplement points at another base contract")
        if self.world != 2 or self.physical_gpus != [2, 3]:
            raise RuntimeError("this runner is frozen to two ranks on physical GPU2/3")
        if self.global_batch != 32 or self.execution["execution_amendment"]["local_batch_sizes_by_rank"] != [16, 16]:
            raise RuntimeError("two-rank execution must preserve global batch 32 as 16/16")
        runtime_parent = self.cpu_runtime["frozen_parent"]
        if runtime_parent["matrix_contract_sha256"] != self.contract_hash:
            raise RuntimeError("CPU runtime does not bind the matrix contract")
        if runtime_parent["execution_contract_sha256"] != self.execution_hash:
            raise RuntimeError("CPU runtime does not bind the execution contract")
        for key in ("matrix_contract", "execution_contract", "abandoned_long_cpu_canary"):
            if sha256(ROOT / runtime_parent[key]) != runtime_parent[f"{key}_sha256"]:
                raise RuntimeError(f"CPU runtime parent hash mismatch: {key}")
        reuse_parent = self.reuse_runtime["frozen_parent"]
        expected_reuse_parents = {
            "matrix_contract": self.contract_hash,
            "execution_contract": self.execution_hash,
            "d14_cpu_runtime": self.cpu_runtime_hash,
        }
        for key, expected_hash in expected_reuse_parents.items():
            if reuse_parent[f"{key}_sha256"] != expected_hash:
                raise RuntimeError(f"four-GPU Reuse runtime does not bind {key}")
            if sha256(ROOT / reuse_parent[key]) != expected_hash:
                raise RuntimeError(f"four-GPU Reuse runtime parent hash mismatch: {key}")
        if self.reuse_world != 4 or self.reuse_physical_gpus != [0, 1, 2, 3]:
            raise RuntimeError("remaining D14 Reuse is frozen to four ranks on GPU0/1/2/3")
        reuse_cpu_sets = [set(map(int, self.reuse_runtime["runtime"][f"rank{rank}_cpu_affinity"])) for rank in range(self.reuse_world)]
        if any(len(values) != 14 for values in reuse_cpu_sets) or len(set().union(*reuse_cpu_sets)) != 56:
            raise RuntimeError("four-GPU Reuse requires 14 disjoint physical CPU cores per rank")
        forced_parent = self.forced_d7_reuse["frozen_parent"]
        expected_forced_parents = {
            "matrix_contract": self.contract_hash,
            "execution_contract": self.execution_hash,
            "four_gpu_runtime_contract": self.reuse_runtime_hash,
        }
        for key, expected_hash in expected_forced_parents.items():
            if forced_parent[f"{key}_sha256"] != expected_hash:
                raise RuntimeError(f"forced D7 Reuse contract does not bind {key}")
            if sha256(ROOT / forced_parent[key]) != expected_hash:
                raise RuntimeError(f"forced D7 Reuse parent hash mismatch: {key}")
        forced_scope = self.forced_d7_reuse["scope"]
        if forced_scope["branch"] != "D7" or list(map(int, forced_scope["horizons_days"])) != [3, 7]:
            raise RuntimeError("forced D7 Reuse scope must be D7 E3/E7")
        expected_edges = [f"v{edge-1}_to_v{edge}" for edge in range(1, 11)]
        if forced_scope["edges"] != expected_edges or int(forced_scope["expected_cells"]) != 20:
            raise RuntimeError("forced D7 Reuse scope must contain all twenty cells")
        expected_forced_output = self.output / "D7" / "forced_reuse_diagnostic_v1"
        if self.forced_d7_output != expected_forced_output:
            raise RuntimeError("forced D7 output must remain separate from formal Reuse")
        forced_runtime = self.forced_d7_reuse["runtime"]
        if list(map(int, forced_runtime["physical_gpus"])) != self.reuse_physical_gpus:
            raise RuntimeError("forced D7 Reuse must use the frozen four-GPU allowlist")
        if int(forced_runtime["world_size"]) != self.reuse_world:
            raise RuntimeError("forced D7 Reuse must use four ranks")
        for key in (
            "cohort_size_per_rank", "query_chunk_size_per_rank", "history_threads_per_rank",
            "arrow_cpu_threads_per_rank", "arrow_io_threads_per_rank",
            "torch_cpu_threads_per_rank", "omp_num_threads_per_rank",
        ):
            if int(forced_runtime[key]) != int(self.reuse_runtime["runtime"][key]):
                raise RuntimeError(f"forced D7 runtime drifted from proven four-GPU setting: {key}")
        for rank in range(self.reuse_world):
            if forced_runtime[f"rank{rank}_cpu_affinity"] != self.reuse_runtime["runtime"][f"rank{rank}_cpu_affinity"]:
                raise RuntimeError(f"forced D7 CPU affinity drifted on rank {rank}")
        admission_hashes = self.forced_d7_reuse["frozen_admission_evidence"]["seal_sha256_by_edge"]
        checkpoint_hashes = self.forced_d7_reuse["frozen_checkpoint_evidence"]["seal_sha256_by_version"]
        for edge_name, expected_hash in admission_hashes.items():
            path = self.output / "D7" / "admission" / f"{edge_name}.seal.json"
            if not path.exists() or sha256(path) != expected_hash:
                raise RuntimeError(f"forced D7 admission evidence changed: {edge_name}")
        for version_name, expected_hash in checkpoint_hashes.items():
            version = int(version_name[1:])
            path = self.checkpoint_dir("D7", version) / "checkpoint.seal.json"
            if not path.exists() or sha256(path) != expected_hash:
                raise RuntimeError(f"forced D7 checkpoint evidence changed: {version_name}")
        scope = self.contract["scope"]
        if scope["complete_source_days_half_open"] != [0, 300]:
            raise RuntimeError("Medium runner requires the complete [0,300) boundary")
        for name, branch in scope["branches"].items():
            duration, updates = int(branch["training_days"]), int(branch["updates"])
            if len(branch["versions"]) != updates + 1:
                raise RuntimeError(f"{name} version count differs from updates")
            if 217 + duration * updates + max(map(int, branch["evaluation_days"])) > 300:
                raise RuntimeError(f"{name} exceeds complete source data")

    def event(self, event: str, **values: object) -> None:
        self.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.log_jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": utc_now(), "event": event, **values}, ensure_ascii=False, sort_keys=True) + "\n")

    def write_state(self, stage: str, **values: object) -> None:
        atomic_json(self.state_path, {
            "status": stage, "updated_at_utc": utc_now(),
            "contract": str(self.contract_path.relative_to(ROOT)), "contract_sha256": self.contract_hash,
            "execution_contract": str(self.execution_path.relative_to(ROOT)), "execution_contract_sha256": self.execution_hash,
            "cpu_runtime_contract": str(self.cpu_runtime_path.relative_to(ROOT)), "cpu_runtime_contract_sha256": self.cpu_runtime_hash,
            "reuse_runtime_contract": str(self.reuse_runtime_path.relative_to(ROOT)), "reuse_runtime_contract_sha256": self.reuse_runtime_hash,
            "world_size": self.world, "physical_gpus": self.physical_gpus, **values,
        })

    def write_forced_d7_state(self, stage: str, **values: object) -> None:
        atomic_json(self.forced_d7_state_path, {
            "status": stage, "updated_at_utc": utc_now(),
            "forced_d7_reuse_contract": str(self.forced_d7_reuse_path.relative_to(ROOT)),
            "forced_d7_reuse_contract_sha256": self.forced_d7_reuse_hash,
            "matrix_contract_sha256": self.contract_hash,
            "execution_contract_sha256": self.execution_hash,
            "four_gpu_runtime_contract_sha256": self.reuse_runtime_hash,
            "world_size": self.reuse_world, "physical_gpus": self.reuse_physical_gpus,
            **values,
        })

    def run(self, name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{name}.log"
        retry = 1
        while log_path.exists():
            log_path = self.logs / f"{name}.retry{retry}.log"
            retry += 1
        self.event("step_start", name=name, command=command, log=str(log_path))
        print("+", " ".join(command), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            returncode = process.wait()
        self.event("step_end", name=name, returncode=returncode)
        if returncode:
            self.write_state("failed", failed_step=name, returncode=returncode)
            raise subprocess.CalledProcessError(returncode, command)

    @property
    def distributed_prefix(self) -> list[str]:
        return ["torchrun", "--standalone", f"--nproc_per_node={self.world}"]

    @property
    def gpu_env(self) -> dict[str, str]:
        omp = str(self.cpu_runtime["runtime"]["omp_num_threads_per_rank"])
        return {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "2,3", "OMP_NUM_THREADS": omp, "PYTHONUNBUFFERED": "1"}

    @property
    def reuse_distributed_prefix(self) -> list[str]:
        return ["torchrun", "--standalone", f"--nproc_per_node={self.reuse_world}"]

    @property
    def reuse_gpu_env(self) -> dict[str, str]:
        omp = str(self.reuse_runtime["runtime"]["omp_num_threads_per_rank"])
        devices = ",".join(map(str, self.reuse_physical_gpus))
        return {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": devices, "OMP_NUM_THREADS": omp, "PYTHONUNBUFFERED": "1"}

    def cpu_runtime_args(self, branch: str) -> list[str]:
        if branch != self.cpu_runtime["scope"]["applies_to_branch"]:
            return []
        runtime = self.cpu_runtime["runtime"]
        affinity = ";".join(
            ",".join(map(str, runtime[f"rank{rank}_cpu_affinity"])) for rank in range(self.world)
        )
        return [
            "--history-threads", str(runtime["history_threads_per_rank"]),
            "--arrow-cpu-threads", str(runtime["arrow_cpu_threads_per_rank"]),
            "--arrow-io-threads", str(runtime["arrow_io_threads_per_rank"]),
            "--torch-cpu-threads", str(runtime["torch_cpu_threads_per_rank"]),
            "--cpu-affinity-by-rank", affinity,
        ]

    def reuse_runtime_args(self) -> list[str]:
        runtime = self.reuse_runtime["runtime"]
        affinity = ";".join(
            ",".join(map(str, runtime[f"rank{rank}_cpu_affinity"]))
            for rank in range(self.reuse_world)
        )
        return [
            "--history-threads", str(runtime["history_threads_per_rank"]),
            "--arrow-cpu-threads", str(runtime["arrow_cpu_threads_per_rank"]),
            "--arrow-io-threads", str(runtime["arrow_io_threads_per_rank"]),
            "--torch-cpu-threads", str(runtime["torch_cpu_threads_per_rank"]),
            "--cpu-affinity-by-rank", affinity,
        ]

    def validate_manifest(self) -> bool:
        descriptor = self.manifest / "manifest.json"
        if not descriptor.exists():
            if self.manifest.exists():
                raise RuntimeError(f"partial manifest directory requires audit: {self.manifest}")
            return False
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        if payload["contract_sha256"] != self.contract_hash:
            raise RuntimeError("Medium request manifest was built from another contract")
        parent = self.execution["frozen_parent"]
        required = {
            descriptor: parent["manifest_sha256"],
            self.manifest / "requests_quality.parquet": parent["requests_quality_sha256"],
            self.manifest / "requests_fidelity.parquet": parent["requests_fidelity_sha256"],
        }
        for path, expected in required.items():
            if not path.exists() or sha256(path) != expected:
                raise RuntimeError(f"execution supplement input hash mismatch: {path}")
        for name, artifact in payload["artifacts"].items():
            path = self.manifest / name
            if not path.exists() or sha256(path) != artifact["sha256"]:
                raise RuntimeError(f"manifest artifact hash mismatch: {name}")
        return True

    def prepare(self) -> None:
        if self.validate_manifest():
            self.event("manifest_skip_valid")
            return
        self.run("prepare_manifest", [sys.executable, "scripts/build_yambda500m_hstu_native_matrix_manifest.py", "--contract", str(self.contract_path), "--output", str(self.manifest), "--threads", str(self.threads)], env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"})
        if not self.validate_manifest():
            raise RuntimeError("manifest builder returned without a valid artifact")
        self.write_state("data_prepared", manifest=str(self.manifest))

    def gpu_preflight(self, *, physical_gpus: list[int] | None = None) -> list[dict]:
        required_gpus = self.physical_gpus if physical_gpus is None else physical_gpus
        rows = subprocess.check_output([
            "nvidia-smi", f"--id={','.join(map(str, required_gpus))}", "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()
        values = []
        for row in rows:
            index, name, total, free = [value.strip() for value in row.split(",", 3)]
            if int(index) in required_gpus:
                values.append({"index": int(index), "name": name, "memory_total_mib": int(total), "memory_free_mib": int(free)})
        values.sort(key=lambda value: value["index"])
        if [value["index"] for value in values] != required_gpus:
            raise RuntimeError(f"Medium execution requires physical GPUs {required_gpus}")
        minimum = int(self.contract["resource_plan"]["minimum_free_memory_mib_per_gpu_for_canary"])
        deficient = [value for value in values if value["memory_free_mib"] < minimum]
        if deficient:
            self.event("gpu_preflight_blocked", gpus=values, deficient=deficient)
            self.write_state("gpu_preflight_blocked", gpus=values, deficient=deficient)
            raise RuntimeError(f"GPU canary preflight lacks free memory: {deficient}")
        self.event("gpu_preflight_pass", gpus=values)
        return values

    def disk_preflight(self) -> None:
        free = shutil.disk_usage(ROOT).free / 2**30
        minimum = float(self.contract["resource_plan"]["minimum_free_workspace_gib_before_formal"])
        if free < minimum:
            raise RuntimeError(f"workspace has {free:.1f} GiB free; contract requires {minimum:.1f}")
        self.event("disk_preflight_pass", free_gib=free)

    def checkpoint_dir(self, branch: str, version: int, *, smoke: bool = False) -> Path:
        root = self.output / "smoke" if smoke else self.output
        return root / "shared_v0" if version == 0 else root / branch / "checkpoints" / f"v{version}"

    def checkpoint(self, branch: str, version: int, *, smoke: bool = False) -> Path:
        return self.checkpoint_dir(branch, version, smoke=smoke) / "checkpoint_100.pt"

    def validate_checkpoint(self, branch: str, version: int, *, smoke: bool = False) -> bool:
        directory = self.checkpoint_dir(branch, version, smoke=smoke)
        checkpoint, result, seal = directory / "checkpoint_100.pt", directory / "train_result.json", directory / "checkpoint.seal.json"
        existing = [path.exists() for path in (checkpoint, result, seal)]
        if not any(existing):
            if directory.exists():
                raise RuntimeError(f"partial checkpoint directory requires audit: {directory}")
            return False
        if not all(existing):
            raise RuntimeError(f"partial checkpoint artifacts require audit: {directory}")
        payload = json.loads(seal.read_text(encoding="utf-8"))
        expected = {
            "status": "medium_checkpoint_sealed", "contract_sha256": self.contract_hash,
            "execution_contract_sha256": self.execution_hash, "world_size": self.world,
            "physical_gpus": self.physical_gpus, "branch": "shared" if version == 0 else branch,
            "version": f"v{version}", "checkpoint_sha256": sha256(checkpoint), "smoke": smoke,
        }
        if payload != expected:
            raise RuntimeError(f"checkpoint seal mismatch: {checkpoint}")
        return True

    def seal_checkpoint(self, branch: str, version: int, *, smoke: bool = False) -> None:
        directory = self.checkpoint_dir(branch, version, smoke=smoke)
        checkpoint = directory / "checkpoint_100.pt"
        result = json.loads((directory / "train_result.json").read_text(encoding="utf-8"))
        if result["contract_sha256"] != self.contract_hash or result["execution_contract_sha256"] != self.execution_hash:
            raise RuntimeError("trainer result differs from frozen contracts")
        if result["version"] != f"v{version}" or result["world_size"] != self.world:
            raise RuntimeError("trainer result differs from frozen version/world")
        atomic_json(directory / "checkpoint.seal.json", {
            "status": "medium_checkpoint_sealed", "contract_sha256": self.contract_hash,
            "execution_contract_sha256": self.execution_hash, "world_size": self.world,
            "physical_gpus": self.physical_gpus, "branch": "shared" if version == 0 else branch,
            "version": f"v{version}", "checkpoint_sha256": sha256(checkpoint), "smoke": smoke,
        })

    def train(self, branch: str, version: int, *, smoke: bool = False) -> None:
        if self.validate_checkpoint(branch, version, smoke=smoke):
            self.event("checkpoint_skip_valid", branch=branch, version=version, smoke=smoke)
            return
        if version == 0:
            start, end, block, parent, branch_arg = 0, 217, "foundation", None, "shared"
        else:
            branch_cfg = self.contract["scope"]["branches"][branch]
            duration = int(branch_cfg["training_days"])
            start, end, block = 217 + (version - 1) * duration, 217 + version * duration, "matrix_horizon"
            parent, branch_arg = self.checkpoint(branch, version - 1, smoke=smoke), branch
            if not parent.exists():
                raise RuntimeError(f"direct parent checkpoint is absent: {parent}")
        command = [
            *self.distributed_prefix, "scripts/train_yambda500m_foundation_fsdp.py",
            "--version", f"v{version}", "--branch", branch_arg,
            "--launch-contract", str(self.contract_path), "--execution-contract", str(self.execution_path),
            "--manifest-dir", str(self.manifest), "--training-block", block,
            "--output", str(self.checkpoint_dir(branch, version, smoke=smoke)),
            "--oov-buckets", str(self.contract["model"]["oov_buckets"]),
            "--passes", str(self.contract["training"]["passes"]), "--global-batch-size", str(self.global_batch),
            "--train-start-day", str(start), "--train-end-day", str(end),
        ]
        if parent is not None:
            command.extend(["--parent", str(parent)])
        if smoke:
            command.extend(["--canary-steps", str(self.execution["execution_amendment"]["focused_correctness_canary"]["training_steps"])])
        self.run(f"{'smoke_' if smoke else ''}train_{branch_arg}_v{version}", command, env=self.gpu_env)
        self.seal_checkpoint(branch, version, smoke=smoke)

    def full_only_dir(self, branch: str, edge: int, horizon: int, *, smoke: bool = False) -> Path:
        root = self.output / "smoke" if smoke else self.output
        return root / branch / "full_only" / f"E{horizon}" / f"v{edge-1}_to_v{edge}"

    def reuse_dir(self, branch: str, edge: int, horizon: int, *, smoke: bool = False,
                  forced_diagnostic: bool = False) -> Path:
        if forced_diagnostic:
            if smoke or branch != "D7":
                raise RuntimeError("forced diagnostic Reuse is formal-checkpoint D7 only")
            return self.forced_d7_output / f"E{horizon}" / f"v{edge-1}_to_v{edge}"
        root = self.output / "smoke" if smoke else self.output
        return root / branch / "reuse" / f"E{horizon}" / f"v{edge-1}_to_v{edge}"

    def evaluate_full_only(self, branch: str, edge: int, horizon: int, *, smoke: bool = False) -> None:
        directory = self.full_only_dir(branch, edge, horizon, smoke=smoke)
        report, raw, seal = directory / "adjudication.json", directory / "raw.parquet", directory / "raw.seal.json"
        if report.exists():
            if json.loads(report.read_text())["raw_sha256"] != json.loads(seal.read_text())["raw_sha256"]:
                raise RuntimeError(f"Full-only report/seal mismatch: {directory}")
            self.event("full_only_skip_valid", branch=branch, edge=edge, horizon=horizon, smoke=smoke)
            return
        duration = int(self.contract["scope"]["branches"][branch]["training_days"])
        cutover, train_start = 217 + edge * duration, 217 + (edge - 1) * duration
        if directory.exists() and not (raw.exists() and seal.exists()):
            raise RuntimeError(f"partial Full-only directory requires audit: {directory}")
        if not directory.exists():
            command = [
                *self.distributed_prefix, "scripts/evaluate_yambda500m_release_candidates_raw.py",
                "--stage", f"medium_{branch}_E{horizon}_edge{edge}_full_only",
                "--block", "matrix_horizon", "--training-block", "matrix_horizon",
                "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
                "--parent", f"v{edge-1}={self.checkpoint(branch, edge-1, smoke=smoke)}",
                "--current", f"v{edge}={self.checkpoint(branch, edge, smoke=smoke)}",
                "--start-day", str(cutover), "--end-day", str(cutover + horizon),
                "--training-start-day", str(train_start), "--training-end-day", str(cutover),
                "--output", str(directory),
                *self.cpu_runtime_args(branch),
            ]
            if smoke:
                command.extend(["--max-users", "16", "--allow-canary-checkpoints"])
            self.run(f"{'smoke_' if smoke else ''}full_{branch}_E{horizon}_edge{edge}_raw", command, env=self.gpu_env)
        if sha256(raw) != json.loads(seal.read_text())["raw_sha256"]:
            raise RuntimeError(f"Full-only raw differs from seal: {directory}")
        self.run(f"{'smoke_' if smoke else ''}full_{branch}_E{horizon}_edge{edge}_adjudicate", [
            sys.executable, "scripts/adjudicate_yambda500m_release_candidates.py", "--raw", str(raw), "--seal", str(seal),
            "--labels", str(self.manifest / "requests_quality.parquet"), "--output", str(report),
        ], env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"})

    def primary_horizon(self, branch: str) -> int:
        return int(self.execution["release_admission"]["primary_horizon_by_branch"][branch])

    def admission_path(self, branch: str, edge: int) -> Path:
        return self.output / branch / "admission" / f"v{edge-1}_to_v{edge}.seal.json"

    def seal_admission(self, branch: str, edge: int) -> dict:
        path = self.admission_path(branch, edge)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            report = self.full_only_dir(branch, edge, self.primary_horizon(branch)) / "adjudication.json"
            if payload["full_only_report_sha256"] != sha256(report):
                raise RuntimeError(f"admission seal/report mismatch: {path}")
            return payload
        horizon = self.primary_horizon(branch)
        report_path = self.full_only_dir(branch, edge, horizon) / "adjudication.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidate = report["candidates"][f"v{edge}"]
        parent_metrics = report["parent_absolute"]["hstu_native"]
        current_metrics = candidate["absolute"]["hstu_native"]
        paired = candidate["paired_release_gain"]["parent_minus_current_log_loss"]
        gates = {
            "current_minus_parent_ROC_AUC_strictly_positive": current_metrics["ROC_AUC"] is not None and parent_metrics["ROC_AUC"] is not None and current_metrics["ROC_AUC"] > parent_metrics["ROC_AUC"],
            "parent_minus_current_log_loss_strictly_positive": parent_metrics["log_loss"] > current_metrics["log_loss"],
            "current_brier_not_greater_than_parent": current_metrics["Brier"] <= parent_metrics["Brier"],
            "bootstrap_95CI_lower_strictly_positive": paired["user_cluster_bootstrap_95CI"]["p2_5"] > 0.0,
        }
        parent_accepted = edge == 1 or bool(self.seal_admission(branch, edge - 1)["reuse_unlocked"])
        metric_pass, unlocked = all(gates.values()), bool(parent_accepted and all(gates.values()))
        reason = "accepted_for_adjacent_reuse_evaluation" if unlocked else ("parent_not_in_accepted_diagnostic_lineage" if not parent_accepted else "full_only_quality_gate_failed")
        payload = {
            "status": "medium_full_only_release_eligibility_sealed",
            "contract_sha256": self.contract_hash, "execution_contract_sha256": self.execution_hash,
            "branch": branch, "edge": f"v{edge-1}_to_v{edge}", "primary_horizon_days": horizon,
            "full_only_report_sha256": sha256(report_path), "parent_in_accepted_diagnostic_lineage": parent_accepted,
            "gates": gates, "all_metric_gates_pass": metric_pass, "reuse_unlocked": unlocked,
            "serving_lineage_promoted": False, "reason": reason,
        }
        atomic_json(path, payload)
        self.event("admission_sealed", branch=branch, edge=edge, reuse_unlocked=unlocked, gates=gates)
        return payload

    def evaluate_reuse(self, branch: str, edge: int, horizon: int, *, smoke: bool = False,
                       forced_diagnostic: bool = False) -> None:
        if forced_diagnostic and (smoke or branch != "D7"):
            raise RuntimeError("forced diagnostic Reuse is formal-checkpoint D7 only")
        if not smoke and not forced_diagnostic:
            admission = self.seal_admission(branch, edge)
            if not admission["reuse_unlocked"]:
                self.event("reuse_locked", branch=branch, edge=edge, horizon=horizon, reason=admission["reason"])
                return
        if forced_diagnostic:
            admission = self.seal_admission(branch, edge)
            if admission["reuse_unlocked"]:
                raise RuntimeError("forced diagnostic path is reserved for a formally locked D7 edge")
        directory = self.reuse_dir(
            branch, edge, horizon, smoke=smoke, forced_diagnostic=forced_diagnostic,
        )
        report, raw, seal = directory / "adjudication.json", directory / "raw.parquet", directory / "raw.seal.json"
        diagnostic_seal = directory / "forced_diagnostic.seal.json"
        if report.exists():
            if json.loads(report.read_text())["raw_sha256"] != json.loads(seal.read_text())["raw_sha256"]:
                raise RuntimeError(f"Reuse report/seal mismatch: {directory}")
            if forced_diagnostic:
                if not diagnostic_seal.exists():
                    raise RuntimeError(f"forced D7 report lacks its diagnostic seal: {directory}")
                payload = json.loads(diagnostic_seal.read_text(encoding="utf-8"))
                expected = {
                    "forced_d7_reuse_contract_sha256": self.forced_d7_reuse_hash,
                    "raw_sha256": sha256(raw), "adjudication_sha256": sha256(report),
                    "admission_seal_sha256": sha256(self.admission_path(branch, edge)),
                }
                if any(payload.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"forced D7 diagnostic seal mismatch: {directory}")
            self.event("reuse_skip_valid", branch=branch, edge=edge, horizon=horizon, smoke=smoke,
                       forced_diagnostic=forced_diagnostic)
            return
        duration = int(self.contract["scope"]["branches"][branch]["training_days"])
        cutover = 217 + edge * duration
        if directory.exists() and not (raw.exists() and seal.exists()):
            raise RuntimeError(f"partial Reuse directory requires audit: {directory}")
        if not directory.exists():
            runtime = self.reuse_runtime["runtime"]
            distributed_prefix = self.distributed_prefix if smoke else self.reuse_distributed_prefix
            cohort_size = self.contract["evaluation"]["evaluation_cohort_size_per_rank"] if smoke else runtime["cohort_size_per_rank"]
            query_chunk_size = self.contract["evaluation"]["query_chunk_size"] if smoke else runtime["query_chunk_size_per_rank"]
            runtime_args = self.cpu_runtime_args(branch) if smoke else self.reuse_runtime_args()
            stage_suffix = "_forced_diagnostic" if forced_diagnostic else ""
            command = [
                *distributed_prefix, "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
                "--stage", f"medium_{branch}_E{horizon}_edge{edge}_reuse{stage_suffix}",
                "--edge", f"v{edge-1}_to_v{edge}",
                "--cutover-day", str(cutover), "--start-day", str(cutover), "--end-day", str(cutover + horizon),
                "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
                "--parent", str(self.checkpoint(branch, edge-1, smoke=smoke)), "--current", str(self.checkpoint(branch, edge, smoke=smoke)),
                "--output", str(directory), "--include-parent-exact",
                "--cohort-size", str(cohort_size), "--query-chunk-size", str(query_chunk_size),
                *runtime_args,
            ]
            if smoke:
                command.extend(["--max-users", "16", "--allow-canary-checkpoints"])
            runtime_env = self.gpu_env if smoke else self.reuse_gpu_env
            name_prefix = "forced_diagnostic_" if forced_diagnostic else ""
            self.run(f"{name_prefix}{'smoke_' if smoke else ''}reuse_{branch}_E{horizon}_edge{edge}_raw", command, env=runtime_env)
        if sha256(raw) != json.loads(seal.read_text())["raw_sha256"]:
            raise RuntimeError(f"Reuse raw differs from seal: {directory}")
        name_prefix = "forced_diagnostic_" if forced_diagnostic else ""
        self.run(f"{name_prefix}{'smoke_' if smoke else ''}reuse_{branch}_E{horizon}_edge{edge}_adjudicate", [
            sys.executable, "scripts/adjudicate_yambda500m_hstu_native_onehop_reuse.py", "--raw", str(raw), "--seal", str(seal),
            "--labels", str(self.manifest / "requests_quality.parquet"), "--output", str(report),
        ], env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"})
        if forced_diagnostic:
            atomic_json(diagnostic_seal, {
                "status": "medium_D7_forced_adjacent_reuse_diagnostic_cell_sealed",
                "forced_d7_reuse_contract": str(self.forced_d7_reuse_path.relative_to(ROOT)),
                "forced_d7_reuse_contract_sha256": self.forced_d7_reuse_hash,
                "branch": branch, "edge": f"v{edge-1}_to_v{edge}", "horizon_days": horizon,
                "raw_sha256": sha256(raw), "raw_seal_sha256": sha256(seal),
                "adjudication_sha256": sha256(report),
                "admission_seal_sha256": sha256(self.admission_path(branch, edge)),
                "formal_admission_unchanged": True, "serving_lineage_promotion": False,
                "interpretation": "forced_diagnostic_only_not_formal_release_qualification",
            })

    def reuse_runtime_canary(self) -> None:
        canary = self.reuse_runtime["correctness_canary"]
        directory = (ROOT / canary["output"]).resolve()
        raw, seal, marker = directory / "raw.parquet", directory / "raw.seal.json", directory / "canary.pass.json"
        if marker.exists():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("reuse_runtime_contract_sha256") != self.reuse_runtime_hash:
                raise RuntimeError("four-GPU Reuse canary marker belongs to another runtime contract")
            if not raw.exists() or sha256(raw) != payload.get("raw_sha256"):
                raise RuntimeError("four-GPU Reuse canary raw differs from its pass marker")
            self.event("reuse_4gpu_canary_skip_valid")
            return
        if directory.exists() and not (raw.exists() and seal.exists()):
            raise RuntimeError(f"partial four-GPU Reuse canary requires audit: {directory}")
        if not directory.exists():
            command = [
                *self.reuse_distributed_prefix, "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
                "--stage", "medium_D14_E3_edge2_reuse_4gpu_runtime_canary",
                "--edge", "v1_to_v2", "--cutover-day", "245", "--start-day", "245", "--end-day", "248",
                "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
                "--parent", str(self.checkpoint("D14", 1)), "--current", str(self.checkpoint("D14", 2)),
                "--output", str(directory), "--include-parent-exact",
                "--cohort-size", str(self.reuse_runtime["runtime"]["cohort_size_per_rank"]),
                "--query-chunk-size", str(self.reuse_runtime["runtime"]["query_chunk_size_per_rank"]),
                "--max-users", str(canary["maximum_users_per_rank"]),
                *self.reuse_runtime_args(),
            ]
            self.run("reuse_D14_4gpu_runtime_canary_raw", command, env=self.reuse_gpu_env)
        payload = json.loads(seal.read_text(encoding="utf-8"))
        execution = payload.get("execution_runtime", {})
        peaks = execution.get("peak_memory_by_rank", [])
        maximum_peak = float(canary["maximum_peak_reserved_mib_per_rank"])
        checks = {
            "raw_hash_matches": sha256(raw) == payload.get("raw_sha256"),
            "three_paths_per_request": payload.get("rows") == 3 * payload.get("requests", -1),
            "world_size": execution.get("world_size") == self.reuse_world,
            "cohort_size": execution.get("cohort_size_per_rank") == self.reuse_runtime["runtime"]["cohort_size_per_rank"],
            "query_chunk_size": execution.get("query_chunk_size_per_rank") == self.reuse_runtime["runtime"]["query_chunk_size_per_rank"],
            "four_memory_records": len(peaks) == self.reuse_world,
            "peak_reserved_below_limit": len(peaks) == self.reuse_world and all(float(value["peak_reserved_mib"]) < maximum_peak for value in peaks),
        }
        if not all(checks.values()):
            raise RuntimeError(f"four-GPU Reuse runtime canary failed: {checks}")
        atomic_json(marker, {
            "status": "medium_D14_remaining_reuse_4gpu_runtime_canary_passed",
            "reuse_runtime_contract_sha256": self.reuse_runtime_hash,
            "raw_sha256": payload["raw_sha256"], "requests": payload["requests"],
            "execution_runtime": execution, "checks": checks, "quality_interpretation": "prohibited_raw_only",
        })
        self.event("reuse_4gpu_canary_pass", execution_runtime=execution)

    def forced_d7_reuse_canary(self) -> None:
        canary = self.forced_d7_reuse["correctness_canary"]
        directory = (ROOT / canary["output"]).resolve()
        raw, seal, marker = directory / "raw.parquet", directory / "raw.seal.json", directory / "canary.pass.json"
        if marker.exists():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            expected = {
                "forced_d7_reuse_contract_sha256": self.forced_d7_reuse_hash,
                "raw_sha256": sha256(raw) if raw.exists() else None,
            }
            if any(payload.get(key) != value for key, value in expected.items()):
                raise RuntimeError("forced D7 Reuse canary marker mismatch")
            self.event("forced_d7_reuse_canary_skip_valid")
            return
        if directory.exists() and not (raw.exists() and seal.exists()):
            raise RuntimeError(f"partial forced D7 Reuse canary requires audit: {directory}")
        edge_name = str(canary["edge"])
        edge = int(edge_name.split("_to_v", 1)[1])
        horizon = int(canary["horizon_days"])
        cutover = 217 + edge * 7
        if not directory.exists():
            runtime = self.forced_d7_reuse["runtime"]
            command = [
                *self.reuse_distributed_prefix,
                "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
                "--stage", f"medium_D7_E{horizon}_edge{edge}_forced_reuse_diagnostic_canary",
                "--edge", edge_name, "--cutover-day", str(cutover),
                "--start-day", str(cutover), "--end-day", str(cutover + horizon),
                "--manifest-dir", str(self.manifest), "--dataset-manifest", str(self.dataset),
                "--parent", str(self.checkpoint("D7", edge - 1)),
                "--current", str(self.checkpoint("D7", edge)),
                "--output", str(directory), "--include-parent-exact",
                "--cohort-size", str(runtime["cohort_size_per_rank"]),
                "--query-chunk-size", str(runtime["query_chunk_size_per_rank"]),
                "--max-users", str(canary["maximum_users_per_rank"]),
                *self.reuse_runtime_args(),
            ]
            self.run("forced_diagnostic_reuse_D7_4gpu_canary_raw", command, env=self.reuse_gpu_env)
        payload = json.loads(seal.read_text(encoding="utf-8"))
        execution = payload.get("execution_runtime", {})
        peaks = execution.get("peak_memory_by_rank", [])
        maximum_peak = float(canary["maximum_peak_reserved_mib_per_rank"])
        checks = {
            "raw_hash_matches": sha256(raw) == payload.get("raw_sha256"),
            "three_paths_per_request": payload.get("rows") == 3 * payload.get("requests", -1),
            "world_size": execution.get("world_size") == self.reuse_world,
            "cohort_size": execution.get("cohort_size_per_rank") == self.forced_d7_reuse["runtime"]["cohort_size_per_rank"],
            "query_chunk_size": execution.get("query_chunk_size_per_rank") == self.forced_d7_reuse["runtime"]["query_chunk_size_per_rank"],
            "four_memory_records": len(peaks) == self.reuse_world,
            "peak_reserved_below_limit": len(peaks) == self.reuse_world and all(
                float(value["peak_reserved_mib"]) < maximum_peak for value in peaks
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"forced D7 Reuse runtime canary failed: {checks}")
        atomic_json(marker, {
            "status": "medium_D7_forced_reuse_4gpu_runtime_canary_passed",
            "forced_d7_reuse_contract": str(self.forced_d7_reuse_path.relative_to(ROOT)),
            "forced_d7_reuse_contract_sha256": self.forced_d7_reuse_hash,
            "raw_sha256": payload["raw_sha256"], "requests": payload["requests"],
            "execution_runtime": execution, "checks": checks,
            "quality_interpretation": "prohibited_raw_only",
        })
        self.event("forced_d7_reuse_canary_pass", execution_runtime=execution)

    def forced_d7_summary(self, *, require_complete: bool) -> None:
        rows = []
        scope = self.forced_d7_reuse["scope"]
        for edge_name in scope["edges"]:
            edge = int(str(edge_name).split("_to_v", 1)[1])
            for horizon in map(int, scope["horizons_days"]):
                directory = self.reuse_dir("D7", edge, horizon, forced_diagnostic=True)
                report_path = directory / "adjudication.json"
                cell_seal_path = directory / "forced_diagnostic.seal.json"
                if not report_path.exists():
                    continue
                if not cell_seal_path.exists():
                    raise RuntimeError(f"forced D7 result lacks diagnostic seal: {directory}")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                summary = report["three_path_summary"]
                old, new, reuse = (
                    summary["old_parent"], summary["new_current"], summary["adjacent_one_hop_reuse"],
                )
                old_auc, new_auc, reuse_auc = old["ROC_AUC"], new["ROC_AUC"], reuse["ROC_AUC"]
                rows.append({
                    "edge": edge_name, "evaluation_days": horizon, "requests": summary["requests"],
                    "new_relative_to_old_AUC_percent": 100.0 * (new_auc - old_auc) / old_auc,
                    "reuse_relative_to_old_AUC_percent": 100.0 * (reuse_auc - old_auc) / old_auc,
                    "reuse_AUC_gain_retained_percent": summary["reuse_AUC_gain_retained_percent"],
                    "old_parent": old, "new_current": new, "adjacent_one_hop_reuse": reuse,
                    "raw_sha256": report["raw_sha256"],
                    "adjudication_sha256": sha256(report_path),
                    "diagnostic_seal_sha256": sha256(cell_seal_path),
                })
        expected = int(scope["expected_cells"])
        complete = len(rows) == expected
        if require_complete and not complete:
            raise RuntimeError(f"forced D7 Reuse matrix incomplete: {len(rows)}/{expected}")
        payload = {
            "status": "medium_D7_forced_reuse_diagnostic_complete" if complete else "medium_D7_forced_reuse_diagnostic_partial",
            "forced_d7_reuse_contract_sha256": self.forced_d7_reuse_hash,
            "matrix_contract_sha256": self.contract_hash,
            "formal_admission_and_lineage_unchanged": True,
            "qualification_interpretation": "prohibited_diagnostic_only",
            "expected_cells": expected, "completed_cells": len(rows), "rows": rows,
        }
        atomic_json(self.forced_d7_output / "summary.json", payload)
        lines = [
            "# Medium D7 forced adjacent-Reuse diagnostic", "",
            "This complete diagnostic bypasses execution locking only. It does not alter the sealed formal admission or serving/cache lineage.", "",
            f"Status: **{payload['status']}** ({len(rows)}/{expected}).", "",
            "Each result is computed from one sealed, aligned Old/New/Reuse three-path cohort.", "",
            "| Edge | E | Requests | New vs Old AUC | Reuse gain retained | Old loss | New loss | Reuse loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            recovery = row["reuse_AUC_gain_retained_percent"]
            recovery_text = "N/A" if recovery is None else f"{recovery:+.2f}%"
            lines.append(
                f"| {row['edge'].replace('_to_', ' → ')} | {row['evaluation_days']} | {row['requests']:,} | "
                f"{row['new_relative_to_old_AUC_percent']:+.2f}% | {recovery_text} | "
                f"{row['old_parent']['log_loss']:.6f} | {row['new_current']['log_loss']:.6f} | "
                f"{row['adjacent_one_hop_reuse']['log_loss']:.6f} |"
            )
        atomic_text(self.forced_d7_output / "summary.md", "\n".join(lines) + "\n")
        self.event("forced_d7_reuse_summary_written", completed_cells=len(rows), complete=complete)

    def run_forced_d7_reuse(self, acknowledgement: str | None) -> None:
        if acknowledgement != FORCED_D7_ACK:
            raise RuntimeError(f"forced D7 Reuse requires --acknowledge-long-run {FORCED_D7_ACK}")
        self.prepare()
        self.disk_preflight()
        self.gpu_preflight(physical_gpus=self.reuse_physical_gpus)
        self.write_forced_d7_state("forced_d7_reuse_started", completed_cells=0, expected_cells=20)
        self.forced_d7_reuse_canary()
        for edge_name in self.forced_d7_reuse["scope"]["edges"]:
            edge = int(str(edge_name).split("_to_v", 1)[1])
            for horizon in map(int, self.forced_d7_reuse["scope"]["horizons_days"]):
                self.evaluate_reuse("D7", edge, horizon, forced_diagnostic=True)
                self.forced_d7_summary(require_complete=False)
                completed = len(list(self.forced_d7_output.glob("E*/v*_to_v*/forced_diagnostic.seal.json")))
                self.write_forced_d7_state(
                    "forced_d7_reuse_running", completed_cells=completed, expected_cells=20,
                    last_completed={"edge": edge_name, "horizon_days": horizon},
                )
        self.forced_d7_summary(require_complete=True)
        self.write_forced_d7_state(
            "forced_d7_reuse_complete", completed_cells=20, expected_cells=20,
            summary=str((self.forced_d7_output / "summary.json").relative_to(ROOT)),
        )

    def smoke(self) -> None:
        marker = self.output / "smoke/smoke_complete.json"
        if marker.exists():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("execution_contract_sha256") != self.execution_hash:
                raise RuntimeError("smoke marker belongs to another execution contract")
            self.event("smoke_skip_valid")
            return
        self.prepare(); gpus = self.gpu_preflight()
        self.train("D7", 0, smoke=True); self.train("D7", 1, smoke=True)
        self.evaluate_full_only("D7", 1, 3, smoke=True); self.evaluate_reuse("D7", 1, 3, smoke=True)
        train_result = json.loads((self.checkpoint_dir("D7", 0, smoke=True) / "train_result.json").read_text())
        reuse_report = json.loads((self.reuse_dir("D7", 1, 3, smoke=True) / "adjudication.json").read_text())
        atomic_json(marker, {
            "status": "medium_gpu2_gpu3_training_full_only_and_reuse_smoke_passed",
            "contract_sha256": self.contract_hash, "execution_contract_sha256": self.execution_hash,
            "cpu_runtime_contract_sha256": self.cpu_runtime_hash,
            "reuse_runtime_contract_sha256": self.reuse_runtime_hash,
            "forced_d7_reuse_contract_sha256": self.forced_d7_reuse_hash,
            "world_size": self.world, "physical_gpus": self.physical_gpus,
            "remaining_reuse_world_size": self.reuse_world, "remaining_reuse_physical_gpus": self.reuse_physical_gpus,
            "global_batch_size": self.global_batch, "local_batch_sizes_by_rank": [16, 16],
            "gpu_preflight": gpus, "v0_canary_global_requests_per_second": train_result["global_requests_per_second"],
            "evaluation_requests": reuse_report["three_path_summary"]["requests"], "quality_interpretation": "none_smoke_only",
        })
        self.write_state("smoke_passed", smoke_marker=str(marker))

    def train_all(self) -> None:
        self.train("D7", 0)
        for branch in ("D7", "D14"):
            for version in range(1, int(self.contract["scope"]["branches"][branch]["updates"]) + 1):
                self.train(branch, version)
        self.write_state("all_checkpoints_complete")

    def evaluate_all(self) -> None:
        for branch in ("D7", "D14"):
            values, primary = self.contract["scope"]["branches"][branch], self.primary_horizon(branch)
            for edge in range(1, int(values["updates"]) + 1):
                self.evaluate_full_only(branch, edge, primary); self.seal_admission(branch, edge)
                for horizon in map(int, values["evaluation_days"]):
                    if horizon != primary:
                        self.evaluate_full_only(branch, edge, horizon)
                self.summarize(require_complete=False)
        self.reuse_runtime_canary()
        for branch in ("D7", "D14"):
            values = self.contract["scope"]["branches"][branch]
            for edge in range(1, int(values["updates"]) + 1):
                for horizon in map(int, values["evaluation_days"]):
                    self.evaluate_reuse(branch, edge, horizon); self.summarize(require_complete=False)
        self.write_state("all_evaluations_complete")

    def expected_cells(self) -> int:
        return sum(int(value["updates"]) * len(value["evaluation_days"]) for value in self.contract["scope"]["branches"].values())

    def summary_rows(self) -> list[dict]:
        rows = []
        for branch in ("D7", "D14"):
            values = self.contract["scope"]["branches"][branch]
            for edge in range(1, int(values["updates"]) + 1):
                admission_path = self.admission_path(branch, edge)
                admission = json.loads(admission_path.read_text()) if admission_path.exists() else None
                for horizon in map(int, values["evaluation_days"]):
                    full_path = self.full_only_dir(branch, edge, horizon) / "adjudication.json"
                    if not full_path.exists():
                        continue
                    full = json.loads(full_path.read_text())
                    parent = full["parent_absolute"]["hstu_native"]
                    candidate = full["candidates"][f"v{edge}"]
                    current = candidate["absolute"]["hstu_native"]
                    row = {
                        "recipe": branch, "edge": f"v{edge-1}_to_v{edge}", "evaluation_days": horizon,
                        "requests": candidate["paired_release_gain"]["parent_minus_current_log_loss"]["requests"],
                        "old_parent": parent, "new_current": current,
                        "admission": None if admission is None else {key: admission[key] for key in ("reuse_unlocked", "reason", "primary_horizon_days")},
                        "adjacent_one_hop_reuse": None, "reuse_AUC_gain_retained_percent": None, "reuse_log_loss_gain_retained_percent": None,
                    }
                    reuse_path = self.reuse_dir(branch, edge, horizon) / "adjudication.json"
                    if reuse_path.exists():
                        reuse = json.loads(reuse_path.read_text())["three_path_summary"]["adjacent_one_hop_reuse"]
                        auc_gain, loss_gain = current["ROC_AUC"] - parent["ROC_AUC"], parent["log_loss"] - current["log_loss"]
                        row["adjacent_one_hop_reuse"] = reuse
                        row["reuse_AUC_gain_retained_percent"] = None if auc_gain <= 0 else 100 * (reuse["ROC_AUC"] - parent["ROC_AUC"]) / auc_gain
                        row["reuse_log_loss_gain_retained_percent"] = None if loss_gain <= 0 else 100 * (parent["log_loss"] - reuse["log_loss"]) / loss_gain
                    rows.append(row)
        return rows

    def summarize(self, *, require_complete: bool) -> None:
        rows = self.summary_rows()
        full_complete = len(rows) == self.expected_cells()
        admissions_complete = sum(self.admission_path(branch, edge).exists() for branch in ("D7", "D14") for edge in range(1, int(self.contract["scope"]["branches"][branch]["updates"]) + 1)) == 14
        reuse_complete = full_complete and all(row["admission"] is not None and (not row["admission"]["reuse_unlocked"] or row["adjacent_one_hop_reuse"] is not None) for row in rows)
        complete = full_complete and admissions_complete and reuse_complete
        if require_complete and not complete:
            raise RuntimeError(f"matrix incomplete: full={len(rows)}/{self.expected_cells()}, admissions={admissions_complete}, reuse={reuse_complete}")
        payload = {
            "status": "medium_full_then_reuse_matrix_complete" if complete else "medium_full_then_reuse_matrix_partial",
            "contract_sha256": self.contract_hash, "execution_contract_sha256": self.execution_hash,
            "cpu_runtime_contract_sha256": self.cpu_runtime_hash,
            "world_size": self.world, "physical_gpus": self.physical_gpus,
            "expected_full_only_cells": self.expected_cells(), "completed_full_only_cells": len(rows),
            "all_admission_seals_complete": admissions_complete, "all_unlocked_reuse_cells_complete": reuse_complete,
            "recursive_reuse": False, "rows": rows,
        }
        atomic_json(self.output / "summary.json", payload)
        lines = [
            "# Medium D7/D14: Old Full / New Full / adjacent one-hop Reuse", "",
            "Training/Full-only used GPU2/3 world size 2. Completed Reuse artifacts are preserved; remaining D14 Reuse uses GPU0/1/2/3 world size 4 after its raw-only runtime canary.", "",
            f"Status: **{payload['status']}** ({len(rows)}/{payload['expected_full_only_cells']} Full-only cells).", "",
            "| Recipe | Edge | E | Requests | Old AUC | New AUC | Reuse AUC | AUC recovery | Old loss | New loss | Reuse loss | Loss recovery | Reuse status |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            reuse = row["adjacent_one_hop_reuse"]
            auc_recovery, loss_recovery = row["reuse_AUC_gain_retained_percent"], row["reuse_log_loss_gain_retained_percent"]
            status = "pending" if row["admission"] is None else ("unlocked" if row["admission"]["reuse_unlocked"] else "locked: " + row["admission"]["reason"])
            reuse_auc = "—" if reuse is None else f"{reuse['ROC_AUC']:.6f}"
            reuse_loss = "—" if reuse is None else f"{reuse['log_loss']:.6f}"
            auc_text = "—" if auc_recovery is None else f"{auc_recovery:+.1f}%"
            loss_text = "—" if loss_recovery is None else f"{loss_recovery:+.1f}%"
            lines.append(f"| {row['recipe']} | {row['edge'].replace('_to_', ' → ')} | {row['evaluation_days']} | {row['requests']:,} | {row['old_parent']['ROC_AUC']:.6f} | {row['new_current']['ROC_AUC']:.6f} | {reuse_auc} | {auc_text} | {row['old_parent']['log_loss']:.6f} | {row['new_current']['log_loss']:.6f} | {reuse_loss} | {loss_text} | {status} |")
        atomic_text(self.output / "summary.md", "\n".join(lines) + "\n")
        self.event("summary_written", completed_full_only_cells=len(rows), complete=complete)

    def plan(self) -> dict:
        tasks = ["prepare_manifest", "gpu2_gpu3_smoke_v0", "gpu2_gpu3_smoke_D7_v1", "smoke_D7_E3_full_only", "smoke_D7_E3_reuse_mechanics", "formal_shared_v0"]
        for branch in ("D7", "D14"):
            tasks.extend(f"formal_{branch}_v{version}" for version in range(1, int(self.contract["scope"]["branches"][branch]["updates"]) + 1))
        tasks.extend(["formal_all_32_full_only_cells", "seal_14_primary_horizon_release_eligibility_decisions", "formal_reuse_only_for_unlocked_accepted_lineage_edges", "summary"])
        return {
            "contract_sha256": self.contract_hash, "execution_contract_sha256": self.execution_hash,
            "cpu_runtime_contract_sha256": self.cpu_runtime_hash,
            "reuse_runtime_contract_sha256": self.reuse_runtime_hash,
            "forced_d7_reuse_contract_sha256": self.forced_d7_reuse_hash,
            "world_size": self.world, "physical_gpus": self.physical_gpus,
            "remaining_reuse_world_size": self.reuse_world,
            "remaining_reuse_physical_gpus": self.reuse_physical_gpus,
            "global_train_batch_size": self.global_batch, "local_batch_sizes_by_rank": [16, 16],
            "D14_cpu_runtime": self.cpu_runtime["runtime"],
            "D14_remaining_reuse_runtime": self.reuse_runtime["runtime"],
            "D7_forced_reuse_diagnostic_runtime": self.forced_d7_reuse["runtime"],
            "tasks": tasks, "formal_checkpoints": 15, "formal_full_only_cells": self.expected_cells(),
            "formal_reuse_cells_maximum": self.expected_cells(), "formal_acknowledgement": FORMAL_ACK,
            "forced_d7_reuse_cells": self.forced_d7_reuse["scope"]["expected_cells"],
            "forced_d7_reuse_acknowledgement": FORCED_D7_ACK,
        }

    def formal(self, acknowledgement: str | None) -> None:
        if acknowledgement != FORMAL_ACK:
            raise RuntimeError(f"formal run requires --acknowledge-long-run {FORMAL_ACK}")
        self.prepare(); self.disk_preflight(); self.smoke(); self.gpu_preflight()
        self.write_state("formal_started"); self.train_all()
        self.gpu_preflight(physical_gpus=self.reuse_physical_gpus)
        self.evaluate_all(); self.summarize(require_complete=True)
        self.write_state("formal_complete", summary=str(self.output / "summary.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(
        "plan", "prepare", "smoke", "reuse-canary", "d7-forced-reuse-canary",
        "d7-forced-reuse", "train", "evaluate", "summarize", "formal",
    ), default="plan")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--execution-contract", type=Path, default=EXECUTION)
    parser.add_argument("--cpu-runtime-contract", type=Path, default=CPU_RUNTIME)
    parser.add_argument("--reuse-runtime-contract", type=Path, default=REUSE_RUNTIME)
    parser.add_argument("--forced-d7-reuse-contract", type=Path, default=FORCED_D7_REUSE)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--acknowledge-long-run")
    args = parser.parse_args()
    pipeline = Pipeline(args.contract, execution_path=args.execution_contract,
                        cpu_runtime_path=args.cpu_runtime_contract,
                        reuse_runtime_path=args.reuse_runtime_contract,
                        forced_d7_reuse_path=args.forced_d7_reuse_contract,
                        threads=args.threads)
    if args.mode == "plan":
        print(json.dumps(pipeline.plan(), ensure_ascii=False, indent=2))
    elif args.mode == "prepare":
        pipeline.prepare()
    elif args.mode == "smoke":
        pipeline.smoke()
    elif args.mode == "reuse-canary":
        pipeline.prepare(); pipeline.disk_preflight()
        pipeline.gpu_preflight(physical_gpus=pipeline.reuse_physical_gpus)
        pipeline.reuse_runtime_canary()
    elif args.mode == "d7-forced-reuse-canary":
        pipeline.prepare(); pipeline.disk_preflight()
        pipeline.gpu_preflight(physical_gpus=pipeline.reuse_physical_gpus)
        pipeline.forced_d7_reuse_canary()
    elif args.mode == "d7-forced-reuse":
        pipeline.run_forced_d7_reuse(args.acknowledge_long_run)
    elif args.mode == "summarize":
        pipeline.summarize(require_complete=False)
    else:
        if args.acknowledge_long_run != FORMAL_ACK:
            raise RuntimeError(f"{args.mode} requires --acknowledge-long-run {FORMAL_ACK}")
        if args.mode == "formal":
            pipeline.formal(args.acknowledge_long_run)
        else:
            pipeline.prepare(); pipeline.disk_preflight(); pipeline.smoke()
            if args.mode == "train":
                pipeline.gpu_preflight()
                pipeline.train_all()
            else:
                pipeline.gpu_preflight(physical_gpus=pipeline.reuse_physical_gpus)
                pipeline.evaluate_all(); pipeline.summarize(require_complete=True)


if __name__ == "__main__":
    main()
