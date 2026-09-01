#!/usr/bin/env python3
"""Complete the Medium D14/E14 direct long-age Reuse triangle."""
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
CONTRACT = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_direct_long_age_reuse_v1.yaml"
REPORTING_AMENDMENT = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_direct_long_age_reuse_reporting_v2.yaml"
ACK = "RUN_MEDIUM_D14_DIRECT_LONG_AGE_REUSE"
EXPECTED_CELLS = (
    (0, 2),
    (0, 3), (1, 3),
    (0, 4), (1, 4), (2, 4),
    (0, 5), (1, 5), (2, 5), (3, 5),
)


def sha256(path: Path) -> str:
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


class Runner:
    def __init__(self, contract_path: Path) -> None:
        self.contract_path = contract_path.resolve()
        self.contract = yaml.safe_load(self.contract_path.read_text(encoding="utf-8"))
        self.contract_hash = sha256(self.contract_path)
        self.amendment_path = REPORTING_AMENDMENT.resolve()
        self.amendment = yaml.safe_load(self.amendment_path.read_text(encoding="utf-8"))
        self.amendment_hash = sha256(self.amendment_path)
        self.output = (ROOT / self.contract["outputs"]["root"]).resolve()
        self.logs = self.output / "logs"
        self.log_jsonl = self.logs / "pipeline.jsonl"
        self.state_path = self.output / "state.json"
        self.dataset = (ROOT / self.contract["frozen_execution"]["dataset_manifest"]["path"]).resolve()
        self.runtime = self.contract["runtime"]
        self.gpus = list(map(int, self.runtime["physical_gpus"]))
        self.world = int(self.runtime["world_size"])
        self._verify_contract_shape()

    def _verify_contract_shape(self) -> None:
        scope = self.contract["scope"]
        observed = tuple(
            (int(row["producer"][1:]), int(row["current"][1:]))
            for row in scope["direct_long_age_cells"]
        )
        if observed != EXPECTED_CELLS:
            raise RuntimeError(f"direct long-age cell list drifted: {observed}")
        if int(scope["expected_new_cells"]) != len(EXPECTED_CELLS):
            raise RuntimeError("expected new-cell count drifted")
        if int(scope["expected_complete_triangle_cells_including_adjacent"]) != 15:
            raise RuntimeError("complete triangle must contain fifteen cells")
        if scope["display_horizon"] != "E14" or int(scope["requested_evaluation_days"]) != 14:
            raise RuntimeError("runner is frozen to the unified E14 reporting horizon")
        for row in scope["direct_long_age_cells"]:
            current = int(row["current"][1:])
            cutover = 217 + 14 * current
            if int(row["cutover_day"]) != cutover:
                raise RuntimeError(f"cutover drifted for {row}")
            if list(map(int, row["day_range_half_open"])) != [cutover, cutover + 14]:
                raise RuntimeError(f"E14 range drifted for {row}")
        if self.world != 4 or self.gpus != [0, 1, 2, 3]:
            raise RuntimeError("Medium long-age Reuse requires one four-rank GPU0/1/2/3 job")
        if int(self.runtime["cohort_size_per_rank"]) != 32:
            raise RuntimeError("cohort size must remain at the proven safe value 32/rank")
        if int(self.runtime["query_chunk_size_per_rank"]) != 256:
            raise RuntimeError("query chunk must remain at the proven safe value 256/rank")
        amendment = self.amendment
        if amendment["frozen_parent"]["contract_sha256"] != self.contract_hash:
            raise RuntimeError("reporting amendment does not bind the frozen execution contract")
        if (ROOT / amendment["frozen_parent"]["contract"]).resolve() != self.contract_path:
            raise RuntimeError("reporting amendment points at another execution contract")
        timing = amendment["timing_and_scope"]
        if int(timing["adopted_after_completed_new_cells"]) != 1 or int(timing["remaining_cells_must_all_run"]) != 9:
            raise RuntimeError("reporting amendment must preserve the one-complete/nine-remaining boundary")
        policy = amendment["reporting_policy"]
        if policy["primary_comparison"] != "within_run_current_exact_vs_direct_reuse":
            raise RuntimeError("reporting amendment must use within-run New versus Reuse")
        if policy["old_path_in_new_cells"] != "not_computed":
            raise RuntimeError("new direct cells must remain two-path New/Reuse evaluations")
        if policy["cross_run_current_equality_gate"] != "prohibited":
            raise RuntimeError("cross-run Current equality must not gate execution")
        cpu_sets = [
            set(map(int, self.runtime[f"rank{rank}_cpu_affinity"]))
            for rank in range(self.world)
        ]
        if any(len(values) != 14 for values in cpu_sets) or len(set().union(*cpu_sets)) != 56:
            raise RuntimeError("four ranks require 56 disjoint physical CPU cores")

    def event(self, event: str, **values: object) -> None:
        self.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.log_jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": utc_now(), "event": event, **values}, ensure_ascii=False, sort_keys=True) + "\n")

    def state(self, status: str, **values: object) -> None:
        atomic_json(self.state_path, {
            "status": status,
            "updated_at_utc": utc_now(),
            "contract": str(self.contract_path.relative_to(ROOT)),
            "contract_sha256": self.contract_hash,
            "reporting_amendment": str(self.amendment_path.relative_to(ROOT)),
            "reporting_amendment_sha256": self.amendment_hash,
            "world_size": self.world,
            "physical_gpus": self.gpus,
            **values,
        })

    def run(self, name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        log = self.logs / f"{name}.log"
        retry = 1
        while log.exists():
            log = self.logs / f"{name}.retry{retry}.log"
            retry += 1
        self.event("step_start", name=name, command=command, log=str(log.relative_to(ROOT)))
        print("+", " ".join(map(str, command)), flush=True)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                stream.write(line)
            returncode = process.wait()
        self.event("step_end", name=name, returncode=returncode)
        if returncode:
            self.state("failed", failed_step=name, returncode=returncode)
            raise subprocess.CalledProcessError(returncode, command)

    @property
    def gpu_env(self) -> dict[str, str]:
        return {
            **os.environ,
            "PYTHONPATH": "src",
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus)),
            "OMP_NUM_THREADS": str(self.runtime["omp_num_threads_per_rank"]),
            "PYTHONUNBUFFERED": "1",
        }

    @property
    def distributed_prefix(self) -> list[str]:
        return ["torchrun", "--standalone", f"--nproc_per_node={self.world}"]

    def cpu_args(self) -> list[str]:
        affinity = ";".join(
            ",".join(map(str, self.runtime[f"rank{rank}_cpu_affinity"]))
            for rank in range(self.world)
        )
        return [
            "--history-threads", str(self.runtime["history_threads_per_rank"]),
            "--arrow-cpu-threads", str(self.runtime["arrow_cpu_threads_per_rank"]),
            "--arrow-io-threads", str(self.runtime["arrow_io_threads_per_rank"]),
            "--torch-cpu-threads", str(self.runtime["torch_cpu_threads_per_rank"]),
            "--cpu-affinity-by-rank", affinity,
        ]

    def checkpoint(self, version: int) -> Path:
        return (ROOT / self.contract["checkpoints"][f"v{version}"]["path"]).resolve()

    def manifest_record(self, current: int) -> dict:
        key = "v5" if current == 5 else "base"
        return self.contract["frozen_execution"]["manifests"][key]

    def manifest_dir(self, current: int) -> Path:
        return (ROOT / self.manifest_record(current)["directory"]).resolve()

    def adjacent_report(self, current: int) -> Path:
        key = f"v{current - 1}_to_v{current}"
        return (ROOT / self.contract["adjacent_evidence"][key]["path"]).resolve()

    def cell_record(self, producer: int, current: int) -> dict:
        for row in self.contract["scope"]["direct_long_age_cells"]:
            if row["producer"] == f"v{producer}" and row["current"] == f"v{current}":
                return row
        raise KeyError((producer, current))

    def cell_dir(self, producer: int, current: int, *, canary: bool = False) -> Path:
        root = self.output / "canary" if canary else self.output
        return root / "E14" / f"v{producer}_to_v{current}"

    def verify_inputs(self) -> None:
        records: list[tuple[Path, str, str]] = []
        for name, record in self.contract["frozen_parents"].items():
            records.append(((ROOT / record["path"]).resolve(), record["sha256"], name))
        for name in ("executor", "adjudicator", "dataset_manifest"):
            record = self.contract["frozen_execution"][name]
            records.append(((ROOT / record["path"]).resolve(), record["sha256"], name))
        for name, record in self.contract["frozen_execution"]["manifests"].items():
            directory = (ROOT / record["directory"]).resolve()
            records.extend([
                (directory / "manifest.json", record["manifest_sha256"], f"{name}_manifest"),
                (directory / "requests_fidelity.parquet", record["requests_fidelity_sha256"], f"{name}_fidelity"),
                (directory / "requests_quality.parquet", record["requests_quality_sha256"], f"{name}_quality"),
            ])
        for name, record in self.contract["checkpoints"].items():
            records.append(((ROOT / record["path"]).resolve(), record["sha256"], name))
        for name, record in self.contract["adjacent_evidence"].items():
            records.append(((ROOT / record["path"]).resolve(), record["sha256"], f"adjacent_{name}"))
        for path, expected, name in records:
            if not path.is_file():
                raise FileNotFoundError(path)
            observed = sha256(path)
            if observed != expected:
                raise RuntimeError(f"frozen input hash mismatch for {name}: {observed} != {expected}")
        first_cell = self.amendment["completed_before_amendment"]
        for name, relative in (
            ("raw", "raw.parquet"),
            ("raw_seal", "raw.seal.json"),
            ("adjudication", "adjudication.json"),
            ("cell_seal", "cell.seal.json"),
        ):
            path = self.cell_dir(0, 2) / relative
            if not path.is_file() or sha256(path) != first_cell[f"{name}_sha256"]:
                raise RuntimeError(f"completed v0_to_v2 evidence changed: {name}")
        canary_marker = self.output / "canary" / "canary.pass.json"
        if not canary_marker.is_file() or sha256(canary_marker) != first_cell["canary_pass_sha256"]:
            raise RuntimeError("pre-amendment canary evidence changed")
        for current in range(1, 6):
            report = json.loads(self.adjacent_report(current).read_text(encoding="utf-8"))
            if report.get("edge") != f"v{current - 1}_to_v{current}":
                raise RuntimeError(f"adjacent report edge drifted for current v{current}")
            expected_range = [217 + 14 * current, 217 + 14 * current + 14]
            if list(map(int, report.get("evaluation_day_range", []))) != expected_range:
                raise RuntimeError(f"adjacent E14 range drifted for current v{current}")
        self.event("frozen_inputs_verified", files=len(records))

    def disk_preflight(self) -> None:
        free_gib = shutil.disk_usage(ROOT).free / 2**30
        if free_gib < 30:
            raise RuntimeError(f"workspace has only {free_gib:.1f} GiB free; 30 GiB required")
        self.event("disk_preflight_pass", free_gib=free_gib)

    def gpu_preflight(self) -> list[dict]:
        output = subprocess.check_output([
            "nvidia-smi",
            f"--id={','.join(map(str, self.gpus))}",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ], text=True)
        rows = []
        for line in output.strip().splitlines():
            index, name, total, free = [part.strip() for part in line.split(",", 3)]
            if int(index) in self.gpus:
                rows.append({
                    "index": int(index), "name": name,
                    "memory_total_mib": int(total), "memory_free_mib": int(free),
                })
        rows.sort(key=lambda row: row["index"])
        if [row["index"] for row in rows] != self.gpus:
            raise RuntimeError(f"required GPU allowlist is unavailable: {rows}")
        deficient = [row for row in rows if row["memory_free_mib"] < 40000]
        if deficient:
            raise RuntimeError(f"four-GPU preflight requires 40000 MiB free per GPU: {deficient}")
        self.event("gpu_preflight_pass", gpus=rows)
        return rows

    @staticmethod
    def validate_raw(directory: Path, *, expected_edge: str, expected_range: list[int]) -> dict:
        raw, seal = directory / "raw.parquet", directory / "raw.seal.json"
        if not raw.is_file() or not seal.is_file():
            raise RuntimeError(f"sealed raw artifacts are incomplete: {directory}")
        payload = json.loads(seal.read_text(encoding="utf-8"))
        checks = {
            "raw_hash": sha256(raw) == payload.get("raw_sha256"),
            "edge": payload.get("edge") == expected_edge,
            "range": list(map(int, payload.get("evaluation_day_range", []))) == expected_range,
            "two_paths": payload.get("rows") == 2 * payload.get("requests", -1),
            "no_parent_exact": payload.get("contains_parent_exact_rolling") is False,
            "non_recursive": payload.get("recursive_reuse") is False,
        }
        if not all(checks.values()):
            raise RuntimeError(f"raw seal validation failed for {directory}: {checks}")
        return payload

    def evaluate_raw(self, producer: int, current: int, *, canary: bool) -> Path:
        record = self.cell_record(producer, current)
        day_range = list(map(int, record["day_range_half_open"]))
        directory = self.cell_dir(producer, current, canary=canary)
        raw, seal = directory / "raw.parquet", directory / "raw.seal.json"
        if directory.exists() and not (raw.is_file() and seal.is_file()):
            raise RuntimeError(f"interrupted unsealed directory requires audit: {directory}")
        if not directory.exists():
            manifest = self.manifest_dir(current)
            command = [
                *self.distributed_prefix,
                "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
                "--stage", f"medium_D14_E14_direct_long_age_v{producer}_to_v{current}{'_canary' if canary else ''}",
                "--edge", f"v{producer}_to_v{current}",
                "--cutover-day", str(record["cutover_day"]),
                "--start-day", str(day_range[0]),
                "--end-day", str(day_range[1]),
                "--manifest-dir", str(manifest),
                "--dataset-manifest", str(self.dataset),
                "--parent", str(self.checkpoint(producer)),
                "--current", str(self.checkpoint(current)),
                "--output", str(directory),
                "--cohort-size", str(self.runtime["cohort_size_per_rank"]),
                "--query-chunk-size", str(self.runtime["query_chunk_size_per_rank"]),
                *self.cpu_args(),
            ]
            if canary:
                command.extend(["--max-users", str(self.contract["canary"]["maximum_users_per_rank"] )])
            self.run(
                f"{'canary_' if canary else ''}E14_v{producer}_to_v{current}_raw",
                command,
                env=self.gpu_env,
            )
        self.validate_raw(
            directory,
            expected_edge=f"v{producer}_to_v{current}",
            expected_range=day_range,
        )
        return directory

    def adjudicate(self, producer: int, current: int) -> Path:
        directory = self.evaluate_raw(producer, current, canary=False)
        raw, seal = directory / "raw.parquet", directory / "raw.seal.json"
        report, cell_seal = directory / "adjudication.json", directory / "cell.seal.json"
        if not report.exists():
            manifest = self.manifest_dir(current)
            self.run(f"E14_v{producer}_to_v{current}_adjudicate", [
                sys.executable,
                "scripts/adjudicate_yambda500m_hstu_native_onehop_reuse.py",
                "--raw", str(raw),
                "--seal", str(seal),
                "--labels", str(manifest / "requests_quality.parquet"),
                "--output", str(report),
            ], env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"})
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("raw_sha256") != json.loads(seal.read_text(encoding="utf-8")).get("raw_sha256"):
            raise RuntimeError(f"adjudication/raw mismatch: {directory}")
        expected = {
            "contract_sha256": self.contract_hash,
            "edge": f"v{producer}_to_v{current}",
            "display_horizon": "E14",
            "day_range_half_open": list(map(int, self.cell_record(producer, current)["day_range_half_open"])),
            "raw_sha256": sha256(raw),
            "raw_seal_sha256": sha256(seal),
            "adjudication_sha256": sha256(report),
        }
        if cell_seal.exists():
            observed = json.loads(cell_seal.read_text(encoding="utf-8"))
            if any(observed.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"cell seal mismatch: {directory}")
        else:
            atomic_json(cell_seal, {
                "status": "medium_D14_E14_direct_long_age_reuse_cell_sealed",
                **expected,
                "reporting_amendment_sha256": self.amendment_hash,
                "producer_materializes_complete_pre_cutover_prefix": True,
                "post_cutover_appends_by_current": True,
                "recursive_reuse": False,
                "serving_lineage_promotion": False,
            })
        return report

    def canary(self) -> None:
        marker = self.output / "canary" / "canary.pass.json"
        if marker.exists():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            directory = self.cell_dir(0, 5, canary=True)
            raw = directory / "raw.parquet"
            if payload.get("contract_sha256") != self.contract_hash:
                raise RuntimeError("canary marker belongs to another contract")
            if not raw.exists() or payload.get("raw_sha256") != sha256(raw):
                raise RuntimeError("canary raw differs from its pass marker")
            self.event("canary_skip_valid")
            return
        directory = self.evaluate_raw(0, 5, canary=True)
        payload = self.validate_raw(directory, expected_edge="v0_to_v5", expected_range=[287, 301])
        execution = payload.get("execution_runtime", {})
        peaks = execution.get("peak_memory_by_rank", [])
        limit = float(self.contract["canary"]["maximum_peak_reserved_mib_per_rank"])
        checks = {
            "world_size": execution.get("world_size") == self.world,
            "cohort_size": execution.get("cohort_size_per_rank") == self.runtime["cohort_size_per_rank"],
            "query_chunk_size": execution.get("query_chunk_size_per_rank") == self.runtime["query_chunk_size_per_rank"],
            "four_peak_records": len(peaks) == self.world,
            "peaks_below_limit": len(peaks) == self.world and all(float(row["peak_reserved_mib"]) < limit for row in peaks),
        }
        if not all(checks.values()):
            raise RuntimeError(f"direct long-age canary failed: {checks}")
        atomic_json(marker, {
            "status": "medium_D14_E14_direct_long_age_four_gpu_canary_passed",
            "contract_sha256": self.contract_hash,
            "cell": "v0_to_v5",
            "raw_sha256": payload["raw_sha256"],
            "requests": payload["requests"],
            "execution_runtime": execution,
            "checks": checks,
            "quality_metric_read": False,
        })
        self.event("canary_pass", checks=checks)

    @staticmethod
    def comparison_row(*, producer: int, current: int, source: str, requests: int,
                       new: dict, reuse: dict, day_range: list[int],
                       historical_new: dict) -> dict:
        return {
            "producer": f"v{producer}",
            "current": f"v{current}",
            "version_gap": current - producer,
            "source": source,
            "display_horizon": "E14",
            "day_range_half_open": day_range,
            "requests": requests,
            "current_minus_reuse_ROC_AUC_pp": 100.0 * (new["ROC_AUC"] - reuse["ROC_AUC"]),
            "reuse_AUC_change_vs_current_percent": 100.0 * (reuse["ROC_AUC"] - new["ROC_AUC"]) / new["ROC_AUC"],
            "reuse_minus_current_log_loss": reuse["log_loss"] - new["log_loss"],
            "reuse_log_loss_change_vs_current_percent": 100.0 * (reuse["log_loss"] - new["log_loss"]) / new["log_loss"],
            "current_minus_historical_current_ROC_AUC_pp": 100.0 * (new["ROC_AUC"] - historical_new["ROC_AUC"]),
            "current_minus_historical_current_log_loss": new["log_loss"] - historical_new["log_loss"],
            "new_current": {"ROC_AUC": new["ROC_AUC"], "log_loss": new["log_loss"]},
            "reuse": {"ROC_AUC": reuse["ROC_AUC"], "log_loss": reuse["log_loss"]},
            "historical_current_reference": {
                "ROC_AUC": historical_new["ROC_AUC"],
                "log_loss": historical_new["log_loss"],
            },
        }

    def summary_rows(self) -> list[dict]:
        rows = []
        for current in range(1, 6):
            adjacent_payload = json.loads(self.adjacent_report(current).read_text(encoding="utf-8"))
            adjacent = adjacent_payload["three_path_summary"]
            canonical_new = adjacent["new_current"]
            day_range = list(map(int, adjacent_payload["evaluation_day_range"]))
            for producer in range(current):
                if producer == current - 1:
                    reuse = adjacent["adjacent_one_hop_reuse"]
                    requests = int(adjacent["requests"])
                    source = "frozen_adjacent"
                    new = canonical_new
                else:
                    report = self.cell_dir(producer, current) / "adjudication.json"
                    if not report.exists():
                        continue
                    direct = json.loads(report.read_text(encoding="utf-8"))
                    new = direct["absolute_metrics"]["current_exact_rolling"]
                    reuse = direct["absolute_metrics"]["one_hop_reuse_rolling"]
                    requests = int(direct["reuse_minus_recompute"]["paired_harm"]["requests"])
                    source = "new_direct_long_age"
                    if requests != int(adjacent["requests"]):
                        raise RuntimeError(f"request cohort differs from adjacent E14 for v{producer}_to_v{current}")
                rows.append(self.comparison_row(
                    producer=producer,
                    current=current,
                    source=source,
                    requests=requests,
                    new=new,
                    reuse=reuse,
                    day_range=day_range,
                    historical_new=canonical_new,
                ))
        rows.sort(key=lambda row: (int(row["current"][1:]), int(row["producer"][1:])))
        return rows

    def summarize(self, *, require_complete: bool) -> dict:
        rows = self.summary_rows()
        completed_new = sum(row["source"] == "new_direct_long_age" for row in rows)
        complete = completed_new == len(EXPECTED_CELLS) and len(rows) == 15
        if require_complete and not complete:
            raise RuntimeError(f"direct long-age triangle incomplete: new={completed_new}/10, total={len(rows)}/15")
        payload = {
            "status": "medium_D14_E14_direct_long_age_triangle_complete" if complete else "medium_D14_E14_direct_long_age_triangle_in_progress",
            "contract": str(self.contract_path.relative_to(ROOT)),
            "contract_sha256": self.contract_hash,
            "reporting_amendment": str(self.amendment_path.relative_to(ROOT)),
            "reporting_amendment_sha256": self.amendment_hash,
            "display_horizon": "E14",
            "primary_comparison": "within_run_current_exact_vs_direct_reuse",
            "old_path_in_new_cells": "not_computed",
            "cross_run_current_drift_is_recorded_not_gating": True,
            "expected_new_cells": 10,
            "completed_new_cells": completed_new,
            "expected_triangle_cells": 15,
            "completed_triangle_cells": len(rows),
            "direct_non_recursive_reuse": True,
            "all_predeclared_cells_reported": complete,
            "rows": rows,
        }
        atomic_json(self.output / "summary.json", payload)
        lines = [
            "# Medium D14/E14 direct long-age Reuse triangle",
            "",
            f"Status: **{payload['status']}**. New non-adjacent cells: {completed_new}/10; full triangle including frozen adjacent cells: {len(rows)}/15.",
            "",
            "Every row is reported as E14. Each comparison uses only New and Reuse produced in the same run. Old is not recomputed for new cells; cross-run Current drift is recorded in JSON and never gates execution.",
            "",
            "| Current | KV producer | Gap | Source | Requests | New AUC | Reuse AUC | New − Reuse AUC (pp) | Reuse AUC vs New | New loss | Reuse loss | Reuse loss vs New |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['current']} | {row['producer']} | {row['version_gap']} | {row['source']} | {row['requests']:,} | "
                f"{row['new_current']['ROC_AUC']:.6f} | {row['reuse']['ROC_AUC']:.6f} | "
                f"{row['current_minus_reuse_ROC_AUC_pp']:+.4f} | {row['reuse_AUC_change_vs_current_percent']:+.4f}% | "
                f"{row['new_current']['log_loss']:.6f} | {row['reuse']['log_loss']:.6f} | "
                f"{row['reuse_log_loss_change_vs_current_percent']:+.4f}% |"
            )
        atomic_text(self.output / "summary.md", "\n".join(lines) + "\n")
        self.event("summary_written", completed_new_cells=completed_new, completed_triangle_cells=len(rows), complete=complete)
        return payload

    def plan(self) -> dict:
        return {
            "contract": str(self.contract_path.relative_to(ROOT)),
            "contract_sha256": self.contract_hash,
            "reporting_amendment": str(self.amendment_path.relative_to(ROOT)),
            "reporting_amendment_sha256": self.amendment_hash,
            "display_horizon": "E14",
            "world_size": self.world,
            "physical_gpus": self.gpus,
            "runtime": self.runtime,
            "new_cells": [f"v{producer}_to_v{current}" for producer, current in EXPECTED_CELLS],
            "new_cell_count": len(EXPECTED_CELLS),
            "final_triangle_cell_count": 15,
            "formal_acknowledgement": ACK,
        }

    def formal(self, acknowledgement: str | None) -> None:
        if acknowledgement != ACK:
            raise RuntimeError(f"formal queue requires --acknowledge-long-run {ACK}")
        self.verify_inputs()
        self.disk_preflight()
        self.gpu_preflight()
        self.state("canary_started", completed_new_cells=0, expected_new_cells=10)
        self.canary()
        completed = sum((self.cell_dir(p, c) / "cell.seal.json").exists() for p, c in EXPECTED_CELLS)
        self.state("formal_started", completed_new_cells=completed, expected_new_cells=10)
        for producer, current in EXPECTED_CELLS:
            self.adjudicate(producer, current)
            payload = self.summarize(require_complete=False)
            self.state(
                "formal_running",
                completed_new_cells=payload["completed_new_cells"],
                expected_new_cells=10,
                last_completed=f"v{producer}_to_v{current}",
            )
        payload = self.summarize(require_complete=True)
        self.state(
            "complete",
            completed_new_cells=payload["completed_new_cells"],
            expected_new_cells=10,
            summary=str((self.output / "summary.json").relative_to(ROOT)),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "preflight", "canary", "formal", "summarize"), default="plan")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--acknowledge-long-run")
    args = parser.parse_args()
    runner = Runner(args.contract)
    if args.mode == "plan":
        print(json.dumps(runner.plan(), ensure_ascii=False, indent=2))
    elif args.mode == "preflight":
        runner.verify_inputs()
        runner.disk_preflight()
        runner.summarize(require_complete=False)
        print(json.dumps({"status": "preflight_passed", "contract_sha256": runner.contract_hash}, indent=2))
    elif args.mode == "canary":
        runner.verify_inputs()
        runner.disk_preflight()
        runner.gpu_preflight()
        runner.canary()
    elif args.mode == "summarize":
        runner.verify_inputs()
        print(json.dumps(runner.summarize(require_complete=False), ensure_ascii=False, indent=2))
    else:
        runner.formal(args.acknowledge_long_run)


if __name__ == "__main__":
    main()
