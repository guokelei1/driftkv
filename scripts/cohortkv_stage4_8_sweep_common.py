from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from pathlib import Path

import run_cohortkv_stage4_7_organic_chain as base
import torch
from cohortkv_stage4_7_common import (
    CHECKPOINT_DIR,
    COMPILER_OUTPUT,
    PREPARED_PATH,
    RUNTIME_DIR,
    TRAINING_PATH,
    load_inputs,
    sha256,
)
from evaluate_cohortkv_stage4_6_lifecycle import (
    LAUNCH,
    exact_batch,
    execute_direct,
    timed_cuda,
)
from motivation_validity import seed_everything
from run_cohortkv_stage4_6_full_chain import task_metrics

from hstu_kvcache.migration import JaggedMigratedKVBatch, append_jagged_suffix
from hstu_kvcache.migration.organic_schedulers import (
    SchedulerRecord,
    select_aoi_maxweight,
    select_model_time_staggered_renewal,
    select_total_token_cumulative_debt,
    select_work_balanced_staggered_renewal,
)
from hstu_kvcache.migration.stage45_oldkv import DirectOldKVFusedOperator
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_organic_windows,
)
from hstu_kvcache.utils import save_json

PROTOCOL = "cohortkv_single_config_stage4_8_scheduler_sweep_v1"
SWEEP_PROTOCOL = "cohortkv_single_config_stage4_8_four_gpu_sweep_v1"
BASELINE_PROTOCOL = (
    "cohortkv_single_config_stage4_8_external_exact_baseline_v1"
)
BASELINE_PATH = (
    "configs/cohortkv_single_config_v1/stage4_8_exact_baseline.json"
)
OUTPUT_DIR = "results/system/cohortkv_single_config_full_chain_v1"
LOG_DIR = "logs/cohortkv_stage4_8_scheduler_sweeps"
TASK_METRICS = ("catalog_auc", "ndcg_at_100", "hit_at_100")
NUM_EDGES = 11
BATCH_SIZE = 4
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BASELINE_SHA256 = (
    "29c0c3d6a6cee5521fe52d0d65ba4a96f739a58c7bf80b52858ef8ee781a6b51"
)
LAUNCHER_PATHS = {
    "staggered_renewal": (
        ROOT / "scripts/run_cohortkv_stage4_8_staggered_renewal_sweep.py"
    ),
    "token_debt": ROOT / "scripts/run_cohortkv_stage4_8_token_debt_sweep.py",
    "aoi_maxweight": (
        ROOT / "scripts/run_cohortkv_stage4_8_aoi_maxweight_sweep.py"
    ),
    "model_time_renewal": (
        ROOT / "scripts/run_cohortkv_stage4_8_model_time_renewal_sweep.py"
    ),
}
IMPLEMENTATION_PATHS = {
    "stage4_8_worker": ROOT / "scripts/cohortkv_stage4_8_sweep_common.py",
    "organic_schedulers": (
        ROOT / "src/hstu_kvcache/migration/organic_schedulers.py"
    ),
    "stage4_7_common": ROOT / "scripts/cohortkv_stage4_7_common.py",
    "stage4_7_organic_chain": (
        ROOT / "scripts/run_cohortkv_stage4_7_organic_chain.py"
    ),
    "stage4_6_lifecycle": (
        ROOT / "scripts/evaluate_cohortkv_stage4_6_lifecycle.py"
    ),
    "stage4_6_full_chain": (
        ROOT / "scripts/run_cohortkv_stage4_6_full_chain.py"
    ),
    "direct_oldkv_operator": (
        ROOT / "src/hstu_kvcache/migration/stage45_oldkv.py"
    ),
    "organic_migration": ROOT / "src/hstu_kvcache/migration/organic.py",
    "organic_streaming": ROOT / "src/hstu_kvcache/streaming/organic.py",
}


@dataclass(frozen=True)
class VariantSpec:
    scheme: str
    grid_index: int
    label: str
    parameter_name: str
    parameter: float

    def to_dict(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "grid_index": self.grid_index,
            "label": self.label,
            self.parameter_name: self.parameter,
        }


SCHEME_GRIDS = {
    "staggered_renewal": (
        ("h08", "renewal_horizon", 8.0),
        ("h10", "renewal_horizon", 10.0),
        ("h12", "renewal_horizon", 12.0),
        ("h16", "renewal_horizon", 16.0),
    ),
    "token_debt": (
        ("total10", "total_exact_token_fraction", 0.10),
        ("total12", "total_exact_token_fraction", 0.12),
        ("total14", "total_exact_token_fraction", 0.14),
        ("total16", "total_exact_token_fraction", 0.16),
    ),
    "aoi_maxweight": (
        ("budget04", "scheduled_token_fraction", 0.04),
        ("budget07", "scheduled_token_fraction", 0.07),
        ("budget10", "scheduled_token_fraction", 0.10),
        ("budget13", "scheduled_token_fraction", 0.13),
    ),
    "model_time_renewal": (
        ("h08", "renewal_horizon", 8.0),
        ("h10", "renewal_horizon", 10.0),
        ("h12", "renewal_horizon", 12.0),
        ("h16", "renewal_horizon", 16.0),
    ),
}


def variant_specs(scheme: str) -> tuple[VariantSpec, ...]:
    try:
        grid = SCHEME_GRIDS[scheme]
    except KeyError as exc:
        raise ValueError(f"unknown Stage 4.8 scheme: {scheme}") from exc
    return tuple(
        VariantSpec(
            scheme=scheme,
            grid_index=index,
            label=label,
            parameter_name=name,
            parameter=value,
        )
        for index, (label, name, value) in enumerate(grid)
    )


def parse_args(scheme: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-index", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--devices",
        nargs=4,
        default=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    )
    parser.add_argument("--prepared-data", default=PREPARED_PATH)
    parser.add_argument("--training-result", default=TRAINING_PATH)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--compiler-result", default=COMPILER_OUTPUT)
    parser.add_argument("--runtime-dir", default=RUNTIME_DIR)
    parser.add_argument("--baseline", default=BASELINE_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--runtime-smoke-test", action="store_true")
    args = parser.parse_args()
    args.scheme = scheme
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke_test and args.runtime_smoke_test:
        raise ValueError("choose one Stage 4.8 smoke mode")
    if args.seed != 0 or args.batch_size != BATCH_SIZE:
        raise ValueError("Stage 4.8 freezes seed 0 and batch size 4")
    specs = variant_specs(args.scheme)
    if len(specs) != 4 or tuple(value.grid_index for value in specs) != tuple(
        range(4)
    ):
        raise ValueError("Stage 4.8 scheme grid must contain four ordered points")
    if args.grid_index is not None:
        if not 0 <= args.grid_index < 4 or args.device is None:
            raise ValueError("Stage 4.8 child requires grid index 0..3 and device")
        if not args.smoke_test:
            child_device = torch.device(args.device)
            if (
                child_device.type != "cuda"
                or child_device.index is None
                or child_device.index >= torch.cuda.device_count()
            ):
                raise ValueError(
                    "Stage 4.8 worker requires an available explicit CUDA index"
                )
    elif args.device is not None:
        raise ValueError("--device is reserved for a child worker")
    if not args.smoke_test and args.grid_index is None:
        devices = tuple(torch.device(value) for value in args.devices)
        indices = tuple(value.index for value in devices)
        if (
            any(
                value.type != "cuda"
                or value.index is None
                or value.index >= torch.cuda.device_count()
                for value in devices
            )
            or len(set(indices)) != 4
        ):
            raise ValueError(
                "Stage 4.8 launcher requires four distinct available "
                "explicit CUDA indices"
            )
    elif len(set(args.devices)) != 4:
        raise ValueError("Stage 4.8 launcher requires four distinct devices")
    if args.runtime_smoke_test and args.grid_index is None:
        raise ValueError("runtime smoke requires one grid index and device")


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def implementation_snapshot(scheme: str) -> dict[str, dict[str, str]]:
    paths = {
        **IMPLEMENTATION_PATHS,
        "scheme_launcher": LAUNCHER_PATHS[scheme],
    }
    return {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for name, path in paths.items()
    }


def _expected_exact_edges(chain: dict) -> list[dict]:
    return [
        {
            "source_version": step["source_version"],
            "target_version": step["target_version"],
            "source_model": f"theta{step['source_version']}",
            "target_model": f"theta{step['target_version']}",
            "target_date": step["prediction_target_date"],
            "exact_reference_records": (
                step["actions"]["migrate"]
                + step["actions"]["scheduled_selector_exact"]
                + step["actions"]["natural_no_reuse_target_exact"]
            ),
            "all_exact_reference_ms": step["cost"][
                "all_exact_reference_ms"
            ],
        }
        for step in chain["steps"]
    ]


def _expected_exact_endpoints(chain: dict) -> list[dict]:
    output = []
    for endpoint in chain["endpoints"]:
        task = endpoint["task_metrics"]
        exact = task["all_exact"]
        output.append(
            {
                "version": endpoint["version"],
                "target_date": endpoint["target_date"],
                "records": task["records"],
                "catalog_auc": exact["catalog_auc"],
                "ndcg_at_100": exact["ndcg@100"],
                "hit_at_100": exact["hit@100"],
            }
        )
    return output


def _validate_baseline_derivation(baseline: dict) -> None:
    artifacts = baseline["source_artifacts"]
    chain_path = _repo_path(artifacts["stage4_7_chain"]["path"])
    summary_path = _repo_path(artifacts["stage4_7_summary"]["path"])
    chain = json.loads(chain_path.read_text())
    summary = json.loads(summary_path.read_text())
    checks = {
        "chain_protocol": chain["protocol"]
        == artifacts["stage4_7_chain"]["protocol"],
        "chain_experiment_protocol": chain["experiment_protocol"]
        == artifacts["stage4_7_chain"]["experiment_protocol"],
        "chain_status": chain["status"] == "complete",
        "chain_commit": chain["repository_commit"]
        == artifacts["stage4_7_chain"]["repository_commit"],
        "chain_checks": all(
            passed
            for family in chain["checks"].values()
            for passed in family.values()
        ),
        "edge_denominators": _expected_exact_edges(chain)
        == baseline["edge_exact_gpu_denominators"],
        "cumulative_denominator": chain["cumulative_gpu_cost"][
            "all_exact_reference_ms"
        ]
        == baseline["cumulative_exact_gpu_denominator_ms"],
        "endpoint_exact_task": _expected_exact_endpoints(chain)
        == baseline["endpoint_exact_task"],
        "summary_protocol": summary["protocol"]
        == artifacts["stage4_7_summary"]["protocol"],
        "summary_status": summary["status"]
        == artifacts["stage4_7_summary"]["status"],
        "summary_chain": summary["result_artifact"]["sha256"]
        == artifacts["stage4_7_chain"]["sha256"],
        "summary_compiler": summary["implementation_snapshot"][
            "compiler_result"
        ]["sha256"]
        == artifacts["stage4_7_compiler"]["sha256"],
        "incumbent_update": summary["gpu_cost"][
            "cumulative_update_only_ratio"
        ]
        == baseline["incumbent_stage4_7"]["primary_update_only_ratio"],
        "incumbent_symmetric": summary["gpu_cost"][
            "symmetric_lifecycle_ratio"
        ]
        == baseline["incumbent_stage4_7"]["symmetric_lifecycle_ratio"],
        "incumbent_common": summary["gpu_cost"][
            "common_inclusive_lifecycle_ratio"
        ]
        == baseline["incumbent_stage4_7"][
            "common_inclusive_lifecycle_ratio"
        ],
    }
    if not all(checks.values()):
        raise ValueError(f"Stage 4.8 baseline derivation differs: {checks}")


def load_exact_baseline(path: str | Path) -> dict:
    baseline_path = _repo_path(path)
    if (
        not baseline_path.exists()
        or sha256(baseline_path) != EXPECTED_BASELINE_SHA256
    ):
        raise ValueError("Stage 4.8 exact baseline SHA256 differs")
    baseline = json.loads(baseline_path.read_text())
    if (
        baseline.get("protocol") != BASELINE_PROTOCOL
        or baseline.get("status") != "complete"
        or len(baseline.get("edge_exact_gpu_denominators", [])) != NUM_EDGES
        or len(baseline.get("endpoint_exact_task", [])) != NUM_EDGES + 1
        or not math.isclose(
            sum(
                float(value["all_exact_reference_ms"])
                for value in baseline["edge_exact_gpu_denominators"]
            ),
            float(baseline["cumulative_exact_gpu_denominator_ms"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("Stage 4.8 exact baseline is invalid")
    for artifact in baseline["source_artifacts"].values():
        source = _repo_path(artifact["path"])
        if not source.exists() or sha256(source) != artifact["sha256"]:
            raise ValueError(
                f"Stage 4.8 frozen source differs: {artifact['path']}"
            )
    _validate_baseline_derivation(baseline)
    return baseline


def output_path(args: argparse.Namespace, spec: VariantSpec) -> Path:
    return Path(args.output_dir) / (
        f"stage4_8_{spec.scheme}_{spec.label}_seed0.json"
    )


def summary_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / (
        f"stage4_8_{args.scheme}_sweep_seed0.json"
    )


def _frozen_argument_binding(args: argparse.Namespace, baseline: dict) -> bool:
    provenance = baseline["input_provenance"]
    checkpoints = provenance["checkpoints"]
    compiler_pairs = provenance["compiler_pairs"]
    return (
        _repo_path(args.prepared_data).resolve()
        == _repo_path(provenance["prepared_data"]["path"]).resolve()
        and _repo_path(args.training_result).resolve()
        == _repo_path(provenance["training_result"]["path"]).resolve()
        and _repo_path(args.checkpoint_dir).resolve()
        == _repo_path(checkpoints[0]["path"]).parent.resolve()
        and _repo_path(args.compiler_result).resolve()
        == _repo_path(
            baseline["source_artifacts"]["stage4_7_compiler"]["path"]
        ).resolve()
        and _repo_path(args.runtime_dir).resolve()
        == _repo_path(
            compiler_pairs[0]["direct_program_path"]
        ).parent.resolve()
        and args.seed == baseline["configuration"]["training_seed"]
        and args.batch_size == baseline["configuration"]["batch_size"]
    )


def _expected_result_inputs(baseline: dict) -> dict:
    provenance = baseline["input_provenance"]
    return {
        "prepared_data": provenance["prepared_data"],
        "training_result": provenance["training_result"],
        "checkpoints": provenance["checkpoints"],
        "manifest_content_sha256": provenance["manifest"][
            "content_sha256"
        ],
        "windows": provenance["windows"],
        "compiler": baseline["source_artifacts"]["stage4_7_compiler"],
    }


def result_complete(
    path: Path,
    spec: VariantSpec,
    args: argparse.Namespace,
    expected_device: str,
) -> bool:
    if not path.exists():
        return False
    baseline = load_exact_baseline(args.baseline)
    if not _frozen_argument_binding(args, baseline):
        return False
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    configuration = value.get("configuration", {})
    exact_baseline = value.get("exact_baseline", {})
    return (
        value.get("protocol") == PROTOCOL
        and value.get("status") == "complete"
        and value.get("scheme") == spec.scheme
        and value.get("variant") == spec.to_dict()
        and value.get("repository_commit") == _repository_commit()
        and value.get("implementation")
        == implementation_snapshot(spec.scheme)
        and configuration.get("dataset")
        == baseline["configuration"]["dataset"]
        and configuration.get("split") == baseline["configuration"]["split"]
        and configuration.get("seed")
        == baseline["configuration"]["training_seed"]
        and configuration.get("batch_size")
        == baseline["configuration"]["batch_size"]
        and configuration.get("device") == str(torch.device(expected_device))
        and configuration.get("device_name")
        == baseline["configuration"]["device_class"]
        and configuration.get("records")
        == baseline["configuration"]["records"]
        and configuration.get("model") == baseline["configuration"]["model"]
        and exact_baseline.get("sha256") == EXPECTED_BASELINE_SHA256
        and exact_baseline.get("protocol") == BASELINE_PROTOCOL
        and exact_baseline.get("source_result")
        == baseline["source_artifacts"]["stage4_7_chain"]
        and value.get("inputs") == _expected_result_inputs(baseline)
        and value.get("checks", {}).get("all_passed") is True
    )


def _forwarded_path_args(args: argparse.Namespace) -> list[str]:
    return [
        "--prepared-data",
        args.prepared_data,
        "--training-result",
        args.training_result,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--compiler-result",
        args.compiler_result,
        "--runtime-dir",
        args.runtime_dir,
        "--baseline",
        args.baseline,
        "--output-dir",
        args.output_dir,
        "--log-dir",
        args.log_dir,
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
    ]


def launch_sweep(args: argparse.Namespace) -> dict:
    specs = variant_specs(args.scheme)
    script = str(Path(sys.argv[0]))
    baseline = load_exact_baseline(args.baseline)
    if not _frozen_argument_binding(args, baseline):
        raise ValueError("Stage 4.8 launcher arguments differ from baseline")
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    processes = []
    completed = []
    pending = []
    for spec, device in zip(specs, args.devices, strict=True):
        result = output_path(args, spec)
        if result.exists() and not args.force:
            if result_complete(result, spec, args, device):
                completed.append(result)
                print(f"skip complete {spec.label}: {result}", flush=True)
                continue
            raise FileExistsError(
                "incompatible Stage 4.8 result exists; "
                f"use --force: {result}"
            )
        pending.append((spec, device, result))
    for spec, device, result in pending:
        command = [
            sys.executable,
            script,
            "--grid-index",
            str(spec.grid_index),
            "--device",
            device,
            *_forwarded_path_args(args),
        ]
        if args.force:
            command.append("--force")
        log_path = Path(args.log_dir) / (
            f"stage4_8_{args.scheme}_{spec.label}.log"
        )
        handle = log_path.open("w")
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((spec, device, result, log_path, handle, process))
        print(
            f"launch {spec.label} on {device}, log={log_path}",
            flush=True,
        )
    failures = []
    for spec, device, result, log_path, handle, process in processes:
        returncode = process.wait()
        handle.close()
        if returncode != 0 or not result_complete(
            result,
            spec,
            args,
            device,
        ):
            failures.append(
                {
                    "variant": spec.label,
                    "device": device,
                    "returncode": returncode,
                    "log": str(log_path),
                }
            )
        else:
            completed.append(result)
            print(f"complete {spec.label}: {result}", flush=True)
    if failures:
        raise RuntimeError(f"Stage 4.8 sweep workers failed: {failures}")
    by_path = {path: json.loads(path.read_text()) for path in completed}
    ordered = [by_path[output_path(args, spec)] for spec in specs]
    payload = {
        "protocol": SWEEP_PROTOCOL,
        "status": "complete",
        "scheme": args.scheme,
        "parameter_gpu_assignment": [
            {
                **spec.to_dict(),
                "device": ordered[spec.grid_index]["configuration"][
                    "device"
                ],
                "requested_device": device,
                "result": str(output_path(args, spec)),
                "result_sha256": sha256(output_path(args, spec)),
            }
            for spec, device in zip(specs, args.devices, strict=True)
        ],
        "development_screen": True,
        "same_device_paired_confirmation_required": True,
        "points": [
            {
                "variant": value["variant"],
                "device": value["configuration"]["device"],
                "quality": value["record_weighted_task"],
                "gpu_cost": value["cumulative_gpu_cost"],
                "cost_gates": value["cost_gates"],
            }
            for value in ordered
        ],
        "all_points_preserved": True,
    }
    save_json(payload, summary_path(args))
    return payload


def _checkpoint_signature(checkpoints: list[dict]) -> list[dict]:
    return [
        {
            "version": value["version"],
            "path": value["path"],
            "sha256": value["sha256"],
            "bytes": value["bytes"],
        }
        for value in checkpoints
    ]


def validate_runtime_provenance(
    args: argparse.Namespace,
    baseline: dict,
    metadata: dict,
    training: dict,
    manifest: dict,
    checkpoints: list[dict],
    windows,
    compiler: dict,
) -> dict[str, bool]:
    provenance = baseline["input_provenance"]
    compiler_artifact = baseline["source_artifacts"]["stage4_7_compiler"]
    current_windows = [
        {
            "version": int(window.version),
            "target_date": str(window.target_date),
            "content_sha256": window.content_sha256,
        }
        for window in windows
    ]
    checks = {
        "prepared_sha256": sha256(_repo_path(args.prepared_data))
        == provenance["prepared_data"]["sha256"],
        "prepared_protocol": metadata["protocol"]
        == provenance["prepared_data"]["protocol"],
        "training_sha256": sha256(_repo_path(args.training_result))
        == provenance["training_result"]["sha256"],
        "training_protocol": training["protocol"]
        == provenance["training_result"]["protocol"],
        "manifest_protocol": manifest["protocol"]
        == provenance["manifest"]["protocol"],
        "manifest_sha256": manifest["content_sha256"]
        == provenance["manifest"]["content_sha256"],
        "manifest_records": len(manifest["records"])
        == provenance["manifest"]["records"],
        "checkpoints": _checkpoint_signature(checkpoints)
        == provenance["checkpoints"],
        "windows": current_windows == provenance["windows"],
        "compiler_sha256": sha256(_repo_path(args.compiler_result))
        == compiler_artifact["sha256"],
        "compiler_protocol": compiler["protocol"]
        == compiler_artifact["protocol"],
        "model": training["model"] == baseline["configuration"]["model"],
        "batch_size": args.batch_size
        == baseline["configuration"]["batch_size"],
    }
    exact_edges = baseline["edge_exact_gpu_denominators"]
    checks["exact_reference_record_shapes"] = all(
        exact_edges[version - 1]["exact_reference_records"]
        == sum(
            record.history is not None and len(record.history) >= 2
            for record in windows[version].records.values()
        )
        for version in range(1, NUM_EDGES + 1)
    )
    if not all(checks.values()):
        raise ValueError(f"Stage 4.8 runtime provenance differs: {checks}")
    return checks


def operator_severity(program) -> float:
    weights = program.weights.detach().to(dtype=torch.float32, device="cpu")
    biases = program.biases.detach().to(dtype=torch.float32, device="cpu")
    identity = torch.eye(weights.shape[1], dtype=torch.float32)
    weight_mean_square = (weights - identity.unsqueeze(0)).square().mean()
    bias_mean_square = biases.square().mean()
    value = float((weight_mean_square + bias_mean_square).sqrt())
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Stage 4.8 operator severity is invalid")
    return value


def _assemble_target_sources(
    record_ids: tuple[int, ...],
    sources: tuple[JaggedMigratedKVBatch, ...],
    target_version: int,
) -> JaggedMigratedKVBatch:
    if not record_ids or not sources:
        raise ValueError("Stage 4.8 target assembly is empty")
    source_by_record = {}
    for source in sources:
        for row, record_id in enumerate(source.record_ids):
            if record_id in source_by_record:
                raise ValueError("Stage 4.8 target record appears twice")
            source_by_record[record_id] = (source, row)
    if set(source_by_record) != set(record_ids):
        raise ValueError("Stage 4.8 target sources do not cover the layout")
    first = sources[0]
    lengths = torch.tensor(
        [
            int(source_by_record[record_id][0].lengths[
                source_by_record[record_id][1]
            ])
            for record_id in record_ids
        ],
        dtype=torch.long,
        device=first.k.device,
    )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=first.k.device),
            lengths.cumsum(0),
        )
    )
    k = torch.empty(
        (first.k.shape[0], int(offsets[-1]), first.k.shape[2]),
        dtype=first.k.dtype,
        device=first.k.device,
    )
    v = torch.empty_like(k)
    for target_row, record_id in enumerate(record_ids):
        source, source_row = source_by_record[record_id]
        if (
            source.k.shape[0] != first.k.shape[0]
            or source.k.shape[2] != first.k.shape[2]
            or source.k.dtype != first.k.dtype
            or source.k.device != first.k.device
        ):
            raise ValueError("Stage 4.8 target source layouts differ")
        source_start = int(source.offsets[source_row])
        source_stop = int(source.offsets[source_row + 1])
        target_start = int(offsets[target_row])
        target_stop = int(offsets[target_row + 1])
        k[:, target_start:target_stop].copy_(
            source.k[:, source_start:source_stop]
        )
        v[:, target_start:target_stop].copy_(
            source.v[:, source_start:source_stop]
        )
    return JaggedMigratedKVBatch(
        record_ids=record_ids,
        migration_anchor_version=f"theta{target_version}",
        served_kv_target=f"theta{target_version}",
        k=k,
        v=v,
        lengths=lengths,
        offsets=offsets,
    )


def _target_prefix(
    record_ids: tuple[int, ...],
    sources: tuple[JaggedMigratedKVBatch, ...],
    target_version: int,
    device: torch.device,
) -> tuple[JaggedMigratedKVBatch, float]:
    if len(sources) == 1 and sources[0].record_ids == record_ids:
        source = sources[0]
        return (
            JaggedMigratedKVBatch(
                record_ids=source.record_ids,
                migration_anchor_version=f"theta{target_version}",
                served_kv_target=f"theta{target_version}",
                k=source.k,
                v=source.v,
                lengths=source.lengths,
                offsets=source.offsets,
            ),
            0.0,
        )
    return timed_cuda(
        partial(
            _assemble_target_sources,
            record_ids,
            sources,
            target_version,
        ),
        device,
    )


def _score_task_rows(
    model,
    hidden: torch.Tensor,
    descriptors: list[dict],
    records: list,
    all_items: torch.Tensor,
) -> list[dict[str, float]]:
    selected = [
        row
        for row, (descriptor, record) in enumerate(
            zip(descriptors, records, strict=True)
        )
        if descriptor["evaluation_role"] == "final_test"
        and record.engaged_positive_item_ids
    ]
    if not selected:
        return []
    selected_hidden = hidden[
        torch.tensor(selected, dtype=torch.long, device=hidden.device)
    ]
    scores = model.item_emb.score(
        selected_hidden,
        all_items.unsqueeze(0).expand(len(selected), -1),
    )
    return [
        task_metrics(
            scores[index],
            list(records[row].engaged_positive_item_ids),
        )
        for index, row in enumerate(selected)
    ]


def _mixed_task_summary(
    values: list[dict[str, float]],
    exact: dict,
) -> dict:
    if len(values) != int(exact["records"]) or not values:
        raise ValueError("Stage 4.8 task records differ from exact baseline")
    mixed = {
        "catalog_auc": sum(value["catalog_auc"] for value in values)
        / len(values),
        "ndcg_at_100": sum(value["ndcg@100"] for value in values)
        / len(values),
        "hit_at_100": sum(value["hit@100"] for value in values)
        / len(values),
    }
    exact_metrics = {metric: float(exact[metric]) for metric in TASK_METRICS}
    return {
        "records": len(values),
        "mixed": mixed,
        "all_exact_external": exact_metrics,
        "mixed_minus_exact": {
            metric: mixed[metric] - exact_metrics[metric]
            for metric in TASK_METRICS
        },
        "mixed_over_exact": {
            metric: (
                mixed[metric] / exact_metrics[metric]
                if exact_metrics[metric] != 0
                else None
            )
            for metric in TASK_METRICS
        },
    }


def _record_weighted_task(endpoints: list[dict]) -> dict:
    records = sum(value["task_metrics"]["records"] for value in endpoints)
    mixed = {
        metric: sum(
            value["task_metrics"]["records"]
            * value["task_metrics"]["mixed"][metric]
            for value in endpoints
        )
        / records
        for metric in TASK_METRICS
    }
    exact = {
        metric: sum(
            value["task_metrics"]["records"]
            * value["task_metrics"]["all_exact_external"][metric]
            for value in endpoints
        )
        / records
        for metric in TASK_METRICS
    }
    return {
        "records": records,
        "mixed": mixed,
        "all_exact_external": exact,
        "mixed_minus_exact": {
            metric: mixed[metric] - exact[metric] for metric in TASK_METRICS
        },
        "mixed_over_exact": {
            metric: mixed[metric] / exact[metric] for metric in TASK_METRICS
        },
    }


def _serialize_state(state: object) -> dict:
    if not is_dataclass(state):
        raise ValueError("Stage 4.8 scheduler state is not serializable")
    return asdict(state)


def _select_actions(
    records: tuple[SchedulerRecord, ...],
    spec: VariantSpec,
    target_version: int,
    edge_severity: float | None,
    state: object | None,
):
    if spec.scheme == "staggered_renewal":
        return select_work_balanced_staggered_renewal(
            records,
            target_version=target_version,
            horizon=int(spec.parameter),
            state=state,
        )
    if spec.scheme == "token_debt":
        return select_total_token_cumulative_debt(
            records,
            budget_fraction=spec.parameter,
            state=state,
        )
    if spec.scheme == "aoi_maxweight":
        return select_aoi_maxweight(
            records,
            budget_fraction=spec.parameter,
            state=state,
        )
    if spec.scheme == "model_time_renewal":
        if edge_severity is None:
            raise ValueError("model-time renewal requires edge severity")
        return select_model_time_staggered_renewal(
            records,
            edge_severity=edge_severity,
            horizon=int(spec.parameter),
            state=state,
        )
    raise ValueError(f"unknown Stage 4.8 scheme: {spec.scheme}")


@torch.inference_mode()
def _initialize_theta0(
    cfg,
    checkpoint_dir: str,
    window,
    groups,
    device: torch.device,
) -> tuple[dict[int, JaggedMigratedKVBatch], dict[int, int], float]:
    model = load_checkpoint_model(cfg, checkpoint_dir, 0, device)
    cache_by_record = {}
    initialization_ms = 0.0
    for group in groups:
        selected = [
            value
            for value in group
            if window.records[int(value["user_id"])].history is not None
        ]
        if not selected:
            continue
        records = [
            window.records[int(value["user_id"])] for value in selected
        ]
        record_ids = tuple(int(value["record_id"]) for value in selected)
        batch = base._history_batch(
            records,
            cfg.max_seq_len,
            device,
            prefix=False,
        )
        (full, _), elapsed = timed_cuda(
            partial(
                base._exact_full_batch,
                model,
                batch,
                record_ids,
                0,
            ),
            device,
        )
        initialization_ms += elapsed
        cache_by_record.update(base._split_cache(full))
        del full, batch
    last_exact = {record_id: 0 for record_id in cache_by_record}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cache_by_record, last_exact, initialization_ms


def _theta0_state_checks(
    window,
    groups,
    record_by_id: dict[int, dict],
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    last_exact_by_record: dict[int, int],
) -> dict[str, bool]:
    residents = {
        int(value["record_id"])
        for group in groups
        for value in group
        if window.records[int(value["user_id"])].history is not None
    }
    return {
        "cache_covers_theta0_residents": set(cache_by_record) == residents,
        "last_exact_covers_theta0_residents": set(last_exact_by_record)
        == residents,
        "cache_lengths_match_theta0_history": all(
            int(cache.lengths[0])
            == len(
                window.records[
                    int(record_by_id[record_id]["user_id"])
                ].history
            )
            for record_id, cache in cache_by_record.items()
        ),
        "cache_versions_are_theta0": all(
            cache.migration_anchor_version == "theta0"
            and cache.served_kv_target == "theta0"
            for cache in cache_by_record.values()
        ),
        "last_exact_versions_are_zero": all(
            value == 0 for value in last_exact_by_record.values()
        ),
    }


def _group_partition(
    group,
    target_window,
    record_by_id: dict[int, dict],
    reusable_ids: set[int],
    natural_ids: set[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    descriptors = [
        record_by_id[int(value["record_id"])] for value in group
    ]
    reusable = [
        value
        for value in descriptors
        if int(value["record_id"]) in reusable_ids
    ]
    natural_prefix = [
        value
        for value in descriptors
        if int(value["record_id"]) in natural_ids
        and target_window.records[int(value["user_id"])].history is not None
        and len(target_window.records[int(value["user_id"])].history) >= 2
    ]
    natural_short = [
        value
        for value in descriptors
        if int(value["record_id"]) in natural_ids
        and target_window.records[int(value["user_id"])].history is not None
        and len(target_window.records[int(value["user_id"])].history) == 1
    ]
    if {
        int(value["record_id"])
        for value in (*reusable, *natural_prefix, *natural_short)
    } != reusable_ids.union(natural_ids).intersection(
        int(value["record_id"]) for value in descriptors
    ):
        raise RuntimeError("Stage 4.8 group partition differs")
    return reusable, natural_prefix, natural_short


def _lineage_value(
    descriptor: dict,
    transition,
    action: str,
    target_version: int,
    last_exact_before: int | None,
    last_exact_after: int | None,
    candidate_executed: bool,
) -> dict:
    return {
        "record_id": int(descriptor["record_id"]),
        "user_id": int(descriptor["user_id"]),
        "evaluation_role": descriptor["evaluation_role"],
        "old_history_sha256": transition.old_history_hash,
        "new_history_sha256": transition.new_history_hash,
        "foreground_status": transition.status,
        "old_history_tokens": transition.old_length,
        "overlap_tokens": transition.overlap,
        "evicted_tokens": transition.evicted,
        "appended_tokens": transition.appended,
        "source_prefix_tokens": transition.new_length,
        "common_latest_tokens": (
            1 if transition.new_history_hash is not None else 0
        ),
        "previous_actual_consumed": transition.previous_actual_consumed,
        "action": action,
        "last_exact_version_before": last_exact_before,
        "last_exact_version_after": last_exact_after,
        "migration_age_after": (
            target_version - last_exact_after
            if last_exact_after is not None
            else None
        ),
        "migration_candidate_executed": candidate_executed,
    }


@torch.inference_mode()
def _run_edge(
    args: argparse.Namespace,
    spec: VariantSpec,
    cfg,
    compiler: dict,
    exact_edge: dict,
    exact_endpoint: dict,
    old_window,
    target_window,
    groups,
    record_by_id: dict[int, dict],
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    last_exact_by_record: dict[int, int],
    scheduler_state: object | None,
    operator,
    all_items: torch.Tensor,
) -> tuple[
    dict[int, JaggedMigratedKVBatch],
    dict[int, int],
    object,
    dict,
    dict,
]:
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    source_version = int(old_window.version)
    target_version = int(target_window.version)
    source_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        source_version,
        device,
    )
    program, program_descriptor, program_cpu = base._load_program(
        args,
        cfg,
        compiler,
        source_version,
        device,
        operator,
    )
    severity = (
        operator_severity(program_cpu)
        if spec.scheme == "model_time_renewal"
        else None
    )
    costs = {
        "foreground_evict": 0.0,
        "foreground_incremental_append": 0.0,
        "candidate_transform": 0.0,
        "router_probe": 0.0,
        "exact_refresh": 0.0,
        "publication": 0.0,
        "common_latest": 0.0,
        "common_publication": 0.0,
        "natural_direct_exact": 0.0,
    }
    previous_resident_records = len(cache_by_record)
    previous_resident_bytes = base.resident_cache_bytes(cache_by_record)
    (
        source_prefixes,
        source_last_exact,
        direct_exact_ids,
        transitions,
        foreground_peak_bytes,
    ) = base._prepare_source_prefix(
        source_model,
        old_window,
        target_window,
        groups,
        cache_by_record,
        last_exact_by_record,
        device,
        costs,
    )
    previous_cache_consumed = not cache_by_record
    source_prefix_lengths_match = all(
        int(cache.lengths[0]) == transitions[record_id].new_length
        for record_id, cache in source_prefixes.items()
    )
    if not previous_cache_consumed:
        raise RuntimeError("Stage 4.8 foreground did not consume prior cache")
    reusable_ids = set(source_prefixes)
    natural_ids = set(direct_exact_ids)
    scheduler_records = tuple(
        [
            SchedulerRecord(
                record_id=record_id,
                prefix_tokens=transitions[record_id].new_length,
                migration_age=source_version
                - int(source_last_exact[record_id]),
                natural_exact=False,
            )
            for record_id in sorted(reusable_ids)
        ]
        + [
            SchedulerRecord(
                record_id=record_id,
                prefix_tokens=transitions[record_id].new_length,
                migration_age=0,
                natural_exact=True,
            )
            for record_id in sorted(natural_ids)
        ]
    )
    selection = _select_actions(
        scheduler_records,
        spec,
        target_version,
        severity,
        scheduler_state,
    )
    scheduled_ids = set(selection.scheduled_exact_ids)
    migrate_ids = set(selection.migrate_ids)
    if (
        set(selection.natural_exact_ids) != natural_ids
        or scheduled_ids | migrate_ids != reusable_ids
        or scheduled_ids & migrate_ids
    ):
        raise RuntimeError("Stage 4.8 scheduler action coverage differs")
    del source_model, program_cpu
    gc.collect()
    torch.cuda.empty_cache()
    target_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        target_version,
        device,
    )
    next_cache = {}
    next_last_exact = {}
    task_values = []
    lineage_by_record = {}
    publication_peak_bytes = base.resident_cache_bytes(source_prefixes)
    for group in groups:
        reusable, natural_prefix, natural_short = _group_partition(
            group,
            target_window,
            record_by_id,
            reusable_ids,
            natural_ids,
        )
        if reusable:
            record_ids = tuple(
                int(value["record_id"]) for value in reusable
            )
            records = [
                target_window.records[int(value["user_id"])]
                for value in reusable
            ]
            group_migrate_ids = tuple(
                value for value in record_ids if value in migrate_ids
            )
            group_exact_ids = tuple(
                value for value in record_ids if value in scheduled_ids
            )
            sources = []
            candidate = None
            if group_migrate_ids:
                source_prefix, assembly_ms = timed_cuda(
                    partial(
                        base._assemble_record_caches,
                        group_migrate_ids,
                        source_prefixes,
                    ),
                    device,
                )
                candidate, transform_ms = timed_cuda(
                    partial(
                        execute_direct,
                        operator,
                        program,
                        source_prefix,
                        target_version,
                    ),
                    device,
                )
                costs["candidate_transform"] += assembly_ms + transform_ms
                sources.append(candidate)
                del source_prefix
            exact_selected = None
            if group_exact_ids:
                exact_descriptors = [
                    value
                    for value in reusable
                    if int(value["record_id"]) in scheduled_ids
                ]
                exact_records = [
                    target_window.records[int(value["user_id"])]
                    for value in exact_descriptors
                ]
                exact_prefix_batch = base._history_batch(
                    exact_records,
                    cfg.max_seq_len,
                    device,
                    prefix=True,
                )
                exact_selected, exact_ms = timed_cuda(
                    partial(
                        exact_batch,
                        target_model,
                        exact_prefix_batch,
                        group_exact_ids,
                        target_version,
                    ),
                    device,
                )
                costs["exact_refresh"] += exact_ms
                sources.append(exact_selected)
                del exact_prefix_batch
            for record_id in record_ids:
                source_prefixes.pop(record_id)
            target_prefix, publication_ms = _target_prefix(
                record_ids,
                tuple(sources),
                target_version,
                device,
            )
            costs["publication"] += publication_ms
            suffix = base._suffix_batch(records, device)
            common, common_ms = timed_cuda(
                partial(
                    append_jagged_suffix,
                    target_model,
                    base.identity_jagged_slice(target_prefix),
                    suffix["item_ids"],
                    suffix["behaviors"],
                    suffix["time_deltas"],
                    suffix["lengths"],
                ),
                device,
            )
            costs["common_latest"] += common_ms
            published, split_ms = timed_cuda(
                partial(base._split_cache, common.cache),
                device,
            )
            costs["common_publication"] += split_ms
            next_cache.update(published)
            hidden = common.last_appended_hidden
            if hidden is None:
                raise RuntimeError("Stage 4.8 common latest has no hidden")
            task_values.extend(
                _score_task_rows(
                    target_model,
                    hidden,
                    reusable,
                    records,
                    all_items,
                )
            )
            for descriptor, record_id in zip(
                reusable,
                record_ids,
                strict=True,
            ):
                before = int(source_last_exact[record_id])
                after = (
                    target_version
                    if record_id in scheduled_ids
                    else before
                )
                next_last_exact[record_id] = after
                lineage_by_record[record_id] = _lineage_value(
                    descriptor,
                    transitions[record_id],
                    (
                        "scheduled_exact"
                        if record_id in scheduled_ids
                        else "migrate"
                    ),
                    target_version,
                    before,
                    after,
                    record_id in migrate_ids,
                )
            del target_prefix, suffix, common, published, hidden, sources
            if candidate is not None:
                del candidate
            if exact_selected is not None:
                del exact_selected
        if natural_prefix:
            record_ids = tuple(
                int(value["record_id"]) for value in natural_prefix
            )
            records = [
                target_window.records[int(value["user_id"])]
                for value in natural_prefix
            ]
            prefix_batch = base._history_batch(
                records,
                cfg.max_seq_len,
                device,
                prefix=True,
            )
            exact_prefix, exact_ms = timed_cuda(
                partial(
                    exact_batch,
                    target_model,
                    prefix_batch,
                    record_ids,
                    target_version,
                ),
                device,
            )
            costs["exact_refresh"] += exact_ms
            costs["natural_direct_exact"] += exact_ms
            suffix = base._suffix_batch(records, device)
            common, common_ms = timed_cuda(
                partial(
                    append_jagged_suffix,
                    target_model,
                    base.identity_jagged_slice(exact_prefix),
                    suffix["item_ids"],
                    suffix["behaviors"],
                    suffix["time_deltas"],
                    suffix["lengths"],
                ),
                device,
            )
            costs["common_latest"] += common_ms
            published, split_ms = timed_cuda(
                partial(base._split_cache, common.cache),
                device,
            )
            costs["common_publication"] += split_ms
            next_cache.update(published)
            hidden = common.last_appended_hidden
            if hidden is None:
                raise RuntimeError("Stage 4.8 natural exact has no hidden")
            task_values.extend(
                _score_task_rows(
                    target_model,
                    hidden,
                    natural_prefix,
                    records,
                    all_items,
                )
            )
            for descriptor, record_id in zip(
                natural_prefix,
                record_ids,
                strict=True,
            ):
                before = last_exact_by_record.get(record_id)
                next_last_exact[record_id] = target_version
                lineage_by_record[record_id] = _lineage_value(
                    descriptor,
                    transitions[record_id],
                    "natural_exact",
                    target_version,
                    before,
                    target_version,
                    False,
                )
            del prefix_batch, exact_prefix, suffix, common, published, hidden
        if natural_short:
            record_ids = tuple(
                int(value["record_id"]) for value in natural_short
            )
            records = [
                target_window.records[int(value["user_id"])]
                for value in natural_short
            ]
            suffix = base._suffix_batch(records, device)
            common, common_ms = timed_cuda(
                partial(
                    append_jagged_suffix,
                    target_model,
                    base.empty_jagged_slice(
                        record_ids,
                        target_version,
                        cfg.num_layers,
                        cfg.num_heads * cfg.head_dim,
                        torch.float16,
                        device,
                    ),
                    suffix["item_ids"],
                    suffix["behaviors"],
                    suffix["time_deltas"],
                    suffix["lengths"],
                ),
                device,
            )
            costs["common_latest"] += common_ms
            published, split_ms = timed_cuda(
                partial(base._split_cache, common.cache),
                device,
            )
            costs["common_publication"] += split_ms
            next_cache.update(published)
            hidden = common.last_appended_hidden
            if hidden is None:
                raise RuntimeError("Stage 4.8 short history has no hidden")
            task_values.extend(
                _score_task_rows(
                    target_model,
                    hidden,
                    natural_short,
                    records,
                    all_items,
                )
            )
            for descriptor, record_id in zip(
                natural_short,
                record_ids,
                strict=True,
            ):
                before = last_exact_by_record.get(record_id)
                next_last_exact[record_id] = target_version
                lineage_by_record[record_id] = _lineage_value(
                    descriptor,
                    transitions[record_id],
                    "natural_exact_short",
                    target_version,
                    before,
                    target_version,
                    False,
                )
            del suffix, common, published, hidden
        publication_peak_bytes = max(
            publication_peak_bytes,
            base.resident_cache_bytes(source_prefixes)
            + base.resident_cache_bytes(next_cache),
        )
    for group in groups:
        for descriptor in group:
            record_id = int(descriptor["record_id"])
            if record_id in lineage_by_record:
                continue
            transition = transitions[record_id]
            lineage_by_record[record_id] = _lineage_value(
                descriptor,
                transition,
                "expire" if transition.status == "expired" else "absent",
                target_version,
                last_exact_by_record.get(record_id),
                None,
                False,
            )
    residents = {
        int(value["record_id"])
        for group in groups
        for value in group
        if target_window.records[int(value["user_id"])].history is not None
    }
    if (
        source_prefixes
        or set(next_cache) != residents
        or set(next_last_exact) != residents
        or set(lineage_by_record) != {
            int(value["record_id"]) for group in groups for value in group
        }
    ):
        raise RuntimeError("Stage 4.8 recursive state coverage differs")
    endpoint_task = _mixed_task_summary(task_values, exact_endpoint)
    step_cost = base.cost_summary(
        costs,
        float(exact_edge["all_exact_reference_ms"]),
    )
    cuda_peak = torch.cuda.max_memory_allocated(device)
    checks = {
        "target_cache_covers_residents": set(next_cache) == residents,
        "last_exact_state_covers_residents": set(next_last_exact) == residents,
        "cache_lengths_match_history": all(
            int(cache.lengths[0])
            == len(
                target_window.records[
                    int(record_by_id[record_id]["user_id"])
                ].history
            )
            for record_id, cache in next_cache.items()
        ),
        "cache_versions_match_target": all(
            cache.migration_anchor_version == f"theta{target_version}"
            and cache.served_kv_target == f"theta{target_version}"
            for cache in next_cache.values()
        ),
        "history_overlap_arithmetic": all(
            value.old_length - value.evicted == value.overlap
            and value.overlap + value.appended == value.new_length
            for value in transitions.values()
        ),
        "source_prefix_lengths_match_target_h_minus_latest": (
            source_prefix_lengths_match
        ),
        "prefix_plus_latest_matches_target_history": all(
            transition.new_length
            + (
                1
                if target_window.records[
                    int(record_by_id[record_id]["user_id"])
                ].history
                is not None
                else 0
            )
            == (
                len(
                    target_window.records[
                        int(record_by_id[record_id]["user_id"])
                    ].history
                )
                if target_window.records[
                    int(record_by_id[record_id]["user_id"])
                ].history
                is not None
                else 0
            )
            for record_id, transition in transitions.items()
        ),
        "lineage_history_hashes_match_edge_windows": all(
            value["old_history_sha256"]
            == old_window.records[value["user_id"]].history_sha256
            and value["new_history_sha256"]
            == target_window.records[value["user_id"]].history_sha256
            for value in lineage_by_record.values()
        ),
        "previous_actual_consumption_disclosed": all(
            value["previous_actual_consumed"]
            == (
                value["foreground_status"] == "continued"
                and value["overlap_tokens"] > 0
            )
            for value in lineage_by_record.values()
        ),
        "previous_cache_consumed_during_foreground": (
            previous_cache_consumed
        ),
        "source_prefix_cache_consumed_during_publication": (
            not source_prefixes
        ),
        "adjacent_versions": target_version == source_version + 1,
        "scheduler_covers_reusable": scheduled_ids | migrate_ids
        == reusable_ids,
        "scheduler_covers_natural": set(selection.natural_exact_ids)
        == natural_ids,
        "scheduled_exact_skips_migration": all(
            not lineage_by_record[value]["migration_candidate_executed"]
            for value in scheduled_ids
        ),
        "only_migrants_execute_candidate": all(
            lineage_by_record[value]["migration_candidate_executed"]
            == (value in migrate_ids)
            for value in reusable_ids
        ),
        "labels_not_used_for_routing": selection.diagnostics.get(
            "labels_used"
        )
        is False,
        "task_records_match_external_exact": endpoint_task["records"]
        == exact_endpoint["records"],
        "external_exact_reference_not_executed": True,
        "lineage_covers_manifest": len(lineage_by_record)
        == sum(len(group) for group in groups),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Stage 4.8 edge checks failed: "
            f"{[key for key, value in checks.items() if not value]}"
        )
    scheduled_tokens = sum(
        transitions[value].new_length for value in scheduled_ids
    )
    natural_tokens = sum(
        transitions[value].new_length for value in natural_ids
    )
    step = {
        "source_version": source_version,
        "target_version": target_version,
        "prediction_target_date": target_window.target_date,
        "actions": {
            "migrate": len(migrate_ids),
            "scheduled_exact": len(scheduled_ids),
            "natural_exact": len(natural_ids),
            "natural_exact_with_prefix": sum(
                transitions[value].new_length > 0 for value in natural_ids
            ),
            "natural_exact_short": sum(
                transitions[value].new_length == 0 for value in natural_ids
            ),
            "scheduled_exact_prefix_tokens": scheduled_tokens,
            "natural_exact_prefix_tokens": natural_tokens,
            "reusable_prefix_tokens": sum(
                transitions[value].new_length for value in reusable_ids
            ),
            "resident_prefix_tokens": sum(
                transitions[value].new_length
                for value in reusable_ids | natural_ids
            ),
        },
        "scheduler": {
            "variant": spec.to_dict(),
            "edge_operator_severity": severity,
            "diagnostics": selection.diagnostics,
            "state_after": _serialize_state(selection.next_state),
            "scheduled_exact_ids": list(selection.scheduled_exact_ids),
        },
        "task_metrics": endpoint_task,
        "cost": step_cost,
        "external_exact_reference": {
            **exact_edge,
            "executed_in_this_run": False,
        },
        "memory": {
            "previous_resident_records": previous_resident_records,
            "output_resident_records": len(next_cache),
            "logical_previous_kv_bytes": previous_resident_bytes,
            "logical_output_kv_bytes": base.resident_cache_bytes(next_cache),
            "logical_foreground_peak_bytes": foreground_peak_bytes,
            "logical_publication_peak_bytes": publication_peak_bytes,
            "cuda_max_memory_allocated_bytes": cuda_peak,
        },
        "checks": checks,
        "lineage": [
            lineage_by_record[value] for value in sorted(lineage_by_record)
        ],
        "program": {
            "sha256": program_descriptor["sha256"],
            "labels_used": False,
        },
    }
    endpoint = {
        "version": target_version,
        "target_date": target_window.target_date,
        "resident_records": len(next_cache),
        "task_metrics": endpoint_task,
    }
    del target_model, program
    gc.collect()
    torch.cuda.empty_cache()
    return (
        next_cache,
        next_last_exact,
        selection.next_state,
        endpoint,
        step,
    )


def _cumulative_cost(steps: list[dict]) -> dict:
    component_keys = (
        "foreground_evict",
        "foreground_incremental_append",
        "candidate_transform",
        "router_probe",
        "exact_refresh",
        "publication",
        "common_latest",
        "common_publication",
        "natural_direct_exact",
    )
    costs = {
        key: sum(float(step["cost"][key]) for step in steps)
        for key in component_keys
    }
    exact_ms = sum(
        float(step["cost"]["all_exact_reference_ms"]) for step in steps
    )
    return base.cost_summary(costs, exact_ms)


def _anchor_endpoint(baseline: dict) -> dict:
    exact = baseline["endpoint_exact_task"][0]
    metrics = {metric: float(exact[metric]) for metric in TASK_METRICS}
    return {
        "version": 0,
        "target_date": exact["target_date"],
        "resident_records": baseline["configuration"]["records"],
        "anchor": "theta0_exact_full_history",
        "task_metrics": {
            "records": exact["records"],
            "mixed": metrics,
            "all_exact_external": metrics,
            "mixed_minus_exact": {metric: 0.0 for metric in TASK_METRICS},
            "mixed_over_exact": {metric: 1.0 for metric in TASK_METRICS},
        },
    }


def run_variant(
    args: argparse.Namespace,
    spec: VariantSpec,
) -> dict:
    device = torch.device(args.device)
    baseline = load_exact_baseline(args.baseline)
    plan, metadata, training, cfg, manifest, checkpoints = load_inputs(
        args.prepared_data,
        args.training_result,
        args.checkpoint_dir,
    )
    user_ids = tuple(
        int(value["user_id"]) for value in manifest["records"]
    )
    windows = reconstruct_organic_windows(plan, user_ids)
    window_checks = base.validate_windows(windows, manifest)
    compiler = json.loads(Path(args.compiler_result).read_text())
    compiler_checks = base.validate_compiler_payload(
        compiler,
        manifest,
        windows,
        checkpoints,
    )
    provenance_checks = validate_runtime_provenance(
        args,
        baseline,
        metadata,
        training,
        manifest,
        checkpoints,
        windows,
        compiler,
    )
    device_name = torch.cuda.get_device_name(device)
    if device_name != baseline["configuration"]["device_class"]:
        raise ValueError(
            f"Stage 4.8 device differs: {device_name}"
        )
    groups = base.fixed_record_groups(manifest, args.batch_size)
    record_by_id = {
        int(value["record_id"]): value for value in manifest["records"]
    }
    cache_by_record, last_exact, initialization_ms = _initialize_theta0(
        cfg,
        args.checkpoint_dir,
        windows[0],
        groups,
        device,
    )
    theta0_checks = _theta0_state_checks(
        windows[0],
        groups,
        record_by_id,
        cache_by_record,
        last_exact,
    )
    if not all(theta0_checks.values()):
        raise RuntimeError(
            f"Stage 4.8 theta0 state differs: {theta0_checks}"
        )
    all_items = torch.arange(
        1,
        cfg.num_prediction_items + 1,
        device=device,
    )
    operator = DirectOldKVFusedOperator(**LAUNCH)
    scheduler_state = None
    endpoints = [_anchor_endpoint(baseline)]
    steps = []
    started = time.perf_counter()
    for source_version in range(NUM_EDGES):
        (
            cache_by_record,
            last_exact,
            scheduler_state,
            endpoint,
            step,
        ) = _run_edge(
            args,
            spec,
            cfg,
            compiler,
            baseline["edge_exact_gpu_denominators"][source_version],
            baseline["endpoint_exact_task"][source_version + 1],
            windows[source_version],
            windows[source_version + 1],
            groups,
            record_by_id,
            cache_by_record,
            last_exact,
            scheduler_state,
            operator,
            all_items,
        )
        endpoints.append(endpoint)
        steps.append(step)
        print(
            json.dumps(
                {
                    "scheme": spec.scheme,
                    "variant": spec.label,
                    "source_version": source_version,
                    "target_version": source_version + 1,
                    "actions": step["actions"],
                    "task_metrics": endpoint["task_metrics"],
                    "cost": step["cost"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    cumulative_cost = _cumulative_cost(steps)
    if not math.isclose(
        cumulative_cost["all_exact_reference_ms"],
        baseline["cumulative_exact_gpu_denominator_ms"],
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError("Stage 4.8 cumulative exact denominator differs")
    update_endpoints = endpoints[1:]
    weighted_task = _record_weighted_task(update_endpoints)
    frozen_weighted = baseline["record_weighted_exact_task"][
        "eleven_update_endpoints"
    ]
    if (
        weighted_task["records"] != frozen_weighted["records"]
        or any(
            not math.isclose(
                weighted_task["all_exact_external"][metric],
                frozen_weighted[metric],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for metric in TASK_METRICS
        )
    ):
        raise RuntimeError("Stage 4.8 weighted exact task reference differs")
    incumbent = baseline["incumbent_stage4_7"]
    cost_gates = {
        "symmetric_lifecycle_strictly_below_stage4_7": (
            cumulative_cost["symmetric_lifecycle_ratio"]
            < incumbent["symmetric_lifecycle_ratio"]
        ),
        "common_inclusive_strictly_below_stage4_7": (
            cumulative_cost["common_inclusive_ratio"]
            < incumbent["common_inclusive_lifecycle_ratio"]
        ),
        "thresholds": incumbent,
        "all_passed": False,
    }
    cost_gates["all_passed"] = all(
        bool(value)
        for key, value in cost_gates.items()
        if key.endswith("stage4_7")
    )
    chain_checks = {
        "twelve_endpoints": len(endpoints) == 12,
        "eleven_updates": len(steps) == 11,
        "fixed_lineage_rows": all(
            len(step["lineage"]) == len(manifest["records"])
            for step in steps
        ),
        "adjacent_versions": all(
            step["target_version"] == step["source_version"] + 1
            for step in steps
        ),
        "previous_actual_consumption_disclosed": all(
            value["previous_actual_consumed"]
            == (
                value["foreground_status"] == "continued"
                and value["overlap_tokens"] > 0
            )
            for step in steps
            for value in step["lineage"]
        ),
        "history_hashes_match_windows": all(
            value["old_history_sha256"]
            == windows[step["source_version"]]
            .records[value["user_id"]]
            .history_sha256
            and value["new_history_sha256"]
            == windows[step["target_version"]]
            .records[value["user_id"]]
            .history_sha256
            for step in steps
            for value in step["lineage"]
        ),
        "all_edge_checks_pass": all(
            all(step["checks"].values()) for step in steps
        ),
        "recursive_state_present": bool(cache_by_record)
        and set(cache_by_record) == set(last_exact),
        "external_exact_never_executed": all(
            step["external_exact_reference"]["executed_in_this_run"] is False
            for step in steps
        ),
        "labels_never_used_for_routing": all(
            step["scheduler"]["diagnostics"]["labels_used"] is False
            for step in steps
        ),
        "all_scheduled_exact_skip_candidate": all(
            value["action"] != "scheduled_exact"
            or not value["migration_candidate_executed"]
            for step in steps
            for value in step["lineage"]
        ),
        "weighted_task_records_match": weighted_task["records"]
        == frozen_weighted["records"],
    }
    all_checks = {
        "causality": window_checks,
        "compiler": compiler_checks,
        "provenance": provenance_checks,
        "theta0": theta0_checks,
        "chain": chain_checks,
    }
    checks_pass = all(
        value
        for family in all_checks.values()
        for value in family.values()
    )
    if not checks_pass:
        raise RuntimeError("Stage 4.8 global protocol checks failed")
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "scheme": spec.scheme,
        "variant": spec.to_dict(),
        "study_stage": "single_configuration_development_screen",
        "repository_commit": _repository_commit(),
        "implementation": implementation_snapshot(spec.scheme),
        "configuration": {
            "dataset": "KuaiRand-1K",
            "split": "4+12",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "device": str(device),
            "device_name": device_name,
            "records": len(manifest["records"]),
            "model": training["model"],
        },
        "measurement_boundary": {
            "external_exact_reference_executed": False,
            "scheduled_exact_candidate_executed": False,
            "gpu_lifecycle_measured": True,
            "catalog_scoring_excluded": True,
            "scheduler_cpu_time_excluded": True,
            "host_sequence_construction_excluded": True,
            "compiler_excluded": True,
            "common_latest_measured_separately": True,
            "source_residency": "prior_actual_kv_hot_in_hbm",
        },
        "exact_baseline": {
            "path": args.baseline,
            "sha256": sha256(_repo_path(args.baseline)),
            "protocol": baseline["protocol"],
            "source_result": baseline["source_artifacts"][
                "stage4_7_chain"
            ],
        },
        "inputs": {
            "prepared_data": baseline["input_provenance"]["prepared_data"],
            "training_result": baseline["input_provenance"][
                "training_result"
            ],
            "checkpoints": checkpoints,
            "manifest_content_sha256": manifest["content_sha256"],
            "windows": baseline["input_provenance"]["windows"],
            "compiler": baseline["source_artifacts"][
                "stage4_7_compiler"
            ],
        },
        "theta0_initialization_gpu_ms": initialization_ms,
        "endpoints": endpoints,
        "steps": steps,
        "record_weighted_task": weighted_task,
        "cumulative_gpu_cost": cumulative_cost,
        "cost_gates": cost_gates,
        "checks": {
            **all_checks,
            "all_passed": checks_pass,
        },
        "final_scheduler_state": _serialize_state(scheduler_state),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _smoke_records() -> tuple[SchedulerRecord, ...]:
    return tuple(
        SchedulerRecord(
            record_id=record_id,
            prefix_tokens=(record_id + 1) * 10,
            migration_age=record_id % 5,
            natural_exact=record_id == 0,
        )
        for record_id in range(12)
    )


def smoke_payload(args: argparse.Namespace) -> dict:
    baseline = load_exact_baseline(args.baseline)
    specs = variant_specs(args.scheme)
    points = []
    for spec in specs:
        state = None
        total_scheduled = 0
        for target_version in range(1, NUM_EDGES + 1):
            records = _smoke_records()
            result = _select_actions(
                records,
                spec,
                target_version,
                1.0 + target_version / 10,
                state,
            )
            state = result.next_state
            expected = {value.record_id for value in records}
            actual = (
                set(result.scheduled_exact_ids)
                | set(result.natural_exact_ids)
                | set(result.migrate_ids)
            )
            if expected != actual:
                raise RuntimeError("Stage 4.8 smoke action coverage differs")
            total_scheduled += len(result.scheduled_exact_ids)
        points.append(
            {
                "variant": spec.to_dict(),
                "simulated_scheduled_exact": total_scheduled,
                "final_state": _serialize_state(state),
            }
        )
    return {
        "protocol": SWEEP_PROTOCOL,
        "status": "smoke_passed",
        "scheme": args.scheme,
        "points": points,
        "baseline": {
            "path": args.baseline,
            "sha256": sha256(_repo_path(args.baseline)),
            "exact_gpu_ms": baseline[
                "cumulative_exact_gpu_denominator_ms"
            ],
            "source_artifacts_verified": True,
        },
        "formal_execution": {
            "workers": 4,
            "one_process_per_gpu": True,
            "external_exact_reference_reused": True,
            "external_exact_reference_executed": False,
            "all_points_preserved": True,
        },
    }


def runtime_smoke_payload(
    args: argparse.Namespace,
    spec: VariantSpec,
) -> dict:
    device = torch.device(args.device)
    baseline = load_exact_baseline(args.baseline)
    plan, metadata, training, cfg, manifest, checkpoints = load_inputs(
        args.prepared_data,
        args.training_result,
        args.checkpoint_dir,
    )
    user_ids = tuple(
        int(value["user_id"]) for value in manifest["records"]
    )
    windows = reconstruct_organic_windows(plan, user_ids)
    window_checks = base.validate_windows(windows, manifest)
    compiler = json.loads(Path(args.compiler_result).read_text())
    compiler_checks = base.validate_compiler_payload(
        compiler,
        manifest,
        windows,
        checkpoints,
    )
    provenance_checks = validate_runtime_provenance(
        args,
        baseline,
        metadata,
        training,
        manifest,
        checkpoints,
        windows,
        compiler,
    )
    if torch.cuda.get_device_name(device) != baseline["configuration"][
        "device_class"
    ]:
        raise ValueError("Stage 4.8 runtime smoke device differs")
    groups = base.fixed_record_groups(manifest, args.batch_size)
    record_by_id = {
        int(value["record_id"]): value for value in manifest["records"]
    }
    cache_by_record, last_exact, initialization_ms = _initialize_theta0(
        cfg,
        args.checkpoint_dir,
        windows[0],
        groups,
        device,
    )
    theta0_checks = _theta0_state_checks(
        windows[0],
        groups,
        record_by_id,
        cache_by_record,
        last_exact,
    )
    all_items = torch.arange(
        1,
        cfg.num_prediction_items + 1,
        device=device,
    )
    operator = DirectOldKVFusedOperator(**LAUNCH)
    (
        cache_by_record,
        last_exact,
        scheduler_state,
        endpoint,
        step,
    ) = _run_edge(
        args,
        spec,
        cfg,
        compiler,
        baseline["edge_exact_gpu_denominators"][0],
        baseline["endpoint_exact_task"][1],
        windows[0],
        windows[1],
        groups,
        record_by_id,
        cache_by_record,
        last_exact,
        None,
        operator,
        all_items,
    )
    checks = {
        "causality": all(window_checks.values()),
        "compiler": all(compiler_checks.values()),
        "provenance": all(provenance_checks.values()),
        "theta0": all(theta0_checks.values()),
        "edge": all(step["checks"].values()),
        "recursive_state": set(cache_by_record) == set(last_exact),
        "external_exact_not_executed": step[
            "external_exact_reference"
        ]["executed_in_this_run"]
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 4.8 runtime smoke failed: {checks}")
    del cache_by_record, last_exact, all_items
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "protocol": PROTOCOL,
        "status": "runtime_smoke_passed",
        "scheme": spec.scheme,
        "variant": spec.to_dict(),
        "device": str(device),
        "theta0_initialization_gpu_ms": initialization_ms,
        "edge": {
            "source_version": 0,
            "target_version": 1,
            "actions": step["actions"],
            "task_metrics": endpoint["task_metrics"],
            "cost": step["cost"],
            "scheduler": {
                "diagnostics": step["scheduler"]["diagnostics"],
                "state_type": type(scheduler_state).__name__,
            },
        },
        "checks": checks,
    }


def worker_main(args: argparse.Namespace) -> None:
    spec = variant_specs(args.scheme)[args.grid_index]
    result = output_path(args, spec)
    if result.exists() and not args.force:
        if result_complete(result, spec, args, args.device):
            print(f"already complete: {result}")
            return
        raise FileExistsError(
            f"incomplete Stage 4.8 result exists; use --force: {result}"
        )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    payload = run_variant(args, spec)
    save_json(payload, result)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(result),
                "variant": payload["variant"],
                "record_weighted_task": payload["record_weighted_task"],
                "cumulative_gpu_cost": payload["cumulative_gpu_cost"],
                "cost_gates": payload["cost_gates"],
            }
        ),
        flush=True,
    )


def main_for_scheme(scheme: str) -> None:
    args = parse_args(scheme)
    validate_args(args)
    if args.smoke_test:
        print(json.dumps(smoke_payload(args), indent=2))
        return
    if args.runtime_smoke_test:
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        seed_everything(args.seed)
        spec = variant_specs(args.scheme)[args.grid_index]
        print(json.dumps(runtime_smoke_payload(args, spec), indent=2))
        return
    if args.grid_index is not None:
        worker_main(args)
        return
    payload = launch_sweep(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scheme": payload["scheme"],
                "summary": str(summary_path(args)),
                "points": payload["points"],
            },
            indent=2,
        )
    )
