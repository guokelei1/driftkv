from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ..models import HSTUConfig, HSTUKVCache
from .kuairand_projected_persistent import (
    _broadcast,
    _distributed,
    _initialize_model,
    _load_checkpoint,
    _manifest_path,
    _read_manifest,
    _seed,
    load_persistent_config,
)
from .kuairand_query_transition import _atomic_json, file_sha256, load_config
from .sharded_edge import ExternalEmbeddingHSTU

PROTOCOL = "evokv_kuairand_large_hotkv_scaling_v0"
METHODS = ("reuse", "recompute")


def load_scaling_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("timing", {}).get("primary_scope")
        != "hot_hbm_hstu_core_only"
        or document.get("timing", {}).get("methods") != list(METHODS)
        or document.get("timing", {}).get("suffix_tokens") != 2
        or document.get("timing", {}).get("both_operations_return_full_target_kv")
        is not True
        or document.get("execution", {}).get("world_size") != 2
        or document.get("execution", {}).get("devices") != [0, 1]
        or set(document.get("profiles", {})) != {"canary", "full"}
        or not isinstance(document.get("outputs", {}).get("root"), str)
    ):
        raise ValueError("KuaiRand hot-K/V scaling config differs")
    for name in ("registry", "large_config", "checkpoint_manifest"):
        artifact = document.get("baseline", {}).get(name)
        artifact_path = (
            Path(artifact.get("path", "")) if isinstance(artifact, dict) else Path()
        )
        if (
            not isinstance(artifact, dict)
            or not artifact_path.is_file()
            or file_sha256(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError("KuaiRand hot-K/V scaling baseline binding differs")
    for artifact in document.get("implementation", []):
        artifact_path = Path(artifact.get("path", ""))
        if (
            not artifact_path.is_file()
            or file_sha256(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError("KuaiRand hot-K/V scaling implementation differs")
    for name, profile in document["profiles"].items():
        if profile.get("name") != name:
            raise ValueError("KuaiRand hot-K/V scaling profile name differs")
        _validate_profile(profile)
    return document


def _validate_lengths(values: Any) -> list[int]:
    if not isinstance(values, list):
        raise ValueError("KuaiRand hot-K/V scaling lengths differ")
    result = [int(value) for value in values]
    if result != sorted(set(result)) or not result or min(result) < 4:
        raise ValueError("KuaiRand hot-K/V scaling lengths differ")
    return result


def _validate_profile(profile: dict[str, Any]) -> None:
    common = profile.get("common")
    if (
        not isinstance(common, dict)
        or int(common.get("maximum_sequence_tokens", 0)) < 512
        or int(common.get("head_dim", 0)) < 1
        or int(common.get("seed", -1)) < 0
    ):
        raise ValueError("KuaiRand hot-K/V scaling common profile differs")
    standalone = profile.get("standalone_latency")
    sequence = profile.get("sequence_scaling")
    requests = profile.get("request_scaling")
    depth = profile.get("depth_length_scaling")
    width = profile.get("width_scaling")
    context = profile.get("context_capacity_scaling")
    anchor = profile.get("checkpoint_anchor")
    if not all(
        isinstance(value, dict)
        for value in (
            standalone,
            sequence,
            requests,
            depth,
            width,
            context,
            anchor,
        )
    ):
        raise ValueError("KuaiRand hot-K/V scaling sweep differs")
    max_sequence = int(common["maximum_sequence_tokens"])
    for sweep in (standalone, sequence, depth):
        if max(_validate_lengths(sweep.get("sequence_tokens"))) > max_sequence:
            raise ValueError("KuaiRand hot-K/V scaling sequence bound differs")
    request_counts = [int(value) for value in requests.get("request_counts", [])]
    local_batch = int(requests.get("local_batch_size", 0))
    global_batch = 2 * local_batch
    if (
        request_counts != sorted(set(request_counts))
        or not request_counts
        or global_batch < 2
        or any(value % global_batch for value in request_counts)
    ):
        raise ValueError("KuaiRand hot-K/V scaling request counts differ")
    for sweep in (anchor, standalone, sequence, depth, width, context):
        local_batch = int(sweep.get("local_batch_size", 0))
        requests_per_point = int(sweep.get("requests", 0))
        if (
            local_batch < 1
            or requests_per_point < local_batch * 2
            or requests_per_point % (local_batch * 2)
            or int(sweep.get("warmup_iterations", -1)) < 1
            or int(sweep.get("measured_repeats", 0)) < 1
        ):
            raise ValueError("KuaiRand hot-K/V scaling execution shape differs")
    layers = [int(value) for value in depth.get("layers", [])]
    hidden = [int(value) for value in width.get("hidden_sizes", [])]
    context_lengths = [
        int(value) for value in context.get("maximum_sequence_tokens", [])
    ]
    head_dim = int(common["head_dim"])
    if (
        layers != sorted(set(layers))
        or not layers
        or min(layers) < 1
        or hidden != sorted(set(hidden))
        or not hidden
        or any(value % head_dim for value in hidden)
        or context_lengths != sorted(set(context_lengths))
        or not context_lengths
        or min(context_lengths) < int(context.get("sequence_tokens", 0))
    ):
        raise ValueError("KuaiRand hot-K/V scaling model geometry differs")


def _sample_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p25_ms": float(np.percentile(array, 25)),
        "p75_ms": float(np.percentile(array, 75)),
        "p95_ms": float(np.percentile(array, 95)),
        "minimum_ms": float(array.min()),
        "maximum_ms": float(array.max()),
        "stdev_ms": float(array.std(ddof=1)) if len(values) > 1 else 0.0,
    }


@torch.no_grad()
def _timed_forward(
    core,
    method: str,
    old_cache: HSTUKVCache,
    suffix: torch.Tensor,
    full_embedded: torch.Tensor,
) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if method == "reuse":
        hidden, cache = core.forward_with_cache_embedded(old_cache, suffix)
    elif method == "recompute":
        hidden, cache = core.forward_embedded(
            full_embedded,
            return_kv=True,
            return_hidden=True,
        )
    else:
        raise ValueError("KuaiRand hot-K/V scaling method differs")
    end.record()
    end.synchronize()
    if cache is None:
        raise RuntimeError("KuaiRand hot-K/V scaling output cache is missing")
    elapsed_ms = float(start.elapsed_time(end))
    checksum = float(
        hidden[:, -1].float().sum().item()
        + cache.k[:, :, -1, :1].float().sum().item()
    )
    del hidden, cache, start, end
    return elapsed_ms, checksum


def _point_id(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _point_path(root: Path, family: str, spec: dict[str, Any]) -> Path:
    return root / "points" / family / f"{_point_id(spec)}.json"


def _cached_point(
    path: Path,
    config_sha256: str,
    profile_name: str,
    family: str,
    spec: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document = json.loads(path.read_text())
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "complete"
        or document.get("config", {}).get("sha256") != config_sha256
        or document.get("profile") != profile_name
        or document.get("family") != family
        or document.get("spec") != spec
    ):
        raise ValueError("KuaiRand hot-K/V cached scaling point differs")
    return document


def _combine_samples(
    gathered: list[dict[str, Any]],
    request_counts: list[int],
    global_batch_size: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    maximum_batches = max(request_counts) // global_batch_size
    methods = {}
    for method in METHODS:
        keys = sorted(gathered[0]["samples"][method])
        if len(keys) != maximum_batches * repeats or any(
            sorted(payload["samples"][method]) != keys for payload in gathered
        ):
            raise RuntimeError("KuaiRand hot-K/V scaling rank samples differ")
        rank_max = {
            key: max(float(payload["samples"][method][key]) for payload in gathered)
            for key in keys
        }
        count_results = {}
        for request_count in request_counts:
            batches = request_count // global_batch_size
            pass_values = [
                sum(rank_max[f"{batch}:{repeat}"] for batch in range(batches))
                for repeat in range(repeats)
            ]
            summary = _sample_summary(pass_values)
            median = float(summary["median_ms"])
            count_results[str(request_count)] = {
                "serial_pass_cuda_ms": summary,
                "median_ms_per_request_amortized": median / request_count,
                "median_requests_per_second": request_count * 1000.0 / median,
                "batches_per_pass": batches,
            }
        methods[method] = {
            "max_rank_batch_cuda_ms": _sample_summary(list(rank_max.values())),
            "request_counts": count_results,
        }
    comparisons = {}
    for request_count in request_counts:
        key = str(request_count)
        reuse = float(
            methods["reuse"]["request_counts"][key]["serial_pass_cuda_ms"][
                "median_ms"
            ]
        )
        recompute = float(
            methods["recompute"]["request_counts"][key]["serial_pass_cuda_ms"][
                "median_ms"
            ]
        )
        comparisons[key] = {
            "absolute_median_difference_ms": recompute - reuse,
            "recompute_over_reuse_ratio": recompute / reuse,
            "reuse_time_saved_percent": 100.0 * (recompute - reuse) / recompute,
        }
    return methods, comparisons


@torch.no_grad()
def _measure_point(
    core,
    family: str,
    spec: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    profile_name: str,
    timing: dict[str, Any],
    capacity: dict[str, Any],
    output_path: Path,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, Any] | None:
    local_batch = int(spec["local_batch_size"])
    global_batch = local_batch * world_size
    request_counts = [int(value) for value in spec["request_counts"]]
    repeats = int(spec["measured_repeats"])
    maximum_batches = max(request_counts) // global_batch
    sequence_tokens = int(spec["sequence_tokens"])
    hidden_size = int(spec["hidden_size"])
    generator = torch.Generator(device=device).manual_seed(
        int(spec["tensor_seed"]) + rank * 1_000_003
    )
    full_embedded = torch.randn(
        local_batch,
        sequence_tokens,
        hidden_size,
        dtype=torch.float32,
        device=device,
        generator=generator,
    ) * 0.02
    _, old_cache = core.forward_embedded(
        full_embedded[:, :-2],
        return_kv=True,
        return_hidden=False,
    )
    if old_cache is None:
        raise RuntimeError("KuaiRand hot-K/V scaling source cache is missing")
    suffix = full_embedded[:, -2:]
    core.eval()
    torch.cuda.synchronize(device)
    for iteration in range(int(spec["warmup_iterations"])):
        methods = METHODS if iteration % 2 == 0 else tuple(reversed(METHODS))
        for method in methods:
            _timed_forward(core, method, old_cache, suffix, full_embedded)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    checksums: dict[str, list[float]] = {method: [] for method in METHODS}
    for repeat in range(repeats):
        for batch in range(maximum_batches):
            methods = (
                METHODS
                if (batch + repeat) % 2 == 0
                else tuple(reversed(METHODS))
            )
            for method in methods:
                elapsed, checksum = _timed_forward(
                    core,
                    method,
                    old_cache,
                    suffix,
                    full_embedded,
                )
                samples[method][f"{batch}:{repeat}"] = elapsed
                checksums[method].append(checksum)
    if not all(
        values and all(np.isfinite(value) for value in values)
        for values in checksums.values()
    ):
        raise RuntimeError("KuaiRand hot-K/V scaling checksum differs")
    rank_payload = {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "samples": samples,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    if dist.is_initialized():
        gathered: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, rank_payload)
    else:
        gathered = [rank_payload]
    if rank == 0:
        methods, comparisons = _combine_samples(
            gathered,
            request_counts,
            global_batch,
            repeats,
        )
        result = {
            "protocol": PROTOCOL,
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": config_sha256},
            "profile": profile_name,
            "family": family,
            "spec": spec,
            "timing_scope": timing,
            "capacity_background": capacity,
            "request_shape": {
                "sequence_tokens": sequence_tokens,
                "cached_prefix_tokens": sequence_tokens - 2,
                "reuse_suffix_tokens": 2,
                "global_batch_size": global_batch,
                "local_batch_size": local_batch,
            },
            "timed_core_geometry": {
                "layers": int(spec["layers"]),
                "hidden_size": hidden_size,
                "num_heads": int(spec["num_heads"]),
                "head_dim": int(spec["head_dim"]),
                "maximum_sequence_tokens": int(
                    spec["maximum_sequence_tokens"]
                ),
                "parameter_bytes": sum(
                    value.numel() * value.element_size()
                    for value in core.parameters()
                ),
            },
            "methods": methods,
            "comparisons": comparisons,
            "rank_runtime": gathered,
        }
        if local_batch == 1 and request_counts == [world_size]:
            result["one_request_per_rank_latency_ms"] = {
                method: methods[method]["max_rank_batch_cuda_ms"]["median_ms"]
                for method in METHODS
            }
        _atomic_json(output_path, result)
    else:
        result = None
    if dist.is_initialized():
        dist.barrier()
    del full_embedded, old_cache, suffix
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _synthetic_core(
    base_model: dict[str, Any],
    layers: int,
    hidden_size: int,
    head_dim: int,
    maximum_sequence_tokens: int,
    seed: int,
    device: torch.device,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = HSTUConfig(
        num_items=1,
        num_prediction_items=1,
        num_behaviors=1,
        hidden_size=hidden_size,
        num_layers=layers,
        num_heads=hidden_size // head_dim,
        head_dim=head_dim,
        max_seq_len=maximum_sequence_tokens,
        input_dropout=0.0,
        activation=str(base_model["activation"]),
        qk_scale=float(base_model["qk_scale"]),
        gating=str(base_model["gating"]),
        block_variant=str(base_model["block_variant"]),
        relative_position_bias=bool(base_model["relative_position_bias"]),
        causal_diagonal=str(base_model["causal_diagonal"]),
    )
    model = ExternalEmbeddingHSTU(config).to(device)
    model.eval()
    return model.core


def _spec(
    source: str,
    layers: int,
    hidden_size: int,
    sequence_tokens: int,
    local_batch_size: int,
    request_counts: list[int],
    warmup_iterations: int,
    measured_repeats: int,
    tensor_seed: int,
    maximum_sequence_tokens: int,
    head_dim: int,
) -> dict[str, Any]:
    return {
        "source": source,
        "layers": layers,
        "hidden_size": hidden_size,
        "sequence_tokens": sequence_tokens,
        "local_batch_size": local_batch_size,
        "request_counts": request_counts,
        "warmup_iterations": warmup_iterations,
        "measured_repeats": measured_repeats,
        "tensor_seed": tensor_seed,
        "maximum_sequence_tokens": maximum_sequence_tokens,
        "head_dim": head_dim,
        "num_heads": hidden_size // head_dim,
    }


def _task_specs(profile: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    common = profile["common"]
    head_dim = int(common["head_dim"])
    maximum = int(common["maximum_sequence_tokens"])
    seed = int(common["seed"])
    tasks = []
    anchor = profile["checkpoint_anchor"]
    tasks.append(
        (
            "checkpoint_anchor",
            _spec(
                "selected_theta8_checkpoint",
                8,
                512,
                int(anchor["sequence_tokens"]),
                int(anchor["local_batch_size"]),
                [int(anchor["requests"])],
                int(anchor["warmup_iterations"]),
                int(anchor["measured_repeats"]),
                seed + 11,
                512,
                head_dim,
            ),
        )
    )
    standalone = profile["standalone_latency"]
    for index, length in enumerate(standalone["sequence_tokens"]):
        tasks.append(
            (
                "standalone_latency",
                _spec(
                    "synthetic_timing_core",
                    8,
                    512,
                    int(length),
                    int(standalone["local_batch_size"]),
                    [int(standalone["requests"])],
                    int(standalone["warmup_iterations"]),
                    int(standalone["measured_repeats"]),
                    seed + 101 + index,
                    maximum,
                    head_dim,
                ),
            )
        )
    sequence = profile["sequence_scaling"]
    for index, length in enumerate(sequence["sequence_tokens"]):
        tasks.append(
            (
                "sequence_scaling",
                _spec(
                    "synthetic_timing_core",
                    8,
                    512,
                    int(length),
                    int(sequence["local_batch_size"]),
                    [int(sequence["requests"])],
                    int(sequence["warmup_iterations"]),
                    int(sequence["measured_repeats"]),
                    seed + 1001 + index,
                    maximum,
                    head_dim,
                ),
            )
        )
    requests = profile["request_scaling"]
    tasks.append(
        (
            "request_scaling",
            _spec(
                "synthetic_timing_core",
                8,
                512,
                int(requests["sequence_tokens"]),
                int(requests["local_batch_size"]),
                [int(value) for value in requests["request_counts"]],
                int(requests["warmup_iterations"]),
                int(requests["measured_repeats"]),
                seed + 2001,
                maximum,
                head_dim,
            ),
        )
    )
    depth = profile["depth_length_scaling"]
    for layer_index, layers in enumerate(depth["layers"]):
        for length_index, length in enumerate(depth["sequence_tokens"]):
            tasks.append(
                (
                    "depth_length_scaling",
                    _spec(
                        "synthetic_timing_core",
                        int(layers),
                        512,
                        int(length),
                        int(depth["local_batch_size"]),
                        [int(depth["requests"])],
                        int(depth["warmup_iterations"]),
                        int(depth["measured_repeats"]),
                        seed + 3001 + layer_index * 101 + length_index,
                        maximum,
                        head_dim,
                    ),
                )
            )
    width = profile["width_scaling"]
    for index, hidden_size in enumerate(width["hidden_sizes"]):
        tasks.append(
            (
                "width_scaling",
                _spec(
                    "synthetic_timing_core",
                    int(width["layers"]),
                    int(hidden_size),
                    int(width["sequence_tokens"]),
                    int(width["local_batch_size"]),
                    [int(width["requests"])],
                    int(width["warmup_iterations"]),
                    int(width["measured_repeats"]),
                    seed + 4001 + index,
                    maximum,
                    head_dim,
                ),
            )
        )
    context = profile["context_capacity_scaling"]
    for index, context_tokens in enumerate(context["maximum_sequence_tokens"]):
        tasks.append(
            (
                "context_capacity_scaling",
                _spec(
                    "synthetic_timing_core",
                    int(context["layers"]),
                    int(context["hidden_size"]),
                    int(context["sequence_tokens"]),
                    int(context["local_batch_size"]),
                    [int(context["requests"])],
                    int(context["warmup_iterations"]),
                    int(context["measured_repeats"]),
                    seed + 5001 + index,
                    int(context_tokens),
                    head_dim,
                ),
            )
        )
    if len({(family, _point_id(spec)) for family, spec in tasks}) != len(tasks):
        raise ValueError("KuaiRand hot-K/V scaling task identity differs")
    if any(spec["hidden_size"] % head_dim for _, spec in tasks):
        raise ValueError("KuaiRand hot-K/V scaling task head geometry differs")
    return tasks


def _summarize_point(point: dict[str, Any]) -> dict[str, Any]:
    count = str(max(int(value) for value in point["spec"]["request_counts"]))
    reuse = point["methods"]["reuse"]["request_counts"][count]
    recompute = point["methods"]["recompute"]["request_counts"][count]
    comparison = point["comparisons"][count]
    result = {
        "family": point["family"],
        "source": point["spec"]["source"],
        "layers": point["spec"]["layers"],
        "hidden_size": point["spec"]["hidden_size"],
        "num_heads": point["spec"]["num_heads"],
        "head_dim": point["spec"]["head_dim"],
        "maximum_sequence_tokens": point["spec"]["maximum_sequence_tokens"],
        "timed_core_parameter_bytes": point["timed_core_geometry"][
            "parameter_bytes"
        ],
        "timed_core_parameter_gib": point["timed_core_geometry"][
            "parameter_bytes"
        ]
        / (1 << 30),
        "sequence_tokens": point["spec"]["sequence_tokens"],
        "requests": int(count),
        "global_batch_size": point["request_shape"]["global_batch_size"],
        "reuse_median_total_ms": reuse["serial_pass_cuda_ms"]["median_ms"],
        "recompute_median_total_ms": recompute["serial_pass_cuda_ms"]["median_ms"],
        "reuse_median_ms_per_request_amortized": reuse[
            "median_ms_per_request_amortized"
        ],
        "recompute_median_ms_per_request_amortized": recompute[
            "median_ms_per_request_amortized"
        ],
        **comparison,
    }
    if "one_request_per_rank_latency_ms" in point:
        result["reuse_one_request_per_rank_latency_ms"] = point[
            "one_request_per_rank_latency_ms"
        ]["reuse"]
        result["recompute_one_request_per_rank_latency_ms"] = point[
            "one_request_per_rank_latency_ms"
        ]["recompute"]
    return result


def _summary_document(
    config_path: Path,
    config_sha256: str,
    profile_name: str,
    timing: dict[str, Any],
    capacity: dict[str, Any],
    points: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    compact = [_summarize_point(point) for point in points]
    request_point = next(
        point for point in points if point["family"] == "request_scaling"
    )
    request_rows = []
    for count in request_point["spec"]["request_counts"]:
        key = str(count)
        reuse = request_point["methods"]["reuse"]["request_counts"][key]
        recompute = request_point["methods"]["recompute"]["request_counts"][key]
        request_rows.append(
            {
                "requests": count,
                "reuse_median_total_ms": reuse["serial_pass_cuda_ms"]["median_ms"],
                "recompute_median_total_ms": recompute["serial_pass_cuda_ms"][
                    "median_ms"
                ],
                "reuse_median_ms_per_request_amortized": reuse[
                    "median_ms_per_request_amortized"
                ],
                "recompute_median_ms_per_request_amortized": recompute[
                    "median_ms_per_request_amortized"
                ],
                **request_point["comparisons"][key],
            }
        )
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": config_sha256},
        "profile": profile_name,
        "timing_scope": timing,
        "capacity_background": capacity,
        "points": compact,
        "tables": {
            "checkpoint_anchor": [
                row for row in compact if row["family"] == "checkpoint_anchor"
            ],
            "standalone_latency": [
                row for row in compact if row["family"] == "standalone_latency"
            ],
            "sequence_scaling": [
                row for row in compact if row["family"] == "sequence_scaling"
            ],
            "request_scaling": request_rows,
            "depth_length_scaling": [
                row for row in compact if row["family"] == "depth_length_scaling"
            ],
            "width_scaling": [
                row for row in compact if row["family"] == "width_scaling"
            ],
            "context_capacity_scaling": [
                row
                for row in compact
                if row["family"] == "context_capacity_scaling"
            ],
        },
        "point_files": [
            {
                "family": point["family"],
                "spec": point["spec"],
                "path": point["result_path"],
                "sha256": point["result_sha256"],
            }
            for point in points
        ],
        "elapsed_seconds_including_untimed_setup": elapsed_seconds,
    }


def preflight_scaling(config_path: str | Path, profile_name: str) -> dict[str, Any]:
    path = Path(config_path)
    document = load_scaling_config(path)
    profile = document["profiles"][profile_name]
    baseline_path = Path(document["baseline"]["large_config"]["path"])
    baseline = load_persistent_config(baseline_path)
    baseline_sha256 = file_sha256(baseline_path)
    version = int(document["baseline"]["checkpoint_version"])
    manifest = _read_manifest(
        Path(baseline["outputs"]["checkpoint_root"]),
        version,
        baseline,
        baseline_sha256,
        False,
    )
    tasks = _task_specs(profile)
    return {
        "protocol": PROTOCOL,
        "status": "ready",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(path), "sha256": file_sha256(path)},
        "profile": profile_name,
        "task_count": len(tasks),
        "families": {
            family: sum(1 for task_family, _ in tasks if task_family == family)
            for family in sorted({family for family, _ in tasks})
        },
        "maximum_sequence_tokens": max(
            int(spec["sequence_tokens"]) for _, spec in tasks
        ),
        "maximum_layers": max(int(spec["layers"]) for _, spec in tasks),
        "maximum_hidden_size": max(
            int(spec["hidden_size"]) for _, spec in tasks
        ),
        "maximum_requests": max(
            max(int(value) for value in spec["request_counts"])
            for _, spec in tasks
        ),
        "baseline": {
            "checkpoint_version": version,
            "checkpoint_bytes": int(manifest["checkpoint_bytes"]),
            "global_model_parameter_bytes": int(
                manifest["geometry"]["global_model_parameter_bytes"]
            ),
            "manifest_sha256": file_sha256(
                _manifest_path(Path(baseline["outputs"]["checkpoint_root"]), version)
            ),
        },
    }


def run_scaling(config_path: str | Path, profile_name: str) -> dict[str, Any]:
    started = time.monotonic()
    path = Path(config_path)
    document = load_scaling_config(path)
    config_sha256 = file_sha256(path)
    profile = document["profiles"][profile_name]
    output_root = Path(document["outputs"]["root"]) / profile_name
    result_path = output_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if (
            result.get("protocol") != PROTOCOL
            or result.get("status") != "complete"
            or result.get("config", {}).get("sha256") != config_sha256
            or result.get("profile") != profile_name
        ):
            raise ValueError("KuaiRand hot-K/V scaling cached summary differs")
        return result
    baseline_path = Path(document["baseline"]["large_config"]["path"])
    baseline = load_persistent_config(baseline_path)
    baseline_sha256 = file_sha256(baseline_path)
    rank, world_size, device = _distributed(baseline)
    try:
        _seed(int(profile["common"]["seed"]))
        base_config = load_config(baseline["parent"]["base_config"]["path"])
        version = int(document["baseline"]["checkpoint_version"])
        manifest = json.loads(
            _manifest_path(
                Path(baseline["outputs"]["checkpoint_root"]), version
            ).read_text()
        )
        semantic_rows = int(manifest["geometry"]["semantic_embedding_rows"])
        dense, embedding, tracker, geometry = _initialize_model(
            baseline,
            base_config,
            semantic_rows,
            rank,
            world_size,
            device,
        )
        _load_checkpoint(
            Path(baseline["outputs"]["checkpoint_root"]),
            version,
            dense,
            embedding,
            tracker,
            baseline,
            baseline_sha256,
            rank,
        )
        if int(geometry["global_model_parameter_bytes"]) != int(
            document["baseline"]["required_global_model_parameter_bytes"]
        ):
            raise RuntimeError("KuaiRand hot-K/V scaling capacity differs")
        capacity = {
            "selected_checkpoint_version": version,
            "selected_checkpoint_manifest_sha256": file_sha256(
                _manifest_path(
                    Path(baseline["outputs"]["checkpoint_root"]), version
                )
            ),
            "global_model_parameter_bytes": int(
                geometry["global_model_parameter_bytes"]
            ),
            "global_model_parameter_gib": float(
                geometry["global_model_parameter_gib"]
            ),
            "global_embedding_bytes": int(geometry["global_embedding_bytes"]),
            "embedding_residency": "selected_checkpoint_shards_resident_in_hbm",
            "embedding_lookup_timed": False,
        }
        tasks = _task_specs(profile)
        points = []
        synthetic_core = None
        synthetic_geometry = None
        for family, spec in tasks:
            point_path = _point_path(output_root, family, spec)
            if rank == 0:
                cached = _cached_point(
                    point_path,
                    config_sha256,
                    profile_name,
                    family,
                    spec,
                )
            else:
                cached = None
            cached = _broadcast(cached, rank)
            if cached is not None:
                point = cached
            else:
                geometry_key = (
                    int(spec["layers"]),
                    int(spec["hidden_size"]),
                    int(spec["maximum_sequence_tokens"]),
                )
                if spec["source"] == "selected_theta8_checkpoint":
                    core = dense.core
                else:
                    if synthetic_geometry != geometry_key:
                        del synthetic_core
                        gc.collect()
                        torch.cuda.empty_cache()
                        synthetic_core = _synthetic_core(
                            base_config["model"],
                            geometry_key[0],
                            geometry_key[1],
                            int(profile["common"]["head_dim"]),
                            geometry_key[2],
                            int(profile["common"]["seed"])
                            + geometry_key[0] * 1009
                            + geometry_key[1] * 17,
                            device,
                        )
                        synthetic_geometry = geometry_key
                    core = synthetic_core
                point = _measure_point(
                    core,
                    family,
                    spec,
                    path,
                    config_sha256,
                    profile_name,
                    document["timing"],
                    capacity,
                    point_path,
                    rank,
                    world_size,
                    device,
                )
                point = _broadcast(point, rank)
                if rank == 0:
                    print(
                        f"phase=kuairand_hotkv_scaling family={family} "
                        f"layers={spec['layers']} hidden={spec['hidden_size']} "
                        f"tokens={spec['sequence_tokens']} "
                        f"requests={max(spec['request_counts'])} "
                        f"ratio={_summarize_point(point)['recompute_over_reuse_ratio']:.3f}",
                        flush=True,
                    )
            if rank == 0:
                points.append(
                    point
                    | {
                        "result_path": str(point_path),
                        "result_sha256": file_sha256(point_path),
                    }
                )
        del synthetic_core, dense, embedding, tracker
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            result = _summary_document(
                path,
                config_sha256,
                profile_name,
                document["timing"],
                capacity,
                points,
                time.monotonic() - started,
            )
            _atomic_json(result_path, result)
        else:
            result = None
        if dist.is_initialized():
            dist.barrier()
        result = _broadcast(result, rank)
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
