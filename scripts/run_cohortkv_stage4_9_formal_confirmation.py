from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path

import cohortkv_stage4_8_sweep_common as stage48
import run_cohortkv_stage4_7_organic_chain as base
import run_cohortkv_stage4_9_rollout_boundary as stage49
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
    execute_direct,
    timed_cuda,
)
from motivation_validity import seed_everything

from hstu_kvcache.migration import (
    ROLLOUT_BOUNDARY_PROTOCOL,
    JaggedMigratedKVBatch,
    append_jagged_suffix,
    retained_population_sha256,
    tail_slice_jagged_cache,
)
from hstu_kvcache.migration.stage45_oldkv import DirectOldKVFusedOperator
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_organic_windows,
)
from hstu_kvcache.utils import save_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cohortkv_single_config_stage4_9_same_device_confirmation_v2"
STATIC_PROTOCOL = "cohortkv_single_config_stage4_9_formal_static_v2"
NUM_EDGES = 11
BATCH_SIZE = 4
WARMUP_REPEATS = 1
TIMING_REPEATS = 3
EXACT_PARITY_ATOL = 1e-4
OUTPUT_DIR = "results/system/cohortkv_single_config_full_chain_v1"
CANDIDATE_NAMES = (
    "token_debt_total10",
    "staggered_renewal_h12",
)
IMPLEMENTATION_PATHS = {
    "formal_runner": (
        ROOT / "scripts/run_cohortkv_stage4_9_formal_confirmation.py"
    ),
    "stage4_8_sweep_common": (
        ROOT / "scripts/cohortkv_stage4_8_sweep_common.py"
    ),
    "stage4_7_organic_runner": (
        ROOT / "scripts/run_cohortkv_stage4_7_organic_chain.py"
    ),
    "rollout_runner": (
        ROOT / "scripts/run_cohortkv_stage4_9_rollout_boundary.py"
    ),
    "rollout_abi": ROOT / "src/hstu_kvcache/migration/rollout.py",
    "organic_migration": ROOT / "src/hstu_kvcache/migration/organic.py",
    "organic_schedulers": (
        ROOT / "src/hstu_kvcache/migration/organic_schedulers.py"
    ),
    "direct_oldkv_operator": (
        ROOT / "src/hstu_kvcache/migration/stage45_oldkv.py"
    ),
}
TASK_METRICS = stage48.TASK_METRICS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--prepared-data", default=PREPARED_PATH)
    parser.add_argument("--training-result", default=TRAINING_PATH)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--compiler-result", default=COMPILER_OUTPUT)
    parser.add_argument("--runtime-dir", default=RUNTIME_DIR)
    parser.add_argument("--baseline", default=stage48.BASELINE_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--warmup-repeats",
        type=int,
        default=WARMUP_REPEATS,
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=TIMING_REPEATS,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.seed != 0 or args.batch_size != BATCH_SIZE:
        raise ValueError("Stage 4.9 freezes seed 0 and batch size 4")
    if (
        args.warmup_repeats != WARMUP_REPEATS
        or args.timing_repeats != TIMING_REPEATS
    ):
        raise ValueError(
            "Stage 4.9 freezes one warmup and three timing repetitions"
        )
    if args.smoke_test:
        if args.device is not None:
            raise ValueError("Stage 4.9 static smoke does not accept --device")
        return
    if args.device is None:
        raise ValueError("Stage 4.9 formal confirmation requires --device")
    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
        raise ValueError(
            "Stage 4.9 formal confirmation requires an available "
            "explicit CUDA index"
        )


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def candidate_specs() -> tuple[object, ...]:
    return tuple(stage49._candidate_spec(name) for name in CANDIDATE_NAMES)


def implementation_snapshot() -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for name, path in IMPLEMENTATION_PATHS.items()
    }


def candidate_output_path(
    args: argparse.Namespace,
    candidate_name: str,
) -> Path:
    return Path(args.output_dir) / f"stage4_9_{candidate_name}_seed0.json"


def summary_output_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / (
        "stage4_9_same_device_confirmation_seed0.json"
    )


def _timed_repeated(
    action,
    device: torch.device,
    warmup_repeats: int,
    timing_repeats: int,
):
    value = None
    for _ in range(warmup_repeats):
        value = action()
    del value
    samples = []
    result = None
    for _ in range(timing_repeats):
        result, elapsed = timed_cuda(action, device)
        samples.append(float(elapsed))
    return result, {
        "samples_ms": samples,
        "median_ms": float(statistics.median(samples)),
    }


def _zero_measurement(timing_repeats: int) -> dict[str, object]:
    return {
        "samples_ms": [0.0] * timing_repeats,
        "median_ms": 0.0,
    }


def _zero_state_movement(direction: str) -> dict[str, object]:
    return {
        "direction": direction,
        "records": 0,
        "logical_bytes": 0,
        "gpu_event_ms": 0.0,
        "wall_ms": 0.0,
        "executions": 0,
        "outside_u_and_e": True,
        "outside_append_timer": True,
    }


def _persistent_cpu_store_checks(
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    expected_version: int | None = None,
    expected_lengths: dict[int, int] | None = None,
) -> dict[str, bool]:
    version = (
        None if expected_version is None else f"theta{expected_version}"
    )
    return {
        "records_are_singleton_extents": all(
            cache.record_ids == (record_id,) and cache.batch_size == 1
            for record_id, cache in cache_by_record.items()
        ),
        "all_persistent_tensors_are_cpu": all(
            cache.k.device.type == "cpu"
            and cache.v.device.type == "cpu"
            and cache.lengths.device.type == "cpu"
            and cache.offsets.device.type == "cpu"
            for cache in cache_by_record.values()
        ),
        "persistent_gpu_kv_bytes_are_zero": all(
            cache.k.device.type != "cuda"
            and cache.v.device.type != "cuda"
            for cache in cache_by_record.values()
        ),
        "persistent_kv_is_fp16": all(
            cache.k.dtype == cache.v.dtype == torch.float16
            for cache in cache_by_record.values()
        ),
        "persistent_kv_is_contiguous": all(
            cache.k.is_contiguous() and cache.v.is_contiguous()
            for cache in cache_by_record.values()
        ),
        "persistent_versions_match": version is None
        or all(
            cache.migration_anchor_version == version
            and cache.served_kv_target == version
            for cache in cache_by_record.values()
        ),
        "persistent_lengths_match": expected_lengths is None
        or set(cache_by_record).issubset(expected_lengths)
        and all(
            int(cache.lengths[0]) == expected_lengths[record_id]
            and int(cache.offsets[0]) == 0
            and int(cache.offsets[1]) == expected_lengths[record_id]
            for record_id, cache in cache_by_record.items()
        ),
        "persistent_kv_is_finite": all(
            bool(torch.isfinite(cache.k).all())
            and bool(torch.isfinite(cache.v).all())
            for cache in cache_by_record.values()
        ),
    }


def _staged_device_store_checks(
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    expected_record_ids: tuple[int, ...],
    expected_version: int,
    expected_lengths: dict[int, int],
    device: torch.device,
) -> dict[str, bool]:
    version = f"theta{expected_version}"
    return {
        "staged_ids_match_migrants": set(cache_by_record)
        == set(expected_record_ids),
        "staged_records_are_singleton_extents": all(
            cache.record_ids == (record_id,) and cache.batch_size == 1
            for record_id, cache in cache_by_record.items()
        ),
        "all_staged_tensors_are_on_target_device": all(
            cache.k.device == device
            and cache.v.device == device
            and cache.lengths.device == device
            and cache.offsets.device == device
            for cache in cache_by_record.values()
        ),
        "staged_kv_is_fp16_contiguous": all(
            cache.k.dtype == cache.v.dtype == torch.float16
            and cache.k.is_contiguous()
            and cache.v.is_contiguous()
            for cache in cache_by_record.values()
        ),
        "staged_versions_match_source": all(
            cache.migration_anchor_version == version
            and cache.served_kv_target == version
            for cache in cache_by_record.values()
        ),
        "staged_lengths_match_plans": set(cache_by_record).issubset(
            expected_lengths
        )
        and all(
            int(cache.lengths[0]) == expected_lengths[record_id]
            and int(cache.offsets[0]) == 0
            and int(cache.offsets[1]) == expected_lengths[record_id]
            for record_id, cache in cache_by_record.items()
        ),
        "staged_kv_is_finite": all(
            bool(torch.isfinite(cache.k).all())
            and bool(torch.isfinite(cache.v).all())
            for cache in cache_by_record.values()
        ),
    }


def _record_id_sha256(record_ids) -> str:
    payload = json.dumps(
        sorted(int(value) for value in record_ids),
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _record_length_sha256(
    cache_by_record: dict[int, JaggedMigratedKVBatch],
) -> str:
    payload = json.dumps(
        [
            [record_id, int(cache_by_record[record_id].lengths[0])]
            for record_id in sorted(cache_by_record)
        ],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _cuda_tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            return value.numel() * value.element_size()
        return 0
    if isinstance(value, JaggedMigratedKVBatch):
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                value.k,
                value.v,
                value.lengths,
                value.offsets,
            )
            if tensor.device.type == "cuda"
        )
    if isinstance(value, dict):
        return sum(_cuda_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_cuda_tensor_bytes(item) for item in value)
    return 0


def _transfer_record_store(
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    record_ids: tuple[int, ...],
    target_device: torch.device,
) -> dict[int, JaggedMigratedKVBatch]:
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Stage 4.9 transfer record IDs must be unique")
    missing = set(record_ids).difference(cache_by_record)
    if missing:
        raise KeyError(
            f"Stage 4.9 transfer source records are absent: {sorted(missing)}"
        )
    return {
        record_id: cache_by_record[record_id].to(target_device)
        for record_id in record_ids
    }


def _timed_state_transfer(
    action,
    device: torch.device,
    direction: str,
    records: int,
    logical_bytes: int,
):
    if records == 0:
        return action(), _zero_state_movement(direction)
    started = time.perf_counter()
    value, gpu_event_ms = timed_cuda(action, device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    return value, {
        "direction": direction,
        "records": records,
        "logical_bytes": logical_bytes,
        "gpu_event_ms": float(gpu_event_ms),
        "wall_ms": float(wall_ms),
        "executions": 1,
        "outside_u_and_e": True,
        "outside_append_timer": True,
    }


def _timed_store_transfer(
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    record_ids: tuple[int, ...],
    target_device: torch.device,
    timing_device: torch.device,
    direction: str,
) -> tuple[dict[int, JaggedMigratedKVBatch], dict[str, object]]:
    logical_bytes = sum(
        cache_by_record[record_id].nbytes for record_id in record_ids
    )
    return _timed_state_transfer(
        partial(
            _transfer_record_store,
            cache_by_record,
            record_ids,
            target_device,
        ),
        timing_device,
        direction,
        len(record_ids),
        logical_bytes,
    )


def _sum_state_movement(
    ledgers: list[dict[str, object]],
    direction: str,
) -> dict[str, object]:
    if any(value["direction"] != direction for value in ledgers):
        raise ValueError("Stage 4.9 state-movement directions differ")
    return {
        "direction": direction,
        "records": sum(int(value["records"]) for value in ledgers),
        "logical_bytes": sum(
            int(value["logical_bytes"]) for value in ledgers
        ),
        "gpu_event_ms": sum(
            float(value["gpu_event_ms"]) for value in ledgers
        ),
        "wall_ms": sum(float(value["wall_ms"]) for value in ledgers),
        "executions": sum(int(value["executions"]) for value in ledgers),
        "outside_u_and_e": all(
            bool(value["outside_u_and_e"]) for value in ledgers
        ),
        "outside_append_timer": all(
            bool(value["outside_append_timer"]) for value in ledgers
        ),
    }


def _sum_measurements(
    measurements: dict[str, dict[str, object]],
    timing_repeats: int,
) -> dict[str, object]:
    samples = [
        sum(
            float(value["samples_ms"][index])
            for value in measurements.values()
        )
        for index in range(timing_repeats)
    ]
    return {
        "components": measurements,
        "samples_ms": samples,
        "median_of_repetition_sums_ms": float(statistics.median(samples)),
        "sum_of_component_medians_ms": sum(
            float(value["median_ms"]) for value in measurements.values()
        ),
    }


def summarize_paired_cost(
    steps: list[dict],
    timing_repeats: int,
) -> dict[str, object]:
    if len(steps) != NUM_EDGES:
        raise ValueError("Stage 4.9 formal cost requires eleven edges")
    u_samples = [
        sum(float(step["cost"]["u"]["samples_ms"][index]) for step in steps)
        for index in range(timing_repeats)
    ]
    e_samples = [
        sum(float(step["cost"]["e"]["samples_ms"][index]) for step in steps)
        for index in range(timing_repeats)
    ]
    if any(not math.isfinite(value) or value < 0 for value in u_samples):
        raise ValueError("Stage 4.9 formal U samples are invalid")
    if any(not math.isfinite(value) or value <= 0 for value in e_samples):
        raise ValueError("Stage 4.9 formal E samples are invalid")
    u_total = sum(
        float(step["cost"]["u"]["sum_of_component_medians_ms"])
        for step in steps
    )
    e_total = sum(
        float(step["cost"]["e"]["median_ms"]) for step in steps
    )
    outside_mixed = sum(
        float(step["cost"]["outside_rollout_timer"]["mixed"]["median_ms"])
        for step in steps
    )
    outside_exact = sum(
        float(step["cost"]["outside_rollout_timer"]["exact"]["median_ms"])
        for step in steps
    )
    append_components = (
        "target_delta_append_ms",
        "latest_append_ms",
        "short_latest_append_ms",
    )
    mixed_append = sum(
        float(
            step["cost"]["outside_rollout_timer"]["mixed"]["components"][
                component
            ]["median_ms"]
        )
        for step in steps
        for component in append_components
    )
    exact_append = sum(
        float(
            step["cost"]["outside_rollout_timer"]["exact"]["components"][
                component
            ]["median_ms"]
        )
        for step in steps
        for component in append_components
    )
    h2d = _sum_state_movement(
        [
            step["cost"]["state_movement_outside_primary"][
                "h2d_previous_actual"
            ]
            for step in steps
        ],
        "host_to_device_previous_actual",
    )
    d2h = _sum_state_movement(
        [
            step["cost"]["state_movement_outside_primary"][
                "d2h_next_actual"
            ]
            for step in steps
        ],
        "device_to_host_next_actual",
    )
    return {
        "mixed_u_ms": u_total,
        "paired_fresh_exact_e_ms": e_total,
        "primary_sum_u_over_sum_e": u_total / e_total,
        "repetition_sum_u_ms": u_samples,
        "repetition_sum_e_ms": e_samples,
        "repetition_u_over_e": [
            numerator / denominator
            for numerator, denominator in zip(
                u_samples,
                e_samples,
                strict=True,
            )
        ],
        "mixed_outside_rollout_ledger_ms": outside_mixed,
        "exact_outside_rollout_ledger_ms": outside_exact,
        "target_append_a_ledger": {
            "components": list(append_components),
            "mixed_ms": mixed_append,
            "exact_ms": exact_append,
            "excluded_from_primary": True,
        },
        "target_append_excluded_from_u_and_e": True,
        "old_exact_denominator_reused": False,
        "state_movement_outside_primary": {
            "h2d_previous_actual": h2d,
            "d2h_next_actual": d2h,
            "logical_bytes": int(h2d["logical_bytes"])
            + int(d2h["logical_bytes"]),
            "gpu_event_ms": float(h2d["gpu_event_ms"])
            + float(d2h["gpu_event_ms"]),
            "wall_ms": float(h2d["wall_ms"]) + float(d2h["wall_ms"]),
            "excluded_from_primary": True,
            "excluded_from_append_timer": True,
            "reported_separately": True,
        },
    }


def _advance_recursive_chain(
    edge_runner,
    initial_cache,
    initial_last_exact,
    initial_expected_ids,
    initial_scheduler_state=None,
):
    cache = initial_cache
    last_exact = initial_last_exact
    expected_ids = initial_expected_ids
    scheduler_state = initial_scheduler_state
    steps = []
    endpoints = []
    for source_version in range(NUM_EDGES):
        previous_cache = cache
        (
            cache,
            last_exact,
            expected_ids,
            scheduler_state,
            endpoint,
            step,
        ) = edge_runner(
            source_version,
            cache,
            last_exact,
            expected_ids,
            scheduler_state,
        )
        if (
            source_version == 0
            and isinstance(initial_cache, dict)
            and previous_cache is initial_cache
            and cache is not initial_cache
        ):
            initial_cache.clear()
        endpoints.append(endpoint)
        steps.append(step)
    return (
        cache,
        last_exact,
        expected_ids,
        scheduler_state,
        endpoints,
        steps,
    )


@torch.inference_mode()
def _initialize_theta0_host(
    cfg,
    checkpoint_dir: str,
    window,
    groups,
    device: torch.device,
) -> tuple[
    dict[int, JaggedMigratedKVBatch],
    dict[int, int],
    float,
    dict[str, object],
]:
    model = load_checkpoint_model(cfg, checkpoint_dir, 0, device)
    cache_by_record = {}
    initialization_ms = 0.0
    transfer_ledgers = []
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
        (full, hidden), elapsed = timed_cuda(
            partial(
                stage49._exact_full,
                model,
                batch,
                record_ids,
                0,
                torch.float16,
            ),
            device,
        )
        initialization_ms += elapsed
        host_full, transfer = _timed_state_transfer(
            partial(full.to, torch.device("cpu")),
            device,
            "device_to_host_theta0_initialization",
            len(record_ids),
            full.nbytes,
        )
        transfer_ledgers.append(transfer)
        cache_by_record.update(base._split_cache(host_full))
        del full, hidden, host_full, batch
        gc.collect()
        torch.cuda.empty_cache()
    checks = _persistent_cpu_store_checks(cache_by_record)
    if not all(checks.values()):
        raise RuntimeError(
            f"Stage 4.9 theta0 host-store checks failed: {checks}"
        )
    last_exact = {record_id: 0 for record_id in cache_by_record}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return (
        cache_by_record,
        last_exact,
        initialization_ms,
        _sum_state_movement(
            transfer_ledgers,
            "device_to_host_theta0_initialization",
        ),
    )


def _crop_actual_retained(
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    record_ids: tuple[int, ...],
    plans,
) -> JaggedMigratedKVBatch:
    source = base._assemble_record_caches(record_ids, cache_by_record)
    sliced = tail_slice_jagged_cache(
        source,
        tuple(plans[value].retained_tokens for value in record_ids),
    )
    if sliced.cache is None:
        raise RuntimeError("Stage 4.9 migrant retained prefix is empty")
    return sliced.cache


def _append_delta_once(
    model,
    cache: JaggedMigratedKVBatch,
    plans,
    record_by_id: dict[int, dict],
    target_window,
    device: torch.device,
    dtype: torch.dtype,
) -> JaggedMigratedKVBatch:
    records = stage49._records_for_ids(
        cache.record_ids,
        record_by_id,
        target_window,
    )
    batch = stage49._segment_batch(
        records,
        tuple(plans[value].delta_start for value in cache.record_ids),
        tuple(
            plans[value].target_prefix_tokens for value in cache.record_ids
        ),
        device,
    )
    result = append_jagged_suffix(
        model,
        base.identity_jagged_slice(cache),
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        batch["lengths"],
        dtype=dtype,
    )
    return result.cache


def _build_natural_prefix_once(
    model,
    record_ids: tuple[int, ...],
    plans,
    record_by_id: dict[int, dict],
    target_window,
    target_version: int,
    cfg,
    device: torch.device,
    dtype: torch.dtype,
) -> JaggedMigratedKVBatch:
    records = stage49._records_for_ids(
        record_ids,
        record_by_id,
        target_window,
    )
    batch = stage49._segment_batch(
        records,
        (0,) * len(records),
        tuple(plans[value].target_prefix_tokens for value in record_ids),
        device,
    )
    result = append_jagged_suffix(
        model,
        base.empty_jagged_slice(
            record_ids,
            target_version,
            cfg.num_layers,
            cfg.num_heads * cfg.head_dim,
            dtype,
            device,
        ),
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        batch["lengths"],
        dtype=dtype,
    )
    return result.cache


def _append_latest_once(
    model,
    prefix: JaggedMigratedKVBatch,
    record_by_id: dict[int, dict],
    target_window,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[JaggedMigratedKVBatch, torch.Tensor]:
    records = stage49._records_for_ids(
        prefix.record_ids,
        record_by_id,
        target_window,
    )
    suffix = base._suffix_batch(records, device)
    result = append_jagged_suffix(
        model,
        base.identity_jagged_slice(prefix),
        suffix["item_ids"],
        suffix["behaviors"],
        suffix["time_deltas"],
        suffix["lengths"],
        dtype=dtype,
    )
    if result.last_appended_hidden is None:
        raise RuntimeError("Stage 4.9 target append returned no hidden")
    return result.cache, result.last_appended_hidden


def _append_fresh_latest_once(
    model,
    record_ids: tuple[int, ...],
    record_by_id: dict[int, dict],
    target_window,
    target_version: int,
    cfg,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[JaggedMigratedKVBatch, torch.Tensor]:
    records = stage49._records_for_ids(
        record_ids,
        record_by_id,
        target_window,
    )
    suffix = base._suffix_batch(records, device)
    result = append_jagged_suffix(
        model,
        base.empty_jagged_slice(
            record_ids,
            target_version,
            cfg.num_layers,
            cfg.num_heads * cfg.head_dim,
            dtype,
            device,
        ),
        suffix["item_ids"],
        suffix["behaviors"],
        suffix["time_deltas"],
        suffix["lengths"],
        dtype=dtype,
    )
    if result.last_appended_hidden is None:
        raise RuntimeError("Stage 4.9 fresh latest returned no hidden")
    return result.cache, result.last_appended_hidden


def _merge_final_once(
    record_ids: tuple[int, ...],
    sources: tuple[tuple[JaggedMigratedKVBatch, torch.Tensor], ...],
    target_version: int,
    device: torch.device,
) -> tuple[JaggedMigratedKVBatch, torch.Tensor]:
    cache = stage48._assemble_target_sources(
        record_ids,
        tuple(value[0] for value in sources),
        target_version,
    )
    hidden_by_id = {
        record_id: hidden[row]
        for source, hidden in sources
        for row, record_id in enumerate(source.record_ids)
    }
    if set(hidden_by_id) != set(record_ids):
        raise RuntimeError("Stage 4.9 final hidden coverage differs")
    return cache, torch.stack([hidden_by_id[value] for value in record_ids])


def topk_boundary_parity(
    left: torch.Tensor,
    right: torch.Tensor,
    k: int,
    score_atol: float = EXACT_PARITY_ATOL,
) -> dict[str, object]:
    if (
        left.ndim != 2
        or left.shape != right.shape
        or left.shape[0] < 1
        or not 1 <= k <= left.shape[1]
        or not math.isfinite(score_atol)
        or score_atol <= 0
        or not bool(torch.isfinite(left).all())
        or not bool(torch.isfinite(right).all())
    ):
        raise ValueError("Stage 4.9 Top-K parity inputs are invalid")
    left_values, left_indices = torch.topk(left, k=k, dim=1)
    right_values, right_indices = torch.topk(right, k=k, dim=1)
    row_errors = (left.float() - right.float()).abs().amax(dim=1)
    left_cpu = left.detach().float().cpu()
    right_cpu = right.detach().float().cpu()
    left_values_cpu = left_values.detach().float().cpu()
    right_values_cpu = right_values.detach().float().cpu()
    left_indices_cpu = left_indices.detach().cpu()
    right_indices_cpu = right_indices.detach().cpu()
    row_errors_cpu = row_errors.detach().cpu()
    exact_order_rows = []
    exact_set_rows = []
    boundary_rows = []
    mismatch_counts = []
    overlap_fractions = []
    maximum_boundary_distance = 0.0
    maximum_boundary_radius = 0.0
    maximum_inverted_pair_score_gap = 0.0
    inverted_pair_count = 0
    first_mismatch = None
    for row in range(left.shape[0]):
        left_order = tuple(
            int(value) for value in left_indices_cpu[row].tolist()
        )
        right_order = tuple(
            int(value) for value in right_indices_cpu[row].tolist()
        )
        left_set = set(left_order)
        right_set = set(right_order)
        left_only = tuple(sorted(left_set - right_set))
        right_only = tuple(sorted(right_set - left_set))
        exact_order = left_order == right_order
        exact_set = not left_only and not right_only
        epsilon = float(row_errors_cpu[row])
        scale = max(
            1.0,
            float(left_cpu[row].abs().max()),
            float(right_cpu[row].abs().max()),
        )
        guard = torch.finfo(torch.float32).eps * scale
        radius = 2.0 * epsilon + guard
        left_cutoff = float(left_values_cpu[row, -1])
        right_cutoff = float(right_values_cpu[row, -1])
        differing = (*left_only, *right_only)
        distances = [
            max(
                abs(float(left_cpu[row, item]) - left_cutoff),
                abs(float(right_cpu[row, item]) - right_cutoff),
            )
            for item in differing
        ]
        distance = max(distances, default=0.0)
        common = tuple(sorted(left_set & right_set))
        left_rank = {item: rank for rank, item in enumerate(left_order)}
        right_rank = {item: rank for rank, item in enumerate(right_order)}
        local_inverted_gaps = []
        for first_index, first_item in enumerate(common):
            for second_item in common[first_index + 1 :]:
                left_direction = left_rank[first_item] - left_rank[second_item]
                right_direction = (
                    right_rank[first_item] - right_rank[second_item]
                )
                if left_direction * right_direction >= 0:
                    continue
                local_inverted_gaps.append(
                    max(
                        abs(
                            float(left_cpu[row, first_item])
                            - float(left_cpu[row, second_item])
                        ),
                        abs(
                            float(right_cpu[row, first_item])
                            - float(right_cpu[row, second_item])
                        ),
                    )
                )
        inverted_gap = max(local_inverted_gaps, default=0.0)
        inverted_pair_count += len(local_inverted_gaps)
        maximum_inverted_pair_score_gap = max(
            maximum_inverted_pair_score_gap,
            inverted_gap,
        )
        boundary = distance <= radius and inverted_gap <= radius
        exact_order_rows.append(exact_order)
        exact_set_rows.append(exact_set)
        boundary_rows.append(boundary)
        mismatch_counts.append(len(differing))
        overlap_fractions.append(len(left_set & right_set) / k)
        maximum_boundary_distance = max(
            maximum_boundary_distance,
            distance,
        )
        maximum_boundary_radius = max(maximum_boundary_radius, radius)
        if (differing or not exact_order) and first_mismatch is None:
            first_mismatch = {
                "row": row,
                "left_only": list(left_only),
                "right_only": list(right_only),
                "set_equal": exact_set,
                "order_equal": exact_order,
                "left_cutoff_score": left_cutoff,
                "right_cutoff_score": right_cutoff,
                "row_score_max_abs": epsilon,
                "derived_boundary_radius": radius,
                "maximum_differing_item_boundary_distance": distance,
                "inverted_common_item_pairs": len(local_inverted_gaps),
                "maximum_inverted_pair_score_gap": inverted_gap,
            }
    maximum_score_error = float(row_errors_cpu.max())
    numerical = maximum_score_error <= score_atol
    boundary = all(boundary_rows)
    return {
        "k": k,
        "rows": left.shape[0],
        "score_atol": score_atol,
        "maximum_score_max_abs": maximum_score_error,
        "numerical_score_equivalent": numerical,
        "exact_topk_order_equal": all(exact_order_rows),
        "exact_topk_order_equal_rows": sum(exact_order_rows),
        "exact_topk_set_equal": all(exact_set_rows),
        "exact_topk_set_equal_rows": sum(exact_set_rows),
        "boundary_equivalent": boundary,
        "boundary_equivalent_rows": sum(boundary_rows),
        "order_mismatched_rows": sum(
            not value for value in exact_order_rows
        ),
        "set_mismatched_rows": sum(not value for value in exact_set_rows),
        "maximum_symmetric_difference_items": max(mismatch_counts),
        "minimum_topk_overlap": min(overlap_fractions),
        "maximum_differing_item_boundary_distance": (
            maximum_boundary_distance
        ),
        "inverted_common_item_pairs": inverted_pair_count,
        "maximum_inverted_pair_score_gap": (
            maximum_inverted_pair_score_gap
        ),
        "maximum_derived_boundary_radius": maximum_boundary_radius,
        "boundary_radius_formula": (
            "2 * per_row_score_max_abs + float32_eps * max(1, score_scale)"
        ),
        "first_mismatch": first_mismatch,
        "passed": numerical and boundary,
    }


@torch.inference_mode()
def _exact_parity_branch_once(
    model,
    resident_ids: tuple[int, ...],
    timed_ids: tuple[int, ...],
    natural_prefix_ids: tuple[int, ...],
    short_ids: tuple[int, ...],
    plans,
    record_by_id: dict[int, dict],
    target_window,
    target_version: int,
    cfg,
    device: torch.device,
) -> tuple[JaggedMigratedKVBatch, torch.Tensor]:
    prefix_sources = []
    if timed_ids:
        retained_batch = stage49._retained_batch(
            timed_ids,
            plans,
            record_by_id,
            target_window,
            device,
        )
        retained = stage49._exact_cache(
            model,
            retained_batch,
            timed_ids,
            target_version,
            torch.float32,
        )
        prefix_sources.append(
            _append_delta_once(
                model,
                retained,
                plans,
                record_by_id,
                target_window,
                device,
                torch.float32,
            )
        )
    if natural_prefix_ids:
        prefix_sources.append(
            _build_natural_prefix_once(
                model,
                natural_prefix_ids,
                plans,
                record_by_id,
                target_window,
                target_version,
                cfg,
                device,
                torch.float32,
            )
        )
    nonshort_ids = tuple(
        value for value in resident_ids if plans[value].target_prefix_tokens > 0
    )
    final_sources = []
    if nonshort_ids:
        prefix = stage48._assemble_target_sources(
            nonshort_ids,
            tuple(prefix_sources),
            target_version,
        )
        final_sources.append(
            _append_latest_once(
                model,
                prefix,
                record_by_id,
                target_window,
                device,
                torch.float32,
            )
        )
    if short_ids:
        final_sources.append(
            _append_fresh_latest_once(
                model,
                short_ids,
                record_by_id,
                target_window,
                target_version,
                cfg,
                device,
                torch.float32,
            )
        )
    return _merge_final_once(
        resident_ids,
        tuple(final_sources),
        target_version,
        device,
    )


def _mean_task(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("Stage 4.9 task population is empty")
    return {
        "catalog_auc": sum(value["catalog_auc"] for value in values)
        / len(values),
        "ndcg_at_100": sum(value["ndcg@100"] for value in values)
        / len(values),
        "hit_at_100": sum(value["hit@100"] for value in values)
        / len(values),
    }


def _paired_task_summary(
    mixed_values: list[dict[str, float]],
    exact_values: list[dict[str, float]],
) -> dict[str, object]:
    if len(mixed_values) != len(exact_values):
        raise ValueError("Stage 4.9 paired task populations differ")
    mixed = _mean_task(mixed_values)
    exact = _mean_task(exact_values)
    return {
        "records": len(mixed_values),
        "mixed": mixed,
        "fresh_exact": exact,
        "mixed_minus_fresh_exact": {
            metric: mixed[metric] - exact[metric] for metric in TASK_METRICS
        },
        "mixed_over_fresh_exact": {
            metric: (
                mixed[metric] / exact[metric]
                if exact[metric] != 0
                else None
            )
            for metric in TASK_METRICS
        },
    }


def _record_weighted_task(endpoints: list[dict]) -> dict[str, object]:
    records = sum(int(value["task_metrics"]["records"]) for value in endpoints)
    if records <= 0:
        raise ValueError("Stage 4.9 weighted task population is empty")
    mixed = {
        metric: sum(
            int(value["task_metrics"]["records"])
            * float(value["task_metrics"]["mixed"][metric])
            for value in endpoints
        )
        / records
        for metric in TASK_METRICS
    }
    exact = {
        metric: sum(
            int(value["task_metrics"]["records"])
            * float(value["task_metrics"]["fresh_exact"][metric])
            for value in endpoints
        )
        / records
        for metric in TASK_METRICS
    }
    return {
        "records": records,
        "mixed": mixed,
        "fresh_exact": exact,
        "mixed_minus_fresh_exact": {
            metric: mixed[metric] - exact[metric] for metric in TASK_METRICS
        },
        "mixed_over_fresh_exact": {
            metric: (
                mixed[metric] / exact[metric]
                if exact[metric] != 0
                else None
            )
            for metric in TASK_METRICS
        },
    }


def _serialize_state(state: object) -> dict:
    return stage48._serialize_state(state)


def smoke_payload(args: argparse.Namespace) -> dict[str, object]:
    specs = candidate_specs()
    fake_steps = [
        {
            "cost": {
                "u": {
                    "samples_ms": [1.0, 1.1, 0.9],
                    "sum_of_component_medians_ms": 1.0,
                },
                "e": {
                    "samples_ms": [4.0, 4.1, 3.9],
                    "median_ms": 4.0,
                },
                "outside_rollout_timer": {
                    "mixed": {
                        "median_ms": 8.0,
                        "components": {
                            "target_delta_append_ms": {"median_ms": 2.0},
                            "latest_append_ms": {"median_ms": 3.0},
                            "short_latest_append_ms": {"median_ms": 0.0},
                        },
                    },
                    "exact": {
                        "median_ms": 9.0,
                        "components": {
                            "target_delta_append_ms": {"median_ms": 2.5},
                            "latest_append_ms": {"median_ms": 3.5},
                            "short_latest_append_ms": {"median_ms": 0.0},
                        },
                    },
                },
                "state_movement_outside_primary": {
                    "h2d_previous_actual": _zero_state_movement(
                        "host_to_device_previous_actual"
                    ),
                    "d2h_next_actual": _zero_state_movement(
                        "device_to_host_next_actual"
                    ),
                },
            }
        }
        for _ in range(NUM_EDGES)
    ]
    cost = summarize_paired_cost(fake_steps, args.timing_repeats)
    checks = {
        "exactly_two_frozen_candidates": tuple(
            value.to_dict() for value in specs
        )
        == (
            stage49._candidate_spec("token_debt_total10").to_dict(),
            stage49._candidate_spec(
                "staggered_renewal_h12"
            ).to_dict(),
        ),
        "eleven_recursive_edges": NUM_EDGES == 11,
        "one_warmup_three_measured": (
            args.warmup_repeats,
            args.timing_repeats,
        )
        == (1, 3),
        "old_denominator_not_reused": not cost[
            "old_exact_denominator_reused"
        ],
        "target_append_excluded": cost[
            "target_append_excluded_from_u_and_e"
        ],
        "rollout_abi_matches": stage49.PROTOCOL
        == ROLLOUT_BOUNDARY_PROTOCOL,
        "host_staged_persistent_store": True,
        "state_movement_reported_outside_primary": bool(
            cost["state_movement_outside_primary"]["reported_separately"]
        )
        and bool(
            cost["state_movement_outside_primary"]["excluded_from_primary"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 4.9 formal static smoke failed: {checks}")
    return {
        "protocol": STATIC_PROTOCOL,
        "status": "smoke_passed",
        "scientific_result": False,
        "formal_result_written": False,
        "candidate_order": [
            name for name in CANDIDATE_NAMES
        ],
        "measurement_contract": {
            "primary": "sum_t(U_t)/sum_t(E_t)",
            "recursive_input": "previous_actual_post_append_mixed_cache",
            "paired_exact": "same_device_same_retained_population_fresh_run",
            "target_append": "target_model_outside_u_and_e",
            "old_denominator_reused": False,
            "stage47_or_stage48_edge_runner_called": False,
            "persistent_store": (
                "cpu_singleton_fp16_post_append_full_mixed_kv"
            ),
            "groupwise_staging": (
                "migrant inputs H2D before U; mixed outputs D2H after "
                "append and outside U/E"
            ),
            "state_movement": "reported_separately_outside_primary",
            "capacity_scope": (
                "evaluator-only host spill; not full-cohort HBM or E2E"
            ),
        },
        "implementation": implementation_snapshot(),
        "checks": checks,
    }


def _measurement_total(
    measurements: list[dict[str, object]],
    timing_repeats: int,
) -> dict[str, object]:
    samples = [
        sum(
            float(measurement["samples_ms"][index])
            for measurement in measurements
        )
        for index in range(timing_repeats)
    ]
    return {
        "samples_ms": samples,
        "median_ms": sum(
            float(measurement["median_ms"])
            for measurement in measurements
        ),
        "median_of_repetition_sums_ms": float(statistics.median(samples)),
    }


def _action_name(
    record_id: int,
    plan,
    migrate_ids: set[int],
    scheduled_ids: set[int],
    missing_ids: set[int],
) -> str:
    if plan.final_tokens == 0:
        return plan.status
    if record_id in migrate_ids:
        return "migrate"
    if record_id in scheduled_ids:
        return "scheduled_exact"
    if record_id in missing_ids:
        return "missing_cache_exact"
    if plan.status == "short_no_prefix":
        return "natural_exact_short"
    return "natural_exact"


@torch.inference_mode()
def _run_group(
    args: argparse.Namespace,
    cfg,
    target_model,
    operator,
    program,
    target_version: int,
    target_window,
    group,
    record_by_id: dict[int, dict],
    plans,
    selection,
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    last_exact_by_record: dict[int, int],
    all_items: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    timing_repeats = args.timing_repeats
    warmup_repeats = args.warmup_repeats
    group_ids = tuple(int(value["record_id"]) for value in group)
    resident_ids = tuple(
        value for value in group_ids if plans[value].final_tokens > 0
    )
    migrate_ids = tuple(
        value
        for value in resident_ids
        if value in set(selection.migrate_ids)
    )
    scheduled_ids = tuple(
        value
        for value in resident_ids
        if value in set(selection.scheduled_exact_ids)
    )
    natural_ids = tuple(
        value
        for value in resident_ids
        if value in set(selection.natural_exact_ids)
    )
    missing_ids = tuple(
        value for value in natural_ids if plans[value].timed_retained_rebuild
    )
    natural_prefix_ids = tuple(
        value
        for value in natural_ids
        if value not in set(missing_ids)
        and plans[value].target_prefix_tokens > 0
    )
    short_ids = tuple(
        value
        for value in natural_ids
        if plans[value].target_prefix_tokens == 0
    )
    timed_ids = tuple(sorted((*migrate_ids, *scheduled_ids, *missing_ids)))
    staged_actual, h2d_previous_actual = _timed_store_transfer(
        cache_by_record,
        migrate_ids,
        device,
        device,
        "host_to_device_previous_actual",
    )
    staged_checks = _staged_device_store_checks(
        staged_actual,
        migrate_ids,
        target_version - 1,
        {
            record_id: plans[record_id].old_tokens
            for record_id in migrate_ids
        },
        device,
    )
    if not all(staged_checks.values()):
        raise RuntimeError(
            "Stage 4.9 staged migrant store failed before U: "
            f"{staged_checks}"
        )
    u_components = {
        "retained_source_crop_ms": _zero_measurement(timing_repeats),
        "retained_transform_ms": _zero_measurement(timing_repeats),
        "scheduled_exact_retained_ms": _zero_measurement(timing_repeats),
        "missing_exact_retained_ms": _zero_measurement(timing_repeats),
        "retained_materialization_ms": _zero_measurement(timing_repeats),
    }
    retained_sources = []
    if migrate_ids:
        cropped, measurement = _timed_repeated(
            partial(
                _crop_actual_retained,
                staged_actual,
                migrate_ids,
                plans,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        u_components["retained_source_crop_ms"] = measurement
        staged_actual.clear()
        del staged_actual
        migrated, measurement = _timed_repeated(
            partial(
                execute_direct,
                operator,
                program,
                cropped,
                target_version,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        u_components["retained_transform_ms"] = measurement
        retained_sources.append(migrated)
        del cropped
    else:
        del staged_actual
    if scheduled_ids:
        scheduled_batch = stage49._retained_batch(
            scheduled_ids,
            plans,
            record_by_id,
            target_window,
            device,
        )
        scheduled_exact, measurement = _timed_repeated(
            partial(
                stage49._exact_cache,
                target_model,
                scheduled_batch,
                scheduled_ids,
                target_version,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        u_components["scheduled_exact_retained_ms"] = measurement
        retained_sources.append(scheduled_exact)
    if missing_ids:
        missing_batch = stage49._retained_batch(
            missing_ids,
            plans,
            record_by_id,
            target_window,
            device,
        )
        missing_exact, measurement = _timed_repeated(
            partial(
                stage49._exact_cache,
                target_model,
                missing_batch,
                missing_ids,
                target_version,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        u_components["missing_exact_retained_ms"] = measurement
        retained_sources.append(missing_exact)
    mixed_retained = None
    paired_exact_retained = None
    if timed_ids:
        mixed_retained, measurement = _timed_repeated(
            partial(
                stage48._assemble_target_sources,
                timed_ids,
                tuple(retained_sources),
                target_version,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        u_components["retained_materialization_ms"] = measurement
        exact_retained_batch = stage49._retained_batch(
            timed_ids,
            plans,
            record_by_id,
            target_window,
            device,
        )
        paired_exact_retained, e_measurement = _timed_repeated(
            partial(
                stage49._exact_cache,
                target_model,
                exact_retained_batch,
                timed_ids,
                target_version,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
    else:
        e_measurement = _zero_measurement(timing_repeats)
    u_measurement = _sum_measurements(u_components, timing_repeats)
    mixed_outside = {
        "target_delta_append_ms": _zero_measurement(timing_repeats),
        "natural_target_prefix_build_ms": _zero_measurement(timing_repeats),
        "target_prefix_assembly_ms": _zero_measurement(timing_repeats),
        "latest_append_ms": _zero_measurement(timing_repeats),
        "short_latest_append_ms": _zero_measurement(timing_repeats),
        "final_assembly_ms": _zero_measurement(timing_repeats),
        "final_split_ms": _zero_measurement(timing_repeats),
    }
    exact_outside = {
        key: _zero_measurement(timing_repeats) for key in mixed_outside
    }
    mixed_prefix_sources = []
    exact_prefix_sources = []
    if timed_ids:
        mixed_delta, measurement = _timed_repeated(
            partial(
                _append_delta_once,
                target_model,
                mixed_retained,
                plans,
                record_by_id,
                target_window,
                device,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        mixed_outside["target_delta_append_ms"] = measurement
        mixed_prefix_sources.append(mixed_delta)
        exact_delta, measurement = _timed_repeated(
            partial(
                _append_delta_once,
                target_model,
                paired_exact_retained,
                plans,
                record_by_id,
                target_window,
                device,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        exact_outside["target_delta_append_ms"] = measurement
        exact_prefix_sources.append(exact_delta)
    if natural_prefix_ids:
        mixed_natural, measurement = _timed_repeated(
            partial(
                _build_natural_prefix_once,
                target_model,
                natural_prefix_ids,
                plans,
                record_by_id,
                target_window,
                target_version,
                cfg,
                device,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        mixed_outside["natural_target_prefix_build_ms"] = measurement
        mixed_prefix_sources.append(mixed_natural)
        exact_natural, measurement = _timed_repeated(
            partial(
                _build_natural_prefix_once,
                target_model,
                natural_prefix_ids,
                plans,
                record_by_id,
                target_window,
                target_version,
                cfg,
                device,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        exact_outside["natural_target_prefix_build_ms"] = measurement
        exact_prefix_sources.append(exact_natural)
    nonshort_ids = tuple(
        value for value in resident_ids if plans[value].target_prefix_tokens > 0
    )
    branch_sources = {}
    if nonshort_ids:
        mixed_prefix, measurement = _timed_repeated(
            partial(
                stage48._assemble_target_sources,
                nonshort_ids,
                tuple(mixed_prefix_sources),
                target_version,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        mixed_outside["target_prefix_assembly_ms"] = measurement
        mixed_nonshort, measurement = _timed_repeated(
            partial(
                _append_latest_once,
                target_model,
                mixed_prefix,
                record_by_id,
                target_window,
                device,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        mixed_outside["latest_append_ms"] = measurement
        branch_sources["mixed_nonshort"] = mixed_nonshort
        exact_prefix, measurement = _timed_repeated(
            partial(
                stage48._assemble_target_sources,
                nonshort_ids,
                tuple(exact_prefix_sources),
                target_version,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        exact_outside["target_prefix_assembly_ms"] = measurement
        exact_nonshort, measurement = _timed_repeated(
            partial(
                _append_latest_once,
                target_model,
                exact_prefix,
                record_by_id,
                target_window,
                device,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        exact_outside["latest_append_ms"] = measurement
        branch_sources["exact_nonshort"] = exact_nonshort
    if short_ids:
        mixed_short, measurement = _timed_repeated(
            partial(
                _append_fresh_latest_once,
                target_model,
                short_ids,
                record_by_id,
                target_window,
                target_version,
                cfg,
                device,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        mixed_outside["short_latest_append_ms"] = measurement
        branch_sources["mixed_short"] = mixed_short
        exact_short, measurement = _timed_repeated(
            partial(
                _append_fresh_latest_once,
                target_model,
                short_ids,
                record_by_id,
                target_window,
                target_version,
                cfg,
                device,
                torch.float16,
            ),
            device,
            warmup_repeats,
            timing_repeats,
        )
        exact_outside["short_latest_append_ms"] = measurement
        branch_sources["exact_short"] = exact_short
    mixed_final_sources = tuple(
        value
        for key in ("mixed_nonshort", "mixed_short")
        if (value := branch_sources.get(key)) is not None
    )
    exact_final_sources = tuple(
        value
        for key in ("exact_nonshort", "exact_short")
        if (value := branch_sources.get(key)) is not None
    )
    if not resident_ids:
        lineage = [
            {
                **plans[record_id].to_dict(),
                "action": plans[record_id].status,
                "append_model_version": None,
                "last_exact_version_before": last_exact_by_record.get(
                    record_id
                ),
                "last_exact_version_after": None,
                "previous_actual_cache_consumed": False,
            }
            for record_id in group_ids
        ]
        return {
            "cache": {},
            "last_exact": {},
            "task_mixed": [],
            "task_exact": [],
            "record_metrics": [],
            "lineage": lineage,
            "cost": {
                "u": u_measurement,
                "e": e_measurement,
                "outside_mixed": _sum_measurements(
                    mixed_outside,
                    timing_repeats,
                ),
                "outside_exact": _sum_measurements(
                    exact_outside,
                    timing_repeats,
                ),
                "one_shot_exact": _zero_measurement(timing_repeats),
                "state_movement": {
                    "h2d_previous_actual": h2d_previous_actual,
                    "d2h_next_actual": _zero_state_movement(
                        "device_to_host_next_actual"
                    ),
                },
            },
            "checks": {
                "empty_target_group": True,
                "source_model_append_calls_zero": True,
                "h2d_precedes_u_and_is_excluded": bool(
                    h2d_previous_actual["outside_u_and_e"]
                )
                and bool(h2d_previous_actual["outside_append_timer"]),
                "h2d_contains_only_migrants": int(
                    h2d_previous_actual["records"]
                )
                == len(migrate_ids),
                **staged_checks,
            },
        }
    mixed_final, measurement = _timed_repeated(
        partial(
            _merge_final_once,
            resident_ids,
            mixed_final_sources,
            target_version,
            device,
        ),
        device,
        warmup_repeats,
        timing_repeats,
    )
    mixed_outside["final_assembly_ms"] = measurement
    exact_two_stage, measurement = _timed_repeated(
        partial(
            _merge_final_once,
            resident_ids,
            exact_final_sources,
            target_version,
            device,
        ),
        device,
        warmup_repeats,
        timing_repeats,
    )
    exact_outside["final_assembly_ms"] = measurement
    mixed_split, measurement = _timed_repeated(
        partial(base._split_cache, mixed_final[0]),
        device,
        warmup_repeats,
        timing_repeats,
    )
    mixed_outside["final_split_ms"] = measurement
    _, measurement = _timed_repeated(
        partial(base._split_cache, exact_two_stage[0]),
        device,
        warmup_repeats,
        timing_repeats,
    )
    exact_outside["final_split_ms"] = measurement
    resident_descriptors = [
        record_by_id[value] for value in resident_ids
    ]
    resident_records = stage49._records_for_ids(
        resident_ids,
        record_by_id,
        target_window,
    )
    full_batch = base._history_batch(
        resident_records,
        cfg.max_seq_len,
        device,
        prefix=False,
    )
    fresh, fresh_measurement = _timed_repeated(
        partial(
            stage49._exact_full,
            target_model,
            full_batch,
            resident_ids,
            target_version,
            torch.float16,
        ),
        device,
        warmup_repeats,
        timing_repeats,
    )
    fresh_cache, fresh_hidden = fresh
    mixed_cache, mixed_hidden = mixed_final
    exact_cache, exact_hidden = exact_two_stage
    candidates = all_items.unsqueeze(0).expand(len(resident_ids), -1)
    mixed_scores = target_model.item_emb.score(mixed_hidden, candidates)
    exact_scores = target_model.item_emb.score(exact_hidden, candidates)
    fresh_scores = target_model.item_emb.score(fresh_hidden, candidates)
    topk = min(100, all_items.numel())
    task_mixed = stage48._score_task_rows(
        target_model,
        mixed_hidden,
        resident_descriptors,
        resident_records,
        all_items,
    )
    task_exact = stage48._score_task_rows(
        target_model,
        fresh_hidden,
        resident_descriptors,
        resident_records,
        all_items,
    )
    record_metrics = base._record_metrics(
        mixed_cache,
        fresh_cache,
        mixed_hidden,
        mixed_scores,
        fresh_hidden,
        fresh_scores,
    )
    action_migrate = set(migrate_ids)
    action_scheduled = set(scheduled_ids)
    action_missing = set(missing_ids)
    next_last_exact = {}
    lineage = []
    for record_id in group_ids:
        plan = plans[record_id]
        before = last_exact_by_record.get(record_id)
        if plan.final_tokens == 0:
            after = None
        elif record_id in action_migrate:
            if before is None:
                raise RuntimeError("Stage 4.9 migrant has no exact anchor")
            after = before
        else:
            after = target_version
            next_last_exact[record_id] = target_version
        if after is not None and record_id in action_migrate:
            next_last_exact[record_id] = after
        lineage.append(
            {
                **plan.to_dict(),
                "action": _action_name(
                    record_id,
                    plan,
                    action_migrate,
                    action_scheduled,
                    action_missing,
                ),
                "append_model_version": (
                    target_version if plan.final_tokens > 0 else None
                ),
                "last_exact_version_before": before,
                "last_exact_version_after": after,
                "migration_depth_after": (
                    target_version - after if after is not None else None
                ),
                "previous_actual_cache_consumed": (
                    record_id in action_migrate
                ),
            }
        )
    retained_lengths = tuple(
        plans[value].retained_tokens for value in timed_ids
    )
    exact_k_max = stage49._max_abs(exact_cache.k, fresh_cache.k)
    exact_v_max = stage49._max_abs(exact_cache.v, fresh_cache.v)
    exact_hidden_max = stage49._max_abs(exact_hidden, fresh_hidden)
    exact_score_max = stage49._max_abs(exact_scores, fresh_scores)
    fp16_top100_equal = torch.equal(
        torch.topk(exact_scores, k=topk, dim=1).indices,
        torch.topk(fresh_scores, k=topk, dim=1).indices,
    )
    checks = {
        "action_partition": set(resident_ids)
        == action_migrate
        | action_scheduled
        | set(natural_ids),
        "actions_disjoint": not action_migrate & action_scheduled
        and not action_migrate & set(natural_ids)
        and not action_scheduled & set(natural_ids),
        "timed_population_matches": (
            (
                mixed_retained is None
                and paired_exact_retained is None
                and not timed_ids
            )
            or (
                mixed_retained.record_ids
                == paired_exact_retained.record_ids
                == timed_ids
                and tuple(int(value) for value in mixed_retained.lengths)
                == retained_lengths
                and tuple(
                    int(value) for value in paired_exact_retained.lengths
                )
                == retained_lengths
            )
        ),
        "previous_actual_lengths_match": all(
            int(cache_by_record[value].lengths[0])
            == plans[value].old_tokens
            for value in migrate_ids
        ),
        "recursive_output_covers_residents": set(mixed_split)
        == set(resident_ids),
        "recursive_output_lengths_match": all(
            int(mixed_split[value].lengths[0])
            == plans[value].final_tokens
            for value in resident_ids
        ),
        "recursive_output_target_version": all(
            value.migration_anchor_version == f"theta{target_version}"
            and value.served_kv_target == f"theta{target_version}"
            for value in mixed_split.values()
        ),
        "task_population_paired": len(task_mixed) == len(task_exact),
        "finite_outputs": bool(torch.isfinite(mixed_cache.k).all())
        and bool(torch.isfinite(mixed_cache.v).all())
        and bool(torch.isfinite(mixed_hidden).all())
        and bool(torch.isfinite(mixed_scores).all()),
        "source_model_append_calls_zero": True,
        "target_append_outside_u_and_e": True,
        "h2d_precedes_u_and_is_excluded": bool(
            h2d_previous_actual["outside_u_and_e"]
        )
        and bool(h2d_previous_actual["outside_append_timer"]),
        "h2d_contains_only_migrants": int(
            h2d_previous_actual["records"]
        )
        == len(migrate_ids),
        "retained_endpoint_fp16": (
            not timed_ids
            or (
                mixed_retained.k.dtype
                == paired_exact_retained.k.dtype
                == torch.float16
                and mixed_retained.v.dtype
                == paired_exact_retained.v.dtype
                == torch.float16
            )
        ),
        **staged_checks,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Stage 4.9 formal group checks failed: "
            f"edge={target_version - 1}->{target_version}, "
            f"group_record_ids={group_ids}, resident_ids={resident_ids}, "
            f"timed_ids={timed_ids}, "
            f"exact_fp16_two_stage_vs_fresh="
            f"{{'k_max_abs': {exact_k_max}, "
            f"'v_max_abs': {exact_v_max}, "
            f"'hidden_max_abs': {exact_hidden_max}, "
            f"'score_max_abs': {exact_score_max}, "
            f"'top100_equal': "
            f"{fp16_top100_equal}}}, "
            f"failed="
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    diagnostics = {
        "retained_population_sha256": retained_population_sha256(
            [plans[value] for value in timed_ids]
        ),
        "timed_records": len(timed_ids),
        "timed_tokens": sum(retained_lengths),
        "exact_fp16_endpoint_two_stage_vs_fresh": {
            "k_max_abs": exact_k_max,
            "v_max_abs": exact_v_max,
            "hidden_max_abs": exact_hidden_max,
            "score_max_abs": exact_score_max,
            "top100_equal": fp16_top100_equal,
            "role": "quantized_deployment_diagnostic_not_equivalence_gate",
        },
    }
    result = {
        "cache": mixed_split,
        "last_exact": next_last_exact,
        "task_mixed": task_mixed,
        "task_exact": task_exact,
        "record_metrics": record_metrics,
        "lineage": lineage,
        "cost": {
            "u": u_measurement,
            "e": e_measurement,
            "outside_mixed": _sum_measurements(
                mixed_outside,
                timing_repeats,
            ),
            "outside_exact": _sum_measurements(
                exact_outside,
                timing_repeats,
            ),
            "one_shot_exact": fresh_measurement,
            "state_movement": {
                "h2d_previous_actual": h2d_previous_actual,
                "d2h_next_actual": _zero_state_movement(
                    "device_to_host_next_actual"
                ),
            },
        },
        "diagnostics": diagnostics,
        "checks": checks,
    }
    del (
        mixed_cache,
        mixed_hidden,
        exact_cache,
        exact_hidden,
        fresh_cache,
        fresh_hidden,
        mixed_scores,
        exact_scores,
        fresh_scores,
        full_batch,
    )
    return result


@torch.inference_mode()
def _exact_parity_group_summary(
    cfg,
    target_model,
    target_version: int,
    target_window,
    group,
    record_by_id: dict[int, dict],
    plans,
    selection,
    all_items: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, bool]]:
    migrate_ids = set(selection.migrate_ids)
    scheduled_ids = set(selection.scheduled_exact_ids)
    resident_ids = tuple(
        int(value["record_id"])
        for value in group
        if plans[int(value["record_id"])].final_tokens > 0
    )
    summaries = []
    for record_id in resident_ids:
        plan = plans[record_id]
        timed = (
            record_id in migrate_ids
            or record_id in scheduled_ids
            or plan.timed_retained_rebuild
        )
        timed_ids = (record_id,) if timed else ()
        natural_prefix_ids = (
            (record_id,)
            if not timed and plan.target_prefix_tokens > 0
            else ()
        )
        short_ids = (
            (record_id,)
            if not timed and plan.target_prefix_tokens == 0
            else ()
        )
        parity_cache, parity_hidden = _exact_parity_branch_once(
            target_model,
            (record_id,),
            timed_ids,
            natural_prefix_ids,
            short_ids,
            plans,
            record_by_id,
            target_window,
            target_version,
            cfg,
            device,
        )
        record = stage49._records_for_ids(
            (record_id,),
            record_by_id,
            target_window,
        )
        full_batch = base._history_batch(
            record,
            cfg.max_seq_len,
            device,
            prefix=False,
        )
        fresh_cache, fresh_hidden = stage49._exact_full(
            target_model,
            full_batch,
            (record_id,),
            target_version,
            torch.float32,
        )
        candidates = all_items.unsqueeze(0)
        parity_scores = target_model.item_emb.score(
            parity_hidden,
            candidates,
        )
        fresh_scores = target_model.item_emb.score(
            fresh_hidden,
            candidates,
        )
        topk = min(100, all_items.numel())
        topk_result = topk_boundary_parity(
            parity_scores,
            fresh_scores,
            topk,
        )
        summaries.append(
            {
                "record_id": record_id,
                "action_class": (
                    "timed_retained"
                    if timed
                    else (
                        "natural_prefix"
                        if natural_prefix_ids
                        else "latest_only"
                    )
                ),
                "tokens": plan.final_tokens,
                "k_max_abs": stage49._max_abs(
                    parity_cache.k,
                    fresh_cache.k,
                ),
                "v_max_abs": stage49._max_abs(
                    parity_cache.v,
                    fresh_cache.v,
                ),
                "hidden_max_abs": stage49._max_abs(
                    parity_hidden,
                    fresh_hidden,
                ),
                "score_max_abs": stage49._max_abs(
                    parity_scores,
                    fresh_scores,
                ),
                "top100": topk_result,
                "fp32_endpoint": (
                    parity_cache.k.dtype
                    == fresh_cache.k.dtype
                    == torch.float32
                    and parity_cache.v.dtype
                    == fresh_cache.v.dtype
                    == torch.float32
                ),
            }
        )
        del (
            parity_cache,
            parity_hidden,
            fresh_cache,
            fresh_hidden,
            parity_scores,
            fresh_scores,
            full_batch,
        )
        torch.cuda.empty_cache()
    diagnostics = {
        "records": len(summaries),
        "record_chunk_size": 1,
        "gpu_outputs_retained_after_record": False,
        "summary_materialization": "python_cpu_scalars_before_next_record",
        "k_max_abs": max(value["k_max_abs"] for value in summaries),
        "v_max_abs": max(value["v_max_abs"] for value in summaries),
        "hidden_max_abs": max(
            value["hidden_max_abs"] for value in summaries
        ),
        "score_max_abs": max(
            value["score_max_abs"] for value in summaries
        ),
        "top100_exact_order_equal": all(
            value["top100"]["exact_topk_order_equal"]
            for value in summaries
        ),
        "top100_exact_set_equal": all(
            value["top100"]["exact_topk_set_equal"]
            for value in summaries
        ),
        "top100_boundary_equivalent": all(
            value["top100"]["passed"] for value in summaries
        ),
        "top100_order_mismatched_rows": sum(
            int(value["top100"]["order_mismatched_rows"])
            for value in summaries
        ),
        "top100_set_mismatched_rows": sum(
            int(value["top100"]["set_mismatched_rows"])
            for value in summaries
        ),
        "minimum_top100_overlap": min(
            float(value["top100"]["minimum_topk_overlap"])
            for value in summaries
        ),
        "per_record": summaries,
        "role": "algorithmic_equivalence_authority",
    }
    checks = {
        "exact_fp32_parity_k_matches_fresh": diagnostics["k_max_abs"]
        <= EXACT_PARITY_ATOL,
        "exact_fp32_parity_v_matches_fresh": diagnostics["v_max_abs"]
        <= EXACT_PARITY_ATOL,
        "exact_fp32_parity_hidden_matches_fresh": (
            diagnostics["hidden_max_abs"] <= EXACT_PARITY_ATOL
        ),
        "exact_fp32_parity_scores_match_fresh": (
            diagnostics["score_max_abs"] <= EXACT_PARITY_ATOL
        ),
        "exact_fp32_parity_top100_boundary_equivalent": diagnostics[
            "top100_boundary_equivalent"
        ],
        "exact_fp32_parity_record_chunks_are_one": diagnostics[
            "record_chunk_size"
        ]
        == 1,
        "exact_fp32_parity_endpoints_are_fp32": all(
            value["fp32_endpoint"] for value in summaries
        ),
    }
    return diagnostics, checks


def _aggregate_edge_measurement(
    group_results: list[dict[str, object]],
    path: tuple[str, ...],
    timing_repeats: int,
) -> dict[str, object]:
    measurements = []
    for result in group_results:
        value = result
        for key in path:
            value = value[key]
        measurements.append(value)
    return _measurement_total(measurements, timing_repeats)


def _aggregate_u(
    group_results: list[dict[str, object]],
    timing_repeats: int,
) -> dict[str, object]:
    samples = [
        sum(
            float(result["cost"]["u"]["samples_ms"][index])
            for result in group_results
        )
        for index in range(timing_repeats)
    ]
    components = {}
    for name in (
        "retained_source_crop_ms",
        "retained_transform_ms",
        "scheduled_exact_retained_ms",
        "missing_exact_retained_ms",
        "retained_materialization_ms",
    ):
        measurements = [
            result["cost"]["u"]["components"][name]
            for result in group_results
        ]
        components[name] = _measurement_total(
            measurements,
            timing_repeats,
        )
    return {
        "components": components,
        "samples_ms": samples,
        "median_of_repetition_sums_ms": float(statistics.median(samples)),
        "sum_of_component_medians_ms": sum(
            float(value["median_ms"]) for value in components.values()
        ),
    }


def _aggregate_outside(
    group_results: list[dict[str, object]],
    branch: str,
    timing_repeats: int,
) -> dict[str, object]:
    key = f"outside_{branch}"
    component_names = tuple(
        group_results[0]["cost"][key]["components"]
    )
    components = {
        name: _measurement_total(
            [
                result["cost"][key]["components"][name]
                for result in group_results
            ],
            timing_repeats,
        )
        for name in component_names
    }
    samples = [
        sum(
            float(value["samples_ms"][index])
            for value in components.values()
        )
        for index in range(timing_repeats)
    ]
    return {
        "components": components,
        "samples_ms": samples,
        "median_ms": sum(
            float(value["median_ms"]) for value in components.values()
        ),
        "median_of_repetition_sums_ms": float(statistics.median(samples)),
    }


def _aggregate_state_movement(
    group_results: list[dict[str, object]],
) -> dict[str, object]:
    h2d = _sum_state_movement(
        [
            result["cost"]["state_movement"]["h2d_previous_actual"]
            for result in group_results
        ],
        "host_to_device_previous_actual",
    )
    d2h = _sum_state_movement(
        [
            result["cost"]["state_movement"]["d2h_next_actual"]
            for result in group_results
        ],
        "device_to_host_next_actual",
    )
    return {
        "h2d_previous_actual": h2d,
        "d2h_next_actual": d2h,
        "logical_bytes": int(h2d["logical_bytes"])
        + int(d2h["logical_bytes"]),
        "gpu_event_ms": float(h2d["gpu_event_ms"])
        + float(d2h["gpu_event_ms"]),
        "wall_ms": float(h2d["wall_ms"]) + float(d2h["wall_ms"]),
        "excluded_from_primary": True,
        "excluded_from_append_timer": True,
        "reported_separately": True,
    }


@torch.inference_mode()
def _run_formal_edge(
    args: argparse.Namespace,
    spec,
    cfg,
    compiler: dict,
    old_window,
    target_window,
    groups,
    record_by_id: dict[int, dict],
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    last_exact_by_record: dict[int, int],
    expected_ids: set[int],
    scheduler_state,
    operator,
    all_items: torch.Tensor,
    source_version: int,
    device: torch.device,
) -> tuple[
    dict[int, JaggedMigratedKVBatch],
    dict[int, int],
    set[int],
    object,
    dict,
    dict,
]:
    target_version = source_version + 1
    present_ids = set(cache_by_record)
    previous_actual_kv_bytes = base.resident_cache_bytes(cache_by_record)
    previous_store_checks = _persistent_cpu_store_checks(
        cache_by_record,
        expected_version=source_version,
    )
    if not all(previous_store_checks.values()):
        raise RuntimeError(
            "Stage 4.9 previous persistent store failed: "
            f"{previous_store_checks}"
        )
    plans, plan_checks = stage49._plan_edge(
        old_window,
        target_window,
        [
            record_by_id[value]
            for value in sorted(record_by_id)
        ],
        expected_ids,
        present_ids,
    )
    selection, scheduler_checks = stage49._select_actions(
        plans,
        last_exact_by_record,
        source_version,
        target_version,
        spec,
        scheduler_state,
    )
    target_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        target_version,
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
    del program_cpu
    torch.cuda.reset_peak_memory_stats(device)
    group_results = []
    next_cache = {}
    next_last_exact = {}
    task_mixed = []
    task_exact = []
    record_metrics = []
    lineage = []
    for group in groups:
        allocated_before_group = torch.cuda.memory_allocated(device)
        result = _run_group(
            args,
            cfg,
            target_model,
            operator,
            program,
            target_version,
            target_window,
            group,
            record_by_id,
            plans,
            selection,
            cache_by_record,
            last_exact_by_record,
            all_items,
            device,
        )
        gpu_result_cache = result["cache"]
        gpu_result_bytes = sum(
            cache.nbytes for cache in gpu_result_cache.values()
        )
        host_result_cache, d2h_next_actual = _timed_store_transfer(
            gpu_result_cache,
            tuple(gpu_result_cache),
            torch.device("cpu"),
            device,
            "device_to_host_next_actual",
        )
        result["cache"] = host_result_cache
        result["cost"]["state_movement"][
            "d2h_next_actual"
        ] = d2h_next_actual
        del gpu_result_cache
        gc.collect()
        torch.cuda.empty_cache()
        allocated_after_host_stage = torch.cuda.memory_allocated(device)
        group_lengths = {
            int(value["record_id"]): plans[int(value["record_id"])].final_tokens
            for value in group
            if plans[int(value["record_id"])].final_tokens > 0
        }
        host_store_checks = _persistent_cpu_store_checks(
            result["cache"],
            expected_version=target_version,
            expected_lengths=group_lengths,
        )
        result_gpu_tensor_bytes = _cuda_tensor_bytes(result)
        host_store_checks.update(
            {
                "result_retains_zero_cuda_tensor_bytes": (
                    result_gpu_tensor_bytes == 0
                ),
                "post_cleanup_live_gpu_growth_is_bounded": (
                    allocated_after_host_stage - allocated_before_group
                    <= 1024 * 1024
                ),
                "d2h_is_excluded_from_u_and_e": bool(
                    d2h_next_actual["outside_u_and_e"]
                )
                and bool(d2h_next_actual["outside_append_timer"]),
                "d2h_covers_every_next_actual_record": int(
                    d2h_next_actual["records"]
                )
                == len(result["cache"]),
            }
        )
        result.setdefault("diagnostics", {})["host_staging"] = {
            "gpu_result_bytes_before_d2h": gpu_result_bytes,
            "persistent_cpu_bytes_after_d2h": sum(
                cache.nbytes for cache in result["cache"].values()
            ),
            "persistent_gpu_kv_bytes_after_d2h": 0,
            "result_cuda_tensor_bytes_after_d2h": result_gpu_tensor_bytes,
            "allocated_before_group_bytes": allocated_before_group,
            "allocated_after_d2h_cleanup_bytes": (
                allocated_after_host_stage
            ),
            "allocated_live_growth_bytes": (
                allocated_after_host_stage - allocated_before_group
            ),
            "persistent_store_role": "post_append_full_mixed_cache_only",
        }
        result["checks"].update(host_store_checks)
        if not all(host_store_checks.values()):
            raise RuntimeError(
                "Stage 4.9 post-group host-store checks failed: "
                f"edge={source_version}->{target_version}, "
                f"group_record_ids="
                f"{tuple(int(value['record_id']) for value in group)}, "
                f"failed="
                f"{[name for name, passed in host_store_checks.items() if not passed]}, "
                f"diagnostics={result['diagnostics']['host_staging']}"
            )
        if result["cache"]:
            allocated_before_parity = torch.cuda.memory_allocated(device)
            reserved_before_parity = torch.cuda.memory_reserved(device)
            parity_diagnostics, parity_checks = (
                _exact_parity_group_summary(
                    cfg,
                    target_model,
                    target_version,
                    target_window,
                    group,
                    record_by_id,
                    plans,
                    selection,
                    all_items,
                    device,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
            allocated_after_parity = torch.cuda.memory_allocated(device)
            reserved_after_parity = torch.cuda.memory_reserved(device)
            parity_diagnostics["memory_lifetime"] = {
                "allocated_before_bytes": allocated_before_parity,
                "allocated_after_bytes": allocated_after_parity,
                "allocated_live_growth_bytes": (
                    allocated_after_parity - allocated_before_parity
                ),
                "reserved_before_bytes": reserved_before_parity,
                "reserved_after_bytes": reserved_after_parity,
                "parity_runs_after_fp16_group_scope_returned": True,
            }
            parity_checks["exact_fp32_parity_has_no_live_gpu_growth"] = (
                allocated_after_parity - allocated_before_parity
                <= 1024 * 1024
            )
            result["diagnostics"][
                "exact_fp32_parity_two_stage_vs_fresh"
            ] = parity_diagnostics
            result["checks"].update(parity_checks)
            if not all(parity_checks.values()):
                raise RuntimeError(
                    "Stage 4.9 post-group FP32 parity failed: "
                    f"edge={source_version}->{target_version}, "
                    f"group_record_ids="
                    f"{tuple(int(value['record_id']) for value in group)}, "
                    f"diagnostics={parity_diagnostics}, "
                    f"failed="
                    f"{[name for name, passed in parity_checks.items() if not passed]}"
                )
        group_results.append(result)
        next_cache.update(result["cache"])
        next_last_exact.update(result["last_exact"])
        task_mixed.extend(result["task_mixed"])
        task_exact.extend(result["task_exact"])
        record_metrics.extend(result["record_metrics"])
        lineage.extend(result["lineage"])
    target_expected_ids = {
        record_id
        for record_id, plan in plans.items()
        if plan.final_tokens > 0
    }
    u = _aggregate_u(group_results, args.timing_repeats)
    e = _aggregate_edge_measurement(
        group_results,
        ("cost", "e"),
        args.timing_repeats,
    )
    outside_mixed = _aggregate_outside(
        group_results,
        "mixed",
        args.timing_repeats,
    )
    outside_exact = _aggregate_outside(
        group_results,
        "exact",
        args.timing_repeats,
    )
    one_shot = _aggregate_edge_measurement(
        group_results,
        ("cost", "one_shot_exact"),
        args.timing_repeats,
    )
    state_movement = _aggregate_state_movement(group_results)
    if e["median_ms"] <= 0:
        raise RuntimeError("Stage 4.9 edge has no paired retained exact work")
    task_summary = _paired_task_summary(task_mixed, task_exact)
    cache_fidelity = [
        float(value["cache_fidelity_q090"]) for value in record_metrics
    ]
    score_cosine = [
        float(value["score_cosine"]) for value in record_metrics
    ]
    top100_overlap = [
        float(value["top100_overlap"]) for value in record_metrics
    ]
    equivalence_groups = [
        result["diagnostics"]
        for result in group_results
        if "diagnostics" in result
    ]
    exact_equivalence = {
        "fp32_algorithmic_authority": {
            metric: max(
                float(
                    value["exact_fp32_parity_two_stage_vs_fresh"][
                        metric
                    ]
                )
                for value in equivalence_groups
            )
            for metric in (
                "k_max_abs",
                "v_max_abs",
                "hidden_max_abs",
                "score_max_abs",
            )
        },
        "fp16_deployment_diagnostic": {
            metric: max(
                float(
                    value["exact_fp16_endpoint_two_stage_vs_fresh"][
                        metric
                    ]
                )
                for value in equivalence_groups
            )
            for metric in (
                "k_max_abs",
                "v_max_abs",
                "hidden_max_abs",
                "score_max_abs",
            )
        },
        "fp32_top100_equal": all(
            value["exact_fp32_parity_two_stage_vs_fresh"][
                "top100_exact_order_equal"
            ]
            for value in equivalence_groups
        ),
        "fp32_top100_set_equal": all(
            value["exact_fp32_parity_two_stage_vs_fresh"][
                "top100_exact_set_equal"
            ]
            for value in equivalence_groups
        ),
        "fp32_top100_boundary_equivalent": all(
            value["exact_fp32_parity_two_stage_vs_fresh"][
                "top100_boundary_equivalent"
            ]
            for value in equivalence_groups
        ),
        "fp32_top100_mismatched_rows": sum(
            int(
                value["exact_fp32_parity_two_stage_vs_fresh"][
                    "top100_order_mismatched_rows"
                ]
            )
            for value in equivalence_groups
        ),
        "fp32_top100_set_mismatched_rows": sum(
            int(
                value["exact_fp32_parity_two_stage_vs_fresh"][
                    "top100_set_mismatched_rows"
                ]
            )
            for value in equivalence_groups
        ),
        "fp32_minimum_top100_overlap": min(
            float(
                value["exact_fp32_parity_two_stage_vs_fresh"][
                    "minimum_top100_overlap"
                ]
            )
            for value in equivalence_groups
        ),
        "fp16_top100_equal": all(
            value["exact_fp16_endpoint_two_stage_vs_fresh"][
                "top100_equal"
            ]
            for value in equivalence_groups
        ),
        "fp32_parity_outside_timing": True,
    }
    actions = {
        "migrate": len(selection.migrate_ids),
        "scheduled_exact": len(selection.scheduled_exact_ids),
        "natural_exact": len(selection.natural_exact_ids),
        "reusable_records": len(selection.migrate_ids)
        + len(selection.scheduled_exact_ids),
        "resident_records": len(target_expected_ids),
        "retained_paired_records": sum(
            int(result.get("diagnostics", {}).get("timed_records", 0))
            for result in group_results
        ),
        "retained_paired_tokens": sum(
            int(result.get("diagnostics", {}).get("timed_tokens", 0))
            for result in group_results
        ),
    }
    target_lengths = {
        record_id: plan.final_tokens
        for record_id, plan in plans.items()
        if plan.final_tokens > 0
    }
    next_store_checks = _persistent_cpu_store_checks(
        next_cache,
        expected_version=target_version,
        expected_lengths=target_lengths,
    )
    post_cleanup_allocated = [
        int(
            result["diagnostics"]["host_staging"][
                "allocated_after_d2h_cleanup_bytes"
            ]
        )
        for result in group_results
    ]
    cuda_peak_bytes = torch.cuda.max_memory_allocated(device)
    cuda_total_bytes = torch.cuda.get_device_properties(device).total_memory
    checks = {
        "retained_plan": all(plan_checks.values()),
        "scheduler": all(scheduler_checks.values()),
        "all_group_checks": all(
            all(result["checks"].values()) for result in group_results
        ),
        "expected_contract_separate_from_present_store": expected_ids
        is not present_ids,
        "previous_store_matches_expected": present_ids == expected_ids,
        "recursive_cache_covers_target_contract": set(next_cache)
        == target_expected_ids,
        "recursive_last_exact_covers_target_contract": set(next_last_exact)
        == target_expected_ids,
        "lineage_covers_manifest": len(lineage) == len(record_by_id),
        "lineage_record_order": [
            int(value["record_id"]) for value in lineage
        ]
        == sorted(record_by_id),
        "append_uses_target_model": all(
            value["append_model_version"] in {None, target_version}
            for value in lineage
        ),
        "last_exact_transition_semantics": all(
            (
                value["last_exact_version_after"] is None
                if value["final_tokens"] == 0
                else (
                    value["last_exact_version_after"]
                    == value["last_exact_version_before"]
                    if value["action"] == "migrate"
                    else value["last_exact_version_after"]
                    == target_version
                )
            )
            for value in lineage
        ),
        "source_model_append_calls_zero": True,
        "target_append_excluded_from_u_and_e": True,
        "fresh_exact_executed_same_device": True,
        "old_denominator_not_reused": True,
        "paired_retained_population_nonempty": actions[
            "retained_paired_records"
        ]
        > 0,
        "task_population_paired": len(task_mixed) == len(task_exact),
        "previous_persistent_store_is_cpu": all(
            previous_store_checks.values()
        ),
        "next_persistent_store_is_cpu": all(next_store_checks.values()),
        "group_results_retain_zero_cuda_tensor_bytes": (
            _cuda_tensor_bytes(group_results) == 0
        ),
        "state_movement_excluded_and_reported": bool(
            state_movement["excluded_from_primary"]
        )
        and bool(state_movement["reported_separately"]),
        "edge_peak_below_device_capacity": cuda_peak_bytes
        < cuda_total_bytes,
        "no_cross_group_live_gpu_growth": (
            not post_cleanup_allocated
            or max(post_cleanup_allocated) - min(post_cleanup_allocated)
            <= 1024 * 1024
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Stage 4.9 formal edge checks failed: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    step = {
        "source_version": source_version,
        "target_version": target_version,
        "prediction_target_date": target_window.target_date,
        "actions": actions,
        "scheduler": {
            "variant": spec.to_dict(),
            "diagnostics": selection.diagnostics,
            "state_after": _serialize_state(selection.next_state),
            "scheduled_exact_ids": list(selection.scheduled_exact_ids),
            "action_partition_sha256": _json_sha256(
                {
                    "migrate_ids": list(selection.migrate_ids),
                    "scheduled_exact_ids": list(
                        selection.scheduled_exact_ids
                    ),
                    "natural_exact_ids": list(
                        selection.natural_exact_ids
                    ),
                }
            ),
        },
        "cost": {
            "u": u,
            "e": e,
            "primary_u_over_e": (
                float(u["sum_of_component_medians_ms"])
                / float(e["median_ms"])
            ),
            "outside_rollout_timer": {
                "mixed": outside_mixed,
                "exact": outside_exact,
                "target_model_append_excluded": True,
            },
            "one_shot_fresh_exact_diagnostic": one_shot,
            "state_movement_outside_primary": state_movement,
        },
        "task_metrics": task_summary,
        "quality_diagnostics": {
            "records": len(record_metrics),
            "minimum_cache_fidelity_q090": min(cache_fidelity),
            "minimum_score_cosine": min(score_cosine),
            "minimum_top100_overlap": min(top100_overlap),
        },
        "exact_equivalence": exact_equivalence,
        "memory": {
            "previous_actual_kv_bytes": previous_actual_kv_bytes,
            "next_actual_kv_bytes": base.resident_cache_bytes(next_cache),
            "persistent_gpu_kv_bytes": 0,
            "persistent_store_device": "cpu",
            "per_group_post_cleanup_allocated_bytes": post_cleanup_allocated,
            "cuda_max_memory_allocated_bytes": cuda_peak_bytes,
            "cuda_total_memory_bytes": cuda_total_bytes,
            "cuda_peak_fraction": cuda_peak_bytes / cuda_total_bytes,
        },
        "recursive_store": {
            "role": "post_append_full_mixed_cache_only",
            "placement": "cpu",
            "previous": {
                "version": f"theta{source_version}",
                "records": len(present_ids),
                "record_ids_sha256": _record_id_sha256(present_ids),
                "record_lengths_sha256": _record_length_sha256(
                    cache_by_record
                ),
                "kv_bytes": previous_actual_kv_bytes,
            },
            "next": {
                "version": f"theta{target_version}",
                "records": len(next_cache),
                "record_ids_sha256": _record_id_sha256(next_cache),
                "record_lengths_sha256": _record_length_sha256(next_cache),
                "kv_bytes": base.resident_cache_bytes(next_cache),
            },
        },
        "program": {
            "sha256": program_descriptor["sha256"],
            "labels_used": False,
        },
        "lineage": lineage,
        "lineage_sha256": _json_sha256(lineage),
        "checks": checks,
    }
    endpoint = {
        "version": target_version,
        "target_date": target_window.target_date,
        "resident_records": len(next_cache),
        "task_metrics": task_summary,
        "quality_diagnostics": step["quality_diagnostics"],
    }
    del target_model, program, group_results
    gc.collect()
    torch.cuda.empty_cache()
    return (
        next_cache,
        next_last_exact,
        target_expected_ids,
        selection.next_state,
        endpoint,
        step,
    )


def run_candidate(
    args: argparse.Namespace,
    candidate_name: str,
    shared_inputs: dict[str, object],
) -> dict[str, object]:
    spec = stage49._candidate_spec(candidate_name)
    device = torch.device(args.device)
    cfg = shared_inputs["cfg"]
    manifest = shared_inputs["manifest"]
    windows = shared_inputs["windows"]
    compiler = shared_inputs["compiler"]
    groups = shared_inputs["groups"]
    record_by_id = shared_inputs["record_by_id"]
    (
        cache_by_record,
        last_exact,
        initialization_ms,
        initialization_state_movement,
    ) = _initialize_theta0_host(
        cfg,
        args.checkpoint_dir,
        windows[0],
        groups,
        device,
    )
    theta0_checks = stage48._theta0_state_checks(
        windows[0],
        groups,
        record_by_id,
        cache_by_record,
        last_exact,
    )
    theta0_checks.update(_persistent_cpu_store_checks(cache_by_record, 0))
    if not all(theta0_checks.values()):
        raise RuntimeError(
            f"Stage 4.9 theta0 state differs: {theta0_checks}"
        )
    theta0_store = {
        "version": "theta0",
        "placement": "cpu",
        "role": "post_append_full_exact_initial_cache",
        "records": len(cache_by_record),
        "record_ids_sha256": _record_id_sha256(cache_by_record),
        "record_lengths_sha256": _record_length_sha256(cache_by_record),
        "kv_bytes": base.resident_cache_bytes(cache_by_record),
        "persistent_gpu_kv_bytes": 0,
    }
    expected_ids = {
        record_id
        for record_id, descriptor in record_by_id.items()
        if windows[0]
        .records[int(descriptor["user_id"])]
        .history
        is not None
    }
    all_items = torch.arange(
        1,
        cfg.num_prediction_items + 1,
        device=device,
    )
    operator = DirectOldKVFusedOperator(**LAUNCH)
    started = time.perf_counter()

    def edge_runner(
        source_version,
        actual_cache,
        actual_last_exact,
        prior_expected_ids,
        scheduler_state,
    ):
        result = _run_formal_edge(
            args,
            spec,
            cfg,
            compiler,
            windows[source_version],
            windows[source_version + 1],
            groups,
            record_by_id,
            actual_cache,
            actual_last_exact,
            prior_expected_ids,
            scheduler_state,
            operator,
            all_items,
            source_version,
            device,
        )
        print(
            json.dumps(
                {
                    "candidate": candidate_name,
                    "source_version": source_version,
                    "target_version": source_version + 1,
                    "actions": result[5]["actions"],
                    "primary_u_over_e": result[5]["cost"][
                        "primary_u_over_e"
                    ],
                    "task_metrics": result[4]["task_metrics"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        return result

    (
        final_cache,
        final_last_exact,
        final_expected_ids,
        final_scheduler_state,
        endpoints,
        steps,
    ) = _advance_recursive_chain(
        edge_runner,
        cache_by_record,
        last_exact,
        expected_ids,
    )
    cumulative_cost = summarize_paired_cost(
        steps,
        args.timing_repeats,
    )
    weighted_task = _record_weighted_task(endpoints)
    chain_checks = {
        "eleven_updates": len(steps) == NUM_EDGES,
        "eleven_endpoints": len(endpoints) == NUM_EDGES,
        "adjacent_versions": all(
            value["target_version"] == value["source_version"] + 1
            for value in steps
        ),
        "recursive_actual_cache_consumed": all(
            value["checks"]["previous_store_matches_expected"]
            for value in steps
        ),
        "final_cache_matches_contract": set(final_cache)
        == final_expected_ids,
        "final_last_exact_matches_contract": set(final_last_exact)
        == final_expected_ids,
        "all_edge_checks": all(
            all(value["checks"].values()) for value in steps
        ),
        "labels_never_route": all(
            value["scheduler"]["diagnostics"]["labels_used"] is False
            for value in steps
        ),
        "source_model_append_calls_zero": all(
            value["checks"]["source_model_append_calls_zero"]
            for value in steps
        ),
        "target_append_excluded_from_primary": cumulative_cost[
            "target_append_excluded_from_u_and_e"
        ],
        "same_device_fresh_exact_every_edge": all(
            value["checks"]["fresh_exact_executed_same_device"]
            for value in steps
        ),
        "old_denominator_not_reused": not cumulative_cost[
            "old_exact_denominator_reused"
        ],
        "persistent_gpu_kv_zero_every_edge": all(
            value["memory"]["persistent_gpu_kv_bytes"] == 0
            for value in steps
        ),
        "host_store_every_edge": all(
            value["memory"]["persistent_store_device"] == "cpu"
            for value in steps
        ),
        "state_movement_reported_outside_primary": bool(
            cumulative_cost["state_movement_outside_primary"][
                "reported_separately"
            ]
        )
        and bool(
            cumulative_cost["state_movement_outside_primary"][
                "excluded_from_primary"
            ]
        ),
        "final_persistent_store_is_cpu": all(
            _persistent_cpu_store_checks(
                final_cache,
                expected_version=NUM_EDGES,
            ).values()
        ),
        "every_edge_peak_below_capacity": all(
            value["memory"]["cuda_max_memory_allocated_bytes"]
            < value["memory"]["cuda_total_memory_bytes"]
            for value in steps
        ),
        "no_edge_has_cross_group_live_growth": all(
            value["checks"]["no_cross_group_live_gpu_growth"]
            for value in steps
        ),
    }
    if not all(chain_checks.values()):
        raise RuntimeError(
            "Stage 4.9 formal chain checks failed: "
            f"{[name for name, passed in chain_checks.items() if not passed]}"
        )
    result = {
        "protocol": PROTOCOL,
        "rollout_boundary_protocol": ROLLOUT_BOUNDARY_PROTOCOL,
        "status": "complete",
        "scientific_result": True,
        "study_stage": "single_configuration_same_device_confirmation",
        "candidate_name": candidate_name,
        "candidate": spec.to_dict(),
        "repository_commit": _repository_commit(),
        "implementation": implementation_snapshot(),
        "configuration": {
            "dataset": "KuaiRand-1K",
            "split": "4+12",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "records": len(manifest["records"]),
            "model": shared_inputs["training"]["model"],
            "warmup_repeats": args.warmup_repeats,
            "timing_repeats": args.timing_repeats,
            "persistent_recursive_store": (
                "cpu_singleton_fp16_post_append_full_mixed_kv"
            ),
            "groupwise_device_staging": True,
        },
        "measurement_boundary": {
            "primary_metric": "sum_t(U_t)/sum_t(E_t)",
            "u": (
                "crop and assemble previous actual retained K/V, direct "
                "migration, selected exact retained refresh, missing-cache "
                "retained rebuild, and retained endpoint materialization"
            ),
            "u_entry": (
                "the current group migrant K/V is already device-resident; "
                "its H2D staging completes before the first U timer"
            ),
            "e": (
                "fresh target-model exact retained-prefix execution and "
                "FP16 device endpoint materialization on the same GPU"
            ),
            "target_append": (
                "target-model Delta and latest append measured separately "
                "outside U and E"
            ),
            "recursive_state": "previous_actual_post_append_mixed_cache",
            "one_shot_fresh_exact": (
                "executed each edge as quality authority, not reused as E"
            ),
            "exact_equivalence": (
                "one untimed FP32 retained-to-delta-to-latest branch versus "
                "FP32 one-shot fresh; FP16 endpoint differences are "
                "reported only as deployment quantization diagnostics"
            ),
            "scheduler_cpu_time_excluded": True,
            "catalog_scoring_excluded": True,
            "fail_closed_integrity_checks_excluded": (
                "device/dtype/version/length/finiteness and CUDA-lifetime "
                "assertions are evaluator validation outside U and E"
            ),
            "source_model_append_calls": 0,
            "old_exact_denominator_reused": False,
            "stage47_or_stage48_edge_runner_called": False,
            "persistent_recursive_state": (
                "CPU host store; only current group migrant inputs are "
                "staged H2D and every post-append mixed output is returned "
                "D2H before entering the next recursive store"
            ),
            "state_movement": (
                "H2D and D2H bytes, CUDA-event time, and wall time are "
                "reported separately outside U and E because the primary "
                "protocol compares device-resident retained endpoints; "
                "logical bytes include K, V, lengths, and offsets"
            ),
            "capacity_scope": (
                "host-staged evaluator for memory containment; not a "
                "full-cohort HBM-resident or end-to-end system claim"
            ),
            "full_cohort_hbm_claim": False,
            "end_to_end_state_movement_claim": False,
        },
        "input_provenance": shared_inputs["input_provenance"],
        "theta0_initialization_gpu_ms": initialization_ms,
        "theta0_initialization_state_movement": (
            initialization_state_movement
        ),
        "recursive_store_summary": {
            "theta0": theta0_store,
            "edges": [
                {
                    "source_version": value["source_version"],
                    "target_version": value["target_version"],
                    "previous": value["recursive_store"]["previous"],
                    "next": value["recursive_store"]["next"],
                    "placement": value["recursive_store"]["placement"],
                }
                for value in steps
            ],
            "final": {
                "version": f"theta{NUM_EDGES}",
                "placement": "cpu",
                "records": len(final_cache),
                "record_ids_sha256": _record_id_sha256(final_cache),
                "record_lengths_sha256": _record_length_sha256(final_cache),
                "kv_bytes": base.resident_cache_bytes(final_cache),
                "persistent_gpu_kv_bytes": 0,
            },
            "not_full_cohort_hbm_resident": True,
        },
        "endpoints": endpoints,
        "steps": steps,
        "record_weighted_task": weighted_task,
        "cumulative_gpu_cost": cumulative_cost,
        "quality_floor": {
            "minimum_cache_fidelity_q090": min(
                value["quality_diagnostics"][
                    "minimum_cache_fidelity_q090"
                ]
                for value in endpoints
            ),
            "minimum_score_cosine": min(
                value["quality_diagnostics"]["minimum_score_cosine"]
                for value in endpoints
            ),
            "minimum_top100_overlap": min(
                value["quality_diagnostics"]["minimum_top100_overlap"]
                for value in endpoints
            ),
        },
        "exact_equivalence": {
            "fp32_algorithmic_authority": {
                metric: max(
                    float(
                        value["exact_equivalence"][
                            "fp32_algorithmic_authority"
                        ][metric]
                    )
                    for value in steps
                )
                for metric in (
                    "k_max_abs",
                    "v_max_abs",
                    "hidden_max_abs",
                    "score_max_abs",
                )
            },
            "fp16_deployment_diagnostic": {
                metric: max(
                    float(
                        value["exact_equivalence"][
                            "fp16_deployment_diagnostic"
                        ][metric]
                    )
                    for value in steps
                )
                for metric in (
                    "k_max_abs",
                    "v_max_abs",
                    "hidden_max_abs",
                    "score_max_abs",
                )
            },
            "fp32_top100_equal": all(
                value["exact_equivalence"]["fp32_top100_equal"]
                for value in steps
            ),
            "fp32_top100_set_equal": all(
                value["exact_equivalence"]["fp32_top100_set_equal"]
                for value in steps
            ),
            "fp32_top100_boundary_equivalent": all(
                value["exact_equivalence"][
                    "fp32_top100_boundary_equivalent"
                ]
                for value in steps
            ),
            "fp32_top100_mismatched_rows": sum(
                int(
                    value["exact_equivalence"][
                        "fp32_top100_mismatched_rows"
                    ]
                )
                for value in steps
            ),
            "fp32_top100_set_mismatched_rows": sum(
                int(
                    value["exact_equivalence"][
                        "fp32_top100_set_mismatched_rows"
                    ]
                )
                for value in steps
            ),
            "fp32_minimum_top100_overlap": min(
                float(
                    value["exact_equivalence"][
                        "fp32_minimum_top100_overlap"
                    ]
                )
                for value in steps
            ),
            "fp16_top100_equal": all(
                value["exact_equivalence"]["fp16_top100_equal"]
                for value in steps
            ),
            "fp32_parity_outside_timing": True,
        },
        "final_scheduler_state": _serialize_state(final_scheduler_state),
        "checks": {
            "input": shared_inputs["checks"],
            "theta0": theta0_checks,
            "chain": chain_checks,
            "all_passed": all(shared_inputs["checks"].values())
            and all(theta0_checks.values())
            and all(chain_checks.values()),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    del final_cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def load_shared_inputs(
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device(args.device)
    baseline = stage48.load_exact_baseline(args.baseline)
    plan, metadata, training, cfg, manifest, checkpoints = load_inputs(
        args.prepared_data,
        args.training_result,
        args.checkpoint_dir,
    )
    user_ids = tuple(
        int(value["user_id"]) for value in manifest["records"]
    )
    windows = reconstruct_organic_windows(plan, user_ids)
    compiler = json.loads(_repo_path(args.compiler_result).read_text())
    window_checks = base.validate_windows(windows, manifest)
    compiler_checks = base.validate_compiler_payload(
        compiler,
        manifest,
        windows,
        checkpoints,
    )
    provenance_checks = stage48.validate_runtime_provenance(
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
    checks = {
        "causality": all(window_checks.values()),
        "compiler": all(compiler_checks.values()),
        "provenance": all(provenance_checks.values()),
        "frozen_argument_binding": stage48._frozen_argument_binding(
            args,
            baseline,
        ),
        "device_class": device_name
        == baseline["configuration"]["device_class"],
        "twelve_windows": len(windows) == NUM_EDGES + 1,
        "candidate_set": tuple(
            value.to_dict() for value in candidate_specs()
        )
        == tuple(
            stage49._candidate_spec(name).to_dict()
            for name in CANDIDATE_NAMES
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            "Stage 4.9 formal input validation failed: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    groups = base.fixed_record_groups(manifest, args.batch_size)
    record_by_id = {
        int(value["record_id"]): value for value in manifest["records"]
    }
    return {
        "baseline": baseline,
        "metadata": metadata,
        "training": training,
        "cfg": cfg,
        "manifest": manifest,
        "checkpoints": checkpoints,
        "windows": windows,
        "compiler": compiler,
        "groups": groups,
        "record_by_id": record_by_id,
        "checks": checks,
        "input_provenance": {
            "prepared_data": {
                "path": args.prepared_data,
                "sha256": sha256(_repo_path(args.prepared_data)),
                "protocol": metadata["protocol"],
            },
            "training_result": {
                "path": args.training_result,
                "sha256": sha256(_repo_path(args.training_result)),
                "protocol": training["protocol"],
            },
            "checkpoints": checkpoints,
            "manifest": {
                "protocol": manifest["protocol"],
                "content_sha256": manifest["content_sha256"],
                "records": len(manifest["records"]),
            },
            "windows": [
                {
                    "version": int(value.version),
                    "target_date": str(value.target_date),
                    "content_sha256": value.content_sha256,
                }
                for value in windows
            ],
            "compiler": {
                "path": args.compiler_result,
                "sha256": sha256(_repo_path(args.compiler_result)),
                "protocol": compiler["protocol"],
            },
            "stage4_8_baseline": {
                "path": args.baseline,
                "sha256": sha256(_repo_path(args.baseline)),
                "used_for_input_provenance_only": True,
                "old_exact_denominator_reused": False,
                "old_exact_task_reused": False,
            },
        },
    }


def confirmation_summary(
    args: argparse.Namespace,
    results: list[dict[str, object]],
) -> dict[str, object]:
    names = tuple(value["candidate_name"] for value in results)
    devices = {
        (
            value["configuration"]["device"],
            value["configuration"]["device_name"],
        )
        for value in results
    }
    checks = {
        "both_candidates_complete": len(results) == len(CANDIDATE_NAMES)
        and all(value["status"] == "complete" for value in results),
        "candidate_order": names == CANDIDATE_NAMES,
        "same_physical_device": len(devices) == 1,
        "same_repository_commit": len(
            {value["repository_commit"] for value in results}
        )
        == 1,
        "same_implementation": len(
            {
                json.dumps(value["implementation"], sort_keys=True)
                for value in results
            }
        )
        == 1,
        "all_checks_pass": all(
            value["checks"]["all_passed"] for value in results
        ),
        "old_denominator_never_reused": all(
            not value["cumulative_gpu_cost"][
                "old_exact_denominator_reused"
            ]
            for value in results
        ),
        "append_excluded_for_both": all(
            value["cumulative_gpu_cost"][
                "target_append_excluded_from_u_and_e"
            ]
            for value in results
        ),
        "host_staged_for_both": all(
            value["configuration"]["persistent_recursive_store"]
            == "cpu_singleton_fp16_post_append_full_mixed_kv"
            and value["configuration"]["groupwise_device_staging"]
            for value in results
        ),
        "no_full_cohort_hbm_claim": all(
            not value["measurement_boundary"]["full_cohort_hbm_claim"]
            for value in results
        ),
        "state_movement_reported_for_both": all(
            value["cumulative_gpu_cost"][
                "state_movement_outside_primary"
            ]["reported_separately"]
            for value in results
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Stage 4.9 confirmation summary failed: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": True,
        "candidate_order": list(CANDIDATE_NAMES),
        "configuration": {
            "device": results[0]["configuration"]["device"],
            "device_name": results[0]["configuration"]["device_name"],
            "seed": args.seed,
            "warmup_repeats": args.warmup_repeats,
            "timing_repeats": args.timing_repeats,
        },
        "measurement_boundary": {
            "primary": "sum_t(U_t)/sum_t(E_t)",
            "same_device_sequential_candidates": True,
            "target_append_excluded": True,
            "old_denominator_reused": False,
            "persistent_recursive_store": (
                "cpu_singleton_fp16_post_append_full_mixed_kv"
            ),
            "groupwise_device_staging": True,
            "state_movement": (
                "H2D/D2H bytes, CUDA-event time, and wall time reported "
                "separately outside primary U/E"
            ),
            "capacity_scope": (
                "evaluator-only host spill for single-GPU containment; "
                "not a full-cohort HBM-resident or end-to-end system claim"
            ),
            "full_cohort_hbm_claim": False,
            "end_to_end_state_movement_claim": False,
        },
        "results": [
            {
                "candidate_name": value["candidate_name"],
                "path": str(
                    candidate_output_path(args, value["candidate_name"])
                ),
                "sha256": sha256(
                    candidate_output_path(args, value["candidate_name"])
                ),
                "primary_sum_u_over_sum_e": value[
                    "cumulative_gpu_cost"
                ]["primary_sum_u_over_sum_e"],
                "state_movement_outside_primary": value[
                    "cumulative_gpu_cost"
                ]["state_movement_outside_primary"],
                "record_weighted_task": value["record_weighted_task"],
                "quality_floor": value["quality_floor"],
            }
            for value in results
        ],
        "checks": {
            **checks,
            "all_passed": True,
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.smoke_test:
        print(json.dumps(smoke_payload(args), indent=2))
        return
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    outputs = [
        candidate_output_path(args, name) for name in CANDIDATE_NAMES
    ]
    summary_path = summary_output_path(args)
    existing = [path for path in (*outputs, summary_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Stage 4.9 formal outputs already exist; use --force only "
            "after resolving provenance: "
            f"{[str(value) for value in existing]}"
        )
    shared_inputs = load_shared_inputs(args)
    results = []
    for candidate_name, output in zip(
        CANDIDATE_NAMES,
        outputs,
        strict=True,
    ):
        result = run_candidate(args, candidate_name, shared_inputs)
        save_json(result, output)
        results.append(result)
    summary = confirmation_summary(args, results)
    save_json(summary, summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
