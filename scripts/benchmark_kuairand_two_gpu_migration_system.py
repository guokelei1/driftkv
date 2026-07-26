from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from motivation_validity import eval_batches, move_batch, seed_everything

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    CompiledCacheAdapter,
    FusedMigrationOperator,
    MigrationCapsuleBatch,
    MigrationProgram,
    MultiGPUCohortExecutor,
    MultiGPUFullRecomputeExecutor,
    PackedMigrationOperator,
    PinnedKVOutputPool,
    RawHistoryBatch,
    ReferenceMigrationOperator,
    benchmark_cuda_operator,
    capture_layerwise_state,
)
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.streaming import (
    TWO_GPU_SYSTEM_PROTOCOL,
    load_checkpoint_model,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

ATTENTION_PROGRAM_PROTOCOL = (
    "kuairand_long_context_4plus12_attention_weighted_search_v1"
)
VERIFIED_PLAN_PROTOCOL = (
    "kuairand_long_context_4plus12_verified_compiler_v1"
)
DEFAULT_PREPARED = (
    "data/processed/kuairand_long_context_4plus12_exploration_v1.npz"
)
DEFAULT_TRAINING = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
DEFAULT_CHECKPOINTS = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)
DEFAULT_PROGRAM_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/attention_weighted_search"
)
DEFAULT_MANIFEST_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/verified_plans"
)
DEFAULT_OUTPUT = (
    "results/system/"
    "kuairand_long_context_4plus12_two_gpu_migration_system_seed0.json"
)


@dataclass(frozen=True)
class RealCacheRecord:
    record_id: int
    user_id: int
    source_version: str
    normed: torch.Tensor
    item_ids: torch.Tensor
    behaviors: torch.Tensor
    time_deltas: torch.Tensor

    @property
    def length(self) -> int:
        return self.item_ids.shape[0]

    @property
    def capsule_nbytes(self) -> int:
        return self.normed.numel() * self.normed.element_size()

    @property
    def raw_nbytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.item_ids,
                self.behaviors,
                self.time_deltas,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--program-dir", default=DEFAULT_PROGRAM_DIR)
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--base-days", type=int, default=4)
    parser.add_argument("--source-versions", type=int, nargs="+", default=[0, 4, 10])
    parser.add_argument("--source-weights", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    parser.add_argument("--target-version", type=int, default=11)
    parser.add_argument("--layout-search-users", type=int, default=32)
    parser.add_argument("--max-users", type=int, default=64)
    parser.add_argument("--materialize-batch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bucket-width", type=int, default=32)
    parser.add_argument("--max-inflight", type=int, default=3)
    parser.add_argument("--exact-batch-size", type=int, default=2)
    parser.add_argument("--exact-bucket-width", type=int, default=32)
    parser.add_argument("--exact-max-inflight", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--operator-warmup", type=int, default=5)
    parser.add_argument("--operator-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-diagnostic-protocol", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_difference(
    actual: HSTUKVCache,
    expected: HSTUKVCache,
) -> dict[str, float]:
    delta_k = (actual.k.float() - expected.k.float()).flatten()
    delta_v = (actual.v.float() - expected.v.float()).flatten()
    reference_k = expected.k.float().flatten()
    reference_v = expected.v.float().flatten()
    squared_error = delta_k.square().sum() + delta_v.square().sum()
    squared_reference = (
        reference_k.square().sum() + reference_v.square().sum()
    )
    return {
        "max_abs": max(
            float(delta_k.abs().max()),
            float(delta_v.abs().max()),
        ),
        "rms": float(
            torch.cat((delta_k, delta_v)).square().mean().sqrt()
        ),
        "fro_relative": float(
            squared_error.sqrt() / squared_reference.sqrt().clamp_min(1e-12)
        ),
    }


def split_verified_test(
    samples: list[dict],
    seed: int,
) -> tuple[list[dict], dict]:
    first_order = np.random.default_rng(9151 + seed).permutation(len(samples))
    fit = [samples[index] for index in first_order[:40]]
    selection = [samples[index] for index in first_order[40:100]]
    remaining = [samples[index] for index in first_order[100:]]
    second_order = np.random.default_rng(27183 + seed).permutation(
        len(remaining)
    )
    certificate = [remaining[index] for index in second_order[:60]]
    test = [remaining[index] for index in second_order[60:]]
    return test, {
        "fit_users": len(fit),
        "selection_users": len(selection),
        "certificate_users": len(certificate),
        "final_test_users": len(test),
        "selection_seed": 9151 + seed,
        "certificate_seed": 27183 + seed,
    }


def fixed_count_assignment(
    count: int,
    versions: list[int],
    weights: list[float],
    seed: int,
) -> tuple[list[int], dict[int, int]]:
    if len(versions) != len(weights) or not versions:
        raise ValueError("source versions and weights must have equal nonzero length")
    if any(weight < 0 for weight in weights):
        raise ValueError("source weights must be nonnegative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("source weights must have positive sum")
    normalized = np.asarray(weights, dtype=np.float64) / total
    expected = normalized * count
    counts = np.floor(expected).astype(np.int64)
    remainder = count - int(counts.sum())
    order = sorted(
        range(len(versions)),
        key=lambda index: (-(expected[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    assignments = [
        version
        for version, cohort_count in zip(versions, counts, strict=True)
        for _ in range(int(cohort_count))
    ]
    np.random.default_rng(seed).shuffle(assignments)
    return assignments, {
        version: int(value)
        for version, value in zip(versions, counts, strict=True)
    }


def load_verified_programs(
    args: argparse.Namespace,
    cfg: HSTUConfig,
) -> tuple[tuple[MigrationProgram, ...], list[dict]]:
    programs = []
    evidence = []
    expected_shape = (
        cfg.num_layers,
        cfg.hidden_size,
        2 * cfg.num_heads * cfg.head_dim,
    )
    for source_version in args.source_versions:
        manifest_path = Path(args.manifest_dir) / (
            f"theta{source_version}_to_theta{args.target_version}_verified.json"
        )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("protocol") != VERIFIED_PLAN_PROTOCOL:
            raise ValueError("verified plan protocol mismatch")
        if manifest.get("selected_action") != "compiled_full_affine":
            raise ValueError("verified plan does not select compiled full affine")
        if manifest.get("labels_used") is not False:
            raise ValueError("verified plan must be label free")
        selected = next(
            action
            for action in manifest["actions"]
            if action["name"] == manifest["selected_action"]
        )
        program_path = Path(args.program_dir) / (
            f"theta{source_version}_to_theta{args.target_version}_"
            "compiled_attention_mix_1.00.pt"
        )
        if Path(selected["program_path"]) != program_path:
            raise ValueError("verified plan and requested program path differ")
        payload = torch.load(
            program_path,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("protocol") != ATTENTION_PROGRAM_PROTOCOL:
            raise ValueError("compiled program protocol mismatch")
        if payload.get("source_version") != f"theta{source_version}":
            raise ValueError("compiled program source mismatch")
        if payload.get("target_version") != f"theta{args.target_version}":
            raise ValueError("compiled program target mismatch")
        if payload["weights"].shape != expected_shape:
            raise ValueError("compiled program weight shape mismatch")
        if payload["biases"].shape != (
            cfg.num_layers,
            expected_shape[-1],
        ):
            raise ValueError("compiled program bias shape mismatch")
        if payload.get("fit", {}).get("labels_used") is not False:
            raise ValueError("compiled program fit must be label free")
        program = MigrationProgram(
            source_version=payload["source_version"],
            target_version=payload["target_version"],
            adapter=CompiledCacheAdapter(
                weights=payload["weights"],
                biases=payload["biases"],
                source_rank=cfg.hidden_size,
                ridge=float(payload["ridge"]),
            ),
        )
        certificate = next(
            value
            for value in manifest["certificates"]
            if value["action_name"] == manifest["selected_action"]
        )
        programs.append(program)
        evidence.append(
            {
                "source_version": source_version,
                "program_path": str(program_path),
                "program_sha256": sha256(program_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "program_fp32_bytes": program.nbytes,
                "certificate_cost_ratio": certificate["cost_ratio"],
                "worst_recovery_lower_bound": certificate[
                    "worst_recovery_lower_bound"
                ],
                "worst_coverage_lower_bound": certificate[
                    "worst_coverage_lower_bound"
                ],
                "fallback_actions": manifest["fallback_actions"],
            }
        )
    return tuple(programs), evidence


@torch.inference_mode()
def materialize_records(
    entries: list[tuple[int, dict, int]],
    cfg: HSTUConfig,
    checkpoint_dir: str,
    device: torch.device,
    batch_size: int,
) -> tuple[list[RealCacheRecord], dict]:
    started = time.perf_counter()
    records = []
    per_cohort = {}
    for source_version in sorted({entry[2] for entry in entries}):
        cohort_entries = [
            entry for entry in entries if entry[2] == source_version
        ]
        by_user = {
            int(sample["history"]["user_id"]): record_id
            for record_id, sample, _ in cohort_entries
        }
        cohort_samples = [
            sample
            for _, sample, _ in sorted(
                cohort_entries,
                key=lambda value: (
                    len(value[1]["history"]["item_ids"]),
                    value[0],
                ),
            )
        ]
        model = load_checkpoint_model(
            cfg,
            checkpoint_dir,
            source_version,
            device,
        )
        cohort_started = time.perf_counter()
        cohort_tokens = 0
        for selected, _, prefix_cpu, _ in eval_batches(
            cohort_samples,
            cfg.max_seq_len,
            batch_size,
        ):
            prefix = move_batch(prefix_cpu, device)
            state = capture_layerwise_state(
                model,
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                prefix["lengths"],
            )
            normed = torch.stack(state.normed_states).half().cpu()
            lengths = prefix_cpu["lengths"]
            for row, sample in enumerate(selected):
                length = int(lengths[row])
                user_id = int(sample["history"]["user_id"])
                record_id = by_user[user_id]
                records.append(
                    RealCacheRecord(
                        record_id=record_id,
                        user_id=user_id,
                        source_version=f"theta{source_version}",
                        normed=normed[:, row, :length].contiguous(),
                        item_ids=prefix_cpu["item_ids"][
                            row, :length
                        ].contiguous(),
                        behaviors=prefix_cpu["behaviors"][
                            row, :length
                        ].contiguous(),
                        time_deltas=prefix_cpu["time_deltas"][
                            row, :length
                        ].contiguous(),
                    )
                )
                cohort_tokens += length
            del prefix, state, normed
        torch.cuda.synchronize(device)
        per_cohort[f"theta{source_version}"] = {
            "records": len(cohort_entries),
            "logical_tokens": cohort_tokens,
            "seconds": time.perf_counter() - cohort_started,
        }
        del model
        torch.cuda.empty_cache()
    records.sort(key=lambda value: value.record_id)
    return records, {
        "records": len(records),
        "logical_tokens": sum(record.length for record in records),
        "capsule_bytes_unpadded": sum(
            record.capsule_nbytes for record in records
        ),
        "raw_history_bytes_unpadded": sum(
            record.raw_nbytes for record in records
        ),
        "per_cohort": per_cohort,
        "elapsed_seconds": time.perf_counter() - started,
    }


def pack_records(
    records: list[RealCacheRecord],
    batch_size: int,
    bucket_width: int,
    maximum_width: int,
) -> tuple[
    tuple[MigrationCapsuleBatch, ...],
    tuple[RawHistoryBatch, ...],
    dict,
]:
    if batch_size < 1 or bucket_width < 1:
        raise ValueError("batch size and bucket width must be positive")
    grouped: dict[tuple[str, int], list[RealCacheRecord]] = defaultdict(list)
    for record in records:
        bucket = math.ceil(record.length / bucket_width)
        grouped[(record.source_version, bucket)].append(record)
    capsules = []
    histories = []
    logical_tokens = 0
    allocated_tokens = 0
    for key in sorted(grouped):
        cohort = sorted(
            grouped[key],
            key=lambda value: (value.length, value.record_id),
        )
        for start in range(0, len(cohort), batch_size):
            selected = cohort[start : start + batch_size]
            width = min(
                maximum_width,
                math.ceil(max(record.length for record in selected) / bucket_width)
                * bucket_width,
            )
            layers = selected[0].normed.shape[0]
            hidden = selected[0].normed.shape[-1]
            shape = (layers, len(selected), width, hidden)
            normed = torch.zeros(
                shape,
                dtype=torch.float16,
                device="cpu",
                pin_memory=True,
            )
            item_ids = torch.zeros(
                len(selected),
                width,
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
            behaviors = torch.zeros_like(item_ids)
            time_deltas = torch.zeros(
                len(selected),
                width,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            lengths = torch.empty(
                len(selected),
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
            for row, record in enumerate(selected):
                length = record.length
                normed[:, row, :length].copy_(record.normed)
                item_ids[row, :length].copy_(record.item_ids)
                behaviors[row, :length].copy_(record.behaviors)
                time_deltas[row, :length].copy_(record.time_deltas)
                lengths[row] = length
                logical_tokens += length
            allocated_tokens += len(selected) * width
            record_ids = tuple(record.record_id for record in selected)
            anchor = selected[0].source_version
            capsules.append(
                MigrationCapsuleBatch(
                    record_ids=record_ids,
                    migration_anchor_version=anchor,
                    normed=normed,
                    lengths=lengths,
                )
            )
            histories.append(
                RawHistoryBatch(
                    record_ids=record_ids,
                    migration_anchor_version=anchor,
                    item_ids=item_ids,
                    behaviors=behaviors,
                    time_deltas=time_deltas,
                    lengths=lengths,
                )
            )
    cohort_batch_counts = {
        source: sum(
            batch.migration_anchor_version == source for batch in capsules
        )
        for source in sorted(
            {batch.migration_anchor_version for batch in capsules}
        )
    }
    return tuple(capsules), tuple(histories), {
        "records": len(records),
        "batches": len(capsules),
        "batch_size_limit": batch_size,
        "bucket_width": bucket_width,
        "logical_tokens": logical_tokens,
        "allocated_tokens": allocated_tokens,
        "padding_fraction": 1.0 - logical_tokens / max(allocated_tokens, 1),
        "capsule_bytes": sum(batch.nbytes for batch in capsules),
        "raw_history_bytes": sum(batch.nbytes for batch in histories),
        "cohort_batch_counts": cohort_batch_counts,
        "sequence_widths": [batch.seq_len for batch in capsules],
        "batch_sizes": [batch.batch_size for batch in capsules],
    }


def pack_raw_records(
    records: list[RealCacheRecord],
    batch_size: int,
    bucket_width: int,
    maximum_width: int,
) -> tuple[tuple[RawHistoryBatch, ...], dict]:
    if batch_size < 1 or bucket_width < 1:
        raise ValueError("batch size and bucket width must be positive")
    grouped: dict[int, list[RealCacheRecord]] = defaultdict(list)
    for record in records:
        grouped[math.ceil(record.length / bucket_width)].append(record)
    histories = []
    logical_tokens = 0
    allocated_tokens = 0
    for bucket in sorted(grouped):
        cohort = sorted(
            grouped[bucket],
            key=lambda value: (value.length, value.record_id),
        )
        for start in range(0, len(cohort), batch_size):
            selected = cohort[start : start + batch_size]
            width = min(
                maximum_width,
                math.ceil(max(record.length for record in selected) / bucket_width)
                * bucket_width,
            )
            item_ids = torch.zeros(
                len(selected),
                width,
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
            behaviors = torch.zeros_like(item_ids)
            time_deltas = torch.zeros(
                len(selected),
                width,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            lengths = torch.empty(
                len(selected),
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
            for row, record in enumerate(selected):
                length = record.length
                item_ids[row, :length].copy_(record.item_ids)
                behaviors[row, :length].copy_(record.behaviors)
                time_deltas[row, :length].copy_(record.time_deltas)
                lengths[row] = length
                logical_tokens += length
            allocated_tokens += len(selected) * width
            histories.append(
                RawHistoryBatch(
                    record_ids=tuple(
                        record.record_id for record in selected
                    ),
                    migration_anchor_version="raw_history",
                    item_ids=item_ids,
                    behaviors=behaviors,
                    time_deltas=time_deltas,
                    lengths=lengths,
                )
            )
    return tuple(histories), {
        "records": len(records),
        "batches": len(histories),
        "batch_size_limit": batch_size,
        "bucket_width": bucket_width,
        "logical_tokens": logical_tokens,
        "allocated_tokens": allocated_tokens,
        "padding_fraction": 1.0 - logical_tokens / max(allocated_tokens, 1),
        "raw_history_bytes": sum(batch.nbytes for batch in histories),
        "sequence_widths": [batch.seq_len for batch in histories],
        "batch_sizes": [batch.batch_size for batch in histories],
        "version_cohort_grouping": False,
    }


def latency_payload(samples) -> dict:
    return {
        "values_ms": list(samples.values_ms),
        "median_ms": samples.median_ms,
        "mean_ms": samples.mean_ms,
    }


def resident_operator_benchmark(
    programs: tuple[MigrationProgram, ...],
    capsule: MigrationCapsuleBatch,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    program = next(
        value
        for value in programs
        if value.source_version == capsule.migration_anchor_version
    )
    resident = capsule.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    operators = (
        ReferenceMigrationOperator(),
        PackedMigrationOperator(torch.float16),
        FusedMigrationOperator(),
    )
    outputs = {}
    latencies = {}
    for operator in operators:
        output, latency = benchmark_cuda_operator(
            operator,
            program,
            resident,
            warmup,
            repeats,
        )
        outputs[operator.name] = output
        latencies[operator.name] = latency
    reference_name = "reference_fp32"
    packed_name = "packed_float16"
    fused_name = next(
        name for name in outputs if name.startswith("fused_triton")
    )
    result = {
        "record_count": capsule.batch_size,
        "sequence_width": capsule.seq_len,
        "logical_tokens": int(capsule.lengths.sum()),
        "operators": {
            name: {
                "latency": latency_payload(latencies[name]),
                "records_per_second": (
                    capsule.batch_size * 1000.0
                    / latencies[name].median_ms
                ),
                "difference_from_fp32_reference": (
                    {
                        "max_abs": 0.0,
                        "rms": 0.0,
                        "fro_relative": 0.0,
                    }
                    if name == reference_name
                    else cache_difference(
                        outputs[name].cache,
                        outputs[reference_name].cache,
                    )
                ),
                "k_contiguous": outputs[name].cache.k.is_contiguous(),
                "v_contiguous": outputs[name].cache.v.is_contiguous(),
            }
            for name in outputs
        },
        "packed_speedup_over_reference": (
            latencies[reference_name].median_ms
            / latencies[packed_name].median_ms
        ),
        "fused_speedup_over_packed": (
            latencies[packed_name].median_ms
            / latencies[fused_name].median_ms
        ),
        "selected_fused_operator": fused_name,
    }
    del outputs
    torch.cuda.empty_cache()
    return result


def collect_migration_runtime(
    programs: tuple[MigrationProgram, ...],
    batches: tuple[MigrationCapsuleBatch, ...],
    devices: list[str],
    operator,
    output_pool: PinnedKVOutputPool | None,
    partition_strategy: str,
    max_inflight: int,
    warmup: int,
    repeats: int,
) -> dict:
    metrics = []
    with MultiGPUCohortExecutor(
        programs,
        devices,
        max_inflight_batches=max_inflight,
        pin_inputs=False,
        operator=operator,
        output_pool=output_pool,
        partition_strategy=partition_strategy,
    ) as executor:
        for iteration in range(warmup + repeats):
            report = executor.run(batches)
            expected_ids = tuple(
                record_id
                for batch in batches
                for record_id in batch.record_ids
            )
            actual_ids = tuple(
                record_id
                for batch in report.batches
                for record_id in batch.record_ids
            )
            if actual_ids != expected_ids:
                raise RuntimeError("migration runtime changed logical batch order")
            if iteration >= warmup:
                metrics.append(report.metrics)
            del report
    elapsed = [value.elapsed_seconds for value in metrics]
    median = statistics.median(elapsed)
    final = metrics[-1]
    return {
        "operator": operator.name,
        "devices": devices,
        "device_count": len(devices),
        "partition_strategy": partition_strategy,
        "persistent_output_pool": output_pool is not None,
        "max_inflight_batches": max_inflight,
        "elapsed_seconds": elapsed,
        "median_elapsed_seconds": median,
        "median_records_per_second": final.record_count / median,
        "median_tokens_per_second": final.token_count / median,
        "median_gib_per_second": (
            (final.input_bytes + final.output_bytes)
            / (2**30 * median)
        ),
        "metrics_last_repeat": asdict(final),
    }


def collect_recompute_runtime(
    models: list[HSTU],
    source_versions: list[str],
    target_version: str,
    batches: tuple[RawHistoryBatch, ...],
    devices: list[str],
    output_pool: PinnedKVOutputPool,
    execution_dtype: torch.dtype | None,
    max_inflight: int,
    warmup: int,
    repeats: int,
) -> dict:
    metrics = []
    with MultiGPUFullRecomputeExecutor(
        models[: len(devices)],
        source_version=source_versions,
        target_version=target_version,
        devices=devices,
        max_inflight_batches=max_inflight,
        pin_inputs=False,
        execution_dtype=execution_dtype,
        publication_dtype=torch.float16,
        output_pool=output_pool,
        partition_strategy="greedy_lpt",
    ) as executor:
        for iteration in range(warmup + repeats):
            report = executor.run(batches)
            expected_ids = tuple(
                record_id
                for batch in batches
                for record_id in batch.record_ids
            )
            actual_ids = tuple(
                record_id
                for batch in report.batches
                for record_id in batch.record_ids
            )
            if actual_ids != expected_ids:
                raise RuntimeError("recompute runtime changed logical batch order")
            if iteration >= warmup:
                metrics.append(report.metrics)
            del report
    elapsed = [value.elapsed_seconds for value in metrics]
    median = statistics.median(elapsed)
    final = metrics[-1]
    dtype_name = (
        "float32"
        if execution_dtype is None
        else str(execution_dtype).removeprefix("torch.")
    )
    return {
        "execution_dtype": dtype_name,
        "publication_dtype": "float16",
        "devices": devices,
        "device_count": len(devices),
        "partition_strategy": "greedy_lpt",
        "persistent_output_pool": True,
        "max_inflight_batches": max_inflight,
        "elapsed_seconds": elapsed,
        "median_elapsed_seconds": median,
        "median_records_per_second": final.record_count / median,
        "median_tokens_per_second": final.token_count / median,
        "median_gib_per_second": (
            (final.input_bytes + final.output_bytes)
            / (2**30 * median)
        ),
        "metrics_last_repeat": asdict(final),
    }


@torch.inference_mode()
def resident_recompute_precision(
    model: HSTU,
    batch: RawHistoryBatch,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    resident = batch.to(device, non_blocking=True)

    def execute(dtype: torch.dtype | None) -> HSTUKVCache:
        if dtype is None:
            cache = model.compute_kv(
                resident.item_ids,
                resident.behaviors,
                resident.time_deltas,
                lengths=resident.lengths,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                cache = model.compute_kv(
                    resident.item_ids,
                    resident.behaviors,
                    resident.time_deltas,
                    lengths=resident.lengths,
                )
        return HSTUKVCache(
            k=cache.k.half(),
            v=cache.v.half(),
            seq_len=cache.seq_len,
        )

    outputs = {}
    timings = {}
    for name, dtype in (
        ("float32", None),
        ("bfloat16", torch.bfloat16),
    ):
        output = None
        for _ in range(warmup):
            output = execute(dtype)
        torch.cuda.synchronize(device)
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = execute(dtype)
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        if output is None:
            raise RuntimeError("resident recompute produced no output")
        outputs[name] = output
        timings[name] = values
    result = {
        "records": batch.batch_size,
        "sequence_width": batch.seq_len,
        "float32": {
            "latency_ms": timings["float32"],
            "median_ms": statistics.median(timings["float32"]),
        },
        "bfloat16": {
            "latency_ms": timings["bfloat16"],
            "median_ms": statistics.median(timings["bfloat16"]),
            "difference_from_float32_published_fp16": cache_difference(
                outputs["bfloat16"],
                outputs["float32"],
            ),
            "finite": bool(
                torch.isfinite(outputs["bfloat16"].k).all()
                and torch.isfinite(outputs["bfloat16"].v).all()
            ),
        },
    }
    del outputs
    torch.cuda.empty_cache()
    return result


def system_topology(devices: list[str]) -> dict:
    resolved = [torch.device(device) for device in devices]
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": [
            {
                "logical_device": str(device),
                "name": torch.cuda.get_device_name(device),
                "total_memory_bytes": torch.cuda.get_device_properties(
                    device
                ).total_memory,
                "compute_capability": list(
                    torch.cuda.get_device_capability(device)
                ),
            }
            for device in resolved
        ],
        "peer_access": [
            [
                bool(torch.cuda.can_device_access_peer(left, right))
                if left != right
                else True
                for right in range(len(resolved))
            ]
            for left in range(len(resolved))
        ],
        "placement": (
            "cache extents are partitioned across GPUs; programs and the "
            "current model are replicated; no cross-GPU cache copy is required"
        ),
    }


def validate_args(args: argparse.Namespace) -> None:
    if len(args.devices) != 2:
        raise ValueError("this protocol requires exactly two CUDA devices")
    devices = [torch.device(device) for device in args.devices]
    if any(device.type != "cuda" for device in devices):
        raise ValueError("system benchmark requires CUDA devices")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA devices must be unique")
    if args.source_versions != [0, 4, 10]:
        raise ValueError("system protocol freezes source versions 0, 4, and 10")
    if args.target_version != 11 or args.base_days != 4:
        raise ValueError("system protocol freezes theta11 under 4+12")
    if args.max_users < 1 or args.layout_search_users < 1:
        raise ValueError("layout search and final user counts must be positive")
    if min(
        args.materialize_batch_size,
        args.batch_size,
        args.bucket_width,
        args.max_inflight,
        args.exact_batch_size,
        args.exact_bucket_width,
        args.exact_max_inflight,
        args.timing_repeats,
        args.operator_repeats,
    ) < 1:
        raise ValueError("batch, bucket, inflight, and repeat settings must be positive")
    if args.warmup_repeats < 0 or args.operator_warmup < 0:
        raise ValueError("warmup counts must be nonnegative")


def formal_configuration(args: argparse.Namespace) -> bool:
    return (
        args.prepared_data == DEFAULT_PREPARED
        and args.training_result == DEFAULT_TRAINING
        and args.checkpoint_dir == DEFAULT_CHECKPOINTS
        and args.program_dir == DEFAULT_PROGRAM_DIR
        and args.manifest_dir == DEFAULT_MANIFEST_DIR
        and args.output == DEFAULT_OUTPUT
        and args.devices == ["cuda:0", "cuda:1"]
        and args.source_weights == [0.2, 0.3, 0.5]
        and args.layout_search_users == 32
        and args.max_users == 64
        and args.materialize_batch_size == 2
        and args.batch_size == 1
        and args.bucket_width == 32
        and args.max_inflight == 3
        and args.exact_batch_size == 2
        and args.exact_bucket_width == 32
        and args.exact_max_inflight == 3
        and args.warmup_repeats == 1
        and args.timing_repeats == 3
        and args.operator_warmup == 5
        and args.operator_repeats == 20
        and args.seed == 0
    )


def smoke(args: argparse.Namespace) -> None:
    generator = torch.Generator().manual_seed(args.seed)
    program = MigrationProgram(
        source_version="theta0",
        target_version="theta1",
        adapter=CompiledCacheAdapter(
            weights=torch.randn(2, 16, 32, generator=generator),
            biases=torch.randn(2, 32, generator=generator),
            source_rank=16,
            ridge=1e-3,
        ),
    )
    capsule = MigrationCapsuleBatch(
        record_ids=(0, 1),
        migration_anchor_version="theta0",
        normed=torch.randn(
            2,
            2,
            8,
            16,
            generator=generator,
            dtype=torch.float16,
        ),
        lengths=torch.tensor([8, 5]),
    ).to(args.devices[0])
    packed, _ = benchmark_cuda_operator(
        PackedMigrationOperator(torch.float16),
        program,
        capsule,
        0,
        1,
    )
    fused, _ = benchmark_cuda_operator(
        FusedMigrationOperator(),
        program,
        capsule,
        0,
        1,
    )
    difference = cache_difference(fused.cache, packed.cache)
    if difference["fro_relative"] > 1e-3:
        raise RuntimeError("fused smoke difference exceeds tolerance")
    print(json.dumps({"status": "ok", "difference": difference}, indent=2))


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.smoke_test:
        smoke(args)
        return
    formal = formal_configuration(args)
    if not formal and not args.allow_diagnostic_protocol:
        raise ValueError(
            "nondefault settings require --allow-diagnostic-protocol"
        )
    seed_everything(args.seed)
    training = json.loads(Path(args.training_result).read_text())
    if training.get("protocol") != training_protocol_for_base_days(
        args.base_days
    ):
        raise ValueError("training protocol mismatch")
    if training.get("status") != "complete":
        raise ValueError("training result is incomplete")
    if int(training["args"]["seed"]) != args.seed:
        raise ValueError("training seed differs from system protocol")
    prepared_hash = sha256(args.prepared_data)
    if prepared_hash != training["prepared_data"]["sha256"]:
        raise ValueError("prepared data differs from training")
    cfg = HSTUConfig(**training["model"])
    plan, metadata = load_prepared_kuairand_plan(args.prepared_data)
    validate_long_context_plan(plan, metadata, args.base_days)
    eval_date, all_samples = reconstruct_online_eval_samples(
        plan,
        (args.target_version,),
        1000,
    )[args.target_version]
    verified_test, split = split_verified_test(all_samples, args.seed)
    selection_order = np.random.default_rng(
        43091 + args.seed
    ).permutation(len(verified_test))
    selected_samples = [
        verified_test[index]
        for index in selection_order[
            args.layout_search_users : (
                args.layout_search_users + args.max_users
            )
        ]
    ]
    if len(selected_samples) != args.max_users:
        raise ValueError("verified final split is too small for system roles")
    assignments, assignment_counts = fixed_count_assignment(
        len(selected_samples),
        args.source_versions,
        args.source_weights,
        58211 + args.seed,
    )
    entries = [
        (record_id, sample, source_version)
        for record_id, (sample, source_version) in enumerate(
            zip(selected_samples, assignments, strict=True)
        )
    ]
    programs, program_evidence = load_verified_programs(args, cfg)
    materialize_device = torch.device(args.devices[0])
    torch.cuda.set_device(materialize_device)
    records, materialization = materialize_records(
        entries,
        cfg,
        args.checkpoint_dir,
        materialize_device,
        args.materialize_batch_size,
    )
    capsules, _, migration_packing = pack_records(
        records,
        args.batch_size,
        args.bucket_width,
        cfg.max_seq_len - 1,
    )
    histories, exact_packing = pack_raw_records(
        records,
        args.exact_batch_size,
        args.exact_bucket_width,
        cfg.max_seq_len - 1,
    )
    del records
    gc.collect()
    representative_index = max(
        range(len(capsules)),
        key=lambda index: (
            capsules[index].batch_size * capsules[index].seq_len,
            capsules[index].seq_len,
        ),
    )
    resident_operator = resident_operator_benchmark(
        programs,
        capsules[representative_index],
        materialize_device,
        args.operator_warmup,
        args.operator_repeats,
    )
    ablations = []
    ablations.append(
        {
            "name": "packed_dynamic_1gpu",
            **collect_migration_runtime(
                programs,
                capsules,
                args.devices[:1],
                PackedMigrationOperator(torch.float16),
                None,
                "greedy_lpt",
                args.max_inflight,
                args.warmup_repeats,
                args.timing_repeats,
            ),
        }
    )
    ablations.append(
        {
            "name": "fused_dynamic_1gpu",
            **collect_migration_runtime(
                programs,
                capsules,
                args.devices[:1],
                FusedMigrationOperator(),
                None,
                "greedy_lpt",
                args.max_inflight,
                args.warmup_repeats,
                args.timing_repeats,
            ),
        }
    )
    destination_pool = PinnedKVOutputPool.allocate(
        capsules,
        served_kv_target=f"theta{args.target_version}",
        num_layers=cfg.num_layers,
        kv_width=cfg.num_heads * cfg.head_dim,
        dtype=torch.float16,
    )
    ablations.append(
        {
            "name": "fused_persistent_1gpu",
            **collect_migration_runtime(
                programs,
                capsules,
                args.devices[:1],
                FusedMigrationOperator(),
                destination_pool,
                "greedy_lpt",
                args.max_inflight,
                args.warmup_repeats,
                args.timing_repeats,
            ),
        }
    )
    ablations.append(
        {
            "name": "fused_persistent_2gpu_round_robin",
            **collect_migration_runtime(
                programs,
                capsules,
                args.devices,
                FusedMigrationOperator(),
                destination_pool,
                "round_robin",
                args.max_inflight,
                args.warmup_repeats,
                args.timing_repeats,
            ),
        }
    )
    ablations.append(
        {
            "name": "fused_persistent_2gpu_lpt",
            **collect_migration_runtime(
                programs,
                capsules,
                args.devices,
                FusedMigrationOperator(),
                destination_pool,
                "greedy_lpt",
                args.max_inflight,
                args.warmup_repeats,
                args.timing_repeats,
            ),
        }
    )
    method_destination_bytes = destination_pool.nbytes
    del destination_pool
    gc.collect()
    exact_destination_pool = PinnedKVOutputPool.allocate(
        histories,
        served_kv_target=f"theta{args.target_version}",
        num_layers=cfg.num_layers,
        kv_width=cfg.num_heads * cfg.head_dim,
        dtype=torch.float16,
    )
    current_models = [
        load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            args.target_version,
            torch.device(device),
        )
        for device in args.devices
    ]
    exact_representative_index = max(
        range(len(histories)),
        key=lambda index: (
            histories[index].batch_size
            * histories[index].seq_len
            * histories[index].seq_len,
            histories[index].seq_len,
        ),
    )
    resident_exact = resident_recompute_precision(
        current_models[0],
        histories[exact_representative_index],
        materialize_device,
        args.operator_warmup,
        args.operator_repeats,
    )
    exact_points = []
    for dtype in (torch.bfloat16, None):
        for device_count in (1, 2):
            exact_points.append(
                collect_recompute_runtime(
                    current_models,
                    ["raw_history"],
                    f"theta{args.target_version}",
                    histories,
                    args.devices[:device_count],
                    exact_destination_pool,
                    dtype,
                    args.exact_max_inflight,
                    args.warmup_repeats,
                    args.timing_repeats,
                )
            )
    points = {value["name"]: value for value in ablations}
    selected_method = points["fused_persistent_2gpu_lpt"]
    strongest_exact = min(
        (
            value
            for value in exact_points
            if value["device_count"] == 2
        ),
        key=lambda value: value["median_elapsed_seconds"],
    )
    bf16_one = next(
        value
        for value in exact_points
        if value["execution_dtype"] == "bfloat16"
        and value["device_count"] == 1
    )
    bf16_two = next(
        value
        for value in exact_points
        if value["execution_dtype"] == "bfloat16"
        and value["device_count"] == 2
    )
    result = {
        "protocol": TWO_GPU_SYSTEM_PROTOCOL,
        "status": (
            "adaptive_system_complete"
            if formal
            else "diagnostic_complete"
        ),
        "formal_protocol": formal,
        "study_stage": "adaptive_seed0_system_development",
        "source_training_result": args.training_result,
        "prepared_data": {
            "path": args.prepared_data,
            "sha256": prepared_hash,
        },
        "checkpoint_dir": args.checkpoint_dir,
        "eval_date": eval_date,
        "seed": args.seed,
        "quality_evidence_boundary": {
            "source": (
                "verified compiler manifests and their held-out "
                "recommendation evaluation"
            ),
            "system_benchmark_uses_labels": False,
            "system_users_change_algorithm_selection": False,
        },
        "split": split,
        "workload": {
            "kind": (
                "controlled mixed-version replay over held-out real "
                "KuaiRand histories"
            ),
            "selection_seed": 43091 + args.seed,
            "selection_offset": args.layout_search_users,
            "cohort_assignment_seed": 58211 + args.seed,
            "users": len(selected_samples),
            "source_versions": args.source_versions,
            "source_weights": args.source_weights,
            "source_counts": {
                f"theta{version}": count
                for version, count in assignment_counts.items()
            },
            "target_version": args.target_version,
            "admission_policy": (
                "none; every stale cohort receives its verified compiled repair"
            ),
        },
        "layout_selection": {
            "stage": "adaptive disjoint system-only search",
            "search_users": args.layout_search_users,
            "labels_used": False,
            "rule": (
                "maximize median records per second; retain only settings "
                "that improve throughput or padding"
            ),
            "migration": {
                "batch_size": args.batch_size,
                "bucket_width": args.bucket_width,
                "max_inflight": args.max_inflight,
                "version_cohort_grouping": True,
            },
            "full_recompute": {
                "batch_size": args.exact_batch_size,
                "bucket_width": args.exact_bucket_width,
                "max_inflight": args.exact_max_inflight,
                "version_cohort_grouping": False,
            },
        },
        "programs": program_evidence,
        "topology": system_topology(args.devices),
        "materialization": materialization,
        "packing": {
            "migration": migration_packing,
            "full_recompute": exact_packing,
        },
        "state_layout": {
            "cached_old_norm_capsule_bytes": migration_packing[
                "capsule_bytes"
            ],
            "logical_old_kv_bytes_at_fp16": (
                2
                * cfg.num_layers
                * migration_packing["logical_tokens"]
                * cfg.num_heads
                * cfg.head_dim
                * torch.empty((), dtype=torch.float16).element_size()
            ),
            "extra_capsule_ratio_to_logical_old_kv": (
                materialization["capsule_bytes_unpadded"]
                / (
                    2
                    * cfg.num_layers
                    * materialization["logical_tokens"]
                    * cfg.num_heads
                    * cfg.head_dim
                    * torch.empty((), dtype=torch.float16).element_size()
                )
            ),
            "migration_persistent_destination_bytes": (
                method_destination_bytes
            ),
            "full_recompute_persistent_destination_bytes": (
                exact_destination_pool.nbytes
            ),
            "raw_history_bytes": exact_packing["raw_history_bytes"],
            "program_replica_bytes_fp16": (
                sum(program.nbytes for program in programs)
                // 2
                * len(args.devices)
            ),
            "current_model_replica_bytes_fp32": sum(
                value.numel() * value.element_size()
                for model in current_models
                for value in (*model.parameters(), *model.buffers())
            ),
        },
        "resident_operator": resident_operator,
        "resident_full_recompute": resident_exact,
        "end_to_end_migration_ablations": ablations,
        "pipelined_full_recompute": exact_points,
        "derived": {
            "fused_dynamic_speedup_over_packed_dynamic_1gpu": (
                points["packed_dynamic_1gpu"]["median_elapsed_seconds"]
                / points["fused_dynamic_1gpu"]["median_elapsed_seconds"]
            ),
            "persistent_pool_speedup_1gpu": (
                points["fused_dynamic_1gpu"]["median_elapsed_seconds"]
                / points["fused_persistent_1gpu"]["median_elapsed_seconds"]
            ),
            "migration_1_to_2_gpu_speedup": (
                points["fused_persistent_1gpu"]["median_elapsed_seconds"]
                / selected_method["median_elapsed_seconds"]
            ),
            "migration_2gpu_parallel_efficiency": (
                points["fused_persistent_1gpu"]["median_elapsed_seconds"]
                / selected_method["median_elapsed_seconds"]
                / 2.0
            ),
            "lpt_speedup_over_round_robin_2gpu": (
                points["fused_persistent_2gpu_round_robin"][
                    "median_elapsed_seconds"
                ]
                / selected_method["median_elapsed_seconds"]
            ),
            "bf16_recompute_1_to_2_gpu_speedup": (
                bf16_one["median_elapsed_seconds"]
                / bf16_two["median_elapsed_seconds"]
            ),
            "selected_method_speedup_over_strongest_2gpu_full_recompute": (
                strongest_exact["median_elapsed_seconds"]
                / selected_method["median_elapsed_seconds"]
            ),
            "strongest_2gpu_full_recompute_dtype": strongest_exact[
                "execution_dtype"
            ],
        },
        "scope_boundary": {
            "included": (
                "real versioned capsules, verified programs, fused K/V "
                "operator, pinned-host movement, persistent publication "
                "extents, mixed-cohort batching, and two-GPU scheduling"
            ),
            "excluded": (
                "online checkpoint training, organic cache-age generation, "
                "capsule compression, SSD spill, and four-GPU scaling"
            ),
        },
    }
    save_json(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
