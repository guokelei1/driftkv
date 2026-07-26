from __future__ import annotations

import argparse
import gc
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from benchmark_kuairand_two_gpu_migration_system import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_PREPARED,
    DEFAULT_PROGRAM_DIR,
    DEFAULT_TRAINING,
    RealCacheRecord,
    fixed_count_assignment,
    load_verified_programs,
    materialize_records,
    pack_records,
    sha256,
    split_verified_test,
    system_topology,
)
from motivation_validity import seed_everything

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    FusedJaggedMigrationOperator,
    FusedMigrationOperator,
    JaggedMigratedKVBatch,
    JaggedMigrationCapsuleBatch,
    MigrationProgram,
    MultiGPUHBMJaggedCohortExecutor,
    MultiGPUJaggedCohortExecutor,
    PackedJaggedMigrationOperator,
    PinnedJaggedKVOutputPool,
    PinnedKVOutputPool,
)
from hstu_kvcache.streaming import (
    COHORT_JAGGED_SYSTEM_PROTOCOL,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

DEFAULT_OUTPUT = (
    "results/system/"
    "kuairand_long_context_4plus12_cohort_jagged_system_seed0.json"
)
DEFAULT_V2_RESULT = (
    "results/system/"
    "kuairand_long_context_4plus12_two_gpu_migration_system_seed0.json"
)


@dataclass(frozen=True)
class CachePageRecord:
    extent_id: int
    record_id: int
    source_version: str
    token_start: int
    normed: torch.Tensor

    @property
    def length(self) -> int:
        return self.normed.shape[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--program-dir", default=DEFAULT_PROGRAM_DIR)
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--v2-result", default=DEFAULT_V2_RESULT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--base-days", type=int, default=4)
    parser.add_argument("--source-versions", type=int, nargs="+", default=[0, 4, 10])
    parser.add_argument("--source-weights", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    parser.add_argument("--target-version", type=int, default=11)
    parser.add_argument("--search-users", type=int, default=32)
    parser.add_argument("--max-users", type=int, default=64)
    parser.add_argument("--materialize-batch-size", type=int, default=2)
    parser.add_argument(
        "--candidate-token-budgets",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384],
    )
    parser.add_argument(
        "--candidate-page-sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
    )
    parser.add_argument(
        "--candidate-page-tile-budgets",
        type=int,
        nargs="+",
        default=[2048, 4096],
    )
    parser.add_argument("--dense-batch-size", type=int, default=1)
    parser.add_argument("--dense-bucket-width", type=int, default=32)
    parser.add_argument("--max-inflight", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--operator-warmup", type=int, default=5)
    parser.add_argument("--operator-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-diagnostic-protocol", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.devices != ["cuda:0", "cuda:1"]:
        raise ValueError("formal cohort-jagged protocol requires cuda:0 and cuda:1")
    if torch.cuda.device_count() < 2:
        raise ValueError("two visible CUDA devices are required")
    if args.source_versions != [0, 4, 10]:
        raise ValueError("protocol freezes source versions 0, 4, and 10")
    if args.target_version != 11 or args.base_days != 4:
        raise ValueError("protocol freezes theta11 under 4+12")
    if min(
        args.search_users,
        args.max_users,
        args.materialize_batch_size,
        args.dense_batch_size,
        args.dense_bucket_width,
        args.max_inflight,
        args.timing_repeats,
        args.operator_repeats,
        *args.candidate_token_budgets,
        *args.candidate_page_sizes,
        *args.candidate_page_tile_budgets,
    ) < 1:
        raise ValueError("counts and token budgets must be positive")
    if args.warmup_repeats < 0 or args.operator_warmup < 0:
        raise ValueError("warmup counts must be nonnegative")


def formal_configuration(args: argparse.Namespace) -> bool:
    return (
        args.prepared_data == DEFAULT_PREPARED
        and args.training_result == DEFAULT_TRAINING
        and args.checkpoint_dir == DEFAULT_CHECKPOINTS
        and args.program_dir == DEFAULT_PROGRAM_DIR
        and args.manifest_dir == DEFAULT_MANIFEST_DIR
        and args.v2_result == DEFAULT_V2_RESULT
        and args.output == DEFAULT_OUTPUT
        and args.devices == ["cuda:0", "cuda:1"]
        and args.source_weights == [0.2, 0.3, 0.5]
        and args.search_users == 32
        and args.max_users == 64
        and args.materialize_batch_size == 2
        and args.candidate_token_budgets == [2048, 4096, 8192, 16384]
        and args.candidate_page_sizes == [128, 256, 512, 1024]
        and args.candidate_page_tile_budgets == [2048, 4096]
        and args.dense_batch_size == 1
        and args.dense_bucket_width == 32
        and args.max_inflight == 3
        and args.warmup_repeats == 1
        and args.timing_repeats == 3
        and args.operator_warmup == 5
        and args.operator_repeats == 20
        and args.seed == 0
    )


def record_chunks(
    records: list[RealCacheRecord],
    max_tokens: int,
) -> list[list[RealCacheRecord]]:
    grouped: dict[str, list[RealCacheRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_version].append(record)
    chunks = []
    for source in sorted(grouped):
        current = []
        current_tokens = 0
        for record in sorted(
            grouped[source],
            key=lambda value: value.record_id,
        ):
            if current and current_tokens + record.length > max_tokens:
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(record)
            current_tokens += record.length
        if current:
            chunks.append(current)
    return chunks


def pack_jagged_records(
    records: list[RealCacheRecord],
    max_tokens: int,
) -> tuple[tuple[JaggedMigrationCapsuleBatch, ...], dict]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    batches = []
    for selected in record_chunks(records, max_tokens):
        layers = selected[0].normed.shape[0]
        hidden = selected[0].normed.shape[-1]
        token_count = sum(record.length for record in selected)
        normed = torch.empty(
            layers,
            token_count,
            hidden,
            dtype=torch.float16,
            device="cpu",
            pin_memory=True,
        )
        lengths = torch.empty(
            len(selected),
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )
        offsets = torch.empty(
            len(selected) + 1,
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )
        offsets[0] = 0
        cursor = 0
        for index, record in enumerate(selected):
            end = cursor + record.length
            normed[:, cursor:end].copy_(record.normed)
            lengths[index] = record.length
            offsets[index + 1] = end
            cursor = end
        batches.append(
            JaggedMigrationCapsuleBatch(
                record_ids=tuple(record.record_id for record in selected),
                migration_anchor_version=selected[0].source_version,
                normed=normed,
                lengths=lengths,
                offsets=offsets,
            )
        )
    sources = sorted({batch.migration_anchor_version for batch in batches})
    return tuple(batches), {
        "records": len(records),
        "batches": len(batches),
        "maximum_tokens_per_batch": max_tokens,
        "logical_tokens": sum(batch.token_count for batch in batches),
        "allocated_tokens": sum(batch.token_count for batch in batches),
        "padding_fraction": 0.0,
        "capsule_bytes": sum(batch.nbytes for batch in batches),
        "cohort_batch_counts": {
            source: sum(
                batch.migration_anchor_version == source for batch in batches
            )
            for source in sources
        },
        "cohort_record_counts": {
            source: sum(
                batch.batch_size
                for batch in batches
                if batch.migration_anchor_version == source
            )
            for source in sources
        },
        "token_counts": [batch.token_count for batch in batches],
        "batch_sizes": [batch.batch_size for batch in batches],
        "layout": "layer-major cohort-jagged valid-token stream",
    }


def make_cache_pages(
    records: list[RealCacheRecord],
    page_size: int,
) -> list[CachePageRecord]:
    if page_size < 1:
        raise ValueError("page_size must be positive")
    pages = []
    for record in sorted(records, key=lambda value: value.record_id):
        for start in range(0, record.length, page_size):
            pages.append(
                CachePageRecord(
                    extent_id=(record.record_id << 16) + start,
                    record_id=record.record_id,
                    source_version=record.source_version,
                    token_start=start,
                    normed=record.normed[:, start : start + page_size],
                )
            )
    return pages


def pack_paged_jagged_records(
    records: list[RealCacheRecord],
    page_size: int,
    tile_tokens: int,
) -> tuple[tuple[JaggedMigrationCapsuleBatch, ...], list[dict], dict]:
    if page_size < 1 or tile_tokens < 1:
        raise ValueError("page size and tile token budget must be positive")
    if page_size > tile_tokens:
        raise ValueError("page size cannot exceed tile token budget")
    grouped: dict[str, list[CachePageRecord]] = defaultdict(list)
    for page in make_cache_pages(records, page_size):
        grouped[page.source_version].append(page)
    selected_bins: list[tuple[str, list[CachePageRecord]]] = []
    for source in sorted(grouped):
        bins: list[list[CachePageRecord]] = []
        loads: list[int] = []
        pages = sorted(
            grouped[source],
            key=lambda value: (
                -value.length,
                value.record_id,
                value.token_start,
            ),
        )
        for page in pages:
            candidates = [
                index
                for index, load in enumerate(loads)
                if load + page.length <= tile_tokens
            ]
            if candidates:
                selected = min(
                    candidates,
                    key=lambda index: (
                        tile_tokens - loads[index] - page.length,
                        index,
                    ),
                )
                bins[selected].append(page)
                loads[selected] += page.length
            else:
                bins.append([page])
                loads.append(page.length)
        selected_bins.extend(
            (source, pages_in_bin)
            for pages_in_bin in bins
        )
    batches = []
    page_table = []
    for source, selected in selected_bins:
        selected = sorted(
            selected,
            key=lambda value: (
                value.record_id,
                value.token_start,
            ),
        )
        layers = selected[0].normed.shape[0]
        hidden = selected[0].normed.shape[-1]
        token_count = sum(page.length for page in selected)
        normed = torch.empty(
            layers,
            token_count,
            hidden,
            dtype=torch.float16,
            device="cpu",
            pin_memory=True,
        )
        lengths = torch.empty(
            len(selected),
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )
        offsets = torch.empty(
            len(selected) + 1,
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )
        offsets[0] = 0
        cursor = 0
        for index, page in enumerate(selected):
            end = cursor + page.length
            normed[:, cursor:end].copy_(page.normed)
            lengths[index] = page.length
            offsets[index + 1] = end
            page_table.append(
                {
                    "extent_id": page.extent_id,
                    "record_id": page.record_id,
                    "source_version": page.source_version,
                    "token_start": page.token_start,
                    "token_count": page.length,
                }
            )
            cursor = end
        batches.append(
            JaggedMigrationCapsuleBatch(
                record_ids=tuple(page.extent_id for page in selected),
                migration_anchor_version=source,
                normed=normed,
                lengths=lengths,
                offsets=offsets,
            )
        )
    sources = sorted({batch.migration_anchor_version for batch in batches})
    logical_tokens = sum(batch.token_count for batch in batches)
    return tuple(batches), page_table, {
        "records": len(records),
        "pages": len(page_table),
        "batches": len(batches),
        "page_size": page_size,
        "maximum_tokens_per_tile": tile_tokens,
        "logical_tokens": logical_tokens,
        "allocated_tokens": logical_tokens,
        "padding_fraction": 0.0,
        "capsule_bytes": sum(batch.nbytes for batch in batches),
        "cohort_batch_counts": {
            source: sum(
                batch.migration_anchor_version == source for batch in batches
            )
            for source in sources
        },
        "token_counts": [batch.token_count for batch in batches],
        "pages_per_batch": [batch.batch_size for batch in batches],
        "minimum_tile_fill": min(
            batch.token_count / tile_tokens for batch in batches
        ),
        "mean_tile_fill": statistics.mean(
            batch.token_count / tile_tokens for batch in batches
        ),
        "layout": (
            "fixed-size serving pages compacted by version cohort into "
            "bounded valid-token tiles"
        ),
    }


def collect_jagged_runtime(
    programs: tuple[MigrationProgram, ...],
    batches: tuple[JaggedMigrationCapsuleBatch, ...],
    devices: list[str],
    operator,
    output_pool: PinnedJaggedKVOutputPool,
    max_inflight: int,
    warmup: int,
    repeats: int,
) -> dict:
    metrics = []
    with MultiGPUJaggedCohortExecutor(
        programs,
        devices,
        max_inflight_batches=max_inflight,
        pin_inputs=False,
        operator=operator,
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
                raise RuntimeError("jagged runtime changed logical batch order")
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
        "partition_strategy": "greedy_lpt",
        "persistent_output_pool": True,
        "max_inflight_batches": max_inflight,
        "elapsed_seconds": elapsed,
        "median_elapsed_seconds": median,
        "median_records_per_second": final.record_count / median,
        "median_tokens_per_second": final.token_count / median,
        "median_gib_per_second": (
            (final.input_bytes + final.output_bytes) / (2**30 * median)
        ),
        "metrics_last_repeat": asdict(final),
    }


def collect_hbm_runtime(
    programs: tuple[MigrationProgram, ...],
    batches: tuple[JaggedMigrationCapsuleBatch, ...],
    devices: list[str],
    operator,
    max_inflight: int,
    warmup: int,
    repeats: int,
) -> dict:
    metrics = []
    executor = MultiGPUHBMJaggedCohortExecutor(
        programs,
        devices,
        batches,
        max_inflight_batches=max_inflight,
        pin_inputs=False,
        operator=operator,
        partition_strategy="greedy_lpt",
    )
    output_pool_nbytes = executor.output_pool_nbytes
    try:
        for iteration in range(warmup + repeats):
            report = executor.run()
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
                raise RuntimeError("HBM runtime changed logical batch order")
            if iteration >= warmup:
                metrics.append(report.metrics)
            del report
    finally:
        executor.close()
    del executor
    for device in devices:
        with torch.cuda.device(torch.device(device)):
            torch.cuda.empty_cache()
    elapsed = [value.elapsed_seconds for value in metrics]
    median = statistics.median(elapsed)
    final = metrics[-1]
    return {
        "operator": operator.name,
        "devices": devices,
        "device_count": len(devices),
        "partition_strategy": "greedy_lpt",
        "publication_boundary": "preallocated serving-native HBM jagged K/V",
        "persistent_output_pool": True,
        "persistent_output_pool_bytes": output_pool_nbytes,
        "max_inflight_batches": max_inflight,
        "elapsed_seconds": elapsed,
        "median_elapsed_seconds": median,
        "median_records_per_second": final.record_count / median,
        "median_tokens_per_second": final.token_count / median,
        "median_h2d_gib_per_second": final.input_bytes / (2**30 * median),
        "metrics_last_repeat": asdict(final),
    }


def difference_payload(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float | int]:
    squared_error = 0.0
    squared_reference = 0.0
    max_abs = 0.0
    mismatched = 0
    elements = 0
    for actual, expected in pairs:
        if actual.shape != expected.shape:
            raise ValueError("comparison tensor shapes differ")
        for layer in range(actual.shape[0]):
            actual_layer = actual[layer]
            expected_layer = expected[layer]
            delta = actual_layer.float() - expected_layer.float()
            squared_error += float(delta.square().sum())
            squared_reference += float(expected_layer.float().square().sum())
            max_abs = max(max_abs, float(delta.abs().max()))
            mismatched += int(torch.count_nonzero(actual_layer != expected_layer))
            elements += actual_layer.numel()
    return {
        "max_abs": max_abs,
        "rms": (squared_error / max(elements, 1)) ** 0.5,
        "fro_relative": (
            squared_error**0.5 / max(squared_reference**0.5, 1e-12)
        ),
        "mismatched_fp16_elements": mismatched,
        "elements": elements,
    }


def resident_jagged_benchmark(
    programs: tuple[MigrationProgram, ...],
    capsule: JaggedMigrationCapsuleBatch,
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
        PackedJaggedMigrationOperator(torch.float16),
        FusedJaggedMigrationOperator(),
    )
    outputs: dict[str, JaggedMigratedKVBatch] = {}
    timings = {}
    for operator in operators:
        prepared = operator.prepare_program(program, device)
        output = None
        for _ in range(warmup):
            output = operator.execute(prepared, resident)
        torch.cuda.synchronize(device)
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = operator.execute(prepared, resident)
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        if output is None:
            raise RuntimeError("resident jagged operator produced no output")
        outputs[operator.name] = output
        timings[operator.name] = values
    packed_name = operators[0].name
    fused_name = operators[1].name
    difference = difference_payload(
        [
            (outputs[fused_name].k, outputs[packed_name].k),
            (outputs[fused_name].v, outputs[packed_name].v),
        ]
    )
    return {
        "records": capsule.batch_size,
        "logical_tokens": capsule.token_count,
        "operators": {
            name: {
                "latency_ms": values,
                "median_ms": statistics.median(values),
                "valid_tokens_per_second": (
                    capsule.token_count * 1000.0
                    / statistics.median(values)
                ),
                "k_contiguous": outputs[name].k.is_contiguous(),
                "v_contiguous": outputs[name].v.is_contiguous(),
            }
            for name, values in timings.items()
        },
        "fused_difference_from_packed_fp16": difference,
        "fused_speedup_over_packed_fp16": (
            statistics.median(timings[packed_name])
            / statistics.median(timings[fused_name])
        ),
    }


def validate_dense_jagged_outputs(
    dense_pool: PinnedKVOutputPool,
    jagged_pool: PinnedJaggedKVOutputPool,
) -> dict:
    dense = {}
    for output in dense_pool.outputs.values():
        for row, record_id in enumerate(output.record_ids):
            length = int(output.lengths[row])
            dense[record_id] = (
                output.cache.k[:, row, :length],
                output.cache.v[:, row, :length],
            )
    jagged = {
        record_id: output.record_kv(record_id)
        for output in jagged_pool.outputs.values()
        for record_id in output.record_ids
    }
    if dense.keys() != jagged.keys():
        raise RuntimeError("dense and jagged outputs contain different records")
    pairs = []
    for record_id in sorted(dense):
        pairs.extend(
            (
                (jagged[record_id][0], dense[record_id][0]),
                (jagged[record_id][1], dense[record_id][1]),
            )
        )
    return {
        "record_count": len(dense),
        "valid_token_only": True,
        "difference_from_dense_fused_fp16": difference_payload(pairs),
    }


def validate_dense_paged_outputs(
    dense_pool: PinnedKVOutputPool,
    paged_pool: PinnedJaggedKVOutputPool,
    page_table: list[dict],
) -> dict:
    dense = {}
    for output in dense_pool.outputs.values():
        for row, record_id in enumerate(output.record_ids):
            length = int(output.lengths[row])
            dense[record_id] = (
                output.cache.k[:, row, :length],
                output.cache.v[:, row, :length],
            )
    paged = {
        extent_id: output.record_kv(extent_id)
        for output in paged_pool.outputs.values()
        for extent_id in output.record_ids
    }
    descriptors = {value["extent_id"]: value for value in page_table}
    if paged.keys() != descriptors.keys():
        raise RuntimeError("paged output and page table extents differ")
    coverage = defaultdict(int)
    pairs = []
    for extent_id in sorted(descriptors):
        descriptor = descriptors[extent_id]
        record_id = descriptor["record_id"]
        start = descriptor["token_start"]
        end = start + descriptor["token_count"]
        actual_k, actual_v = paged[extent_id]
        expected_k, expected_v = dense[record_id]
        pairs.extend(
            (
                (actual_k, expected_k[:, start:end]),
                (actual_v, expected_v[:, start:end]),
            )
        )
        coverage[record_id] += descriptor["token_count"]
    expected_coverage = {
        record_id: values[0].shape[1]
        for record_id, values in dense.items()
    }
    if dict(coverage) != expected_coverage:
        raise RuntimeError("paged publication does not cover every valid token")
    return {
        "record_count": len(dense),
        "page_count": len(page_table),
        "complete_record_coverage": True,
        "difference_from_dense_fused_fp16": difference_payload(pairs),
    }


def summarize_record_group(records: list[RealCacheRecord]) -> dict:
    sources = sorted({record.source_version for record in records})
    return {
        "records": len(records),
        "logical_tokens": sum(record.length for record in records),
        "capsule_bytes_unpadded": sum(record.capsule_nbytes for record in records),
        "source_counts": {
            source: sum(record.source_version == source for record in records)
            for source in sources
        },
        "source_tokens": {
            source: sum(
                record.length
                for record in records
                if record.source_version == source
            )
            for source in sources
        },
    }


def v2_reference(path: str) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("protocol") != (
        "kuairand_long_context_4plus12_two_gpu_migration_system_v2"
    ):
        raise ValueError("v2 reference protocol mismatch")
    exact = min(
        (
            value
            for value in payload["pipelined_full_recompute"]
            if value["device_count"] == 2
        ),
        key=lambda value: value["median_elapsed_seconds"],
    )
    return {
        "path": path,
        "sha256": sha256(path),
        "users": payload["workload"]["users"],
        "source_counts": payload["workload"]["source_counts"],
        "logical_tokens": payload["packing"]["migration"]["logical_tokens"],
        "strongest_two_gpu_exact_dtype": exact["execution_dtype"],
        "strongest_two_gpu_exact_seconds": exact["median_elapsed_seconds"],
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    formal = formal_configuration(args)
    if not formal and not args.allow_diagnostic_protocol:
        raise ValueError("nondefault settings require --allow-diagnostic-protocol")
    seed_everything(args.seed)
    training = json.loads(Path(args.training_result).read_text())
    if training.get("protocol") != training_protocol_for_base_days(args.base_days):
        raise ValueError("training protocol mismatch")
    if training.get("status") != "complete":
        raise ValueError("training result is incomplete")
    if int(training["args"]["seed"]) != args.seed:
        raise ValueError("training seed differs from system protocol")
    prepared_hash = sha256(args.prepared_data)
    if prepared_hash != training["prepared_data"]["sha256"]:
        raise ValueError("prepared data differs from training")
    cfg = training["model"]
    from hstu_kvcache.models import HSTUConfig

    model_config = HSTUConfig(**cfg)
    plan, metadata = load_prepared_kuairand_plan(args.prepared_data)
    validate_long_context_plan(plan, metadata, args.base_days)
    eval_date, all_samples = reconstruct_online_eval_samples(
        plan,
        (args.target_version,),
        1000,
    )[args.target_version]
    verified_test, split = split_verified_test(all_samples, args.seed)
    ordering = np.random.default_rng(43091 + args.seed).permutation(
        len(verified_test)
    )
    search_samples = [
        verified_test[index] for index in ordering[: args.search_users]
    ]
    final_samples = [
        verified_test[index]
        for index in ordering[
            args.search_users : args.search_users + args.max_users
        ]
    ]
    if len(search_samples) != args.search_users or len(final_samples) != args.max_users:
        raise ValueError("verified split is too small for disjoint system roles")
    search_assignments, search_counts = fixed_count_assignment(
        len(search_samples),
        args.source_versions,
        args.source_weights,
        58210 + args.seed,
    )
    final_assignments, final_counts = fixed_count_assignment(
        len(final_samples),
        args.source_versions,
        args.source_weights,
        58211 + args.seed,
    )
    entries = [
        (record_id, sample, source)
        for record_id, (sample, source) in enumerate(
            zip(search_samples, search_assignments, strict=True)
        )
    ]
    entries.extend(
        (
            args.search_users + record_id,
            sample,
            source,
        )
        for record_id, (sample, source) in enumerate(
            zip(final_samples, final_assignments, strict=True)
        )
    )
    programs, program_evidence = load_verified_programs(args, model_config)
    materialize_device = torch.device(args.devices[0])
    torch.cuda.set_device(materialize_device)
    records, materialization = materialize_records(
        entries,
        model_config,
        args.checkpoint_dir,
        materialize_device,
        args.materialize_batch_size,
    )
    search_records = [
        record for record in records if record.record_id < args.search_users
    ]
    final_records = [
        record for record in records if record.record_id >= args.search_users
    ]
    search_summary = summarize_record_group(search_records)
    final_summary = summarize_record_group(final_records)
    search_results = []
    for token_budget in args.candidate_token_budgets:
        search_batches, search_packing = pack_jagged_records(
            search_records,
            token_budget,
        )
        search_pool = PinnedJaggedKVOutputPool.allocate(
            search_batches,
            served_kv_target=f"theta{args.target_version}",
            num_layers=model_config.num_layers,
            kv_width=model_config.num_heads * model_config.head_dim,
            dtype=torch.float16,
        )
        host_runtime = collect_jagged_runtime(
            programs,
            search_batches,
            args.devices,
            FusedJaggedMigrationOperator(),
            search_pool,
            args.max_inflight,
            args.warmup_repeats,
            args.timing_repeats,
        )
        hbm_runtime = collect_hbm_runtime(
            programs,
            search_batches,
            args.devices,
            FusedJaggedMigrationOperator(),
            args.max_inflight,
            args.warmup_repeats,
            args.timing_repeats,
        )
        search_results.append(
            {
                "token_budget": token_budget,
                "packing": search_packing,
                "host_backed_runtime": host_runtime,
                "hbm_direct_runtime": hbm_runtime,
            }
        )
        del search_batches, search_pool
        gc.collect()
    page_search_results = []
    for page_size in args.candidate_page_sizes:
        for tile_tokens in args.candidate_page_tile_budgets:
            page_batches, _, page_packing = pack_paged_jagged_records(
                search_records,
                page_size,
                tile_tokens,
            )
            hbm_runtime = collect_hbm_runtime(
                programs,
                page_batches,
                args.devices,
                FusedJaggedMigrationOperator(),
                args.max_inflight,
                args.warmup_repeats,
                args.timing_repeats,
            )
            page_search_results.append(
                {
                    "page_size": page_size,
                    "tile_tokens": tile_tokens,
                    "packing": page_packing,
                    "hbm_direct_runtime": hbm_runtime,
                }
            )
            del page_batches
            gc.collect()
    selected_host_search = max(
        search_results,
        key=lambda value: (
            value["host_backed_runtime"]["median_tokens_per_second"],
            -value["token_budget"],
        ),
    )
    selected_hbm_search = max(
        search_results,
        key=lambda value: (
            value["hbm_direct_runtime"]["median_tokens_per_second"],
            -value["token_budget"],
        ),
    )
    selected_host_token_budget = selected_host_search["token_budget"]
    selected_hbm_token_budget = selected_hbm_search["token_budget"]
    selected_page_search = max(
        page_search_results,
        key=lambda value: (
            value["hbm_direct_runtime"]["median_tokens_per_second"],
            -value["packing"]["pages"],
        ),
    )
    selected_page_size = selected_page_search["page_size"]
    selected_page_tile_tokens = selected_page_search["tile_tokens"]
    dense_capsules, dense_histories, dense_packing = pack_records(
        final_records,
        args.dense_batch_size,
        args.dense_bucket_width,
        model_config.max_seq_len - 1,
    )
    del dense_histories
    jagged_capsules, jagged_packing = pack_jagged_records(
        final_records,
        selected_host_token_budget,
    )
    single_record_capsules, single_record_packing = pack_jagged_records(
        final_records,
        1,
    )
    hbm_jagged_capsules, hbm_jagged_packing = pack_jagged_records(
        final_records,
        selected_hbm_token_budget,
    )
    paged_capsules, page_table, paged_packing = pack_paged_jagged_records(
        final_records,
        selected_page_size,
        selected_page_tile_tokens,
    )
    del records, search_records, final_records
    gc.collect()
    dense_pool = PinnedKVOutputPool.allocate(
        dense_capsules,
        served_kv_target=f"theta{args.target_version}",
        num_layers=model_config.num_layers,
        kv_width=model_config.num_heads * model_config.head_dim,
        dtype=torch.float16,
    )
    jagged_pool = PinnedJaggedKVOutputPool.allocate(
        jagged_capsules,
        served_kv_target=f"theta{args.target_version}",
        num_layers=model_config.num_layers,
        kv_width=model_config.num_heads * model_config.head_dim,
        dtype=torch.float16,
    )
    paged_pool = PinnedJaggedKVOutputPool.allocate(
        paged_capsules,
        served_kv_target=f"theta{args.target_version}",
        num_layers=model_config.num_layers,
        kv_width=model_config.num_heads * model_config.head_dim,
        dtype=torch.float16,
    )
    from benchmark_kuairand_two_gpu_migration_system import (
        collect_migration_runtime,
    )

    ablations = []
    for device_count in (1, 2):
        ablations.append(
            {
                "name": f"dense_fused_persistent_{device_count}gpu",
                "layout": "dense_length_bucketed",
                **collect_migration_runtime(
                    programs,
                    dense_capsules,
                    args.devices[:device_count],
                    FusedMigrationOperator(),
                    dense_pool,
                    "greedy_lpt",
                    args.max_inflight,
                    args.warmup_repeats,
                    args.timing_repeats,
                ),
            }
        )
    for operator_name, operator in (
        ("jagged_packed", PackedJaggedMigrationOperator(torch.float16)),
        ("jagged_fused", FusedJaggedMigrationOperator()),
    ):
        for device_count in (1, 2):
            ablations.append(
                {
                    "name": f"{operator_name}_persistent_{device_count}gpu",
                    "layout": "cohort_jagged_direct_publish",
                    **collect_jagged_runtime(
                        programs,
                        jagged_capsules,
                        args.devices[:device_count],
                        operator,
                        jagged_pool,
                        args.max_inflight,
                        args.warmup_repeats,
                        args.timing_repeats,
                    ),
                }
            )
    for operator_name, operator in (
        ("paged_packed", PackedJaggedMigrationOperator(torch.float16)),
        ("paged_fused", FusedJaggedMigrationOperator()),
    ):
        for device_count in (1, 2):
            ablations.append(
                {
                    "name": f"{operator_name}_persistent_{device_count}gpu",
                    "layout": "cohort-page compacted direct host publish",
                    **collect_jagged_runtime(
                        programs,
                        paged_capsules,
                        args.devices[:device_count],
                        operator,
                        paged_pool,
                        args.max_inflight,
                        args.warmup_repeats,
                        args.timing_repeats,
                    ),
                }
            )
    hbm_ablations = []
    for device_count in (1, 2):
        hbm_ablations.append(
            {
                "name": f"single_record_fused_hbm_{device_count}gpu",
                "layout": "one-record jagged direct HBM publish",
                **collect_hbm_runtime(
                    programs,
                    single_record_capsules,
                    args.devices[:device_count],
                    FusedJaggedMigrationOperator(),
                    args.max_inflight,
                    args.warmup_repeats,
                    args.timing_repeats,
                ),
            }
        )
    for operator_name, operator in (
        ("cohort_jagged_packed", PackedJaggedMigrationOperator(torch.float16)),
        ("cohort_jagged_fused", FusedJaggedMigrationOperator()),
    ):
        for device_count in (1, 2):
            hbm_ablations.append(
                {
                    "name": f"{operator_name}_hbm_{device_count}gpu",
                    "layout": "cohort-jagged direct HBM publish",
                    **collect_hbm_runtime(
                        programs,
                        hbm_jagged_capsules,
                        args.devices[:device_count],
                        operator,
                        args.max_inflight,
                        args.warmup_repeats,
                        args.timing_repeats,
                    ),
                }
            )
    for operator_name, operator in (
        ("cohort_paged_packed", PackedJaggedMigrationOperator(torch.float16)),
        ("cohort_paged_fused", FusedJaggedMigrationOperator()),
    ):
        for device_count in (1, 2):
            hbm_ablations.append(
                {
                    "name": f"{operator_name}_hbm_{device_count}gpu",
                    "layout": "cohort-page compacted direct HBM publish",
                    **collect_hbm_runtime(
                        programs,
                        paged_capsules,
                        args.devices[:device_count],
                        operator,
                        args.max_inflight,
                        args.warmup_repeats,
                        args.timing_repeats,
                    ),
                }
            )
    correctness = validate_dense_jagged_outputs(dense_pool, jagged_pool)
    paged_correctness = validate_dense_paged_outputs(
        dense_pool,
        paged_pool,
        page_table,
    )
    representative = max(
        paged_capsules,
        key=lambda value: (value.token_count, value.batch_size),
    )
    resident = resident_jagged_benchmark(
        programs,
        representative,
        materialize_device,
        args.operator_warmup,
        args.operator_repeats,
    )
    points = {value["name"]: value for value in ablations}
    host_selected = points["jagged_fused_persistent_2gpu"]
    hbm_points = {value["name"]: value for value in hbm_ablations}
    hbm_selected = hbm_points["cohort_paged_fused_hbm_2gpu"]
    exact_reference = v2_reference(args.v2_result)
    if exact_reference["users"] != args.max_users:
        raise ValueError("v2 exact reference user count differs")
    if exact_reference["source_counts"] != {
        f"theta{version}": count for version, count in final_counts.items()
    }:
        raise ValueError("v2 exact reference cohort mix differs")
    if exact_reference["logical_tokens"] != final_summary["logical_tokens"]:
        raise ValueError("v2 exact reference trace differs")
    result = {
        "protocol": COHORT_JAGGED_SYSTEM_PROTOCOL,
        "status": (
            "adaptive_system_complete" if formal else "diagnostic_complete"
        ),
        "formal_protocol": formal,
        "study_stage": "adaptive_seed0_operator_development",
        "source_training_result": args.training_result,
        "prepared_data": {
            "path": args.prepared_data,
            "sha256": prepared_hash,
        },
        "checkpoint_dir": args.checkpoint_dir,
        "eval_date": eval_date,
        "seed": args.seed,
        "quality_evidence_boundary": {
            "source": "unchanged verified compiler manifests",
            "system_benchmark_uses_labels": False,
            "operator_search_uses_labels": False,
            "system_users_change_algorithm_selection": False,
        },
        "split": split,
        "workload": {
            "kind": "controlled mixed-version replay over real held-out histories",
            "ordering_seed": 43091 + args.seed,
            "search_users": args.search_users,
            "final_users": args.max_users,
            "search_assignment_seed": 58210 + args.seed,
            "final_assignment_seed": 58211 + args.seed,
            "search_source_counts": {
                f"theta{version}": count
                for version, count in search_counts.items()
            },
            "final_source_counts": {
                f"theta{version}": count
                for version, count in final_counts.items()
            },
            "target_version": args.target_version,
            "admission_policy": (
                "none; every stale cohort receives its verified compiled repair"
            ),
        },
        "programs": program_evidence,
        "topology": system_topology(args.devices),
        "materialization": materialization,
        "search_trace": search_summary,
        "final_trace": final_summary,
        "layout_search": {
            "stage": "adaptive disjoint label-free system search",
            "candidate_token_budgets": args.candidate_token_budgets,
            "selection_rule": (
                "maximum median valid-token throughput on two GPUs, selected "
                "independently for host-backed and direct-HBM publication"
            ),
            "results": search_results,
            "selected_host_token_budget": selected_host_token_budget,
            "selected_hbm_token_budget": selected_hbm_token_budget,
        },
        "page_layout_search": {
            "stage": "adaptive disjoint label-free HBM page-layout search",
            "candidate_page_sizes": args.candidate_page_sizes,
            "candidate_tile_budgets": args.candidate_page_tile_budgets,
            "selection_rule": "maximum median valid-token throughput on two GPUs",
            "results": page_search_results,
            "selected_page_size": selected_page_size,
            "selected_tile_tokens": selected_page_tile_tokens,
        },
        "packing": {
            "dense_baseline": dense_packing,
            "host_cohort_jagged": jagged_packing,
            "hbm_single_record": single_record_packing,
            "hbm_cohort_jagged": hbm_jagged_packing,
            "cohort_paged": paged_packing,
        },
        "page_table": page_table,
        "resident_operator": resident,
        "host_backed_ablations": ablations,
        "hbm_direct_ablations": hbm_ablations,
        "correctness": correctness,
        "paged_correctness": paged_correctness,
        "v2_exact_reference": exact_reference,
        "derived": {
            "jagged_fused_speedup_over_dense_fused_1gpu": (
                points["dense_fused_persistent_1gpu"]["median_elapsed_seconds"]
                / points["jagged_fused_persistent_1gpu"]["median_elapsed_seconds"]
            ),
            "jagged_fused_speedup_over_dense_fused_2gpu": (
                points["dense_fused_persistent_2gpu"]["median_elapsed_seconds"]
                / host_selected["median_elapsed_seconds"]
            ),
            "jagged_fused_speedup_over_jagged_packed_1gpu": (
                points["jagged_packed_persistent_1gpu"]["median_elapsed_seconds"]
                / points["jagged_fused_persistent_1gpu"]["median_elapsed_seconds"]
            ),
            "jagged_fused_speedup_over_jagged_packed_2gpu": (
                points["jagged_packed_persistent_2gpu"]["median_elapsed_seconds"]
                / host_selected["median_elapsed_seconds"]
            ),
            "jagged_fused_1_to_2_gpu_speedup": (
                points["jagged_fused_persistent_1gpu"]["median_elapsed_seconds"]
                / host_selected["median_elapsed_seconds"]
            ),
            "jagged_fused_2gpu_parallel_efficiency": (
                points["jagged_fused_persistent_1gpu"]["median_elapsed_seconds"]
                / host_selected["median_elapsed_seconds"]
                / 2.0
            ),
            "migration_batch_reduction": (
                dense_packing["batches"] / jagged_packing["batches"]
            ),
            "input_byte_reduction": (
                dense_packing["capsule_bytes"] / jagged_packing["capsule_bytes"]
            ),
            "output_byte_reduction": dense_pool.nbytes / jagged_pool.nbytes,
            "selected_speedup_over_v2_two_gpu_exact": (
                exact_reference["strongest_two_gpu_exact_seconds"]
                / host_selected["median_elapsed_seconds"]
            ),
            "hbm_whole_record_cohort_speedup_over_single_record_1gpu": (
                hbm_points["single_record_fused_hbm_1gpu"][
                    "median_elapsed_seconds"
                ]
                / hbm_points["cohort_jagged_fused_hbm_1gpu"][
                    "median_elapsed_seconds"
                ]
            ),
            "hbm_whole_record_cohort_speedup_over_single_record_2gpu": (
                hbm_points["single_record_fused_hbm_2gpu"][
                    "median_elapsed_seconds"
                ]
                / hbm_points["cohort_jagged_fused_hbm_2gpu"][
                    "median_elapsed_seconds"
                ]
            ),
            "hbm_paged_speedup_over_single_record_1gpu": (
                hbm_points["single_record_fused_hbm_1gpu"][
                    "median_elapsed_seconds"
                ]
                / hbm_points["cohort_paged_fused_hbm_1gpu"][
                    "median_elapsed_seconds"
                ]
            ),
            "hbm_paged_speedup_over_single_record_2gpu": (
                hbm_points["single_record_fused_hbm_2gpu"][
                    "median_elapsed_seconds"
                ]
                / hbm_selected["median_elapsed_seconds"]
            ),
            "hbm_paged_speedup_over_whole_record_cohort_2gpu": (
                hbm_points["cohort_jagged_fused_hbm_2gpu"][
                    "median_elapsed_seconds"
                ]
                / hbm_selected["median_elapsed_seconds"]
            ),
            "hbm_paged_fused_speedup_over_packed_1gpu": (
                hbm_points["cohort_paged_packed_hbm_1gpu"][
                    "median_elapsed_seconds"
                ]
                / hbm_points["cohort_paged_fused_hbm_1gpu"][
                    "median_elapsed_seconds"
                ]
            ),
            "hbm_paged_fused_speedup_over_packed_2gpu": (
                hbm_points["cohort_paged_packed_hbm_2gpu"][
                    "median_elapsed_seconds"
                ]
                / hbm_selected["median_elapsed_seconds"]
            ),
            "hbm_paged_1_to_2_gpu_speedup": (
                hbm_points["cohort_paged_fused_hbm_1gpu"][
                    "median_elapsed_seconds"
                ]
                / hbm_selected["median_elapsed_seconds"]
            ),
            "hbm_paged_2gpu_parallel_efficiency": (
                hbm_points["cohort_paged_fused_hbm_1gpu"][
                    "median_elapsed_seconds"
                ]
                / hbm_selected["median_elapsed_seconds"]
                / 2.0
            ),
            "paged_hbm_direct_speedup_over_host_publication_2gpu": (
                points["paged_fused_persistent_2gpu"]["median_elapsed_seconds"]
                / hbm_selected["median_elapsed_seconds"]
            ),
            "hbm_paged_batch_reduction": (
                single_record_packing["batches"]
                / paged_packing["batches"]
            ),
            "host_paged_speedup_over_dense_fused_2gpu": (
                points["dense_fused_persistent_2gpu"]["median_elapsed_seconds"]
                / points["paged_fused_persistent_2gpu"][
                    "median_elapsed_seconds"
                ]
            ),
        },
        "mechanism_boundary": {
            "specialized": (
                "same-version cache pages are compacted into bounded valid-token "
                "tiles; one compiled cohort program transforms each tile; the "
                "fused operator writes page-addressable jagged K/V directly "
                "into host or serving-resident HBM extents"
            ),
            "generic_support": (
                "FP16 tensor cores, Triton tiling, pinned memory, asynchronous "
                "copies, and LPT device partitioning"
            ),
            "excluded": (
                "organic version lifecycle, finite repair budget, progressive "
                "fallbacks, SSD, foreground interference, four-GPU scaling, "
                "and a same-boundary HBM full-recompute comparison"
            ),
        },
    }
    save_json(result, args.output)
    print(json.dumps({"output": args.output, "derived": result["derived"]}, indent=2))


if __name__ == "__main__":
    main()
