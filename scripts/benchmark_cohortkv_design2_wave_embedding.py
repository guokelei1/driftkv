from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration.design2_distributed import (
    close_d2_distributed_runtime,
    init_d2_distributed_runtime,
)
from hstu_kvcache.migration.design2_embedding import (
    ModuloRowShardedEmbedding,
)
from hstu_kvcache.migration.design2_embedding_capsule import (
    D2MaterializedEmbeddingCapsuleRankPlan,
    compile_d2_embedding_capsule,
    execute_d2_embedding_capsule,
    materialize_d2_embedding_capsule,
)
from hstu_kvcache.migration.design2_plan import (
    D2ActionPlan,
    build_d2_record_owner_map,
    canonical_sha256,
    d2_record_owner_map_sha256,
    file_sha256,
)
from hstu_kvcache.migration.design2_wave_embedding import (
    D2_WAVE_EMBEDDING_BRANCHES,
    D2_WAVE_EMBEDDING_LOOKUP_KERNEL,
    D2WaveEmbeddingLookupPlan,
    build_d2_wave_embedding_logical_request,
    build_d2_wave_embedding_lookup_plan,
    d2_wave_embedding_demand_calls,
    execute_d2_wave_embedding_lookup_plan,
    summarize_d2_wave_embedding_execution,
)
from hstu_kvcache.streaming import (
    reconstruct_organic_windows,
    validate_long_context_plan,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_CHECKPOINT = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/theta_2.pt"
)
PROTOCOL = "cohortkv_d2_wave_embedding_capsule_w3_v3"
_VERSION = re.compile(r"^theta([0-9]+)$")
_MODE_EXECUTION_ORDER = (
    "one_batch_no_dedup",
    "demand_token_microbatch",
    "wave_scope_unique_cache",
    "compiled_vector_only_capsule",
)
_EXPECTED_BRANCH_TOKENS = {
    "mixed": 347062,
    "all_exact": 934917,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--prepared-data")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--token-microbatch", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--expected-visible-devices",
        nargs="+",
        default=["0", "1", "3"],
    )
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _visible_devices() -> tuple[str, ...]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicit")
    devices = tuple(part.strip() for part in value.split(","))
    if not devices or any(not part for part in devices):
        raise RuntimeError("CUDA_VISIBLE_DEVICES is invalid")
    return devices


def _checkpoint_descriptor(
    action_plan: D2ActionPlan,
) -> dict[str, object]:
    upstream_path = _path(action_plan.provenance.artifact)
    if file_sha256(upstream_path) != action_plan.provenance.artifact_sha256:
        raise RuntimeError("D2 upstream artifact hash differs")
    upstream = json.loads(upstream_path.read_text())
    return next(
        value
        for value in upstream["input_provenance"]["checkpoints"]
        if value["version"] == action_plan.target_version
    )


def _load_target_histories(
    action_plan: D2ActionPlan,
    prepared_path: Path,
) -> tuple[dict[int, torch.Tensor], str]:
    if file_sha256(prepared_path) != (
        action_plan.provenance.prepared_data_sha256
    ):
        raise RuntimeError("D2 prepared data hash differs")
    data_plan, metadata = load_prepared_kuairand_plan(prepared_path)
    validate_long_context_plan(data_plan, metadata, 4)
    user_ids = tuple(
        record.prepared_user_id for record in action_plan.records
    )
    windows = reconstruct_organic_windows(data_plan, user_ids)
    match = _VERSION.fullmatch(action_plan.target_version)
    if match is None:
        raise RuntimeError("D2 target version differs")
    target_window = windows[int(match.group(1))]
    if (
        target_window.content_sha256
        != action_plan.provenance.target_window_content_sha256
    ):
        raise RuntimeError("D2 target window hash differs")
    histories = {}
    for record in action_plan.records:
        window_record = target_window.records[
            record.prepared_user_id
        ]
        history = window_record.history
        if (
            history is None
            or window_record.history_sha256
            != record.target_history_sha256
            or len(history) != record.final_tokens
        ):
            raise RuntimeError("D2 target history identity differs")
        histories[record.record_id] = torch.as_tensor(
            history.item_ids,
            dtype=torch.int64,
        ).contiguous()
    return histories, target_window.content_sha256


def _load_embedding_shard(
    checkpoint_path: Path,
    checkpoint_descriptor: dict[str, object],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[ModuloRowShardedEmbedding, dict[str, object]]:
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint_sha256 != checkpoint_descriptor["sha256"]:
        raise RuntimeError("D2 target checkpoint hash differs")
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    full_weight = state["item_emb.weight"]
    if (
        full_weight.ndim != 2
        or not full_weight.is_floating_point()
    ):
        raise RuntimeError("D2 target item embedding differs")
    local_weight = (
        full_weight[rank::world_size]
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )
    embedding = ModuloRowShardedEmbedding(
        local_weight=local_weight,
        num_embeddings=full_weight.shape[0],
        rank=rank,
        world_size=world_size,
    )
    descriptor = {
        "checkpoint_sha256": checkpoint_sha256,
        "num_embeddings": full_weight.shape[0],
        "hidden_size": full_weight.shape[1],
        "source_dtype": str(full_weight.dtype),
        "transport_dtype": str(embedding.local_weight.dtype),
        "local_rows": embedding.local_weight.shape[0],
        "local_weight_bytes": (
            embedding.local_weight.numel()
            * embedding.local_weight.element_size()
        ),
    }
    del local_weight
    del full_weight
    del state
    gc.collect()
    torch.cuda.synchronize(device)
    return embedding, descriptor


def _prepare_lookup_plan(
    action_plan: D2ActionPlan,
    histories: dict[int, torch.Tensor],
    owner_map: dict[int, int],
    *,
    branch: str,
    mode: str,
    rank: int,
    world_size: int,
    token_microbatch: int,
    hidden_size: int,
    device: torch.device,
) -> tuple[D2WaveEmbeddingLookupPlan, dict[str, object]]:
    dist.barrier()
    started = time.perf_counter()
    logical = build_d2_wave_embedding_logical_request(
        action_plan.records,
        histories,
        owner_map,
        branch=branch,
        rank=rank,
        world_size=world_size,
    )
    plan_collective_seconds = 0.0
    lookup_calls = 1
    local_demand_calls = 1
    if mode == "demand_token_microbatch":
        local_demand_calls = d2_wave_embedding_demand_calls(
            logical.logical_tokens,
            token_microbatch,
        )
        calls = torch.tensor(
            local_demand_calls,
            dtype=torch.int64,
            device=device,
        )
        torch.cuda.synchronize(device)
        collective_started = time.perf_counter()
        dist.all_reduce(calls, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize(device)
        plan_collective_seconds = (
            time.perf_counter() - collective_started
        )
        lookup_calls = int(calls.item())
    plan = build_d2_wave_embedding_lookup_plan(
        logical,
        mode=mode,
        token_microbatch=token_microbatch,
        lookup_calls=lookup_calls,
    )
    plan_build_seconds = time.perf_counter() - started
    transfer_started = time.perf_counter()
    device_plan = plan.to(device)
    torch.cuda.synchronize(device)
    prepared_transfer_seconds = time.perf_counter() - transfer_started
    full_prepare_seconds = (
        plan_build_seconds + prepared_transfer_seconds
    )
    report = {
        "plan_build_seconds": plan_build_seconds,
        "plan_collective_seconds": plan_collective_seconds,
        "prepared_transfer_seconds_excluded": (
            prepared_transfer_seconds
        ),
        "plan_build_and_materialization_seconds": (
            full_prepare_seconds
        ),
        "full_prepare_seconds": full_prepare_seconds,
        "lookup_calls": device_plan.lookup_calls,
        "local_demand_calls": local_demand_calls,
        "empty_alignment_calls": (
            device_plan.lookup_calls - local_demand_calls
            if mode == "demand_token_microbatch"
            else 0
        ),
        "logical_tokens": logical.logical_tokens,
        "logical_unique_tokens": logical.logical_unique_tokens,
        "logical_remote_tokens": logical.logical_remote_tokens,
        "logical_remote_unique_tokens": (
            logical.logical_remote_unique_tokens
        ),
        "phase_token_counts": dict(logical.phase_token_counts),
        "cache_item_id_bytes": device_plan.cache_item_id_bytes,
        "cache_vector_bytes": device_plan.cache_vector_bytes(hidden_size),
        "inverse_bytes": device_plan.inverse_bytes,
        "cache_and_inverse_bytes": (
            device_plan.cache_item_id_bytes
            + device_plan.cache_vector_bytes(hidden_size)
            + device_plan.inverse_bytes
        ),
    }
    return device_plan, report


def _prepare_compiled_capsule(
    action_plan: D2ActionPlan,
    histories: dict[int, torch.Tensor],
    owner_map: dict[int, int],
    embedding: ModuloRowShardedEmbedding,
    *,
    branch: str,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[
    D2MaterializedEmbeddingCapsuleRankPlan,
    dict[str, object],
]:
    dist.barrier()
    prepare_started = time.perf_counter()
    logical = build_d2_wave_embedding_logical_request(
        action_plan.records,
        histories,
        owner_map,
        branch=branch,
        rank=rank,
        world_size=world_size,
    )
    local_item_ids = tuple(logical.item_ids.tolist())
    requester_item_ids: list[object] = [None] * world_size
    gather_started = time.perf_counter()
    dist.all_gather_object(requester_item_ids, local_item_ids)
    request_manifest_gather_seconds = (
        time.perf_counter() - gather_started
    )
    if not all(isinstance(value, tuple) for value in requester_item_ids):
        raise RuntimeError("D2 capsule request manifest differs")
    capsule_plan = compile_d2_embedding_capsule(
        tuple(requester_item_ids),
        num_embeddings=embedding.num_embeddings,
        world_size=world_size,
    )
    materialized = materialize_d2_embedding_capsule(
        capsule_plan,
        rank,
        device,
    )
    prepare_seconds = time.perf_counter() - prepare_started
    plan_build_seconds = (
        prepare_seconds - materialized.materialization_seconds
    )
    element_bytes = embedding.local_weight.element_size()
    unique_vector_bytes = (
        materialized.unique_tokens
        * embedding.hidden_size
        * element_bytes
    )
    reconstructed_vector_bytes = (
        materialized.requested_tokens
        * embedding.hidden_size
        * element_bytes
    )
    report = {
        "plan_build_seconds": plan_build_seconds,
        "plan_collective_seconds": (
            request_manifest_gather_seconds
        ),
        "prepared_transfer_seconds_excluded": (
            materialized.materialization_seconds
        ),
        "capsule_prepare_seconds_excluded": prepare_seconds,
        "plan_build_and_materialization_seconds": (
            prepare_seconds
        ),
        "full_prepare_seconds": prepare_seconds,
        "request_manifest_gather_seconds_excluded": (
            request_manifest_gather_seconds
        ),
        "capsule_compile_seconds_excluded": (
            capsule_plan.compile_seconds
        ),
        "capsule_materialization_seconds_excluded": (
            materialized.materialization_seconds
        ),
        "compiled_global_plan_bytes": capsule_plan.plan_nbytes,
        "compiled_rank_plan_bytes": materialized.rank_plan_bytes,
        "materialized_rank_plan_bytes": (
            materialized.materialized_plan_bytes
        ),
        "steady_resident_materialized_plan_bytes": (
            materialized.materialized_plan_bytes
        ),
        "unique_vector_capsule_bytes_per_repeat": (
            unique_vector_bytes
        ),
        "reconstructed_vector_bytes_per_repeat": (
            reconstructed_vector_bytes
        ),
        "plan_plus_unique_capsule_live_bytes_per_repeat": (
            materialized.materialized_plan_bytes
            + unique_vector_bytes
        ),
        "lookup_calls": 1,
        "local_demand_calls": 1,
        "empty_alignment_calls": 0,
        "logical_tokens": logical.logical_tokens,
        "logical_unique_tokens": logical.logical_unique_tokens,
        "logical_remote_tokens": logical.logical_remote_tokens,
        "logical_remote_unique_tokens": (
            logical.logical_remote_unique_tokens
        ),
        "phase_token_counts": dict(logical.phase_token_counts),
        "cache_item_id_bytes": 0,
        "cache_vector_bytes": unique_vector_bytes,
        "inverse_bytes": (
            materialized.inverse_slots.numel()
            * materialized.inverse_slots.element_size()
        ),
        "cache_and_inverse_bytes": (
            materialized.materialized_plan_bytes
            + unique_vector_bytes
        ),
    }
    return materialized, report


def _compare_vectors(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[bool, float]:
    bitwise = torch.equal(candidate, reference)
    if bitwise:
        return True, 0.0
    maximum = torch.max(
        torch.abs(candidate.float() - reference.float())
    )
    return False, float(maximum.item())


def _capsule_repeat_summary(
    lookup,
    plan_report: dict[str, object],
    *,
    branch: str,
    rank: int,
    world_size: int,
    repeat: int,
    bitwise: bool,
    max_abs_error: float,
) -> dict[str, object]:
    metrics = lookup.metrics
    return {
        "branch": branch,
        "mode": "compiled_vector_only_capsule",
        "rank": rank,
        "world_size": world_size,
        "phase_token_counts": plan_report["phase_token_counts"],
        "logical_tokens": plan_report["logical_tokens"],
        "logical_unique_tokens": plan_report[
            "logical_unique_tokens"
        ],
        "logical_remote_tokens": plan_report[
            "logical_remote_tokens"
        ],
        "logical_remote_unique_tokens": plan_report[
            "logical_remote_unique_tokens"
        ],
        "lookup_calls": 1,
        "lookup_requested_tokens": metrics.unique_tokens,
        "lookup_unique_tokens_sum": metrics.unique_tokens,
        "lookup_remote_requested_tokens": (
            metrics.remote_unique_tokens
        ),
        "lookup_remote_unique_tokens_sum": (
            metrics.remote_unique_tokens
        ),
        "served_remote_requested_tokens": (
            metrics.served_remote_unique_tokens
        ),
        "counts_collective_payload_bytes": (
            metrics.counts_collective_bytes
        ),
        "id_collective_payload_bytes": metrics.id_collective_bytes,
        "vector_collective_payload_bytes": (
            metrics.vector_collective_payload_bytes
        ),
        "collective_tensor_input_bytes": (
            metrics.vector_collective_input_bytes
        ),
        "collective_tensor_output_bytes": (
            metrics.vector_collective_output_bytes
        ),
        "actual_collective_tensor_payload_bytes": (
            metrics.vector_collective_payload_bytes
        ),
        "off_diagonal_send_bytes": (
            metrics.off_diagonal_send_bytes
        ),
        "off_diagonal_receive_bytes": (
            metrics.off_diagonal_receive_bytes
        ),
        "off_diagonal_bytes": metrics.off_diagonal_bytes,
        "collective_calls": metrics.collective_calls,
        "off_diagonal_collective_calls": (
            metrics.collective_calls
        ),
        "counts_collective_seconds": 0.0,
        "id_collective_seconds": 0.0,
        "vector_collective_seconds": metrics.collective_seconds,
        "collective_seconds": metrics.collective_seconds,
        "off_diagonal_collective_seconds": (
            metrics.collective_seconds
        ),
        "makespan_seconds": metrics.execution_seconds,
        "reconstruction_seconds": (
            metrics.execution_seconds - metrics.collective_seconds
        ),
        "cache_item_id_bytes": plan_report["cache_item_id_bytes"],
        "cache_vector_bytes": plan_report["cache_vector_bytes"],
        "inverse_bytes": plan_report["inverse_bytes"],
        "cache_and_inverse_bytes": plan_report[
            "cache_and_inverse_bytes"
        ],
        "capsule_compile_seconds_excluded": (
            metrics.plan_compile_seconds
        ),
        "capsule_materialization_seconds_excluded": (
            metrics.plan_materialization_seconds
        ),
        "compiled_global_plan_bytes": metrics.global_plan_bytes,
        "compiled_rank_plan_bytes": metrics.rank_plan_bytes,
        "materialized_rank_plan_bytes": (
            metrics.materialized_plan_bytes
        ),
        "unique_vector_capsule_bytes_per_repeat": plan_report[
            "unique_vector_capsule_bytes_per_repeat"
        ],
        "reconstructed_vector_bytes_per_repeat": plan_report[
            "reconstructed_vector_bytes_per_repeat"
        ],
        "lookup_kernel": "compiled_vector_only_capsule_v1",
        "timed_payload_hashing": False,
        "timing_cuda_synchronized": True,
        "repeat": repeat,
        "bitwise_equal_no_dedup": bitwise,
        "max_abs_error_no_dedup": max_abs_error,
    }


def _execute_compiled_capsule_mode(
    materialized: D2MaterializedEmbeddingCapsuleRankPlan,
    embedding: ModuloRowShardedEmbedding,
    plan_report: dict[str, object],
    *,
    branch: str,
    repeats: int,
    warmup_repeats: int,
    reference: torch.Tensor,
) -> dict[str, object]:
    warmup_bitwise = []
    for _ in range(warmup_repeats):
        dist.barrier()
        lookup = execute_d2_embedding_capsule(
            materialized,
            embedding.local_weight,
        )
        bitwise, _ = _compare_vectors(
            lookup.item_vectors,
            reference,
        )
        warmup_bitwise.append(bitwise)
        del lookup
    repeat_reports = []
    for repeat in range(repeats):
        dist.barrier()
        lookup = execute_d2_embedding_capsule(
            materialized,
            embedding.local_weight,
        )
        bitwise, max_abs_error = _compare_vectors(
            lookup.item_vectors,
            reference,
        )
        repeat_reports.append(
            _capsule_repeat_summary(
                lookup,
                plan_report,
                branch=branch,
                rank=materialized.rank,
                world_size=materialized.world_size,
                repeat=repeat,
                bitwise=bitwise,
                max_abs_error=max_abs_error,
            )
        )
        del lookup
    return {
        "warmup_repeats": warmup_repeats,
        "warmup_bitwise_equal_no_dedup": all(warmup_bitwise),
        "repeats": repeat_reports,
    }


def _execute_mode(
    embedding: ModuloRowShardedEmbedding,
    plan,
    *,
    repeats: int,
    warmup_repeats: int,
    reference: torch.Tensor | None,
) -> tuple[dict[str, object], torch.Tensor | None]:
    warmup_bitwise = []
    retained_reference = reference
    for _ in range(warmup_repeats):
        dist.barrier()
        execution = execute_d2_wave_embedding_lookup_plan(
            embedding,
            plan,
        )
        if retained_reference is None:
            retained_reference = execution.item_vectors.detach()
            bitwise = True
        else:
            bitwise, _ = _compare_vectors(
                execution.item_vectors,
                retained_reference,
            )
        warmup_bitwise.append(bitwise)
        if execution.item_vectors is not retained_reference:
            del execution
    repeat_reports = []
    for repeat in range(repeats):
        dist.barrier()
        execution = execute_d2_wave_embedding_lookup_plan(
            embedding,
            plan,
        )
        if retained_reference is None:
            retained_reference = execution.item_vectors.detach()
            bitwise = True
            max_abs_error = 0.0
        else:
            bitwise, max_abs_error = _compare_vectors(
                execution.item_vectors,
                retained_reference,
            )
        summary = summarize_d2_wave_embedding_execution(
            plan,
            execution,
            hidden_size=embedding.hidden_size,
        )
        summary.update(
            {
                "repeat": repeat,
                "bitwise_equal_no_dedup": bitwise,
                "max_abs_error_no_dedup": max_abs_error,
            }
        )
        repeat_reports.append(summary)
        if execution.item_vectors is not retained_reference:
            del execution
    return (
        {
            "warmup_repeats": warmup_repeats,
            "warmup_bitwise_equal_no_dedup": all(warmup_bitwise),
            "repeats": repeat_reports,
        },
        retained_reference,
    )


def _aggregate_mode(
    rank_modes: tuple[dict[str, object], ...],
    *,
    branch: str,
    mode: str,
) -> dict[str, object]:
    if (
        tuple(value["rank"] for value in rank_modes)
        != tuple(range(len(rank_modes)))
        or any(
            value["branch"] != branch or value["mode"] != mode
            for value in rank_modes
        )
    ):
        raise RuntimeError("D2 wave embedding rank mode differs")
    repeat_count = len(rank_modes[0]["execution"]["repeats"])
    if any(
        len(value["execution"]["repeats"]) != repeat_count
        for value in rank_modes
    ):
        raise RuntimeError("D2 wave embedding repeat count differs")
    repeats = []
    for repeat in range(repeat_count):
        ranks = tuple(
            value["execution"]["repeats"][repeat]
            for value in rank_modes
        )
        send_bytes = sum(
            value["off_diagonal_send_bytes"] for value in ranks
        )
        receive_bytes = sum(
            value["off_diagonal_receive_bytes"] for value in ranks
        )
        endpoint_bytes = sum(
            value["off_diagonal_bytes"] for value in ranks
        )
        rank_single_wave_seconds = [
            rank_mode["plan"]["full_prepare_seconds"]
            + rank_execution["makespan_seconds"]
            for rank_mode, rank_execution in zip(
                rank_modes,
                ranks,
                strict=True,
            )
        ]
        repeats.append(
            {
                "repeat": repeat,
                "max_rank_makespan_seconds": max(
                    value["makespan_seconds"] for value in ranks
                ),
                "max_rank_collective_seconds": max(
                    value["collective_seconds"] for value in ranks
                ),
                "max_rank_single_wave_including_plan_seconds": max(
                    rank_single_wave_seconds
                ),
                "sum_rank_logical_tokens": sum(
                    value["logical_tokens"] for value in ranks
                ),
                "sum_rank_logical_unique_tokens": sum(
                    value["logical_unique_tokens"] for value in ranks
                ),
                "sum_rank_logical_remote_tokens": sum(
                    value["logical_remote_tokens"] for value in ranks
                ),
                "sum_rank_lookup_requested_tokens": sum(
                    value["lookup_requested_tokens"]
                    for value in ranks
                ),
                "sum_rank_lookup_remote_requested_tokens": sum(
                    value["lookup_remote_requested_tokens"]
                    for value in ranks
                ),
                "sum_rank_collective_tensor_payload_bytes": sum(
                    value["actual_collective_tensor_payload_bytes"]
                    for value in ranks
                ),
                "sum_rank_collective_tensor_input_bytes": sum(
                    value["collective_tensor_input_bytes"]
                    for value in ranks
                ),
                "sum_rank_collective_tensor_output_bytes": sum(
                    value["collective_tensor_output_bytes"]
                    for value in ranks
                ),
                "sum_rank_counts_collective_payload_bytes": sum(
                    value["counts_collective_payload_bytes"]
                    for value in ranks
                ),
                "sum_rank_id_collective_payload_bytes": sum(
                    value["id_collective_payload_bytes"]
                    for value in ranks
                ),
                "sum_rank_vector_collective_payload_bytes": sum(
                    value["vector_collective_payload_bytes"]
                    for value in ranks
                ),
                "collective_calls_per_rank": [
                    value["collective_calls"] for value in ranks
                ],
                "off_diagonal_one_way_send_bytes": send_bytes,
                "off_diagonal_one_way_receive_bytes": receive_bytes,
                "off_diagonal_one_way_vector_payload_bytes": (
                    send_bytes
                    if mode == "compiled_vector_only_capsule"
                    else None
                ),
                "off_diagonal_endpoint_summed_bytes": endpoint_bytes,
                "off_diagonal_send_receive_match": (
                    send_bytes == receive_bytes
                    and endpoint_bytes == send_bytes + receive_bytes
                ),
                "all_ranks_bitwise_equal_no_dedup": all(
                    value["bitwise_equal_no_dedup"]
                    and value["max_abs_error_no_dedup"] == 0.0
                    for value in ranks
                ),
                "rank_makespan_seconds": [
                    value["makespan_seconds"] for value in ranks
                ],
                "rank_collective_seconds": [
                    value["collective_seconds"] for value in ranks
                ],
                "rank_single_wave_including_plan_seconds": (
                    rank_single_wave_seconds
                ),
            }
        )
    return {
        "branch": branch,
        "mode": mode,
        "rank_reports": list(rank_modes),
        "repeats": repeats,
        "summary": {
            "repeats": repeat_count,
            "median_max_rank_makespan_seconds": statistics.median(
                value["max_rank_makespan_seconds"] for value in repeats
            ),
            "median_max_rank_collective_seconds": statistics.median(
                value["max_rank_collective_seconds"] for value in repeats
            ),
            "median_max_rank_single_wave_including_plan_seconds": (
                statistics.median(
                    value[
                        "max_rank_single_wave_including_plan_seconds"
                    ]
                    for value in repeats
                )
            ),
            "median_off_diagonal_one_way_send_bytes": (
                statistics.median(
                    value["off_diagonal_one_way_send_bytes"]
                    for value in repeats
                )
            ),
            "median_off_diagonal_one_way_vector_payload_bytes": (
                statistics.median(
                    value[
                        "off_diagonal_one_way_vector_payload_bytes"
                    ]
                    for value in repeats
                )
                if mode == "compiled_vector_only_capsule"
                else None
            ),
            "median_off_diagonal_endpoint_summed_bytes": (
                statistics.median(
                    value["off_diagonal_endpoint_summed_bytes"]
                    for value in repeats
                )
            ),
            "median_collective_tensor_endpoint_payload_bytes": (
                statistics.median(
                    value[
                        "sum_rank_collective_tensor_payload_bytes"
                    ]
                    for value in repeats
                )
            ),
            "all_bitwise_equal_no_dedup": all(
                value["all_ranks_bitwise_equal_no_dedup"]
                for value in repeats
            ),
            "all_off_diagonal_send_receive_match": all(
                value["off_diagonal_send_receive_match"]
                for value in repeats
            ),
            "lookup_kernels": sorted(
                {
                    repeat["lookup_kernel"]
                    for value in rank_modes
                    for repeat in value["execution"]["repeats"]
                }
            ),
            "all_timed_payload_hashing_disabled": all(
                not repeat["timed_payload_hashing"]
                for value in rank_modes
                for repeat in value["execution"]["repeats"]
            ),
            "all_timing_cuda_synchronized": all(
                repeat["timing_cuda_synchronized"]
                for value in rank_modes
                for repeat in value["execution"]["repeats"]
            ),
            "max_plan_build_seconds": max(
                value["plan"]["plan_build_seconds"]
                for value in rank_modes
            ),
            "max_plan_collective_seconds": max(
                value["plan"]["plan_collective_seconds"]
                for value in rank_modes
            ),
            "max_prepared_transfer_seconds_excluded": max(
                value["plan"]["prepared_transfer_seconds_excluded"]
                for value in rank_modes
            ),
            "max_rank_plan_build_and_materialization_seconds": max(
                value["plan"][
                    "plan_build_and_materialization_seconds"
                ]
                for value in rank_modes
            ),
            "max_rank_full_prepare_seconds": max(
                value["plan"]["full_prepare_seconds"]
                for value in rank_modes
            ),
            "max_capsule_compile_seconds_excluded": max(
                value["plan"].get(
                    "capsule_compile_seconds_excluded",
                    0.0,
                )
                for value in rank_modes
            ),
            "max_capsule_materialization_seconds_excluded": max(
                value["plan"].get(
                    "capsule_materialization_seconds_excluded",
                    0.0,
                )
                for value in rank_modes
            ),
            "sum_rank_materialized_plan_bytes": sum(
                value["plan"].get(
                    "materialized_rank_plan_bytes",
                    0,
                )
                for value in rank_modes
            ),
            "max_rank_steady_resident_materialized_plan_bytes": max(
                value["plan"].get(
                    "steady_resident_materialized_plan_bytes",
                    0,
                )
                for value in rank_modes
            ),
            "max_rank_plan_plus_unique_capsule_live_bytes_per_repeat": max(
                value["plan"].get(
                    "plan_plus_unique_capsule_live_bytes_per_repeat",
                    0,
                )
                for value in rank_modes
            ),
            "lookup_calls_per_rank": [
                value["plan"]["lookup_calls"] for value in rank_modes
            ],
            "empty_alignment_calls_per_rank": [
                value["plan"]["empty_alignment_calls"]
                for value in rank_modes
            ],
        },
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _amortization_report(
    candidate: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, object]:
    candidate_prepare = float(
        candidate["max_rank_full_prepare_seconds"]
    )
    baseline_prepare = float(
        baseline["max_rank_full_prepare_seconds"]
    )
    candidate_steady = float(
        candidate["median_max_rank_makespan_seconds"]
    )
    baseline_steady = float(
        baseline["median_max_rank_makespan_seconds"]
    )
    additional_prepare = candidate_prepare - baseline_prepare
    steady_savings = baseline_steady - candidate_steady
    minimum_executions = None
    status = "steady_not_faster"
    if steady_savings > 0:
        minimum_executions = max(
            1,
            math.ceil(max(0.0, additional_prepare) / steady_savings),
        )
        while (
            candidate_prepare
            + minimum_executions * candidate_steady
            > baseline_prepare
            + minimum_executions * baseline_steady
        ):
            minimum_executions += 1
        status = (
            "wins_first_execution"
            if minimum_executions == 1
            else "amortizes_with_identical_manifest_reuse"
        )
    return {
        "model": (
            "max-rank full prepare plus N times median "
            "max-rank steady execution"
        ),
        "scope": "identical frozen whole-branch request manifest",
        "candidate_full_prepare_seconds": candidate_prepare,
        "baseline_full_prepare_seconds": baseline_prepare,
        "additional_prepare_seconds": additional_prepare,
        "candidate_steady_seconds": candidate_steady,
        "baseline_steady_seconds": baseline_steady,
        "steady_savings_seconds_per_execution": steady_savings,
        "candidate_single_wave_seconds": (
            candidate_prepare + candidate_steady
        ),
        "baseline_single_wave_seconds": (
            baseline_prepare + baseline_steady
        ),
        "minimum_total_executions_to_break_even": (
            minimum_executions
        ),
        "minimum_reuses_after_first_to_break_even": (
            None
            if minimum_executions is None
            else minimum_executions - 1
        ),
        "status": status,
    }


def _aggregate(
    args: argparse.Namespace,
    action_plan: D2ActionPlan,
    prepared_path: Path,
    checkpoint_path: Path,
    target_window_sha256: str,
    owner_map: dict[int, int],
    gathered: tuple[dict[str, object], ...],
) -> dict[str, object]:
    branches = {}
    checks = {}
    for branch in D2_WAVE_EMBEDDING_BRANCHES:
        modes = {}
        for mode in _MODE_EXECUTION_ORDER:
            rank_modes = tuple(
                value["branches"][branch]["modes"][mode]
                for value in gathered
            )
            modes[mode] = _aggregate_mode(
                rank_modes,
                branch=branch,
                mode=mode,
            )
        baseline = modes["one_batch_no_dedup"]["summary"]
        cached = modes["wave_scope_unique_cache"]["summary"]
        demand = modes["demand_token_microbatch"]["summary"]
        capsule = modes["compiled_vector_only_capsule"]["summary"]
        baseline_first = modes["one_batch_no_dedup"]["repeats"][0]
        cached_first = modes["wave_scope_unique_cache"]["repeats"][0]
        demand_first = modes["demand_token_microbatch"]["repeats"][0]
        capsule_first = modes["compiled_vector_only_capsule"][
            "repeats"
        ][0]
        branches[branch] = {
            "modes": modes,
            "comparisons": {
                "wave_cache_over_one_batch": {
                    "lookup_requested_token_ratio": _ratio(
                        cached_first[
                            "sum_rank_lookup_requested_tokens"
                        ],
                        baseline_first[
                            "sum_rank_lookup_requested_tokens"
                        ],
                    ),
                    "one_way_off_diagonal_byte_ratio": _ratio(
                        cached[
                            "median_off_diagonal_one_way_send_bytes"
                        ],
                        baseline[
                            "median_off_diagonal_one_way_send_bytes"
                        ],
                    ),
                    "collective_second_ratio": _ratio(
                        cached[
                            "median_max_rank_collective_seconds"
                        ],
                        baseline[
                            "median_max_rank_collective_seconds"
                        ],
                    ),
                    "makespan_ratio": _ratio(
                        cached[
                            "median_max_rank_makespan_seconds"
                        ],
                        baseline[
                            "median_max_rank_makespan_seconds"
                        ],
                    ),
                    "single_wave_including_plan_ratio": _ratio(
                        cached[
                            "median_max_rank_single_wave_including_plan_seconds"
                        ],
                        baseline[
                            "median_max_rank_single_wave_including_plan_seconds"
                        ],
                    ),
                    "amortization": _amortization_report(
                        cached,
                        baseline,
                    ),
                },
                "demand_microbatch_over_one_batch": {
                    "lookup_requested_token_ratio": _ratio(
                        demand_first[
                            "sum_rank_lookup_requested_tokens"
                        ],
                        baseline_first[
                            "sum_rank_lookup_requested_tokens"
                        ],
                    ),
                    "one_way_off_diagonal_byte_ratio": _ratio(
                        demand[
                            "median_off_diagonal_one_way_send_bytes"
                        ],
                        baseline[
                            "median_off_diagonal_one_way_send_bytes"
                        ],
                    ),
                    "collective_second_ratio": _ratio(
                        demand[
                            "median_max_rank_collective_seconds"
                        ],
                        baseline[
                            "median_max_rank_collective_seconds"
                        ],
                    ),
                    "makespan_ratio": _ratio(
                        demand[
                            "median_max_rank_makespan_seconds"
                        ],
                        baseline[
                            "median_max_rank_makespan_seconds"
                        ],
                    ),
                    "single_wave_including_plan_ratio": _ratio(
                        demand[
                            "median_max_rank_single_wave_including_plan_seconds"
                        ],
                        baseline[
                            "median_max_rank_single_wave_including_plan_seconds"
                        ],
                    ),
                    "amortization": _amortization_report(
                        demand,
                        baseline,
                    ),
                },
                "compiled_capsule_over_one_batch": {
                    "lookup_requested_token_ratio": _ratio(
                        capsule_first[
                            "sum_rank_lookup_requested_tokens"
                        ],
                        baseline_first[
                            "sum_rank_lookup_requested_tokens"
                        ],
                    ),
                    "one_way_off_diagonal_byte_ratio": _ratio(
                        capsule[
                            "median_off_diagonal_one_way_send_bytes"
                        ],
                        baseline[
                            "median_off_diagonal_one_way_send_bytes"
                        ],
                    ),
                    "collective_second_ratio": _ratio(
                        capsule[
                            "median_max_rank_collective_seconds"
                        ],
                        baseline[
                            "median_max_rank_collective_seconds"
                        ],
                    ),
                    "makespan_ratio": _ratio(
                        capsule[
                            "median_max_rank_makespan_seconds"
                        ],
                        baseline[
                            "median_max_rank_makespan_seconds"
                        ],
                    ),
                    "single_wave_including_plan_ratio": _ratio(
                        capsule[
                            "median_max_rank_single_wave_including_plan_seconds"
                        ],
                        baseline[
                            "median_max_rank_single_wave_including_plan_seconds"
                        ],
                    ),
                    "amortization": _amortization_report(
                        capsule,
                        baseline,
                    ),
                },
                "compiled_capsule_over_dynamic_wave_cache": {
                    "lookup_requested_token_ratio": _ratio(
                        capsule_first[
                            "sum_rank_lookup_requested_tokens"
                        ],
                        cached_first[
                            "sum_rank_lookup_requested_tokens"
                        ],
                    ),
                    "one_way_off_diagonal_byte_ratio": _ratio(
                        capsule[
                            "median_off_diagonal_one_way_send_bytes"
                        ],
                        cached[
                            "median_off_diagonal_one_way_send_bytes"
                        ],
                    ),
                    "collective_second_ratio": _ratio(
                        capsule[
                            "median_max_rank_collective_seconds"
                        ],
                        cached[
                            "median_max_rank_collective_seconds"
                        ],
                    ),
                    "makespan_ratio": _ratio(
                        capsule[
                            "median_max_rank_makespan_seconds"
                        ],
                        cached[
                            "median_max_rank_makespan_seconds"
                        ],
                    ),
                    "single_wave_including_plan_ratio": _ratio(
                        capsule[
                            "median_max_rank_single_wave_including_plan_seconds"
                        ],
                        cached[
                            "median_max_rank_single_wave_including_plan_seconds"
                        ],
                    ),
                    "amortization": _amortization_report(
                        capsule,
                        cached,
                    ),
                },
            },
        }
        branch_checks = {
            "logical_tokens_match_frozen_h12": all(
                mode_value["repeats"][0]["sum_rank_logical_tokens"]
                == _EXPECTED_BRANCH_TOKENS[branch]
                for mode_value in modes.values()
            ),
            "all_modes_bitwise_equal_no_dedup": all(
                mode_value["summary"]["all_bitwise_equal_no_dedup"]
                for mode_value in modes.values()
            ),
            "all_modes_disable_timed_payload_hashing": all(
                mode_value["summary"][
                    "all_timed_payload_hashing_disabled"
                ]
                for mode_value in modes.values()
            ),
            "all_modes_use_synchronized_timing": all(
                mode_value["summary"]["all_timing_cuda_synchronized"]
                for mode_value in modes.values()
            ),
            "dynamic_modes_use_same_nohash_kernel": all(
                modes[mode]["summary"]["lookup_kernels"]
                == [D2_WAVE_EMBEDDING_LOOKUP_KERNEL]
                for mode in (
                    "one_batch_no_dedup",
                    "demand_token_microbatch",
                    "wave_scope_unique_cache",
                )
            ),
            "off_diagonal_accounting_closes": all(
                mode_value["summary"][
                    "all_off_diagonal_send_receive_match"
                ]
                for mode_value in modes.values()
            ),
            "demand_calls_aligned": len(
                set(demand["lookup_calls_per_rank"])
            )
            == 1,
            "no_dedup_submits_all_logical_tokens": (
                baseline_first["sum_rank_lookup_requested_tokens"]
                == baseline_first["sum_rank_logical_tokens"]
            ),
            "demand_submits_all_logical_tokens": (
                demand_first["sum_rank_lookup_requested_tokens"]
                == demand_first["sum_rank_logical_tokens"]
            ),
            "wave_cache_submits_requester_unique_tokens": (
                cached_first["sum_rank_lookup_requested_tokens"]
                == cached_first["sum_rank_logical_unique_tokens"]
            ),
            "compiled_capsule_submits_requester_unique_tokens": (
                capsule_first["sum_rank_lookup_requested_tokens"]
                == capsule_first["sum_rank_logical_unique_tokens"]
            ),
            "compiled_capsule_uses_one_collective_per_rank": all(
                all(call == 1 for call in value["collective_calls_per_rank"])
                for value in modes[
                    "compiled_vector_only_capsule"
                ]["repeats"]
            ),
            "compiled_capsule_one_way_vector_payload_is_explicit": all(
                value["off_diagonal_one_way_vector_payload_bytes"]
                == value["sum_rank_collective_tensor_input_bytes"]
                == value["off_diagonal_one_way_send_bytes"]
                for value in modes[
                    "compiled_vector_only_capsule"
                ]["repeats"]
            ),
            "compiled_capsule_vector_endpoint_accounting_closes": all(
                value["sum_rank_collective_tensor_input_bytes"]
                + value["sum_rank_collective_tensor_output_bytes"]
                == value["sum_rank_vector_collective_payload_bytes"]
                for value in modes[
                    "compiled_vector_only_capsule"
                ]["repeats"]
            ),
            "compiled_capsule_has_zero_count_and_id_bytes": all(
                value["sum_rank_counts_collective_payload_bytes"] == 0
                and value["sum_rank_id_collective_payload_bytes"] == 0
                for value in modes[
                    "compiled_vector_only_capsule"
                ]["repeats"]
            ),
            "compiled_capsule_payload_is_vector_only": all(
                value["sum_rank_collective_tensor_payload_bytes"]
                == value["sum_rank_vector_collective_payload_bytes"]
                for value in modes[
                    "compiled_vector_only_capsule"
                ]["repeats"]
            ),
        }
        checks[branch] = branch_checks
    checks["all_passed"] = all(
        all(value.values()) for value in checks.values()
    )
    result = {
        "protocol": PROTOCOL,
        "predecessor_protocol": (
            "cohortkv_d2_wave_embedding_capsule_w3_v2"
        ),
        "predecessor_evidence": {
            "v1": {
                "protocol": (
                    "cohortkv_d2_wave_embedding_capsule_w3_v1"
                ),
                "timing_status": (
                    "superseded_by_fair_nohash_v3"
                ),
                "bytes_and_collective_call_evidence": "retained",
            },
            "v2": {
                "protocol": (
                    "cohortkv_d2_wave_embedding_capsule_w3_v2"
                ),
                "timing_status": (
                    "invalidated_by_timed_sha_in_dynamic_modes"
                ),
                "bytes_and_collective_call_evidence": "retained",
            },
        },
        "status": "complete" if checks["all_passed"] else "failed",
        "scientific_result": False,
        "development_design_validation": True,
        "formal_stage_b_gate": False,
        "formal_stage_c": False,
        "artifact_semantics": {
            "boundary": (
                "whole-branch requester-scope embedding capsule "
                "microbenchmark"
            ),
            "request_timing": (
                "all target histories and the frozen branch request "
                "multiset are available before lookup"
            ),
            "wave_cache_scope": (
                "one unique-ID cache and inverse per requester rank "
                "over the complete branch, crossing logical phases"
            ),
            "one_batch_semantics": (
                "communication capsule baseline, not integrated wave "
                "execution semantics"
            ),
            "compiled_capsule_semantics": (
                "all-rank requester IDs, source rows, destination "
                "slots and inverse are compiled and materialized "
                "before steady execution; each repeat exchanges only "
                "vectors in one collective"
            ),
            "timed": (
                "payload-hash-free dynamic sharded lookup plus "
                "reconstruction, or compiled capsule vector exchange "
                "plus reconstruction; all CUDA work is synchronized"
            ),
            "reported_separately": (
                "request-manifest gather, plan compile, plan "
                "materialization, full preparation, composed single-wave "
                "time, identical-manifest break-even and resident bytes"
            ),
            "single_wave_including_plan": (
                "composed per rank as measured full preparation plus "
                "one steady execution; process startup and warmups are "
                "not included"
            ),
            "break_even": (
                "diagnostic amortization over repeated execution of the "
                "same frozen whole-branch manifest, not an organic "
                "cross-wave reuse claim"
            ),
            "excluded": (
                "process startup, input loading, checkpoint loading, "
                "history reconstruction, model compute, K/V work, "
                "transaction and publication"
            ),
        },
        "inputs": {
            "action_plan": {
                "path": str(_path(args.action_plan).relative_to(ROOT)),
                "file_sha256": file_sha256(_path(args.action_plan)),
                "content_sha256": action_plan.content_sha256,
            },
            "prepared_data": {
                "path": str(prepared_path.relative_to(ROOT)),
                "sha256": file_sha256(prepared_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": file_sha256(checkpoint_path),
            },
            "target_window_content_sha256": target_window_sha256,
        },
        "configuration": {
            "world_size": 3,
            "backend": "nccl",
            "visible_devices": list(_visible_devices()),
            "record_owner": "strict_cow_lpt",
            "record_owner_map_sha256": (
                d2_record_owner_map_sha256(owner_map)
            ),
            "embedding_owner": "item_id_mod_world_size",
            "branches": list(D2_WAVE_EMBEDDING_BRANCHES),
            "mode_execution_order": list(_MODE_EXECUTION_ORDER),
            "token_microbatch": args.token_microbatch,
            "warmup_repeats": args.warmup_repeats,
            "measured_repeats": args.repeats,
            "embedding_transport_dtype": "float32",
            "dynamic_lookup_kernel": (
                D2_WAVE_EMBEDDING_LOOKUP_KERNEL
            ),
            "timed_payload_hashing": False,
        },
        "rank_inputs": [
            {
                "rank": value["rank"],
                "device": value["device"],
                "embedding": value["embedding"],
            }
            for value in gathered
        ],
        "branches": branches,
        "checks": checks,
        "unsupported_claims": [
            "integrated mixed-wave speedup",
            "online serving request-order equivalence",
            "NCCL wire bytes",
            "embedding-tier foreground interference isolation",
            "capacity admission",
            "formal Stage-B or Stage-C completion",
            "capsule-plan reuse across changing organic wave manifests",
        ],
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def _run_rank(
    args: argparse.Namespace,
    runtime,
) -> tuple[
    dict[str, object],
    D2ActionPlan,
    Path,
    Path,
    str,
    dict[int, int],
]:
    if (
        runtime.world_size != 3
        or tuple(args.expected_visible_devices) != _visible_devices()
        or len(args.expected_visible_devices) != 3
        or args.repeats < 1
        or args.warmup_repeats < 0
        or args.token_microbatch < 1
    ):
        raise RuntimeError("D2 W3 benchmark configuration is invalid")
    if _path(args.output).exists():
        raise FileExistsError(
            "D2 W3 v3 output exists; choose a new output path"
        )
    action_path = _path(args.action_plan)
    action_plan = D2ActionPlan.load(action_path)
    prepared_path = _path(
        args.prepared_data or action_plan.provenance.prepared_data
    )
    checkpoint_path = _path(args.checkpoint)
    checkpoint_descriptor = _checkpoint_descriptor(action_plan)
    histories, target_window_sha256 = _load_target_histories(
        action_plan,
        prepared_path,
    )
    owner_map = build_d2_record_owner_map(
        action_plan,
        runtime.world_size,
        "strict_cow_lpt",
    )
    embedding, embedding_descriptor = _load_embedding_shard(
        checkpoint_path,
        checkpoint_descriptor,
        rank=runtime.rank,
        world_size=runtime.world_size,
        device=runtime.device,
    )
    device_properties = torch.cuda.get_device_properties(runtime.device)
    dist.barrier()
    branches = {}
    for branch in D2_WAVE_EMBEDDING_BRANCHES:
        modes = {}
        reference = None
        for mode in _MODE_EXECUTION_ORDER:
            if mode == "compiled_vector_only_capsule":
                if reference is None:
                    raise RuntimeError(
                        "D2 capsule requires one-batch reference"
                    )
                materialized, plan_report = (
                    _prepare_compiled_capsule(
                        action_plan,
                        histories,
                        owner_map,
                        embedding,
                        branch=branch,
                        rank=runtime.rank,
                        world_size=runtime.world_size,
                        device=runtime.device,
                    )
                )
                execution = _execute_compiled_capsule_mode(
                    materialized,
                    embedding,
                    plan_report,
                    branch=branch,
                    repeats=args.repeats,
                    warmup_repeats=args.warmup_repeats,
                    reference=reference,
                )
                del materialized
            else:
                plan, plan_report = _prepare_lookup_plan(
                    action_plan,
                    histories,
                    owner_map,
                    branch=branch,
                    mode=mode,
                    rank=runtime.rank,
                    world_size=runtime.world_size,
                    token_microbatch=args.token_microbatch,
                    hidden_size=embedding.hidden_size,
                    device=runtime.device,
                )
                execution, reference = _execute_mode(
                    embedding,
                    plan,
                    repeats=args.repeats,
                    warmup_repeats=args.warmup_repeats,
                    reference=reference,
                )
                del plan
            modes[mode] = {
                "rank": runtime.rank,
                "branch": branch,
                "mode": mode,
                "plan": plan_report,
                "execution": execution,
            }
        del reference
        torch.cuda.empty_cache()
        branches[branch] = {"modes": modes}
    local = {
        "rank": runtime.rank,
        "device": {
            "local_rank": runtime.local_rank,
            "logical_device": str(runtime.device),
            "name": device_properties.name,
            "uuid": f"GPU-{device_properties.uuid}",
            "total_memory_bytes": device_properties.total_memory,
        },
        "embedding": embedding_descriptor,
        "branches": branches,
    }
    return (
        local,
        action_plan,
        prepared_path,
        checkpoint_path,
        target_window_sha256,
        owner_map,
    )


def run(args: argparse.Namespace) -> dict[str, object] | None:
    runtime = init_d2_distributed_runtime(
        backend="nccl",
        timeout_seconds=args.timeout_seconds,
    )
    try:
        (
            local,
            action_plan,
            prepared_path,
            checkpoint_path,
            target_window_sha256,
            owner_map,
        ) = _run_rank(args, runtime)
        dist.barrier()
        gathered: list[object] = [None] * runtime.world_size
        dist.all_gather_object(gathered, local)
        output = None
        if runtime.is_primary:
            if not all(isinstance(value, dict) for value in gathered):
                raise RuntimeError("D2 W3 benchmark rank report is invalid")
            output = _aggregate(
                args,
                action_plan,
                prepared_path,
                checkpoint_path,
                target_window_sha256,
                owner_map,
                tuple(gathered),
            )
            if output["status"] != "complete":
                raise RuntimeError(
                    f"D2 W3 benchmark checks failed: {output['checks']}"
                )
            output_path = _path(args.output)
            if output_path.exists():
                raise FileExistsError(
                    "D2 W3 v3 output appeared during execution"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(
                f"{output_path.suffix}.{os.getpid()}.tmp"
            )
            temporary.write_text(
                json.dumps(output, indent=2, sort_keys=True) + "\n"
            )
            os.replace(temporary, output_path)
        return output
    finally:
        close_d2_distributed_runtime(runtime)


def main() -> None:
    output = run(parse_args())
    if output is not None:
        print(
            json.dumps(
                {
                    "status": output["status"],
                    "protocol": output["protocol"],
                    "scientific_result": output["scientific_result"],
                    "checks": output["checks"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
