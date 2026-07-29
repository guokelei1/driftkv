from __future__ import annotations

import argparse
import bisect
import gc
import json
import os
import re
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration.design2_distributed import (
    close_d2_distributed_runtime,
    init_d2_distributed_runtime,
)
from hstu_kvcache.migration.design2_embedding_capsule import (
    compile_d2_embedding_capsule,
    materialize_d2_embedding_capsule,
)
from hstu_kvcache.migration.design2_plan import (
    D2ActionPlan,
    build_d2_record_owner_map,
    canonical_sha256,
    d2_record_owner_map_sha256,
    file_sha256,
)
from hstu_kvcache.migration.design2_resource_isolation import (
    D2_RESOURCE_ISOLATION_PROTOCOL,
    D2CollectiveLaunchCoordinator,
    D2FixedRateSchedule,
    D2ForegroundSample,
    D2VectorExchangeSample,
    D2VectorExchangeWorkspace,
    build_d2_fixed_rate_schedule,
    build_d2_synthetic_foreground_request_ring,
    build_d2_vector_exchange_workspace,
    execute_d2_vector_exchange,
    summarize_d2_foreground_samples,
)
from hstu_kvcache.migration.design2_wave_embedding import (
    build_d2_wave_embedding_logical_request,
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
DEFAULT_SCENARIOS = (
    "idle",
    "rank_local_dense_control",
    "mixed_embedding",
    "all_exact_embedding",
)
_VERSION = re.compile(r"^theta([0-9]+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--prepared-data")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--foreground-rate-per-rank",
        type=float,
        default=250.0,
    )
    parser.add_argument(
        "--foreground-duration-seconds",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--foreground-batch-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--foreground-ring-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--foreground-seed",
        type=int,
        default=20260729,
    )
    parser.add_argument(
        "--foreground-deadline-ms",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--maintenance-start-seconds",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--analysis-window-seconds",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--maintenance-repeats",
        type=int,
        default=8,
    )
    parser.add_argument("--dense-rows", type=int, default=2048)
    parser.add_argument("--dense-iterations", type=int, default=1024)
    parser.add_argument(
        "--scenario-start-delay-seconds",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
    )
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
        value = target_window.records[record.prepared_user_id]
        history = value.history
        if (
            history is None
            or value.history_sha256 != record.target_history_sha256
            or len(history) != record.final_tokens
        ):
            raise RuntimeError("D2 target history identity differs")
        histories[record.record_id] = torch.tensor(
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
) -> tuple[torch.Tensor, dict[str, object]]:
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
    descriptor = {
        "checkpoint_sha256": checkpoint_sha256,
        "num_embeddings": full_weight.shape[0],
        "hidden_size": full_weight.shape[1],
        "source_dtype": str(full_weight.dtype),
        "transport_dtype": str(local_weight.dtype),
        "local_rows": local_weight.shape[0],
        "local_weight_bytes": (
            local_weight.numel() * local_weight.element_size()
        ),
    }
    del full_weight
    del state
    gc.collect()
    torch.cuda.synchronize(device)
    return local_weight, descriptor


def _build_maintenance_workspaces_with_vocabulary(
    action_plan: D2ActionPlan,
    histories: dict[int, torch.Tensor],
    owner_map: dict[int, int],
    local_weight: torch.Tensor,
    *,
    num_embeddings: int,
    rank: int,
    world_size: int,
) -> tuple[
    dict[str, D2VectorExchangeWorkspace],
    dict[str, dict[str, object]],
]:
    workspaces = {}
    descriptors = {}
    for branch in ("mixed", "all_exact"):
        rank_requests = []
        rank_descriptors = []
        for requester in range(world_size):
            logical = build_d2_wave_embedding_logical_request(
                action_plan.records,
                histories,
                owner_map,
                branch=branch,
                rank=requester,
                world_size=world_size,
            )
            rank_requests.append(tuple(logical.item_ids.tolist()))
            rank_descriptors.append(
                {
                    "rank": requester,
                    "logical_tokens": logical.logical_tokens,
                    "logical_unique_tokens": (
                        logical.logical_unique_tokens
                    ),
                    "logical_remote_tokens": (
                        logical.logical_remote_tokens
                    ),
                    "logical_remote_unique_tokens": (
                        logical.logical_remote_unique_tokens
                    ),
                    "phase_token_counts": dict(
                        logical.phase_token_counts
                    ),
                }
            )
        compiled = compile_d2_embedding_capsule(
            tuple(rank_requests),
            num_embeddings=num_embeddings,
            world_size=world_size,
        )
        materialized = materialize_d2_embedding_capsule(
            compiled,
            rank,
            local_weight.device,
        )
        workspaces[branch] = build_d2_vector_exchange_workspace(
            materialized,
            local_weight,
            reconstruct_requested=False,
        )
        descriptors[branch] = {
            "rank_requests": rank_descriptors,
            "plan_compile_seconds": compiled.compile_seconds,
            "global_plan_bytes": compiled.plan_nbytes,
            "rank_plan_bytes": materialized.rank_plan_bytes,
            "materialized_rank_plan_bytes": (
                materialized.materialized_plan_bytes
            ),
            "timed_reconstruct_requested": False,
        }
    return workspaces, descriptors


def _build_foreground_workspaces(
    local_weight: torch.Tensor,
    *,
    num_embeddings: int,
    world_size: int,
    rank: int,
    batch_tokens: int,
    ring_size: int,
    seed: int,
) -> tuple[
    tuple[D2VectorExchangeWorkspace, ...],
    dict[str, object],
]:
    ring = build_d2_synthetic_foreground_request_ring(
        num_embeddings=num_embeddings,
        world_size=world_size,
        batch_tokens_per_rank=batch_tokens,
        ring_size=ring_size,
        seed=seed,
    )
    workspaces = []
    slot_descriptors = []
    for slot, requests in enumerate(ring):
        compiled = compile_d2_embedding_capsule(
            requests,
            num_embeddings=num_embeddings,
            world_size=world_size,
        )
        materialized = materialize_d2_embedding_capsule(
            compiled,
            rank,
            local_weight.device,
        )
        workspaces.append(
            build_d2_vector_exchange_workspace(
                materialized,
                local_weight,
                reconstruct_requested=True,
            )
        )
        slot_descriptors.append(
            {
                "slot": slot,
                "global_plan_bytes": compiled.plan_nbytes,
                "rank_plan_bytes": materialized.rank_plan_bytes,
                "materialized_rank_plan_bytes": (
                    materialized.materialized_plan_bytes
                ),
                "requested_tokens": materialized.requested_tokens,
                "unique_tokens": materialized.unique_tokens,
                "remote_unique_tokens": (
                    materialized.remote_unique_tokens
                ),
                "served_remote_unique_tokens": (
                    materialized.served_remote_unique_tokens
                ),
            }
        )
    return tuple(workspaces), {
        "request_source": (
            "deterministic synthetic fixed ring over real checkpoint "
            "vocabulary"
        ),
        "seed": seed,
        "ring_size": ring_size,
        "batch_tokens_per_rank": batch_tokens,
        "reconstruct_requested": True,
        "slots": slot_descriptors,
    }


def _sleep_until(target: float) -> None:
    while True:
        remaining = target - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.01))


def _foreground_worker(
    schedule: D2FixedRateSchedule,
    epoch: float,
    workspaces: tuple[D2VectorExchangeWorkspace, ...],
    local_weight: torch.Tensor,
    process_group: dist.ProcessGroup,
    stream: torch.cuda.Stream,
    coordinator: D2CollectiveLaunchCoordinator,
    result: dict[str, object],
) -> None:
    try:
        samples = []
        exchange_totals = {
            "vector_send_bytes": 0,
            "vector_receive_bytes": 0,
            "collective_calls": 0,
        }
        for sequence, release in enumerate(
            schedule.release_offsets_seconds
        ):
            _sleep_until(epoch + release)
            issued = time.perf_counter()
            exchange = execute_d2_vector_exchange(
                workspaces[sequence % len(workspaces)],
                local_weight,
                process_group=process_group,
                stream=stream,
                collective_launch_guard=lambda sequence=sequence: coordinator.phase(
                    "foreground",
                    sequence,
                ),
            )
            completed = time.perf_counter()
            samples.append(
                D2ForegroundSample(
                    sequence=sequence,
                    release_offset_seconds=release,
                    issue_offset_seconds=max(issued - epoch, release),
                    completion_offset_seconds=completed - epoch,
                    execution_wall_seconds=exchange.wall_seconds,
                    execution_device_seconds=exchange.device_seconds,
                )
            )
            exchange_totals["vector_send_bytes"] += (
                exchange.vector_send_bytes
            )
            exchange_totals["vector_receive_bytes"] += (
                exchange.vector_receive_bytes
            )
            exchange_totals["collective_calls"] += (
                exchange.collective_calls
            )
        checksum_tensor = next(
            workspace.requested_vectors
            for workspace in workspaces
            if workspace.requested_vectors is not None
        )
        result["samples"] = tuple(samples)
        result["exchange_totals"] = exchange_totals
        result["checksum"] = float(checksum_tensor[0, 0].item())
    except BaseException as error:
        result["error"] = repr(error)


def _run_embedding_maintenance(
    workspace: D2VectorExchangeWorkspace,
    local_weight: torch.Tensor,
    *,
    repeats: int,
    process_group: dist.ProcessGroup,
    stream: torch.cuda.Stream,
    coordinator: D2CollectiveLaunchCoordinator,
) -> tuple[D2VectorExchangeSample, ...]:
    return tuple(
        execute_d2_vector_exchange(
            workspace,
            local_weight,
            process_group=process_group,
            stream=stream,
            collective_launch_guard=lambda repeat=repeat: (
                coordinator.phase("maintenance", repeat)
            ),
        )
        for repeat in range(repeats)
    )


def _build_dense_control(
    local_weight: torch.Tensor,
    *,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=local_weight.device)
    generator.manual_seed(9173)
    left = torch.randn(
        (rows, local_weight.shape[1]),
        dtype=local_weight.dtype,
        device=local_weight.device,
        generator=generator,
    )
    right = torch.randn(
        (local_weight.shape[1], local_weight.shape[1]),
        dtype=local_weight.dtype,
        device=local_weight.device,
        generator=generator,
    )
    output = torch.empty_like(left)
    return left, right, output


@torch.inference_mode()
def _run_dense_control(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    iterations: int,
    stream: torch.cuda.Stream,
) -> dict[str, object]:
    left, right, output = tensors
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    with torch.cuda.stream(stream):
        start_event.record(stream)
        for _ in range(iterations):
            torch.mm(left, right, out=output)
        end_event.record(stream)
    end_event.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    device_seconds = start_event.elapsed_time(end_event) / 1000.0
    return {
        "iterations": iterations,
        "rows": left.shape[0],
        "hidden_size": left.shape[1],
        "estimated_flops": (
            2
            * iterations
            * left.shape[0]
            * left.shape[1]
            * right.shape[1]
        ),
        "tensor_resident_bytes": sum(
            value.numel() * value.element_size()
            for value in tensors
        ),
        "wall_seconds": wall_seconds,
        "device_seconds": device_seconds,
        "checksum": float(output[0, 0].item()),
    }


def _foreground_report(
    samples: tuple[D2ForegroundSample, ...],
    schedule: D2FixedRateSchedule,
    *,
    deadline_seconds: float,
    analysis_start: float,
    analysis_end: float,
    maintenance_start: float,
    maintenance_end: float,
) -> dict[str, object]:
    report = {
        "full_scenario": summarize_d2_foreground_samples(
            samples,
            schedule,
            deadline_seconds=deadline_seconds,
        ),
        "fixed_analysis_window": summarize_d2_foreground_samples(
            samples,
            schedule,
            deadline_seconds=deadline_seconds,
            window_start_seconds=analysis_start,
            window_end_seconds=analysis_end,
        ),
    }
    if maintenance_end > maintenance_start:
        overlap_end = min(maintenance_end, schedule.duration_seconds)
        if overlap_end > maintenance_start:
            report["actual_maintenance_overlap"] = (
                summarize_d2_foreground_samples(
                    samples,
                    schedule,
                    deadline_seconds=deadline_seconds,
                    window_start_seconds=maintenance_start,
                    window_end_seconds=overlap_end,
                )
            )
    return report


def _collective_launch_order(
    schedule: D2FixedRateSchedule,
    *,
    maintenance_start_seconds: float,
    maintenance_repeats: int,
    embedding_maintenance: bool,
) -> tuple[tuple[str, int], ...]:
    if not embedding_maintenance:
        return tuple(
            ("foreground", sequence)
            for sequence in range(schedule.request_count)
        )
    insertion = bisect.bisect_left(
        schedule.release_offsets_seconds,
        maintenance_start_seconds,
    )
    if insertion + maintenance_repeats > schedule.request_count:
        raise ValueError(
            "D2 maintenance launch order exceeds foreground schedule"
        )
    output = [
        ("foreground", sequence)
        for sequence in range(insertion)
    ]
    for repeat in range(maintenance_repeats):
        output.append(("maintenance", repeat))
        output.append(("foreground", insertion + repeat))
    output.extend(
        ("foreground", sequence)
        for sequence in range(
            insertion + maintenance_repeats,
            schedule.request_count,
        )
    )
    return tuple(output)


def _run_scenario(
    name: str,
    args: argparse.Namespace,
    schedule: D2FixedRateSchedule,
    foreground_workspaces: tuple[D2VectorExchangeWorkspace, ...],
    maintenance_workspaces: dict[
        str,
        D2VectorExchangeWorkspace,
    ],
    dense_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    local_weight: torch.Tensor,
    *,
    foreground_group: dist.ProcessGroup,
    maintenance_group: dist.ProcessGroup,
    foreground_stream: torch.cuda.Stream,
    maintenance_stream: torch.cuda.Stream,
) -> dict[str, object]:
    dist.barrier()
    embedding_maintenance = name in {
        "mixed_embedding",
        "all_exact_embedding",
    }
    launch_order = _collective_launch_order(
        schedule,
        maintenance_start_seconds=args.maintenance_start_seconds,
        maintenance_repeats=args.maintenance_repeats,
        embedding_maintenance=embedding_maintenance,
    )
    coordinator = D2CollectiveLaunchCoordinator(launch_order)
    epoch = time.perf_counter() + args.scenario_start_delay_seconds
    foreground_result: dict[str, object] = {}
    thread = threading.Thread(
        target=_foreground_worker,
        args=(
            schedule,
            epoch,
            foreground_workspaces,
            local_weight,
            foreground_group,
            foreground_stream,
            coordinator,
            foreground_result,
        ),
        daemon=False,
    )
    thread.start()
    _sleep_until(epoch + args.maintenance_start_seconds)
    maintenance_started = time.perf_counter()
    maintenance: dict[str, object]
    if name == "idle":
        maintenance = {
            "kind": "idle",
            "repeats": 0,
            "samples": [],
            "wall_seconds": 0.0,
        }
    elif name == "rank_local_dense_control":
        dense = _run_dense_control(
            dense_tensors,
            iterations=args.dense_iterations,
            stream=maintenance_stream,
        )
        maintenance = {
            "kind": "rank_local_dense_control",
            "repeats": args.dense_iterations,
            "samples": [],
            **dense,
        }
    else:
        branch = (
            "mixed"
            if name == "mixed_embedding"
            else "all_exact"
        )
        samples = _run_embedding_maintenance(
            maintenance_workspaces[branch],
            local_weight,
            repeats=args.maintenance_repeats,
            process_group=maintenance_group,
            stream=maintenance_stream,
            coordinator=coordinator,
        )
        maintenance = {
            "kind": "planned_unique_embedding_exchange",
            "branch": branch,
            "repeats": args.maintenance_repeats,
            "samples": [value.to_dict() for value in samples],
            "wall_seconds": sum(
                value.wall_seconds for value in samples
            ),
            "device_seconds": sum(
                value.device_seconds for value in samples
            ),
            "collective_device_seconds": sum(
                value.collective_device_seconds for value in samples
            ),
            "vector_send_bytes": sum(
                value.vector_send_bytes for value in samples
            ),
            "vector_receive_bytes": sum(
                value.vector_receive_bytes for value in samples
            ),
            "collective_calls": sum(
                value.collective_calls for value in samples
            ),
            "requested_tokens_per_repeat": (
                samples[0].requested_tokens
            ),
            "unique_tokens_per_repeat": samples[0].unique_tokens,
            "remote_unique_tokens_per_repeat": (
                samples[0].remote_unique_tokens
            ),
            "served_remote_unique_tokens_per_repeat": (
                samples[0].served_remote_unique_tokens
            ),
            "logical_inverse_reconstruction_timed": False,
        }
    maintenance_completed = time.perf_counter()
    thread.join()
    if "error" in foreground_result:
        raise RuntimeError(
            f"D2 foreground thread failed: {foreground_result['error']}"
        )
    samples = foreground_result["samples"]
    if not isinstance(samples, tuple):
        raise RuntimeError("D2 foreground samples are absent")
    coordinator.assert_complete()
    started_offset = maintenance_started - epoch
    completed_offset = maintenance_completed - epoch
    analysis_end = (
        args.maintenance_start_seconds
        + args.analysis_window_seconds
    )
    foreground = _foreground_report(
        samples,
        schedule,
        deadline_seconds=args.foreground_deadline_ms / 1000.0,
        analysis_start=args.maintenance_start_seconds,
        analysis_end=analysis_end,
        maintenance_start=started_offset,
        maintenance_end=completed_offset,
    )
    foreground["exchange_totals"] = foreground_result[
        "exchange_totals"
    ]
    foreground["checksum"] = foreground_result["checksum"]
    maintenance["start_offset_seconds"] = started_offset
    maintenance["end_offset_seconds"] = completed_offset
    maintenance["overran_foreground_schedule"] = (
        completed_offset > schedule.duration_seconds
    )
    dist.barrier()
    return {
        "scenario": name,
        "collective_launch_order": {
            "operation_count": len(launch_order),
            "embedding_maintenance_interleaved": (
                embedding_maintenance
            ),
            "maintenance_collectives": (
                args.maintenance_repeats
                if embedding_maintenance
                else 0
            ),
            "foreground_collectives": schedule.request_count,
        },
        "foreground": foreground,
        "maintenance": maintenance,
    }


def _warm_up(
    foreground_workspaces: tuple[D2VectorExchangeWorkspace, ...],
    maintenance_workspaces: dict[
        str,
        D2VectorExchangeWorkspace,
    ],
    dense_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    local_weight: torch.Tensor,
    *,
    foreground_group: dist.ProcessGroup,
    maintenance_group: dist.ProcessGroup,
    foreground_stream: torch.cuda.Stream,
    maintenance_stream: torch.cuda.Stream,
) -> None:
    for workspace in foreground_workspaces:
        execute_d2_vector_exchange(
            workspace,
            local_weight,
            process_group=foreground_group,
            stream=foreground_stream,
        )
    for branch in ("mixed", "all_exact"):
        execute_d2_vector_exchange(
            maintenance_workspaces[branch],
            local_weight,
            process_group=maintenance_group,
            stream=maintenance_stream,
        )
    _run_dense_control(
        dense_tensors,
        iterations=1,
        stream=maintenance_stream,
    )
    dist.barrier()


def _aggregate_foreground(
    ranks: tuple[dict[str, object], ...],
    section: str,
) -> dict[str, object]:
    values = tuple(
        value["foreground"][section] for value in ranks
    )
    completed = tuple(
        value
        for value in values
        if value["has_completed_requests"]
    )
    p50s = [
        value["response_p50_seconds"]
        for value in completed
        if value["response_p50_seconds"] is not None
    ]
    p99s = [
        value["response_p99_seconds"]
        for value in completed
        if value["response_p99_seconds"] is not None
    ]
    response_maxima = [
        value["response_max_seconds"]
        for value in completed
        if value["response_max_seconds"] is not None
    ]
    queue_p99s = [
        value["queue_p99_seconds"]
        for value in completed
        if value["queue_p99_seconds"] is not None
    ]
    queue_depths = [
        value["estimated_max_queue_depth_requests"]
        for value in completed
        if value["estimated_max_queue_depth_requests"] is not None
    ]
    deadline_misses = [
        value["deadline_miss_count"]
        for value in completed
        if value["deadline_miss_count"] is not None
    ]
    all_ranks_observed = len(completed) == len(values)
    any_rank_observed = bool(completed)
    observation_status = (
        "complete"
        if all_ranks_observed
        else (
            "partial_rank_coverage"
            if any_rank_observed
            else "no_completed_requests"
        )
    )
    return {
        "observation_status": observation_status,
        "no_completed_requests": not any_rank_observed,
        "ranks_with_completed_requests": len(completed),
        "ranks_without_completed_requests": (
            len(values) - len(completed)
        ),
        "queue_observation_status": (
            "observed_all_ranks"
            if all_ranks_observed
            else (
                "observed_partial_ranks"
                if any_rank_observed
                else "no_completed_requests"
            )
        ),
        "deadline_observation_status": (
            "observed_all_ranks"
            if all_ranks_observed
            else (
                "observed_partial_ranks"
                if any_rank_observed
                else "no_completed_requests"
            )
        ),
        "rank_summaries": list(values),
        "sum_rank_scheduled_requests": sum(
            value["scheduled_requests"] for value in values
        ),
        "sum_rank_completed_requests": sum(
            value["completed_requests"] for value in values
        ),
        "global_actual_offered_rate_per_second": sum(
            value["actual_offered_rate_per_second"]
            for value in values
        ),
        "global_achieved_rate_per_second": (
            sum(
                value["achieved_rate_per_second"]
                for value in completed
            )
            if completed
            else None
        ),
        "global_achieved_rate_is_partial": not all_ranks_observed,
        "worst_rank_response_p50_seconds": (
            max(p50s) if p50s else None
        ),
        "worst_rank_response_p99_seconds": (
            max(p99s) if p99s else None
        ),
        "worst_rank_response_max_seconds": (
            max(response_maxima) if response_maxima else None
        ),
        "sum_rank_deadline_miss_count": (
            sum(deadline_misses)
            if all_ranks_observed
            else None
        ),
        "observed_sum_rank_deadline_miss_count": (
            sum(deadline_misses) if deadline_misses else None
        ),
        "max_rank_estimated_queue_depth_requests": (
            max(queue_depths) if queue_depths else None
        ),
        "worst_rank_queue_p99_seconds": (
            max(queue_p99s) if queue_p99s else None
        ),
    }


def _aggregate_maintenance(
    ranks: tuple[dict[str, object], ...],
) -> dict[str, object]:
    values = tuple(value["maintenance"] for value in ranks)
    kind = values[0]["kind"]
    if any(value["kind"] != kind for value in values):
        raise RuntimeError("D2 maintenance kind differs across ranks")
    output = {
        "kind": kind,
        "rank_summaries": list(values),
        "max_rank_wall_seconds": max(
            value["wall_seconds"] for value in values
        ),
        "max_rank_end_offset_seconds": max(
            value["end_offset_seconds"] for value in values
        ),
        "any_rank_overran_foreground_schedule": any(
            value["overran_foreground_schedule"]
            for value in values
        ),
    }
    if kind == "planned_unique_embedding_exchange":
        send_bytes = sum(
            value["vector_send_bytes"] for value in values
        )
        receive_bytes = sum(
            value["vector_receive_bytes"] for value in values
        )
        output.update(
            {
                "branch": values[0]["branch"],
                "repeats": values[0]["repeats"],
                "off_diagonal_one_way_send_bytes": send_bytes,
                "off_diagonal_one_way_receive_bytes": receive_bytes,
                "off_diagonal_endpoint_bytes": (
                    send_bytes + receive_bytes
                ),
                "off_diagonal_send_receive_match": (
                    send_bytes == receive_bytes
                ),
                "sum_rank_collective_calls": sum(
                    value["collective_calls"] for value in values
                ),
                "max_rank_collective_device_seconds": max(
                    value["collective_device_seconds"]
                    for value in values
                ),
                "sum_rank_requested_tokens_per_repeat": sum(
                    value["requested_tokens_per_repeat"]
                    for value in values
                ),
                "sum_rank_unique_tokens_per_repeat": sum(
                    value["unique_tokens_per_repeat"]
                    for value in values
                ),
                "sum_rank_remote_unique_tokens_per_repeat": sum(
                    value["remote_unique_tokens_per_repeat"]
                    for value in values
                ),
            }
        )
    elif kind == "rank_local_dense_control":
        output.update(
            {
                "sum_rank_estimated_flops": sum(
                    value["estimated_flops"] for value in values
                ),
                "sum_rank_tensor_resident_bytes": sum(
                    value["tensor_resident_bytes"] for value in values
                ),
                "max_rank_device_seconds": max(
                    value["device_seconds"] for value in values
                ),
            }
        )
    return output


def _aggregate_scenarios(
    gathered: tuple[dict[str, object], ...],
    scenario_order: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    scenarios = {}
    checks = {}
    for scenario in scenario_order:
        ranks = tuple(
            value["scenarios"][scenario] for value in gathered
        )
        full = _aggregate_foreground(ranks, "full_scenario")
        analysis = _aggregate_foreground(
            ranks,
            "fixed_analysis_window",
        )
        maintenance = _aggregate_maintenance(ranks)
        scenario_output = {
            "foreground_full_scenario": full,
            "foreground_fixed_analysis_window": analysis,
            "maintenance": maintenance,
        }
        if all(
            "actual_maintenance_overlap" in value["foreground"]
            for value in ranks
        ):
            scenario_output["foreground_actual_maintenance_overlap"] = (
                _aggregate_foreground(
                    ranks,
                    "actual_maintenance_overlap",
                )
            )
        scenarios[scenario] = scenario_output
        checks[f"{scenario}_foreground_completed"] = (
            full["sum_rank_scheduled_requests"]
            == full["sum_rank_completed_requests"]
        )
        if maintenance["kind"] == "planned_unique_embedding_exchange":
            checks[f"{scenario}_byte_accounting_closes"] = (
                maintenance["off_diagonal_send_receive_match"]
            )
    idle = scenarios["idle"]["foreground_fixed_analysis_window"]
    for value in scenarios.values():
        foreground = value["foreground_fixed_analysis_window"]
        value["foreground_over_idle"] = {
            "worst_rank_p50_ratio": (
                foreground["worst_rank_response_p50_seconds"]
                / idle["worst_rank_response_p50_seconds"]
            ),
            "worst_rank_p99_ratio": (
                foreground["worst_rank_response_p99_seconds"]
                / idle["worst_rank_response_p99_seconds"]
            ),
            "global_achieved_rate_ratio": (
                foreground["global_achieved_rate_per_second"]
                / idle["global_achieved_rate_per_second"]
            ),
            "additional_deadline_misses": (
                foreground["sum_rank_deadline_miss_count"]
                - idle["sum_rank_deadline_miss_count"]
            ),
        }
    mixed = scenarios["mixed_embedding"]["maintenance"]
    exact = scenarios["all_exact_embedding"]["maintenance"]
    comparisons = {
        "all_exact_over_mixed_maintenance": {
            "one_way_vector_byte_ratio": (
                exact["off_diagonal_one_way_send_bytes"]
                / mixed["off_diagonal_one_way_send_bytes"]
            ),
            "max_rank_wall_ratio": (
                exact["max_rank_wall_seconds"]
                / mixed["max_rank_wall_seconds"]
            ),
            "unique_token_ratio": (
                exact["sum_rank_unique_tokens_per_repeat"]
                / mixed["sum_rank_unique_tokens_per_repeat"]
            ),
        },
        "foreground_interference": {
            scenario: value["foreground_over_idle"]
            for scenario, value in scenarios.items()
        },
    }
    checks["all_exact_has_more_planned_embedding_bytes"] = (
        exact["off_diagonal_one_way_send_bytes"]
        > mixed["off_diagonal_one_way_send_bytes"]
    )
    checks["all_passed"] = all(checks.values())
    return scenarios, {"comparisons": comparisons, "checks": checks}


def _validate_args(args: argparse.Namespace) -> None:
    scenarios = tuple(args.scenarios)
    analysis_end = (
        args.maintenance_start_seconds
        + args.analysis_window_seconds
    )
    if (
        scenarios != DEFAULT_SCENARIOS
        or tuple(args.expected_visible_devices) != _visible_devices()
        or len(args.expected_visible_devices) != 3
        or args.foreground_rate_per_rank <= 0
        or args.foreground_duration_seconds <= 0
        or args.foreground_batch_tokens < 1
        or args.foreground_ring_size < 1
        or args.foreground_seed < 0
        or args.foreground_deadline_ms <= 0
        or args.maintenance_start_seconds < 0
        or args.analysis_window_seconds <= 0
        or analysis_end > args.foreground_duration_seconds
        or args.maintenance_repeats < 1
        or args.dense_rows < 1
        or args.dense_iterations < 1
        or args.scenario_start_delay_seconds <= 0
    ):
        raise RuntimeError("D2 resource isolation configuration is invalid")


def _run_rank(
    args: argparse.Namespace,
    runtime,
    foreground_group: dist.ProcessGroup,
    maintenance_group: dist.ProcessGroup,
) -> tuple[
    dict[str, object],
    D2ActionPlan,
    Path,
    Path,
    str,
    dict[int, int],
    dict[str, object],
    dict[str, dict[str, object]],
]:
    _validate_args(args)
    if runtime.world_size != 3:
        raise RuntimeError("D2 resource isolation requires three ranks")
    output_path = _path(args.output)
    if output_path.exists():
        raise FileExistsError(
            "D2 resource isolation output exists; choose a new path"
        )
    action_plan = D2ActionPlan.load(_path(args.action_plan))
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
    local_weight, embedding_descriptor = _load_embedding_shard(
        checkpoint_path,
        checkpoint_descriptor,
        rank=runtime.rank,
        world_size=runtime.world_size,
        device=runtime.device,
    )
    num_embeddings = int(embedding_descriptor["num_embeddings"])
    maintenance_workspaces, maintenance_descriptors = (
        _build_maintenance_workspaces_with_vocabulary(
            action_plan,
            histories,
            owner_map,
            local_weight,
            num_embeddings=num_embeddings,
            rank=runtime.rank,
            world_size=runtime.world_size,
        )
    )
    foreground_workspaces, foreground_descriptor = (
        _build_foreground_workspaces(
            local_weight,
            num_embeddings=num_embeddings,
            world_size=runtime.world_size,
            rank=runtime.rank,
            batch_tokens=args.foreground_batch_tokens,
            ring_size=args.foreground_ring_size,
            seed=args.foreground_seed,
        )
    )
    dense_tensors = _build_dense_control(
        local_weight,
        rows=args.dense_rows,
    )
    torch.cuda.synchronize(runtime.device)
    foreground_stream = torch.cuda.Stream(device=runtime.device)
    maintenance_stream = torch.cuda.Stream(device=runtime.device)
    _warm_up(
        foreground_workspaces,
        maintenance_workspaces,
        dense_tensors,
        local_weight,
        foreground_group=foreground_group,
        maintenance_group=maintenance_group,
        foreground_stream=foreground_stream,
        maintenance_stream=maintenance_stream,
    )
    schedule = build_d2_fixed_rate_schedule(
        args.foreground_rate_per_rank,
        args.foreground_duration_seconds,
    )
    scenarios = {}
    for scenario in DEFAULT_SCENARIOS:
        scenarios[scenario] = _run_scenario(
            scenario,
            args,
            schedule,
            foreground_workspaces,
            maintenance_workspaces,
            dense_tensors,
            local_weight,
            foreground_group=foreground_group,
            maintenance_group=maintenance_group,
            foreground_stream=foreground_stream,
            maintenance_stream=maintenance_stream,
        )
    properties = torch.cuda.get_device_properties(runtime.device)
    local = {
        "rank": runtime.rank,
        "device": {
            "local_rank": runtime.local_rank,
            "logical_device": str(runtime.device),
            "name": properties.name,
            "uuid": f"GPU-{properties.uuid}",
            "total_memory_bytes": properties.total_memory,
        },
        "embedding": embedding_descriptor,
        "foreground_plan": foreground_descriptor,
        "maintenance_plans": maintenance_descriptors,
        "scenarios": scenarios,
    }
    return (
        local,
        action_plan,
        prepared_path,
        checkpoint_path,
        target_window_sha256,
        owner_map,
        foreground_descriptor,
        maintenance_descriptors,
    )


def _aggregate(
    args: argparse.Namespace,
    action_plan: D2ActionPlan,
    prepared_path: Path,
    checkpoint_path: Path,
    target_window_sha256: str,
    owner_map: dict[int, int],
    gathered: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if tuple(value["rank"] for value in gathered) != (0, 1, 2):
        raise RuntimeError("D2 resource isolation rank reports differ")
    scenarios, analysis = _aggregate_scenarios(
        gathered,
        DEFAULT_SCENARIOS,
    )
    result = {
        "protocol": D2_RESOURCE_ISOLATION_PROTOCOL,
        "status": (
            "complete"
            if analysis["checks"]["all_passed"]
            else "failed"
        ),
        "scientific_result": False,
        "development_design_validation": True,
        "formal_stage_c": False,
        "workload_scope": {
            "foreground": (
                "fixed-rate deterministic synthetic lookup stressor "
                "over the real checkpoint vocabulary"
            ),
            "not_claimed": [
                "production serving trace",
                "serving arrival distribution",
                "end-to-end recommendation latency",
                "foreground admission or QoS policy",
            ],
        },
        "timed_boundary": {
            "foreground": (
                "scheduled release through local shard gather, one "
                "preplanned vector all-to-all, requester reconstruction "
                "and stream completion"
            ),
            "maintenance": (
                "local shard gather, one preplanned unique-vector "
                "all-to-all and requester unique-capsule assembly per "
                "repeat"
            ),
            "maintenance_excluded": (
                "request manifest compilation, plan materialization, "
                "logical duplicate reconstruction, D1 K/V compute, "
                "transaction and publication"
            ),
            "dense_control": (
                "rank-local FP32 dense matrix multiplication on the "
                "maintenance CUDA stream with no collective"
            ),
            "process_startup_and_input_loading": "excluded",
        },
        "concurrency": {
            "default_group": (
                "scenario barriers and final metadata gather only"
            ),
            "foreground_group": (
                "dedicated all-rank NCCL process group"
            ),
            "maintenance_group": (
                "separate dedicated all-rank NCCL process group"
            ),
            "foreground_stream": "dedicated CUDA stream",
            "maintenance_stream": "separate dedicated CUDA stream",
            "device_wide_synchronize_inside_timed_exchange": False,
            "collective_order": (
                "a per-rank deterministic launch coordinator enforces "
                "the same cross-process-group NCCL launch order on every "
                "rank; maintenance and foreground launches alternate "
                "after the fixed insertion point"
            ),
            "maintenance_pacing": (
                "one maintenance launch is interleaved with one "
                "foreground launch for the configured repeat count; "
                "this is a deadlock-safe contention probe, not "
                "unconstrained maintenance saturation"
            ),
            "deadline_semantics": (
                "deadline misses are observed, never cancelled, so "
                "collective order remains intact"
            ),
        },
        "fairness": {
            "same_foreground_schedule_across_scenarios": True,
            "same_foreground_request_ring_across_scenarios": True,
            "same_local_embedding_shard_storage": True,
            "same_fixed_analysis_window_across_scenarios": True,
            "embedding_branch_semantics": (
                "mixed and all-exact use the frozen H12 action-plan "
                "request sets with requester-scope wave uniqueness"
            ),
            "maintenance_repeats_fixed_across_embedding_branches": True,
            "dense_control_matching": (
                "fixed FLOP work, not duration- or byte-matched; its "
                "measured active wall and overlap window are reported"
            ),
            "scenario_order": list(DEFAULT_SCENARIOS),
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
            "foreground_rate_per_rank": (
                args.foreground_rate_per_rank
            ),
            "foreground_duration_seconds": (
                args.foreground_duration_seconds
            ),
            "foreground_batch_tokens": (
                args.foreground_batch_tokens
            ),
            "foreground_ring_size": args.foreground_ring_size,
            "foreground_seed": args.foreground_seed,
            "foreground_deadline_ms": args.foreground_deadline_ms,
            "maintenance_start_seconds": (
                args.maintenance_start_seconds
            ),
            "analysis_window_seconds": args.analysis_window_seconds,
            "maintenance_repeats": args.maintenance_repeats,
            "dense_rows": args.dense_rows,
            "dense_iterations": args.dense_iterations,
        },
        "rank_inputs": [
            {
                "rank": value["rank"],
                "device": value["device"],
                "embedding": value["embedding"],
                "foreground_plan": value["foreground_plan"],
                "maintenance_plans": value["maintenance_plans"],
            }
            for value in gathered
        ],
        "scenarios": scenarios,
        "comparisons": analysis["comparisons"],
        "checks": analysis["checks"],
        "unsupported_claims": [
            "real serving workload QoS",
            "causal production interference",
            "end-to-end D1 or D2 speedup",
            "NCCL wire bytes",
            "formal Stage-C completion",
        ],
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def run(args: argparse.Namespace) -> dict[str, object] | None:
    runtime = init_d2_distributed_runtime(
        backend="nccl",
        timeout_seconds=args.timeout_seconds,
    )
    foreground_group = None
    maintenance_group = None
    try:
        ranks = list(range(runtime.world_size))
        foreground_group = dist.new_group(ranks=ranks, backend="nccl")
        maintenance_group = dist.new_group(ranks=ranks, backend="nccl")
        (
            local,
            action_plan,
            prepared_path,
            checkpoint_path,
            target_window_sha256,
            owner_map,
            _,
            _,
        ) = _run_rank(
            args,
            runtime,
            foreground_group,
            maintenance_group,
        )
        dist.barrier()
        gathered: list[object] = [None] * runtime.world_size
        dist.all_gather_object(gathered, local)
        output = None
        if runtime.is_primary:
            if not all(isinstance(value, dict) for value in gathered):
                raise RuntimeError(
                    "D2 resource isolation rank report is invalid"
                )
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
                    "D2 resource isolation structural checks failed: "
                    f"{output['checks']}"
                )
            output_path = _path(args.output)
            if output_path.exists():
                raise FileExistsError(
                    "D2 resource isolation output appeared"
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
        if maintenance_group is not None:
            dist.destroy_process_group(maintenance_group)
        if foreground_group is not None:
            dist.destroy_process_group(foreground_group)
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
                    "comparisons": output["comparisons"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
