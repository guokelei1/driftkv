from __future__ import annotations

import runpy
import threading
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hstu_kvcache.migration.design2_embedding_capsule import (
    compile_d2_embedding_capsule,
    materialize_d2_embedding_capsule,
)
from hstu_kvcache.migration.design2_resource_isolation import (
    D2CollectiveLaunchCoordinator,
    D2ForegroundSample,
    build_d2_fixed_rate_schedule,
    build_d2_synthetic_foreground_request_ring,
    build_d2_vector_exchange_workspace,
    execute_d2_vector_exchange,
    summarize_d2_foreground_samples,
)


def test_collective_launch_coordinator_enforces_cross_thread_order() -> None:
    coordinator = D2CollectiveLaunchCoordinator(
        (
            ("foreground", 0),
            ("maintenance", 0),
            ("foreground", 1),
            ("maintenance", 1),
        )
    )
    observed = []

    def foreground() -> None:
        for ordinal in range(2):
            with coordinator.phase("foreground", ordinal):
                observed.append(("foreground", ordinal))

    def maintenance() -> None:
        for ordinal in range(2):
            with coordinator.phase("maintenance", ordinal):
                observed.append(("maintenance", ordinal))

    maintenance_thread = threading.Thread(target=maintenance)
    foreground_thread = threading.Thread(target=foreground)
    maintenance_thread.start()
    time.sleep(0.01)
    foreground_thread.start()
    foreground_thread.join()
    maintenance_thread.join()
    coordinator.assert_complete()
    assert tuple(observed) == coordinator.order


def test_fixed_rate_summary_preserves_queue_and_deadline_semantics() -> None:
    schedule = build_d2_fixed_rate_schedule(4.0, 1.0)
    samples = tuple(
        D2ForegroundSample(
            sequence=index,
            release_offset_seconds=release,
            issue_offset_seconds=release + queue,
            completion_offset_seconds=release + queue + service,
            execution_wall_seconds=service,
            execution_device_seconds=service * 0.8,
        )
        for index, (release, queue, service) in enumerate(
            zip(
                schedule.release_offsets_seconds,
                (0.0, 0.01, 0.30, 0.5),
                (0.05, 0.05, 0.10, 0.10),
                strict=True,
            )
        )
    )
    summary = summarize_d2_foreground_samples(
        samples,
        schedule,
        deadline_seconds=0.5,
        window_start_seconds=0.25,
        window_end_seconds=1.0,
    )
    assert summary["scheduled_requests"] == 3
    assert summary["completed_requests"] == 3
    assert summary["deadline_miss_count"] == 1
    assert summary["estimated_max_queue_depth_requests"] == 1
    assert summary["response_p50_seconds"] == pytest.approx(0.4)
    assert summary["response_p99_seconds"] == pytest.approx(0.596)
    assert summary["achieved_rate_per_second"] < 4.0


def test_empty_observation_window_is_explicit_and_aggregate_safe() -> None:
    namespace = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "scripts"
            / "benchmark_cohortkv_design2_resource_isolation.py"
        )
    )
    aggregate_foreground = namespace["_aggregate_foreground"]

    schedule = build_d2_fixed_rate_schedule(2.0, 1.0)
    samples = tuple(
        D2ForegroundSample(
            sequence=index,
            release_offset_seconds=release,
            issue_offset_seconds=release,
            completion_offset_seconds=release + 0.01,
            execution_wall_seconds=0.01,
            execution_device_seconds=0.008,
        )
        for index, release in enumerate(
            schedule.release_offsets_seconds
        )
    )
    empty = summarize_d2_foreground_samples(
        samples,
        schedule,
        deadline_seconds=0.02,
        window_start_seconds=0.1,
        window_end_seconds=0.2,
    )
    assert empty["observation_status"] == (
        "no_scheduled_requests_in_window"
    )
    assert empty["no_completed_requests"] is True
    assert empty["achieved_rate_per_second"] is None
    assert empty["deadline_miss_count"] is None
    assert empty["deadline_miss_fraction"] is None
    assert empty["positive_queue_count"] is None
    assert empty["estimated_max_queue_depth_requests"] is None
    assert empty["response_p50_seconds"] is None
    aggregate = aggregate_foreground(
        tuple(
            {
                "foreground": {
                    "actual_maintenance_overlap": empty,
                }
            }
            for _ in range(3)
        ),
        "actual_maintenance_overlap",
    )
    assert aggregate["observation_status"] == "no_completed_requests"
    assert aggregate["ranks_with_completed_requests"] == 0
    assert aggregate["ranks_without_completed_requests"] == 3
    assert aggregate["global_achieved_rate_per_second"] is None
    assert aggregate["worst_rank_response_p50_seconds"] is None
    assert aggregate["worst_rank_queue_p99_seconds"] is None
    assert aggregate["sum_rank_deadline_miss_count"] is None


def test_synthetic_ring_is_valid_and_has_remote_pressure() -> None:
    ring = build_d2_synthetic_foreground_request_ring(
        num_embeddings=101,
        world_size=3,
        batch_tokens_per_rank=12,
        ring_size=4,
        seed=17,
    )
    assert len(ring) == 4
    for requesters in ring:
        assert len(requesters) == 3
        for rank, ids in enumerate(requesters):
            assert len(ids) == 12
            assert all(0 <= value < 101 for value in ids)
            assert sum(value % 3 != rank for value in ids) == 8


def test_world_one_vector_exchange_matches_requested_order() -> None:
    weight = torch.arange(36, dtype=torch.float32).reshape(9, 4)
    requests = ((5, 1, 5, 8, 0),)
    plan = compile_d2_embedding_capsule(
        requests,
        num_embeddings=9,
        world_size=1,
    )
    materialized = materialize_d2_embedding_capsule(
        plan,
        0,
        "cpu",
    )
    workspace = build_d2_vector_exchange_workspace(
        materialized,
        weight,
        reconstruct_requested=True,
    )
    sample = execute_d2_vector_exchange(
        workspace,
        weight,
        process_group=None,
    )
    assert workspace.requested_vectors is not None
    assert torch.equal(
        workspace.requested_vectors,
        weight.index_select(0, torch.tensor(requests[0])),
    )
    assert sample.requested_tokens == 5
    assert sample.unique_tokens == 4
    assert sample.collective_calls == 0
    assert sample.vector_endpoint_bytes == 0


def _world_two_exchange_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    foreground_group = dist.new_group(
        ranks=[0, 1],
        backend="gloo",
    )
    maintenance_group = dist.new_group(
        ranks=[0, 1],
        backend="gloo",
    )
    try:
        full = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        local = full[rank::2].contiguous()
        requests = ((1, 2, 9, 1), (0, 7, 10))
        plan = compile_d2_embedding_capsule(
            requests,
            num_embeddings=11,
            world_size=2,
        )
        materialized = materialize_d2_embedding_capsule(
            plan,
            rank,
            "cpu",
        )
        workspace = build_d2_vector_exchange_workspace(
            materialized,
            local,
            reconstruct_requested=True,
        )
        sample = execute_d2_vector_exchange(
            workspace,
            local,
            process_group=foreground_group,
        )
        assert workspace.requested_vectors is not None
        assert torch.equal(
            workspace.requested_vectors,
            full.index_select(
                0,
                torch.tensor(requests[rank]),
            ),
        )
        assert sample.collective_calls == 1
        assert sample.vector_send_bytes >= 0
        assert sample.vector_receive_bytes >= 0
        foreground_workspace = build_d2_vector_exchange_workspace(
            materialized,
            local,
            reconstruct_requested=True,
        )
        maintenance_workspace = build_d2_vector_exchange_workspace(
            materialized,
            local,
            reconstruct_requested=False,
        )
        coordinator = D2CollectiveLaunchCoordinator(
            (
                ("foreground", 0),
                ("maintenance", 0),
                ("foreground", 1),
                ("maintenance", 1),
            )
        )
        errors = []

        def run_foreground() -> None:
            try:
                for ordinal in range(2):
                    execute_d2_vector_exchange(
                        foreground_workspace,
                        local,
                        process_group=foreground_group,
                        collective_launch_guard=(
                            lambda ordinal=ordinal: coordinator.phase(
                                "foreground",
                                ordinal,
                            )
                        ),
                    )
            except BaseException as error:
                errors.append(error)

        def run_maintenance() -> None:
            try:
                for ordinal in range(2):
                    execute_d2_vector_exchange(
                        maintenance_workspace,
                        local,
                        process_group=maintenance_group,
                        collective_launch_guard=(
                            lambda ordinal=ordinal: coordinator.phase(
                                "maintenance",
                                ordinal,
                            )
                        ),
                    )
            except BaseException as error:
                errors.append(error)

        foreground_thread = threading.Thread(target=run_foreground)
        maintenance_thread = threading.Thread(target=run_maintenance)
        foreground_thread.start()
        maintenance_thread.start()
        foreground_thread.join()
        maintenance_thread.join()
        assert not errors
        coordinator.assert_complete()
    finally:
        dist.destroy_process_group(maintenance_group)
        dist.destroy_process_group(foreground_group)
        dist.destroy_process_group()


def test_explicit_group_vector_exchange_matches_full_embedding(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _world_two_exchange_worker,
        args=(str(tmp_path / "isolation-gloo"),),
        nprocs=2,
        join=True,
    )
