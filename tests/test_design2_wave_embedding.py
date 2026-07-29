import torch
import torch.distributed as dist

from hstu_kvcache.migration.design2_embedding import (
    ModuloRowShardedEmbedding,
)
from hstu_kvcache.migration.design2_embedding_capsule import (
    compile_d2_embedding_capsule,
    execute_d2_embedding_capsule,
    materialize_d2_embedding_capsule,
)
from hstu_kvcache.migration.design2_plan import D2ActionRecord
from hstu_kvcache.migration.design2_wave_embedding import (
    D2_WAVE_EMBEDDING_LOOKUP_KERNEL,
    D2WaveEmbeddingLogicalRequest,
    build_d2_wave_embedding_logical_request,
    build_d2_wave_embedding_lookup_plan,
    d2_wave_embedding_demand_calls,
    execute_d2_wave_embedding_lookup_plan,
    summarize_d2_wave_embedding_execution,
)


def _record(
    record_id: int,
    requested_action: str,
    requested_reason: str,
    retained_tokens: int,
    delta_tokens: int,
) -> D2ActionRecord:
    digest = f"{record_id + 1:064x}"
    present = requested_reason != "natural_exact"
    old_tokens = retained_tokens if present else 0
    return D2ActionRecord(
        record_id=record_id,
        prepared_user_id=record_id + 1,
        requested_action=requested_action,
        requested_reason=requested_reason,
        old_tokens=old_tokens,
        retained_start=0,
        retained_tokens=retained_tokens,
        delta_start=retained_tokens,
        delta_tokens=delta_tokens,
        target_prefix_tokens=retained_tokens + delta_tokens,
        latest_tokens=1,
        final_tokens=retained_tokens + delta_tokens + 1,
        last_exact_version="theta1" if present else None,
        migration_depth=0,
        previous_cache_expected=present,
        previous_cache_present=present,
        old_history_sha256=digest if present else None,
        target_history_sha256=digest,
        retained_identity_sha256=digest,
        delta_identity_sha256=digest,
        target_prefix_identity_sha256=digest,
    )


def _records() -> tuple[D2ActionRecord, ...]:
    return (
        _record(0, "compiled", "migrate", 2, 2),
        _record(1, "exact", "scheduled_exact", 2, 1),
        _record(2, "exact", "natural_exact", 0, 3),
    )


def test_build_branch_requests_preserves_phase_and_demand_order() -> None:
    histories = {
        0: (1, 2, 3, 4, 5),
        1: (2, 3, 4, 6),
        2: (7, 8, 9, 10),
    }
    owner_map = {0: 0, 1: 0, 2: 1}
    mixed_rank0 = build_d2_wave_embedding_logical_request(
        _records(),
        histories,
        owner_map,
        branch="mixed",
        rank=0,
        world_size=2,
    )
    assert mixed_rank0.item_ids.tolist() == [2, 3, 3, 4, 4, 5, 6]
    assert dict(mixed_rank0.phase_token_counts) == {
        "scheduled_exact_retained": 2,
        "natural_exact_target_prefix": 0,
        "delta_append": 3,
        "latest_append": 2,
    }
    assert mixed_rank0.logical_tokens == 7
    assert mixed_rank0.logical_unique_tokens == 5
    assert mixed_rank0.logical_remote_tokens == 3
    assert mixed_rank0.logical_remote_unique_tokens == 2

    all_exact_rank0 = build_d2_wave_embedding_logical_request(
        _records(),
        histories,
        owner_map,
        branch="all_exact",
        rank=0,
        world_size=2,
    )
    assert all_exact_rank0.item_ids.tolist() == [
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        6,
    ]
    assert dict(all_exact_rank0.phase_token_counts) == {
        "all_exact_retained": 4,
        "natural_exact_target_prefix": 0,
        "delta_append": 3,
        "latest_append": 2,
    }

    mixed_rank1 = build_d2_wave_embedding_logical_request(
        _records(),
        histories,
        owner_map,
        branch="mixed",
        rank=1,
        world_size=2,
    )
    assert mixed_rank1.item_ids.tolist() == [7, 8, 9, 10]
    assert dict(mixed_rank1.phase_token_counts) == {
        "scheduled_exact_retained": 0,
        "natural_exact_target_prefix": 3,
        "delta_append": 0,
        "latest_append": 1,
    }


def test_lookup_plans_pad_demand_and_reconstruct_wave_unique() -> None:
    request = D2WaveEmbeddingLogicalRequest(
        branch="mixed",
        rank=0,
        world_size=1,
        item_ids=torch.tensor([5, 2, 5, 3, 2, 7, 7]),
        phase_token_counts=(("wave", 7),),
    )
    assert d2_wave_embedding_demand_calls(7, 3) == 3
    demand = build_d2_wave_embedding_lookup_plan(
        request,
        mode="demand_token_microbatch",
        token_microbatch=3,
        lookup_calls=4,
    )
    assert [value.numel() for value in demand.lookup_batches] == [
        3,
        3,
        1,
        0,
    ]
    one_batch = build_d2_wave_embedding_lookup_plan(
        request,
        mode="one_batch_no_dedup",
        token_microbatch=3,
    )
    assert one_batch.lookup_calls == 1
    assert one_batch.inverse is None
    cached = build_d2_wave_embedding_lookup_plan(
        request,
        mode="wave_scope_unique_cache",
        token_microbatch=3,
    )
    assert cached.lookup_batches[0].tolist() == [2, 3, 5, 7]
    assert cached.inverse is not None
    assert torch.equal(
        cached.lookup_batches[0].index_select(0, cached.inverse),
        request.item_ids,
    )
    assert cached.cache_item_id_bytes == 32
    assert cached.cache_vector_bytes(4) == 64
    assert cached.inverse_bytes == 56


def test_world_one_execution_is_bitwise_across_all_modes(
    tmp_path,
) -> None:
    rendezvous = tmp_path / "wave-embedding-gloo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        weight = torch.arange(48, dtype=torch.float32).reshape(12, 4)
        embedding = ModuloRowShardedEmbedding(
            local_weight=weight,
            num_embeddings=12,
            rank=0,
            world_size=1,
        )
        request = D2WaveEmbeddingLogicalRequest(
            branch="mixed",
            rank=0,
            world_size=1,
            item_ids=torch.tensor([5, 2, 5, 3, 2, 7, 7]),
            phase_token_counts=(("wave", 7),),
        )
        executions = {}
        plans = {}
        for mode in (
            "one_batch_no_dedup",
            "demand_token_microbatch",
            "wave_scope_unique_cache",
        ):
            plans[mode] = build_d2_wave_embedding_lookup_plan(
                request,
                mode=mode,
                token_microbatch=3,
                lookup_calls=(
                    4 if mode == "demand_token_microbatch" else 1
                ),
            )
            executions[mode] = execute_d2_wave_embedding_lookup_plan(
                embedding,
                plans[mode],
            )
        reference = executions["one_batch_no_dedup"].item_vectors
        expected = weight.index_select(0, request.item_ids)
        assert torch.equal(reference, expected)
        assert torch.equal(
            executions["demand_token_microbatch"].item_vectors,
            reference,
        )
        assert torch.equal(
            executions["wave_scope_unique_cache"].item_vectors,
            reference,
        )
        assert all(
            not hasattr(metric, "requested_ids_sha256")
            for execution in executions.values()
            for metric in execution.lookup_metrics
        )
        demand_summary = summarize_d2_wave_embedding_execution(
            plans["demand_token_microbatch"],
            executions["demand_token_microbatch"],
            hidden_size=4,
        )
        assert demand_summary["logical_tokens"] == 7
        assert demand_summary["lookup_requested_tokens"] == 7
        assert demand_summary["lookup_calls"] == 4
        assert demand_summary[
            "actual_collective_tensor_payload_bytes"
        ] == 0
        assert demand_summary["off_diagonal_send_bytes"] == 0
        assert demand_summary["off_diagonal_receive_bytes"] == 0
        assert demand_summary["off_diagonal_bytes"] == 0
        assert demand_summary["off_diagonal_collective_seconds"] == 0.0
        assert (
            demand_summary["lookup_kernel"]
            == D2_WAVE_EMBEDDING_LOOKUP_KERNEL
        )
        assert demand_summary["timed_payload_hashing"] is False
        assert demand_summary["timing_cuda_synchronized"] is True
        cached_summary = summarize_d2_wave_embedding_execution(
            plans["wave_scope_unique_cache"],
            executions["wave_scope_unique_cache"],
            hidden_size=4,
        )
        assert cached_summary["lookup_requested_tokens"] == 4
        assert cached_summary["cache_item_id_bytes"] == 32
        assert cached_summary["cache_vector_bytes"] == 64
        assert cached_summary["inverse_bytes"] == 56
        capsule_plan = compile_d2_embedding_capsule(
            (tuple(request.item_ids.tolist()),),
            num_embeddings=12,
            world_size=1,
        )
        materialized = materialize_d2_embedding_capsule(
            capsule_plan,
            0,
            "cpu",
        )
        capsule = execute_d2_embedding_capsule(
            materialized,
            embedding.local_weight,
        )
        assert torch.equal(capsule.item_vectors, reference)
        assert capsule.metrics.requested_tokens == 7
        assert capsule.metrics.unique_tokens == 4
        assert capsule.metrics.counts_collective_bytes == 0
        assert capsule.metrics.id_collective_bytes == 0
        assert capsule.metrics.collective_calls == 0
    finally:
        dist.destroy_process_group()


def test_invalid_history_and_lookup_call_count_fail() -> None:
    try:
        build_d2_wave_embedding_logical_request(
            _records(),
            {0: (1,), 1: (1, 2, 3, 4), 2: (1, 2, 3, 4)},
            {0: 0, 1: 0, 2: 0},
            branch="mixed",
            rank=0,
            world_size=1,
        )
    except ValueError as error:
        assert "history" in str(error)
    else:
        raise AssertionError("invalid target history did not fail")
    request = D2WaveEmbeddingLogicalRequest(
        branch="mixed",
        rank=0,
        world_size=1,
        item_ids=torch.arange(7),
        phase_token_counts=(("wave", 7),),
    )
    try:
        build_d2_wave_embedding_lookup_plan(
            request,
            mode="demand_token_microbatch",
            token_microbatch=3,
            lookup_calls=2,
        )
    except ValueError as error:
        assert "drop requests" in str(error)
    else:
        raise AssertionError("short collective plan did not fail")
