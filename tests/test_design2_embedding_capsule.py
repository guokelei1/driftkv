from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hstu_kvcache.migration.design2_embedding import (
    ModuloRowShardedEmbedding,
)
from hstu_kvcache.migration.design2_embedding_capsule import (
    D2EmbeddingCapsulePlan,
    D2EmbeddingCapsuleRankPlan,
    compile_d2_embedding_capsule,
    execute_d2_embedding_capsule,
    materialize_d2_embedding_capsule,
)


def _reference_compile(
    requester_item_ids: tuple[tuple[int, ...], ...],
    num_embeddings: int,
    world_size: int,
) -> D2EmbeddingCapsulePlan:
    unique_item_ids = tuple(
        tuple(sorted(set(values))) for values in requester_item_ids
    )
    inverse_slots = []
    local_rows = []
    local_slots = []
    send_rows = [
        [[] for _ in range(world_size)] for _ in range(world_size)
    ]
    receive_slots = [
        [[] for _ in range(world_size)] for _ in range(world_size)
    ]
    for requester in range(world_size):
        slot_by_item = {
            item_id: slot
            for slot, item_id in enumerate(unique_item_ids[requester])
        }
        inverse_slots.append(
            tuple(
                slot_by_item[value]
                for value in requester_item_ids[requester]
            )
        )
        requester_local_rows = []
        requester_local_slots = []
        for slot, item_id in enumerate(unique_item_ids[requester]):
            owner = item_id % world_size
            row = item_id // world_size
            if owner == requester:
                requester_local_rows.append(row)
                requester_local_slots.append(slot)
            else:
                send_rows[owner][requester].append(row)
                receive_slots[requester][owner].append(slot)
        local_rows.append(tuple(requester_local_rows))
        local_slots.append(tuple(requester_local_slots))
    ranks = tuple(
        D2EmbeddingCapsuleRankPlan(
            rank=rank,
            world_size=world_size,
            unique_item_ids=unique_item_ids[rank],
            inverse_slots=inverse_slots[rank],
            local_rows=local_rows[rank],
            local_capsule_slots=local_slots[rank],
            send_local_rows_by_requester=tuple(
                tuple(values) for values in send_rows[rank]
            ),
            receive_capsule_slots_by_owner=tuple(
                tuple(values) for values in receive_slots[rank]
            ),
        )
        for rank in range(world_size)
    )
    return D2EmbeddingCapsulePlan(
        num_embeddings=num_embeddings,
        world_size=world_size,
        ranks=ranks,
        compile_seconds=0.0,
    )


@pytest.mark.parametrize(
    ("requester_item_ids", "num_embeddings", "world_size"),
    (
        (((3, 1, 3, 0, 1),), 7, 1),
        (((1, 1, 2, 5, 8, 2), ()), 9, 2),
        (
            (
                (14, 0, 11, 14, 6, 3, 11),
                (1, 13, 4, 7, 1, 10),
                (2, 5, 8, 2, 14, 0),
            ),
            15,
            3,
        ),
        (
            tuple(
                tuple(
                    torch.randint(
                        0,
                        97,
                        (1000,),
                        generator=torch.Generator().manual_seed(rank),
                    ).tolist()
                )
                for rank in range(4)
            ),
            97,
            4,
        ),
        (((999999, 4, 4), (7,)), 1_000_000, 2),
    ),
)
def test_vectorized_compiler_matches_reference_plan(
    requester_item_ids: tuple[tuple[int, ...], ...],
    num_embeddings: int,
    world_size: int,
) -> None:
    expected = _reference_compile(
        requester_item_ids,
        num_embeddings,
        world_size,
    )
    actual = compile_d2_embedding_capsule(
        requester_item_ids,
        num_embeddings,
        world_size,
    )
    assert actual.ranks == expected.ranks
    assert actual.plan_nbytes == expected.plan_nbytes
    assert actual.protocol == expected.protocol
    D2EmbeddingCapsulePlan(
        num_embeddings=actual.num_embeddings,
        world_size=actual.world_size,
        ranks=actual.ranks,
        compile_seconds=actual.compile_seconds,
    )


def test_world_one_capsule_matches_dynamic_lookup_bitwise() -> None:
    weight = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    requested = (3, 1, 3, 0, 1)
    plan = compile_d2_embedding_capsule(
        (requested,),
        num_embeddings=7,
        world_size=1,
    )
    dynamic = ModuloRowShardedEmbedding(
        local_weight=weight,
        num_embeddings=7,
        rank=0,
        world_size=1,
    ).lookup(
        torch.tensor([requested]),
        torch.tensor([len(requested)]),
    )
    materialized = materialize_d2_embedding_capsule(plan, 0, "cpu")
    inverse_pointer = materialized.inverse_slots.data_ptr()
    capsule = execute_d2_embedding_capsule(materialized, weight)
    repeated = execute_d2_embedding_capsule(materialized, weight)
    assert torch.equal(
        capsule.item_vectors,
        dynamic.item_vectors.reshape(-1, 5),
    )
    assert plan.ranks[0].unique_item_ids == (0, 1, 3)
    assert plan.ranks[0].inverse_slots == (2, 1, 2, 0, 1)
    assert plan.ranks[0].send_splits == (0,)
    assert plan.ranks[0].receive_splits == (0,)
    assert plan.plan_nbytes > 0
    assert plan.compile_seconds >= 0
    assert materialized.materialized_plan_bytes > 0
    assert materialized.inverse_slots.data_ptr() == inverse_pointer
    assert torch.equal(repeated.item_vectors, capsule.item_vectors)
    assert capsule.metrics.requested_tokens == 5
    assert capsule.metrics.unique_tokens == 3
    assert capsule.metrics.local_unique_tokens == 3
    assert capsule.metrics.remote_unique_tokens == 0
    assert capsule.metrics.counts_collective_bytes == 0
    assert capsule.metrics.id_collective_bytes == 0
    assert capsule.metrics.vector_collective_payload_bytes == 0
    assert capsule.metrics.off_diagonal_bytes == 0
    assert capsule.metrics.collective_calls == 0
    assert (
        capsule.metrics.execution_seconds
        >= capsule.metrics.collective_seconds
    )
    assert capsule.metrics.global_plan_bytes == plan.plan_nbytes
    assert (
        capsule.metrics.materialized_plan_bytes
        == materialized.materialized_plan_bytes
    )
    assert (
        capsule.metrics.plan_compile_seconds
        == plan.compile_seconds
    )
    assert (
        capsule.metrics.plan_materialization_seconds
        == materialized.materialization_seconds
    )


def _world_two_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        full_weight = torch.arange(
            36,
            dtype=torch.float32,
        ).reshape(9, 4)
        local_weight = full_weight[rank::2].contiguous()
        requests = ((1, 1, 2, 5, 8, 2), ())
        local_ids = requests[rank]
        plan = compile_d2_embedding_capsule(
            requests,
            num_embeddings=9,
            world_size=2,
        )
        dynamic = ModuloRowShardedEmbedding(
            local_weight=local_weight,
            num_embeddings=9,
            rank=rank,
            world_size=2,
        ).lookup(
            torch.tensor([local_ids], dtype=torch.long)
            if local_ids
            else torch.empty((0, 0), dtype=torch.long),
            torch.tensor([len(local_ids)], dtype=torch.long)
            if local_ids
            else torch.empty(0, dtype=torch.long),
        )
        materialized = materialize_d2_embedding_capsule(
            plan,
            rank,
            "cpu",
        )
        capsule = execute_d2_embedding_capsule(
            materialized,
            local_weight,
        )
        assert torch.equal(
            capsule.item_vectors,
            dynamic.item_vectors.reshape(-1, 4),
        )
        assert capsule.metrics.collective_calls == 1
        assert (
            capsule.metrics.execution_seconds
            >= capsule.metrics.collective_seconds
        )
        assert capsule.metrics.counts_collective_bytes == 0
        assert capsule.metrics.id_collective_bytes == 0
        if rank == 0:
            assert plan.ranks[0].unique_item_ids == (1, 2, 5, 8)
            assert plan.ranks[0].inverse_slots == (0, 0, 1, 2, 3, 1)
            assert plan.ranks[0].receive_splits == (0, 2)
            assert capsule.metrics.requested_tokens == 6
            assert capsule.metrics.unique_tokens == 4
            assert capsule.metrics.local_unique_tokens == 2
            assert capsule.metrics.remote_unique_tokens == 2
            assert capsule.metrics.served_remote_unique_tokens == 0
            assert capsule.metrics.vector_collective_input_bytes == 0
            assert capsule.metrics.vector_collective_output_bytes == 32
        else:
            assert plan.ranks[1].unique_item_ids == ()
            assert plan.ranks[1].inverse_slots == ()
            assert plan.ranks[1].send_splits == (2, 0)
            assert capsule.item_vectors.shape == (0, 4)
            assert capsule.metrics.requested_tokens == 0
            assert capsule.metrics.unique_tokens == 0
            assert capsule.metrics.remote_unique_tokens == 0
            assert capsule.metrics.served_remote_unique_tokens == 2
            assert capsule.metrics.vector_collective_input_bytes == 32
            assert capsule.metrics.vector_collective_output_bytes == 0
    finally:
        dist.destroy_process_group()


def test_world_two_asymmetric_empty_requester_matches_dynamic_lookup(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _world_two_worker,
        args=(str(tmp_path / "capsule-gloo"),),
        nprocs=2,
        join=True,
    )


def test_capsule_compiler_rejects_invalid_request_graph() -> None:
    with pytest.raises(ValueError):
        compile_d2_embedding_capsule(
            ((0,),),
            num_embeddings=4,
            world_size=2,
        )
    with pytest.raises(ValueError):
        compile_d2_embedding_capsule(
            ((0, 4), ()),
            num_embeddings=4,
            world_size=2,
        )
