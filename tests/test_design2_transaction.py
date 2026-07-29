import hashlib

import torch.distributed as dist

from hstu_kvcache.migration.design2_distributed import (
    D2CollectiveGuard,
    broadcast_d2_metadata,
    capture_d2_preflight_failures,
    d2_distributed_runtime,
    d2_file_init_method,
    gather_d2_rank_metadata,
    vote_d2_preflight,
)
from hstu_kvcache.migration.design2_transaction import (
    D2RankCapacity,
    D2RankFragmentMetadata,
    validate_d2_private_fragments,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _trace():
    from hstu_kvcache.migration.design2_distributed import D2CollectiveStep

    return (
        D2CollectiveStep(0, "preflight_vote"),
        D2CollectiveStep(1, "compiled_retained"),
        D2CollectiveStep(2, "exact_and_append"),
        D2CollectiveStep(3, "fragment_ready"),
    )


def _metadata(
    rank: int,
    world_size: int,
    record_ids: tuple[int, ...],
) -> D2RankFragmentMetadata:
    capacity = D2RankCapacity(
        required_bytes=100 + rank,
        capacity_bytes=1_000,
        measured_peak_bytes=120 + rank,
    )
    if not record_ids:
        return D2RankFragmentMetadata.empty(
            action_plan_sha256=_sha("plan"),
            target_version="theta2",
            rank=rank,
            world_size=world_size,
            phase_trace=_trace(),
            capacity=capacity,
        )
    return D2RankFragmentMetadata(
        action_plan_sha256=_sha("plan"),
        target_version="theta2",
        rank=rank,
        world_size=world_size,
        owner_record_ids=record_ids,
        fragment_sha256=_sha(f"fragment-{rank}"),
        payload_bytes=512 * len(record_ids),
        phase_trace=_trace(),
        capacity=capacity,
    )


def test_world_size_one_gloo_runtime_is_always_initialized(
    tmp_path,
    monkeypatch,
) -> None:
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    with d2_distributed_runtime(
        backend="gloo",
        init_method=d2_file_init_method(tmp_path / "d2-store"),
        rank=0,
        world_size=1,
        local_rank=0,
        device="cpu",
        timeout_seconds=10,
    ) as runtime:
        assert dist.is_initialized()
        assert runtime.world_size == 1
        assert runtime.owns_process_group
        guard = D2CollectiveGuard(
            runtime,
            (
                "preflight_vote",
                "rank_metadata_gather",
                "metadata_broadcast",
            ),
        )
        failures = capture_d2_preflight_failures(
            {
                "shape": lambda: True,
                "capacity": lambda: None,
            }
        )
        decision = vote_d2_preflight(
            runtime,
            failures,
            guard=guard,
        )
        assert decision.passed
        gathered = gather_d2_rank_metadata(
            runtime,
            {"rank": runtime.rank, "empty": True},
            guard=guard,
        )
        assert gathered == ({"rank": 0, "empty": True},)
        broadcast = broadcast_d2_metadata(
            runtime,
            {"transaction": "private-only"},
            guard=guard,
        )
        assert broadcast == {"transaction": "private-only"}
        assert [value.token for value in guard.require_complete()] == [
            "0000:preflight_vote",
            "0001:rank_metadata_gather",
            "0002:metadata_broadcast",
        ]
    assert not dist.is_initialized()


def test_private_fragments_validate_global_owner_coverage_with_empty_rank() -> None:
    metadata = (
        _metadata(0, 3, (0, 1)),
        _metadata(1, 3, (2,)),
        _metadata(2, 3, ()),
    )
    decision = validate_d2_private_fragments(
        action_plan_sha256=_sha("plan"),
        target_version="theta2",
        world_size=3,
        record_owner_map={0: 0, 1: 0, 2: 1},
        rank_metadata=metadata,
        expected_phase_trace=_trace(),
    )
    assert decision.ready
    assert decision.status == "ready"
    assert decision.ready_ranks == (0, 1, 2)
    assert decision.covered_record_ids == (0, 1, 2)
    assert decision.total_fragment_bytes == 1_536
    assert not decision.publishes_target_epoch
    assert metadata[2].ready
    assert metadata[2].payload_bytes == 0


def test_private_fragment_decision_is_deterministic() -> None:
    metadata = (
        _metadata(0, 2, (0,)),
        _metadata(1, 2, (1,)),
    )
    arguments = {
        "action_plan_sha256": _sha("plan"),
        "target_version": "theta2",
        "world_size": 2,
        "record_owner_map": {0: 0, 1: 1},
        "rank_metadata": metadata,
    }
    first = validate_d2_private_fragments(**arguments)
    second = validate_d2_private_fragments(**arguments)
    assert first == second
    assert first.fragment_set_sha256 == second.fragment_set_sha256
    assert first.to_dict()["publishes_target_epoch"] is False
