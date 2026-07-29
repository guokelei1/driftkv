import hashlib
from dataclasses import replace

import pytest
import torch.distributed as dist

from hstu_kvcache.migration.design2_distributed import (
    D2CollectiveGuard,
    D2CollectiveStep,
    capture_d2_preflight_failures,
    d2_distributed_runtime,
    d2_file_init_method,
    vote_d2_preflight,
)
from hstu_kvcache.migration.design2_transaction import (
    D2RankCapacity,
    D2RankFragmentMetadata,
    validate_d2_private_fragments,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _trace(suffix: str = "fragment_ready") -> tuple[D2CollectiveStep, ...]:
    return (
        D2CollectiveStep(0, "preflight_vote"),
        D2CollectiveStep(1, suffix),
    )


def _fragment(
    rank: int,
    records: tuple[int, ...],
    *,
    capacity: D2RankCapacity | None = None,
) -> D2RankFragmentMetadata:
    return D2RankFragmentMetadata(
        action_plan_sha256=_sha("plan"),
        target_version="theta2",
        rank=rank,
        world_size=2,
        owner_record_ids=records,
        fragment_sha256=_sha(f"fragment-{rank}"),
        payload_bytes=256 * len(records),
        phase_trace=_trace(),
        capacity=capacity
        or D2RankCapacity(
            required_bytes=100,
            capacity_bytes=1_000,
        ),
    )


def test_cooperative_preflight_converts_local_exceptions_to_abort_votes(
    tmp_path,
    monkeypatch,
) -> None:
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    with d2_distributed_runtime(
        backend="gloo",
        init_method=d2_file_init_method(tmp_path / "d2-fault-store"),
        rank=0,
        world_size=1,
        local_rank=0,
        device="cpu",
        timeout_seconds=10,
    ) as runtime:
        guard = D2CollectiveGuard(runtime, ("preflight_vote",))

        def fail() -> None:
            raise RuntimeError("missing old extent")

        failures = capture_d2_preflight_failures(
            {
                "artifact": lambda: True,
                "old_extent": fail,
                "capacity": lambda: False,
            }
        )
        decision = vote_d2_preflight(
            runtime,
            failures,
            guard=guard,
        )
        assert not decision.passed
        assert len(decision.failure_reasons) == 2
        assert "missing old extent" in decision.failure_reasons[0]
        assert "returned false" in decision.failure_reasons[1]
        guard.require_complete()
    assert not dist.is_initialized()


def test_collective_guard_rejects_wrong_phase_without_advancing_trace(
    tmp_path,
    monkeypatch,
) -> None:
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    with d2_distributed_runtime(
        backend="gloo",
        init_method=d2_file_init_method(tmp_path / "d2-order-store"),
        rank=0,
        world_size=1,
        local_rank=0,
        device="cpu",
        timeout_seconds=10,
    ) as runtime:
        guard = D2CollectiveGuard(runtime, ("compiled_retained",))
        with pytest.raises(RuntimeError, match="order mismatch"):
            guard.enter("exact_prefix")
        assert guard.trace == ()
        guard.enter("compiled_retained")
        guard.require_complete()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate record coverage"),
        ("missing", "missing record coverage"),
        ("owner", "owner mismatch"),
        ("hash", "action plan hash mismatch"),
        ("target", "target version mismatch"),
        ("capacity", "capacity rejected"),
        ("phase", "phase traces differ"),
        ("synthetic", "synthetic: injected mid-exact"),
    ],
)
def test_private_fragment_faults_abort_without_publication(
    mutation: str,
    message: str,
) -> None:
    first = _fragment(0, (0,))
    second = _fragment(1, (1,))
    owner_map = {0: 0, 1: 1}
    if mutation == "duplicate":
        second = replace(second, owner_record_ids=(0, 1), payload_bytes=512)
    elif mutation == "missing":
        second = D2RankFragmentMetadata.empty(
            action_plan_sha256=_sha("plan"),
            target_version="theta2",
            rank=1,
            world_size=2,
            phase_trace=_trace(),
            capacity=D2RankCapacity(100, 1_000),
        )
    elif mutation == "owner":
        owner_map = {0: 1, 1: 0}
    elif mutation == "hash":
        second = replace(second, action_plan_sha256=_sha("other"))
    elif mutation == "target":
        second = replace(second, target_version="theta3")
    elif mutation == "capacity":
        second = replace(
            second,
            capacity=D2RankCapacity(
                required_bytes=1_001,
                capacity_bytes=1_000,
            ),
        )
    elif mutation == "phase":
        second = replace(second, phase_trace=_trace("different"))
    elif mutation == "synthetic":
        second = second.with_synthetic_failure("injected mid-exact")
    decision = validate_d2_private_fragments(
        action_plan_sha256=_sha("plan"),
        target_version="theta2",
        world_size=2,
        record_owner_map=owner_map,
        rank_metadata=(first, second),
    )
    assert decision.status == "abort"
    assert not decision.ready
    assert not decision.publishes_target_epoch
    assert any(
        message in reason for reason in decision.failure_reasons
    )


def test_missing_rank_metadata_is_an_abort_not_a_partial_ready() -> None:
    decision = validate_d2_private_fragments(
        action_plan_sha256=_sha("plan"),
        target_version="theta2",
        world_size=2,
        record_owner_map={0: 0, 1: 1},
        rank_metadata=(_fragment(0, (0,)),),
    )
    assert decision.status == "abort"
    assert not decision.publishes_target_epoch
    assert any(
        "rank metadata coverage mismatch" in reason
        for reason in decision.failure_reasons
    )
