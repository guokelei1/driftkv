import hashlib
from dataclasses import replace

import pytest

from hstu_kvcache.migration.design2_dev_epoch import (
    D2_DEV_EPOCH_SCOPE,
    D2DevEpochPointer,
    D2DevEpochRecord,
    D2DevEpochSpec,
    D2DevEpochStateMachine,
    D2DevPrivatePrepare,
    D2DevReadbackAck,
    validate_d2_dev_prepares,
    validate_d2_dev_readbacks,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _record(
    record_id: int,
    owner_rank: int,
    token_length: int,
    action: str,
) -> D2DevEpochRecord:
    return D2DevEpochRecord(
        record_id=record_id,
        owner_rank=owner_rank,
        token_length=token_length,
        action=action,
        lineage_sha256=_sha(f"lineage-{record_id}"),
        payload_sha256=_sha(f"payload-{record_id}"),
    )


def _spec() -> D2DevEpochSpec:
    return D2DevEpochSpec(
        action_plan_sha256=_sha("action-plan"),
        source_version="theta1",
        target_version="theta2",
        source_epoch=11,
        target_epoch=12,
        world_size=3,
        records=(
            _record(10, 0, 17, "compiled"),
            _record(11, 1, 23, "scheduled_exact"),
            _record(12, 0, 31, "natural_exact"),
        ),
    )


def _source_pointer() -> D2DevEpochPointer:
    return D2DevEpochPointer(
        version="theta1",
        epoch=11,
        certificate_sha256=_sha("source-certificate"),
    )


def _records_by_rank(
    spec: D2DevEpochSpec,
) -> tuple[tuple[D2DevEpochRecord, ...], ...]:
    return tuple(
        tuple(
            value
            for value in spec.records
            if value.owner_rank == rank
        )
        for rank in range(spec.world_size)
    )


def _prepares(
    spec: D2DevEpochSpec,
) -> tuple[D2DevPrivatePrepare, ...]:
    records = _records_by_rank(spec)
    return tuple(
        D2DevPrivatePrepare.create(
            spec=spec,
            rank=rank,
            records=records[rank],
        )
        for rank in range(spec.world_size)
    )


def _readbacks(
    spec: D2DevEpochSpec,
    commit,
) -> tuple[D2DevReadbackAck, ...]:
    records = _records_by_rank(spec)
    return tuple(
        D2DevReadbackAck.create(
            spec=spec,
            commit=commit,
            rank=rank,
            records=records[rank],
        )
        for rank in range(spec.world_size)
    )


def test_w3_empty_rank_stays_private_until_readback_publication() -> None:
    spec = _spec()
    source = _source_pointer()
    machine = D2DevEpochStateMachine(spec, source)
    assert machine.state == "private_prepare"
    assert machine.visible_pointer == source
    assert not machine.target_visible
    with pytest.raises(KeyError, match="not globally visible"):
        machine.resolve_visible("theta2")

    prepares = _prepares(spec)
    assert prepares[2].records == ()
    commit = machine.decide_prepares(tuple(reversed(prepares)))
    assert commit.committed
    assert commit.status == "commit"
    assert commit.prepared_ranks == (0, 1, 2)
    assert commit.covered_record_ids == (10, 11, 12)
    assert not commit.publishes_target_epoch
    assert machine.state == "awaiting_readback"
    assert machine.visible_pointer == source
    assert not machine.target_visible
    with pytest.raises(KeyError, match="not globally visible"):
        machine.resolve_visible("theta2")

    readbacks = _readbacks(spec, commit)
    assert readbacks[2].records == ()
    publication = machine.validate_readbacks(tuple(reversed(readbacks)))
    assert publication.status == "published"
    assert publication.publishes_target_epoch
    assert publication.acknowledged_ranks == (0, 1, 2)
    assert publication.scope == D2_DEV_EPOCH_SCOPE
    assert publication.scientific_result is False
    assert publication.formal_stage_c is False
    assert publication.timing_included is False
    assert machine.state == "published"
    assert machine.target_visible
    assert machine.visible_pointer.version == "theta2"
    assert machine.visible_pointer.epoch == 12
    assert (
        machine.visible_pointer.certificate_sha256
        == publication.certificate_sha256
    )
    assert machine.resolve_visible("theta2") == machine.visible_pointer
    with pytest.raises(KeyError, match="not globally visible"):
        machine.resolve_visible("theta1")


def test_commit_certificate_is_deterministic_across_arrival_order() -> None:
    spec = _spec()
    prepares = _prepares(spec)
    first = validate_d2_dev_prepares(spec, prepares)
    second = validate_d2_dev_prepares(spec, tuple(reversed(prepares)))
    assert first == second
    assert first.certificate_sha256 == second.certificate_sha256
    payload = first.to_dict()
    assert payload["scope"] == D2_DEV_EPOCH_SCOPE
    assert payload["scientific_result"] is False
    assert payload["formal_stage_c"] is False
    assert payload["timing_included"] is False
    assert "elapsed_seconds" not in payload


def _replace_rank_records(
    spec: D2DevEpochSpec,
    prepares: tuple[D2DevPrivatePrepare, ...],
    rank: int,
    records: tuple[D2DevEpochRecord, ...],
    failure_reason: str | None = None,
) -> tuple[D2DevPrivatePrepare, ...]:
    values = list(prepares)
    values[rank] = D2DevPrivatePrepare.create(
        spec=spec,
        rank=rank,
        records=records,
        failure_reason=failure_reason,
    )
    return tuple(values)


@pytest.mark.parametrize(
    "case",
    (
        "missing_rank",
        "duplicate_rank",
        "wrong_owner",
        "wrong_length",
        "wrong_action",
        "wrong_lineage",
        "wrong_payload_checksum",
        "wrong_fragment_checksum",
        "rank_fault",
    ),
)
def test_prepare_validation_aborts_every_invalid_global_epoch(
    case: str,
) -> None:
    spec = _spec()
    prepares = _prepares(spec)
    if case == "missing_rank":
        candidate = prepares[:2]
    elif case == "duplicate_rank":
        candidate = prepares + (prepares[0],)
    elif case == "wrong_fragment_checksum":
        candidate = (
            replace(prepares[0], fragment_sha256=_sha("wrong-fragment")),
            *prepares[1:],
        )
    elif case == "rank_fault":
        candidate = _replace_rank_records(
            spec,
            prepares,
            1,
            prepares[1].records,
            failure_reason="synthetic worker fault",
        )
    else:
        original = prepares[0].records[0]
        if case == "wrong_owner":
            changed = replace(original, owner_rank=1)
        elif case == "wrong_length":
            changed = replace(original, token_length=18)
        elif case == "wrong_action":
            changed = replace(original, action="natural_exact")
        elif case == "wrong_lineage":
            changed = replace(
                original,
                lineage_sha256=_sha("wrong-lineage"),
            )
        else:
            changed = replace(
                original,
                payload_sha256=_sha("wrong-payload"),
            )
        candidate = _replace_rank_records(
            spec,
            prepares,
            0,
            (changed, prepares[0].records[1]),
        )
    machine = D2DevEpochStateMachine(spec, _source_pointer())
    decision = machine.decide_prepares(candidate)
    assert decision.status == "abort"
    assert decision.failure_reasons
    assert not decision.publishes_target_epoch
    assert machine.state == "aborted"
    assert machine.visible_pointer.version == "theta1"
    assert not machine.target_visible
    with pytest.raises(KeyError, match="not globally visible"):
        machine.resolve_visible("theta2")
    with pytest.raises(RuntimeError, match="cannot publish"):
        machine.validate_readbacks(())


def _replace_readback_records(
    spec: D2DevEpochSpec,
    commit,
    readbacks: tuple[D2DevReadbackAck, ...],
    rank: int,
    records: tuple[D2DevEpochRecord, ...],
    failure_reason: str | None = None,
) -> tuple[D2DevReadbackAck, ...]:
    values = list(readbacks)
    values[rank] = D2DevReadbackAck.create(
        spec=spec,
        commit=commit,
        rank=rank,
        records=records,
        failure_reason=failure_reason,
    )
    return tuple(values)


@pytest.mark.parametrize(
    "case",
    (
        "missing_rank",
        "duplicate_rank",
        "wrong_owner",
        "wrong_length",
        "wrong_action",
        "wrong_lineage",
        "wrong_payload_checksum",
        "wrong_ack_checksum",
        "readback_fault",
    ),
)
def test_readback_validation_never_publishes_invalid_target(
    case: str,
) -> None:
    spec = _spec()
    machine = D2DevEpochStateMachine(spec, _source_pointer())
    commit = machine.decide_prepares(_prepares(spec))
    readbacks = _readbacks(spec, commit)
    if case == "missing_rank":
        candidate = readbacks[:2]
    elif case == "duplicate_rank":
        candidate = readbacks + (readbacks[0],)
    elif case == "wrong_ack_checksum":
        candidate = (
            replace(readbacks[0], ack_sha256=_sha("wrong-ack")),
            *readbacks[1:],
        )
    elif case == "readback_fault":
        candidate = _replace_readback_records(
            spec,
            commit,
            readbacks,
            1,
            readbacks[1].records,
            failure_reason="synthetic readback fault",
        )
    else:
        original = readbacks[0].records[0]
        if case == "wrong_owner":
            changed = replace(original, owner_rank=1)
        elif case == "wrong_length":
            changed = replace(original, token_length=18)
        elif case == "wrong_action":
            changed = replace(original, action="natural_exact")
        elif case == "wrong_lineage":
            changed = replace(
                original,
                lineage_sha256=_sha("wrong-readback-lineage"),
            )
        else:
            changed = replace(
                original,
                payload_sha256=_sha("wrong-readback-payload"),
            )
        candidate = _replace_readback_records(
            spec,
            commit,
            readbacks,
            0,
            (changed, readbacks[0].records[1]),
        )
    publication = machine.validate_readbacks(candidate)
    assert publication.status == "abort"
    assert publication.failure_reasons
    assert not publication.publishes_target_epoch
    assert machine.state == "aborted"
    assert machine.visible_pointer.version == "theta1"
    assert not machine.target_visible
    with pytest.raises(KeyError, match="not globally visible"):
        machine.resolve_visible("theta2")


def test_state_machine_rejects_out_of_order_or_repeated_decisions() -> None:
    spec = _spec()
    machine = D2DevEpochStateMachine(spec, _source_pointer())
    with pytest.raises(RuntimeError, match="precedes"):
        machine.validate_readbacks(())
    commit = machine.decide_prepares(_prepares(spec))
    with pytest.raises(RuntimeError, match="already exists"):
        machine.decide_prepares(_prepares(spec))
    publication = machine.validate_readbacks(_readbacks(spec, commit))
    assert publication.publishes_target_epoch
    with pytest.raises(RuntimeError, match="already exists"):
        machine.validate_readbacks(_readbacks(spec, commit))


def test_source_pointer_must_match_epoch_specification() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="differs"):
        D2DevEpochStateMachine(
            spec,
            replace(_source_pointer(), epoch=10),
        )
    commit = validate_d2_dev_prepares(spec, _prepares(spec))
    with pytest.raises(ValueError, match="commit is invalid"):
        validate_d2_dev_readbacks(
            spec,
            commit,
            replace(_source_pointer(), epoch=10),
            _readbacks(spec, commit),
        )


def test_record_rejects_empty_final_cache_and_unknown_action() -> None:
    with pytest.raises(ValueError, match="record is invalid"):
        _record(10, 0, 0, "compiled")
    with pytest.raises(ValueError, match="record is invalid"):
        _record(10, 0, 1, "recompute")
