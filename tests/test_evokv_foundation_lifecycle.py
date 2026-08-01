from __future__ import annotations

from dataclasses import replace

import pytest

from hstu_kvcache.migration.foundation_lifecycle import (
    FoundationGroupSpec,
    FoundationRecordSpec,
    FoundationRollingLifecycle,
    deterministic_extent_payload,
)


def _records(*, padded: bool = False) -> tuple[FoundationRecordSpec, ...]:
    lengths = ((7, 9), (11, 13), (15, 17), (19, 19))
    return tuple(
        FoundationRecordSpec(
            record_id=index + 1,
            route="action_plan_overlay_pending",
            source_version="theta0",
            target_version="theta1",
            old_valid_tokens=old,
            old_allocated_tokens=32 if padded else old,
            target_valid_tokens=target,
            target_allocated_tokens=32 if padded else target,
        )
        for index, (old, target) in enumerate(lengths)
    )


def _lifecycle(
    records: tuple[FoundationRecordSpec, ...],
    *,
    bytes_per_token: int = 4,
) -> FoundationRollingLifecycle:
    payloads = {
        record.record_id: deterministic_extent_payload(
            record.record_id,
            record.source_version,
            record.old_allocated_tokens * bytes_per_token,
        )
        for record in records
    }
    max_allocated = max(
        record.target_allocated_tokens * bytes_per_token
        for record in records
    )
    return FoundationRollingLifecycle(
        records,
        payloads,
        bytes_per_token=bytes_per_token,
        shadow_capacity_bytes=max_allocated,
        staging_capacity_bytes=max_allocated,
        workspace_capacity_bytes=16,
    )


def _group(
    ordinal: int,
    record: FoundationRecordSpec,
    *,
    bytes_per_token: int = 4,
) -> FoundationGroupSpec:
    return FoundationGroupSpec(
        group_id=f"group-{ordinal}",
        records=(record,),
        staging_bytes=record.target_allocated_tokens * bytes_per_token,
        workspace_bytes=8,
    )


def test_rolling_groups_validate_commit_reclaim_and_reuse_shadow() -> None:
    records = _records()
    lifecycle = _lifecycle(records)
    last = None
    for ordinal, record in enumerate(records):
        group = _group(ordinal, record)
        payload = deterministic_extent_payload(
            record.record_id,
            record.target_version,
            record.target_allocated_tokens * 4,
        )
        target = lifecycle.prepare_target(record.record_id, payload)
        receipt = lifecycle.execute_group(group, (target,))
        assert receipt.status == "committed"
        assert lifecycle.state(record.record_id)["version"] == "theta1"
        last = (group, target)
    assert last is not None
    replay = lifecycle.execute_group(last[0], (last[1],))
    assert replay.status == "already_committed"
    ledger = lifecycle.ledger()
    assert ledger["scientific_result"] is False
    assert ledger["executes_d1_d2_numeric"] is False
    assert ledger["groups_committed"] == 4
    assert ledger["groups_failed"] == 0
    assert ledger["idempotent_group_replays"] == 1
    assert ledger["coverage"]["complete"]
    assert ledger["coverage"]["exactly_once"]
    assert ledger["capacity"]["shadow_reuse_groups"] == 3
    assert (
        ledger["capacity"]["peak_shadow_allocated_bytes"]
        == max(record.target_allocated_tokens for record in records) * 4
    )
    assert (
        ledger["capacity"]["sum_group_shadow_allocated_bytes"]
        == sum(record.target_allocated_tokens for record in records) * 4
    )
    assert (
        ledger["transaction"]["reclaimed_old_allocated_bytes"]
        == sum(record.old_allocated_tokens for record in records) * 4
    )


def test_failed_full_payload_validation_does_not_publish() -> None:
    record = _records()[0]
    lifecycle = _lifecycle((record,))
    group = _group(0, record)
    payload = deterministic_extent_payload(
        record.record_id,
        record.target_version,
        record.target_allocated_tokens * 4,
    )
    target = lifecycle.prepare_target(record.record_id, payload)
    corrupt = replace(target, payload=target.payload[:-1])
    before = lifecycle.state(record.record_id)
    with pytest.raises(ValueError, match="payload, length"):
        lifecycle.execute_group(group, (corrupt,))
    assert lifecycle.state(record.record_id) == before
    ledger = lifecycle.ledger()
    assert ledger["groups_committed"] == 0
    assert ledger["groups_failed"] == 1
    assert ledger["coverage"]["pending_record_ids"] == [record.record_id]
    assert (
        ledger["transaction"]["discarded_failed_shadow_bytes"]
        == record.target_allocated_tokens * 4
    )


def test_group_coverage_and_capacity_failures_are_not_committed() -> None:
    records = _records()[:2]
    lifecycle = _lifecycle(records)
    group = FoundationGroupSpec(
        group_id="too-large",
        records=records,
        staging_bytes=1,
        workspace_bytes=1,
    )
    outputs = tuple(
        lifecycle.prepare_target(
            record.record_id,
            deterministic_extent_payload(
                record.record_id,
                record.target_version,
                record.target_allocated_tokens * 4,
            ),
        )
        for record in records
    )
    with pytest.raises(MemoryError, match="shadow capacity"):
        lifecycle.execute_group(group, outputs)
    assert lifecycle.ledger()["groups_committed"] == 0

    one = _group(1, records[0])
    with pytest.raises(ValueError, match="coverage"):
        lifecycle.execute_group(one, ())
    assert lifecycle.state(records[0].record_id)["version"] == "theta0"


def test_hom_tracks_valid_and_allocated_bytes_separately() -> None:
    records = _records(padded=True)
    lifecycle = _lifecycle(records)
    initial = lifecycle.ledger()["capacity"]
    assert initial["initial_live_valid_bytes"] < (
        initial["initial_live_allocated_bytes"]
    )
    for ordinal, record in enumerate(records):
        payload = deterministic_extent_payload(
            record.record_id,
            record.target_version,
            record.target_allocated_tokens * 4,
        )
        lifecycle.execute_group(
            _group(ordinal, record),
            (lifecycle.prepare_target(record.record_id, payload),),
        )
    ledger = lifecycle.ledger()
    assert (
        ledger["capacity"]["final_live_allocated_bytes"]
        == ledger["capacity"]["initial_live_allocated_bytes"]
    )
    assert (
        ledger["capacity"]["final_live_valid_bytes"]
        > ledger["capacity"]["initial_live_valid_bytes"]
    )
