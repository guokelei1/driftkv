from __future__ import annotations

import numpy as np

from hstu_kvcache.migration.xp_exact_baseline import XPBaselineRecord
from hstu_kvcache.migration.xp_m2_lookup_baseline import (
    account_lookup_requests,
    build_lookup_requests,
    record_extents,
    select_retained_budget,
)


def _record(
    record_id: int,
    owner_rank: int,
    retained: int,
    evicted: int = 0,
) -> XPBaselineRecord:
    items = np.arange(1, evicted + retained + 33, dtype=np.int64)
    return XPBaselineRecord(
        record_id=record_id,
        user_id=record_id,
        owner_rank=owner_rank,
        item_ids=items,
        old_start=0,
        old_length=evicted + retained,
        target_start=evicted,
        target_length=retained + 32,
        old_valid_bytes=(evicted + retained) * 16,
        target_valid_bytes=(retained + 32) * 16,
    )


def test_natural_extents_separate_retained_and_common_append() -> None:
    record = _record(0, 0, 480, 32)
    retained, append = record_extents(record)
    assert len(retained) == 480
    assert len(append) == 32
    assert np.array_equal(record.items("old")[32:], retained)
    assert np.array_equal(record.items("target"), np.concatenate((retained, append)))


def test_shape_stratified_token_budget_is_deterministic_and_nested() -> None:
    records = tuple(
        _record(index, index % 2, retained)
        for index, retained in enumerate(
            (64, 80, 96, 127, 128, 160, 224, 255, 256, 300, 383, 384, 420, 480)
        )
    )
    low = select_retained_budget(records, 0.2, "fixture")
    repeat = select_retained_budget(records, 0.2, "fixture")
    middle = select_retained_budget(records, 0.5, "fixture")
    full = select_retained_budget(records, 1.0, "fixture")
    assert low == repeat
    assert set(low.selected_record_ids).issubset(middle.selected_record_ids)
    assert set(middle.selected_record_ids).issubset(full.selected_record_ids)
    assert full.selected_retained_tokens == sum(
        record.target_length - 32 for record in records
    )
    assert {value["name"] for value in low.strata} == {
        "64_127",
        "128_255",
        "256_383",
        "384_480",
    }


def test_accounting_separates_retained_append_and_remote_payloads() -> None:
    records = (_record(0, 0, 64), _record(1, 1, 128))
    selected = select_retained_budget(records, 1.0, "fixture")
    retained = build_lookup_requests(
        records,
        "retained",
        selected.selected_record_ids,
    )
    append = build_lookup_requests(records, "append")
    retained_counts = account_lookup_requests(
        retained,
        world_size=2,
        hidden_size=1536,
    )
    append_counts = account_lookup_requests(
        append,
        world_size=2,
        hidden_size=1536,
    )
    assert retained_counts["requested_tokens"] == 192
    assert append_counts["requested_tokens"] == 64
    assert append_counts["remote_tokens"] == 32
    assert append_counts["id_request_bytes"] == 32 * 8
    assert append_counts["h1536_fp32_response_bytes"] == 32 * 1536 * 4
