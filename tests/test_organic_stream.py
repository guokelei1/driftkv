import json

import pandas as pd
import pytest

from hstu_kvcache.data import KuaiRandTrace, StreamingDataPlan
from hstu_kvcache.streaming import (
    ORGANIC_STREAM_PROTOCOL,
    OrganicStreamSnapshot,
    build_4plus12_causal_snapshots,
    stable_event_identity,
)

DATES = [f"202204{day:02d}" for day in range(8, 24)]
DAY_MS = 86400 * 1000


def make_plan(max_seq_len: int = 5) -> StreamingDataPlan:
    rows = []
    item_id = 1
    for day_index, date in enumerate(DATES):
        rows.append(
            {
                "date": date,
                "user_idx": 1,
                "item_idx": item_id,
                "behavior": 2,
                "label": 1,
                "time_ms": day_index * DAY_MS + 2000,
            }
        )
        item_id += 1
    rows.extend(
        [
            {
                "date": "20220411",
                "user_idx": 2,
                "item_idx": item_id,
                "behavior": 3,
                "label": 1,
                "time_ms": 3 * DAY_MS + 1000,
            },
            {
                "date": "20220412",
                "user_idx": 2,
                "item_idx": item_id + 1,
                "behavior": 1,
                "label": 0,
                "time_ms": 4 * DAY_MS + 1000,
            },
            {
                "date": "20220412",
                "user_idx": 2,
                "item_idx": item_id + 2,
                "behavior": 2,
                "label": 1,
                "time_ms": 4 * DAY_MS + 3000,
            },
            {
                "date": "20220408",
                "user_idx": 3,
                "item_idx": item_id + 3,
                "behavior": 1,
                "label": 1,
                "time_ms": 500,
            },
            {
                "date": "20220420",
                "user_idx": 3,
                "item_idx": item_id + 4,
                "behavior": 2,
                "label": 1,
                "time_ms": 12 * DAY_MS + 1000,
            },
        ]
    )
    interactions = pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)
    trace = KuaiRandTrace(
        interactions=interactions,
        num_users=3,
        num_items=item_id + 4,
        num_behaviors=9,
        user_map={101: 1, 102: 2, 103: 3},
        item_map={value: value for value in range(1, item_id + 5)},
    )
    return StreamingDataPlan(
        trace=trace,
        base_dates=DATES[:4],
        stream_dates=DATES[4:],
        max_seq_len=max_seq_len,
        history_window_days=8,
    )


def by_user(snapshot: OrganicStreamSnapshot) -> dict[int, object]:
    return dict(snapshot.records)


def test_theta0_through_theta11_map_to_next_unseen_dates():
    snapshots = build_4plus12_causal_snapshots(make_plan(), (3, 1, 2))

    assert [snapshot.version for snapshot in snapshots] == list(range(12))
    assert [snapshot.target_date for snapshot in snapshots] == DATES[4:]
    assert [record.user_id for record in snapshots[0].records.values()] == [
        1,
        2,
        3,
    ]
    assert snapshots[0].to_dict()["protocol"] == ORGANIC_STREAM_PROTOCOL


def test_snapshot_is_built_before_target_day_ingestion():
    snapshots = build_4plus12_causal_snapshots(make_plan(max_seq_len=32), (1, 2, 3))
    first = by_user(snapshots[0])
    second = by_user(snapshots[1])

    assert first[1].new_events.item_ids.tolist() == [5]
    assert first[1].positives == first[1].engaged_positive_item_ids
    assert 5 not in first[1].history.events.item_ids
    assert 5 in second[1].history.events.item_ids
    assert set(first[1].history_event_identities).isdisjoint(
        first[1].new_event_identities
    )
    assert first[1].new_event_identities[0] in second[1].history_event_identities
    assert all(
        timestamp < first[1].as_of_timestamp_ms
        for timestamp in first[1].history.events.timestamps
    )


def test_active_history_matches_streaming_plan_eval_set():
    snapshots = build_4plus12_causal_snapshots(make_plan(), (1, 2, 3))
    reference = make_plan()
    reference.init_base()

    for version, target_date in enumerate(reference.stream_dates):
        samples = {
            int(sample["history"]["user_id"]): sample
            for sample in reference.get_eval_set(target_date)
        }
        records = by_user(snapshots[version])
        for user_id, sample in samples.items():
            history = records[user_id].history
            assert history is not None
            assert history.events.item_ids.tolist() == sample["history"][
                "item_ids"
            ].tolist()
            assert history.events.behaviors.tolist() == sample["history"][
                "behaviors"
            ].tolist()
            assert history.events.time_deltas.tolist() == sample["history"][
                "time_deltas"
            ].tolist()
            assert history.events.timestamps.tolist() == sample["history"][
                "timestamps"
            ].tolist()
            assert records[user_id].engaged_positive_item_ids == tuple(
                sample["pos_items"]
            )
        if version < len(reference.stream_dates) - 1:
            reference.ingest_day(target_date)


def test_inactive_wall_clock_none_and_short_histories_are_preserved():
    snapshots = build_4plus12_causal_snapshots(
        make_plan(max_seq_len=32),
        (1, 2, 3),
    )
    first = by_user(snapshots[0])
    d20 = by_user(snapshots[8])

    assert not first[3].active
    assert (
        first[3].as_of_timestamp_ms
        == snapshots[0].wall_clock_as_of_timestamp_ms
    )
    assert first[2].history is not None
    assert len(first[2].history.events) == 1
    assert d20[3].active
    assert d20[3].history is None
    assert d20[3].engaged_positive_item_ids == tuple(
        d20[3].new_events.item_ids.tolist()
    )


def test_history_applies_rolling_window_and_tail_cap():
    capped = build_4plus12_causal_snapshots(make_plan(max_seq_len=3), (1,))
    first_history = capped[0].records[1].history

    assert first_history is not None
    assert first_history.events.item_ids.tolist() == [2, 3, 4]
    assert first_history.available_length_before_token_cap == 4
    assert first_history.token_truncated

    windowed = build_4plus12_causal_snapshots(make_plan(max_seq_len=64), (1,))
    d20_history = windowed[8].records[1].history

    assert d20_history is not None
    assert len(d20_history.events) == 8
    assert d20_history.available_length_before_token_cap == 8
    assert not d20_history.token_truncated
    assert d20_history.events.timestamps[0] == (
        windowed[8].records[1].as_of_timestamp_ms - 8 * DAY_MS
    )


def test_snapshot_serialization_hashes_and_arrays_are_stable_copies():
    plan = make_plan()
    snapshot = build_4plus12_causal_snapshots(plan, (1, 2, 3))[0]
    payload = json.loads(json.dumps(snapshot.to_dict()))
    restored = OrganicStreamSnapshot.from_dict(payload)
    record = by_user(snapshot)[1]

    assert restored.content_sha256 == snapshot.content_sha256
    assert by_user(restored)[1].history_sha256 == record.history_sha256
    assert record.new_event_identities == (
        stable_event_identity(
            1,
            int(record.new_events.item_ids[0]),
            int(record.new_events.behaviors[0]),
            int(record.new_events.labels[0]),
            int(record.new_events.timestamps[0]),
        ),
    )
    plan.trace.interactions.loc[:, "item_idx"] = 999
    assert record.new_events.item_ids.tolist() == [5]
    with pytest.raises(ValueError):
        record.new_events.item_ids[0] = 999
