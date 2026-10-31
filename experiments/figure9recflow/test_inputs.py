"""CPU checks for chronological cache input, coverage, and worker transport."""
import json
import pickle

import numpy as np
import pytest

from experiments.figure9recflow.inputs import RecFlowInputs, event_deltas
from experiments.recflow_full_population.code.protocol import day_bounds_ms


EVENT = np.dtype([("uid", "<i4"), ("ts", "<i8"), ("rid", "<i8"),
                  ("features", "<i4", (5,)), ("behavior", "u1"), ("label", "i1")])
REQUEST = np.dtype([("uid", "<i4"), ("ts", "<i8"), ("rid", "<i8"),
    ("start", "<i8"), ("stop", "<i8"), ("observed", "<i4"), ("positive", "<i4"),
    ("history_stop", "<i8"), ("history_length", "<i2"), ("day", "u1"),
    ("role", "i1"), ("resident", "?"), ("first_request", "?"), ("valid_request", "?")])


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "data"
    for part in ("history", "selections", "stage_v2/d29"):
        (data / part).mkdir(parents=True)
    np.savez(data / "population.npz", train=np.array([0, 1, 2], np.int32),
             cache_fit=np.array([3, 4], np.int32), validation=np.array([5], np.int32),
             cache_validation=np.array([6], np.int32))
    start, _ = day_bounds_ms(29)
    # UID0 has a >1024 prefix; UID1 only appends; UID2 is entirely inactive.
    times = [(0, t) for t in range(start - 1026000, start, 1000)]
    times += [(0, start + t) for t in [2000, 5000, 5000, 10000]]
    times += [(1, start - 1000), (1, start + 20000), (3, start - 5000)]
    events = np.zeros(len(times), EVENT)
    for i, (uid, ts) in enumerate(times):
        events[i] = (uid, ts, i, [i, 1, 2, 3, 4], i % 8, i % 2)
    offsets = np.r_[0, np.bincount(events["uid"], minlength=7).cumsum()]
    np.save(data / "history/events.npy", events)
    np.save(data / "history/user_offsets.npy", offsets)
    features = np.arange(40, dtype=np.int32).reshape(8, 5)
    labels = np.array([-1, 0, -1, -1, 1, 0, 1, 0], np.int8)
    requests = np.zeros(4, REQUEST)
    for i, (ts, valid) in enumerate([(start + 5000, True), (start + 5000, True),
                                    (start + 8000, True), (start + 12000, False)]):
        stop = int(np.searchsorted(events["ts"][:offsets[1]], ts, side="left"))
        labs = labels[2 * i:2 * i + 2]
        requests[i] = (0, ts, 100 + i, 2 * i, 2 * i + 2, int((labs >= 0).sum()),
                       int((labs == 1).sum()), stop, min(stop, 1024), 29, 0,
                       True, i == 0, valid)
    stage = data / "stage_v2/d29"
    np.save(stage / "features.npy", features)
    np.save(stage / "labels.npy", labels)
    np.save(stage / "index.npy", requests)
    np.save(data / "selections/role2_d29_d29.npy", requests[:3])
    (data / "selection_manifest.json").write_text(json.dumps({"datasets": {"2:29": {"requests": 3}}}))
    old, new = tmp_path / "old.pt", tmp_path / "new.pt"
    old.touch()
    new.touch()
    manifest = tmp_path / "chain_manifest.json"
    manifest.write_text(json.dumps({"history_limit": 1024,
        "nodes": [{"id": "V0", "checkpoint": str(old)}, {"id": "V1", "checkpoint": str(new)},
                  {"id": "V2", "checkpoint": None}],
        "edges": [{"id": "V0_to_V1", "source": "V0", "target": "V1", "evaluation_day": 29,
                   "evidence": {"kind": "fixture"}},
                  {"id": "V1_to_V2", "source": "V1", "target": "V2", "evaluation_day": 30}]}))
    return RecFlowInputs(data, manifest, fit_users=2)


def test_fixed_roles_and_inactive_users_are_not_filtered(inputs):
    assert len(inputs.edges) == 1
    np.testing.assert_array_equal(inputs.eval_uids,
        np.random.default_rng(20261030).permutation(np.array([0, 1, 2], np.int32)))
    assert set(inputs.day(29).uids) == {0, 1, 2}
    np.testing.assert_array_equal(inputs.day(29, limit=2).uids, inputs.eval_uids[:2])
    assert set(inputs.fit_uids) == {3, 4}
    assert [len(x.events) for x in inputs.fit_snapshots(29)].count(0) == 1
    with pytest.raises(ValueError, match="original train"):
        inputs.day(29, uids=inputs.fit_uids)


def test_timestamp_blocks_and_cross_prefix_delta(inputs):
    user = inputs.day(29, uids=[0])[0]
    assert len(user.initial_history) == 1024
    assert user.initial_deltas[0] == 0
    assert user.initial_deltas[1] == 1
    np.testing.assert_array_equal(user.event_deltas, [3, 3, 0, 5])
    blocks = list(user.timeline())
    assert [b.kind for b in blocks] == ["append", "request", "append", "request", "append"]
    assert len(blocks[1].requests) == 2
    assert blocks[1].ts == blocks[2].ts
    assert len(blocks[2].events) == 2
    assert blocks[1].requests[0].query_delta == 3
    assert blocks[3].requests[0].query_delta == 3
    # The post-query event is retained, and its delta is not reset by FIFO.
    assert blocks[-1].deltas[0] == 5


def test_full_candidates_zero_positive_requests_and_views(inputs):
    day = inputs.day(29, uids=[0])
    user = day[0]
    assert len(user.requests) == 3
    assert [r.rid for r in user.requests] == [100, 101, 102]
    assert [len(r.labels) for r in user.requests] == [2, 2, 2]
    assert [int((r.labels == 1).sum()) for r in user.requests] == [0, 0, 1]
    assert np.shares_memory(user.requests[0].features, day._features)
    assert np.shares_memory(user.requests[0].labels, day._labels)
    assert not user.requests[0].features.flags.writeable


def test_coverage_includes_behavior_only_and_inactive_population(inputs):
    result = inputs.day(29).coverage()
    assert result["quality_users"] == result["cost_users"] == 3
    assert result["requests"] == 3 and result["candidates"] == 6
    assert result["zero_positive_requests"] == 2
    assert result["zero_observed_requests"] == 1
    assert result["excluded_invalid_requests"] == 1
    assert result["request_inactive_users"] == 2
    assert result["first_consumer_users"] == 2
    assert result["once_repair_history_rows"] == 1025
    assert result["appended_history_events"] == 5
    assert result["maintenance_after_last_request_events"] == 2
    assert result["fit_empty_history_users"] == 1
    behavior_only = inputs.day(29, uids=[1])[0]
    assert not behavior_only.requests
    assert [b.kind for b in behavior_only.timeline()] == ["append"]
    assert not list(inputs.day(29, uids=[2])[0].timeline())


def test_missing_request_and_missing_checkpoint_are_rejected(inputs):
    path = inputs.data / "selections/role2_d29_d29.npy"
    records = np.load(path)
    np.save(path, records[1:])
    with pytest.raises(ValueError, match="every valid"):
        inputs.day(29).coverage()
    inputs.edges[0].target_checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="Accepted checkpoint"):
        RecFlowInputs(inputs.data, inputs.chain_manifest, fit_users=2)


def test_worker_pickle_reopens_inputs_and_transports_only_user_slices(inputs):
    day = inputs.day(29, uids=[0])
    original = day[0]
    encoded = pickle.dumps(day)
    assert len(encoded) < 10000
    restored = pickle.loads(encoded)
    assert restored._records is None and restored.inputs._events is None
    copy = restored[0]
    np.testing.assert_array_equal(original.events, copy.events)
    np.testing.assert_array_equal(original.requests[0].labels, copy.requests[0].labels)
    # A worker result contains only this user's slices, not a reference to DayInputs.
    transported = pickle.loads(pickle.dumps(original))
    assert [x.kind for x in transported.timeline()] == [x.kind for x in original.timeline()]
    assert len(pickle.dumps(original)) < 100000


def test_event_deltas_reject_reverse_time():
    rows = np.zeros(2, EVENT)
    rows["ts"] = [10, 9]
    with pytest.raises(ValueError, match="chronological"):
        event_deltas(rows)
