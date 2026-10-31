import numpy as np
import pytest

from hstu_kvcache.recflow.data import RAW_DTYPE, PreparedRecFlow, prepare_arrays, user_roles


def test_complete_request_and_simultaneous_requests_are_excluded(tmp_path):
    uid = int(np.flatnonzero(user_roles(np.arange(100)) == 0)[0])
    rows = [
        (uid, 1000, 1, 10, 1, 3, 1, 1),
        (uid, 2000, 2, 20, 2, 4, 1, 18),
        (uid, 2000, 3, 10, 1, 3, 0, 18),
        (uid, 3000, 4, 30, 9, 9, 1, 19),
        (uid, 3000, 4, 30, 9, 9, 1, 19),
        (uid, 3000, 4, 10, 1, 3, 1, 19),
        (uid, 4000, 5, 20, 5, 5, 0, 20),
    ]
    audit = prepare_arrays(np.array(rows, dtype=RAW_DTYPE)[::-1], tmp_path)
    data = PreparedRecFlow(tmp_path)
    assert data.catalog["raw_item_ids"].tolist() == [10, 20]
    assert data.catalog["c1"].tolist() == [1, 2]
    for idx in [1, 2]:
        assert data.history(idx)[0].tolist() == [1]
    assert data.targets(3)[0].tolist() == [10, 30]
    assert data.targets(3)[1].tolist() == [1, 3]
    assert data.history(4)[0].tolist() == [1, 2, 1, 1, 3, 3]
    assert data.history(4)[2].tolist() == [0, 1, 0, 1, 0, 0]
    # Eviction must not change the retained events' time embeddings.
    assert data.history(4, max_length=3)[2].tolist() == [1, 0, 0]
    np.testing.assert_array_equal(data.history(4, max_length=3)[2], data.history(4)[2][-3:])
    assert audit["duplicate_request_video_rows_retained_in_history"] == 1
    assert audit["same_user_timestamp_extra_requests"] == 1
    assert data.request_indices(19, 20).tolist() == [3]


def test_ambiguous_request_timestamp_cannot_silently_split(tmp_path):
    rows = [(1, 1000, 7, 10, 1, 1, 1, 1), (1, 2000, 7, 20, 1, 1, 1, 1)]
    with pytest.raises(ValueError, match="multiple timestamps"):
        prepare_arrays(np.array(rows, dtype=RAW_DTYPE), tmp_path)


def test_development_selection_excludes_heldout_and_late_users(tmp_path):
    uids = np.arange(100)
    dev = int(uids[user_roles(uids) == 0][0])
    heldout = int(uids[user_roles(uids) == 2][0])
    rows = [(dev, 1000, 1, 10, 1, 1, 1, 1), (heldout, 1000, 2, 20, 1, 1, 1, 1), (101, 2000, 3, 30, 1, 1, 1, 19)]
    prepare_arrays(np.array(rows, dtype=RAW_DTYPE), tmp_path)
    data = PreparedRecFlow(tmp_path)
    assert data.cohort(min_history=1).tolist() == [dev]
    selected = data.request_indices(1, 37)
    assert data.requests["uid"][selected].tolist() == [dev]
