import numpy as np
import pytest

from experiments.recflow_request_ranking.code.data_common import (
    RAW_DTYPE,
    check_exposure_join,
    deduplicate_records,
    rank_membership,
    roles,
)


def rows():
    raw = np.zeros(4, RAW_DTYPE)
    raw["uid"] = [10, 10, 11, 10]
    raw["rid"] = [100, 100, 101, 100]
    raw["ts"] = [2000, 2000, 2500, 2000]
    raw["features"][:, 0] = [30, 10, 40, 10]
    raw["label"] = [-1, 1, 0, 1]
    raw["behavior"] = [0, 1, 0, 1]
    return raw


def test_ranking_members_include_zero_and_exclude_early_stage_sentinels():
    assert rank_membership([0, 1, 120, 121, 3000, 8000]).tolist() == [True] * 4 + [False] * 2
    with pytest.raises(ValueError, match="Unrecognized"):
        rank_membership([-1, 9000])


def test_canonical_dedup_and_join_preserve_unknowns():
    raw, keys, requests, count = deduplicate_records(rows())
    assert count == 1
    assert raw["features"][:, 0].tolist() == [10, 30, 40]
    assert requests["observed"].tolist() == [1, 1]
    assert requests["positive"].tolist() == [1, 0]
    assert check_exposure_join(raw, keys, raw[raw["label"] >= 0])["matched_exposures"] == 2
    assert raw["label"].tolist() == [1, -1, 0]


def test_contradictory_duplicate_cannot_be_silently_collapsed():
    raw = rows()
    raw["label"][-1] = 0
    with pytest.raises(ValueError, match="Inconsistent duplicated"):
        deduplicate_records(raw)


def test_request_cannot_cross_user_or_time():
    raw = rows()[:3]
    raw["uid"][1] = 12
    with pytest.raises(ValueError, match="multiple users"):
        deduplicate_records(raw)


def test_exposure_missing_or_unmatched_is_fatal():
    raw, keys, _, _ = deduplicate_records(rows())
    exposed = raw[raw["label"] >= 0].copy()
    exposed["features"][0, 0] = 999
    with pytest.raises(ValueError, match="missing"):
        check_exposure_join(raw, keys, exposed)
    exposed = raw[raw["label"] >= 0].copy()
    exposed["behavior"][0] = 3
    with pytest.raises(ValueError, match="behavior"):
        check_exposure_join(raw, keys, exposed)


def test_roles_are_stable_and_partitioned():
    ids = np.arange(10000)
    assert np.array_equal(roles(ids)[::2], roles(ids[::2]))
    assert set(roles(ids)) == {0, 1, 2, 3}
