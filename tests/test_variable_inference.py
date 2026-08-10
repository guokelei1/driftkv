import numpy as np
import pytest

from hstu_kvcache.migration.variable_inference import (
    ROLE_CODES,
    load_corpus,
    prefix_schedule,
    write_corpus,
)


def test_prefix_schedule_is_variable_progressive_and_reserves_target() -> None:
    lengths = np.asarray([96, 153, 402, 512], dtype=np.int64)
    schedule = prefix_schedule(lengths, 3)
    assert schedule.tolist() == [
        [64, 74, 84, 95],
        [76, 101, 126, 152],
        [200, 267, 334, 401],
        [255, 340, 425, 511],
    ]
    assert np.all(schedule[:, 1:] > schedule[:, :-1])
    assert np.array_equal(schedule[:, -1], lengths - 1)


def test_prefix_schedule_supports_two_edge_qb_chain() -> None:
    lengths = np.asarray([104, 255, 512], dtype=np.int64)
    schedule = prefix_schedule(lengths, 2)
    assert schedule.tolist() == [
        [64, 83, 103],
        [127, 190, 254],
        [255, 383, 511],
    ]


def _arrays() -> dict[str, np.ndarray]:
    lengths = np.asarray([96, 104], dtype=np.int64)
    offsets = np.asarray([0, 96, 200], dtype=np.int64)
    events = int(offsets[-1])
    return {
        "record_source_ids": np.asarray([1, 2], dtype=np.int64),
        "record_user_ids": np.asarray([10, 11], dtype=np.int64),
        "record_role": np.full(2, ROLE_CODES["qualification"], dtype=np.uint8),
        "record_offsets": offsets,
        "record_valid_lengths": lengths,
        "edge_prefix_lengths": prefix_schedule(lengths, 2),
        "feature_ids": np.ones((events, 1), dtype=np.uint32),
        "target_item_ids": np.ones(events, dtype=np.uint32),
        "behaviors": np.ones(events, dtype=np.uint8),
        "time_deltas": np.ones(events, dtype=np.float32),
        "labels": np.ones(events, dtype=np.uint8),
        "is_prediction_item": np.ones(events, dtype=np.uint8),
    }


def test_variable_corpus_roundtrip_binds_content(tmp_path) -> None:
    path = tmp_path / "variable.npz"
    descriptor = write_corpus(
        path,
        _arrays(),
        {
            "dataset": "qk",
            "edge_count": 2,
            "feature_fields": 1,
            "minimum_initial_tokens": 64,
        },
    )
    corpus = load_corpus(path)
    assert descriptor["content_sha256"] == corpus.content_sha256
    assert corpus.role_records("qualification").tolist() == [0, 1]


def test_variable_corpus_rejects_duplicate_users(tmp_path) -> None:
    arrays = _arrays()
    arrays["record_user_ids"][1] = arrays["record_user_ids"][0]
    with pytest.raises(ValueError, match="corpus differs"):
        write_corpus(
            tmp_path / "duplicate.npz",
            arrays,
            {
                "dataset": "qk",
                "edge_count": 2,
                "feature_fields": 1,
                "minimum_initial_tokens": 64,
            },
        )
