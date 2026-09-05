from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hstu_kvcache.data import (
    YambdaScaleDataset,
    apply_stable_oov_buckets,
    select_medium_uids,
    stable_oov_bucket,
    uid_selector_digest,
)


def test_uid_selector_is_order_independent_and_nested() -> None:
    eligible = [91, 3, 42, 5, 100]
    selected = select_medium_uids(eligible, count=3)
    assert selected == select_medium_uids(reversed(eligible), count=3)
    assert set(selected).issubset(eligible)
    expected = sorted(eligible, key=lambda uid: (uid_selector_digest(uid), uid))[:3]
    assert selected == expected


def test_stable_oov_buckets_preserve_known_ids_and_are_repeatable() -> None:
    raw = np.asarray(["known", "new-a", "new-b", "new-a"])
    mapped = np.asarray([12, 0, 0, 0])
    values = apply_stable_oov_buckets(raw, mapped, known_vocab_size=100, buckets=32)
    assert values[0] == 12
    assert values[1] == values[3] == stable_oov_bucket("new-a", known_vocab_size=100, buckets=32)
    assert (100 <= values[1] < 132) and (100 <= values[2] < 132)


def test_zero_oov_buckets_retains_legacy_single_oov() -> None:
    raw = np.asarray(["new-a"])
    assert apply_stable_oov_buckets(raw, np.asarray([0]), known_vocab_size=100, buckets=0).tolist() == [0]


def test_loader_rejects_future_feedback_before_reading_data(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps({
        "rank_limit": 2,
        "shared_listens_glob": "listens/*.parquet",
        "shared_feedback_glob": "feedback/*.parquet",
        "item_mapping_path": "items.parquet",
        "feedback_access": {
            "default_training_end_exclusive": 20,
            "default_evaluation_end_exclusive": 10,
        },
    }))
    dataset = YambdaScaleDataset(manifest, threads=1)
    # No payload files exist: the access check must precede any data scan.
    with pytest.raises(PermissionError, match="evaluation feedback is locked at 10"):
        next(dataset.iter_feedback(0, 11, purpose="evaluation"))
