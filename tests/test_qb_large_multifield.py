from __future__ import annotations

import numpy as np
import pandas as pd

from hstu_kvcache.data.qb_large_multifield import (
    QBLargeProfile,
    build_catalog,
    extend_role_horizon,
    materialize_corpus,
)


def frame() -> pd.DataFrame:
    value = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "item_id": [10, 11, 12, 10, 13, 14],
            "click": [1, 0, 1, 0, 1, 0],
            "follow": [0, 0, 0, 0, 0, 0],
            "like": [0, 1, 0, 0, 0, 0],
            "share": [0, 0, 0, 1, 0, 0],
            "video_category_code": [0, 1, 1, 0, 1, 0],
            "watch_bucket": [1, 2, 2, 1, 3, 2],
            "gender": [0, 0, 0, 1, 1, 1],
            "age": [2, 2, 2, 3, 3, 3],
            "behavior_signature": [1, 4, 1, 8, 1, 0],
            "raw_ordinal": [0, 1, 2, 0, 1, 2],
            "user_length": [3, 3, 3, 3, 3, 3],
        }
    )
    return value


def test_catalog_is_base_only_and_fallback_is_field_local() -> None:
    profile = QBLargeProfile(
        name="test",
        fields=("item", "user", "user_item"),
        embedding_width=16,
    )
    catalog = build_catalog(frame(), profile, base_prefix=3)
    features, targets, direct = catalog.map_frame(frame())
    assert features.shape == (6, 3)
    assert targets.shape == (6,)
    assert direct[:5].all()
    assert not direct[5]
    query = frame().iloc[:1].copy()
    query["item_id"] = 999
    mapped, target, item_direct = catalog.map_frame(query)
    assert not item_direct[0]
    assert catalog.offsets["item"] <= target[0] < catalog.offsets["user"]
    assert mapped[0, 1] == features[0, 1]
    assert np.all(mapped > 0)


def test_role_horizon_extension_preserves_every_parent_event() -> None:
    value = pd.concat(
        [
            frame(),
            pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "item_id": [15, 16],
                    "click": [1, 1],
                    "follow": [0, 0],
                    "like": [0, 0],
                    "share": [0, 0],
                    "video_category_code": [1, 0],
                    "watch_bucket": [2, 2],
                    "gender": [0, 1],
                    "age": [2, 3],
                    "behavior_signature": [1, 1],
                    "raw_ordinal": [3, 3],
                    "user_length": [4, 4],
                }
            ),
        ],
        ignore_index=True,
    ).sort_values(["user_id", "raw_ordinal"], kind="stable")
    value["user_length"] = 4
    profile = QBLargeProfile(
        name="test",
        fields=("item", "user", "user_item"),
        embedding_width=16,
    )
    catalog = build_catalog(value, profile, base_prefix=2)
    catalog = type(catalog)(
        profile=catalog.profile,
        keys=catalog.keys,
        offsets=catalog.offsets,
        metadata={**catalog.metadata, "content_sha256": "test-catalog"},
    )
    parent, metadata = materialize_corpus(
        value,
        catalog,
        base_prefix=2,
        required_horizon=3,
        users=2,
        train_users=1,
        tuning_users=1,
        qualification_users=0,
        role_salt="test-extension",
    )
    extended, extended_metadata = extend_role_horizon(
        value,
        catalog,
        parent,
        metadata,
        required_horizon=4,
    )
    assert np.array_equal(extended["role_record_user_ids"], parent["role_record_user_ids"])
    assert np.array_equal(extended["role_record_role"], parent["role_record_role"])
    assert np.array_equal(np.diff(extended["role_record_offsets"]), np.asarray([4, 4]))
    for name in (
        "feature_ids",
        "target_item_ids",
        "behavior",
        "raw_label",
        "label",
        "raw_ordinal",
        "is_prediction_item",
    ):
        expected = parent[f"role_{name}"]
        if expected.ndim == 2:
            observed = extended[f"role_{name}"].reshape(2, 4, -1)[:, :3].reshape(-1, 3)
        else:
            observed = extended[f"role_{name}"].reshape(2, 4)[:, :3].reshape(-1)
        assert np.array_equal(observed, expected)
    assert extended_metadata["parent_corpus_content_sha256"] == metadata["content_sha256"]
