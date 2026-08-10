from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hstu_kvcache.data.qk_theta0 import (
    PROTOCOL,
    QKTheta0CorpusConfig,
    _open_scan_cache,
    artifact_sha256,
    build_rank_batch,
    epoch_record_order,
    load_qk_theta0_corpus,
    select_training_users,
    training_ends,
)
from hstu_kvcache.streaming.trainer import build_next_item_targets


def test_training_ends_require_a_future_effective_target() -> None:
    labels = np.asarray(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    lengths = np.asarray([4, 4, 3], dtype=np.uint16)
    assert training_ends(labels, lengths).tolist() == [2, 0, 3]


def test_user_selection_reaches_real_next_item_row_gate() -> None:
    items = np.asarray(
        [
            [5, 1],
            [6, 2],
            [7, 3],
            [8, 4],
        ],
        dtype=np.uint32,
    )
    labels = np.asarray([[0, 1]] * 4, dtype=np.uint8)
    lengths = np.asarray([2] * 4, dtype=np.uint16)
    users = np.asarray([100, 101, 102, 103], dtype=np.int64)
    selected, ends, audit = select_training_users(
        items,
        labels,
        lengths,
        users,
        semantic_rows=8,
        representative_users=1,
        minimum_eligible_rows=8,
        selection_seed=17,
        block_size=2,
    )
    assert set(selected.tolist()) == {0, 1, 2, 3}
    assert ends.tolist() == [2, 2, 2, 2]
    assert audit["eligible_semantic_rows"] == 8
    assert audit["selected_covered_rows"] == 8


def test_scan_cache_rolls_back_uncommitted_chunk(tmp_path: Path) -> None:
    config = QKTheta0CorpusConfig(
        source=tmp_path / "source.zip",
        member="qk.csv",
        catalog=tmp_path / "catalog.npz",
        user_lengths=tmp_path / "lengths.npz",
        cache_dir=tmp_path / "cache",
        output=tmp_path / "corpus.npz",
        summary=tmp_path / "summary.json",
        base_prefix=4,
        prediction_rows=2,
        representative_users=1,
        minimum_eligible_rows=1,
    )
    identity = {
        "protocol": PROTOCOL,
        "phase": "base_prefix_scan",
        "source": {},
        "catalog_file_sha256": "catalog",
        "user_lengths_file_sha256": "users",
        "users": 2,
        "base_prefix": 4,
        "prediction_rows": 2,
        "shape": [2, 4],
    }
    item, behavior, label, seen, _ = _open_scan_cache(config, identity)
    item[0, :2] = [1, 2]
    behavior[0, :2] = [1, 2]
    label[0, :2] = [0, 1]
    seen[0] = 2
    for value in (item, behavior, label, seen):
        value.flush()
    del item, behavior, label, seen
    item, behavior, label, seen, state = _open_scan_cache(config, identity)
    assert state["completed_chunks"] == 0
    assert np.count_nonzero(item) == 0
    assert np.count_nonzero(behavior) == 0
    assert np.count_nonzero(label) == 0
    assert np.count_nonzero(seen) == 0


def _write_corpus(path: Path) -> None:
    arrays = {
        "record_user_ids": np.asarray([10, 20], dtype=np.int64),
        "record_offsets": np.asarray([0, 3, 7], dtype=np.int64),
        "record_lengths": np.asarray([3, 4], dtype=np.uint16),
        "record_selection": np.asarray([1, 2], dtype=np.uint8),
        "item_idx": np.asarray([5, 1, 2, 6, 7, 3, 4], dtype=np.uint32),
        "behavior": np.asarray([1, 2, 3, 1, 1, 4, 5], dtype=np.uint8),
        "label": np.asarray([0, 1, 1, 0, 0, 1, 1], dtype=np.uint8),
        "raw_ordinal": np.asarray([0, 1, 2, 0, 1, 2, 3], dtype=np.uint16),
    }
    metadata = {
        "protocol": PROTOCOL,
        "dataset": "tenrec-qk",
        "scientific_result": False,
        "formal_result": False,
        "catalog": {"semantic_rows": 8, "prediction_rows": 4},
        "base_only_boundary": {
            "base_prefix": 64,
            "post_base_rows_materialized": False,
            "vocabulary_fit": "fixture",
        },
        "content_sha256": artifact_sha256(arrays),
    }
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_corpus_loader_and_rank_batch_preserve_causal_targets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qk_theta0.npz"
    _write_corpus(path)
    corpus = load_qk_theta0_corpus(
        path,
        num_embeddings=9,
        num_prediction_items=4,
    )
    batch = build_rank_batch(
        corpus,
        np.asarray([1], dtype=np.int64),
        batch_size=2,
    )
    targets, valid = build_next_item_targets(
        batch["item_ids"],
        batch["lengths"],
        batch["labels"],
        batch["train_mask"],
    )
    assert targets[valid].tolist() == [3, 4]
    assert batch["lengths"].tolist() == [4, 0]
    assert batch["time_deltas"][0].tolist() == [0.0, 1.0, 1.0, 1.0]
    first = epoch_record_order(corpus, seed=7, epoch=0, bucket_size=2)
    second = epoch_record_order(corpus, seed=7, epoch=0, bucket_size=2)
    assert first.tolist() == second.tolist()
