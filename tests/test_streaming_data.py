import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hstu_kvcache.data import (
    KuaiRandTrace,
    StreamingDataPlan,
    collate_batch,
    load_prepared_exposure_plan,
)
from hstu_kvcache.data.kuairand import load_kuairand
from hstu_kvcache.streaming.trainer import build_next_item_targets


def test_next_item_targets_use_shift_labels_lengths_and_train_mask():
    item_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    lengths = torch.tensor([3, 2])
    labels = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]])
    train_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)

    targets, valid = build_next_item_targets(item_ids, lengths, labels, train_mask)

    assert targets.tolist() == [[2, 3, 0], [5, 0, 0]]
    assert valid.tolist() == [[True, False, False], [True, False, False]]


def test_collate_preserves_labels_lengths_and_training_mask():
    sequences = [
        {
            "item_ids": np.array([1, 2, 3]),
            "behaviors": np.array([1, 2, 2]),
            "time_deltas": np.array([0.0, 1.0, 2.0]),
            "labels": np.array([0, 1, 0]),
            "train_mask": np.array([False, True, True]),
        },
        {
            "item_ids": np.array([4, 5]),
            "behaviors": np.array([1, 3]),
            "time_deltas": np.array([0.0, 3.0]),
            "labels": np.array([0, 1]),
            "train_mask": np.array([False, True]),
        },
    ]

    batch = collate_batch(sequences, max_seq_len=4, pad_to=4)

    assert batch["lengths"].tolist() == [3, 2]
    assert batch["labels"].tolist() == [[0, 1, 0, 0], [0, 1, 0, 0]]
    assert batch["train_mask"].tolist() == [
        [False, True, True, False],
        [False, True, False, False],
    ]


def test_streaming_plan_uses_engaged_eval_items_and_current_day_train_mask():
    interactions = pd.DataFrame(
        {
            "date": ["d1", "d1", "d1", "d2", "d2"],
            "user_idx": [1, 1, 1, 1, 1],
            "item_idx": [1, 2, 3, 4, 5],
            "behavior": [1, 2, 1, 1, 3],
            "label": [0, 1, 0, 0, 1],
            "time_ms": [1000, 2000, 3000, 4000, 5000],
        }
    )
    trace = KuaiRandTrace(interactions, 1, 5, 9, {10: 1}, {i: i for i in range(1, 6)})
    plan = StreamingDataPlan(trace, ["d1"], ["d2"], max_seq_len=8)
    plan.init_base()

    samples = plan.get_eval_set("d2")

    assert len(samples) == 1
    assert samples[0]["pos_items"] == [5]

    plan.ingest_day("d2")
    batch = next(plan.iter_train_batches("d2", batch_size=1))

    assert batch["lengths"].tolist() == [5]
    assert batch["train_mask"][0, :5].tolist() == [False, False, False, True, True]


def test_eval_user_limit_preserves_first_exposure_order():
    interactions = pd.DataFrame(
        {
            "date": ["d1", "d1", "d2", "d2", "d2"],
            "user_idx": [1, 2, 1, 2, 1],
            "item_idx": [1, 2, 3, 4, 5],
            "behavior": [1, 1, 1, 2, 2],
            "label": [0, 0, 0, 1, 1],
            "time_ms": [1000, 1000, 2000, 3000, 4000],
        }
    )
    trace = KuaiRandTrace(interactions, 2, 5, 2, {}, {})
    plan = StreamingDataPlan(trace, ["d1"], ["d2"], max_seq_len=8)
    plan.init_base()

    samples = plan.get_eval_set("d2", max_users=1)

    assert samples[0]["history"]["user_id"] == 1
    assert samples[0]["pos_items"] == [5]


def test_all_chunk_base_training_covers_each_next_item_pair_once():
    interactions = pd.DataFrame(
        {
            "date": ["d1"] * 8,
            "user_idx": [1] * 8,
            "item_idx": list(range(1, 9)),
            "behavior": [2] * 8,
            "label": [1] * 8,
            "time_ms": [1000 * value for value in range(1, 9)],
        }
    )
    trace = KuaiRandTrace(interactions, 1, 8, 9, {10: 1}, {i: i for i in range(1, 9)})
    plan = StreamingDataPlan(trace, ["d1"], [], max_seq_len=4)
    plan.init_base()

    batch = next(plan.iter_base_train_batches(batch_size=8, all_chunks=True))
    _, valid = build_next_item_targets(
        batch["item_ids"],
        batch["lengths"],
        batch["labels"],
        batch["train_mask"],
    )

    assert sorted(batch["lengths"].tolist()) == [2, 4, 4]
    assert int(valid.sum()) == 7


def test_all_chunk_stream_training_covers_current_day_targets_once():
    interactions = pd.DataFrame(
        {
            "date": ["d1"] * 3 + ["d2"] * 6,
            "user_idx": [1] * 9,
            "item_idx": list(range(1, 10)),
            "behavior": [2] * 9,
            "label": [1] * 9,
            "time_ms": [1000 * value for value in range(1, 10)],
        }
    )
    trace = KuaiRandTrace(interactions, 1, 9, 9, {10: 1}, {i: i for i in range(1, 10)})
    plan = StreamingDataPlan(trace, ["d1"], ["d2"], max_seq_len=4)
    plan.init_base()
    plan.ingest_day("d2")

    batch = next(plan.iter_train_batches("d2", batch_size=8, all_chunks=True))
    _, valid = build_next_item_targets(
        batch["item_ids"],
        batch["lengths"],
        batch["labels"],
        batch["train_mask"],
    )

    assert sorted(batch["lengths"].tolist()) == [3, 4, 4]
    assert int(valid.sum()) == 6


def test_kuairand_behavior_priority_is_preserved(tmp_path: Path):
    rows = []
    flags = [
        {},
        {"long_view": 1},
        {"long_view": 1, "is_click": 1},
        {"is_click": 1, "is_like": 1},
        {"is_like": 1, "is_follow": 1},
        {"is_follow": 1, "is_comment": 1},
        {"is_comment": 1, "is_forward": 1},
        {"is_forward": 1, "is_hate": 1},
    ]
    for index, values in enumerate(flags):
        row = {
            "user_id": 1,
            "video_id": index + 1,
            "time_ms": 1000 * (index + 1),
            "date": 20220408,
            "hourmin": 1200,
            "is_click": 0,
            "is_like": 0,
            "is_follow": 0,
            "is_comment": 0,
            "is_forward": 0,
            "is_hate": 0,
            "long_view": 0,
            "play_time_ms": 1000,
            "duration_ms": 1000,
        }
        row.update(values)
        rows.append(row)
    path = tmp_path / "log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    trace = load_kuairand([path], min_interactions_per_user=1, max_items=None)

    assert trace.interactions["behavior"].tolist() == [1, 8, 2, 3, 4, 5, 6, 7]


def test_prepared_exposure_plan_preserves_base_windows_and_positive_targets(
    tmp_path: Path,
):
    metadata = {
        "dataset": "synthetic",
        "selected_users": 2,
        "fitted_items": 6,
        "num_behaviors": 2,
        "window_count": 2,
    }
    path = tmp_path / "prepared.npz"
    np.savez_compressed(
        path,
        user_idx=np.array([1, 1, 2, 2, 1, 2, 1, 2]),
        item_idx=np.array([1, 2, 3, 4, 5, 5, 6, 6]),
        behavior=np.array([1, 2, 1, 2, 1, 2, 2, 1]),
        label=np.array([0, 1, 0, 1, 0, 1, 1, 0]),
        time_ms=np.array([1000, 2000, 1000, 2000, 3000, 3000, 4000, 4000]),
        window_index=np.array([-1, -1, -1, -1, 0, 0, 1, 1]),
        metadata_json=np.array(json.dumps(metadata)),
    )

    plan, loaded = load_prepared_exposure_plan(path, max_seq_len=8)
    plan.init_base()

    assert loaded == metadata
    assert plan.base_dates == ["base"]
    assert plan.stream_dates == ["window_0", "window_1"]
    assert plan.get_eval_set("window_0")[0]["pos_items"] == [5]
    plan.ingest_day("window_0")
    assert plan.get_eval_set("window_1")[0]["pos_items"] == [6]
