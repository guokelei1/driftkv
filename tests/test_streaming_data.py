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
    load_prepared_kuairand_plan,
    save_prepared_kuairand_plan,
)
from hstu_kvcache.data.kuairand import load_kuairand
from hstu_kvcache.streaming import prepared_protocol_for_base_days
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
    assert samples[0]["history"]["available_length_before_token_cap"] == 3
    assert not samples[0]["history"]["token_truncated"]

    plan.ingest_day("d2")
    batch = next(plan.iter_train_batches("d2", batch_size=1))

    assert batch["lengths"].tolist() == [5]
    assert batch["train_mask"][0, :5].tolist() == [False, False, False, True, True]


def test_eval_context_records_pre_cap_length_and_tail_truncation():
    interactions = pd.DataFrame(
        {
            "date": ["d1"] * 6 + ["d2"],
            "user_idx": [1] * 7,
            "item_idx": list(range(1, 8)),
            "behavior": [1] * 7,
            "label": [1] * 7,
            "time_ms": [1000 * value for value in range(1, 8)],
        }
    )
    trace = KuaiRandTrace(interactions, 1, 7, 2, {}, {})
    plan = StreamingDataPlan(trace, ["d1"], ["d2"], max_seq_len=4)
    plan.init_base()

    sample = plan.get_eval_set("d2")[0]

    assert sample["history"]["item_ids"].tolist() == [3, 4, 5, 6]
    assert sample["history"]["available_length_before_token_cap"] == 6
    assert sample["history"]["token_truncated"]


def test_as_of_timestamp_excludes_current_and_future_events():
    interactions = pd.DataFrame(
        {
            "date": ["d1", "d1", "d2", "d3"],
            "user_idx": [1, 1, 1, 1],
            "item_idx": [1, 2, 3, 4],
            "behavior": [1, 1, 1, 1],
            "label": [1, 1, 1, 1],
            "time_ms": [1000, 2000, 3000, 4000],
        }
    )
    trace = KuaiRandTrace(interactions, 1, 4, 2, {}, {})
    plan = StreamingDataPlan(trace, ["d1"], ["d2", "d3"], max_seq_len=8)
    plan.init_base()
    plan.ingest_day("d2")
    plan.ingest_day("d3")

    history = plan._build_seq(1, as_of_timestamp=3000)

    assert history["item_ids"].tolist() == [1, 2]
    assert history["timestamps"].tolist() == [1000, 2000]
    assert history["available_length_before_token_cap"] == 2
    assert not history["token_truncated"]


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


def test_context_hashing_preserves_long_tail_rows_but_not_targets(tmp_path: Path):
    rows = []
    for index, (date, video_id) in enumerate(
        [
            (20220408, 10),
            (20220408, 10),
            (20220408, 20),
            (20220409, 30),
            (20220409, 10),
        ]
    ):
        rows.append(
            {
                "user_id": 1,
                "video_id": video_id,
                "time_ms": 1000 * (index + 1),
                "date": date,
                "hourmin": 1200,
                "is_click": 1,
                "is_like": 0,
                "is_follow": 0,
                "is_comment": 0,
                "is_forward": 0,
                "is_hate": 0,
                "long_view": 0,
                "play_time_ms": 1000,
                "duration_ms": 1000,
            }
        )
    path = tmp_path / "log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    trace = load_kuairand(
        [path],
        min_interactions_per_user=1,
        max_items=1,
        fit_num_days=1,
        context_hash_buckets=8,
    )

    assert len(trace.interactions) == 5
    assert trace.num_prediction_items == 1
    assert trace.num_items == 9
    assert trace.interactions["is_prediction_item"].tolist() == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert trace.interactions["label"].tolist() == [1, 1, 0, 0, 1]
    assert trace.interactions.loc[
        ~trace.interactions["is_prediction_item"], "item_idx"
    ].between(2, 9).all()


def test_engaged_only_prediction_catalog_preserves_unengaged_context(tmp_path: Path):
    rows = []
    for index, (date, video_id, clicked) in enumerate(
        [
            (20220408, 10, 0),
            (20220408, 10, 0),
            (20220408, 20, 1),
            (20220409, 10, 1),
            (20220409, 20, 1),
        ]
    ):
        rows.append(
            {
                "user_id": 1,
                "video_id": video_id,
                "time_ms": 1000 * (index + 1),
                "date": date,
                "hourmin": 1200,
                "is_click": clicked,
                "is_like": 0,
                "is_follow": 0,
                "is_comment": 0,
                "is_forward": 0,
                "is_hate": 0,
                "long_view": 0,
                "play_time_ms": 1000,
                "duration_ms": 1000,
            }
        )
    path = tmp_path / "log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    trace = load_kuairand(
        [path],
        min_interactions_per_user=1,
        max_items=1,
        fit_num_days=1,
        context_hash_buckets=4,
        prediction_items_from_engaged_only=True,
    )

    assert trace.item_map == {20: 1}
    assert len(trace.interactions) == 5
    assert trace.interactions["is_prediction_item"].tolist() == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert trace.interactions["label"].tolist() == [0, 0, 1, 0, 1]


def test_history_window_prunes_old_events_without_token_truncation():
    day_ms = 86400 * 1000
    interactions = pd.DataFrame(
        {
            "date": [f"d{index}" for index in range(1, 7)],
            "user_idx": [1] * 6,
            "item_idx": list(range(1, 7)),
            "behavior": [2] * 6,
            "label": [1] * 6,
            "time_ms": [index * day_ms for index in range(6)],
        }
    )
    trace = KuaiRandTrace(interactions, 1, 6, 9, {}, {})
    plan = StreamingDataPlan(
        trace,
        [f"d{index}" for index in range(1, 7)],
        [],
        max_seq_len=16,
        history_window_days=4,
    )

    plan.init_base()
    history = plan.user_histories[1]

    assert history["item_ids"].tolist() == [2, 3, 4, 5, 6]
    assert history["timestamps"][0] == history["timestamps"][-1] - 4 * day_ms


def test_long_context_prepared_artifact_round_trip(tmp_path: Path):
    dates = [f"d{index:02d}" for index in range(16)]
    interactions = pd.DataFrame(
        {
            "date": dates,
            "user_idx": [1] * 16,
            "item_idx": [1, 2, 3, 4] * 4,
            "behavior": [2] * 16,
            "label": [1, 1, 0, 0] * 4,
            "time_ms": [1000 * index for index in range(16)],
            "is_prediction_item": [True, True, False, False] * 4,
        }
    )
    trace = KuaiRandTrace(
        interactions=interactions,
        num_users=1,
        num_items=6,
        num_behaviors=9,
        user_map={100: 1},
        item_map={10: 1, 20: 2},
        num_prediction_items=2,
        context_hash_buckets=4,
    )
    plan = StreamingDataPlan(
        trace=trace,
        base_dates=dates[:8],
        stream_dates=dates[8:],
        max_seq_len=8,
        max_items=2,
        history_window_days=4,
    )
    path = tmp_path / "prepared.npz"

    metadata = save_prepared_kuairand_plan(
        plan,
        path,
        source_paths=["synthetic.csv"],
    )
    loaded, loaded_metadata = load_prepared_kuairand_plan(path)

    assert loaded_metadata == metadata
    assert loaded.base_dates == dates[:8]
    assert loaded.stream_dates == dates[8:]
    assert loaded.num_users == 1
    assert loaded.num_items == 6
    assert loaded.num_prediction_items == 2
    assert loaded.trace.interactions["label"].tolist() == [1, 1, 0, 0] * 4


def test_exploration_split_prepared_artifact_round_trip(tmp_path: Path):
    dates = [f"d{index:02d}" for index in range(16)]
    interactions = pd.DataFrame(
        {
            "date": dates,
            "user_idx": [1] * 16,
            "item_idx": [1, 2] * 8,
            "behavior": [2] * 16,
            "label": [1, 0] * 8,
            "time_ms": [1000 * index for index in range(16)],
            "is_prediction_item": [True] * 16,
        }
    )
    trace = KuaiRandTrace(
        interactions=interactions,
        num_users=1,
        num_items=2,
        num_behaviors=9,
        user_map={100: 1},
        item_map={10: 1, 20: 2},
        num_prediction_items=2,
        context_hash_buckets=0,
    )
    plan = StreamingDataPlan(
        trace=trace,
        base_dates=dates[:4],
        stream_dates=dates[4:],
        max_seq_len=8,
        max_items=2,
        history_window_days=4,
    )
    path = tmp_path / "prepared_4plus12.npz"

    metadata = save_prepared_kuairand_plan(
        plan,
        path,
        source_paths=["synthetic.csv"],
        protocol=prepared_protocol_for_base_days(4),
    )
    loaded, loaded_metadata = load_prepared_kuairand_plan(path)

    assert loaded_metadata == metadata
    assert loaded.base_dates == dates[:4]
    assert loaded.stream_dates == dates[4:]
    assert loaded.trace.interactions["label"].tolist() == [1, 0] * 8


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
