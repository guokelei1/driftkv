import numpy as np

from hstu_kvcache.streaming.kuairand_next_item_rolling_chain import rolling_sequences


def _history(length: int) -> dict[str, np.ndarray]:
    values = np.arange(length, dtype=np.int64)
    return {
        "item_ids": values + 1,
        "behaviors": np.ones(length, dtype=np.int64),
        "time_deltas": values.astype(np.float32),
        "labels": np.ones(length, dtype=np.int64),
        "timestamps": values,
    }


def test_rolling_context_assigns_each_target_once() -> None:
    sequences = rolling_sequences([_history(15)], 6, 4, [3])
    targets = []
    for sequence in sequences:
        target_items = sequence["item_ids"][sequence["train_mask"]]
        targets.extend(target_items.tolist())
        assert len(sequence["item_ids"]) <= 10
    assert targets == list(range(4, 16))


def test_rolling_context_preserves_preceding_history() -> None:
    sequences = rolling_sequences([_history(20)], 6, 4, [7])
    assert sequences[0]["item_ids"].tolist() == list(range(2, 12))
    assert sequences[0]["train_mask"].tolist() == [False] * 6 + [True] * 4
    assert sequences[1]["item_ids"].tolist() == list(range(6, 16))
    assert sequences[-1]["item_ids"].tolist() == list(range(14, 21))
