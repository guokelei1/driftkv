import numpy as np
import torch

from hstu_kvcache.streaming.kuairand_next_item_hard_update import (
    _logged_negatives,
)
from hstu_kvcache.streaming.kuairand_next_item_rollout import (
    nested_logged_unengaged_candidate_ids,
)


def test_logged_unengaged_candidates_are_nested_and_label_clean() -> None:
    items = np.asarray([11, 21, 12, 22, 23, 13, 24], dtype=np.int64)
    labels = np.asarray([True, False, True, False, False, True, False])
    positives = torch.tensor([11, 12, 13])
    fallback = torch.arange(1, 101)
    candidates = nested_logged_unengaged_candidate_ids(
        positives,
        items,
        labels,
        fallback,
        num_prediction_items=100,
        maximum_negative_count=5,
        seed=None,
    )
    assert candidates.shape == (3, 6)
    assert torch.equal(candidates[:, 0], positives)
    assert candidates[0, 1].item() == 21
    assert candidates[1, 1].item() == 21
    assert candidates[2, 1].item() == 23
    for row in candidates:
        assert len(torch.unique(row)) == len(row)
        assert not any(value in {11, 12, 13} for value in row[1:].tolist())


def test_logged_unengaged_seed_is_deterministic() -> None:
    items = np.arange(1, 41, dtype=np.int64)
    labels = np.zeros(40, dtype=np.bool_)
    labels[[3, 19]] = True
    positives = torch.from_numpy(items[labels].copy())
    fallback = torch.arange(1, 101)
    first = nested_logged_unengaged_candidate_ids(
        positives,
        items,
        labels,
        fallback,
        num_prediction_items=100,
        maximum_negative_count=20,
        seed=61031,
    )
    second = nested_logged_unengaged_candidate_ids(
        positives,
        items,
        labels,
        fallback,
        num_prediction_items=100,
        maximum_negative_count=20,
        seed=61031,
    )
    assert torch.equal(first, second)


def test_training_logged_negatives_exclude_engaged_items() -> None:
    torch.manual_seed(17)
    items = torch.tensor([[11, 21, 12, 22, 11, 23, 0]])
    labels = torch.tensor([[True, False, True, False, False, False, False]])
    lengths = torch.tensor([6])
    targets = torch.tensor([[11, 12, 11, 23, 21, 22]])
    negatives = _logged_negatives(
        items,
        labels,
        lengths,
        targets,
        count=20,
        num_prediction_items=100,
    )
    assert negatives.shape == (1, 6, 20)
    assert set(negatives.flatten().tolist()) <= {21, 22, 23}
