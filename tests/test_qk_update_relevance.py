from __future__ import annotations

import numpy as np
import torch

from hstu_kvcache.streaming.qk_protocol_sweep_runner import (
    nested_uniform_candidate_ids,
)
from hstu_kvcache.streaming.qk_update_relevance_runner import (
    COHORTS,
    SparseRelationIndex,
    _admission_gate,
    _relation_index,
    context_hard_candidate_ids,
    relation_cohort_masks,
)


def test_sparse_relation_index_aggregates_context_support() -> None:
    index = _relation_index(
        [np.asarray([2, 2, 3, 3], dtype=np.uint32)],
        [np.asarray([7, 7, 7, 9], dtype=np.uint32)],
    )
    targets, counts = index.lookup(np.asarray([2, 3], dtype=np.uint32))
    assert isinstance(index, SparseRelationIndex)
    assert targets.tolist() == [7, 9]
    assert counts.tolist() == [3, 1]


def test_relation_cohorts_are_fixed_from_training_support_and_offset() -> None:
    masks = relation_cohort_masks(
        np.asarray([0, 3, 15, 20]),
        np.asarray([0, 1, 2, 3]),
        np.asarray([1, 0, 2, 0]),
        np.asarray([1, 2, 0, 0]),
        np.asarray([0, 0, 1, 0]),
    )
    assert tuple(masks) == COHORTS
    assert masks["context_h32_support_ge1"].tolist() == [True, True, False, False]
    assert masks["context_h32_support_ge2"].tolist() == [False, True, False, False]
    assert masks["successor_or_copositive"].tolist() == [True, True, True, False]
    assert masks["context_h32_support_ge1_first_positive"].tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_context_hard_candidates_are_nested_unique_and_filled() -> None:
    positives = torch.tensor([7, 11])
    uniform = nested_uniform_candidate_ids(
        positives,
        num_prediction_items=100,
        maximum_negative_count=5,
        seed=29,
    )
    values, selected = context_hard_candidate_ids(
        positives,
        np.asarray([7, 9, 12], dtype=np.uint32),
        np.arange(1, 101, dtype=np.int64),
        uniform,
        maximum_negative_count=5,
    )
    assert values.shape == (2, 6)
    assert values[:, 0].tolist() == [7, 11]
    assert values[0, 1:3].tolist() == [9, 12]
    assert values[1, 1:4].tolist() == [7, 9, 12]
    assert selected == 5
    for row in values:
        assert len(torch.unique(row)) == len(row)


def test_admission_gate_handles_zero_reuse_ndcg() -> None:
    summary = {
        "full_catalog": {
            "rolling_next_item": {
                "cohorts": {
                    "context_h32_support_ge1": {
                        "positive_targets": 10,
                        "gaps": {
                            "ndcg_at_10": {
                                "relative_percent": None,
                                "positive_direction_with_ci": False,
                            },
                            "mrr": {"positive_direction_with_ci": False},
                        },
                    }
                }
            }
        }
    }
    result = _admission_gate(
        summary,
        candidates=[
            {
                "mode": "rolling_next_item",
                "cohort": "context_h32_support_ge1",
            }
        ],
        minimum_targets=1,
        relative_range=[5.0, 10.0],
    )
    assert result["status"] == "no_preferred_full_catalog_gap"
