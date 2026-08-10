from __future__ import annotations

import numpy as np

from hstu_kvcache.streaming.qk_alignment_runner import (
    COHORTS,
    _alignment_gate,
    alignment_cohort_masks,
)


def test_alignment_cohorts_use_fixed_training_visible_attributes() -> None:
    masks = alignment_cohort_masks(
        np.asarray([0, 1, 3, 7, 15, 20]),
        np.asarray([0, 1, 2, 3, 4, 5]),
        np.asarray([1, 1, 1, 0, 0, 1], dtype=np.bool_),
        np.asarray([1, 1, 0, 1, 0, 1], dtype=np.bool_),
        4,
    )
    assert tuple(masks) == COHORTS
    assert masks["first_positive"].tolist() == [True, False, False, False, False, False]
    assert masks["offset_lt_4"].tolist() == [True, True, True, False, False, False]
    assert masks["early_8_train_positive_supported"].tolist() == [
        True,
        True,
        False,
        True,
        False,
        False,
    ]
    assert masks["prefix_positive_mass_q4"].all()


def test_alignment_gate_requires_target_volume_and_supporting_mrr() -> None:
    def value(relative: float, targets: int, mrr: bool) -> dict[str, object]:
        return {
            "positive_targets": targets,
            "gaps": {
                "ndcg_at_10": {
                    "relative_percent": relative,
                    "positive_direction_with_ci": True,
                },
                "mrr": {"positive_direction_with_ci": mrr},
            },
        }

    summary = {
        "rolling_next_item": {
            "cohorts": {
                "first_positive": value(7.0, 6000, True),
                "offset_lt_4": value(8.0, 4000, True),
            }
        }
    }
    gate = _alignment_gate(
        summary,
        candidates=[
            {"mode": "rolling_next_item", "cohort": "first_positive"},
            {"mode": "rolling_next_item", "cohort": "offset_lt_4"},
        ],
        minimum_targets=5000,
        relative_range=[5.0, 10.0],
    )
    assert gate["status"] == "aligned_protocol_found"
    assert gate["admitted"][0]["cohort"] == "first_positive"
    assert gate["all_checked"][1]["preferred_gap_passed"] is False
