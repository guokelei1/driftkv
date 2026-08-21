from __future__ import annotations

import math

import pytest
import torch

from hstu_kvcache.data import build_q_main_rank_decay
from hstu_kvcache.models import conditional_reranking_loss


def test_qmain_records_causal_rank_weight_and_log_probability() -> None:
    proposal = build_q_main_rank_decay([7, 3, 11, 5], causal_cutoff=100, decay=1.2)
    assert proposal.item_ids == (7, 3, 11, 5)
    assert [row.proposal_rank for row in proposal.candidates] == [1, 2, 3, 4]
    assert math.isclose(sum(row.weight for row in proposal.candidates), 1.0)
    assert all(math.isclose(row.log_q_main, math.log(row.weight)) for row in proposal.candidates)
    assert all(row.causal_cutoff == 100 for row in proposal.candidates)


def test_negative_rule_excludes_only_current_positive_and_keeps_seen_items() -> None:
    proposal = build_q_main_rank_decay([7, 3, 11, 5], causal_cutoff=100)
    negatives = proposal.negatives_for(current_positive_id=11)
    assert [row.item_id for row in negatives] == [7, 3, 5]
    # A historical/seen item has no special exclusion path.
    assert 3 in [row.item_id for row in negatives]
    proposal.validate_query_timestamp(100)
    proposal.validate_query_timestamp(101)
    with pytest.raises(ValueError):
        proposal.validate_query_timestamp(99)


def test_qmain_builder_rejects_duplicate_or_noncausal_inputs() -> None:
    with pytest.raises(ValueError):
        build_q_main_rank_decay([1, 1], causal_cutoff=5)
    with pytest.raises(ValueError):
        build_q_main_rank_decay([1, 2], causal_cutoff=5, decay=0.0)


def test_conditional_loss_stays_within_candidate_panel() -> None:
    scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    loss = conditional_reranking_loss(scores, torch.tensor([0, 1]))
    assert float(loss.detach()) < 0.2
    loss.backward()
    assert scores.grad is not None
