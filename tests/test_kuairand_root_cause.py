import numpy as np
import pandas as pd
import torch

from hstu_kvcache.data import KuaiRandTrace, StreamingDataPlan
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming.kuairand_root_cause import (
    _evaluation_sequence,
    _full_catalog_pair,
    _length_matched_donor,
)


def test_evaluation_sequence_predicts_unseen_day_from_prior_history():
    interactions = pd.DataFrame(
        {
            "date": ["d1", "d1", "d2", "d2", "d3", "d3"],
            "user_idx": [1] * 6,
            "item_idx": [1, 2, 3, 4, 5, 6],
            "behavior": [1, 2, 3, 4, 5, 6],
            "label": [0, 1, 1, 1, 1, 0],
            "time_ms": [1000, 2000, 3000, 4000, 5000, 6000],
            "time_delta": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    trace = KuaiRandTrace(interactions, 1, 6, 9, {10: 1}, {value: value for value in range(1, 7)})
    plan = StreamingDataPlan(trace, ["d1"], ["d2", "d3"], max_seq_len=8)
    plan.init_base()
    plan.ingest_day("d2")

    sequence = _evaluation_sequence(plan, 1, "d3")

    assert sequence["prefix"]["item_ids"].tolist() == [1, 2, 3]
    assert sequence["suffix"]["item_ids"].tolist() == [4, 5]
    assert sequence["targets"].tolist() == [5, 6]
    assert sequence["labels"].tolist() == [True, False]


def test_full_catalog_pair_matches_direct_scores():
    torch.manual_seed(3)
    model = HSTU(
        HSTUConfig(
            num_items=11,
            num_prediction_items=9,
            num_behaviors=3,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            input_dropout=0.0,
        )
    )
    hidden = torch.randn(2, 5, 8)
    positives = torch.tensor([1, 3, 5, 7, 9])

    nll, ranks = _full_catalog_pair(
        model,
        hidden,
        positives,
        target_chunk=3,
        item_chunk=4,
        device=torch.device("cpu"),
        phase="test",
    )

    scores = torch.matmul(hidden, model.item_emb.weight[1:10].t())
    expected_nll = torch.logsumexp(scores, dim=-1) - scores.gather(
        2, (positives - 1).view(1, -1, 1).expand(2, -1, -1)
    ).squeeze(-1)
    positive_scores = scores.gather(
        2, (positives - 1).view(1, -1, 1).expand(2, -1, -1)
    ).squeeze(-1)
    ids = torch.arange(1, 10)
    expected_ranks = 1 + (
        (scores >= positive_scores.unsqueeze(-1))
        & (ids.view(1, 1, -1) != positives.view(1, -1, 1))
    ).sum(dim=-1)

    assert torch.allclose(nll, expected_nll, atol=1e-6, rtol=1e-6)
    assert torch.equal(ranks, expected_ranks)


def test_length_matched_donor_preserves_requested_extent():
    sequence = {
        "item_ids": np.array([1, 2]),
        "behaviors": np.array([3, 4]),
        "time_deltas": np.array([0.0, 1.0], dtype=np.float32),
    }

    donor = _length_matched_donor(sequence, 5)

    assert donor["item_ids"].tolist() == [1, 2, 1, 2, 1]
    assert all(len(value) == 5 for value in donor.values())
