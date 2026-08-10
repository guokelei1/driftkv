import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming.kuairand_cache_compatible import (
    OutputHead,
    _full_catalog_multi,
    _output_only_training_step,
)


def test_full_catalog_multi_matches_direct_scores():
    torch.manual_seed(13)
    weights = [torch.randn(10, 8), torch.randn(10, 8), torch.randn(10, 8)]
    hidden = torch.randn(3, 5, 8)
    positives = torch.tensor([1, 3, 5, 7, 9])

    nll, ranks = _full_catalog_multi(
        weights,
        hidden,
        positives,
        prediction_items=9,
        target_chunk=3,
        item_chunk=4,
        device=torch.device("cpu"),
        phase="test",
    )

    expected_scores = torch.stack(
        [torch.matmul(hidden[index], weights[index][1:].t()) for index in range(3)]
    )
    target_indices = (positives - 1).view(1, -1, 1).expand(3, -1, -1)
    positive_scores = expected_scores.gather(2, target_indices).squeeze(-1)
    expected_nll = torch.logsumexp(expected_scores, dim=-1) - positive_scores
    ids = torch.arange(1, 10)
    expected_ranks = 1 + (
        (expected_scores >= positive_scores.unsqueeze(-1))
        & (ids.view(1, 1, -1) != positives.view(1, -1, 1))
    ).sum(dim=-1)

    assert torch.allclose(nll, expected_nll, atol=1e-6, rtol=1e-6)
    assert torch.equal(ranks, expected_ranks)


def test_output_only_step_does_not_change_backbone():
    torch.manual_seed(17)
    model = HSTU(
        HSTUConfig(
            num_items=12,
            num_prediction_items=9,
            num_behaviors=3,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=5,
            input_dropout=0.0,
        )
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = OutputHead(model.item_emb.weight[:10])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    batch = {
        "item_ids": torch.tensor([[1, 2, 3, 4, 0], [2, 3, 5, 6, 7]]),
        "behaviors": torch.tensor([[1, 1, 2, 2, 0], [1, 2, 1, 2, 1]]),
        "time_deltas": torch.tensor(
            [[0.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 2.0, 1.0, 1.0]]
        ),
        "lengths": torch.tensor([4, 5]),
        "labels": torch.tensor(
            [[False, True, True, True, False], [False, True, True, True, True]]
        ),
        "train_mask": torch.tensor(
            [[False, True, True, True, False], [False, True, True, True, True]]
        ),
    }
    before_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before_head = head.weight.detach().clone()

    loss, targets = _output_only_training_step(
        model,
        head,
        batch,
        optimizer,
        torch.device("cpu"),
        negative_count=3,
    )

    assert loss > 0.0
    assert targets > 0
    assert all(torch.equal(value, before_model[name]) for name, value in model.state_dict().items())
    assert not torch.equal(head.weight, before_head)
