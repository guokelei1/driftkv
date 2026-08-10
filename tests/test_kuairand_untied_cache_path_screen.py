from __future__ import annotations

import numpy as np
import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming.kuairand_root_cause import _training_step
from hstu_kvcache.streaming.kuairand_untied_cache_path_screen import (
    _bootstrap,
    _cache_producer_parameters,
    _untied_source_state,
)


def _model(tied: bool) -> HSTU:
    return HSTU(
        HSTUConfig(
            num_items=50,
            num_prediction_items=40,
            num_behaviors=4,
            hidden_size=16,
            num_layers=3,
            num_heads=2,
            head_dim=8,
            input_dropout=0.0,
            tie_item_embeddings=tied,
        )
    )


def test_untied_bootstrap_preserves_target_weighting() -> None:
    targets = np.concatenate((np.asarray([1.0]), np.full(99, 99.0)))
    sums = np.concatenate((np.asarray([1.0]), np.zeros(99)))
    lower, upper = _bootstrap(targets, sums, 2000, 17)
    assert lower == 0.0
    assert upper < 0.001


def test_untied_source_preserves_tied_outputs_and_cache(tmp_path) -> None:
    torch.manual_seed(11)
    tied = _model(True).eval()
    checkpoint = tmp_path / "tied.pt"
    torch.save(tied.state_dict(), checkpoint)
    untied = _model(False).eval()
    untied.load_state_dict(
        _untied_source_state(
            {"source": {"tied_theta1": {"path": str(checkpoint)}}},
            untied,
        )
    )
    items = torch.randint(1, 51, (2, 7))
    behaviors = torch.randint(1, 5, (2, 7))
    deltas = torch.rand(2, 7)
    candidates = torch.randint(1, 41, (2, 9))
    tied_hidden, tied_cache = tied(items, behaviors, deltas, return_kv=True)
    untied_hidden, untied_cache = untied(items, behaviors, deltas, return_kv=True)
    assert torch.equal(tied_hidden, untied_hidden)
    assert torch.equal(tied_cache.k, untied_cache.k)
    assert torch.equal(tied_cache.v, untied_cache.v)
    assert torch.equal(
        tied.score_candidates(tied_hidden, candidates),
        untied.score_candidates(untied_hidden, candidates),
    )


def test_cache_producer_group_freezes_output_and_final_readout() -> None:
    model = _model(False)
    _, names = _cache_producer_parameters(model)
    last = model.cfg.num_layers - 1
    assert "item_emb.weight" in names
    assert "output_emb.weight" not in names
    assert "final_norm.weight" not in names
    assert not any(name.startswith(f"blocks.{last}.attn.q_proj.") for name in names)
    assert not any(name.startswith(f"blocks.{last}.attn.out_proj.") for name in names)
    assert not any(name.startswith(f"blocks.{last}.gate_proj.") for name in names)


def test_untied_training_ignores_context_only_invalid_targets() -> None:
    model = _model(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss, targets = _training_step(
        model,
        {
            "item_ids": torch.tensor([[1, 50, 2]]),
            "behaviors": torch.tensor([[1, 1, 1]]),
            "time_deltas": torch.tensor([[0.0, 1.0, 1.0]]),
            "lengths": torch.tensor([3]),
            "labels": torch.tensor([[0, 0, 1]]),
            "train_mask": torch.tensor([[False, True, True]]),
        },
        optimizer,
        torch.device("cpu"),
        8,
    )
    assert targets == 1
    assert loss > 0.0
