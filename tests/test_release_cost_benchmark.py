from __future__ import annotations

import pytest

from hstu_kvcache.benchmark import (
    RELEASE_COST_CONFIGURATIONS,
    estimate_release_card_hours,
    make_random_hstu,
)


def test_release_cost_configurations_match_the_planned_table() -> None:
    assert [(c.num_layers, c.context_length, c.hidden_size, c.num_heads)
            for c in RELEASE_COST_CONFIGURATIONS] == [
        (4, 512, 128, 4),
        (6, 1024, 256, 4),
        (8, 2048, 512, 8),
        (16, 4096, 512, 8),
        (24, 8192, 512, 8),
    ]


def test_extrapolation_is_linear_and_uses_card_hours() -> None:
    estimate = estimate_release_card_hours(
        elapsed_seconds=36.0, sampled_users=1_000, target_users=10_000_000
    )
    assert estimate.seconds_per_user == pytest.approx(0.036)
    assert estimate.card_hours == pytest.approx(100.0)


def test_random_model_is_deterministic_and_untrained() -> None:
    configuration = RELEASE_COST_CONFIGURATIONS[0]
    first = make_random_hstu(configuration, seed=17, num_items=32, num_behaviors=4)
    second = make_random_hstu(configuration, seed=17, num_items=32, num_behaviors=4)
    assert first.cfg == second.cfg
    assert next(first.parameters()).equal(next(second.parameters()))
    assert not first.training
