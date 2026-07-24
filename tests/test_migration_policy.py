import pytest

from hstu_kvcache.migration import (
    cache_fidelity_recovery,
    select_maximum_fidelity_actions,
    select_minimum_cost_actions,
)


def configs():
    return {
        "reuse": {
            "cache_error_rel": 1.0,
            "migration_ratio_to_recompute": 0.0,
        },
        "cheap": {
            "cache_error_rel": 0.7,
            "migration_ratio_to_recompute": 0.2,
        },
        "partial": {
            "cache_error_rel": 0.35,
            "migration_ratio_to_recompute": 0.5,
        },
        "recompute": {
            "cache_error_rel": 0.0,
            "migration_ratio_to_recompute": 1.0,
        },
    }


def test_cache_fidelity_recovery_uses_full_reference():
    assert cache_fidelity_recovery(1.0, 1.0, 0.0) == 0.0
    assert cache_fidelity_recovery(0.35, 1.0, 0.0) == pytest.approx(0.65)
    assert cache_fidelity_recovery(0.0, 1.0, 0.0) == 1.0


def test_minimum_cost_selector_meets_fidelity_target():
    selected = select_minimum_cost_actions(
        configs(),
        (0.25, 0.5, 0.9),
    )

    assert selected["0.25"]["selected"] == "cheap"
    assert selected["0.5"]["selected"] == "partial"
    assert selected["0.9"]["selected"] == "recompute"


def test_maximum_fidelity_selector_respects_budget():
    selected = select_maximum_fidelity_actions(
        configs(),
        (0.1, 0.3, 0.6, 1.0),
    )

    assert selected["0.1"]["selected"] == "reuse"
    assert selected["0.3"]["selected"] == "cheap"
    assert selected["0.6"]["selected"] == "partial"
    assert selected["1.0"]["selected"] == "recompute"


def test_selector_rejects_invalid_targets():
    with pytest.raises(ValueError):
        select_minimum_cost_actions(configs(), (0.0,))
    with pytest.raises(ValueError):
        select_maximum_fidelity_actions(configs(), (1.1,))
