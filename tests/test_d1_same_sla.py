from __future__ import annotations

import pytest

from hstu_kvcache.migration.d1_same_sla import (
    build_d1_same_sla_candidate_manifest,
    select_d1_same_sla_families,
)


def _config(error: float, cost: float) -> dict[str, float]:
    return {
        "cache_error_rel": error,
        "migration_ratio_to_recompute": cost,
    }


def test_manifest_covers_every_declared_structural_family() -> None:
    manifest, first_hash = build_d1_same_sla_candidate_manifest(3)
    repeated, second_hash = build_d1_same_sla_candidate_manifest(3)
    actions = manifest["actions"]
    families = manifest["families"]

    assert manifest == repeated
    assert first_hash == second_hash
    assert set(actions) == {
        "current_projection",
        "fixed_suffix_d1",
        "fixed_suffix_d2",
        "plain_prefix_d1",
        "plain_prefix_d2",
        "recent_rectangle_d1_r25",
        "recent_rectangle_d1_r50",
        "recent_rectangle_d2_r25",
        "recent_rectangle_d2_r50",
        "recent_rectangle_d3_r25",
        "recent_rectangle_d3_r50",
        "contiguous_interval_l1_l1",
        "contiguous_interval_l1_l2",
        "contiguous_interval_l2_l2",
        "contiguous_interval_l2_l3",
        "contiguous_interval_l3_l3",
        "recompute",
    }
    assert set(families) == {
        "fixed_deep_suffix",
        "plain_progressive_prefix",
        "recent_token_rectangles",
        "arbitrary_contiguous_intervals",
    }
    assert all(
        value["candidate_names"][0] == "current_projection"
        and value["fallback"] == "recompute"
        for value in families.values()
    )
    assert all(
        "compiled" not in name
        and "proposed" not in name
        and "residual" not in name
        for name in actions
    )


def test_each_family_selects_by_probe_cost_or_exact_fallback() -> None:
    manifest, _ = build_d1_same_sla_candidate_manifest(
        2,
        rectangle_depth_fractions=(0.5, 1.0),
        rectangle_recent_fractions=(0.5,),
    )
    probe = {
        "reuse": _config(1.0, 0.0),
        "recompute": _config(0.0, 1.0),
    }
    test = {
        "reuse": _config(1.1, 0.0),
        "recompute": _config(0.1, 1.0),
    }
    for index, name in enumerate(manifest["actions"]):
        if name == "recompute":
            continue
        probe[name] = _config(0.8, 0.1 + index / 100)
        test[name] = _config(0.75, 0.2 + index / 100)
    probe["fixed_suffix_d1"] = _config(0.4, 0.45)
    probe["recent_rectangle_d2_r50"] = _config(0.35, 0.55)
    probe["contiguous_interval_l2_l2"] = _config(0.3, 0.6)
    test["fixed_suffix_d1"] = _config(0.45, 0.46)
    test["recent_rectangle_d2_r50"] = _config(0.4, 0.56)
    test["contiguous_interval_l2_l2"] = _config(0.35, 0.61)

    selected = select_d1_same_sla_families(
        probe,
        test,
        manifest,
    )["families"]

    assert selected["fixed_deep_suffix"]["selected"] == (
        "fixed_suffix_d1"
    )
    assert not selected["fixed_deep_suffix"]["fallback_used"]
    assert selected["plain_progressive_prefix"]["selected"] == (
        "recompute"
    )
    assert selected["plain_progressive_prefix"]["fallback_used"]
    assert selected["recent_token_rectangles"]["selected"] == (
        "recent_rectangle_d2_r50"
    )
    assert selected["arbitrary_contiguous_intervals"]["selected"] == (
        "contiguous_interval_l2_l2"
    )
    assert (
        selected["fixed_deep_suffix"]["test"][
            "cache_fidelity_recovery"
        ]
        == pytest.approx(0.65)
    )
