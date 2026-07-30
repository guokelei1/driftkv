from __future__ import annotations

import pytest

from hstu_kvcache.migration.design3_residency import (
    D3ResidencyGroup,
    D3ResidencyPlan,
    D3RouteGranularity,
    D3RouteStageProfile,
    _best_route_interleaving_dp,
    _group_stages,
    bounded_flowshop_makespan,
    d3_stack_revision_sha256,
    deterministic_route_interleavings,
    select_residency_plan,
)


def _profile(
    route: str,
    *,
    input_seconds: float,
    compute_seconds: float,
    output_seconds: float,
    reserved: int = 60,
    pinned: int = 10,
    granularity: int = 8,
    source: str = "a",
) -> D3RouteStageProfile:
    return D3RouteStageProfile(
        route=route,
        granularity=D3RouteGranularity(
            input_segment_records=granularity,
            compute_batch_records=granularity,
            output_segment_records=granularity,
        ),
        reference_records=64,
        input_seconds_by_rank=(input_seconds, input_seconds * 0.9),
        compute_seconds_by_rank=(
            compute_seconds,
            compute_seconds * 0.9,
        ),
        output_seconds_by_rank=(
            output_seconds,
            output_seconds * 0.9,
        ),
        peak_hbm_reserved_bytes_by_rank=(reserved, reserved - 1),
        pinned_bytes_by_rank=(pinned, pinned),
        sample_groups=3,
        source_sha256=source * 64,
    )


def test_bounded_flowshop_respects_lookahead_and_output_credit() -> None:
    assert bounded_flowshop_makespan(
        ((2.0, 5.0, 3.0), (4.0, 1.0, 6.0), (1.0, 2.0, 1.0))
    ) == pytest.approx(17.0)


def test_route_interleavings_preserve_each_route_order() -> None:
    groups = (
        D3ResidencyGroup(0, "compiled", (64, 64)),
        D3ResidencyGroup(1, "compiled", (64, 64)),
        D3ResidencyGroup(2, "exact", (64, 64)),
        D3ResidencyGroup(3, "exact", (64, 64)),
    )
    orders = deterministic_route_interleavings(groups)
    assert len(orders) == 6
    for order in orders:
        assert tuple(value for value in order if value < 2) == (0, 1)
        assert tuple(value for value in order if value >= 2) == (2, 3)


def test_selector_uses_route_asymmetry_and_roundtrips(tmp_path) -> None:
    groups = (
        D3ResidencyGroup(0, "compiled", (64, 64)),
        D3ResidencyGroup(1, "compiled", (64, 64)),
        D3ResidencyGroup(2, "exact", (64, 64)),
    )
    plan = select_residency_plan(
        groups,
        {
            "compiled": (
                _profile(
                    "compiled",
                    input_seconds=5.0,
                    compute_seconds=1.0,
                    output_seconds=1.0,
                ),
            ),
            "exact": (
                _profile(
                    "exact",
                    input_seconds=0.0,
                    compute_seconds=10.0,
                    output_seconds=0.0,
                ),
            ),
        },
        (100, 100),
        group_plan_sha256="d" * 64,
        stack_revision_sha256="e" * 64,
        hbm_margin_bytes=10,
        pinned_limit_bytes_by_rank=(100, 100),
    )
    assert plan.original_group_order == (0, 1, 2)
    assert plan.launch_order == (2, 0, 1)
    assert plan.predicted_makespan_seconds == pytest.approx(17.0)
    output = tmp_path / "plan.json"
    plan.write(output)
    restored = D3ResidencyPlan.load(output)
    assert restored == plan
    assert restored.content_sha256 == plan.content_sha256
    assert restored.hbm_total_bytes_by_rank == (100, 100)
    assert restored.pinned_limit_bytes_by_rank == (100, 100)
    assert (
        restored.compiled_profile.source_sha256
        == restored.exact_profile.source_sha256
    )
    tampered = plan.to_dict()
    tampered["launch_order"] = [0, 2, 1]
    with pytest.raises(ValueError, match="hash differs"):
        D3ResidencyPlan.from_dict(tampered)


def test_selector_capacity_and_near_tie_prefer_smaller_plan() -> None:
    groups = (
        D3ResidencyGroup(0, "compiled", (64, 64)),
        D3ResidencyGroup(1, "exact", (64, 64)),
    )
    fast_large = _profile(
        "compiled",
        input_seconds=1.0,
        compute_seconds=1.0,
        output_seconds=1.0,
        reserved=89,
        pinned=20,
        granularity=16,
    )
    stable_small = _profile(
        "compiled",
        input_seconds=1.02,
        compute_seconds=1.0,
        output_seconds=1.0,
        reserved=70,
        pinned=10,
        source="b",
    )
    exact = _profile(
        "exact",
        input_seconds=0.1,
        compute_seconds=2.0,
        output_seconds=0.5,
        reserved=60,
        pinned=5,
        source="b",
    )
    exact_fast_source = _profile(
        "exact",
        input_seconds=0.1,
        compute_seconds=2.0,
        output_seconds=0.5,
        reserved=60,
        pinned=5,
    )
    plan = select_residency_plan(
        groups,
        {
            "compiled": (fast_large, stable_small),
            "exact": (exact_fast_source, exact),
        },
        (100, 100),
        group_plan_sha256="d" * 64,
        stack_revision_sha256="e" * 64,
        hbm_margin_bytes=10,
        pinned_limit_bytes_by_rank=(100, 100),
    )
    assert plan.compiled == stable_small.granularity
    assert plan.projected_peak_hbm_reserved_bytes_by_rank == (70, 69)
    with pytest.raises(
        ValueError,
        match="exceed capacity",
    ):
        select_residency_plan(
            groups,
            {
                "compiled": (fast_large,),
                "exact": (exact_fast_source,),
            },
            (90, 90),
            group_plan_sha256="d" * 64,
            stack_revision_sha256="e" * 64,
            hbm_margin_bytes=10,
            pinned_limit_bytes_by_rank=(100, 100),
        )


def test_selector_rejects_cross_run_profile_synthesis() -> None:
    groups = (
        D3ResidencyGroup(0, "compiled", (64, 64)),
        D3ResidencyGroup(1, "exact", (64, 64)),
    )
    with pytest.raises(ValueError, match="joint source"):
        select_residency_plan(
            groups,
            {
                "compiled": (
                    _profile(
                        "compiled",
                        input_seconds=1.0,
                        compute_seconds=1.0,
                        output_seconds=1.0,
                        source="a",
                    ),
                ),
                "exact": (
                    _profile(
                        "exact",
                        input_seconds=1.0,
                        compute_seconds=1.0,
                        output_seconds=1.0,
                        source="b",
                    ),
                ),
            },
            (100, 100),
            group_plan_sha256="d" * 64,
            stack_revision_sha256="e" * 64,
            hbm_margin_bytes=10,
            pinned_limit_bytes_by_rank=(100, 100),
        )


def test_selector_near_tie_is_anchored_to_global_minimum() -> None:
    groups = (
        D3ResidencyGroup(0, "compiled", (64, 64)),
        D3ResidencyGroup(1, "exact", (64, 64)),
    )
    profiles: dict[str, list[D3RouteStageProfile]] = {
        "compiled": [],
        "exact": [],
    }
    for source, compute, reserved, pinned in (
        ("a", 7.0, 90, 30),
        ("b", 7.2, 80, 20),
        ("c", 7.4, 70, 10),
    ):
        profiles["compiled"].append(
            _profile(
                "compiled",
                input_seconds=1.0,
                compute_seconds=compute,
                output_seconds=1.0,
                reserved=reserved,
                pinned=pinned,
                source=source,
            )
        )
        profiles["exact"].append(
            _profile(
                "exact",
                input_seconds=0.0,
                compute_seconds=1.0,
                output_seconds=0.0,
                reserved=reserved,
                pinned=pinned,
                source=source,
            )
        )
    plan = select_residency_plan(
        groups,
        profiles,
        (120, 120),
        group_plan_sha256="d" * 64,
        stack_revision_sha256="e" * 64,
        hbm_margin_bytes=10,
        pinned_limit_bytes_by_rank=(100, 100),
        selector_tie_fraction=0.03,
    )
    assert plan.compiled_profile.source_sha256 == "b" * 64


def test_tail_stage_scaling_uses_discrete_segment_counts() -> None:
    profile = D3RouteStageProfile(
        route="compiled",
        granularity=D3RouteGranularity(8, 16, 32),
        reference_records=64,
        input_seconds_by_rank=(8.0,),
        compute_seconds_by_rank=(4.0,),
        output_seconds_by_rank=(2.0,),
        peak_hbm_reserved_bytes_by_rank=(10,),
        pinned_bytes_by_rank=(5,),
        sample_groups=2,
        source_sha256="a" * 64,
    )
    assert _group_stages(
        D3ResidencyGroup(0, "compiled", (9,)),
        profile,
    ) == pytest.approx((2.0, 1.0, 1.0))


def test_beam_dp_matches_exhaustive_small_oracle() -> None:
    groups = tuple(
        [
            D3ResidencyGroup(index, "compiled", (64,))
            for index in range(4)
        ]
        + [
            D3ResidencyGroup(index, "exact", (64,))
            for index in range(4, 8)
        ]
    )
    stages = {
        group.ordinal: (
            (4.0, 1.0, 2.0)
            if group.route == "compiled"
            else (0.5, 5.0, 0.5)
        )
        for group in groups
    }
    exhaustive_order = min(
        deterministic_route_interleavings(groups),
        key=lambda order: (
            bounded_flowshop_makespan(
                tuple(stages[value] for value in order)
            ),
            order,
        ),
    )
    beam_order, beam_makespan = _best_route_interleaving_dp(
        groups,
        stages,
        beam_width=128,
    )
    assert beam_order == exhaustive_order
    assert beam_makespan == pytest.approx(
        bounded_flowshop_makespan(
            tuple(stages[value] for value in exhaustive_order)
        )
    )


def test_large_route_space_is_plannable_without_enumeration() -> None:
    groups = tuple(
        [
            D3ResidencyGroup(index, "compiled", (64,))
            for index in range(30)
        ]
        + [
            D3ResidencyGroup(index, "exact", (64,))
            for index in range(30, 60)
        ]
    )
    order = deterministic_route_interleavings(
        groups,
        max_interleavings=64,
    )[0]
    assert len(order) == 60
    assert tuple(value for value in order if value < 30) == tuple(
        range(30)
    )
    assert tuple(value for value in order if value >= 30) == tuple(
        range(30, 60)
    )
    plan = select_residency_plan(
        groups,
        {
            "compiled": (
                _profile(
                    "compiled",
                    input_seconds=4.0,
                    compute_seconds=1.0,
                    output_seconds=2.0,
                ),
            ),
            "exact": (
                _profile(
                    "exact",
                    input_seconds=0.5,
                    compute_seconds=5.0,
                    output_seconds=0.5,
                ),
            ),
        },
        (100, 100),
        group_plan_sha256="d" * 64,
        stack_revision_sha256="e" * 64,
        hbm_margin_bytes=10,
        pinned_limit_bytes_by_rank=(100, 100),
        max_interleavings=64,
    )
    assert len(plan.launch_order) == 60


def test_stack_revision_binds_execution_identity() -> None:
    inputs = {
        "bindings": {"work": "a"},
        "world_size": 1,
        "group_records_per_rank": 64,
        "records": 64,
        "counts": {"compiled": 32, "exact": 32},
        "embedding_scale_role": "capacity",
        "device_names": ("A40",),
        "source_checkpoint_sha256": ("a" * 64,),
        "target_checkpoint_sha256": ("b" * 64,),
        "capacity_groups": {"compiled": 64, "exact": 64},
    }
    first = d3_stack_revision_sha256(
        **inputs,
        execution_identity={"runtime": "first"},
    )
    second = d3_stack_revision_sha256(
        **inputs,
        execution_identity={"runtime": "second"},
    )
    assert first != second
