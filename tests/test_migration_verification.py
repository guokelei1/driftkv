import pytest

from hstu_kvcache.migration import (
    FidelityContract,
    MigrationActionSpec,
    compile_verified_plan,
)


def make_records(fast_recovery: float, users: int = 60) -> list[dict]:
    records = []
    for user in range(users):
        reuse_cache = 0.5 + user * 0.001
        reuse_score_error = 0.1 + user * 0.0001
        reuse_top_error = 0.4 + user * 0.0002
        records.append(
            {
                "configs": {
                    "reuse": {
                        "cache_error_rel": reuse_cache,
                        "score_cosine": 1.0 - reuse_score_error,
                        "top100_overlap": 1.0 - reuse_top_error,
                    },
                    "fast": {
                        "cache_error_rel": reuse_cache
                        * (1.0 - fast_recovery),
                        "score_cosine": 1.0
                        - reuse_score_error * (1.0 - fast_recovery),
                        "top100_overlap": 1.0
                        - reuse_top_error * (1.0 - fast_recovery),
                    },
                    "structural": {
                        "cache_error_rel": reuse_cache * 0.15,
                        "score_cosine": 1.0 - reuse_score_error * 0.15,
                        "top100_overlap": 1.0 - reuse_top_error * 0.15,
                    },
                    "recompute": {
                        "cache_error_rel": 0.0,
                        "score_cosine": 1.0,
                        "top100_overlap": 1.0,
                    },
                }
            }
        )
    return records


def make_contract() -> FidelityContract:
    return FidelityContract(
        recovery_target=0.7,
        minimum_coverage=0.8,
        confidence_level=0.9,
        max_cost_ratio=0.3,
        bootstrap_samples=200,
        minimum_probe_users=60,
    )


def make_actions() -> tuple[MigrationActionSpec, ...]:
    return (
        MigrationActionSpec(
            name="fast",
            kind="compiled",
            required_state="normalized_capsule",
            program_path="fast.pt",
        ),
        MigrationActionSpec(
            name="structural",
            kind="structural_replay",
            required_state="history_and_capsule",
            replay_depth=8,
        ),
        MigrationActionSpec(
            name="recompute",
            kind="exact",
            required_state="raw_history",
        ),
    )


def test_verified_plan_selects_minimum_cost_certified_action():
    plan = compile_verified_plan(
        protocol="test",
        source_version="theta0",
        target_version="theta1",
        actions=make_actions(),
        records=make_records(0.8),
        cost_ratios={"fast": 0.2, "structural": 0.5, "recompute": 1.0},
        contract=make_contract(),
        seed=3,
    )

    assert plan.selected_action == "fast"
    assert plan.selection_reason == "minimum_cost_certified_within_budget"
    assert plan.fallback_actions == ("structural", "recompute")
    assert plan.next_fallback("fast") == "structural"
    assert plan.next_fallback("structural") == "recompute"
    assert plan.next_fallback("recompute") is None
    assert plan.certificate("fast").fidelity_passed
    assert plan.certificate("fast").budget_passed
    assert not plan.labels_used


def test_verified_plan_falls_back_when_fast_action_fails():
    plan = compile_verified_plan(
        protocol="test",
        source_version="theta0",
        target_version="theta1",
        actions=make_actions(),
        records=make_records(0.4),
        cost_ratios={"fast": 0.2, "structural": 0.5, "recompute": 1.0},
        contract=make_contract(),
        seed=3,
    )

    assert plan.selected_action == "structural"
    assert plan.selection_reason == "minimum_cost_certified_budget_overflow"
    assert plan.fallback_actions == ("recompute",)
    assert not plan.certificate("fast").fidelity_passed
    assert plan.certificate("structural").fidelity_passed
    assert not plan.certificate("structural").budget_passed


def test_verified_plan_requires_complete_probe_and_exact_action():
    with pytest.raises(ValueError, match="recompute"):
        compile_verified_plan(
            protocol="test",
            source_version="theta0",
            target_version="theta1",
            actions=make_actions()[:-1],
            records=make_records(0.8),
            cost_ratios={"fast": 0.2, "structural": 0.5},
            contract=make_contract(),
            seed=3,
        )

    with pytest.raises(ValueError, match="insufficient"):
        compile_verified_plan(
            protocol="test",
            source_version="theta0",
            target_version="theta1",
            actions=make_actions(),
            records=make_records(0.8, users=59),
            cost_ratios={
                "fast": 0.2,
                "structural": 0.5,
                "recompute": 1.0,
            },
            contract=make_contract(),
            seed=3,
        )
