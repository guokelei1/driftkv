import pytest

from hstu_kvcache.migration.lifecycle import (
    BalancedLifecyclePolicy,
    CacheLifecycleState,
    LifecyclePolicy,
    LinearSketchRiskCalibration,
    MonotoneRiskCalibration,
    SketchLifecyclePolicy,
    fit_monotone_risk_calibration,
)


def calibration() -> MonotoneRiskCalibration:
    return MonotoneRiskCalibration(
        correction_upper_bounds=(0.1, 0.2, 0.4),
        one_hop_risks=(0.02, 0.05, 0.1),
        propagation_gain=1.5,
        quantile=0.9,
    )


def test_exact_state_and_migration_advance() -> None:
    policy = LifecyclePolicy(
        max_migration_depth=3,
        risk_threshold=0.2,
        calibration=calibration(),
    )
    state = CacheLifecycleState.exact(7, 0)
    decision = policy.decide(state, 1, 0.15)
    assert decision.action == "migrate"
    assert decision.reason == "risk_accepted"
    migrated = policy.advance(state, decision)
    assert migrated == CacheLifecycleState(
        record_id=7,
        served_version=1,
        last_exact_version=0,
        migration_depth=1,
        risk_score=0.05,
        state_kind="migrated",
    )


def test_zero_record_id_matches_frozen_manifest() -> None:
    assert CacheLifecycleState.exact(0, 0).record_id == 0


def test_risk_threshold_forces_exact_and_resets() -> None:
    policy = LifecyclePolicy(
        max_migration_depth=4,
        risk_threshold=0.12,
        calibration=calibration(),
    )
    state = CacheLifecycleState(
        record_id=4,
        served_version=2,
        last_exact_version=0,
        migration_depth=2,
        risk_score=0.05,
        state_kind="migrated",
    )
    decision = policy.decide(state, 3, 0.3)
    assert decision.action == "exact"
    assert decision.reason == "risk_threshold"
    assert decision.predicted_risk == pytest.approx(0.175)
    assert policy.advance(state, decision) == CacheLifecycleState.exact(4, 3)


def test_maximum_depth_skips_candidate() -> None:
    policy = LifecyclePolicy(
        max_migration_depth=2,
        risk_threshold=10.0,
        calibration=calibration(),
    )
    state = CacheLifecycleState(
        record_id=3,
        served_version=5,
        last_exact_version=3,
        migration_depth=2,
        risk_score=0.02,
        state_kind="migrated",
    )
    assert not policy.requires_candidate(state)
    decision = policy.decide(state, 6)
    assert decision.action == "exact"
    assert decision.reason == "max_migration_depth"
    assert not decision.candidate_evaluated


def test_nonadjacent_target_is_rejected() -> None:
    policy = LifecyclePolicy(
        max_migration_depth=2,
        risk_threshold=1.0,
        calibration=calibration(),
    )
    with pytest.raises(ValueError, match="adjacent"):
        policy.decide(CacheLifecycleState.exact(1, 0), 2, 0.1)


def test_migrated_lineage_must_match_depth() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        CacheLifecycleState(
            record_id=1,
            served_version=4,
            last_exact_version=1,
            migration_depth=2,
            risk_score=0.1,
            state_kind="migrated",
        )


def test_monotone_calibration_fit_and_roundtrip() -> None:
    fitted = fit_monotone_risk_calibration(
        correction_magnitudes=[0.4, 0.1, 0.3, 0.2],
        one_hop_errors=[0.2, 0.02, 0.04, 0.08],
        propagation_ratios=[0.8, 1.2, 1.0],
        bins=2,
        quantile=0.9,
    )
    assert fitted.correction_upper_bounds == (0.2, 0.4)
    assert fitted.one_hop_risks[0] <= fitted.one_hop_risks[1]
    assert fitted.propagation_gain == pytest.approx(1.16)
    assert MonotoneRiskCalibration.from_dict(fitted.to_dict()) == fitted


def test_policy_roundtrip() -> None:
    policy = LifecyclePolicy(
        max_migration_depth=3,
        risk_threshold=0.2,
        calibration=calibration(),
    )
    assert LifecyclePolicy.from_dict(policy.to_dict()) == policy


def linear_calibration() -> LinearSketchRiskCalibration:
    groups = 8
    return LinearSketchRiskCalibration(
        feature_name="absolute_log_norm_ratio",
        layer_quantile=0.5,
        intercept=-4.0,
        feature_mean=0.1,
        feature_scale=0.05,
        feature_coefficient=0.5,
        group_means=(0.5,) * groups,
        group_scales=(0.5,) * groups,
        group_coefficients=(0.0,) * groups,
        num_edges=2,
        maximum_depth=4,
        ridge=0.1,
        target="log_cache_error_q090",
    )


def test_sketch_policy_routes_and_roundtrips() -> None:
    policy = SketchLifecyclePolicy(
        max_migration_depth=3,
        risk_threshold=0.02,
        calibration=linear_calibration(),
    )
    state = CacheLifecycleState.exact(0, 0)
    low = policy.decide(state, 1, 0.0)
    high = policy.decide(state, 1, 0.3)
    assert low.action == "migrate"
    assert high.action == "exact"
    assert policy.advance(state, low).risk_score == low.predicted_risk
    assert SketchLifecyclePolicy.from_dict(policy.to_dict()) == policy


def balanced_policy() -> BalancedLifecyclePolicy:
    return BalancedLifecyclePolicy(
        max_migration_depth=2,
        exact_fractions=(0.25, 0.25, 0.25),
        edge_severities=(0.1, 0.2, 0.3),
        scheduler_seed=0,
    )


def test_balanced_policy_spreads_exact_refreshes() -> None:
    policy = balanced_policy()
    states = tuple(CacheLifecycleState.exact(record_id, 0) for record_id in range(8))
    first = policy.plan(states, 1)
    assert sum(value.action == "exact" for value in first) == 2
    after = tuple(
        policy.advance(state, decision)
        for state, decision in zip(states, first, strict=True)
    )
    second = policy.plan(after, 2)
    assert sum(value.action == "exact" for value in second) == 2
    first_exact = {
        value.record_id for value in first if value.action == "exact"
    }
    second_exact = {
        value.record_id for value in second if value.action == "exact"
    }
    assert first_exact.isdisjoint(second_exact)
    assert all(not value.candidate_evaluated for value in first)
    assert all(not value.candidate_evaluated for value in second)
    assert BalancedLifecyclePolicy.from_dict(policy.to_dict()) == policy


def test_balanced_policy_mandatory_depth_overrides_quota() -> None:
    policy = balanced_policy()
    states = tuple(
        CacheLifecycleState(
            record_id=record_id,
            served_version=2,
            last_exact_version=0,
            migration_depth=2,
            risk_score=0.1,
            state_kind="migrated",
        )
        for record_id in range(4)
    )
    decisions = policy.plan(states, 3)
    assert all(value.action == "exact" for value in decisions)
    assert all(value.reason == "max_migration_depth" for value in decisions)
