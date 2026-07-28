import math

import pytest

from hstu_kvcache.migration.organic_schedulers import (
    AoIMaxWeightState,
    ModelTimeRenewalState,
    SchedulerRecord,
    TotalTokenDebtState,
    WorkBalancedRenewalState,
    select_aoi_maxweight,
    select_model_time_staggered_renewal,
    select_total_token_cumulative_debt,
    select_work_balanced_staggered_renewal,
)


def records(
    count: int,
    tokens: int = 10,
    age: int = 0,
) -> tuple[SchedulerRecord, ...]:
    return tuple(
        SchedulerRecord(
            record_id=index,
            prefix_tokens=tokens,
            migration_age=age,
        )
        for index in range(count)
    )


def test_scheduler_record_and_states_reject_invalid_values() -> None:
    assert SchedulerRecord(0, 0, 0, natural_exact=True).prefix_tokens == 0
    with pytest.raises(ValueError, match="record"):
        SchedulerRecord(True, 1, 0)
    with pytest.raises(ValueError, match="record"):
        SchedulerRecord(0, 0, 0)
    with pytest.raises(ValueError, match="horizon"):
        WorkBalancedRenewalState(0)
    with pytest.raises(ValueError, match="sorted"):
        WorkBalancedRenewalState(4, ((2, 3), (1, 4)))
    with pytest.raises(ValueError, match="debt"):
        TotalTokenDebtState(0.1, ((0, math.inf),))
    with pytest.raises(ValueError, match="AoI"):
        AoIMaxWeightState(1.0)
    with pytest.raises(ValueError, match="model-time"):
        ModelTimeRenewalState(4, model_time=-1)


def test_empty_scheduler_populations_are_supported() -> None:
    renewal = select_work_balanced_staggered_renewal((), 1, 8)
    debt = select_total_token_cumulative_debt((), 0.1)
    aoi = select_aoi_maxweight((), 0.1)
    model_time = select_model_time_staggered_renewal((), 0.0, 8)

    for result in (renewal, debt, aoi, model_time):
        assert result.exact_ids == ()
        assert result.migrate_ids == ()
        assert result.diagnostics["labels_used"] is False
    assert debt.next_state.balance_tokens == 0
    assert aoi.next_state.balance_tokens == 0
    assert model_time.next_state.model_time == 0


def test_work_balanced_renewal_is_staggered_and_bounded() -> None:
    current = records(16)
    state = None
    seen: dict[int, list[int]] = {index: [] for index in range(16)}
    per_edge = []
    for target_version in range(1, 13):
        result = select_work_balanced_staggered_renewal(
            tuple(reversed(current)),
            target_version,
            4,
            state,
        )
        state = result.next_state
        per_edge.append(len(result.scheduled_exact_ids))
        for record_id in result.scheduled_exact_ids:
            seen[record_id].append(target_version)
        selected = set(result.scheduled_exact_ids)
        current = tuple(
            SchedulerRecord(
                record_id=value.record_id,
                prefix_tokens=value.prefix_tokens,
                migration_age=(
                    0
                    if value.record_id in selected
                    else value.migration_age + 1
                ),
            )
            for value in current
        )

    assert per_edge == [4] * 12
    assert all(
        right - left == 4
        for versions in seen.values()
        for left, right in zip(versions, versions[1:], strict=False)
    )
    assert all(len(versions) == 3 for versions in seen.values())


def test_work_balanced_renewal_balances_tokens_and_resets_natural() -> None:
    current = tuple(
        SchedulerRecord(index, tokens, 0)
        for index, tokens in enumerate((40, 30, 20, 10))
    )
    first = select_work_balanced_staggered_renewal(
        current,
        target_version=1,
        horizon=2,
    )
    due = dict(first.next_state.due_versions)
    phase_tokens = [0, 0]
    for value in current:
        phase_tokens[(due[value.record_id] - 1) % 2] += value.prefix_tokens
    assert phase_tokens == [50, 50]

    natural_records = (
        SchedulerRecord(0, 40, 0, natural_exact=True),
        *current[1:],
    )
    second = select_work_balanced_staggered_renewal(
        natural_records,
        target_version=2,
        horizon=2,
        state=first.next_state,
    )
    assert 0 in second.natural_exact_ids
    assert 0 not in second.scheduled_exact_ids
    assert dict(second.next_state.due_versions)[0] == 4


def test_total_token_debt_charges_natural_exact_before_scheduled() -> None:
    current = tuple(
        SchedulerRecord(
            record_id=index,
            prefix_tokens=10,
            migration_age=0,
            natural_exact=index == 9,
        )
        for index in range(10)
    )
    first = select_total_token_cumulative_debt(
        tuple(reversed(current)),
        budget_fraction=0.2,
    )

    assert first.natural_exact_ids == (9,)
    assert first.scheduled_exact_ids == (0,)
    assert first.next_state.balance_tokens == 0
    assert first.next_state.cumulative_resident_tokens == 100
    assert first.next_state.cumulative_exact_tokens == 20
    assert first.diagnostics["cumulative_exact_fraction"] == 0.2


def test_total_token_debt_repays_unavoidable_natural_overage() -> None:
    natural = tuple(
        SchedulerRecord(
            record_id=index,
            prefix_tokens=10,
            migration_age=0,
            natural_exact=index < 3,
        )
        for index in range(10)
    )
    first = select_total_token_cumulative_debt(natural, 0.2)
    assert first.scheduled_exact_ids == ()
    assert first.next_state.balance_tokens == -10

    reusable = records(10, tokens=10, age=1)
    second = select_total_token_cumulative_debt(
        reusable,
        0.2,
        first.next_state,
    )
    assert second.scheduled_exact_ids == (3,)
    assert second.next_state.balance_tokens == 0
    assert second.next_state.cumulative_exact_tokens == 40
    assert second.next_state.cumulative_resident_tokens == 200


def test_total_token_debt_rotates_service_without_random_ties() -> None:
    current = records(8, tokens=10)
    state = None
    selected = []
    for _ in range(4):
        result = select_total_token_cumulative_debt(
            tuple(reversed(current)),
            0.25,
            state,
        )
        state = result.next_state
        selected.extend(result.scheduled_exact_ids)
        exact = set(result.scheduled_exact_ids)
        current = tuple(
            SchedulerRecord(
                value.record_id,
                value.prefix_tokens,
                0 if value.record_id in exact else value.migration_age + 1,
            )
            for value in current
        )

    assert selected == list(range(8))
    assert state.cumulative_exact_tokens == 80
    assert state.cumulative_resident_tokens == 320


def test_aoi_maxweight_uses_age_cost_index_and_reusable_budget() -> None:
    current = (
        SchedulerRecord(0, 10, 5),
        SchedulerRecord(1, 1, 1),
        SchedulerRecord(2, 100, 0, natural_exact=True),
    )
    result = select_aoi_maxweight(current, budget_fraction=0.5)

    assert result.natural_exact_ids == (2,)
    assert result.scheduled_exact_ids == (0, 1)
    assert result.next_state.balance_tokens == -5.5
    assert result.diagnostics["reusable_tokens"] == 11
    assert result.diagnostics["natural_exact_tokens"] == 100
    assert result.diagnostics["one_prefix_borrowing"]
    indices = dict(result.diagnostics["indices"])
    assert indices[1] > indices[0]


def test_aoi_maxweight_is_order_invariant_and_carries_budget() -> None:
    current = (
        SchedulerRecord(0, 8, 0),
        SchedulerRecord(1, 8, 3),
        SchedulerRecord(2, 8, 1),
    )
    first = select_aoi_maxweight(current, 0.2)
    reversed_first = select_aoi_maxweight(
        tuple(reversed(current)),
        0.2,
    )
    assert first.scheduled_exact_ids == reversed_first.scheduled_exact_ids
    assert first.scheduled_exact_ids == (1,)
    assert first.next_state.balance_tokens == pytest.approx(-3.2)

    exact = set(first.scheduled_exact_ids)
    advanced = tuple(
        SchedulerRecord(
            value.record_id,
            value.prefix_tokens,
            0 if value.record_id in exact else value.migration_age + 1,
        )
        for value in current
    )
    second = select_aoi_maxweight(
        advanced,
        0.2,
        first.next_state,
    )
    assert second.scheduled_exact_ids == (2,)
    assert second.next_state.balance_tokens == pytest.approx(-6.4)

    third = select_aoi_maxweight(
        advanced,
        0.2,
        second.next_state,
    )
    assert third.scheduled_exact_ids == ()
    assert third.next_state.balance_tokens == pytest.approx(-1.6)

    fourth = select_aoi_maxweight(
        advanced,
        0.2,
        third.next_state,
    )
    assert fourth.scheduled_exact_ids
    assert fourth.next_state.balance_tokens == pytest.approx(-4.8)


@pytest.mark.parametrize("family", ("debt", "aoi"))
def test_one_prefix_borrowing_serves_positive_head_and_repays(
    family: str,
) -> None:
    current = (
        SchedulerRecord(0, 20, 20),
        SchedulerRecord(1, 1, 0),
    )
    if family == "debt":
        state = TotalTokenDebtState(
            budget_fraction=0.1,
            debts=((0, 20.0), (1, 0.0)),
            balance_tokens=-2.0,
        )
        run = lambda value: select_total_token_cumulative_debt(
            current,
            0.1,
            value,
        )
    else:
        state = AoIMaxWeightState(
            budget_fraction=0.1,
            balance_tokens=-2.0,
        )
        run = lambda value: select_aoi_maxweight(
            current,
            0.1,
            value,
        )

    first = run(state)
    assert first.diagnostics["balance_before_scheduled_tokens"] == (
        pytest.approx(0.1)
    )
    assert first.scheduled_exact_ids == (0,)
    assert first.next_state.balance_tokens == pytest.approx(-19.9)
    assert first.diagnostics["one_prefix_borrowing"]

    state = first.next_state
    for _ in range(9):
        result = run(state)
        assert result.scheduled_exact_ids == ()
        state = result.next_state
    repaid = run(state)
    assert repaid.diagnostics["balance_before_scheduled_tokens"] > 0
    assert repaid.scheduled_exact_ids == (0,)


@pytest.mark.parametrize("family", ("debt", "aoi"))
def test_one_prefix_borrowing_avoids_heterogeneous_head_idle(
    family: str,
) -> None:
    current = (
        SchedulerRecord(0, 3, 10),
        SchedulerRecord(1, 4, 8),
        SchedulerRecord(2, 10, 10),
    )
    if family == "debt":
        result = select_total_token_cumulative_debt(
            current,
            0.5,
            TotalTokenDebtState(
                budget_fraction=0.5,
                debts=((0, 9.0), (1, 8.0), (2, 10.0)),
            ),
        )
    else:
        result = select_aoi_maxweight(current, 0.5)

    assert result.diagnostics["balance_before_scheduled_tokens"] == (
        pytest.approx(8.5)
    )
    assert result.scheduled_exact_ids == (0, 1, 2)
    assert result.next_state.balance_tokens == pytest.approx(-8.5)
    selected_tokens = {
        value.record_id: value.prefix_tokens for value in current
    }
    maximum_selected_prefix = max(
        selected_tokens[record_id]
        for record_id in result.scheduled_exact_ids
    )
    assert -result.next_state.balance_tokens < maximum_selected_prefix


def test_model_time_renewal_scales_with_current_edge_severity() -> None:
    current = records(8)
    first = select_model_time_staggered_renewal(
        tuple(reversed(current)),
        edge_severity=2.0,
        horizon=4,
    )
    assert first.scheduled_exact_ids == (0, 4)
    assert first.next_state.model_time == 1.0

    second = select_model_time_staggered_renewal(
        current,
        edge_severity=0.0,
        horizon=4,
        state=first.next_state,
    )
    assert second.scheduled_exact_ids == ()
    assert second.next_state.model_time == 1.0

    third = select_model_time_staggered_renewal(
        current,
        edge_severity=4.0,
        horizon=4,
        state=second.next_state,
    )
    assert third.next_state.model_time == pytest.approx(3.0)
    assert len(third.scheduled_exact_ids) == 4
    assert third.diagnostics["delta_model_time"] == pytest.approx(2.0)


def test_model_time_natural_exact_reanchors_due_clock() -> None:
    current = records(4)
    first = select_model_time_staggered_renewal(current, 1.0, 4)
    next_records = (
        SchedulerRecord(0, 10, 0, natural_exact=True),
        *current[1:],
    )
    second = select_model_time_staggered_renewal(
        next_records,
        1.0,
        4,
        first.next_state,
    )

    assert second.natural_exact_ids == (0,)
    assert 0 not in second.scheduled_exact_ids
    assert dict(second.next_state.due_model_times)[0] == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("function", "args"),
    (
        (
            select_work_balanced_staggered_renewal,
            {"target_version": 1, "horizon": 4},
        ),
        (
            select_total_token_cumulative_debt,
            {"budget_fraction": 0.1},
        ),
        (
            select_aoi_maxweight,
            {"budget_fraction": 0.1},
        ),
        (
            select_model_time_staggered_renewal,
            {"edge_severity": 1.0, "horizon": 4},
        ),
    ),
)
def test_schedulers_reject_duplicate_record_ids(function, args) -> None:
    duplicate = (
        SchedulerRecord(0, 10, 0),
        SchedulerRecord(0, 20, 1),
    )
    with pytest.raises(ValueError, match="records"):
        function(duplicate, **args)


def test_policy_state_rejects_parameter_changes() -> None:
    with pytest.raises(ValueError, match="horizon changed"):
        select_work_balanced_staggered_renewal(
            records(4),
            1,
            8,
            WorkBalancedRenewalState(10),
        )
    with pytest.raises(ValueError, match="budget changed"):
        select_total_token_cumulative_debt(
            records(4),
            0.1,
            TotalTokenDebtState(0.2),
        )
    with pytest.raises(ValueError, match="budget changed"):
        select_aoi_maxweight(
            records(4),
            0.1,
            AoIMaxWeightState(0.2),
        )
    with pytest.raises(ValueError, match="horizon changed"):
        select_model_time_staggered_renewal(
            records(4),
            1.0,
            8,
            ModelTimeRenewalState(10),
        )
