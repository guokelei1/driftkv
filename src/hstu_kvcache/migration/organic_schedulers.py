from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _valid_int(value: int, minimum: int = 0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= minimum
    )


def _valid_fraction(value: float) -> bool:
    return math.isfinite(value) and 0 < value < 1


def _pairs(
    values: tuple[tuple[int, int | float], ...],
    name: str,
) -> dict[int, int | float]:
    output: dict[int, int | float] = {}
    for record_id, value in values:
        if not _valid_int(record_id) or record_id in output:
            raise ValueError(f"{name} record ids are invalid")
        output[record_id] = value
    if tuple(sorted(output)) != tuple(record_id for record_id, _ in values):
        raise ValueError(f"{name} must be sorted by record id")
    return output


@dataclass(frozen=True)
class SchedulerRecord:
    record_id: int
    prefix_tokens: int
    migration_age: int
    natural_exact: bool = False

    def __post_init__(self) -> None:
        if (
            not _valid_int(self.record_id)
            or not _valid_int(self.prefix_tokens)
            or (not self.natural_exact and self.prefix_tokens < 1)
            or not _valid_int(self.migration_age)
            or not isinstance(self.natural_exact, bool)
        ):
            raise ValueError("organic scheduler record is invalid")


@dataclass(frozen=True)
class SchedulerSelection:
    scheduled_exact_ids: tuple[int, ...]
    natural_exact_ids: tuple[int, ...]
    migrate_ids: tuple[int, ...]
    next_state: object
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        groups = (
            self.scheduled_exact_ids,
            self.natural_exact_ids,
            self.migrate_ids,
        )
        if any(
            values != tuple(sorted(values))
            or len(values) != len(set(values))
            or any(not _valid_int(value) for value in values)
            for values in groups
        ):
            raise ValueError("organic scheduler selection ids are invalid")
        sets = tuple(set(values) for values in groups)
        if any(
            sets[left] & sets[right]
            for left in range(len(sets))
            for right in range(left + 1, len(sets))
        ):
            raise ValueError("organic scheduler actions overlap")
        if not isinstance(self.diagnostics, dict):
            raise ValueError("organic scheduler diagnostics are invalid")

    @property
    def exact_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(self.scheduled_exact_ids + self.natural_exact_ids)
        )


@dataclass(frozen=True)
class WorkBalancedRenewalState:
    horizon: int
    due_versions: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not _valid_int(self.horizon, 1):
            raise ValueError("renewal horizon is invalid")
        values = _pairs(self.due_versions, "renewal due versions")
        if any(not _valid_int(value) for value in values.values()):
            raise ValueError("renewal due version is invalid")


@dataclass(frozen=True)
class TotalTokenDebtState:
    budget_fraction: float
    debts: tuple[tuple[int, float], ...] = ()
    balance_tokens: float = 0.0
    cumulative_resident_tokens: int = 0
    cumulative_exact_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            not _valid_fraction(self.budget_fraction)
            or not math.isfinite(self.balance_tokens)
            or not _valid_int(self.cumulative_resident_tokens)
            or not _valid_int(self.cumulative_exact_tokens)
        ):
            raise ValueError("total token debt state is invalid")
        values = _pairs(self.debts, "token debts")
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("token debt is nonfinite")


@dataclass(frozen=True)
class AoIMaxWeightState:
    budget_fraction: float
    balance_tokens: float = 0.0
    cumulative_reusable_tokens: int = 0
    cumulative_scheduled_exact_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            not _valid_fraction(self.budget_fraction)
            or not math.isfinite(self.balance_tokens)
            or not _valid_int(self.cumulative_reusable_tokens)
            or not _valid_int(self.cumulative_scheduled_exact_tokens)
        ):
            raise ValueError("AoI MaxWeight state is invalid")


@dataclass(frozen=True)
class ModelTimeRenewalState:
    horizon: int
    model_time: float = 0.0
    severity_sum: float = 0.0
    severity_count: int = 0
    due_model_times: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if (
            not _valid_int(self.horizon, 1)
            or not math.isfinite(self.model_time)
            or self.model_time < 0
            or not math.isfinite(self.severity_sum)
            or self.severity_sum < 0
            or not _valid_int(self.severity_count)
        ):
            raise ValueError("model-time renewal state is invalid")
        values = _pairs(
            self.due_model_times,
            "model-time renewal due values",
        )
        if any(
            not math.isfinite(float(value)) or float(value) < 0
            for value in values.values()
        ):
            raise ValueError("model-time renewal due value is invalid")


def _records(
    records: Sequence[SchedulerRecord],
) -> tuple[SchedulerRecord, ...]:
    prepared = tuple(sorted(records, key=lambda value: value.record_id))
    if (
        any(not isinstance(value, SchedulerRecord) for value in prepared)
        or len({value.record_id for value in prepared}) != len(prepared)
    ):
        raise ValueError("organic scheduler records are invalid")
    return prepared


def _selection(
    records: tuple[SchedulerRecord, ...],
    scheduled: set[int],
    next_state: object,
    diagnostics: dict[str, object],
) -> SchedulerSelection:
    natural = {
        value.record_id for value in records if value.natural_exact
    }
    reusable = {
        value.record_id for value in records if not value.natural_exact
    }
    if not scheduled <= reusable:
        raise ValueError("scheduled exact records are not reusable")
    return SchedulerSelection(
        scheduled_exact_ids=tuple(sorted(scheduled)),
        natural_exact_ids=tuple(sorted(natural)),
        migrate_ids=tuple(sorted(reusable - scheduled)),
        next_state=next_state,
        diagnostics=diagnostics,
    )


def _integer_phase_assignment(
    records: tuple[SchedulerRecord, ...],
    horizon: int,
    target_version: int,
    existing: dict[int, int],
) -> dict[int, int]:
    loads = [0] * horizon
    by_id = {value.record_id: value for value in records}
    for record_id, due in existing.items():
        record = by_id.get(record_id)
        if record is not None:
            loads[(due - target_version) % horizon] += record.prefix_tokens
    output = dict(existing)
    missing = tuple(
        value
        for value in records
        if not value.natural_exact and value.record_id not in output
    )
    for record in sorted(
        missing,
        key=lambda value: (-value.prefix_tokens, value.record_id),
    ):
        phase = min(range(horizon), key=lambda value: (loads[value], value))
        output[record.record_id] = target_version + phase
        loads[phase] += record.prefix_tokens
    return output


def select_work_balanced_staggered_renewal(
    records: Sequence[SchedulerRecord],
    target_version: int,
    horizon: int,
    state: WorkBalancedRenewalState | None = None,
) -> SchedulerSelection:
    prepared = _records(records)
    if not _valid_int(target_version, 1) or not _valid_int(horizon, 1):
        raise ValueError("work-balanced renewal inputs are invalid")
    current = state or WorkBalancedRenewalState(horizon=horizon)
    if current.horizon != horizon:
        raise ValueError("work-balanced renewal horizon changed")
    current_due = {
        record_id: int(value)
        for record_id, value in _pairs(
            current.due_versions,
            "renewal due versions",
        ).items()
    }
    active_ids = {value.record_id for value in prepared}
    current_due = {
        record_id: value
        for record_id, value in current_due.items()
        if record_id in active_ids
    }
    due = _integer_phase_assignment(
        prepared,
        horizon,
        target_version,
        current_due,
    )
    scheduled: set[int] = set()
    for record in prepared:
        if record.natural_exact:
            due[record.record_id] = target_version + horizon
        elif due[record.record_id] <= target_version:
            scheduled.add(record.record_id)
            while due[record.record_id] <= target_version:
                due[record.record_id] += horizon
    next_state = WorkBalancedRenewalState(
        horizon=horizon,
        due_versions=tuple(sorted(due.items())),
    )
    scheduled_tokens = sum(
        value.prefix_tokens
        for value in prepared
        if value.record_id in scheduled
    )
    natural_tokens = sum(
        value.prefix_tokens for value in prepared if value.natural_exact
    )
    return _selection(
        prepared,
        scheduled,
        next_state,
        {
            "family": "work_balanced_staggered_renewal",
            "horizon": horizon,
            "target_version": target_version,
            "resident_records": len(prepared),
            "resident_tokens": sum(
                value.prefix_tokens for value in prepared
            ),
            "scheduled_exact_records": len(scheduled),
            "scheduled_exact_tokens": scheduled_tokens,
            "natural_exact_records": sum(
                value.natural_exact for value in prepared
            ),
            "natural_exact_tokens": natural_tokens,
            "labels_used": False,
        },
    )


def select_total_token_cumulative_debt(
    records: Sequence[SchedulerRecord],
    budget_fraction: float,
    state: TotalTokenDebtState | None = None,
) -> SchedulerSelection:
    prepared = _records(records)
    if not _valid_fraction(budget_fraction):
        raise ValueError("total token debt budget is invalid")
    current = state or TotalTokenDebtState(
        budget_fraction=budget_fraction
    )
    if current.budget_fraction != budget_fraction:
        raise ValueError("total token debt budget changed")
    old_debts = {
        record_id: float(value)
        for record_id, value in _pairs(
            current.debts,
            "token debts",
        ).items()
    }
    resident_tokens = sum(value.prefix_tokens for value in prepared)
    natural_tokens = sum(
        value.prefix_tokens for value in prepared if value.natural_exact
    )
    balance_before_scheduled = (
        current.balance_tokens
        + budget_fraction * resident_tokens
        - natural_tokens
    )
    debts: dict[int, float] = {}
    reusable = []
    for record in prepared:
        if record.natural_exact:
            debts[record.record_id] = 0.0
        else:
            debt = (
                old_debts.get(record.record_id, 0.0)
                + budget_fraction * record.prefix_tokens
            )
            debts[record.record_id] = debt
            reusable.append(record)
    ordered = sorted(
        reusable,
        key=lambda value: (
            -(debts[value.record_id] / value.prefix_tokens),
            -(value.migration_age + 1),
            value.record_id,
        ),
    )
    scheduled: set[int] = set()
    balance = balance_before_scheduled
    for record in ordered:
        if balance <= 1e-12:
            break
        scheduled.add(record.record_id)
        balance -= record.prefix_tokens
        debts[record.record_id] -= record.prefix_tokens
    scheduled_tokens = sum(
        value.prefix_tokens
        for value in reusable
        if value.record_id in scheduled
    )
    cumulative_resident = (
        current.cumulative_resident_tokens + resident_tokens
    )
    cumulative_exact = (
        current.cumulative_exact_tokens
        + natural_tokens
        + scheduled_tokens
    )
    next_state = TotalTokenDebtState(
        budget_fraction=budget_fraction,
        debts=tuple(sorted(debts.items())),
        balance_tokens=balance,
        cumulative_resident_tokens=cumulative_resident,
        cumulative_exact_tokens=cumulative_exact,
    )
    return _selection(
        prepared,
        scheduled,
        next_state,
        {
            "family": "total_token_cumulative_debt",
            "budget_fraction": budget_fraction,
            "resident_records": len(prepared),
            "resident_tokens": resident_tokens,
            "natural_exact_records": sum(
                value.natural_exact for value in prepared
            ),
            "natural_exact_tokens": natural_tokens,
            "scheduled_exact_records": len(scheduled),
            "scheduled_exact_tokens": scheduled_tokens,
            "balance_before_scheduled_tokens": balance_before_scheduled,
            "balance_after_scheduled_tokens": balance,
            "one_prefix_borrowing": True,
            "cumulative_resident_tokens": cumulative_resident,
            "cumulative_exact_tokens": cumulative_exact,
            "cumulative_exact_fraction": (
                cumulative_exact / cumulative_resident
                if cumulative_resident
                else 0.0
            ),
            "labels_used": False,
        },
    )


def select_aoi_maxweight(
    records: Sequence[SchedulerRecord],
    budget_fraction: float,
    state: AoIMaxWeightState | None = None,
) -> SchedulerSelection:
    prepared = _records(records)
    if not _valid_fraction(budget_fraction):
        raise ValueError("AoI MaxWeight budget is invalid")
    current = state or AoIMaxWeightState(
        budget_fraction=budget_fraction
    )
    if current.budget_fraction != budget_fraction:
        raise ValueError("AoI MaxWeight budget changed")
    reusable = tuple(
        value for value in prepared if not value.natural_exact
    )
    reusable_tokens = sum(value.prefix_tokens for value in reusable)
    balance_before_scheduled = (
        current.balance_tokens + budget_fraction * reusable_tokens
    )
    indices = {
        value.record_id: (
            (value.migration_age + 1) * (value.migration_age + 2)
            / (2 * value.prefix_tokens)
        )
        for value in reusable
    }
    ordered = sorted(
        reusable,
        key=lambda value: (
            -indices[value.record_id],
            -(value.migration_age + 1),
            value.record_id,
        ),
    )
    scheduled: set[int] = set()
    balance = balance_before_scheduled
    for record in ordered:
        if balance <= 1e-12:
            break
        scheduled.add(record.record_id)
        balance -= record.prefix_tokens
    scheduled_tokens = sum(
        value.prefix_tokens
        for value in reusable
        if value.record_id in scheduled
    )
    cumulative_reusable = (
        current.cumulative_reusable_tokens + reusable_tokens
    )
    cumulative_scheduled = (
        current.cumulative_scheduled_exact_tokens + scheduled_tokens
    )
    next_state = AoIMaxWeightState(
        budget_fraction=budget_fraction,
        balance_tokens=balance,
        cumulative_reusable_tokens=cumulative_reusable,
        cumulative_scheduled_exact_tokens=cumulative_scheduled,
    )
    natural_tokens = sum(
        value.prefix_tokens for value in prepared if value.natural_exact
    )
    return _selection(
        prepared,
        scheduled,
        next_state,
        {
            "family": "aoi_maxweight",
            "budget_fraction": budget_fraction,
            "resident_records": len(prepared),
            "reusable_records": len(reusable),
            "reusable_tokens": reusable_tokens,
            "natural_exact_records": sum(
                value.natural_exact for value in prepared
            ),
            "natural_exact_tokens": natural_tokens,
            "scheduled_exact_records": len(scheduled),
            "scheduled_exact_tokens": scheduled_tokens,
            "balance_before_scheduled_tokens": balance_before_scheduled,
            "balance_after_scheduled_tokens": balance,
            "one_prefix_borrowing": True,
            "cumulative_reusable_tokens": cumulative_reusable,
            "cumulative_scheduled_exact_tokens": cumulative_scheduled,
            "indices": tuple(sorted(indices.items())),
            "labels_used": False,
        },
    )


def _model_time_phase_assignment(
    records: tuple[SchedulerRecord, ...],
    horizon: int,
    model_time: float,
    existing: dict[int, float],
) -> dict[int, float]:
    loads = [0] * horizon
    by_id = {value.record_id: value for value in records}
    for record_id, due in existing.items():
        record = by_id.get(record_id)
        if record is not None:
            phase = max(0, math.ceil(due - model_time) - 1) % horizon
            loads[phase] += record.prefix_tokens
    output = dict(existing)
    missing = tuple(
        value
        for value in records
        if not value.natural_exact and value.record_id not in output
    )
    for record in sorted(
        missing,
        key=lambda value: (-value.prefix_tokens, value.record_id),
    ):
        phase = min(range(horizon), key=lambda value: (loads[value], value))
        output[record.record_id] = model_time + phase + 1
        loads[phase] += record.prefix_tokens
    return output


def select_model_time_staggered_renewal(
    records: Sequence[SchedulerRecord],
    edge_severity: float,
    horizon: int,
    state: ModelTimeRenewalState | None = None,
) -> SchedulerSelection:
    prepared = _records(records)
    if (
        not math.isfinite(edge_severity)
        or edge_severity < 0
        or not _valid_int(horizon, 1)
    ):
        raise ValueError("model-time renewal inputs are invalid")
    current = state or ModelTimeRenewalState(horizon=horizon)
    if current.horizon != horizon:
        raise ValueError("model-time renewal horizon changed")
    severity_sum = current.severity_sum + edge_severity
    severity_count = current.severity_count + 1
    mean_severity = severity_sum / severity_count
    delta_model_time = (
        edge_severity / mean_severity if mean_severity > 0 else 0.0
    )
    next_model_time = current.model_time + delta_model_time
    current_due = {
        record_id: float(value)
        for record_id, value in _pairs(
            current.due_model_times,
            "model-time renewal due values",
        ).items()
    }
    active_ids = {value.record_id for value in prepared}
    current_due = {
        record_id: value
        for record_id, value in current_due.items()
        if record_id in active_ids
    }
    due = _model_time_phase_assignment(
        prepared,
        horizon,
        current.model_time,
        current_due,
    )
    scheduled: set[int] = set()
    for record in prepared:
        if record.natural_exact:
            due[record.record_id] = next_model_time + horizon
        elif due[record.record_id] <= next_model_time + 1e-12:
            scheduled.add(record.record_id)
            while due[record.record_id] <= next_model_time + 1e-12:
                due[record.record_id] += horizon
    next_state = ModelTimeRenewalState(
        horizon=horizon,
        model_time=next_model_time,
        severity_sum=severity_sum,
        severity_count=severity_count,
        due_model_times=tuple(sorted(due.items())),
    )
    scheduled_tokens = sum(
        value.prefix_tokens
        for value in prepared
        if value.record_id in scheduled
    )
    natural_tokens = sum(
        value.prefix_tokens for value in prepared if value.natural_exact
    )
    return _selection(
        prepared,
        scheduled,
        next_state,
        {
            "family": "model_time_staggered_renewal",
            "horizon": horizon,
            "edge_severity": edge_severity,
            "running_mean_severity": mean_severity,
            "delta_model_time": delta_model_time,
            "model_time_before": current.model_time,
            "model_time_after": next_model_time,
            "resident_records": len(prepared),
            "resident_tokens": sum(
                value.prefix_tokens for value in prepared
            ),
            "scheduled_exact_records": len(scheduled),
            "scheduled_exact_tokens": scheduled_tokens,
            "natural_exact_records": sum(
                value.natural_exact for value in prepared
            ),
            "natural_exact_tokens": natural_tokens,
            "labels_used": False,
        },
    )
