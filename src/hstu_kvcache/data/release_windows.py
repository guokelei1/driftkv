"""Pure-time planning primitives for prospective streaming version chains."""

from __future__ import annotations

from dataclasses import asdict, dataclass


DAY_SECONDS = 86_400


@dataclass(frozen=True)
class TimeWindow:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid half-open time window")


@dataclass(frozen=True)
class ReleaseWindowRecipe:
    name: str
    train_days: int
    admission_days: int
    cadence_days: int
    evaluation_days: int
    role: str
    fixed_endpoint: bool = False

    def __post_init__(self) -> None:
        for field_name in ("train_days", "admission_days", "cadence_days", "evaluation_days"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.train_days < 1 or self.cadence_days < 1:
            raise ValueError("train_days and cadence_days must be positive")
        if self.admission_days == 0 and not self.fixed_endpoint:
            raise ValueError("a recipe without admission must use a fixed endpoint")
        if self.cadence_days < self.train_days + self.admission_days:
            raise ValueError("overlapping update data is not allowed in the scale planner")


@dataclass(frozen=True)
class ReleaseSlot:
    update_index: int
    parent_version: str
    child_version: str
    train: TimeWindow
    admission: TimeWindow | None
    cutover: int
    evaluation: TimeWindow
    blind_locked: bool

    def as_dict(self) -> dict:
        return asdict(self)


def complete_day_end(dataset_end: int) -> int:
    if dataset_end < 0:
        raise ValueError("dataset_end must be non-negative")
    return dataset_end // DAY_SECONDS * DAY_SECONDS


def daily_slices(start: int, end: int) -> list[TimeWindow]:
    """Return complete, aligned daily slices in ``[start, end)``."""

    if start % DAY_SECONDS or end % DAY_SECONDS:
        raise ValueError("daily slice endpoints must be day aligned")
    if end < start:
        raise ValueError("daily slice end precedes start")
    return [TimeWindow(value, value + DAY_SECONDS) for value in range(start, end, DAY_SECONDS)]


def max_equal_train_days(
    *,
    base_cutoff: int,
    usable_end: int,
    updates: int,
    admission_days: int,
    evaluation_days: int,
) -> int:
    """Max equal non-overlapping train days for a fixed number of updates."""

    if updates < 1 or admission_days < 0 or evaluation_days < 0:
        raise ValueError("invalid equal-allocation arguments")
    if base_cutoff % DAY_SECONDS or usable_end % DAY_SECONDS:
        raise ValueError("equal-allocation endpoints must be day aligned")
    available_days = (usable_end - base_cutoff) // DAY_SECONDS - evaluation_days
    return max(0, available_days // updates - admission_days)


def plan_release_slots(
    *,
    base_cutoff: int,
    usable_end: int,
    recipe: ReleaseWindowRecipe,
    blind_lock_from_update: int = 3,
    max_updates: int | None = None,
) -> list[ReleaseSlot]:
    """Plan fully evaluable releases without reading events or labels."""

    if base_cutoff < 0 or usable_end <= base_cutoff:
        raise ValueError("usable_end must follow base_cutoff")
    if base_cutoff % DAY_SECONDS or usable_end % DAY_SECONDS:
        raise ValueError("scale release planning uses aligned complete days")
    if blind_lock_from_update < 1:
        raise ValueError("blind lock index must be positive")
    slots: list[ReleaseSlot] = []
    update = 1
    cycle_start = base_cutoff
    while max_updates is None or update <= max_updates:
        train = TimeWindow(cycle_start, cycle_start + recipe.train_days * DAY_SECONDS)
        admission = None
        if recipe.admission_days:
            admission = TimeWindow(train.end, train.end + recipe.admission_days * DAY_SECONDS)
        cutover = cycle_start + (recipe.train_days + recipe.admission_days) * DAY_SECONDS
        evaluation = TimeWindow(cutover, cutover + recipe.evaluation_days * DAY_SECONDS)
        if evaluation.end > usable_end:
            break
        slots.append(
            ReleaseSlot(
                update_index=update,
                parent_version=f"theta{update - 1}",
                child_version=f"theta{update}",
                train=train,
                admission=admission,
                cutover=cutover,
                evaluation=evaluation,
                blind_locked=update >= blind_lock_from_update,
            )
        )
        update += 1
        cycle_start += recipe.cadence_days * DAY_SECONDS
    return slots
