from pathlib import Path

import yaml

from hstu_kvcache.data.release_windows import (
    DAY_SECONDS,
    ReleaseWindowRecipe,
    daily_slices,
    max_equal_train_days,
    plan_release_slots,
)


ROOT = Path(__file__).resolve().parents[1]


def recipe(name: str, train: int, admission: int, cadence: int, evaluation: int, fixed: bool = False):
    return ReleaseWindowRecipe(
        name=name,
        train_days=train,
        admission_days=admission,
        cadence_days=cadence,
        evaluation_days=evaluation,
        role="test",
        fixed_endpoint=fixed,
    )


def test_daily_physical_slices_are_aligned_and_complete() -> None:
    slices = daily_slices(217 * DAY_SECONDS, 300 * DAY_SECONDS)
    assert len(slices) == 83
    assert slices[0].start == 217 * DAY_SECONDS
    assert slices[-1].end == 300 * DAY_SECONDS


def test_main_fixed_endpoint_recipe_supports_five_edges_but_locks_theta3() -> None:
    slots = plan_release_slots(
        base_cutoff=217 * DAY_SECONDS,
        usable_end=300 * DAY_SECONDS,
        recipe=recipe("main", 14, 0, 14, 7, fixed=True),
        blind_lock_from_update=3,
    )
    assert len(slots) == 5
    assert slots[0].cutover == 231 * DAY_SECONDS
    assert slots[1].cutover == 245 * DAY_SECONDS
    assert slots[2].child_version == "theta3"
    assert [slot.blind_locked for slot in slots] == [False, False, True, True, True]
    assert all(slot.admission is None for slot in slots)
    assert all(slot.train.end == slot.cutover for slot in slots)


def test_dense_recipe_capacity_and_seven_day_reserve() -> None:
    dense = recipe("dense", 2, 0, 2, 1, fixed=True)
    assert len(plan_release_slots(
        base_cutoff=217 * DAY_SECONDS,
        usable_end=300 * DAY_SECONDS,
        recipe=dense,
    )) == 41
    assert len(plan_release_slots(
        base_cutoff=217 * DAY_SECONDS,
        usable_end=293 * DAY_SECONDS,
        recipe=dense,
    )) == 37


def test_fewer_versions_can_use_larger_equal_windows() -> None:
    assert max_equal_train_days(
        base_cutoff=217 * DAY_SECONDS,
        usable_end=300 * DAY_SECONDS,
        updates=3,
        admission_days=0,
        evaluation_days=7,
    ) == 25
    assert max_equal_train_days(
        base_cutoff=217 * DAY_SECONDS,
        usable_end=293 * DAY_SECONDS,
        updates=3,
        admission_days=0,
        evaluation_days=7,
    ) == 23


def test_streaming_contract_has_no_training_or_theta3_access() -> None:
    value = yaml.safe_load(
        (ROOT / "configs/contracts/yambda500m_streaming_windows_v1.yaml").read_text()
    )
    assert value["scope"]["training_authorized"] is False
    assert value["scope"]["label_access"] is False
    assert value["blind_boundary"]["labels_requests_training_and_metrics"] == "prohibited"
    assert value["blind_boundary"]["this_is_not_the_theta3_qualification_contract"] is True
    assert all(recipe["admission_days"] == 0 for recipe in value["recipes"].values())
    assert all(recipe["fixed_endpoint"] is True for recipe in value["recipes"].values())
