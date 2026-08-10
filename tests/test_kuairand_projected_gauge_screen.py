from hstu_kvcache.streaming.kuairand_projected_gauge_screen import (
    _select_angle,
    _user_partition,
    load_projected_gauge_screen_config,
)

CONFIG = "configs/evokv_root_cause/kuairand_projected_gauge_screen_20260808_v0.json"


def test_projected_gauge_screen_config_and_partition_are_stable():
    document = load_projected_gauge_screen_config(CONFIG)
    assert document["source"]["target_version"] == 2
    assert _user_partition(17, 91, 0.25) == _user_partition(17, 91, 0.25)


def test_projected_gauge_screen_selects_only_inside_band():
    config = load_projected_gauge_screen_config(CONFIG)
    values = [
        {
            "angle_radians": 0.02,
            "selection": {
                "inside_predeclared_band": True,
                "mean_relative_percent": 4.0,
            },
        },
        {
            "angle_radians": 0.08,
            "selection": {
                "inside_predeclared_band": True,
                "mean_relative_percent": 7.5,
            },
        },
        {
            "angle_radians": 0.12,
            "selection": {
                "inside_predeclared_band": False,
                "mean_relative_percent": 8.0,
            },
        },
    ]
    assert _select_angle(values, config) == 0.08
