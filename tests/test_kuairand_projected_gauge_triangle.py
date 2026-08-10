import pytest

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming.kuairand_projected_gauge_triangle import (
    _angle,
    _apply_transform,
    _matrix,
    load_projected_gauge_triangle_config,
)


def test_projected_gauge_angles_accumulate_after_anchor():
    assert _angle(0, 1, 0.14) == 0.0
    assert _angle(1, 1, 0.14) == 0.0
    assert _angle(8, 1, 0.14) == pytest.approx(0.98)


def test_projected_gauge_matrix_is_lower_triangular():
    cells = []
    for target in range(1, 9):
        for source in range(target):
            cells.append(
                {
                    "target_version": target,
                    "source_version": source,
                    "holdout": {
                        "comparisons": {
                            "recompute_over_reuse": {
                                "mrr": {"relative_percent": target - source}
                            }
                        }
                    },
                }
            )
    values = _matrix(cells, "holdout", "mrr")
    assert values[7][0] == 8.0
    assert values[7][7] == 1.0
    assert values[0][1] is None


def test_projected_value_scale_triangle_config_and_transform():
    config = load_projected_gauge_triangle_config(
        "configs/evokv_root_cause/kuairand_function_preserving_scale13_k005_v015_20260809_v0.json"
    )
    model = HSTU(
        HSTUConfig(
            num_items=20,
            num_behaviors=2,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
        )
    )
    certificate = _apply_transform(model, 3, config)
    assert certificate["value_log_scale"] == pytest.approx(0.3)


def test_projected_gauge_matrix_supports_ten_versions():
    cells = []
    for target in range(1, 11):
        for source in range(target):
            cells.append(
                {
                    "target_version": target,
                    "source_version": source,
                    "holdout": {
                        "comparisons": {
                            "recompute_over_reuse": {
                                "mrr": {"relative_percent": target - source}
                            }
                        }
                    },
                }
            )
    values = _matrix(cells, "holdout", "mrr")
    assert len(values) == 10
    assert values[9][9] == 1.0


def test_projected_gauge_incremental_extension_starts_at_theta11():
    config = load_projected_gauge_triangle_config(
        "configs/evokv_root_cause/kuairand_function_preserving_scale11_k005_v015_20260809_v0.json"
    )
    assert config["final_version"] == 11
    assert config["minimum_source_version"] == 3
    assert config["prior_triangle"]["path"].endswith("result.json")
