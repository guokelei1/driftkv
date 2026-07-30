import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_evokv_design3_m1_qk_edge.py"
)
SPEC = importlib.util.spec_from_file_location(
    "train_evokv_design3_m1_qk_edge",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_qk_model_shape() -> None:
    cfg = MODULE.qk_model_config(312144, 5, 50000)
    assert cfg.hidden_size == 512
    assert cfg.num_layers == 16
    assert cfg.num_heads == 8
    assert cfg.head_dim == 64
    assert cfg.max_seq_len == 512
    assert cfg.num_prediction_items == 50000


def test_qk_boundary_metadata_stays_flexible() -> None:
    metadata = {
        "protocol": "ordered_exposure_prepared_v1",
        "dataset": "tenrec-qk",
        "selected_users": 4096,
        "fitted_items": 250000,
        "base_filtered_events": 512,
        "window_filtered_events": 32,
        "window_count": 1,
        "extra_future_field": "allowed",
    }
    MODULE.validate_prepared_metadata(metadata)
    boundary = MODULE.boundary_metadata(metadata)
    assert boundary["base_filtered_events"] == 512
    assert boundary["window_filtered_events"] == 32
    assert "extra_future_field" not in boundary


def test_tiny_cpu_two_version_smoke() -> None:
    result = MODULE.run_tiny_cpu_smoke()
    assert result["status"] == "complete"
    assert result["base_coverage"]["eligible_targets"] > 0
    assert result["update_coverage"]["eligible_targets"] > 0
    assert result["theta0_to_theta1_dtheta_rel"] > 0
