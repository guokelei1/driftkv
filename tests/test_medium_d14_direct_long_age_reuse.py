import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_yambda500m_medium_d14_direct_long_age_reuse.py"
CONTRACT = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_direct_long_age_reuse_v1.yaml"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("medium_direct_long_age", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_uses_canonical_e14_output_and_exact_ranges():
    module = load_runner_module()
    runner = module.Runner(CONTRACT)
    assert runner.cell_dir(0, 5).parts[-2:] == ("E14", "v0_to_v5")
    assert runner.cell_record(0, 2)["day_range_half_open"] == [245, 259]
    assert runner.cell_record(3, 5)["day_range_half_open"] == [287, 301]
    assert runner.plan()["new_cell_count"] == 10
    assert runner.plan()["final_triangle_cell_count"] == 15


def test_comparison_row_uses_only_within_run_new_and_reuse():
    module = load_runner_module()
    row = module.Runner.comparison_row(
        producer=0,
        current=3,
        source="new_direct_long_age",
        requests=100,
        new={"ROC_AUC": 0.6, "log_loss": 0.3},
        reuse={"ROC_AUC": 0.55, "log_loss": 0.35},
        day_range=[259, 273],
        historical_new={"ROC_AUC": 0.59, "log_loss": 0.31},
    )
    assert abs(row["current_minus_reuse_ROC_AUC_pp"] - 5.0) < 1e-12
    assert abs(row["reuse_AUC_change_vs_current_percent"] + 100.0 / 12.0) < 1e-12
    assert abs(row["reuse_log_loss_change_vs_current_percent"] - 100.0 / 6.0) < 1e-12
    assert abs(row["current_minus_historical_current_ROC_AUC_pp"] - 1.0) < 1e-12
