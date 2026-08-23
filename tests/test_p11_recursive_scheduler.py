from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_recursive_scheduler_reuses_frozen_algorithm_and_actions() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p11_2_recursive_scheduler_replay_v1.yaml").read_text())
    assert contract["frozen_scheduler"]["primary_probe_rate"] == 0.01
    assert contract["frozen_scheduler"]["predictor"] == {
        "preprocessing": "StandardScaler", "family": "Ridge", "alpha": 1.0, "solver": "lsqr"
    }
    assert list(contract["action_mapping"]) == [
        "noop", "layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128", "exact_all"
    ]
    assert "future_quality_labels" in contract["prohibited"]
    assert "theta3_blind_edge" in contract["prohibited"]
