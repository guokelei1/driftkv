from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_method_contract_is_frozen_and_label_free():
    value = yaml.safe_load((ROOT / "configs/contracts/scale_8l_method_v1.yaml").read_text())
    assert value["status"] == "frozen_before_any_8l_partial_action_score"
    assert value["scope"]["actions"] == ["noop", "layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128", "exact_all"]
    assert value["scope"]["labels_or_future_requests_in_profiler"] is False
    assert value["authorization"]["theta3"] == "prohibited"


def test_method_population_uses_1024_cap():
    for edge in ("edge1", "edge2"):
        path = ROOT / "data/manifests/scale_8l_population_v1" / edge / "manifest.json"
        if path.exists():
            import json
            value = json.loads(path.read_text())
            assert value["future_labels_or_requests_materialized"] is False
            assert value["effective_prefix_length"]["max"] <= 1024


def test_total_runner_has_no_theta3_path():
    text = (ROOT / "scripts/run_scale_8l_method_full.py").read_text()
    assert "train_theta3" not in text.lower()
    assert '"theta3_access":False' in text
    assert all(name in text for name in ("r0", "r1_edge1", "r1_edge2", "r2"))
    assert "eval_scale_8l_policy_runtime.py" in text
