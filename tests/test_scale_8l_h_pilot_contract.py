from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_h_pilot_v1.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scale_h_pilot_is_development_only_and_single_seed() -> None:
    value = yaml.safe_load(CONTRACT.read_text())
    assert value["model"] == {
        "condition": "m0_f",
        "seed": 17,
        "checkpoint_selection": "epoch_with_lowest_frozen_F_development_deployment_loss",
        "selected_epoch": 1,
    }
    assert value["data_access"]["split"] == "development"
    assert value["data_access"]["qualification_or_theta3"] is False
    assert value["authorization"]["cross_seed_scale_claim"] is False
    assert value["authorization"]["automatic_replication_seeds"] is False


def test_scale_h_pilot_seals_checkpoint_code_and_inputs() -> None:
    value = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "scale_contract_sha256": ROOT / "configs/contracts/scale_8l_v1.yaml",
        "development_manifest_sha256": ROOT / "data/manifests/p7_full_v1/development/manifest.index.json",
        "frozen_base_bundle_sha256": ROOT / "results/p7/base_fit/frozen_base_bundle_v1/bundle_manifest.json",
        "checkpoint_sha256": ROOT / "results/scale_8l_v1/theta0/m0_f_seed17/theta0_selected.pt",
        "raw_evaluator_sha256": ROOT / "scripts/eval_scale_8l_h_raw.py",
        "adjudicator_sha256": ROOT / "scripts/adjudicate_scale_8l_h_pilot.py",
    }
    for key, path in paths.items():
        assert value["sealed_inputs"][key] == sha256(path)


def test_scale_h_pilot_only_authorizes_release_chain_after_pass() -> None:
    value = yaml.safe_load(CONTRACT.read_text())
    assert "release_chain_pilot" in value["authorization"]["pass"]
    assert "stop_before_release_training" in value["authorization"]["fail"]
    assert value["gates"]["pass_requires_all_target_free_and_quality_gates"] is True
