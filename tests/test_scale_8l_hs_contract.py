from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_hs_v1.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scale_hs_uses_true_uid_rolling_lineage() -> None:
    value = yaml.safe_load(CONTRACT.read_text())
    lineage = value["lineage"]
    assert lineage["materialize_once_per_uid_at_cutover"] is True
    assert lineage["append_between_release_and_query"].startswith("every_listen_strictly_before_query")
    assert lineage["query_mutates_persistent_state"] is False
    assert lineage["rolling_cap"] == 1024
    assert lineage["eviction"] == "before_each_append"
    assert lineage["reuse_state_across_all_uid_queries"] is True


def test_scale_hs_separates_H_S_and_model_update_quality() -> None:
    paths = yaml.safe_load(CONTRACT.read_text())["paths"]
    assert paths["H_request_local"]["comparison"] == "RequestLocalCurrentFull1024_vs_CurrentRecent32"
    assert paths["S_rolling"]["comparison"] == "CurrentExactRolling_vs_ReuseParentRolling"
    assert paths["release_quality"]["comparison"] == "RequestLocalCurrentFull1024_vs_PreviousFull1024"
    assert paths["request_local_full_vs_exact_rolling"]["role"].startswith("mandatory")


def test_scale_hs_remains_development_only_and_label_free_during_scoring() -> None:
    value = yaml.safe_load(CONTRACT.read_text())
    assert value["data_access"]["qualification_or_theta3"] is False
    assert value["data_access"]["labels_enter_score_generation"] is False
    assert value["data_access"]["quality_metadata_joined_after_quality_population_logits_are_computed"] is True
    assert value["data_access"]["quality_and_fidelity_populations_scored_independently"] is True
    invalidation = value["data_access"]["failed_canary_invalidation"]
    assert invalidation["population_assumption_failure"]["failure_stage"].startswith("before_model_load")
    assert invalidation["relative_path_serialization_failure"]["scope"] == "deterministic_canary_only"
    assert value["statistics"]["formal_cross_seed_claim"] is False


def test_scale_hs_seals_code_inputs_and_all_three_checkpoints() -> None:
    value = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "scale_contract_sha256": ROOT / "configs/contracts/scale_8l_v1.yaml",
        "p8_manifest_summary_sha256": ROOT / "data/manifests/p8_release_v1/materialization_summary.json",
        "frozen_base_bundle_sha256": ROOT / "results/p7/base_fit/frozen_base_bundle_v1/bundle_manifest.json",
        "state_transition_source_sha256": ROOT / "src/hstu_kvcache/models/state_transition.py",
        "r1_edge1_checkpoint_sha256": ROOT / "results/scale_8l_v1/releases/r1_edge1/m0_f_seed17/selected.pt",
        "r1_edge2_checkpoint_sha256": ROOT / "results/scale_8l_v1/releases/r1_edge2/m0_f_seed17/selected.pt",
        "r2_checkpoint_sha256": ROOT / "results/scale_8l_v1/releases/r2/m0_f_seed17/selected.pt",
        "raw_evaluator_sha256": ROOT / "scripts/eval_scale_8l_hs_raw.py",
        "raw_sealer_sha256": ROOT / "scripts/seal_scale_8l_hs_raw.py",
        "adjudicator_sha256": ROOT / "scripts/adjudicate_scale_8l_hs.py",
    }
    for key, path in paths.items():
        assert value["sealed_inputs"][key] == sha256(path)
