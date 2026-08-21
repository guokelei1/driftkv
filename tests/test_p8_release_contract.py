from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from hstu_kvcache.data import feedback_history_stratum_v2

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8_contract_freezes_all_models_seeds_releases_and_stop_boundary() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/f_release_chain_contract_v1.yaml").read_text())
    assert contract["model_roles"]["primary_isolation_model"] == "M0-F"
    assert contract["model_roles"]["shared_state_companion"] == "M1"
    assert contract["model_roles"]["seeds"] == [17, 37, 71]
    assert set(contract["release_types"]) == {
        "R0_output_only", "R1_routine_continual", "R2_periodic_encoder_refresh"
    }
    assert contract["authorization"]["theta3_blind_qualification"] is False
    assert contract["authorization"]["tomography"] is False
    assert contract["authorization"]["migration_actions_or_controller"] is False


def test_p8_windows_are_causal_and_training_budgets_are_coverage_only() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/f_release_chain_contract_v1.yaml").read_text())
    windows = contract["time_windows_seconds"]
    assert windows["update1_admission_dev"][1] == windows["edge1_cutover"]
    assert windows["update2_admission_dev"][1] == windows["edge2_cutover"]
    assert windows["edge1_evaluation"][0] == windows["edge1_cutover"]
    assert windows["edge2_evaluation"][0] == windows["edge2_cutover"]
    audit_path = ROOT / "results/data_audit/yambda50m_p8/release_window_coverage_v1.json"
    assert sha256(audit_path) == contract["input_hashes"]["coverage_audit"]
    audit = json.loads(audit_path.read_text())
    assert audit["selection_uses_H_or_S"] is False
    budgets = contract["prospective_data"]["training_unique_query_budget_per_task"]
    assert budgets == {"update1": 2176, "update2": 2304}
    assert audit["coverage"]["update1_train"]["R_rankable"] == 2176
    assert audit["coverage"]["update2_train"]["R_rankable"] == 2304


def test_p8_materialization_is_sealed_and_evaluation_fidelity_is_label_free() -> None:
    summary = json.loads((ROOT / "data/manifests/p8_release_v1/materialization_summary.json").read_text())
    assert summary["P7_8_rewritten"] is False
    assert summary["selection"]["selection_uses_H_or_S"] is False
    assert len(summary["indices"]) == 6
    contract_hash = sha256(ROOT / "configs/contracts/f_release_chain_contract_v1.yaml")
    for split, artifact in summary["indices"].items():
        path = ROOT / artifact["path"]
        assert sha256(path) == artifact["sha256"]
        index = json.loads(path.read_text())
        assert index["contract_hash"] == contract_hash
        assert index["P7_8_rewritten"] is False
        if "evaluation" in split:
            assert {"F:quality", "F:fidelity"} <= set(index["views"])


def test_feedback_history_strata_v2_uses_lifetime_before_cap() -> None:
    events = [(3, index, 1) for index in range(1, 700)]
    events[0] = (11, 1, 1)
    events[-100] = (12, 600, 1)
    events[-5] = (13, 695, 1)
    assert feedback_history_stratum_v2(events, 11, 800) == "seen_only_before_512"
    assert feedback_history_stratum_v2(events, 12, 800) == "old_seen"
    assert feedback_history_stratum_v2(events, 13, 800) == "recent_seen"
    assert feedback_history_stratum_v2(events, 14, 800) == "never_seen"


def test_p8_eval_split_contains_all_four_feedback_strata() -> None:
    import pyarrow.parquet as pq

    root = ROOT / "data/manifests/p8_release_v1/edge1_evaluation"
    index = json.loads((root / "manifest.index.json").read_text())
    tables = [pq.read_table(root / shard["path"], columns=["workload", "manifest_kind", "target_stratum"]) for shard in index["request_shards"]]
    values = set()
    for table in tables:
        rows = table.to_pydict()
        values.update(
            str(stratum)
            for workload, kind, stratum in zip(rows["workload"], rows["manifest_kind"], rows["target_stratum"], strict=True)
            if workload == "F" and kind == "quality"
        )
    assert values == {"recent_seen", "old_seen", "seen_only_before_512", "never_seen"}
