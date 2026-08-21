"""Executable invariants for a frozen materialized-state release snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("edge", ["theta0_theta1", "theta1_theta2"])
def test_release_snapshot_uses_pre_release_materialized_states_only(edge: str) -> None:
    snapshot_path = ROOT / f"data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet"
    meta_path = snapshot_path.with_suffix(".meta.json")
    probe_path = ROOT / f"data/manifests/yambda50m_v2_cutover_probe_{edge}.jsonl"
    if not snapshot_path.exists():
        pytest.skip("Yambda release snapshot is not materialized in this checkout")

    metadata = json.loads(meta_path.read_text())
    table = pq.read_table(snapshot_path)
    values = table.to_pydict()
    release = metadata["release_cutoff"]
    assert metadata["primary_population"] == "all_materialized_states_at_release"
    assert metadata["eligibility"]["require_future_request"] is False
    assert metadata["eligibility"]["require_future_append"] is False
    assert metadata["eligibility"]["require_future_target"] is False
    assert len(values["uid"]) == metadata["state_count"]
    assert len(values["uid"]) == len(set(values["uid"]))
    assert all(timestamp < release for timestamp in values["state_timestamp"])
    assert all(1 <= length <= metadata["history_cap"] for length in values["effective_prefix_length"])
    assert all(work == length * 4 for work, length in zip(values["exact_token_layer_work"], values["effective_prefix_length"]))
    assert all(bool(value) for value in values["parent_checkpoint_hash"])

    with probe_path.open() as stream:
        probes = [json.loads(line) for line in stream]
    assert len(probes) == len(values["uid"])
    assert all(probe["target_injected"] is False for probe in probes)
    assert all(probe["request_timestamp"] == release for probe in probes)


def test_proxy_validity_records_are_bounded_by_the_next_release() -> None:
    result_path = ROOT / "results/data_audit/yambda50m_v2/cutover_probe_validity_v2.json"
    if not result_path.exists():
        pytest.skip("strict cutover-proxy validity result is not materialized in this checkout")

    result = json.loads(result_path.read_text())
    for edge in result["edges"].values():
        assert edge["evaluated_users"] <= edge["snapshot_users"]
        assert 0.0 <= edge["proxy_coverage_rate"] <= 1.0
        horizon = edge["next_release_timestamp"] - edge["release_timestamp"]
        assert all(0 < row["first_observed_event_delay_seconds"] < horizon for row in edge["records"])


def test_qmain_external_validity_uses_disjoint_panels_and_strict_proxy_window() -> None:
    result_path = ROOT / "results/data_audit/yambda50m_v2/external_validity_v3.json"
    csv_path = ROOT / "results/data_audit/yambda50m_v2/external_validity_v3.csv"
    if not result_path.exists():
        pytest.skip("Q_main external-validity result is not materialized in this checkout")

    result = json.loads(result_path.read_text())
    assert result["distribution"] == "Q_main_rank_decay_v1"
    for edge, value in result["edges"].items():
        assert value["target_injection"] is False
        assert value["proxy_eligible_states"] <= value["snapshot_states"]
        assert 0 < value["strict_proxy_coverage"] < 1
        assert "development panels 0..15" in value["cutover_risk"]
        assert "held-out panels 16..31" in value["proxy_risk"]
        assert value["skipped_reasons"]["event_after_next_release"] + value["proxy_eligible_states"] == value["snapshot_states"]
    # The compact CSV gives executable timestamp bounds without embedding rows in JSON.
    import csv
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            edge = result["edges"][row["edge_id"]]
            assert edge["release_timestamp"] < int(row["proxy_timestamp"]) < edge["next_release_timestamp"]


def test_panel_free_distortion_uses_frozen_qmain_support_without_targets() -> None:
    result_path = ROOT / "results/data_audit/yambda50m_v2/panel_free_score_distortion_v1.json"
    if not result_path.exists():
        pytest.skip("panel-free distortion audit is not materialized in this checkout")
    result = json.loads(result_path.read_text())
    assert result["distribution"] == "Q_main_rank_decay_v1"
    assert result["target_injection"] is False
    for edge, value in result["edges"].items():
        assert value["states"] > 0
        assert "1000-item" in value["metric"]
        assert value["heldout_panel_frontier"][0]["budget_ratio_requested"] == 0.0
        assert value["heldout_panel_frontier"][-1]["budget_ratio_requested"] == 1.0
        artifact = ROOT / "results/data_audit/yambda50m_v2" / f"panel_free_score_distortion_v1_{edge}.parquet"
        assert artifact.exists()


@pytest.mark.parametrize("edge", ["theta0_theta1", "theta1_theta2"])
def test_qmain_has_complete_disjoint_panel_halves(edge: str) -> None:
    panel_path = ROOT / f"data/manifests/yambda50m_v2_qmain32_v2_{edge}.parquet"
    snapshot_path = ROOT / f"data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet"
    if not panel_path.exists():
        pytest.skip("Q_main panels are not materialized in this checkout")

    panel_rows = pq.read_table(panel_path).to_pylist()
    snapshot_uids = set(pq.read_table(snapshot_path, columns=["uid"]).column("uid").to_pylist())
    by_uid = {}
    for row in panel_rows:
        assert row["edge_id"] == edge
        assert len(row["candidate_item_ids"]) == 100
        assert len(set(row["candidate_item_ids"])) == 100
        by_uid.setdefault(row["uid"], set()).add(row["panel_id"])
    assert set(by_uid) == snapshot_uids
    assert all(panel_ids == set(range(32)) for panel_ids in by_uid.values())
