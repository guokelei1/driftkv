"""Paired teacher-specific residuals, scale confounding, and block provenance."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.analyze_kv_version_refresh import aggregate_users, analyze, write_outputs


def _user(uid, refresh=6.25):
    # Target logits have unrelated offsets and their residual scales differ100x.
    # The unchanged map keeps exactly the same residual/source ratio (.25).
    endpoints = {}
    for target, version, teacher, residuals in (
        ("old_target", 2, 0., dict(source_unmapped=1., unchanged=.25, teacher=0.)),
        ("new_target", 3, 1000., dict(source_unmapped=100., unchanged=25., refresh=refresh, teacher=0.))):
        endpoints[target] = dict(version=version,
            scores={method: [teacher + np.sqrt(mse)] * 16 for method, mse in residuals.items()},
            score_mse=residuals)
    return dict(uid=uid, history_length=1024, query_delta=100,
                panel_candidates=list(range(16)), endpoints=endpoints)


def test_different_teacher_scales_do_not_turn_raw_change_into_normalized_decay():
    result = aggregate_users([_user(uid) for uid in range(2000)], replicates=32)
    point = result["contrasts"][0]
    assert point["cross_version_unchanged_residual_change"]["mean"] == 24.75
    assert point["cross_version_source_normalized_residual_change"]["mean"] == 0.
    assert point["cross_version_source_normalized_residual_change"]["ci95"] == [0., 0.]
    assert point["new_target_refresh_improvement"]["mean"] == 18.75
    assert point["new_target_refresh_relative_improvement_percent"]["mean"] == 75.
    assert len(result["endpoints"]) == 7


def test_adverse_refresh_and_zero_baseline_are_preserved():
    users = [_user(uid, refresh=100.) for uid in range(8)]
    point = aggregate_users(users, replicates=16)["contrasts"][0]
    assert point["new_target_refresh_improvement"]["mean"] == -75.
    assert point["new_target_refresh_relative_improvement_percent"]["mean"] == -300.
    for user in users:
        for endpoint in user["endpoints"].values():
            endpoint["score_mse"] = {method: 0. for method in endpoint["score_mse"]}
    result = aggregate_users(users, replicates=16)
    assert result["contrasts"][0]["new_target_refresh_relative_improvement_percent"]["mean"] is None
    assert result["contrasts"][0]["cross_version_source_normalized_residual_change"]["ci95"] is None
    assert result["endpoints"][0]["relative_to_source_unmapped"]["invalid_bootstrap_draws"] == 16


def _run(tmp_path, users, diagnostic):
    (tmp_path / "blocks").mkdir(parents=True)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config = dict(format="fixed_history_version_refresh_population_v1", history_edge=2, target_version=3,
        users=[user["uid"] for user in users], diagnostic=diagnostic,
        predecessor_checkpoint_sha256="previous",
        refresh_fitting_identity=dict(protocol="frozen_mapped_source_version_refresh_v1",
            source_kind="actual_predecessor_mapped_cache_not_native_old_target",
            predecessor_checkpoint_sha256="previous", history_edge=2, target_version=3))
    (tmp_path / "configuration.json").write_text(json.dumps(config, indent=2))
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    path = tmp_path / "blocks/part_000000.json"
    path.write_text(json.dumps(dict(status="complete", identity_sha256=identity, offset=0,
                                   uids=config["users"], users=users)))
    summary = dict(status="complete", users=len(users), diagnostic=diagnostic,
        history_edge=2, target_version=3, feedback_labels_used=False, quality_auc_computed=False,
        artifacts=[dict(path="blocks/part_000000.json", sha256=sha(path))],
        costs=dict(once_rebuild_flops=1000, current_refresh_calibration_flops=200,
            current_refresh_population_preparation_flops=100, current_incremental_refresh_flops=300,
            current_incremental_percent_of_once_rebuild=30., prior_sunk_calibration_flops=150,
            prior_sunk_population_preparation_flops=100, prior_sunk_total_flops=250))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    return path


def test_diagnostic_cannot_become_mechanism_evidence_and_own_teacher_mse_is_verified(tmp_path):
    path = _run(tmp_path / "run", [_user(uid) for uid in range(8)], True)
    result = analyze(tmp_path / "run", replicates=16)
    assert not result["quality_use_permitted"] and result["evidence_status"] == "diagnostic_only"
    assert result["costs"]["prior_sunk_total_flops"] == 250
    write_outputs(result, tmp_path / "analysis")
    text = (tmp_path / "analysis/report.md").read_text()
    assert "缺少同预算G_old" in text and "没有后续事件" in text
    block = json.loads(path.read_text())
    block["users"][0]["endpoints"]["new_target"]["score_mse"]["unchanged"] = 1.
    path.write_text(json.dumps(block))
    with pytest.raises(ValueError, match="OWN target teacher"):
        analyze(tmp_path / "run", replicates=16)


def test_fixed2000_complete_report_and_block_hash_integrity(tmp_path):
    path = _run(tmp_path / "run", [_user(uid) for uid in range(2000)], False)
    result = analyze(tmp_path / "run", replicates=16)
    assert result["quality_use_permitted"]
    block = json.loads(path.read_text())
    block["users"][0]["query_delta"] += 1
    path.write_text(json.dumps(block))
    with pytest.raises(ValueError, match="committed hashes"):
        analyze(tmp_path / "run", replicates=16)
