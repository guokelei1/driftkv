"""Cohort attrition, observed-feedback confounding and committed input binding."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.analyze_kv_evolution import (
    METHODS, aggregate_snapshots, analyze_feedback, analyze, write_outputs,
)


def snapshot(uid, appended=0):
    # The nine unavailable users have large cutover error.  Comparing the
    # unpaired cutover mean would incorrectly suggest later improvement.
    f_mse = 2. if appended else (1. if uid < 511 else 1000.)
    forced_mse = .25 if appended else f_mse
    teacher = np.linspace(-1., 1., 16)
    scores = dict(rebuild=teacher.tolist(), reuse=(teacher + 2.).tolist(),
                  f=(teacher + np.sqrt(f_mse)).tolist(),
                  teacher_forced=(teacher + np.sqrt(forced_mse)).tolist())
    return dict(checkpoint=f"append_{appended}" if appended else "cutover", target_appends=appended,
        appended_events=appended, inherited_fraction=.8 if appended else 1.,
        candidate_residual=dict(summary=dict(raw_read_mse=.1,
            length_normalized_read_mse=.001, centered_energy_ratio=.25)),
        score_mse=dict(reuse=4., f=f_mse, teacher_forced=forced_mse), fixed_panel_scores=scores)


def test_each_checkpoint_pairs_its_own_available_users_and_keeps_support_floor():
    users = [dict(uid=i, snapshots=[snapshot(i)] + ([snapshot(i, 8)] if i < 511 else []))
             for i in range(520)]
    initial, later = aggregate_snapshots(users, replicates=64)
    assert initial["score_mse"]["f"]["current"]["mean"] > 1.
    assert initial["quality_use_permitted"]
    assert later["unique_users"] == 511 and not later["quality_use_permitted"]
    assert later["evidence_status"] == "diagnostic_only_lt512_users"
    f = later["score_mse"]["f"]
    assert f["cutover_same_users"]["mean"] == 1.
    assert f["paired_change"]["mean"] == 1.
    assert f["paired_change"]["ci95"] == [1., 1.]
    assert later["teacher_forcing_mse_improvement_over_f"]["mean"] == 1.75
    assert not later["f_read_residual"]["centered_energy_ratio"]["paired_change"]["quality_use_permitted"]


def test_real_stages_share_distinct_users_and_teacher_relative_change_removes_mix_shift():
    users, rows = [], []
    for uid in range(520):
        first, last = (100, 200) if uid < 512 else ((150, 150) if uid < 517 else (None, None))
        users.append(dict(uid=uid, first_request_timestamp=first, last_request_timestamp=last))
        phases = (("first", first), ("last", last)) if uid < 512 else (
            (("first_last", first),) if uid < 517 else ())
        for phase, timestamp in phases:
            label = uid % 2
            # A later candidate/label task can be harder even for the teacher.
            teacher = float(label) if phase == "first" else 0.
            scores = dict(rebuild=teacher, reuse=0., f=teacher + (1. if phase == "first" else 10.),
                          teacher_forced=teacher)
            for method in METHODS:
                rows.append(dict(uid=uid, query_timestamp=timestamp, raw_item_id=timestamp,
                    item_idx=timestamp, label=label, phase=phase, method=method, score=scores[method]))
    result = analyze_feedback(users, rows, replicates=64)
    assert result["support"]["committed_resident_users"] == 520
    assert result["support"]["users_with_any_real_feedback"] == 517
    assert result["support"]["single_timestamp_feedback_users"] == 5
    assert result["support"]["users_without_real_feedback"] == 3
    assert result["unique_users"] == 512 and result["quality_use_permitted"]
    assert result["primary_uids"] == list(range(512))
    f = next(row for row in result["changes"] if row["method"] == "f")
    assert f["auc_last_minus_first"] == -.5
    assert f["teacher_relative_auc_change"] == 0.
    assert f["teacher_relative_auc_change_interval"]["ci95"] == [0., 0.]
    assert all(row["auc_minus_teacher"] == 0. for row in result["points"] if row["method"] == "f")


def test_committed_digest_and_feedback_hash_bind_analysis_and_diagnostic_report(tmp_path):
    root = tmp_path / "run"
    (root / "blocks").mkdir(parents=True)
    (root / "feedback").mkdir()
    cost = tmp_path / "cost.json"
    cost.write_text(json.dumps(dict(complete_percent=59.2, within_rebuild_budget=True,
        costs=dict(candidate_reads_230=dict(users=2000), percent_of_once_rebuild=61.2))))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config = dict(format="kv_evolution_population_v1", users=[1], edge=2, diagnostic=True,
                  cost_preflight_reference=dict(path=str(cost), sha256=sha(cost)))
    (root / "configuration.json").write_text(json.dumps(config, indent=2))
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    feedback = root / "feedback/part_000000.parquet"
    pq.write_table(pa.table({"uid": pa.array([], type=pa.int64())}), feedback)
    user = dict(uid=1, snapshots=[snapshot(1)], panel_candidates=list(range(16)),
                first_request_timestamp=None, last_request_timestamp=None)
    block = dict(status="complete", offset=0, uids=[1], users=[user], identity_sha256=identity,
                 feedback_path="feedback/part_000000.parquet", feedback_sha256=sha(feedback))
    (root / "blocks/part_000000.json").write_text(json.dumps(block))
    result = analyze(root, replicates=32)
    assert result["all_requested_users_committed"]
    assert result["checkpoints"][0]["evidence_status"] == "runner_diagnostic_only"
    assert not result["checkpoints"][0]["quality_use_permitted"]
    assert result["deployment_cost"]["users"] == 2000
    write_outputs(result, tmp_path / "analysis")
    assert "没有真实标签，不计算AUC" in (tmp_path / "analysis/report.md").read_text()
    assert json.loads((tmp_path / "analysis/summary.json").read_text())["runner_diagnostic"]
    feedback.write_bytes(feedback.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="feedback hash changed"):
        analyze(root, replicates=32)
