"""E6 snapshot/label alignment checks, not partial-population quality evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory
from experiments.ch3_motivation.code.probe_projected_fast import (
    ProjectedFastProbeTrajectory, ProjectedGraphPool,
)
from experiments.ch3_motivation.code.probes import ProbeConfig, ProbeLadder
from experiments.ch3_motivation.code.probe_variants import build_variant
from experiments.ch3_motivation.code.run_lowrank import (
    bootstrap_retention, candidate_panel, compress_snapshot, release_candidate_panels, scenario_inputs,
    validate_fit_roles,
)
from experiments.ch3_motivation.code.fg_trajectory_control import TRAINED_FG_METHOD, H_DISABLED_DIAGNOSTIC
from hstu_kvcache.models import HSTU, HSTUConfig


def test_candidate_panel_is_unique_seeded_and_preserves_all_real_rows():
    requests = [dict(item_idx=9), dict(item_idx=15), dict(item_idx=9)]
    panel, positions = candidate_panel(requests, uid=83, timestamp=909, known_items=1000)
    assert len(panel) == len(set(panel)) == 230
    assert [panel[index] for index in positions] == [9, 15, 9]
    assert candidate_panel(requests, uid=83, timestamp=909, known_items=1000) == (panel, positions)
    assert candidate_panel(requests, uid=83, timestamp=910, known_items=1000)[0] != panel
    last, last_positions = candidate_panel([dict(item_idx=1000)], uid=83, timestamp=909,
        known_items=1000)
    assert last[last_positions[0]] == 1000
    many = [dict(item_idx=i) for i in range(1, 234)]
    assert len(candidate_panel(many, uid=1, timestamp=1, known_items=1000)[0]) == 233


def test_release_panels_preserve_all_future_targets_and_bound_overflow():
    by_time = {11: [dict(item_idx=item) for item in range(1, 232)],
               12: [dict(item_idx=item) for item in (19, 232, 233)]}
    panels, locations = release_candidate_panels(by_time, uid=19, release_timestamp=10, known_items=1000)
    assert len(panels) == 2
    assert all(len(panel) == len(set(panel)) == 230 for panel in panels)
    assert set(locations) == set(range(1, 234))
    assert all(panels[group][position] == item for item, (group, position) in locations.items())
    assert release_candidate_panels(by_time, uid=19, release_timestamp=10, known_items=1000) == (panels, locations)
    reversed_order = {12: list(reversed(by_time[12])), 11: list(reversed(by_time[11]))}
    assert release_candidate_panels(reversed_order, uid=19, release_timestamp=10, known_items=1000) == (panels, locations)


@torch.no_grad()
@pytest.mark.parametrize("variant", ["head", "full_item", "layerwise_kv"])
def test_e6_replays_full_model_and_does_not_change_the_shared_snapshot(monkeypatch, variant):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(83)
    config = HSTUConfig(num_items=32, num_behaviors=2, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent = HSTU(config).eval().requires_grad_(False)
    current = copy.deepcopy(parent)
    current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight) * .01)
    probe = build_variant(variant, current, stage="fgh", hidden_width=8, depth=1,
        query_chunk=3, row_chunk=2).eval().requires_grad_(False)
    for parameter in probe.parameters():
        parameter.normal_(std=.02)
    events = (np.array([100, 200, 300, 86401, 86403]), np.array([1, 2, 3, 4, 5]), np.zeros(5, dtype=np.int64))
    cfg = dict(edge=1, release_day=1, end_day=2, history_length=8, append_band_size=4)
    trajectory = ProbeTrajectory(parent, current, probe, events, cfg)
    trajectory.advance(86402)
    trajectory._prepare()
    saved = trajectory.cache.k.clone(), trajectory.cache.v.clone(), trajectory.native.cursor
    panel = [9, 10, 11, 12, 13, 14]
    values, report = compress_snapshot(current, probe, trajectory, panel, [1, 4], torch.tensor([1.]),
        ranks=(0, 1, 2), variants=("both_norm", "both_raw", "key_norm", "value_norm"), method="exact")
    assert set(values) == {"fg", "full", *[f"{v}_r{r}" for v in (
        "both_norm", "both_raw", "key_norm", "value_norm") for r in (0, 1, 2)]}
    assert all(value.shape == (2,) for value in values.values())
    assert report["full_rank_logit_max_error"] < 1e-5
    assert report["full_rank_read_max_error"] < 1e-5
    assert report["inherited_rows"] == 3
    assert torch.equal(trajectory.cache.k, saved[0]) and torch.equal(trajectory.cache.v, saved[1])
    assert trajectory.native.cursor == saved[2]


def test_retention_bootstrap_is_paired_and_keeps_main_effect_control_separate():
    def rows(scores):
        return [dict(uid=user, query_timestamp=pos, raw_item_id=pos, label=pos % 2,
                     score=scores[pos % 2]) for user in (1, 2, 3) for pos in (0, 1)]
    groups = dict(fg=rows([.5, .5]), full=rows([.1, .9]), both_norm_r0=rows([.5, .5]),
                  both_norm_r2=rows([.1, .9]))
    points, metadata = bootstrap_retention(groups, [1, 2, 3], replicates=50,
        baseline_method="fg", structural_baseline_method=None)
    result = {row["method"]: row for row in points}
    assert result["full"]["interaction_gain_retention"] == 1
    assert result["both_norm_r0"]["interaction_gain_retention"] == 0
    assert result["both_norm_r2"]["interaction_gain_retention_ci95"] == [1, 1]
    assert metadata["nonpositive_or_invalid_interaction_gain_draws"] == 0
    groups["full"][0]["raw_item_id"] = 10
    with pytest.raises(ValueError, match="same feedback rows"):
        bootstrap_retention(groups, [1, 2, 3], replicates=5,
            baseline_method="fg", structural_baseline_method=None)


def test_main_retention_uses_trained_fg_while_same_state_deletion_remains_distinct():
    def rows(scores):
        return [dict(uid=user, query_timestamp=index, raw_item_id=index, label=int(index >= 2),
            score=scores[index]) for user in (1, 2, 3) for index in range(4)]
    groups = {TRAINED_FG_METHOD: rows([.5, .5, .5, .5]),
        H_DISABLED_DIAGNOSTIC: rows([.1, .8, .7, .9]),
        "full": rows([.1, .2, .8, .9]), "both_norm_r0": list(reversed(rows([.1, .8, .7, .9])))}
    points, confidence = bootstrap_retention(groups, [1, 2, 3], replicates=30)
    actual = {point["method"]: point for point in points}
    structural = {point["method"]: point for point in confidence["same_state_h_disabled_diagnostic"]["points"]}
    assert actual["both_norm_r0"]["interaction_gain_retention"] == .5
    assert actual["both_norm_r0"]["interaction_gain_retention_ci95"] == [.5, .5]
    assert structural["both_norm_r0"]["interaction_gain_retention"] == 0
    assert confidence["baseline_method"] == TRAINED_FG_METHOD
    assert structural["both_norm_r0"]["baseline_method"] == H_DISABLED_DIAGNOSTIC
    groups.pop(TRAINED_FG_METHOD)
    with pytest.raises(KeyError):
        bootstrap_retention(groups, [1, 2, 3], replicates=5)


@torch.no_grad()
@pytest.mark.parametrize("variant", ["head", "full_item", "layerwise_kv"])
def test_projected_e6_snapshot_matches_dense_after_real_fifo_writes(monkeypatch, variant):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(41)
    cfg = HSTUConfig(num_items=32, num_behaviors=2, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent = HSTU(cfg).eval().requires_grad_(False)
    current = copy.deepcopy(parent)
    current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight)*.02)
    probe = build_variant(variant, current, stage="fgh", hidden_width=8, depth=2,
        query_chunk=3, row_chunk=2).eval().requires_grad_(False)
    for parameter in probe.parameters():
        parameter.normal_(std=.04)
    events = (np.r_[np.arange(8)*100+100, [86401, 86402, 86403]],
              np.arange(1, 12), np.arange(11)%2)
    settings = dict(edge=1, release_day=1, end_day=2, history_length=8, append_band_size=4)
    dense = ProbeTrajectory(parent, current, probe, events, settings)
    pool = ProjectedGraphPool(current, probe, window=8, capture=False)
    projected = ProjectedFastProbeTrajectory(parent, current, probe, events, settings, pool=pool)
    dense.advance(86404)
    projected.advance(86404)
    assert dense.native.cursor == projected.native.cursor == 11
    assert dense.source.seq_len == projected.source.seq_len == 5
    torch.testing.assert_close(projected.cache.k, dense.cache.k, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(projected.cache.v, dense.cache.v, atol=1e-6, rtol=1e-6)
    panel, positions, delta = [13, 14, 15, 16, 17, 18], [1, 4], torch.tensor([1.])
    options = dict(ranks=(0, 1, 2), variants=("both_norm", "both_raw", "key_norm", "value_norm"), method="exact")
    reference, _ = compress_snapshot(current, probe, dense, panel, positions, delta,
        interaction_execution="dense", **options)
    actual, report = compress_snapshot(current, probe, projected, panel, positions, delta,
        interaction_execution="projected", **options)
    for name in reference:
        torch.testing.assert_close(actual[name], reference[name], atol=1e-6, rtol=1e-6)
    batched, batch_report = compress_snapshot(current, probe, projected, panel, positions, delta,
        interaction_execution="projected", batch_rank_scores=True, **options)
    for name in actual:
        torch.testing.assert_close(batched[name], actual[name], atol=1e-6, rtol=1e-6)
    assert batch_report["rank_score_execution"] == "batched across ranks"
    assert report["full_rank_logit_max_error"] < 1e-5
    assert report["full_rank_read_max_error"] < 1e-5


def scenario_fixture(tmp_path):
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    def save(path, value):
        path.write_text(json.dumps(value))
        return path
    population = dict(fit=list(range(1, 65)), validation=list(range(65, 81)))
    population_path = save(tmp_path / "population.json", population)
    history = tmp_path / "histories.npz"
    users = list(range(501, 2501))
    np.savez(history, uids=np.asarray(users+population["fit"]+population["validation"]))
    requests = tmp_path / "requests.parquet"
    pq.write_table(pa.table(dict(uid=users)), requests)
    binding = dict(settings=dict(edge=1), population_sha256=sha(population_path),
        history=dict(path=history.name, sha256=sha(history)),
        requests=dict(path=requests.name, sha256=sha(requests)))
    binding_path = save(tmp_path / "inputs.json", binding)
    models = {"v0": "old-model-hash", "v1": "new-model-hash"}
    qualification = dict(new_rebuild_minus_old_full=.62 - .6, new_rebuild_minus_reuse=.62 - .59,
                         point_qualifies=True, uid95_qualifies=True)
    summary = dict(quality_users=2000, quality_uids=users,
        quality_uids_sha256=hashlib.sha256(json.dumps(users).encode()).hexdigest(),
        qualification=qualification, **qualification,
        auc=dict(old_full=.6, reuse=.59, rebuild=.62),
        new_rebuild_minus_old_full_ci95=[.01, .03], new_rebuild_minus_reuse_ci95=[.02, .04],
        identity=dict(input_binding_sha256=sha(binding_path), models=models))
    summary_path = save(tmp_path / "summary.json", summary)
    reference = lambda path: dict(path=str(path), sha256=sha(path))
    manifest = dict(schema_version=1, edge=1, uids=users, roles=dict(quality=users, e6=users),
        qualification=dict(**qualification, summary_path=str(summary_path),
                           summary_sha256=sha(summary_path), scope="outcome-conditioned diagnostic examples"),
        history=reference(history), requests=reference(requests), input_binding=reference(binding_path),
        source_population=reference(population_path), selection=dict(panel_name="feedback_union_2000"))
    manifest_path = save(tmp_path / "scenario_manifest.json", manifest)
    return manifest_path, manifest, population, models, sha(binding_path)


def test_formal_e6_requires_same_qualified_feedback_population_and_checked_files(tmp_path):
    path, manifest, population, models, binding_hash = scenario_fixture(tmp_path)
    _, users, roles, provenance = scenario_inputs(tmp_path, 1, population_file=path)
    assert len(users) == 2000 and roles == population
    assert provenance["qualification"]["point_qualifies"]
    manifest["roles"]["e6"] = manifest["roles"]["e6"][:-1]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="membership/count|exactly its feedback population"):
        scenario_inputs(tmp_path, 1, population_file=path)
    manifest["roles"]["e6"] = manifest["roles"]["quality"]
    path.write_text(json.dumps(manifest))
    Path(manifest["requests"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="requests path/hash"):
        scenario_inputs(tmp_path, 1, population_file=path)


def test_fit_role_validation_accepts_explicit_tagged_variant_and_rejects_leakage(tmp_path):
    _, _, population, models, binding_hash = scenario_fixture(tmp_path)
    metadata = dict(identity=dict(edge=1, models=models, input_binding_sha256=binding_hash,
        fit_uids=population["fit"], validation_users=population["validation"], seed=17, variant="full_item"))
    result = validate_fit_roles(metadata, population, list(range(501, 2501)), 1, models, binding_hash)
    assert result["variant"] == "full_item"
    with pytest.raises(ValueError, match="overlap"):
        validate_fit_roles(metadata, population, [1, 501], 1, models, binding_hash)
    with pytest.raises(ValueError, match="different edge/backbone"):
        validate_fit_roles(metadata, population, [501], 1, {"v0": "another-model"}, binding_hash)


def test_measured_negative_update_edge_is_retained_without_changing_roles_or_files(tmp_path):
    path, manifest, _, _, _ = scenario_fixture(tmp_path)
    summary_path = Path(manifest["qualification"]["summary_path"])
    summary = json.loads(summary_path.read_text())
    summary["auc"] = dict(old_full=.60, reuse=.58, rebuild=.59)
    fields = dict(new_rebuild_minus_old_full=.59-.60, new_rebuild_minus_reuse=.59-.58,
        point_qualifies=False, uid95_qualifies=False)
    summary.update(fields, qualification=fields,
        new_rebuild_minus_old_full_ci95=[-.02, .01], new_rebuild_minus_reuse_ci95=[.001, .03])
    summary_path.write_text(json.dumps(summary))
    manifest["qualification"].update(fields,
        summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest())
    path.write_text(json.dumps(manifest))
    _, users, _, provenance = scenario_inputs(tmp_path, 1, population_file=path)
    assert len(users) == 2000 and not provenance["qualification"]["point_qualifies"]
    assert users == manifest["roles"]["e6"]


def test_unqualified_original_population_is_diagnostic_only(tmp_path):
    with pytest.raises(ValueError, match="requires --population-file"):
        scenario_inputs(tmp_path, 1, diagnostic=False)
