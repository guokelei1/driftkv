"""Independent trained-FG control and immutable shared-prefill CPU oracles."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.fg_trajectory_control import (
    TRAINED_FG_METHOD, load_fg_control, make_paired_trajectories, replay_fg_user, score_fg_timestamp,
    validate_fg_pair,
)
from experiments.ch3_motivation.code.probe_replay import replay_probe_user
from experiments.ch3_motivation.code.probe_variants import build_variant, save_variant
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.common.artifacts import sha256


def fixture(variant="full_item"):
    torch.manual_seed(87)
    cfg = HSTUConfig(num_items=64, num_behaviors=2, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent = HSTU(cfg).eval().requires_grad_(False)
    current = copy.deepcopy(parent)
    current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight)*.01)
    fg = build_variant(variant, current, stage="fg", hidden_width=8, depth=2,
        query_chunk=2, row_chunk=3).eval().requires_grad_(False)
    for parameter in fg.parameters():
        parameter.normal_(std=.1)
    full = fg.expanded("fgh").eval().requires_grad_(False)
    for module in full.h:
        for parameter in module.parameters():
            parameter.normal_(std=.2)
        module.biases[-1].fill_(.3)
    population = dict(fit=[1, 2, 3, 4], validation=[5, 6])
    models = {"v0": "parent-sha", "v1": "current-sha"}
    identity = dict(edge=1, models=models, variant=variant, seed=17,
        fit_uids=[1, 2], validation_users=[5, 6], input_binding_sha256="calibration-sha",
        candidate_seed=17, queries=16, candidate_source="feedback",
        recent_fraction=.5, calibration_candidate_sha256="pre-release-panel-sha")
    fg_metadata = dict(identity=copy.deepcopy(identity), variant=variant, stage="fg",
        method_label="fg", validation_users=2, fitting_uses_future_feedback=False)
    full_metadata = dict(identity=copy.deepcopy(identity), variant=variant, stage="fgh",
        method_label="fgh", validation_users=2, fitting_uses_future_feedback=False)
    validation = dict(source_population=population, evaluation_uids=[91, 92], edge=1,
        models=models, calibration_binding_sha256="calibration-sha")
    return parent, current, fg, full, fg_metadata, full_metadata, validation


@pytest.mark.parametrize("variant", ["head", "full_item", "layerwise_kv"])
def test_loads_real_tagged_stage_fg_and_matches_paired_fitting_identity(tmp_path, variant):
    _, _, fg, full, fg_meta, full_meta, validation = fixture(variant)
    path = tmp_path/"fg.pt"
    save_variant(path, fg, fg_meta)
    loaded, metadata, identity = load_fg_control(path, full, full_meta, **validation)
    assert loaded.stage == "fg" and loaded.h is None
    assert identity["method"] == TRAINED_FG_METHOD
    assert identity["variant"] == variant and identity["fit_uids"] == [1, 2]
    assert metadata["validation_users"] == 2 and identity["validation_uids"] == [5, 6]
    assert identity["e5_method_label"] == "fg" and identity["independent_state"]
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), fg.state_dict().values(), strict=True))


def test_loads_actual_legacy_head_format_without_inventing_variant_or_uid_membership(tmp_path):
    _, _, fg, full, fg_meta, full_meta, validation = fixture("head")
    for metadata in (fg_meta, full_meta):
        metadata.pop("variant")
        metadata.pop("method_label")
        metadata["identity"].pop("variant")
        metadata["identity"].pop("fit_uids")
        metadata["identity"]["fit_users"] = 2
        metadata["identity"]["sample_candidates"] = metadata["identity"].pop("queries")
    path = tmp_path/"fg_legacy.pt"
    torch.save(dict(config=asdict(fg.cfg), stage="fg", state_dict=fg.state_dict(), metadata=fg_meta), path)
    _, _, identity = load_fg_control(path, full, full_meta, **validation)
    assert identity["variant"] == "head" and identity["fit_uids"] == [1, 2]


@pytest.mark.parametrize("key,value,error", [
    ("seed", 19, "variants, seeds"),
    ("fit_uids", [1, 3], "fitting/validation membership"),
    ("validation_users", [5], "fitting/validation membership"),
    ("models", {"v0": "wrong-sha"}, "backbone identity"),
    ("input_binding_sha256", "different-pack", "calibration binding"),
    ("calibration_candidate_sha256", "future-candidates", "candidate protocol"),
    ("queries", 32, "query counts"),
])
def test_rejects_unpaired_fg_fit_and_data_sources(key, value, error):
    _, _, fg, full, fg_meta, full_meta, validation = fixture()
    fg_meta["identity"][key] = value
    with pytest.raises(ValueError, match=error):
        validate_fg_pair(fg, fg_meta, full, full_meta, **validation)


def test_extended_label_diagnostic_weights_and_evaluation_leakage_are_explicit():
    _, _, fg, full, fg_meta, full_meta, validation = fixture()
    fg_meta["method_label"] = "fg_extended"
    with pytest.raises(ValueError, match="method label"):
        validate_fg_pair(fg, fg_meta, full, full_meta, **validation)

    result = validate_fg_pair(fg, fg_meta, full, full_meta, expected_method_label="fg_extended", **validation)
    assert result["e5_method_label"] == "fg_extended"
    fg_meta["method_label"] = "fg"
    fg_meta["diagnostic_only"] = True
    with pytest.raises(ValueError, match="diagnostic checkpoint"):
        validate_fg_pair(fg, fg_meta, full, full_meta, **validation)
    validate_fg_pair(fg, fg_meta, full, full_meta, allow_diagnostic=True, **validation)
    fg_meta.pop("diagnostic_only")
    with pytest.raises(ValueError, match="overlap"):
        validate_fg_pair(fg, fg_meta, full, full_meta, **dict(validation, evaluation_uids=[1, 91]))
    fg_meta["fitting_uses_future_feedback"] = True
    with pytest.raises(ValueError, match="future-feedback"):
        validate_fg_pair(fg, fg_meta, full, full_meta, **validation)


@pytest.mark.parametrize("rule", ["gap_mix50", "gap_empirical100"])
def test_temporal_pairs_bind_exact_rule_manifest_and_time_provenance(rule):
    _, _, fg, full, fg_meta, full_meta, validation = fixture()
    for metadata in (fg_meta, full_meta):
        metadata["identity"].update(query_delta_rule=rule, gap_manifest_sha256="a"*64,
            time_rule={"strict": "behavior timestamp < query timestamp"}, source_cutoff=86400,
            candidate_order="feedback8_then_uniform8")
    result = validate_fg_pair(fg, fg_meta, full, full_meta, **validation)
    assert result["query_delta_rule"] == rule and result["gap_manifest_sha256"] == "a"*64
    for key, value in [("query_delta_rule", "cutover"), ("gap_manifest_sha256", "b"*64),
                       ("source_cutoff", 86401), ("candidate_order", "reordered")]:
        modified = copy.deepcopy(fg_meta)
        modified["identity"][key] = value
        with pytest.raises(ValueError, match="temporal"):
            validate_fg_pair(fg, modified, full, full_meta, **validation)
    for metadata in (fg_meta, full_meta):
        metadata["identity"].pop("gap_manifest_sha256")
    with pytest.raises(ValueError, match="manifest SHA256"):
        validate_fg_pair(fg, fg_meta, full, full_meta, **validation)


def sampling_pair(tmp_path, rule="cutover", n=128, mode=None):
    """Real six-view manifest with a small loaded model and full reserved roles."""
    from tests.experiments.ch3_motivation.test_calibration_sampling import fixture as sampling_fixture
    manifest,candidates=sampling_fixture()
    root=tmp_path/'fit256';folder=root/'edge_1';folder.mkdir(parents=True)
    population=dict(fit=manifest['target_fit_uids'],validation=manifest['validation_uids'])
    (root/'population.json').write_text(json.dumps(population))
    (folder/'inputs.json').write_text(json.dumps(dict(settings=dict(release_day=20))))
    (folder/'calibration_feedback_candidates.json').write_text(json.dumps(candidates))
    (folder/'histories.npz').write_bytes(b'frozen CPU identity fixture history')
    source=dict(input_root=str(root),population_sha256=sha256(root/'population.json'),
        sha256=sha256(folder/'inputs.json'),candidate_sha256=sha256(folder/'calibration_feedback_candidates.json'),
        history_path=str(folder/'histories.npz'),history_sha256=sha256(folder/'histories.npz'))
    manifest.update(sourcebinding=source,source_sha256={})
    path=tmp_path/'sampling.json';path.write_text(json.dumps(manifest))
    values=fixture();parent,current,fg,full,fg_meta,full_meta,validation=values
    driver='run_probe_calibration.py' if mode is None else 'run_probe_calibration_init.py'
    driver_sha=sha256(ROOT/'experiments/ch3_motivation/code'/driver)
    for metadata in (fg_meta,full_meta):
        metadata['identity'].update(fitting_driver=driver,fitting_driver_sha256=driver_sha,
            numerical_sources={f'experiments/ch3_motivation/code/{driver}':driver_sha},
            query_delta_rule=rule,sampling_manifest=str(path),sampling_manifest_sha256=sha256(path),
            gap_manifest=str(path),gap_manifest_sha256=sha256(path),
            input_binding_sha256=source['sha256'],population_sha256=source['population_sha256'],
            calibration_candidate_sha256=source['candidate_sha256'],fit_users=n,fit_uids=population['fit'][:n],
            validation_users=population['validation'],gradient_fitting_users=n,teacher_fitting_users=n,
            candidate_catalog_source_users=256,time_law_source_users=256,recent_fraction=.75,
            query_delta_layout='[B,16]',time_rule=manifest['time_rule'],source_cutoff=manifest['source_cutoff'],
            candidate_order=manifest['candidate_order'],fitting_uses_future_feedback=False)
        metadata.update(training_users=n,validation_users=16,teacher_preparation_users=n+16,
            actual_teacher_users_captured=n,actual_validation_users_captured=16)
        if mode is not None:
            from experiments.ch3_motivation.code.probe_initialization import initialization_spec
            spec=initialization_spec(fg.cfg.head_dim,fg.cfg.row_dim,mode)
            metadata['identity'].update(h_initialization=mode,h_initialization_spec=spec,
                initialization_helper_sha256=spec['helper_sha256'])
            metadata['identity']['numerical_sources']['experiments/ch3_motivation/code/probe_initialization.py']=spec['helper_sha256']
    validation.update(source_population=population,evaluation_uids=[90001,90002],calibration_binding_sha256=source['sha256'])
    return values


@pytest.mark.parametrize('rule',['cutover','gap_mix50','gap_empirical100'])
def test_sampling_v1_pair_binds_real_manifest_including_cutover(tmp_path,rule):
    _,_,fg,full,fg_meta,full_meta,validation=sampling_pair(tmp_path,rule)
    result=validate_fg_pair(fg,fg_meta,full,full_meta,**validation)
    assert result['query_delta_rule']==rule
    assert result['sampling_manifest_sha256']==result['gap_manifest_sha256']==fg_meta['identity']['sampling_manifest_sha256']
    assert result['sampling_view_key']==f'feedback75_{rule}'
    assert result['candidate_catalog_source_users']==result['time_law_source_users']==256
    assert result['gradient_fitting_users']==result['teacher_fitting_users']==128


@pytest.mark.parametrize('fault',['null_sha','null_alias','null_path','driver_sha','catalog64','law64',
    'gradient64','missing_teacher_n','teacher_capture','validation_capture','teacher_prep','time_rule','cutoff','candidate_order','fit_order'])
def test_sampling_v1_pair_rejects_matching_but_invalid_new_identities(tmp_path,fault):
    _,_,fg,full,fg_meta,full_meta,validation=sampling_pair(tmp_path)
    for metadata in (fg_meta,full_meta):
        identity=metadata['identity']
        if fault=='null_sha':identity['sampling_manifest_sha256']=None
        elif fault=='null_alias':identity['gap_manifest_sha256']=None
        elif fault=='null_path':identity['sampling_manifest']=None
        elif fault=='driver_sha':identity['fitting_driver_sha256']='b'*64
        elif fault=='catalog64':identity['candidate_catalog_source_users']=64
        elif fault=='law64':identity['time_law_source_users']=64
        elif fault=='gradient64':identity['gradient_fitting_users']=64
        elif fault=='missing_teacher_n':identity.pop('teacher_fitting_users')
        elif fault=='teacher_capture':metadata['actual_teacher_users_captured']=256
        elif fault=='validation_capture':metadata.pop('actual_validation_users_captured')
        elif fault=='teacher_prep':metadata['teacher_preparation_users']=272
        elif fault=='time_rule':identity['time_rule']='different chronology'
        elif fault=='cutoff':identity['source_cutoff']+=1
        elif fault=='candidate_order':identity['candidate_order']='permuted'
        else:identity['fit_uids'].reverse()
    with pytest.raises(ValueError,match='sampling calibration'):
        validate_fg_pair(fg,fg_meta,full,full_meta,**validation)


def test_sampling_pair_cannot_mix_driver_recipes_or_drop_into_legacy_cutover(tmp_path):
    _,_,fg,full,fg_meta,full_meta,validation=sampling_pair(tmp_path)
    fg_meta['identity']['learning_rate']=1e-5
    with pytest.raises(ValueError,match='sampling calibration fitting identities'):
        validate_fg_pair(fg,fg_meta,full,full_meta,**validation)
    for metadata in (fg_meta,full_meta):
        metadata['identity'].pop('fitting_driver')
    with pytest.raises(ValueError,match='no recognized explicit fitting driver'):
        validate_fg_pair(fg,fg_meta,full,full_meta,**validation)


@pytest.mark.parametrize('mode',['standard','balanced'])
def test_sampling_initialization_pair_has_explicit_mode_and_actual_dimensions(tmp_path,mode):
    _,_,fg,full,fg_meta,full_meta,validation=sampling_pair(tmp_path,mode=mode)
    result=validate_fg_pair(fg,fg_meta,full,full_meta,**validation)
    assert result['h_initialization']==mode and result['fitting_driver']=='run_probe_calibration_init.py'
    assert result['h_initialization_spec']['query_dim']==fg.cfg.head_dim
    assert result['h_initialization_spec']['row_dim']==fg.cfg.row_dim
    for metadata in (fg_meta,full_meta):
        metadata['identity']['h_initialization_spec']['query_dim']+=1
    with pytest.raises(ValueError,match='actual dimensions'):
        validate_fg_pair(fg,fg_meta,full,full_meta,**validation)


def test_cross_initialization_mode_fg_pair_cannot_be_pooled(tmp_path):
    _,_,fg,full,fg_meta,full_meta,validation=sampling_pair(tmp_path,mode='standard')
    fg_meta['identity']['h_initialization']='balanced'
    with pytest.raises(ValueError,match='sampling calibration fitting identities'):
        validate_fg_pair(fg,fg_meta,full,full_meta,**validation)


def events_and_queries():
    events = (np.r_[np.arange(8)*100+100, [86401, 86402, 86404, 86405, 86408]],
              np.arange(1, 14), np.arange(13)%2)
    settings = dict(edge=1, release_day=1, end_day=2, history_length=8,
        append_band_size=4, query_batch=2)
    queries = {86402: [dict(uid=91, query_timestamp=86402, raw_item_id=100+i,
                           item_idx=20+i, label=i%2) for i in range(3)],
               86406: [dict(uid=91, query_timestamp=86406, raw_item_id=200+i,
                           item_idx=23+i, label=i%2) for i in range(2)]}
    return events, settings, queries


@pytest.mark.parametrize("variant", ["head", "full_item", "layerwise_kv"])
@torch.no_grad()
def test_shared_prefill_is_read_only_and_fg_matches_its_own_dense_e14_reference(monkeypatch, variant):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    parent, current, fg, full, _, _, _ = fixture(variant)
    events, cfg, requests = events_and_queries()
    calls, original = [], parent.compute_kv
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(parent, "compute_kv", counted)
    pair = make_paired_trajectories(parent, current, fg, full, events, cfg, capture=False)
    assert len(calls) == 1
    assert pair.fg.pool is not pair.full_h.pool
    initial = (pair.initial.cache.k.clone(), pair.initial.cache.v.clone(), pair.initial.cursor,
               pair.initial.time, copy.copy(pair.initial.ledger), pair.initial.producers.copy())
    pair.fg._prepare()
    pair.full_h._prepare()
    # Identical f/g weights initially; only full-H owns nonzero h.
    torch.testing.assert_close(pair.fg.cache.k, pair.full_h.cache.k)
    pair.full_h.advance(86406)
    saved_h = pair.full_h.cache.k.clone(), pair.full_h.cache.v.clone(), pair.full_h.native.cursor
    actual = []
    for timestamp, rows in requests.items():
        actual.extend(score_fg_timestamp(pair.fg, rows, uid=91, timestamp=timestamp, query_batch=2))
    assert pair.fg.native.cursor == 12 and pair.full_h.native.cursor == 12
    assert pair.fg.source.seq_len == pair.full_h.source.seq_len == 4
    assert not torch.allclose(pair.fg.cache.k, pair.full_h.cache.k, atol=1e-8, rtol=1e-8)
    assert torch.equal(pair.full_h.cache.k, saved_h[0]) and torch.equal(pair.full_h.cache.v, saved_h[1])
    assert torch.equal(pair.initial.cache.k, initial[0]) and torch.equal(pair.initial.cache.v, initial[1])
    assert (pair.initial.cursor, pair.initial.time, pair.initial.ledger) == initial[2:5]
    assert np.array_equal(pair.initial.producers, initial[5])
    reference, _ = replay_probe_user(parent, current, fg, events, requests, 91, cfg)
    assert [row["method"] for row in actual] == [TRAINED_FG_METHOD]*5
    assert [row["raw_item_id"] for row in actual] == [row["raw_item_id"] for row in reference]
    torch.testing.assert_close(torch.tensor([row["score"] for row in actual]),
        torch.tensor([row["score"] for row in reference]), atol=1e-6, rtol=1e-6)
    # Reusing the shared initial for a complete FG replay gives the same scores;
    # flipping feedback labels cannot alter either cached state or predictions.
    changed = {timestamp: [dict(row, label=1-row["label"]) for row in rows] for timestamp, rows in requests.items()}
    replayed, report = replay_fg_user(parent, current, fg, events, changed, 91, cfg,
        initial=pair.initial, capture=False)
    torch.testing.assert_close(torch.tensor([row["score"] for row in actual]),
        torch.tensor([row["score"] for row in replayed]), atol=1e-6, rtol=1e-6)
    assert report["counts"]["appended_rows"] == 4
    assert not report["unscored_suffix_advanced"]


def test_rejects_cross_branch_pool_or_wrong_shared_user_prefill(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    parent, current, fg, full, _, _, _ = fixture()
    events, cfg, _ = events_and_queries()
    pair = make_paired_trajectories(parent, current, fg, full, events, cfg, capture=False)
    with pytest.raises(ValueError, match="share a mutable graph pool"):
        make_paired_trajectories(parent, current, fg, full, events, cfg,
            fg_pool=pair.fg.pool, full_h_pool=pair.fg.pool, initial=pair.initial)
    other = (events[0], events[1]+1, events[2])
    with pytest.raises(ValueError, match="different raw user events"):
        make_paired_trajectories(parent, current, fg, full, other, cfg, initial=pair.initial, capture=False)


@pytest.mark.parametrize("variant", ["head", "full_item", "layerwise_kv"])
@pytest.mark.parametrize("execution", ["dense", "projected"])
def test_formal_e6_user_records_two_controls_and_retains_every_feedback(monkeypatch, variant, execution):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    from experiments.ch3_motivation.code.run_lowrank import evaluate_e6_user
    from experiments.ch3_motivation.code.fg_trajectory_control import H_DISABLED_DIAGNOSTIC
    parent, current, fg, full, _, _, _ = fixture(variant)
    events, cfg, requests = events_and_queries()
    cfg["stop_after_last_query"] = True
    rows, snapshots, report = evaluate_e6_user(parent, current, full, fg, events, requests, 91, cfg,
        known_items=64, candidate_size=2, execution=execution, verification_heads=0,
        compression_options=dict(ranks=(0, 2), variants=("both_norm",), method="exact"))
    groups = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    assert set(groups) == {TRAINED_FG_METHOD, H_DISABLED_DIAGNOSTIC, "full", "both_norm_r0", "both_norm_r2"}
    expected_keys = {(row["uid"], row["query_timestamp"], row["raw_item_id"]) for items in requests.values() for row in items}
    for values in groups.values():
        assert len(values) == 5
        assert {(row["uid"], row["query_timestamp"], row["raw_item_id"]) for row in values} == expected_keys
    reference, _ = replay_fg_user(parent, current, fg, events, requests, 91, cfg, capture=False)
    torch.testing.assert_close(torch.tensor([row["score"] for row in groups[TRAINED_FG_METHOD]]),
        torch.tensor([row["score"] for row in reference]), atol=1e-6, rtol=1e-6)
    assert len(snapshots) >= 2 and all(item["full_rank_logit_max_error"] < 1e-5 for item in snapshots)
    assert report["shared_old_prefill"]["calls"] == 1
    assert report["trained_fg_own_trajectory"]["counts"]["appended_rows"] == 4


def test_diagnostic_metadata_can_use_the_actual_top_level_seed():
    _, _, fg, full, fg_meta, full_meta, validation = fixture()
    for metadata in (fg_meta, full_meta):
        metadata["seed"] = metadata["identity"].pop("seed")
        metadata["diagnostic_only"] = True
    result = validate_fg_pair(fg, fg_meta, full, full_meta, allow_diagnostic=True, **validation)
    assert result["seed"] == 17
