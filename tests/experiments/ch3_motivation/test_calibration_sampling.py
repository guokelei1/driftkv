import copy
import json
from pathlib import Path
import sys

import pytest
import torch

from experiments.ch3_motivation.code import calibration_sampling as sampling
from experiments.ch3_motivation.code.calibration_gaps import fit_gap_law


def fixture():
    fit, validation, cutoff = list(range(1, 257)), list(range(1001, 1017)), 20 * 86400
    candidate = dict(fitting_pool_uids=fit, validation_uids=validation, known_items=1000,
        recent_items={str(uid):list(range(1, 17)) for uid in fit + validation})
    feedback = {1:[cutoff - 100, cutoff - 50, cutoff],2:[cutoff - 1],1001:[cutoff - 2]}
    behavior = {1:[cutoff - 200, cutoff - 100],2:[cutoff - 10],1001:[cutoff - 3]}
    law = fit_gap_law(feedback,behavior,fit,cutoff)
    assert law["observations"]["1"][0]["gap_seconds"] == 100  # Equal-time behavior is excluded.
    manifest = dict(schema_version=1,status="prepared",edge=1,source_cutoff=cutoff,candidate_count=16,
        candidate_seed=17,gap_seed=17,catalog_source_users=256,time_law_source_users=256,
        candidate_catalog_source_fit_uids=fit,time_law_source_fit_uids=fit,target_fit_uids=fit,
        validation_uids=validation,gap_law=law,time_rule="strict-past feedback gap",candidate_order="fixed feedback prefix then uniform",
        views=sampling.make_sampling_views(fit,validation,candidate,{uid:30 for uid in fit + validation},law,edge=1,cutoff=cutoff),cpu_seconds=dict(total=0.))
    return manifest,candidate


def test_all_six_views_have_exact_candidate_source_and_time_quotas():
    manifest,candidate = fixture()
    checked = sampling.validate_sampling_views(manifest,candidate)
    assert checked["users_per_view"] == 272 and checked["mix50_cutover_counts"]["feedback75"] == [6,2]
    assert manifest["gap_law"]["users"] == [1,2]
    for fraction,quota in ((.5,8),(.75,12)):
        for uid in (1,256,1001,1016):
            choices = [manifest["views"][sampling.view_key(fraction,rule)]["users"][str(uid)] for rule in sampling.RULES]
            assert choices[0]["candidate_ids"] == choices[1]["candidate_ids"] == choices[2]["candidate_ids"]
            assert choices[0]["candidate_ids"][:quota] == list(range(1,quota + 1))
            assert sum(row["candidate_type"] == "feedback" for row in choices[1]["assignments"]) == quota


@pytest.mark.parametrize("n", [64,128,256])
def test_loaded_prefix_keeps_fixed256_sources_and_invalidates_targets(tmp_path,n):
    manifest,candidate = fixture()
    root = tmp_path / "fit256"
    folder = root / "edge_1"
    folder.mkdir(parents=True)
    population = dict(fit=manifest["target_fit_uids"],validation=manifest["validation_uids"])
    (root / "population.json").write_text(json.dumps(population))
    (folder / "inputs.json").write_text(json.dumps(dict(settings=dict(release_day=20))))
    (folder / "calibration_feedback_candidates.json").write_text(json.dumps(candidate))
    (folder / "histories.npz").write_bytes(b"frozen fixture history identity")
    manifest["sourcebinding"] = dict(input_root=str(root),population_sha256=sampling.sha256(root / "population.json"),
        sha256=sampling.sha256(folder / "inputs.json"),candidate_sha256=sampling.sha256(folder / "calibration_feedback_candidates.json"),
        history_path=str(folder / "histories.npz"),history_sha256=sampling.sha256(folder / "histories.npz"))
    manifest["source_sha256"] = {}
    path = tmp_path / "sampling.json"
    path.write_text(json.dumps(manifest))
    views,provenance = sampling.load_sampling_manifest(path,root,1,.75,"gap_mix50",fit_users=n)
    assert list(views) == manifest["target_fit_uids"][:n] + manifest["validation_uids"]
    assert provenance["catalog_source_users"] == provenance["time_law_source_users"] == 256
    assert len(provenance["time_law_source_fit_uids"]) == 256
    samples = {uid:dict(candidates=torch.zeros(1,16,dtype=torch.long),query_delta=torch.tensor([30.]),
        teacher_scores=torch.ones(1,16),read_moment_sum=torch.tensor(1.),score_scale=torch.tensor(1.),raw=object()) for uid in views}
    applied = sampling.apply_sampling_view(samples,views,provenance)
    for uid in views:
        assert applied[uid]["candidates"].shape == applied[uid]["query_delta"].shape == (1,16)
        assert not {"teacher_scores","read_moment_sum","score_scale"} & applied[uid].keys()
        assert applied[uid]["raw"] is samples[uid]["raw"]
        assert samples[uid]["candidates"].eq(0).all() and "teacher_scores" in samples[uid]


def test_validation_cannot_contribute_time_law_and_mixed_counts_cannot_drift():
    manifest,candidate = fixture()
    invalid = copy.deepcopy(manifest)
    invalid["gap_law"]["observations"]["1001"] = invalid["gap_law"]["observations"].pop("1")
    with pytest.raises(ValueError,match="contributed"):
        sampling.validate_sampling_views(invalid,candidate)
    invalid = copy.deepcopy(manifest)
    view = invalid["views"][sampling.view_key(.75,"gap_mix50")]
    target = view["users"]["1"]["assignments"]
    index = next(i for i,row in enumerate(target) if row["source"] == "cutover_release_gap" and row["candidate_type"] == "feedback")
    target[index] = copy.deepcopy(invalid["views"][sampling.view_key(.75,"gap_empirical100")]["users"]["1"]["assignments"][index])
    view["users_sha256"] = sampling.digest(view["users"])
    with pytest.raises(ValueError,match="exactly half"):
        sampling.validate_sampling_views(invalid,candidate)


def test_twelve_feedback_supply_is_required_without_dropping_user():
    manifest,candidate = fixture()
    candidate["recent_items"]["1016"] = list(range(1,12))
    with pytest.raises(ValueError,match="cannot supply twelve"):
        sampling.make_sampling_views(manifest["target_fit_uids"],manifest["validation_uids"],candidate,
            {uid:30 for uid in manifest["target_fit_uids"] + manifest["validation_uids"]},manifest["gap_law"],edge=1,cutoff=manifest["source_cutoff"])
