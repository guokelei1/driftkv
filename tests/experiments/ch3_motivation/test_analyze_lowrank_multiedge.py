"""Known synthetic score oracles; no model-quality or paper results are made."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import analyze_lowrank_multiedge as analysis
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.analyze_probe_quality import ShardLoader
from hstu_kvcache.common.artifacts import sha256


def item(edge, seed=17, *, users=(1, 2), protocol="release", family="main", full=None, rank=None):
    rows = [dict(uid=uid, query_timestamp=edge*100, raw_item_id=label+1, item_idx=label+1,
        label=label, request_instance_ordinal=0) for uid in users for label in (0, 1)]
    keys = analysis.event_keys(pa.Table.from_pylist(rows))
    labels = keys["label"].astype(float)
    full = labels if full is None else np.asarray(full, dtype=float)
    rank = full if rank is None else np.asarray(rank, dtype=float)
    variants = analysis.VARIANTS if family == "main" and seed == 17 else ("both_norm",)
    scores = {analysis.FG: np.zeros(len(keys)), "full": full, analysis.STRUCTURAL: np.zeros(len(keys))}
    scores.update({f"{variant}_r{r}": rank.copy() for variant in variants for r in analysis.RANKS})
    return dict(id=f"{family}_{protocol}_edge_{edge}_seed_{seed}", family=family, protocol=protocol,
        edge=edge, seed=seed, users=list(users), keys=keys, scores=scores)


def complete_items():
    return [item(e, s) for e in analysis.EDGES for s in analysis.SEEDS]+[
        item(e, protocol=p, family="protocol") for e in (2, 5) for p in ("release", "timestamp")]


def brute_auc(scores, labels, indices, weights):
    result = []
    for draw in weights:
        mass = draw[indices].astype(float)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        denominator = mass[positive].sum()*mass[negative].sum()
        numerator = sum(mass[p]*mass[n]*((scores[p] > scores[n])+.5*(scores[p] == scores[n]))
            for p in positive for n in negative)
        result.append(numerator/denominator if denominator else np.nan)
    return np.asarray(result)


def test_ratio_of_equal_edge_gains_retains_negative_and_zero_edges():
    loaded = complete_items()
    # Edge full-minusFG: .5/.25/-.5/.5/0. Rank gain: .5/0/0/0/0.
    full = ([0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0])
    for row in loaded:
        if row["family"] != "main":
            continue
        row["scores"]["full"] = np.asarray(full[row["edge"]-1], dtype=float)
        for method in row["scores"]:
            if "_r" in method:
                row["scores"][method] = np.asarray(full[0] if row["edge"] == 1 else [0]*4, dtype=float)
    value = analysis.aggregate_loaded(loaded, replicates=50, seed=7)
    assert value["status"] == "complete" and value["results_count"] == 81
    family = value["families"]["main_both_norm"]
    for record in family["per_seed"]:
        metric = record["methods"]["both_norm_r2"]
        assert metric["auc_gain_over_trained_fg"] == pytest.approx(.1)
        assert metric["full_h_minus_trained_fg"] == pytest.approx(.15)
        assert metric["interaction_gain_retention_percent"] == pytest.approx(100*2/3)
    assert len(family["per_edge"]) == 15
    assert next(row for row in family["per_edge"] if row["edge"] == 3)["methods"]["full"]["full_h_minus_trained_fg"] < 0


def test_opposite_edges_cancel_in_every_shared_uid_draw():
    loaded = complete_items()
    a = np.asarray([.9, .2, .3, .8])
    for row in loaded:
        if row["family"] != "main":
            continue
        rank = a if row["edge"] == 1 else -a if row["edge"] == 2 else np.zeros(4)
        for method in row["scores"]:
            if "_r" in method:
                row["scores"][method] = rank
    result = analysis.aggregate_loaded(loaded, replicates=100, seed=9)
    for seed in result["families"]["main_both_norm"]["per_seed"]:
        metric = seed["methods"]["both_norm_r6"]
        assert metric["auc_gain_over_trained_fg"] == pytest.approx(0.)
        assert metric["auc_gain_over_trained_fg_ci95"] == pytest.approx([0., 0.])
    assert result["confidence"]["edge_sample_size"].startswith("random")


@pytest.mark.parametrize("second_users", [(2, 3), (3, 4)])
def test_union_draws_map_overlapping_and_disjoint_uid_identity(second_users):
    first = item(1, users=(1, 2), rank=[.9, .2, .3, .8])
    second = item(2, users=second_users, rank=[.4, .8, .7, .1])
    population, weights = analysis.union_weights([first, second], 100, 13)
    assert len(population) == len(set(first["users"]+second["users"]))
    assert np.all(weights.sum(axis=1) == len(population))
    counts = weights[1:, np.searchsorted(population, first["users"])].sum(axis=1)
    assert np.any(counts != len(first["users"]))  # union draws do not fix each edge's size.
    cache = {}
    result, measured = analysis._family([first, second], edges=(1, 2), seeds=(17,), variants=("both_norm",),
        population=population, weights=weights, cache=cache)
    expected = np.mean([brute_auc(row["scores"]["both_norm_r2"], row["keys"]["label"],
        np.searchsorted(population, row["keys"]["uid"]), weights) for row in (first, second)], axis=0)
    np.testing.assert_allclose(measured[17]["both_norm_r2"], expected, equal_nan=True)
    assert result["per_seed"][0]["methods"]["both_norm_r2"]["auc_ci95"] == pytest.approx(analysis._interval(expected[1:]))


@pytest.mark.parametrize("full", [[0, 0, 0, 0], [1, 0, 1, 0]])
def test_zero_or_negative_h_gain_has_absolute_intervals_without_benefit_ratio(full):
    loaded = complete_items()
    for row in loaded:
        row["scores"]["full"] = np.asarray(full, dtype=float)
    value = analysis.aggregate_loaded(loaded, replicates=25)
    for seed in value["families"]["main_both_norm"]["per_seed"]:
        metric = seed["methods"]["both_norm_r1"]
        assert metric["full_h_minus_trained_fg"] <= 0
        assert metric["interaction_gain_retention_percent"] is None
        assert metric["interaction_gain_retention_ci95"] is None
        assert metric["auc_gain_over_trained_fg_ci95"] is not None
        assert metric["signed_ratio_percent"] is None if metric["full_h_minus_trained_fg"] == 0 else metric["signed_ratio_percent"] < 0


def test_protocol_family_is_two_edge512_and_uses_direct_paired_contrasts():
    loaded = complete_items()
    for row in loaded:
        if row["family"] == "protocol" and row["protocol"] == "timestamp":
            row["scores"]["both_norm_r4"] = np.zeros(len(row["keys"]))
    result = analysis.aggregate_loaded(loaded, replicates=50)
    protocol = result["families"]["protocol512_sensitivity"]
    assert protocol["expected_edges"] == [2, 5] and protocol["users_per_edge"] == 512
    contrast = next(row for row in protocol["contrasts"] if row["method"] == "both_norm_r4")
    assert contrast["auc_release_minus_timestamp"] == pytest.approx(.5)
    assert contrast["auc_release_minus_timestamp_ci95"] == pytest.approx([.5, .5])
    assert all(seed["methods"]["both_norm_r4"]["auc"] == 1. for seed in result["families"]["main_both_norm"]["per_seed"])


def test_missing_grid_and_duplicate_id_are_explicit():
    loaded = complete_items()
    incomplete = [row for row in loaded if not(row["family"] == "main" and row["seed"] == 29 and row["edge"] == 5)]
    result = analysis.aggregate_loaded(incomplete, replicates=9)
    assert result["status"] == "pending"
    assert result["families"]["main_both_norm"]["results_count"] == 0
    assert result["families"]["seed17_kv_sensitivity"]["status"] == "complete"
    with pytest.raises(ValueError, match="duplicate E6 artifact id"):
        analysis.aggregate_loaded(loaded+[loaded[0]], replicates=9)


def test_same_edge_seeds_cannot_change_population():
    loaded = complete_items()
    loaded[1] = item(1, 23, users=(1, 3))
    with pytest.raises(ValueError, match="same-edge training seeds"):
        analysis.aggregate_loaded(loaded, replicates=9)


def test_pending_formal_produces_no_quality_results(tmp_path):
    path = tmp_path/"plan.json"
    write_json(path, dict(status="waiting_selection", jobs=[]))
    result = analysis.analyze(path, tmp_path/"analysis")
    assert result["status"] == "pending" and result["results_count"] == 0 and not result["gpu_used"]
    assert not result["families"]
    assert not (tmp_path/"analysis/points.csv").exists()


@pytest.fixture
def committed(tmp_path):
    """Actual marker/parquet layout with known synthetic scores and hash identities.

    Weight bytes are synthetic identity references here: the formal planner's
    separate real typed FG/H tests own numerical-model validation.
    """
    users = list(range(10000, 12000))
    rows = [dict(uid=uid, query_timestamp=100, raw_item_id=i+1, item_idx=i+1, label=i%2)
        for i, uid in enumerate(users)]
    requests = tmp_path/"requests.parquet"
    pq.write_table(pa.Table.from_pylist(rows), requests)
    weights = {}
    for name in ("full_h_checkpoint", "trained_fg_checkpoint", "population_file"):
        path = tmp_path/(name+".identity")
        path.write_text("synthetic committed-identity fixture "+name)
        weights[name] = str(path)
        weights["population_sha256" if name == "population_file" else name+"_sha256"] = sha256(path)
    provenance = dict(requests_path=str(requests), requests_sha256=sha256(requests), manifest_sha256=weights["population_sha256"])
    binding = dict(id="edge_1_seed_23", edge=1, seed=23, users=2000, uids=users,
        models={"model0": "old-sha", "model1": "new-sha"}, population_provenance=provenance,
        paired_control=dict(checkpoint_sha256=weights["trained_fg_checkpoint_sha256"], seed=23),
        fitting=dict(training_seed=23), qualification=dict(evaluation_window={"name": "E14"}), **weights)
    output = tmp_path/"job"
    job = dict(id="main_edge_1_seed_23", protocol="release", edge=1, seed=23, users=2000,
        variants=["both_norm"], output=str(output), ranks=list(analysis.RANKS))
    config = dict(protocol="joint_e14_request_instance_release_v1", edge=1, users=users, role="main_2000_feedback",
        models=binding["models"], ranks=list(analysis.RANKS), variants=job["variants"], population=provenance,
        full_h_checkpoint_sha256=binding["full_h_checkpoint_sha256"], trained_fg_control=binding["paired_control"],
        full_h_fitting=binding["fitting"], candidates=230, sources=analysis.release_sources(),
        decomposition=dict(method="randomized", oversample=16, power_iterations=2, batch_heads=25),
        execution=dict(checkpoint_users=4, candidate_chunk=230, row_chunk=1024, mode="projected", batch_rank_scores=True))
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    write_json(output/"configuration.json", config)
    names = analysis._methods(job["variants"])
    for offset in range(0, len(users), 4):
        block_rows = [dict(row, method=method, score=float(row["label"]) if method != analysis.FG else 0.,
            request_instance_ordinal=0) for row in rows[offset:offset+4] for method in names]
        path = output/"blocks"/f"part_{offset:06d}.parquet"
        path.parent.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pylist(block_rows), path)
        write_json(path.with_suffix(".json"), dict(identity=identity, parquet=path.name,
            sha256=sha256(path), report=dict(uids=users[offset:offset+4], users=[])))
    summary = dict(status="complete", users=2000, quality_auc_computed=True, identity_hash=identity,
        role="main_2000_feedback", full_rank_max_error=1e-7,
        points=[dict(method=method, auc=.5 if method == analysis.FG else 1.) for method in names])
    write_json(output/"summary.json", summary)
    return job, binding, rows


def test_actual_committed_columnar_format_full_event_pairing(committed):
    job, binding, rows = committed
    result = analysis.load_job(job, binding, ShardLoader(), {})
    assert len(result["keys"]) == 2000 and result["score_rows"] == 20000 and result["blocks"] == 500
    assert result["scores"][analysis.FG].sum() == 0
    assert result["keys"]["request_instance_ordinal"].sum() == 0


@pytest.mark.parametrize("fault", ["hash", "uid_slice", "missing_block", "rank_missing", "event_changed", "sources_missing", "h_checkpoint", "summary_nan"])
def test_committed_evidence_faults_are_not_silently_partial(committed, fault):
    job, binding, _ = committed
    folder = Path(job["output"])
    marker_path = folder/"blocks/part_000000.json"
    marker = json.loads(marker_path.read_text())
    if fault == "hash":
        marker["sha256"] = "0"*64
    elif fault == "uid_slice":
        marker["report"]["uids"].reverse()
    elif fault == "missing_block":
        marker_path.unlink()
    elif fault in ("rank_missing", "event_changed"):
        path = marker_path.with_suffix(".parquet")
        rows = pq.read_table(path).to_pylist()
        if fault == "rank_missing":
            rows = [row for row in rows if row["method"] != "both_norm_r16"]
        else:
            rows[0]["raw_item_id"] = 999999
        pq.write_table(pa.Table.from_pylist(rows), path)
        marker["sha256"] = sha256(path)
    elif fault == "summary_nan":
        path = folder/"summary.json"
        summary = json.loads(path.read_text())
        summary["points"][0]["auc"] = float("nan")
        path.write_text(json.dumps(summary))
    else:
        path = folder/"configuration.json"
        config = json.loads(path.read_text())
        if fault == "sources_missing":
            config["sources"] = {}
        else:
            config["full_h_checkpoint_sha256"] = "another-model"
        write_json(path, config)
        summary = json.loads((folder/"summary.json").read_text())
        summary["identity_hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        write_json(folder/"summary.json", summary)
    if fault not in ("missing_block", "sources_missing", "h_checkpoint", "summary_nan"):
        write_json(marker_path, marker)
    message = {"sources_missing": "complete numerical dependency", "h_checkpoint": "actual population/FG/H/ranks",
        "summary_nan": "nonfinite E6 summary AUC"}.get(fault)
    with pytest.raises(ValueError, match=message):
        analysis.load_job(job, binding, ShardLoader(), {})


def test_request_cache_does_not_hide_a_conflicting_declared_source_sha(committed):
    job, binding, _ = committed
    source = binding["population_provenance"]["requests_path"]
    expected = np.sort(analysis.event_keys(pq.read_table(source), assign_ordinals=True), order=analysis.KEY_FIELDS)
    cache = {(source, tuple(binding["uids"])): expected}
    binding = copy.deepcopy(binding)
    binding["population_provenance"]["requests_sha256"] = "0"*64
    folder = Path(job["output"])
    config = json.loads((folder/"configuration.json").read_text())
    config["population"] = binding["population_provenance"]
    write_json(folder/"configuration.json", config)
    summary = json.loads((folder/"summary.json").read_text())
    summary["identity_hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    write_json(folder/"summary.json", summary)
    with pytest.raises(ValueError, match="artifact hash changed"):
        analysis.load_job(job, binding, ShardLoader(), cache)


def test_ordinal_multiplicity_is_preserved_and_timestamp_missing_means_zero():
    rows = [dict(uid=1, query_timestamp=100, raw_item_id=4, item_idx=4, label=1)]*2
    expected = analysis.event_keys(pa.Table.from_pylist(rows), assign_ordinals=True)
    np.testing.assert_array_equal(expected["request_instance_ordinal"], [0, 1])
    timestamp = analysis.event_keys(pa.Table.from_pylist(rows[:1]))
    assert timestamp["request_instance_ordinal"][0] == 0
