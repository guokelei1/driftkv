"""Synthetic score/identity oracles; no real-model quality result is read."""
import hashlib
import json
import copy
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    analyze_natural_quality as analysis, calibration_cost_correction as correction, qualify_scenario,
)
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population
from hstu_kvcache.evaluation.statistics import paired_user_bootstrap
from hstu_kvcache.common.artifacts import sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fast_blocks(folder, users, identity, rows):
    for offset in range(0, len(users), 256):
        block = users[offset:offset+256]
        selected = set(block)
        actual = [row for row in rows if row["uid"] in selected]
        path = folder/f"part_{offset:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(actual), path)
        write_json(path.with_suffix(".json"), dict(fingerprint=digest(identity),
            offset=offset, users=len(block), completed_users=offset+len(block),
            uids_sha256=hashlib.sha256(json.dumps(block).encode()).hexdigest(),
            scores=len(actual), scores_sha256=sha256(path)))


@pytest.fixture(scope="module")
def committed(tmp_path_factory):
    """Actual natural marker formats with known score offsets and full 2k UIDs."""
    root = tmp_path_factory.mktemp("natural")
    users = list(range(1000, 3000))
    fit, validation = list(range(1, 257)), list(range(300, 316))
    inputs, anchors, e3, e5 = [root/name for name in ("inputs", "anchors/edge_1", "e3/edge_1", "e5")]
    inputs.mkdir()
    population = dict(fit=fit, validation=validation, main=[], confirmation=[])
    write_json(inputs/"population.json", population)
    packed = np.asarray(fit+validation+users, dtype=np.uint64)
    np.savez(inputs/"histories.npz", uids=packed, offsets=np.arange(len(packed)+1),
        timestamps=np.full(len(packed), 231*86400-1), item_ids=np.ones(len(packed)), behaviors=np.ones(len(packed)))
    events = [dict(uid=uid, query_timestamp=231*86400+i, raw_item_id=i+1, item_idx=i+1, label=i)
        for uid in users for i in (0, 1)]
    pq.write_table(pa.Table.from_pylist(events), inputs/"requests.parquet")
    cfg = dict(edge=1, release_day=231, end_day=245, history_length=1024,
        profile_users=64, evaluation_checkpoint_users=256, confidence_replicates=1000)
    binding = dict(settings=cfg, population_sha256=sha256(inputs/"population.json"),
        history=dict(path="histories.npz", sha256=sha256(inputs/"histories.npz")),
        requests=dict(path="requests.parquet", sha256=sha256(inputs/"requests.parquet")),
        candidate_panels=dict(long_feedback_active=users))
    write_json(inputs/"inputs.json", binding)
    strata = root/"strata.json"
    write_json(strata, dict(subgroups_by_edge={"1": dict(long_feedback_active=dict(first_2000_uid_hashes=users))}))
    ref = lambda path: dict(path=str(path.resolve()), sha256=sha256(path))
    manifest = dict(schema_version=1, edge=1, uids=users, roles=dict(quality=users), qualification=None,
        input_binding=ref(inputs/"inputs.json"), history=ref(inputs/"histories.npz"),
        requests=ref(inputs/"requests.parquet"), source_population=ref(inputs/"population.json"),
        selection=dict(panel_name="long_feedback_active", definition=analysis.natural_plan.DEFINITION,
            strata_manifest_path=str(strata), strata_manifest_sha256=sha256(strata)))
    candidate = root/"candidate_manifest.json"
    write_json(candidate, manifest)
    _, _, _, original = load_scenario_population(candidate, 1, minimum_users=2000,
        require_qualification=False, require_positive=False)
    model = root/"model.bin"
    model.write_bytes(b"synthetic known identity, not a foundation model")
    models = {str(model): sha256(model)}
    source = dict(qualify_scenario.sources())
    for name in ("scenario_inputs.py", "qualify_natural_scenario.py"):
        path = ROOT/"experiments/ch3_motivation/code"/name
        source[str(path.relative_to(ROOT))] = sha256(path)
    anchor_identity = dict(experiment="fixed_natural_three_anchors", edge=1, models=models,
        input_binding_sha256=sha256(inputs/"inputs.json"), settings=cfg, sources=source, scenario=original)
    labels = np.asarray([row["label"] for row in events])
    noise = np.random.default_rng(72).normal(size=len(events))
    scores = dict(reuse=noise, rebuild=noise+.4*labels, old_full=noise+.05*labels)
    anchor_rows = [dict(row, method=method, score=float(value))
        for method, values in scores.items() for row, value in zip(events, values, strict=True)]
    for branch, methods in (("old_full", {"old_full"}), ("current", {"reuse", "rebuild"})):
        fast_blocks(anchors/branch/"quality_shards", users, dict(anchor_identity, branch=branch),
            [row for row in anchor_rows if row["method"] in methods])
    anchor_summary = qualify_scenario.bootstrap_anchors(anchor_rows, users, replicates=1000, seed=28122)
    anchor_summary.update(status="complete", edge=1, panel="long_feedback_active", identity=anchor_identity,
        evaluation_window={"name": "E14"})
    write_json(anchors/"identity.json", anchor_identity)
    write_json(anchors/"summary.json", anchor_summary)
    manifest["qualification"] = dict(summary_path=str(anchors/"summary.json"),
        summary_sha256=sha256(anchors/"summary.json"), scope="synthetic score oracle",
        **{key: anchor_summary[key] for key in ("new_rebuild_minus_old_full", "new_rebuild_minus_reuse", "point_qualifies", "uid95_qualifies")})
    write_json(anchors/"scenario_manifest.json", manifest)
    _, _, _, provenance = load_scenario_population(anchors/"scenario_manifest.json", 1,
        minimum_users=2000, require_positive=False)

    policies = [dict(tag=method, method=method, calibration_flops=0) for method in ("reuse", "rebuild")]
    policies += [dict(tag=f"tail_{i}", method="tail", fraction=.1, calibration_flops=0) for i in range(22)]
    profile = dict(fitting_uids=fit[:64], selected={}, standalone_calibration_flops_by_width={})
    write_json(e3/"layer_profile.json", profile)
    e3_identity = dict(settings=cfg, checkpoint_sha256=models,
        input_binding_sha256=provenance["input_binding_sha256"], source_sha256=analysis.run_selective_population.sources(),
        evaluation_users_sha256=hashlib.sha256(json.dumps(users, sort_keys=True).encode()).hexdigest(),
        calibration_users_sha256=hashlib.sha256(json.dumps(fit[:64], sort_keys=True).encode()).hexdigest())
    write_json(e3/"identity.json", e3_identity)
    write_json(e3/"status.json", dict(state="complete", mode="run"))
    write_json(e3/"policies.json", dict(policies=policies))
    e3_values = {policy["tag"]: scores.get(policy["tag"], noise+.1*labels) for policy in policies}
    e3_rows = [dict(row, method=method, score=float(value))
        for method, values in e3_values.items() for row, value in zip(events, values, strict=True)]
    fast_blocks(e3/"quality_shards", users,
        dict(e3_identity, layer_profile_sha256=sha256(e3/"layer_profile.json")), e3_rows)
    costs = {}
    for account, denominator in (("actual_event_ledger", 200000), ("candidate_reads_230", 400000)):
        costs[account] = dict(users=2000, once_rebuild_flops=denominator, results=[])
        for policy in policies:
            added = 0 if policy["tag"] == "reuse" else denominator if policy["tag"] == "rebuild" else 40000
            cost = dict(method=policy["tag"], added_flops=added, percent_of_once_rebuild=100*added/denominator)
            if account == "actual_event_ledger":
                cost.update(calibration_flops=0, selection_and_repair_flops=added)
            costs[account]["results"].append(cost)
    write_json(e3/"costs.json", costs)
    groups = analysis.selective_analysis._groups(e3_rows)
    intervals, confidence = paired_user_bootstrap(groups, users, replicates=1000, seed=17022)
    points = [dict(setting={key: policy[key] for key in ("fraction",) if key in policy},
        auc=float(roc_auc_score(labels, e3_values[policy["tag"]])),
        **costs["actual_event_ledger"]["results"][index],
        normalized_230_percent_of_once_rebuild=costs["candidate_reads_230"]["results"][index]["percent_of_once_rebuild"],
        **intervals[policy["tag"]]) for index, policy in enumerate(policies)]
    write_json(e3/"summary.json", dict(identity=e3_identity, results=points, confidence=confidence,
        quality_users=2000, users_with_feedback=2000, feedback_events=len(events),
        reference_gap=anchor_summary["new_rebuild_minus_reuse"], evaluation_window={"name": "E14"}, costs=costs))

    jobs, weights, paths = [], [], []
    execution_sources = {"experiments/ch3_motivation/code/evaluate_probes.py": sha256(ROOT/"experiments/ch3_motivation/code/evaluate_probes.py")}
    ladders = dict(f=noise+.1*labels, fg=noise+.1*labels+3., fgh=noise+.3*labels+3.)
    first_identity = None
    for index, method in enumerate(analysis.METHODS):
        checkpoint = root/f"{method}.pt"
        raw = 100000+90000*index
        payload = dict(format="ch3_probe_variant_v1", variant="full_item", stage=method,
            config=dict(num_layers=2, num_heads=3, head_dim=4, item_dim=8, normalize=True,
                include_item=True, hidden_width=128, depth=1),
            metadata=dict(stage=method, method_label=method, completed_steps=2,
                cumulative_calibration_flops_estimate=raw, total_calibration_flops_estimate=90000,
                operation_counts=dict(training_probe_rows=7, validation_probe_rows=11,
                    training_sample_calls=2, training_backward_calls=2),
                identity=dict(edge=1, seed=17, numerical_sources={correction.CORE_SOURCE: correction.AFFECTED_CORE_SHA256,
                    correction.VARIANT_SOURCE: correction.AFFECTED_VARIANT_SHA256})))
        torch.save(payload, checkpoint)
        job = dict(name=f"natural_edge_1_seed_17_{method}", edge=1, panel="natural_active", backend="projected",
            checkpoint=str(checkpoint), population_file=str(anchors/"scenario_manifest.json"))
        jobs.append(job)
        weights.append(dict(checkpoint=str(checkpoint), method_label=method, checkpoint_sha256=sha256(checkpoint)))
        folder = e5/job["name"]
        identity = dict(edge=1, panel="natural_active", users=users, models=models, backend="projected",
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), sources=execution_sources,
            input_binding_sha256=provenance["input_binding_sha256"], scenario=provenance)
        write_json(folder/"configuration.json", identity)
        first_identity = identity if first_identity is None else first_identity
        for offset in range(0, len(users), 256):
            block = users[offset:offset+256]
            selected = set(block)
            rows = [dict(row, method=method, score=float(value))
                for row, value in zip(events, ladders[method], strict=True) if row["uid"] in selected]
            path = folder/"probe_blocks"/f"part_{offset:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
            write_json(path.with_suffix(".json"), dict(users=block, fingerprint=digest(identity), sha256=sha256(path)))
        account = dict(calibration_flops=raw, release_preparation_flops=10000, extra_read_flops=15000,
            extra_native_writer_flops=4000, shared_graph_setup_flops=1000,
            complete_added_flops=raw+30000, once_rebuild_flops=200000,
            percent_of_once_rebuild=100*(raw+30000)/200000)
        account["candidate_reads_230"] = dict(users=2000, calibration_flops=raw,
            release_preparation_flops=10000, extra_candidate_read_flops=18000,
            actual_behavior_read_extra_flops=7000, extra_native_writer_flops=4000, shared_graph_setup_flops=1000,
            complete_added_flops=raw+40000, once_rebuild_flops=400000,
            percent_of_once_rebuild=100*(raw+40000)/400000)
        anchor_folder = e5/jobs[0]["name"]/"anchors"
        key = digest({name: first_identity[name] for name in ("edge", "models", "users", "input_binding_sha256", "sources")})
        stage_rows = [dict(row, method=method, score=float(value)) for row, value in zip(events, ladders[method], strict=True)]
        stage_groups = analysis.selective_analysis._groups(stage_rows+[row for row in anchor_rows if row["method"] in ("reuse", "rebuild")])
        stage_intervals, stage_confidence = paired_user_bootstrap(stage_groups, users, replicates=1000, seed=17022)
        write_json(folder/"summary.json", dict(status="complete", experiment="E5", edge=1, seed=17,
            variant="full_item", width=128, depth=1, fit_users=256, quality_users=2000,
            feedback_events=len(events), method_label=method, stage=method, identity=identity,
            auc=float(roc_auc_score(labels, ladders[method])), reuse_auc=anchor_summary["auc"]["reuse"],
            rebuild_auc=anchor_summary["auc"]["rebuild"], costs=account,
            intervals=stage_intervals, confidence=stage_confidence,
            anchor_source=dict(path=str(anchor_folder), key=key)))
        paths.append(folder/"summary.json")
    fast_blocks(e5/jobs[0]["name"]/"anchors/quality_shards", users, first_identity,
        [row for row in anchor_rows if row["method"] in ("reuse", "rebuild")])
    plan = dict(edges={"1": dict(weights=weights)}, source_sha256=execution_sources)
    return dict(root=root, anchors=anchors, e3=e3, e5=e5, jobs=jobs, plan=plan, paths=paths)


def test_real_natural_anchor_layout_and_e3_full_points(committed):
    hashes = {}
    context = analysis._anchors(committed["anchors"], 1, hashes)
    result = analysis._e3(committed["e3"], committed["anchors"]/"scenario_manifest.json", 1, context, hashes)
    assert result["status"] == "complete" and result["points"] == 24
    assert result["audit"]["block_audit"]["blocks"] == 8
    assert result["audit"]["fresh_anchor_comparison"]["maximum_absolute_difference"] == 0
    assert set(result["costs"]) == {"actual_event_ledger", "candidate_reads_230"}


def test_seed17_direct_g_h_and_raw_corrected_accounts(committed):
    hashes = {}
    context = analysis._anchors(committed["anchors"], 1, hashes)
    result = analysis._e5(committed["jobs"], committed["plan"], committed["e5"], 1, context, hashes, {"status": "pending"})
    assert result["status"] == "complete" and result["expected_seeds"] == [17]
    assert result["three_seed_claim"] is False
    assert result["increments"]["g"]["auc_difference_ci95"] == [0., 0.]
    assert result["increments"]["h"]["auc_difference_ci95"][0] > 0
    for method, accounts in result["costs"].items():
        deduction = 6144 if method == "fgh" else 0
        for raw, fixed in ((accounts["raw"], accounts["corrected"]),
                (accounts["raw"]["candidate_reads_230"], accounts["corrected"]["candidate_reads_230"])):
            assert fixed["complete_added_flops"] == raw["complete_added_flops"]-deduction
            assert fixed["once_rebuild_flops"] == raw["once_rebuild_flops"]
    tables = analysis._tables({"1": dict(evaluation_window={"name": "E14"}, e3={"status": "pending"}, e5=result)})
    assert len(tables["e5_points"]) == 3 and len(tables["e5_increments"]) == 2
    assert len(tables["e5_costs"]) == 12
    assert {row["version"] for row in tables["e5_costs"]} == {"raw", "corrected"}


def test_missing_natural_inputs_produce_pending_and_no_csv(tmp_path):
    config = tmp_path/"config.json"
    write_json(config, dict(cohort="long_feedback_active", users_per_edge=2000,
        definition=analysis.natural_plan.DEFINITION, all_edges_retained=True, require_positive=False))
    plan = tmp_path/"plan.json"
    write_json(plan, dict(status="waiting_selection", jobs=[]))
    result = analysis.analyze(tmp_path/"output", comparison_config=config, e5_plan=plan,
        anchors_root=tmp_path/"missing_anchors", e3_root=tmp_path/"e3", e5_root=tmp_path/"e5")
    assert result["status"] == "pending" and result["e5_plan_status"] == "waiting_selection"
    assert len(result["edges"]) == 5 and not result["cross_edge_aggregate_computed"]
    assert not list((tmp_path/"output").glob("*.csv"))


def test_invalid_zero_job_ready_plan_is_rejected(tmp_path):
    config = tmp_path/"config.json"
    write_json(config, dict(cohort="long_feedback_active", users_per_edge=2000,
        definition=analysis.natural_plan.DEFINITION, all_edges_retained=True, require_positive=False))
    plan = tmp_path/"plan.json"
    write_json(plan, dict(status="prepared_for_root_review", jobs=[]))
    with pytest.raises(ValueError, match="exactly15"):
        analysis.analyze(tmp_path/"output", comparison_config=config, e5_plan=plan)


def test_cost_arithmetic_cannot_be_silently_corrected(committed):
    saved = json.loads(committed["paths"][0].read_text())
    costs = saved["costs"]
    analysis._check_costs(costs)
    costs["candidate_reads_230"]["complete_added_flops"] += 1
    with pytest.raises(ValueError, match="cost arithmetic"):
        analysis._check_costs(costs)


@pytest.mark.parametrize("fault", ["missing_entry", "stale_entry", "stale_other_source"])
def test_stale_execution_sources_are_rejected_before_pairing(committed, monkeypatch, fault):
    original = analysis._json
    first = committed["paths"][0]
    def stale(path, hashes):
        value = original(path, hashes)
        if Path(path) == first:
            value = copy.deepcopy(value)
            sources = value["identity"]["sources"]
            entry = "experiments/ch3_motivation/code/evaluate_probes.py"
            if fault == "missing_entry":
                sources.clear()
            elif fault == "stale_entry":
                sources[entry] = "0"*64
            else:
                sources["experiments/ch3_motivation/code/probe_cost.py"] = "0"*64
        return value
    monkeypatch.setattr(analysis, "_json", stale)
    with pytest.raises(ValueError, match="execution source"):
        analysis._e5(committed["jobs"], committed["plan"], committed["e5"], 1, {}, {}, {"status": "pending"})


def test_fewer_than_2000_or_changed_natural_definition_is_rejected(committed, monkeypatch):
    original = analysis.load_scenario_population
    def fewer(*args, **kwargs):
        binding, users, population, provenance = original(*args, **kwargs)
        return binding, users[:511], population, provenance
    monkeypatch.setattr(analysis, "load_scenario_population", fewer)
    with pytest.raises(ValueError, match="fixed active2000"):
        analysis._anchors(committed["anchors"], 1, {})


def test_missing_or_changed_committed_scores_are_rejected(committed, monkeypatch):
    original = analysis.audit_blocks
    def changed(*args, **kwargs):
        rows, report = original(*args, **kwargs)
        if "current" in str(args[0]):
            rows[0] = dict(rows[0], score=rows[0]["score"]+100)
        return rows, report
    monkeypatch.setattr(analysis, "audit_blocks", changed)
    with pytest.raises(ValueError, match="summary differs"):
        analysis._anchors(committed["anchors"], 1, {})


def test_driver_assembles_complete_e3_with_distinct_edge_uids_but_pending_e5(tmp_path, monkeypatch):
    config, plan = tmp_path/"config.json", tmp_path/"plan.json"
    write_json(config, dict(cohort="long_feedback_active", users_per_edge=2000,
        definition=analysis.natural_plan.DEFINITION, all_edges_retained=True, require_positive=False))
    write_json(plan, dict(status="waiting_selection", jobs=[]))
    for edge in analysis.EDGES:
        for name in ("scenario_manifest.json", "summary.json", "identity.json"):
            write_json(tmp_path/f"anchors/edge_{edge}"/name, {})
        for name in ("summary.json", "identity.json", "status.json", "policies.json", "layer_profile.json", "costs.json"):
            write_json(tmp_path/f"e3/edge_{edge}"/name, {})
    cohorts = []
    def anchors(folder, edge, hashes):
        users = list(range(edge*10000, edge*10000+2000))
        cohorts.append(users)
        window = {"name": "E14_partial" if edge == 5 else "E14"}
        return dict(binding=dict(settings=dict(evaluation_window=window)),
            summary=dict(quality_uids_sha256=digest(users)), audit={}, users=users)
    def selective(folder, manifest, edge, context, hashes):
        points = [dict(method=f"synthetic_{i}", setting={}, auc=.5) for i in range(24)]
        costs = {name: dict(once_rebuild_flops=100, results=[dict(method=f"synthetic_{i}", added_flops=0)
            for i in range(24)]) for name in ("actual_event_ledger", "candidate_reads_230")}
        return dict(status="complete", points=24, results=points, costs=costs)
    monkeypatch.setattr(analysis, "_anchors", anchors)
    monkeypatch.setattr(analysis, "_e3", selective)
    result = analysis.analyze(tmp_path/"output", comparison_config=config, e5_plan=plan,
        anchors_root=tmp_path/"anchors", e3_root=tmp_path/"e3", e5_root=tmp_path/"e5")
    assert len(set.union(*(set(users) for users in cohorts))) == 10000
    assert result["status"] == "pending" and result["families"]["e3"]["status"] == "complete"
    import csv
    with (tmp_path/"output/e3_points.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 120
    assert all(row["evaluation_window"] == "E14_partial" for row in rows if row["edge"] == "5")
    assert not (tmp_path/"output/e5_points.csv").exists()
