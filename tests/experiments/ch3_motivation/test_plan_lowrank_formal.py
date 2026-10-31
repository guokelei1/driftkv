"""CPU handoff checks bind real typed FG/H, causal packs and all fixed roles."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code import (
    plan_lowrank_formal as planner, plan_probe_calibration_formal as fitting,
    plan_probe_quality as quality,
)
from experiments.ch3_motivation.code.probe_variants import build_variant, save_variant
from experiments.ch3_motivation.code.select_probe_common import sha256, write_json
from hstu_kvcache.models import HSTU, HSTUConfig
from tests.experiments.ch3_motivation.test_plan_probe_calibration_formal import assets, selection


@pytest.fixture(scope="module")
def world(tmp_path_factory, assets):
    root = tmp_path_factory.mktemp("e6_handoff")
    input_root = root/"fit256"
    population = assets.population
    write_json(input_root/"population.json", population)
    manifests = {}
    all_users = population["fit"]+population["validation"]+population["main"]
    for edge in planner.EDGES:
        folder = input_root/f"edge_{edge}"
        folder.mkdir()
        cutoff = (20+(edge-1)*14)*86400
        np.savez(folder/"histories.npz", uids=np.array(all_users), offsets=np.arange(len(all_users)+1)*3,
            timestamps=np.tile([cutoff-20, cutoff-10, cutoff+2], len(all_users)),
            item_ids=np.tile([1, 2, 3], len(all_users)), behaviors=np.zeros(3*len(all_users), dtype=np.int64))
        pq.write_table(pa.Table.from_pylist([dict(uid=uid, query_timestamp=cutoff+5, item_idx=1,
            raw_item_id=1, label=i%2) for i, uid in enumerate(population["main"])]), folder/"requests.parquet")
        binding = dict(settings=dict(edge=edge, release_day=cutoff//86400, end_day=cutoff//86400+14,
            history_length=1024), population_sha256=sha256(input_root/"population.json"),
            history=dict(path="histories.npz", sha256=sha256(folder/"histories.npz")),
            requests=dict(path="requests.parquet", sha256=sha256(folder/"requests.parquet")))
        write_json(folder/"inputs.json", binding)
        candidate = json.loads((assets.root/f"edge_{edge}/calibration_feedback_candidates.json").read_text())
        write_json(folder/"calibration_feedback_candidates.json", candidate)
        manifest = json.loads(assets.manifests[edge].read_text())
        manifest["sourcebinding"] = dict(input_root=str(input_root), population_sha256=sha256(input_root/"population.json"),
            sha256=sha256(folder/"inputs.json"), candidate_sha256=sha256(folder/"calibration_feedback_candidates.json"),
            history_path=str(folder/"histories.npz"), history_sha256=sha256(folder/"histories.npz"))
        path = root/f"sampling_edge_{edge}.json"
        write_json(path, manifest)
        manifests[edge] = path
    setup = SimpleNamespace(root=input_root, population=population, manifests=manifests, scope=assets.scope)
    selected = selection(root/"selected", setup, rule="gap_mix50", n=128)
    formal = fitting.generate(selected.path, assets.scope)
    formal_path = root/"formal.json"
    write_json(formal_path, formal)
    model = HSTU(HSTUConfig(num_items=64, num_behaviors=2, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.))
    fg = build_variant("full_item", model, stage="fg", hidden_width=128, depth=1)
    full = fg.expanded("fgh")  # h=0 is a permitted result, never screened out.
    main_root = root/"main"
    scenarios = root/"scenarios"
    for job in formal["main_jobs"]:
        edge, seed = job["edge"], job["seed"]
        source = json.loads(selected.path.read_text())["recommendation"]["checkpoints_by_edge"][str(edge)]["checkpoint"]
        import torch
        metadata = torch.load(source, map_location="cpu", weights_only=False)["metadata"]
        metadata["identity"].update(seed=seed, controls=True)
        metadata.update(seed=seed, cumulative_calibration_flops_estimate=1234.)
        for method, probe in (("fg", fg), ("fgh", full)):
            actual = copy.deepcopy(metadata)
            actual["method_label"] = method
            save_variant(quality.checkpoint_path(main_root, job, method), probe, actual)
    for edge in planner.EDGES:
        folder = input_root/f"edge_{edge}"
        target = scenarios/f"edge_{edge}/feedback_union_2000"
        models = {f"version{edge-1}": f"model{edge-1}", f"version{edge}": f"model{edge}"}
        gain = -.01 if edge == 4 else .01
        qualifying = gain > 0
        qualification = dict(new_rebuild_minus_old_full=gain, new_rebuild_minus_reuse=.02,
            point_qualifies=qualifying, uid95_qualifies=qualifying)
        summary = dict(quality_uids=population["main"], quality_users=2000,
            quality_uids_sha256=planner.uid_hash(population["main"]), qualification=qualification, **qualification,
            auc=dict(old_full=.6, rebuild=.6+gain, reuse=.58+gain),
            new_rebuild_minus_old_full_ci95=[gain-.001, gain+.001], new_rebuild_minus_reuse_ci95=[.01, .03],
            identity=dict(input_binding_sha256=sha256(folder/"inputs.json"), models=models))
        write_json(target/"summary.json", summary)
        manifest = dict(schema_version=1, edge=edge, uids=population["main"],
            roles=dict(quality=population["main"], e6=population["main"]),
            qualification=dict(qualification, summary_path=str(target/"summary.json"),
                summary_sha256=sha256(target/"summary.json"), scope="fixed cohort; negative edge retained"),
            selection=dict(panel_name="feedback_union_2000"),
            input_binding=dict(path=str(folder/"inputs.json"), sha256=sha256(folder/"inputs.json")),
            history=dict(path=str(folder/"histories.npz"), sha256=sha256(folder/"histories.npz")),
            requests=dict(path=str(folder/"requests.parquet"), sha256=sha256(folder/"requests.parquet")),
            source_population=dict(path=str(input_root/"population.json"), sha256=sha256(input_root/"population.json")))
        write_json(target/"scenario_manifest.json", manifest)
    return SimpleNamespace(root=root, formal=formal, formal_path=formal_path, main_root=main_root,
        scenario_root=scenarios, scope=assets.scope, output_root=root/"results")


def generate(world, **kwargs):
    return planner.generate(world.formal_path, world.main_root, scope_path=world.scope,
        scenario_root=world.scenario_root, output_root=world.output_root, **kwargs)


def test_pending_selection_and_missing_weights_never_emit_fake_commands(tmp_path, world):
    pending = tmp_path/"pending.json"
    write_json(pending, dict(status="waiting_selection", main_jobs=[]))
    value = planner.generate(pending, world.main_root, output_root=world.output_root, scope_path=world.scope)
    assert value["status"] == "waiting_selection" and not value["jobs"] and not value["readiness_jobs"]
    value = planner.generate(world.formal_path, tmp_path/"not_fitted", output_root=world.output_root,
        scope_path=world.scope, scenario_root=world.scenario_root)
    assert value["status"] == "waiting_weights_or_population" and len(value["missing_dependencies"]) == 30
    assert not value["jobs"] and not value["weight_bindings"] and not value["gpu_launch_performed"]


def test_exact19_commands_real_pairs_adverse_edge_and_shared_four_variants(world):
    value = generate(world)
    assert value["status"] == "prepared_for_root_review"
    assert len(value["jobs"]) == 19 and len(value["weight_bindings"]) == 15
    assert len(value["readiness_jobs"]) == 2 and not value["ready_to_launch_quality"]
    assert not value["complete_serving_cost_ready"]
    assert value["weight_bindings"]["edge_4_seed_17"]["qualification"]["new_rebuild_minus_old_full"] < 0
    assert all(binding["h_best_step"] == 0 for binding in value["weight_bindings"].values())
    for job in value["jobs"]:
        assert job["ranks"] == [0, 1, 2, 4, 6, 8, 16]
        assert job["bootstrap"]["replicates"] == 1000
        assert "--bootstrap" not in job["argv"] and "--seed" not in job["argv"]
        assert job["quality_auc_permitted"]
        if job["id"].startswith("main_"):
            assert job["users"] == 2000
            assert job["variants"] == (list(planner.ALL_VARIANTS) if job["seed"] == 17 else ["both_norm"])
        else:
            assert job["edge"] in (2, 5) and job["seed"] == 17 and job["users"] == 512
            assert job["variants"] == ["both_norm"]
    for job in value["readiness_jobs"]:
        assert job["users"] == 8 and not job["quality_auc_permitted"]
        assert job["variants"] == list(planner.ALL_VARIANTS)
    assert value["job_counts"]["total_quality_commands"] == 19
    assert sum(len(order) for order in value["scheduling"]["gpu_orders"].values()) == 19
    planner._check_files(value["frozen_sha256"])


@pytest.mark.parametrize("fault", ["missing_cell", "different_recipe", "escape_root"])
def test_formal_grid_is_not_silently_repaired(tmp_path, world, fault):
    formal = copy.deepcopy(world.formal)
    if fault == "missing_cell":
        formal["main_jobs"].pop()
    elif fault == "different_recipe":
        formal["main_jobs"][1]["learning_rate"] *= 2
    else:
        formal["main_jobs"][0]["output"] = str(tmp_path/"other")
    path = tmp_path/"formal.json"
    write_json(path, formal)
    with pytest.raises(ValueError, match="all15|one frozen|escapes"):
        planner.generate(path, world.main_root, output_root=world.output_root,
            scope_path=world.scope, scenario_root=world.scenario_root)


def test_lpt_balances_work_instead_of_edge_parity():
    jobs = [dict(id=str(i), workload_proxy=float(score), argv=["python"]) for i, score in enumerate([9, 8, 7, 6, 5, 4])]
    value = planner.assign_gpus(jobs)
    loads = list(value["gpu_proxy_totals"].values())
    assert max(loads)-min(loads) <= 1
    assert all(job["argv"][-2:] == ["--gpu", str(job["gpu"])] for job in jobs)
    assert not value["proxy_is_measured_runtime"]


def test_alias_snapshot_cannot_override_current_source_sha(tmp_path):
    source = tmp_path/"source.py"
    source.write_text("current numerical source")
    digest = sha256(source)
    alias = tmp_path/"not_required_to_exist/../source.py"
    assert planner.merge_frozen({str(source): digest}, {str(alias): digest}) == {str(source.resolve()): digest}
    with pytest.raises(ValueError, match="conflicting E6 frozen snapshots"):
        planner.merge_frozen({str(source): digest}, {str(alias): "0"*64})
    repository_source = ROOT/"experiments/ch3_motivation/code/common.py"
    value = planner.merge_frozen({str(repository_source.relative_to(ROOT)): sha256(repository_source)})
    assert list(value) == [str(repository_source.resolve())]


def test_main_weights_cannot_substitute_another_backbone(tmp_path, world):
    import torch
    main = tmp_path/"main"
    for job in world.formal["main_jobs"]:
        for method in ("fg", "fgh"):
            original = quality.checkpoint_path(world.main_root, job, method)
            target = quality.checkpoint_path(main, job, method)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(original)
    target = quality.checkpoint_path(main, world.formal["main_jobs"][0], "fgh")
    payload = torch.load(target, map_location="cpu", weights_only=False)
    target.unlink()
    payload["metadata"]["identity"]["models"]["version0"] = "different-old-model"
    torch.save(payload, target)
    with pytest.raises(ValueError, match="different edge/backbone"):
        planner.generate(world.formal_path, main, scope_path=world.scope,
            scenario_root=world.scenario_root, output_root=world.output_root)


def test_workload_counts_reads_strictly_before_real_context():
    histories = SimpleNamespace(rows={1: (np.array([9, 10, 11, 20]), None, None)})
    rows = [dict(uid=1, query_timestamp=10, item_idx=1), dict(uid=1, query_timestamp=20, item_idx=1),
        dict(uid=1, query_timestamp=20, item_idx=2)]
    value = planner.workload_counts(histories, rows, [1], dict(release_day=0, end_day=1, history_length=2))
    assert value["feedback_instances"] == 3 and value["timestamps"] == 2
    assert value["initial_rows"] == 0 and value["behavior_appends_before_last_read"] == 3
    assert value["timestamp_matrix_row_candidate_work"] == (1+2)*230


@pytest.mark.parametrize("protocol", ["release", "timestamp"])
def test_readiness_rejects_old_pair_or_mislabelled_quality(tmp_path, protocol):
    from experiments.ch3_motivation.code.run_lowrank_timestamp_guarded import sources as timestamp_sources
    recorded_sources = planner.release_sources() if protocol == "release" else timestamp_sources()
    binding = dict(uids=list(range(8)), population_sha256="population", models={"v0": "old", "v1": "new"},
        full_h_checkpoint_sha256="selected_h", trained_fg_checkpoint_sha256="selected_fg")
    job = dict(edge=2, protocol=protocol)
    identity = dict(edge=2, users=list(range(8)), variants=list(planner.ALL_VARIANTS), ranks=list(planner.RANKS),
        models=binding["models"], population=dict(manifest_sha256="population"),
        full_h_checkpoint_sha256="selected_h", trained_fg_control=dict(checkpoint_sha256="selected_fg"),
        decomposition=dict(method="randomized", oversample=16, power_iterations=2, batch_heads=25),
        candidates=230, execution=dict(candidate_chunk=230, row_chunk=1024, mode="projected", batch_rank_scores=True),
        sources=recorded_sources)
    result = dict(status="complete", users=8, quality_auc_computed=False, full_rank_max_error=1e-7,
        energy_guard=dict(matrices=32, expected_matrices=32, all_matrix_scope=True,
            passed_without_exact=31, exact_fallback_matrices=1),
        rank_execution_comparison=dict(checks=4, logit_max_error=1e-7), peak_reserved_bytes=20*1024**3, seconds=80.)
    path = tmp_path/"diagnostic.json"
    result["identity_hash"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    write_json(path, result)
    write_json(tmp_path/"configuration.json", identity)
    frozen = {str((ROOT/name).resolve()): digest for name, digest in recorded_sources.items()}
    assert planner.audit_readiness(path, job, binding, frozen)["status"] == "passed"
    identity["full_h_checkpoint_sha256"] = "old_h"
    write_json(tmp_path/"configuration.json", identity)
    with pytest.raises(ValueError, match="another FG/H"):
        planner.audit_readiness(path, job, binding, frozen)
    identity["full_h_checkpoint_sha256"] = "selected_h"
    write_json(tmp_path/"configuration.json", identity)
    result["quality_auc_computed"] = True
    write_json(path, result)
    with pytest.raises(ValueError, match="reports quality"):
        planner.audit_readiness(path, job, binding, frozen)
