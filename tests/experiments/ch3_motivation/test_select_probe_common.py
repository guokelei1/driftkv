"""CPU-only fixtures for fixed-problem, five-edge recipe selection."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import select_probe_common as analysis


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def metrics(value, uids):
    row = dict(loss=value, read_mse=value * .6, score_mse=value * .4, raw_score_mse=value * .00004)
    return dict(users=16, **row, per_user=[dict(uid=uid, queries=16, history_length=100 + uid % 2, **row)
                                         for uid in uids])


def fixture(tmp_path):
    plans = tmp_path / "plans"
    summaries = {}
    fit_uids, validation_uids = list(range(1000, 1064)), list(range(2000, 2016))
    for edge in analysis.EDGES:
        candidates = {str(uid): list(range(1 + edge, 17 + edge)) for uid in fit_uids + validation_uids}
        common = dict(format=analysis.FORMAT, edge=edge,
            population_sha256="same-frozen-population", scale_fit_uids=fit_uids, validation_uids=validation_uids,
            fitting_candidate_pool_uids=list(range(1000, 1256)),
            models={f"v{edge - 1}": f"sha-v{edge - 1}", f"v{edge}": f"sha-v{edge}"},
            queries=16, recent_fraction=.5, candidate_seed=17, backend="torch",
            candidate_ids=candidates, candidate_ids_sha256=analysis.fingerprint(candidates),
            candidate_file_sha256=f"different-candidate-file-edge-{edge}",
            numerical_sources={"validator": "same-frozen-validator"},
            objective=dict(read_weight=1., score_weight=1.),
            loss_scales=dict(users=64, source="fitting_users_only", validation_uses_fitting_scales=True,
                version="fit_pool_equal_user_length_normalized_v1",
                read_scales=[[edge / 10 + 1]], score_scale=edge / 100 + .001))
        token = analysis.fingerprint(common)
        summaries[edge] = dict(format=analysis.FORMAT, status="complete", common_validation=common,
            common_validation_sha256=token, quality_auc_computed=False, fitting_performed=False,
            post_release_labels_loaded=False,
            native_reuse=dict(common_validation_sha256=token, metrics=metrics(3., validation_uids)), points=[])
    # Same source name can appear in another sweep. Candidate SHA changes every
    # edge and must never split one recipe's five-edge mean.
    recipes = [
        ("sweep_a", "minimum", "full_item", 256, 256, 1., 500, 17),
        ("sweep_a", "prefer_fit", "full_item", 64, 128, 1.008, 500, 17),
        ("sweep_a", "prefer_width", "full_item", 64, 64, 1.009, 1000, 17),
        ("sweep_a", "winner", "full_item", 64, 64, 1.0095, 100, 17),
        ("sweep_a", "outside_fixed_threshold", "full_item", 64, 32, 1.018, 1, 17),
        ("sweep_a", "fails_edge5", "full_item", 64, 16, .9, 1, 17),
        ("sweep_a", "head_smaller_loss", "head", 64, 32, .01, 1, 17),
        ("sweep_a", "layer_smaller_loss", "layerwise_kv", 64, 32, .02, 1, 17),
        ("sweep_b", "winner", "full_item", 64, 64, 2., 1, 17),
        ("sweep_b", "other_seed", "full_item", 64, 64, .001, 1, 23),
    ]
    source_points, source_jobs = {}, {}
    for sweep, name, variant, fit_users, width, value, cost, seed in recipes:
        source_points.setdefault(sweep, [])
        source_jobs.setdefault(sweep, [])
        for edge in analysis.EDGES:
            source_jobs[sweep].append(dict(name=name, edge=edge, seed=seed, variant=variant,
                fit_users=fit_users, width=width, depth=1, stages="f,fg,fgh", controls=True,
                input_root=str(tmp_path / "causal_inputs"), learning_rate=.0001, steps=256,
                read_weight=1., score_weight=1., patience=128, validation_every=32))
            for label, increment in (("f", .2), ("fg", .1), ("fgh", 0.),
                                     ("f_extended", -.8), ("fg_extended", -.7)):
                if name == "winner" and sweep == "sweep_a" and label == "fg":
                    increment = 0.  # The chosen step-zero h is exactly its FG control.
                checkpoint = tmp_path / sweep / "attempt_1" / name / f"edge_{edge}" / f"seed_{seed}" / f"{label}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(f"fixture {sweep}/{name}/{edge}/{seed}/{label}\n")
                step = 0 if name == "winner" and sweep == "sweep_a" and label == "fgh" else 32
                entry = dict(name=name, edge=edge, seed=seed, variant=variant, width=width, depth=1,
                    fit_users=fit_users, method_label=label, best_step=step, completed_steps=256,
                    calibration_flops_estimate=cost * edge,
                    actual_search_preparation_reused=edge % 2 == 0,
                    checkpoint=str(checkpoint), checkpoint_sha256=analysis.sha256(checkpoint))
                source_points[sweep].append(entry)
                point = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=entry["checkpoint_sha256"],
                    common_validation_sha256=summaries[edge]["common_validation_sha256"],
                    status="validated", variant=variant, method_label=label,
                    stage=label.replace("_extended", ""), seed=seed,
                    fitting_users=fit_users, config=dict(hidden_width=width, depth=1),
                    best_step=step, step_zero=step == 0, selection_eligible=True,
                    training_identity=dict(learning_rate=.0001, steps=256, read_weight=1., score_weight=1.,
                        patience=128, validation_every=32, candidate_source="feedback", recent_fraction=.5,
                        queries=16, candidate_seed=17, stages="f,fg,fgh", controls=True,
                        calibration_candidate_sha256=f"edge-specific-training-candidates-{edge}"),
                    metrics=metrics(max(.0001, value + increment), validation_uids))
                if name == "fails_edge5" and edge == 5:
                    point.update(status="failed", error="nonfinite output", selection_eligible=False)
                    point.pop("metrics")
                summaries[edge]["points"].append(point)
    inventory = []
    for sweep, points in source_points.items():
        plan_path = plans / f"{sweep}.json"
        dump(plan_path, dict(jobs=source_jobs[sweep]))
        source_path = tmp_path / sweep / "attempt_1/summary.json"
        dump(source_path, dict(status="complete", identity=dict(plan_sha256=analysis.sha256(plan_path)), points=points))
        for point in points:
            inventory.append(dict(point, source_summary=str(source_path),
                                  source_summary_sha256=analysis.sha256(source_path)))
    inventory_path = tmp_path / "queue/checkpoint_inventory.json"
    dump(inventory_path, inventory)
    summary_paths = []
    for edge, summary in summaries.items():
        failed = sum(point["status"] == "failed" for point in summary["points"])
        summary.update(failed=failed, validated=len(summary["points"]) - failed,
                       status="complete_with_failures" if failed else "complete")
        path = tmp_path / f"common/edge_{edge}/summary.json"
        dump(path, summary)
        summary_paths.append(path)
    return SimpleNamespace(inventory=inventory, inventory_path=inventory_path, plans=plans,
        summaries=summaries, summary_paths=summary_paths)


def analyze(fx, paths=None):
    return analysis.rollup(fx.inventory_path, paths if paths is not None else fx.summary_paths,
                           plan_directory=fx.plans, expected_points=len(fx.inventory))


def test_fixed_one_percent_hierarchy_all_families_steps_and_seeds_retained(tmp_path):
    fx = fixture(tmp_path)
    result = analyze(fx)
    selection = result["selection"]
    assert selection["status"] == "recommendation_for_review"
    chosen = selection["recommendation"]
    assert chosen["recipe"] == "sweep_a/attempt_1/winner"
    assert (chosen["fit_users"], chosen["width"], chosen["seed"]) == (64, 64, 17)
    assert chosen["mean_common_loss"] == pytest.approx(1.0095)
    assert chosen["complete_calibration_flops_estimate_sum"] == 100 * 15
    assert selection["minimum_mean_common_loss"] == pytest.approx(1.)
    assert selection["fixed_one_percent_threshold"] == pytest.approx(1.01)
    assert not any("outside_fixed_threshold" in recipe for recipe in selection["within_one_percent"])
    assert chosen["step_zero_edges"] == [1, 2, 3, 4, 5]
    assert selection["future_correction_seeds"] == [17, 23, 29]
    assert selection["development_seed_robustness_established"] is False
    assert selection["training_authorized"] is False and chosen["backbone_retraining"] is False
    assert len(result["points"]) == len(fx.inventory) == 250
    assert {group["variant"] for group in result["groups"]} == {"head", "full_item", "layerwise_kv"}
    assert {group["method_label"] for group in result["groups"]} == {"f", "fg", "fgh", "f_extended", "fg_extended"}
    assert {group["seed"] for group in result["groups"]} == {17, 23}
    groups = [group for group in result["groups"] if group["name"] == "winner" and group["method_label"] == "fgh"]
    assert len(groups) == 2
    assert {group["recipe"] for group in groups} == {"sweep_a/attempt_1/winner", "sweep_b/attempt_1/winner"}
    failed = next(group for group in result["groups"] if group["name"] == "fails_edge5" and group["method_label"] == "fgh")
    assert failed["mean_common_loss"] is None and failed["failed_edges"] == [5]
    assert not failed["selection_eligible"]
    h = next(row for row in result["stage_comparisons"] if row["recipe"] == chosen["recipe"] and row["after"] == "fgh")
    assert h["mean_common_loss_delta"] == 0.
    assert chosen["settings"]["input_root"].endswith("causal_inputs")
    assert chosen["settings"]["learning_rate"] == .0001


def test_no_selection_from_four_edges_or_an_unfinished_fifth(tmp_path):
    fx = fixture(tmp_path)
    result = analyze(fx, fx.summary_paths[:4])
    assert result["selection"]["status"] == "not_ready"
    assert result["selection"]["recommendation"] is None
    assert result["selection"]["missing_edges"] == [5]
    assert all(group["mean_common_loss"] is None for group in result["groups"])
    fifth = fx.summaries[5]
    fifth.update(status="in_progress", points=fifth["points"][:2])
    dump(fx.summary_paths[4], fifth)
    result = analyze(fx)
    assert result["selection"]["status"] == "not_ready"
    assert result["selection"]["recommendation"] is None
    assert result["selection"]["missing_edges"] == []


@pytest.mark.parametrize("tamper", ["complete_missing_point", "checkpoint_bytes", "loss_mean", "inventory_hash", "source_plan", "duplicate_inventory", "status_count"])
def test_evidence_tampering_rejected(tmp_path, tamper):
    fx = fixture(tmp_path)
    if tamper == "complete_missing_point":
        fx.summaries[1]["points"].pop()
        dump(fx.summary_paths[0], fx.summaries[1])
    elif tamper == "checkpoint_bytes":
        Path(fx.inventory[0]["checkpoint"]).write_text("changed weights")
    elif tamper == "loss_mean":
        fx.summaries[1]["points"][0]["metrics"]["loss"] += 1.
        dump(fx.summary_paths[0], fx.summaries[1])
    elif tamper == "inventory_hash":
        fx.inventory[0]["checkpoint_sha256"] = "changed inventory"
        dump(fx.inventory_path, fx.inventory)
    elif tamper == "source_plan":
        (fx.plans / "sweep_a.json").write_text("changed plan")
    elif tamper == "duplicate_inventory":
        fx.inventory.append(copy.deepcopy(fx.inventory[0]))
        dump(fx.inventory_path, fx.inventory)
    else:
        fx.summaries[1]["failed"] = 1
        dump(fx.summary_paths[0], fx.summaries[1])
    with pytest.raises(ValueError):
        analyze(fx)


def test_different_common_protocol_between_edges_rejected_even_with_valid_hashes(tmp_path):
    fx = fixture(tmp_path)
    fifth = fx.summaries[5]
    fifth["common_validation"]["backend"] = "triton"
    token = analysis.fingerprint(fifth["common_validation"])
    fifth["common_validation_sha256"] = token
    fifth["native_reuse"]["common_validation_sha256"] = token
    for point in fifth["points"]:
        point["common_validation_sha256"] = token
    dump(fx.summary_paths[4], fifth)
    with pytest.raises(ValueError, match="protocol.*across edges"):
        analyze(fx)


def test_same_recipe_cannot_silently_change_optimizer_between_edges(tmp_path):
    fx = fixture(tmp_path)
    point = next(point for point in fx.summaries[5]["points"]
                 if point["method_label"] == "fgh" and "/winner/" in point["checkpoint"] and "/sweep_a/" in point["checkpoint"])
    point["training_identity"]["learning_rate"] = .0003
    dump(fx.summary_paths[4], fx.summaries[5])
    with pytest.raises(ValueError, match="changes its fitting configuration"):
        analyze(fx)


def test_zero_loss_and_no_eligible_primary_have_explicit_rules(tmp_path):
    fx = fixture(tmp_path)
    for edge, summary in fx.summaries.items():
        for point in summary["points"]:
            if "/sweep_a/" in point["checkpoint"] and "/winner/" in point["checkpoint"] and point["method_label"] in ("fg", "fgh"):
                point["metrics"] = metrics(0., summary["common_validation"]["validation_uids"])
        dump(fx.summary_paths[edge - 1], summary)
    result = analyze(fx)
    assert result["selection"]["fixed_one_percent_threshold"] == 0.
    assert result["selection"]["within_one_percent"] == ["sweep_a/attempt_1/winner"]
    assert result["selection"]["recommendation"]["mean_common_loss"] == 0.
    for edge, summary in fx.summaries.items():
        for point in summary["points"]:
            if point["variant"] == "full_item" and point["method_label"] == "fgh":
                point["selection_eligible"] = False
        dump(fx.summary_paths[edge - 1], summary)
    result = analyze(fx)
    assert result["selection"]["all_five_edges_complete"] is True
    assert result["selection"]["status"] == "no_complete_eligible_primary_recipe"
    assert result["selection"]["recommendation"] is None
    assert len(result["points"]) == 250


def test_cli_outputs_groups_all_points_and_future_settings_without_training(tmp_path):
    fx = fixture(tmp_path)
    args = SimpleNamespace(inventory=fx.inventory_path, summary=fx.summary_paths,
        common_root=tmp_path / "common", plan_directory=fx.plans,
        expected_points=len(fx.inventory), output=tmp_path / "analysis")
    analysis.main(args)
    result = json.loads((args.output / "summary.json").read_text())
    selection = json.loads((args.output / "selection.json").read_text())
    assert len(result["points"]) == 250 and result["selection"] == selection
    assert len((args.output / "groups.csv").read_text().splitlines()) == 1 + len(result["groups"])
    assert len(result["analysis_source_sha256"]) == 64
    assert selection["recommendation"]["correction_seeds"] == [17, 23, 29]
    assert selection["training_authorized"] is False
