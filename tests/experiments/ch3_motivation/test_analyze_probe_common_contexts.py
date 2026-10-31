"""Known common losses: context reversals, paired signs, and incomplete inputs."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.analyze_probe_common_contexts import CONTEXTS, aggregate_loaded, analyze


def metric(value):
    row = dict(loss=value, read_mse=value*.4, score_mse=value*.6, raw_score_mse=value/10)
    return dict(users=16, **row, per_user=[dict(uid=uid, **row) for uid in range(16)])


def fixture():
    points, groups = [], []
    losses = {"a": {"f": [5,5,5], "fg": [6,4,3], "fgh": [2,3,4]},
              "b": {"f": [6,6,6], "fg": [5,5,5], "fgh": [3,2,1]}}
    for recipe, stages in losses.items():
        for method, values in stages.items():
            groups.append(dict(recipe=recipe, method_label=method, seed=17, variant="full_item",
                width=128, depth=1, fit_users=256, selection_eligible=True,
                mean_loss_by_context={context:value+.3 for context,value in zip(CONTEXTS,values)}))
            for edge in range(1,6):
                point=dict(status="validated", config=dict(hidden_width=128,depth=1), variant="full_item",
                    fitting_users=256, training_identity=dict(edge=edge,seed=17,fit_users=256,h_initialization="standard"),
                    best_step=0 if edge==5 and method=="fgh" else 8, completed_steps=16,
                    step_zero=edge==5 and method=="fgh",selection_eligible=True,
                    contexts={context:dict(status="validated",common_validation_sha256=f"edge{edge}/{context}",
                              metrics=metric(value+edge/10)) for context,value in zip(CONTEXTS,values)})
                points.append(dict(recipe=recipe,method_label=method,seed=17,edge=edge,
                    checkpoint=f"/{recipe}/edge_{edge}/{method}.pt",validation=point))
    return dict(format="ch3_final_calibration_common_selection_v1",expected_weights=30,
        points=points,groups=groups,selection=dict(status="recommendation_for_review",recommendation=dict(recipe="a")))


def get(report, recipe, method, context):
    return next(row for row in report["context_metrics"] if
                (row["recipe"],row["method_label"],row["context"])==(recipe,method,context))


def contrast(report, recipe, before, context):
    return next(row for row in report["stage_differences"] if
                (row["recipe"],row["before"],row["context"])==(recipe,before,context))


def test_equal_edge_four_metrics_context_rank_reversal_and_negative_improvements():
    source=fixture(); report=aggregate_loaded(source)
    assert report["status"]=="complete" and report["validated_context_cells"]==90
    a=get(report,"a","fgh","cutover"); b=get(report,"b","fgh","cutover")
    assert a["loss"]==pytest.approx(2.3) and a["read_mse"]==pytest.approx(.92)
    assert a["score_mse"]==pytest.approx(1.38) and a["raw_score_mse"]==pytest.approx(.23)
    assert a["fgh_rank"]==1 and b["fgh_rank"]==2 and a["step_zero_edges"]==[5]
    assert get(report,"a","fgh","gap_empirical100")["fgh_rank"]==2
    assert get(report,"b","fgh","gap_empirical100")["fgh_rank"]==1
    assert contrast(report,"a","f","cutover")["mean_loss_after_minus_before"]==pytest.approx(1.)
    assert contrast(report,"a","f","gap_empirical100")["mean_loss_after_minus_before"]==pytest.approx(-2.)
    assert contrast(report,"a","fg","gap_empirical100")["mean_loss_after_minus_before"]==pytest.approx(1.)
    assert contrast(report,"a","fg","cutover")["mean_loss_after_minus_before"]==pytest.approx(-4.)
    assert not report["selection_changed"] and report["source_recommendation"]=="a"
    assert not report["quality_metrics_computed"]


def test_partial_edge_or_context_keeps_cells_and_step0_but_no_mean_or_rank():
    source=fixture()
    point=next(row["validation"] for row in source["points"] if row["recipe"]=="a" and row["method_label"]=="fgh" and row["edge"]==5)
    point["status"]="failed"; point["error"]="later context failed"
    point["contexts"].pop("gap_empirical100")
    report=aggregate_loaded(source)
    row=get(report,"a","fgh","cutover")
    assert row["loss"] is None and row["fgh_rank"] is None and row["step_zero_edges"]==[5]
    assert row["missing_or_failed_edges"]==[5]
    cell=next(row for row in report["points"] if row["recipe"]=="a" and row["method_label"]=="fgh" and row["edge"]==5 and row["context"]=="cutover")
    assert cell["loss"]==2.5 and cell["point_status"]=="failed" and cell["step_zero"]
    delta=contrast(report,"a","fg","cutover")
    assert delta["mean_loss_after_minus_before"] is None
    assert delta["edges"][0]["loss_after_minus_before"]==pytest.approx(-4.)
    assert delta["edges"][4]["loss_after_minus_before"] is None


def test_missing_inventory_pending_and_absent_stage_are_explicit():
    source=fixture();source["expected_weights"]=475
    source["points"]=[row for row in source["points"] if row["recipe"]=="a" and row["method_label"]=="f"]
    source["groups"]=[row for row in source["groups"] if row["recipe"]=="a" and row["method_label"]=="f"]
    source["points"][0]["validation"]={"status":"pending","contexts":{}}
    report=aggregate_loaded(source)
    assert report["status"]=="partial_common_validation" and not report["inventory_complete"]
    assert report["complete_method_groups"]==0 and report["context_metrics"][0]["loss"] is None
    assert all(edge["status"]=="stage_not_planned" for row in report["stage_differences"] for edge in row["edges"])
    for row in source["points"]:row["validation"]={"status":"missing","contexts":{}}
    waiting=aggregate_loaded(source)
    assert waiting["status"]=="waiting_common_validation" and waiting["validated_context_cells"]==0


def test_never_pair_different_training_settings_even_same_recipe_and_seed():
    source=fixture()
    for row in source["points"]:
        if row["recipe"]=="a" and row["method_label"]=="fg":
            row["validation"]["training_identity"]["h_initialization"]="balanced"
    report=aggregate_loaded(source)
    delta=contrast(report,"a","f","cutover")
    assert delta["mean_loss_after_minus_before"] is None
    assert all(row["status"]=="configuration_or_identity_mismatch" for row in delta["edges"])


def test_opposing_edge_differences_cancel_with_equal_edge_weights():
    source=fixture()
    for row in source["points"]:
        if row["recipe"]=="a" and row["method_label"] in ("f","fg"):
            loss=5.+({1:2.,2:-2.}.get(row["edge"],0.) if row["method_label"]=="fg" else 0.)
            for cell in row["validation"]["contexts"].values():cell["metrics"]=metric(loss)
    for group in source["groups"]:
        if group["recipe"]=="a" and group["method_label"] in ("f","fg"):
            group["mean_loss_by_context"]={context:5. for context in CONTEXTS}
    delta=contrast(aggregate_loaded(source),"a","f","cutover")
    assert [row["loss_after_minus_before"] for row in delta["edges"]]==[2.,-2.,0.,0.,0.]
    assert delta["mean_loss_after_minus_before"]==0.


@pytest.mark.parametrize("fault",["nan","wrong_mean","fingerprint","duplicate","old_only"])
def test_rejects_nonfinite_inconsistent_or_different_common_tasks(fault):
    source=fixture();cell=source["points"][0]["validation"]["contexts"]["cutover"]
    if fault=="nan":cell["metrics"]["loss"]=float("nan")
    elif fault=="wrong_mean":cell["metrics"]["raw_score_mse"]*=2
    elif fault=="fingerprint":cell["common_validation_sha256"]="different"
    elif fault=="duplicate":source["points"].append(copy.deepcopy(source["points"][0]))
    else:source["format"]="ch3_common_validation_selection_v1"
    with pytest.raises(ValueError):aggregate_loaded(source)


def test_export_only_saved_json_with_hashes_csv_and_no_new_selection(tmp_path):
    source=tmp_path/"selection_summary.json";source.write_text(json.dumps(fixture()))
    output=tmp_path/"analysis"
    result=analyze(source,output)
    assert result["source"]["sha256"]==hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads((output/"summary.json").read_text())["complete_method_groups"]==6
    assert len((output/"context_metrics.csv").read_text().splitlines())==19
    assert len((output/"point_metrics.csv").read_text().splitlines())==91
    assert len((output/"stage_differences.csv").read_text().splitlines())==13
    with pytest.raises(ValueError,match="new output attempt"):analyze(source,output)
