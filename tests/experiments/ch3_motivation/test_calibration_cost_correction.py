"""Only shared f/h input normalization is removed from calibration costs."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.calibration_cost_correction import (
    AFFECTED_CORE_SHA256, CORE_SOURCE, AFFECTED_VARIANT_SHA256, VARIANT_SOURCE, correction_record,
    apply_to_costs, audit_checkpoints,
)


def payload(label="fgh"):
    stage={"f_extended":"f","fg_extended":"fg"}.get(label,label)
    return dict(format="ch3_probe_variant_v1",variant="full_item",stage=stage,
        config=dict(num_layers=2,num_heads=3,head_dim=4,item_dim=8,normalize=True,include_item=True),
        metadata=dict(stage=stage,method_label=label,completed_steps=2,
            cumulative_calibration_flops_estimate=100000,total_calibration_flops_estimate=90000,
            operation_counts=dict(training_probe_rows=7,validation_probe_rows=11,
                training_sample_calls=2,training_backward_calls=2),
            identity=dict(edge=1,seed=17,numerical_sources={CORE_SOURCE:AFFECTED_CORE_SHA256,
                VARIANT_SOURCE:AFFECTED_VARIANT_SHA256})))


def test_tiny_hand_calculation_retains_the_declared_backward_multiplier():
    raw=payload();before=copy.deepcopy(raw)
    record=correction_record(raw)
    # row_dim16, layers*heads6: duplicated cost192 per historical row.
    assert record["duplicated_forward_flops_per_probe_row"]==192
    assert record["deduction_components_flops"]==dict(training_forward=1344,
        validation_forward=2112,training_backward_estimate=2688)
    assert record["shared_row_normalization_deduction_flops"]==6144
    assert record["corrected_stage_calibration_flops_estimate"]==83856
    assert record["corrected_cumulative_calibration_flops_estimate"]==93856
    assert raw==before and record["original_metadata_and_weights_modified"] is False


@pytest.mark.parametrize("label",["f","fg","f_extended","fg_extended"])
def test_non_h_branches_have_zero_deduction(label):
    record=correction_record(payload(label))
    assert record["shared_row_normalization_deduction_flops"]==0
    assert record["corrected_cumulative_calibration_flops_estimate"]==100000


def test_disabled_normalization_does_not_deduct_arithmetic():
    value=payload();value["config"]["normalize"]=False
    assert correction_record(value)["shared_row_normalization_deduction_flops"]==0


@pytest.mark.parametrize("fault",["missing_training_rows","missing_validation_rows","negative_rows",
    "source","variant_source","variant","backward_calls","normalization_config"])
def test_no_guessing_of_missing_counts_geometry_or_affected_source(fault):
    value=payload()
    if fault=="missing_training_rows":value["metadata"]["operation_counts"].pop("training_probe_rows")
    elif fault=="missing_validation_rows":value["metadata"]["operation_counts"].pop("validation_probe_rows")
    elif fault=="negative_rows":value["metadata"]["operation_counts"]["training_probe_rows"]=-1
    elif fault=="source":value["metadata"]["identity"]["numerical_sources"][CORE_SOURCE]="new_fixed_source"
    elif fault=="variant_source":value["metadata"]["identity"]["numerical_sources"].pop(VARIANT_SOURCE)
    elif fault=="variant":value["variant"]="layerwise_kv"
    elif fault=="backward_calls":value["metadata"]["operation_counts"]["training_backward_calls"]=1
    else:value["config"].pop("normalize")
    with pytest.raises(ValueError):correction_record(value)


def test_two_deployment_accounts_only_change_calibration_and_derived_totals():
    correction=correction_record(payload())
    account=dict(calibration_flops=100000,complete_added_flops=130000,once_rebuild_flops=200000,
        percent_of_once_rebuild=65.,release_preparation_flops=10000,extra_read_flops=15000,
        extra_native_writer_flops=4000,shared_graph_setup_flops=1000)
    costs=dict(account,candidate_reads_230=dict(account,complete_added_flops=140000,
        once_rebuild_flops=400000,percent_of_once_rebuild=35.),
        comparison_to_raw_graph=dict(complete_added_flops=150000,normalized_230_complete_added_flops=160000))
    before=copy.deepcopy(costs);result=apply_to_costs(costs,correction)
    assert costs==before and result["raw"]==before
    for value,total,denominator in ((result["corrected"],123856,200000),
            (result["corrected"]["candidate_reads_230"],133856,400000)):
        assert value["calibration_flops"]==93856 and value["complete_added_flops"]==total
        assert value["percent_of_once_rebuild"]==pytest.approx(100*total/denominator)
        for key in ("release_preparation_flops","extra_read_flops","extra_native_writer_flops","shared_graph_setup_flops"):
            assert value[key]==account[key]
        assert value["once_rebuild_flops"]==denominator
    assert result["corrected"]["comparison_to_raw_graph"]["complete_added_flops"]==143856
    bad=copy.deepcopy(costs);bad["candidate_reads_230"]["calibration_flops"]+=1
    with pytest.raises(ValueError,match="raw checkpoint ledger"):apply_to_costs(bad,correction)
    with pytest.raises(ValueError,match="raw checkpoint ledger"):
        apply_to_costs(result["corrected"],correction)


def test_actual_three_edges_nine_weights_keep_raw_ledgers_and_correct_only_h():
    base=ROOT/"experiments/ch3_motivation/runs/e5/calibration_sampling/attempt_1"/(
        "full_item128_fit256_fb50_cutover_lr3e5_cap2048_pat1024")
    paths=[base/f"edge_{edge}/seed_17/{stage}.pt" for edge in (1,2,3) for stage in ("f","fg","fgh")]
    references=[ROOT/f"experiments/ch3_motivation/runs/e3/edge_{edge}/main/attempt_1/summary.json" for edge in (1,2,3)]
    if not all(path.exists() for path in paths+references):
        pytest.skip("retained PhaseA weights and fixed2000 Rebuild references are unavailable")
    report=audit_checkpoints(paths,references)
    assert report["stage_records"]==9 and report["complete_ladders"]==3
    expected={1:301355827200,2:558570086400,3:627746918400}
    for row in report["records"]:
        deduction=expected[row["edge"]] if row["method_label"]=="fgh" else 0
        assert row["shared_row_normalization_deduction_flops"]==deduction
        assert row["corrected_cumulative_calibration_flops_estimate"]==(
            row["raw_cumulative_calibration_flops_estimate"]-deduction)
        calls=row["executed_validation"]
        assert calls["validation_sample_calls"]==calls["rounds"]*16
        if row["method_label"]=="f":
            assert calls["validation_sample_calls"]=={1:2512,2:4112,3:4112}[row["edge"]]
        assert not row["linear_amortization_lower_bound"]["replaces_main2000_cost_denominator"]
    h=[row for row in report["records"] if row["method_label"]=="fgh"]
    assert h[0]["executed_validation"]["best_step"]==0
    assert h[0]["executed_validation"]["completed_steps"]==1024
    json.dumps(report)  # Every provenance/count/cost field is an exportable CPU record.
