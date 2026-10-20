"""Causal scheduler accounting and finite-size time comparison boundaries."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from system_study_2026_10.time_model import joint_shapes, break_even, operation_paging_shapes, Coefficients, project_batches, reader_padding, zero_extra_read_sensitivity, planned_backend, graph_policy, graph_preparation, graph_preparation_flops, graph_shape_coverage
from system_study_2026_10.time_batch_shapes import simulate, first_query_redundancy


def test_batched_frontier_preserves_individual_causal_shapes_and_old_anchor_version():
    releases, end = [("v3", 10), ("v4", 30), ("v5", 50)], 70
    histories = {1: (np.arange(1, 70), np.ones(69, dtype=int)),
                 2: (np.array([1,2,10,10,11,30,31,50,51]), np.ones(9,dtype=int)),
                 3: (np.array([1,2]), np.ones(2,dtype=int))}
    from query_relay_2026_10.workload import trigger_indices
    expected = Counter()
    for events in histories.values():
        selections = trigger_indices(events, stride=16, known_vocab_size=10, releases=releases, end=end)
        requests = [dict(version=v,query_timestamp=int(events[0][i]),candidates=16)
                    for v, ids in selections.items() for i in ids]
        shapes, _, _ = joint_shapes(events,requests,releases,end,cap=16)
        for (v,pv,op,n,w,c),count in shapes.items():
            expected[(pv,op,n,w,c)] += count
    batches, actual, _, extra, _ = simulate(histories,list(histories),releases,end,10,maximum=2,cap=16)
    assert actual == expected
    assert not any(key[1] == "v5" and key[2] == "anchorize" for key in batches)
    assert extra["additional_coalescing_flops"] >= 0
    _, windowed, _, bound, transfer = simulate(histories,list(histories),releases,end,10,maximum=2,cap=16,window_users=2)
    assert windowed == expected
    assert bound["max_window_native_and_code_bytes"] <= 2*16*10368
    for method in ("relay","rebuild"):
        for direction in ("h2d","d2h"):
            assert sum(key[-1]*count for key,count in transfer.items() if key[0]==method and key[2]==direction) == len(histories)*len(releases)


def test_operation_paging_does_not_assume_whole_frontier_residency():
    common = dict(method="relay",version="v4",history_length=1024,batch=8,coded_rows=500,calls=2)
    rows = [dict(common,operation="materialize",width=0),
            dict(common,operation="append",width=16),dict(common,operation="score",width=16)]
    paging = operation_paging_shapes(rows)
    assert sum(r["payload_bytes"] for r in paging if r["direction"] == "h2d") == 3*2*8*1024*(9216+1152)
    assert sum(r["payload_bytes"] for r in paging if r["direction"] == "d2h") == 2*8*1024*(9216+2*1152)
    native = operation_paging_shapes([dict(common,method="rebuild",operation="rebuild",width=0)])
    assert len(native) == 1 and native[0]["direction"] == "d2h"
    assert native[0]["payload_bytes"] == 2*8*1024*9216


def test_population_cannot_amortize_a_per_user_disadvantage():
    assert break_even(10,2,1)["minimum_users"] is None
    assert break_even(10,1,1)["minimum_users"] is None
    assert break_even(10,1,2)["minimum_users"] == 11


def test_zero_extra_reads_still_retain_common_service_preparation_and_paging():
    method = dict(sample_accounting={"normalization_to_bulk_median":1.},
                  operations={"materialize":{"point_seconds":2.},"anchorize":{"point_seconds":1.}})
    native = dict(sample_accounting={"normalization_to_bulk_median":1.},
                  operations={"rebuild":{"point_seconds":10.}})
    scenario = dict(methods={"relay":method,"rebuild":native},fixed_S_midpoint_seconds=20.,
                    primary_complete_scenario={"additional_state_paging_seconds":{"relay":4.,"rebuild":2.},
                                               "serial_resource_rebuild_seconds":100.})
    result = zero_extra_read_sensitivity(scenario,population=10)
    assert result["method_total_seconds"] == 115.
    assert result["maximum_allowed_fixed_S_for_reference_break_even_seconds"] == 5.
    assert result["conditional_break_even"]["minimum_users"] == 41
    assert result["asymptotic_total_time_saving_percent"] == 5.
    scenario["fixed_baseline_seconds"] = 3.
    scenario["primary_complete_scenario"]["serial_resource_rebuild_seconds"] = 103.
    result = zero_extra_read_sensitivity(scenario,population=10)
    assert result["method_total_seconds"] == 115.
    assert result["fixed_difference_seconds"] == 17.
    assert result["conditional_break_even"]["minimum_users"] == 35


def test_graph_preparation_counts_one_setup_for_each_method_not_all_warm_traces():
    from read_correction_2026_09.cost import CostModel,eager_read
    units = []
    for method,owner,values in (("relay","relay",(1.,3.)),("rebuild","native",(2.,4.))):
        for value in values:
            shape = dict(owner=owner,operation="score",version="v3",batch=1,width=16,
                         history_length=1024,initial_coded_rows=0,initial_prepare_model_calls=4)
            description = dict(preparation_seconds_by_method={owner:value},graph_shapes=[shape],
                               allocated_delta_bytes_by_method={owner:100})
            units.append(dict(method=method,execution="batch8",graph_state_before=description,
                              graph_captures_during_timing=0,full_trace_warm_seconds=1000.))
    setups = graph_preparation({"time_units":units})
    assert setups["relay"]["point_seconds"] == 2.
    assert setups["rebuild"]["point_seconds"] == 3.
    counts = graph_preparation_flops(setups)
    expected = 3*eager_read(CostModel(192,6,6,"torch"),1024,queries=16)
    assert counts["methods"]["relay"]["total_flops"] == expected
    assert counts["methods"]["rebuild"]["total_flops"] == expected
    assert counts["method_minus_native_setup_flops"] == 0


def test_graph_shape_coverage_counts_positive_prefix_once_but_distinct_versions():
    shape = dict(operation="score",version="v3",batch=8,width=16,history_length=1024,
                 initial_coded_rows=100)
    setup = dict(point_seconds=2.,allocated_live_delta_bytes=1000,shapes=[shape])
    setups = {m:setup for m in ("relay","rebuild")}
    rows = [dict(shape,method="relay",coded_rows=c,calls=999999) for c in (50,500)]
    rows += [dict(rows[0],version="v4"),dict(rows[0],history_length=512)]
    result = graph_shape_coverage(rows,setups,graph_enabled=True,policy=graph_policy({}))
    assert result["relay"]["required_shapes"] == 2
    assert result["relay"]["covered_required_shapes"] == 1
    assert len(result["relay"]["missing_shapes"]) == 1
    assert result["relay"]["missing_shapes"][0]["version"] == "v4"


def test_measured_batch_preserved_before_nearby_shape_fallback():
    common = dict(method="relay",version="v3",operation="score",history_length=1024,width=16,coded_rows=600)
    rows = [dict(common,batch=8,wall_seconds=.008,stage_seconds={"compute":.008}),
            dict(common,batch=1,wall_seconds=.003,stage_seconds={"compute":.003})]
    coefficients = Coefficients(rows)
    shapes = [dict(common,batch=8,calls=10),dict(common,batch=1,calls=3)]
    result = project_batches(shapes,coefficients,"relay")
    assert abs(result["point_seconds"] - .089) < 1e-12
    assert result["user_operations"] == 83
    assert result["weighted_coefficient_coverage"]["exact_batch"] == 83
    # A previously unmeasured prefix retains actual B8 using an explicit tier.
    nearby = dict(common,coded_rows=601)
    match = coefficients.requested(nearby,8)
    assert match["tier"] > 0 and match["batch_matched"]


def test_backend_dispatch_is_bound_to_measured_configuration():
    original = graph_policy({})
    wider = graph_policy({"graph_batch_sizes":list(range(1,9)),"graph_append_widths":[1,2,4,8,16,32]})
    for n in (128,1024):
        for b in (1,2,8):
            for operation,widths in (("score",(1,16,32)),("append",(1,2,4,8,16,32)),("rebuild",(0,))):
                for width in widths:
                    expected = n == 1024 and b in (1,8) and (
                        (operation == "score" and width == 16) or
                        (operation == "append" and width in (1,16,32)) or operation == "rebuild")
                    row = dict(operation=operation,history_length=n,batch=b,width=width)
                    assert (planned_backend(row,True,**original) == "cuda_graph") == expected
                    assert planned_backend(row,False) == "eager"
    row = dict(operation="append",history_length=1024,batch=2,width=8)
    assert planned_backend(row,True,**original) == "eager"
    assert planned_backend(row,True,**wider) == "cuda_graph"


def test_missing_small_graph_batch_uses_full_observed_call_not_per_user_speed():
    row = dict(method="rebuild",version="v3",operation="rebuild",history_length=1024,
               width=0,coded_rows=0,batch=3,compute_backend="cuda_graph",wall_seconds=.009,
               stage_seconds={"compute":.009})
    coefficients = Coefficients([row],graph_enabled=True)
    requested = coefficients.requested(dict(row,batch=1),1)
    assert requested["seconds_per_user_operation"] == .009
    assert requested["batch_proxy_kind"] == "larger_batch_full_call_empirical_proxy"
    assert requested["compute_backend"] == "cuda_graph"


def test_nearby_shape_fallback_cannot_borrow_another_compute_backend():
    common = dict(method="relay",version="v3",operation="score",width=16,coded_rows=600)
    graph = dict(common,history_length=1024,batch=8,compute_backend="cuda_graph",
                 wall_seconds=.008,stage_seconds={"compute":.008})
    eager = dict(common,history_length=900,batch=1,compute_backend="eager",
                 wall_seconds=.003,stage_seconds={"compute":.003})
    coefficients = Coefficients([graph,eager],graph_enabled=True)
    # N512/B8 is unsupported; only a same-backend B1 proxy is admissible.
    unsupported = dict(common,history_length=512,batch=8)
    match = coefficients.requested(unsupported,8)
    assert match["compute_backend"] == "eager" and match["batch"] == 1
    supported = coefficients.requested(dict(common,history_length=1024,batch=8),8)
    assert supported["compute_backend"] == "cuda_graph" and supported["batch_matched"]
    # Missing eager evidence is an error, even though a graph observation exists.
    with pytest.raises(ValueError,match="no measured coefficient"):
        Coefficients([graph],graph_enabled=True).requested(unsupported,8)


def test_reader_dot_padding_counts_project_decode_once_not_per_history_block():
    common = dict(method="relay",operation="score",batch=1,width=16,coded_rows=64,calls=1)
    first = reader_padding([common])
    assert first["additional_reader_dot_padding_flops"] == 2211840
    twice = reader_padding([dict(common,coded_rows=128)])
    assert twice["key_projection_and_value_decode_tile_flops"] == first["key_projection_and_value_decode_tile_flops"]
    assert twice["history_code_dot_tile_flops"] == 2*first["history_code_dot_tile_flops"]
    vector = reader_padding([dict(common,width=4)])
    assert vector.get("additional_reader_dot_padding_flops",0) == 0
    assert vector["vector_coded_queries"] == 4


def test_first_query_overwrite_excludes_queryless_users_and_current_target():
    times = np.arange(1,30)
    items = np.ones(len(times),dtype=int)
    items[9] = 20  # Release's first trigger is OOV; next valid trigger is t=26.
    histories = {1:(times,items),2:(times,np.full(len(times),20))}
    result = first_query_redundancy(histories,[1,2],[("v3",10)],30,10)
    c = result["by_release"]["v3"]["counts"]
    assert c["users_with_first_query"] == c["users_with_overwritten_append"] == 1
    assert c["users_without_query_excluded"] == 1
    assert c["tokens"] == 16 and c["append_user_operations"] == 1
    assert c["native_append_shape_9:16"] == 1
