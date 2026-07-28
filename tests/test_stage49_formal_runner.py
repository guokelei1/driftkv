import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hstu_kvcache.migration import (
    JaggedMigratedKVBatch,
    plan_retained_prefix,
)
from hstu_kvcache.models import HSTU, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SCRIPT = SCRIPT_DIR / "run_cohortkv_stage4_9_formal_confirmation.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_cohortkv_stage4_9_formal_confirmation",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SyntheticHistory:
    def __init__(self, item_ids, behaviors, time_deltas):
        self.item_ids = item_ids
        self.behaviors = behaviors
        self.time_deltas = time_deltas

    def __len__(self):
        return len(self.item_ids)


def frozen_args(**overrides):
    values = {
        "device": None,
        "prepared_data": MODULE.PREPARED_PATH,
        "training_result": MODULE.TRAINING_PATH,
        "checkpoint_dir": MODULE.CHECKPOINT_DIR,
        "compiler_result": MODULE.COMPILER_OUTPUT,
        "runtime_dir": MODULE.RUNTIME_DIR,
        "baseline": MODULE.stage48.BASELINE_PATH,
        "output_dir": MODULE.OUTPUT_DIR,
        "seed": 0,
        "batch_size": MODULE.BATCH_SIZE,
        "warmup_repeats": MODULE.WARMUP_REPEATS,
        "timing_repeats": MODULE.TIMING_REPEATS,
        "force": False,
        "smoke_test": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def fake_steps():
    return [
        {
            "cost": {
                "u": {
                    "samples_ms": [2.0, 3.0, 4.0],
                    "sum_of_component_medians_ms": 3.0,
                },
                "e": {
                    "samples_ms": [10.0, 11.0, 12.0],
                    "median_ms": 11.0,
                },
                "outside_rollout_timer": {
                    "mixed": {
                        "median_ms": 100.0,
                        "components": {
                            "target_delta_append_ms": {"median_ms": 20.0},
                            "latest_append_ms": {"median_ms": 30.0},
                            "short_latest_append_ms": {"median_ms": 0.0},
                        },
                    },
                    "exact": {
                        "median_ms": 120.0,
                        "components": {
                            "target_delta_append_ms": {"median_ms": 25.0},
                            "latest_append_ms": {"median_ms": 35.0},
                            "short_latest_append_ms": {"median_ms": 0.0},
                        },
                    },
                },
                "state_movement_outside_primary": {
                    "h2d_previous_actual": {
                        "direction": "host_to_device_previous_actual",
                        "records": 2,
                        "logical_bytes": 100,
                        "gpu_event_ms": 1.0,
                        "wall_ms": 2.0,
                        "executions": 1,
                        "outside_u_and_e": True,
                        "outside_append_timer": True,
                    },
                    "d2h_next_actual": {
                        "direction": "device_to_host_next_actual",
                        "records": 4,
                        "logical_bytes": 200,
                        "gpu_event_ms": 1.5,
                        "wall_ms": 2.5,
                        "executions": 1,
                        "outside_u_and_e": True,
                        "outside_append_timer": True,
                    },
                },
            }
        }
        for _ in range(MODULE.NUM_EDGES)
    ]


def synthetic_cache(
    record_id: int = 7,
    version: int = 2,
    dtype: torch.dtype = torch.float16,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=(record_id,),
        migration_anchor_version=f"theta{version}",
        served_kv_target=f"theta{version}",
        k=torch.zeros((2, 3, 4), dtype=dtype),
        v=torch.ones((2, 3, 4), dtype=dtype),
        lengths=torch.tensor([3], dtype=torch.long),
        offsets=torch.tensor([0, 3], dtype=torch.long),
    )


def test_static_smoke_freezes_only_selected_candidates() -> None:
    result = MODULE.smoke_payload(frozen_args())

    assert result["status"] == "smoke_passed"
    assert not result["scientific_result"]
    assert result["candidate_order"] == [
        "token_debt_total10",
        "staggered_renewal_h12",
    ]
    assert result["measurement_contract"]["old_denominator_reused"] is False
    assert result["measurement_contract"][
        "stage47_or_stage48_edge_runner_called"
    ] is False
    assert (
        result["measurement_contract"]["persistent_store"]
        == "cpu_singleton_fp16_post_append_full_mixed_kv"
    )
    assert (
        "evaluator-only host spill"
        in result["measurement_contract"]["capacity_scope"]
    )
    assert all(result["checks"].values())


def test_persistent_store_contract_is_cpu_singleton_fp16_and_versioned() -> None:
    store = {7: synthetic_cache()}

    checks = MODULE._persistent_cpu_store_checks(
        store,
        expected_version=2,
        expected_lengths={7: 3},
    )

    assert all(checks.values())
    assert MODULE._cuda_tensor_bytes(store) == 0
    assert MODULE._record_id_sha256(store) == MODULE._record_id_sha256({7})
    assert MODULE._record_length_sha256(store)


def test_persistent_store_contract_fails_closed_on_dtype_and_version() -> None:
    store = {7: synthetic_cache(version=1, dtype=torch.float32)}

    checks = MODULE._persistent_cpu_store_checks(
        store,
        expected_version=2,
        expected_lengths={7: 3},
    )

    assert not checks["persistent_kv_is_fp16"]
    assert not checks["persistent_versions_match"]


def test_record_store_transfer_preserves_singleton_cpu_extents() -> None:
    source = {
        7: synthetic_cache(record_id=7),
        9: synthetic_cache(record_id=9),
    }

    transferred = MODULE._transfer_record_store(
        source,
        (9,),
        torch.device("cpu"),
    )

    assert set(transferred) == {9}
    assert transferred[9].record_ids == (9,)
    assert transferred[9].k.device.type == "cpu"
    assert all(
        MODULE._persistent_cpu_store_checks(
            transferred,
            expected_version=2,
            expected_lengths={9: 3},
        ).values()
    )
    assert all(
        MODULE._staged_device_store_checks(
            transferred,
            (9,),
            2,
            {9: 3},
            torch.device("cpu"),
        ).values()
    )


def test_formal_requires_one_explicit_available_cuda(monkeypatch) -> None:
    with pytest.raises(ValueError, match="requires --device"):
        MODULE.validate_args(frozen_args(smoke_test=False))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="available explicit CUDA"):
        MODULE.validate_args(
            frozen_args(
                smoke_test=False,
                device="cuda:1",
            )
        )
    MODULE.validate_args(
        frozen_args(
            smoke_test=False,
            device="cuda:0",
        )
    )


def test_timing_repetitions_are_frozen() -> None:
    with pytest.raises(ValueError, match="one warmup and three"):
        MODULE.validate_args(
            frozen_args(
                warmup_repeats=0,
            )
        )
    with pytest.raises(ValueError, match="one warmup and three"):
        MODULE.validate_args(
            frozen_args(
                timing_repeats=1,
            )
        )


def test_primary_cost_uses_new_paired_e_and_excludes_append() -> None:
    result = MODULE.summarize_paired_cost(
        fake_steps(),
        MODULE.TIMING_REPEATS,
    )

    assert result["mixed_u_ms"] == 33.0
    assert result["paired_fresh_exact_e_ms"] == 121.0
    assert result["primary_sum_u_over_sum_e"] == 33.0 / 121.0
    assert result["mixed_outside_rollout_ledger_ms"] == 1100.0
    assert result["exact_outside_rollout_ledger_ms"] == 1320.0
    assert result["target_append_a_ledger"]["mixed_ms"] == 550.0
    assert result["target_append_a_ledger"]["exact_ms"] == 660.0
    assert result["target_append_excluded_from_u_and_e"]
    assert not result["old_exact_denominator_reused"]
    movement = result["state_movement_outside_primary"]
    assert movement["h2d_previous_actual"]["logical_bytes"] == 1100
    assert movement["d2h_next_actual"]["logical_bytes"] == 2200
    assert movement["logical_bytes"] == 3300
    assert movement["gpu_event_ms"] == 27.5
    assert movement["wall_ms"] == 49.5
    assert movement["excluded_from_primary"]
    assert movement["reported_separately"]


def test_recursive_chain_consumes_previous_actual_post_append_state() -> None:
    initial_cache = object()
    initial_last_exact = object()
    initial_expected = object()
    seen = []

    def edge_runner(
        source_version,
        actual_cache,
        actual_last_exact,
        expected_ids,
        scheduler_state,
    ):
        seen.append(
            (
                source_version,
                actual_cache,
                actual_last_exact,
                expected_ids,
                scheduler_state,
            )
        )
        return (
            ("post_append_actual", source_version),
            ("last_exact", source_version),
            ("expected", source_version),
            ("scheduler", source_version),
            {"version": source_version + 1},
            {
                "source_version": source_version,
                "target_version": source_version + 1,
            },
        )

    result = MODULE._advance_recursive_chain(
        edge_runner,
        initial_cache,
        initial_last_exact,
        initial_expected,
    )

    assert len(seen) == MODULE.NUM_EDGES
    assert seen[0][1:4] == (
        initial_cache,
        initial_last_exact,
        initial_expected,
    )
    for source_version in range(1, MODULE.NUM_EDGES):
        assert seen[source_version][1] == (
            "post_append_actual",
            source_version - 1,
        )
        assert seen[source_version][2] == (
            "last_exact",
            source_version - 1,
        )
        assert seen[source_version][3] == (
            "expected",
            source_version - 1,
        )
    assert result[0] == ("post_append_actual", MODULE.NUM_EDGES - 1)
    assert len(result[4]) == MODULE.NUM_EDGES
    assert len(result[5]) == MODULE.NUM_EDGES


def test_recursive_chain_releases_outer_initial_cache_mapping() -> None:
    initial = {"theta0": object()}

    def edge_runner(
        source_version,
        actual_cache,
        actual_last_exact,
        expected_ids,
        scheduler_state,
    ):
        assert actual_cache
        return (
            {f"theta{source_version + 1}": object()},
            actual_last_exact,
            expected_ids,
            scheduler_state,
            {"version": source_version + 1},
            {
                "source_version": source_version,
                "target_version": source_version + 1,
            },
        )

    MODULE._advance_recursive_chain(
        edge_runner,
        initial,
        {},
        set(),
    )

    assert initial == {}


def test_formal_edge_is_independent_of_append_first_runner() -> None:
    edge_source = inspect.getsource(MODULE._run_formal_edge)
    candidate_source = inspect.getsource(MODULE.run_candidate)
    cost_source = inspect.getsource(MODULE.summarize_paired_cost)

    assert "stage48._run_edge" not in edge_source
    assert "stage48._run_edge" not in candidate_source
    assert "base._run_edge" not in edge_source
    assert "cumulative_exact_gpu_denominator_ms" not in cost_source
    assert "all_exact_reference_ms" not in cost_source


def test_formal_runner_uses_groupwise_host_staging_not_gpu_theta0_store() -> None:
    group_source = inspect.getsource(MODULE._run_group)
    edge_source = inspect.getsource(MODULE._run_formal_edge)
    candidate_source = inspect.getsource(MODULE.run_candidate)
    init_source = inspect.getsource(MODULE._initialize_theta0_host)

    h2d_position = group_source.index("_timed_store_transfer")
    first_u_timer = group_source.index("_timed_repeated")
    group_position = edge_source.index("result = _run_group")
    d2h_position = edge_source.index(
        "host_result_cache, d2h_next_actual",
        group_position,
    )
    parity_position = edge_source.index(
        "_exact_parity_group_summary",
        d2h_position,
    )
    next_store_position = edge_source.index(
        "next_cache.update",
        parity_position,
    )

    assert h2d_position < first_u_timer
    assert group_position < d2h_position < parity_position
    assert d2h_position < next_store_position
    assert "stage48._initialize_theta0" not in candidate_source
    assert "_initialize_theta0_host" in candidate_source
    assert "partial(full.to, torch.device(\"cpu\"))" in init_source


def test_formal_runner_gates_cuda_lifetime_and_explicitly_limits_claim() -> None:
    edge_source = inspect.getsource(MODULE._run_formal_edge)
    candidate_source = inspect.getsource(MODULE.run_candidate)
    summary_source = inspect.getsource(MODULE.confirmation_summary)

    assert "result_retains_zero_cuda_tensor_bytes" in edge_source
    assert "post_cleanup_live_gpu_growth_is_bounded" in edge_source
    assert "no_cross_group_live_gpu_growth" in edge_source
    assert "edge_peak_below_device_capacity" in edge_source
    assert '"persistent_gpu_kv_bytes": 0' in edge_source
    assert '"full_cohort_hbm_claim": False' in candidate_source
    assert '"end_to_end_state_movement_claim": False' in candidate_source
    assert "evaluator-only host spill" in summary_source


def test_fp32_parity_branch_is_algorithmic_equivalence_authority() -> None:
    model = HSTU(
        HSTUConfig(
            num_items=16,
            num_behaviors=4,
            hidden_size=8,
            num_layers=2,
            num_heads=1,
            head_dim=8,
            max_seq_len=4,
            input_dropout=0.0,
        )
    )
    model.eval()
    history = SyntheticHistory(
        [2, 3, 4, 5],
        [1, 2, 1, 3],
        [0.0, 1.0, 2.0, 3.0],
    )
    window = SimpleNamespace(
        records={7: SimpleNamespace(history=history)},
    )
    plans = {
        0: plan_retained_prefix(
            0,
            7,
            ("A", "B"),
            ("A", "B", "C", "D"),
            "old",
            "target",
            True,
            True,
        )
    }
    parity_cache, parity_hidden = MODULE._exact_parity_branch_once(
        model,
        (0,),
        (0,),
        (),
        (),
        plans,
        {0: {"record_id": 0, "user_id": 7}},
        window,
        1,
        model.cfg,
        torch.device("cpu"),
    )
    full_batch = MODULE.stage49._segment_batch(
        [window.records[7]],
        (0,),
        (4,),
        torch.device("cpu"),
    )
    fresh_cache, fresh_hidden = MODULE.stage49._exact_full(
        model,
        full_batch,
        (0,),
        1,
        torch.float32,
    )

    assert parity_cache.k.dtype == torch.float32
    assert parity_cache.v.dtype == torch.float32
    assert torch.allclose(
        parity_cache.k,
        fresh_cache.k,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        parity_cache.v,
        fresh_cache.v,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        parity_hidden,
        fresh_hidden,
        atol=1e-5,
        rtol=1e-5,
    )
    diagnostics, checks = MODULE._exact_parity_group_summary(
        model.cfg,
        model,
        1,
        window,
        ({"record_id": 0, "user_id": 7},),
        {0: {"record_id": 0, "user_id": 7}},
        plans,
        SimpleNamespace(
            migrate_ids=(0,),
            scheduled_exact_ids=(),
        ),
        torch.arange(1, model.cfg.num_prediction_items + 1),
        torch.device("cpu"),
    )
    assert diagnostics["record_chunk_size"] == 1
    assert not diagnostics["gpu_outputs_retained_after_record"]
    assert diagnostics["records"] == 1
    assert diagnostics["k_max_abs"] <= 1e-5
    assert diagnostics["v_max_abs"] <= 1e-5
    assert all(checks.values())


def test_fp16_endpoint_is_not_used_as_algorithmic_equivalence_gate() -> None:
    group_source = inspect.getsource(MODULE._run_group)
    parity_source = inspect.getsource(MODULE._exact_parity_group_summary)

    assert "_exact_parity_branch_once" not in group_source
    assert "exact_fp32_parity_k_matches_fresh" in parity_source
    assert "exact_fp32_parity_v_matches_fresh" in parity_source
    assert '"exact_two_stage_k_matches_fresh"' not in group_source
    assert '"exact_two_stage_v_matches_fresh"' not in group_source
    assert (
        "quantized_deployment_diagnostic_not_equivalence_gate"
        in group_source
    )


def test_fp32_parity_runs_after_fp16_group_scope_and_releases_each_record() -> None:
    source = inspect.getsource(MODULE._run_formal_edge)
    group_position = source.index("result = _run_group")
    cleanup_position = source.index("gc.collect()", group_position)
    parity_position = source.index(
        "_exact_parity_group_summary",
        cleanup_position,
    )
    summary_source = inspect.getsource(
        MODULE._exact_parity_group_summary
    )

    assert group_position < cleanup_position < parity_position
    assert '"record_chunk_size": 1' in summary_source
    assert "del (" in summary_source
    assert "torch.cuda.empty_cache()" in summary_source
    assert '"gpu_outputs_retained_after_record": False' in summary_source


def test_topk_boundary_parity_accepts_only_numerically_ambiguous_swap() -> None:
    left = torch.tensor([[1.0, 0.9, 0.8, 0.799999, 0.2]])
    right = torch.tensor([[1.0, 0.9, 0.799999, 0.800001, 0.2]])

    result = MODULE.topk_boundary_parity(left, right, 3)

    assert result["numerical_score_equivalent"]
    assert not result["exact_topk_order_equal"]
    assert not result["exact_topk_set_equal"]
    assert result["boundary_equivalent"]
    assert result["passed"]
    assert result["order_mismatched_rows"] == 1
    assert result["set_mismatched_rows"] == 1
    assert result["maximum_symmetric_difference_items"] == 2
    assert result["minimum_topk_overlap"] == 2 / 3
    assert result["first_mismatch"]["left_only"] == [2]
    assert result["first_mismatch"]["right_only"] == [3]


def test_topk_boundary_parity_rejects_non_equivalent_scores() -> None:
    left = torch.tensor([[1.0, 0.9, 0.8, 0.7]])
    right = torch.tensor([[1.0, 0.9, 0.6, 0.85]])

    result = MODULE.topk_boundary_parity(left, right, 3)

    assert not result["numerical_score_equivalent"]
    assert not result["passed"]
    assert result["maximum_score_max_abs"] > MODULE.EXACT_PARITY_ATOL


def test_topk_boundary_parity_tracks_near_tied_internal_reordering() -> None:
    left = torch.tensor([[1.0, 0.9, 0.800001, 0.8]])
    right = torch.tensor([[1.0, 0.9, 0.8, 0.800001]])

    result = MODULE.topk_boundary_parity(left, right, 4)

    assert not result["exact_topk_order_equal"]
    assert result["exact_topk_set_equal"]
    assert result["order_mismatched_rows"] == 1
    assert result["set_mismatched_rows"] == 0
    assert result["inverted_common_item_pairs"] == 1
    assert result["boundary_equivalent"]
    assert result["passed"]


def test_topk_boundary_parity_preserves_exact_equality_evidence() -> None:
    scores = torch.tensor([[0.4, 0.3, 0.2, 0.1]])

    result = MODULE.topk_boundary_parity(scores, scores.clone(), 2)

    assert result["numerical_score_equivalent"]
    assert result["exact_topk_order_equal"]
    assert result["exact_topk_set_equal"]
    assert result["boundary_equivalent"]
    assert result["order_mismatched_rows"] == 0
    assert result["set_mismatched_rows"] == 0
    assert result["first_mismatch"] is None
    assert result["passed"]
