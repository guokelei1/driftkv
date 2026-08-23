from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml
import hashlib

from hstu_kvcache.models import (
    HSTU,
    HSTUConfig,
    append_with_rolling_cap,
    hybrid_tail_refresh,
    project_exact_layer0_segment,
    retain_latest_cache,
    transition_work,
)

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model() -> HSTU:
    torch.manual_seed(7)
    return HSTU(HSTUConfig(
        num_items=64, num_prediction_items=64, num_behaviors=4,
        hidden_size=16, num_layers=4, num_heads=2, max_seq_len=32,
        input_dropout=0.0, attn_dropout=0.0,
    )).eval()


def raw() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long),
        torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]], dtype=torch.long),
        torch.tensor([[0, 5, 9, 3, 7, 2, 11, 4]], dtype=torch.float32),
    )


def test_p9_4_contract_limits_actions_and_keeps_controller_closed() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p9_4_executor_contract_v1.yaml").read_text())
    assert [row["name"] for row in contract["actions"]] == [
        "noop", "layer0_recent128", "layer0_middle", "layer0_full",
        "hybrid_tail32", "hybrid_tail128", "exact_all",
    ]
    assert contract["scope"]["all_24_F_cells_required_after_canary"] is True
    assert contract["authorization"]["controller"] is False
    assert contract["hybrid_tail"]["prohibited_claim"] == "exact_CurrentFull_tail_KV"
    paths = {
        "p9_3_result": "configs/contracts/p9_3_2d_tomography_result_v1.yaml",
        "p9_3_raw_seal": "results/p9/p9_3_2d_tomography_raw_seal_v1.json",
        "p9_3_target_free": "results/p9/p9_3_2d_tomography_v1.json",
        "p9_3_quality": "results/p9/p9_3_2d_quality_companions_v1.json",
    }
    for name, relative in paths.items():
        assert sha256(ROOT / relative) == contract["input_hashes"][name]


def test_layer0_segment_projection_matches_current_exact_and_preserves_other_state() -> None:
    current = model()
    parent = copy.deepcopy(current)
    with torch.no_grad():
        parent.blocks[0].attn.k_proj.weight.add_(0.2)
        parent.blocks[0].attn.v_proj.weight.sub_(0.1)
    items, behaviors, deltas = raw()
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    exact = current.compute_kv(items, behaviors, deltas)
    migrated = project_exact_layer0_segment(current, parent_cache, items, behaviors, deltas, "middle")
    selected = slice(2, 6)
    assert torch.allclose(migrated.k[0, :, selected], exact.k[0, :, selected], atol=1e-7, rtol=0)
    assert torch.allclose(migrated.v[0, :, selected], exact.v[0, :, selected], atol=1e-7, rtol=0)
    assert torch.equal(migrated.k[0, :, :2], parent_cache.k[0, :, :2])
    assert torch.equal(migrated.k[0, :, 6:], parent_cache.k[0, :, 6:])
    assert torch.equal(migrated.k[1:], parent_cache.k[1:])
    assert torch.equal(migrated.v[1:], parent_cache.v[1:])


def test_hybrid_tail_is_exact_when_parent_prefix_is_current_model() -> None:
    current = model()
    items, behaviors, deltas = raw()
    exact = current.compute_kv(items, behaviors, deltas)
    replayed = hybrid_tail_refresh(current, exact, items, behaviors, deltas, width=3)
    assert torch.allclose(replayed.k, exact.k, atol=1e-6, rtol=0)
    assert torch.allclose(replayed.v, exact.v, atol=1e-6, rtol=0)


def test_retain_latest_cache_keeps_tail_not_prefix() -> None:
    current = model()
    items, behaviors, deltas = raw()
    cache = current.compute_kv(items, behaviors, deltas)
    latest = retain_latest_cache(cache, 3)
    assert latest.seq_len == 3
    assert torch.equal(latest.k, cache.k[:, :, -3:])
    assert torch.equal(latest.v, cache.v[:, :, -3:])


def test_rolling_append_evicts_before_each_event() -> None:
    current = model()
    items, behaviors, deltas = raw()
    initial = current.compute_kv(items[:, :4], behaviors[:, :4], deltas[:, :4])
    rolled = append_with_rolling_cap(
        current, initial, items[:, 4:6], behaviors[:, 4:6], deltas[:, 4:6], max_length=4
    )
    manual = initial
    for position in range(4, 6):
        manual = retain_latest_cache(manual, 3)
        _, manual = current.forward_with_cache(
            manual,
            items[:, position : position + 1],
            behaviors[:, position : position + 1],
            deltas[:, position : position + 1],
        )
    assert rolled.seq_len == 4
    assert torch.equal(rolled.k, manual.k)
    assert torch.equal(rolled.v, manual.v)


def test_transition_work_keeps_compute_and_io_dimensions_separate() -> None:
    current = model()
    items, behaviors, deltas = raw()
    cache = current.compute_kv(items, behaviors, deltas)
    layer0 = transition_work("layer0_middle", cache, items, behaviors, deltas)
    hybrid = transition_work("hybrid_tail32", cache, items, behaviors, deltas)
    exact = transition_work("exact_all", cache, items, behaviors, deltas)
    assert layer0.recomputed_token_layers == 4
    assert layer0.attention_pair_work == 0
    assert hybrid.recomputed_token_layers == exact.recomputed_token_layers == 32
    assert hybrid.attention_pair_work == exact.attention_pair_work
    assert layer0.raw_history_read_bytes < exact.raw_history_read_bytes


def test_p9_4_ledger_replays_all_cells_only_on_gpu01() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_p9_executor as ledger

    jobs = ledger.jobs()
    assert len(jobs) == 24
    assert {(job.release, job.model, job.seed) for job in jobs} == {
        (release, model, seed)
        for release in ("r0", "r1_edge1", "r1_edge2", "r2")
        for model in ("m0_f", "m1")
        for seed in (17, 37, 71)
    }
    assert "CUDA_VISIBLE_DEVICES=1" in jobs[0].command(1)
    assert "CUDA_VISIBLE_DEVICES=2" not in jobs[0].command(1)


def test_materialized_lineage_canary_is_sealed_and_keeps_scheduler_closed() -> None:
    contract_path = ROOT / "configs/contracts/p9_4_materialized_lineage_canary_v1.yaml"
    result = yaml.safe_load(
        (ROOT / "configs/contracts/p9_4_materialized_lineage_canary_result_v1.yaml").read_text()
    )
    assert result["status"] == "passed"
    assert result["sealed_inputs"]["contract_sha256"] == sha256(contract_path)
    assert result["blocking_gates"]["exact_all_vs_current_online_max_abs_logit"] == 0.0
    assert result["blocking_gates"]["r0_noop_JS_max"] == 0.0
    assert result["adjudication"]["expanded_materialized_lineage_matrix"] == "authorized"
    assert result["adjudication"]["scheduler"] == "prohibited"


def test_p9_5_matrix_covers_all_cells_and_only_gpu01() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/contracts/p9_5_rolling_validation_matrix_v1.yaml").read_text()
    )
    assert contract["scope"]["cells"] == 24
    assert contract["scope"]["evidence_level"] == "expanded_development_validation_not_full_population"
    assert contract["adjudication"]["GPU_allowlist"] == [0, 1]
    assert contract["adjudication"]["scheduler"] == "prohibited"

    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_p9_rolling_validation_queue as queue
    assert len(queue.jobs()) == 24
    source = (ROOT / "scripts/run_p9_rolling_validation_queue.py").read_text()
    assert "for device in (0, 1)" in source
    assert "cuda:2" not in source and "cuda:3" not in source


def test_p9_6_cost_contract_is_state_keyed_and_scheduler_closed() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/contracts/p9_6_transition_cost_contract_v1.yaml").read_text()
    )
    assert contract["logical_population"]["unit"] == "unique_user_state_at_cutover"
    assert contract["logical_population"]["duplicate_queries_per_user_count_once"] is True
    assert contract["runtime_canary"]["batch_size"] == 16
    assert contract["interpretation"]["formal_frontier"] == "pending_full_population_fidelity_join"
    assert contract["interpretation"]["scheduler"] == "prohibited"


def test_p9_7_separates_all_cutover_states_from_future_request_evaluation() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/contracts/p9_7_full_population_contract_v1.yaml").read_text()
    )
    primary = contract["populations"]["migration_primary"]
    heldout = contract["populations"]["heldout_request_evaluation"]
    assert primary["require_future_request"] is False
    assert primary["require_future_activity"] is False
    assert primary["require_future_label"] is False
    assert heldout["may_define_migration_population"] is False
    assert heldout["may_supply_policy_features"] is False
    audit = __import__("json").loads(
        (ROOT / "results/p9/p9_7_full_population_audit_v1.json").read_text()
    )
    assert audit["status"] == "P9_7_full_population_and_probe_audit_passed"
    assert [row["states"] for row in audit["edges"]] == [8229, 8488]
    assert all(row["probe_audit"]["future_or_label_fields"] == [] for row in audit["edges"])


def test_p9_7_uid_executor_migrates_once_and_queries_do_not_mutate_state() -> None:
    result = __import__("json").loads(
        (ROOT / "results/p9/p9_7_uid_executor_canary_v1.json").read_text()
    )
    assert result["status"] == "passed"
    assert result["migration_invocations"] == result["expected_migration_invocations"]
    assert result["max_shared_vs_independent_abs_logit"] == 0.0
    assert result["max_exact_action_vs_current_online_abs_logit"] == 0.0
    assert result["max_query_state_mutation"] == 0.0
    assert result["scheduler_authorized"] is False


def test_p9_8_profiler_is_all_state_label_free_and_gpu01_only() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/contracts/p9_8_cutover_profiler_contract_v1.yaml").read_text()
    )
    assert contract["scope"]["cells"] == 24
    assert contract["scope"]["population"] == "all_materialized_states_at_cutover"
    assert contract["scope"]["labels_or_future_requests_read"] is False
    assert contract["execution"]["GPU_allowlist"] == [0, 1]
    assert contract["evidence_boundary"]["scheduler"] == "prohibited"
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_p9_cutover_profiler_queue as queue
    assert len(queue.jobs()) == 24


def test_p9_9_quality_contract_uses_sealed_requests_and_true_uid_lineage() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/contracts/p9_9_heldout_rolling_quality_contract_v1.yaml").read_text()
    )
    assert contract["scope"]["cells"] == 24
    assert contract["scope"]["requests"] == "all_frozen_request_ids_in_sealed_P8_F_quality_artifact"
    assert contract["scope"]["cold_start_without_cutover_state"] == "reported_and_excluded_from_migration_action_frontier"
    assert contract["lineage"]["transition_once_per_uid_per_action"] is True
    assert contract["lineage"]["reuse_transitioned_state_across_all_uid_queries"] is True
    assert contract["lineage"]["evict_before_each_append_at_context_cap"] == 512
    assert contract["execution"]["GPU_allowlist"] == [0, 1]
    assert contract["evidence_boundary"]["scheduler"] == "prohibited"


def test_p9_9_batched_canary_is_numerically_equivalent_and_r0_is_zero() -> None:
    result = yaml.safe_load(
        (ROOT / "configs/contracts/p9_9_heldout_rolling_quality_canary_result_v1.yaml").read_text()
    )
    assert result["status"] == "passed"
    assert result["contract_sha256"] == sha256(
        ROOT / "configs/contracts/p9_9_heldout_rolling_quality_contract_v1.yaml"
    )
    assert result["evaluator_sha256"] == sha256(
        ROOT / "scripts/eval_p9_heldout_rolling_quality_raw.py"
    )
    assert result["cells"]["r0_m0_f_seed17"]["r0_all_action_JS_max"] < 1e-8
    assert result["batching_equivalence"]["max_abs_action_logit_delta"] < result["batching_equivalence"]["tolerance"]
    assert result["authorization"]["full_24_cell_raw_execution"] is True
    assert result["authorization"]["scheduler"] is False


def test_p9_10_runtime_contract_measures_full_state_cost_without_storage_claim() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/contracts/p9_10_full_population_runtime_contract_v1.yaml").read_text()
    )
    assert contract["scope"]["population"] == "every_materialized_state_at_cutover"
    assert len(contract["scope"]["benchmark_conditions"]) == 3
    assert contract["batching"]["exact_effective_prefix_length_groups"] is True
    assert contract["batching"]["GPU_allowlist"] == [0, 1]
    assert contract["timing_boundaries"]["PCIe_proxy"]["storage_KV_IO"] == "logical_bytes_only_not_claimed_as_measured_storage"
    assert contract["evidence_boundary"]["scheduler"] == "prohibited"
    result = yaml.safe_load(
        (ROOT / "configs/contracts/p9_10_full_population_runtime_canary_result_v1.yaml").read_text()
    )
    assert result["status"] == "passed"
    assert result["contract_sha256"] == sha256(
        ROOT / "configs/contracts/p9_10_full_population_runtime_contract_v1.yaml"
    )
    assert result["evaluator_sha256"] == sha256(
        ROOT / "scripts/eval_p9_full_population_runtime.py"
    )
    assert result["coverage"]["mixed_prefix_lengths"] is True
    assert result["authorization"]["full_three_condition_runtime"] is True
    assert result["authorization"]["scheduler"] is False


def test_p9_11_frontier_keeps_oracle_separate_from_deployable_scheduler() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/contracts/p9_11_frontier_contract_v1.yaml").read_text()
    )
    assert contract["cost_axes"]["primary_allocation_axis"] == "exact_equivalent_recomputed_token_layers"
    assert contract["cost_axes"]["posthoc_combined_scalar"] == "prohibited"
    assert contract["policies"]["near_optimal_state_action"]["exact_solver_claim"] is False
    assert contract["quality_join"]["labels_used_for_policy_selection"] is False
    assert contract["adjudication"]["scheduler_authorized_by_this_step"] is False
    result = __import__("json").loads(
        (ROOT / "results/p9/p9_11_frontier_v1.json").read_text()
    )
    assert result["status"] == "P9_11_uniform_and_offline_state_action_frontiers_adjudicated"
    assert result["state_level_policy_is_offline_oracle"] is True
    assert result["quality_labels_used_for_policy_selection"] is False
    assert result["scheduler_authorized"] is False
