"""Deployment-cost checks against the retained ledger and boundary arithmetic."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from system_study_2026_10.scale_cost import (INPUT, decompose, evaluate, fixed_cost,
                                            memory_at_scale, read, thresholds)


def branches():
    return read(INPUT)["branches"]


def test_reference_population_reproduces_all_branches_with_source_sensitivity():
    for branch in branches().values():
        retained = branch["protocols"]["stride16_candidates16"]
        parts = decompose(branch, 30000)
        row = evaluate(parts, 30000)
        assert row["added_flops"] == retained["complete_added_flops"]
        assert row["full_flops"] == retained["total_modeled_service_full_flops"]
        assert row["relay_flops"] == retained["total_modeled_service_relay_flops"]
        with_source = evaluate(parts, 30000, with_source=True)
        assert with_source["relay_flops"] - row["relay_flops"] == retained["source_reconstruction_sensitivity_flops"]
        assert with_source["full_flops"] == row["full_flops"]


def test_reported_minimum_populations_are_first_integers_meeting_targets():
    for branch in branches().values():
        parts = decompose(branch, 30000)
        for source in (False, True):
            t = thresholds(parts, with_source=source)
            n = t["positive_net_saving"]["minimum_users"]
            assert evaluate(parts, n, with_source=source)["saved_flops"] > 0
            assert evaluate(parts, n - 1, with_source=source)["saved_flops"] <= 0
            for target in (10, 20, 30):
                n = t["complete_service_saving"][str(target)]["minimum_users"]
                if n is not None:
                    assert evaluate(parts, n, with_source=source)["net_saved_percent"] >= target
                    assert evaluate(parts, n - 1, with_source=source)["net_saved_percent"] < target
            n = t["at_most_20_percent_added_over_rebuild"]["minimum_users"]
            if n is not None:
                assert evaluate(parts, n, with_source=source)["added_percent_of_rebuild"] <= 20
                assert evaluate(parts, n - 1, with_source=source)["added_percent_of_rebuild"] > 20


def test_wider_read_cannot_amortize_away_its_per_request_cost():
    parts = decompose(branches()["Q1_read6plus6"], 30000)
    t = thresholds(parts)
    assert t["complete_service_saving"]["30"]["minimum_users"] is None
    assert t["asymptotic_complete_service_saving_percent"] < 30
    assert t["at_most_20_percent_added_over_rebuild"]["minimum_users"] is None
    assert t["asymptotic_added_over_rebuild_percent"] > 20


def test_common_foundation_training_preserves_absolute_saving_and_dilutes_ratio():
    parts = decompose(branches()["current"], 30000)
    without = evaluate(parts, 30000)
    # Test input only, not a measured or assumed foundation-training budget.
    with_training = evaluate(parts, 30000, common_training=10**15)
    assert with_training["full_flops"] - without["full_flops"] == 10**15
    assert with_training["relay_flops"] - without["relay_flops"] == 10**15
    assert with_training["saved_flops"] == without["saved_flops"]
    assert with_training["net_saved_percent"] < without["net_saved_percent"]
    assert with_training["added_percent_of_rebuild"] == without["added_percent_of_rebuild"]


def test_catalogue_and_replica_growth_change_fixed_preparation_not_user_service():
    parts = decompose(branches()["current"], 30000)
    calibration = parts["fixed_components_flops"]["correction_calibration_including_origin_once"]
    tables, small = parts["catalogue_linear_preparation_flops"], parts["small_matrix_preparation_flops"]
    scaled = evaluate(parts, 30000, catalogue_rows=2 * parts["catalogue_rows"], replicas=8)
    assert scaled["fixed_flops"] == calibration + 4 * tables + 2 * small
    assert fixed_cost(parts) == calibration + tables + small
    old = evaluate(parts, 30000)
    for name in ("variable_flops", "common_native_flops", "once_rebuild_flops"):
        assert scaled[name] == old[name]


def test_memory_counts_one_code_and_shared_tables_per_replica():
    memory = branches()["current"]["memory"]
    base = memory_at_scale(memory, 30000, 30000)
    assert base["native_kv_total_bytes"] == memory["total_population_native_kv_bytes"]
    assert base["code_total_bytes"] == memory["total_population_code_bytes"]
    assert base["shared_bytes_all_replicas"] == 4 * memory["shared_state_bytes_per_replica"]
    assert base["handoff_peak_shared_bytes_per_replica"] == memory["publication_peak_shared_state_bytes_per_producer"]
    assert base["per_replica_state_and_shared_lower_bound_bytes"] > 48 * 10**9
    doubled_users = memory_at_scale(memory, 60000, 30000)
    assert doubled_users["native_and_code_total_bytes"] == 2 * base["native_and_code_total_bytes"]
    assert doubled_users["shared_bytes_per_replica"] == base["shared_bytes_per_replica"]
    more_replicas = memory_at_scale(memory, 30000, 30000, replicas=8)
    assert more_replicas["per_replica_native_kv_bytes"] == base["per_replica_native_kv_bytes"] / 2
    assert more_replicas["shared_bytes_all_replicas"] == 2 * base["shared_bytes_all_replicas"]
    doubled_catalogue = memory_at_scale(memory, 30000, 30000, catalogue_multiplier=2)
    assert doubled_catalogue["shared_bytes_per_replica"] - base["shared_bytes_per_replica"] == (
        memory["active_compiled_item_table_bytes_per_replica"] + memory["immutable_origin_bank_bytes_per_replica"])
