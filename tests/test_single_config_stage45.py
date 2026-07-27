import hashlib
import json
from pathlib import Path

from hstu_kvcache.migration.stage45_oldkv import (
    choose_stage45_source_action,
)

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "configs"
    / "cohortkv_single_config_v1"
    / "stage4_5_source_plan_summary.json"
)


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_stage45_freezes_zero_extra_normx_source_plan() -> None:
    summary = load_summary()

    assert summary["protocol"] == (
        "cohortkv_single_config_stage4_5_frozen_v1"
    )
    assert summary["status"] == "stage4_5_source_plan_frozen"
    assert summary["source_plan"]["normal_action"] == "compiled_old_kv"
    assert summary["source_plan"]["source_representation"] == (
        "existing_old_kv_fp16"
    )
    assert summary["source_plan"]["additional_normx_bytes"] == 0
    assert (
        summary["source_plan"]["additional_per_record_source_state_bytes"]
        == 0
    )
    assert summary["source_plan"]["program_bytes"] == 100_777_103
    assert summary["source_plan"]["fallback_action"] == "exact"
    assert summary["gate"]["extra_normx_eliminated"] is True
    assert summary["gate"]["stable_end_to_end_pareto_point"] is True
    assert summary["gate"]["stage5_admitted"] is True


def test_stage45_preserves_semantics_and_full_real_transport() -> None:
    summary = load_summary()
    certificate = summary["semantic_certificate"]
    transport = summary["full_real_transport"]

    assert certificate["all_selected_direct_oldkv"] is True
    assert certificate["all_exact_fallback"] is True
    assert certificate["minimum_worst_view_recovery"] >= 0.7
    assert certificate["maximum_cost_ratio_to_exact"] <= 0.3
    assert len(certificate["pairs"]) == 3
    assert all(
        pair["selected_action"] == "compiled_old_kv"
        and pair["fallback_actions"] == ["recompute"]
        and pair["worst_recovery_lower_bound"] >= 0.7
        and pair["worst_coverage_lower_bound"] >= 0.8
        for pair in certificate["pairs"]
    )
    assert transport["records"] == 682
    assert transport["prefix_tokens"] == 1_087_785
    assert transport["valid_elements"] == 17_822_269_440
    assert transport["roles"]["final_test"] == 522
    assert transport["recommendation_labels_used"] is False
    assert transport["finite"] is True
    assert transport["allclose"] is True
    assert transport["mismatched_elements"] == 0
    assert transport["maximum_absolute_error"] <= 0.02


def test_stage45_end_to_end_gate_passes_on_one_two_and_four_gpus() -> None:
    summary = load_summary()
    system = summary["system"]
    comparisons = system["comparisons"]

    assert [value["gpu_count"] for value in comparisons] == [1, 2, 4]
    assert system["representative_gate_passed"] is True
    assert system["two_gpu_expansion_passed"] is True
    assert system["all_capacity_preflights_passed"] is True
    assert system["all_outputs_finite_and_allclose"] is True
    assert system["all_manifests_complete"] is True
    for comparison in comparisons:
        assert comparison["compiled_median_seconds"] < (
            comparison["exact_median_seconds"]
        )
        assert comparison["speedup"] > 1
        assert comparison["all_compiled_below_all_exact"] is True
        assert comparison["difference_exceeds_variation_band"] is True
        assert comparison["completion_gate_passed"] is True
        assert comparison["additional_compiled_source_state_bytes"] == 0
        assert comparison["final_old_kv_bytes"] == 0
    for point in system["points"]:
        assert len(point["samples_seconds"]) == 5
        assert point["capacity_preflight"]["passed"] is True
        assert point["correctness"]["allclose"] is True
        assert point["correctness"]["valid_element_count"] == 17_822_269_440
        assert point["manifest"]["record_count"] == 682
        assert point["reclamation"]["final_old_kv_bytes"] == 0


def test_stage45_source_policy_is_monotone_to_exact() -> None:
    direct = choose_stage45_source_action(True, True, True)

    assert direct.action == "compiled_old_kv"
    assert direct.fallback_action == "exact"
    for preflight, old_kv, program in (
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ):
        decision = choose_stage45_source_action(
            preflight,
            old_kv,
            program,
        )
        assert decision.action == "exact"
        assert decision.fallback_action == "exact"


def test_stage45_evidence_descriptors_are_content_addressed() -> None:
    summary = load_summary()

    for value in summary["evidence_artifacts"].values():
        path = ROOT / value["path"]
        assert path.stat().st_size == value["bytes"]
        assert sha256_file(path) == value["sha256"]


def test_stage45_source_plan_is_registered_in_parent_blueprint() -> None:
    blueprint = json.loads(
        (
            ROOT
            / "configs"
            / "cohortkv_single_config_v1"
            / "blueprint.json"
        ).read_text()
    )
    value = blueprint["frozen_inputs"]["stage4_5_source_plan_summary"]

    assert blueprint["scope"]["completed_amendments"] == [
        "stage4_5_source_plan"
    ]
    assert blueprint["source_plan_contract"]["normal_action"] == (
        "compiled_old_kv"
    )
    assert blueprint["source_plan_contract"]["additional_normx_bytes"] == 0
    assert blueprint["source_plan_contract"]["fallback_action"] == "exact"
    assert blueprint["source_plan_contract"]["stage5_admitted"] is True
    assert value["path"] == (
        "configs/cohortkv_single_config_v1/"
        "stage4_5_source_plan_summary.json"
    )
    assert value["bytes"] == SUMMARY.stat().st_size
    assert sha256_file(SUMMARY) == value["sha256"]
