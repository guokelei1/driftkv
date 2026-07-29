import json
from pathlib import Path

import pytest

from hstu_kvcache.migration import (
    D2ActionPlan,
    D2WavePlan,
    D2WaveReport,
    build_d2_phase_ledger,
    build_d2_record_owner_map,
    characterize_d2_capacity,
    characterize_d2_requests,
    d2_record_owner_map_sha256,
    export_stage49_h12_action_plan,
)

ROOT = Path(__file__).resolve().parents[1]
H12 = (
    ROOT
    / "results/system/cohortkv_single_config_full_chain_v1"
    / "stage4_9_staggered_renewal_h12_seed0.json"
)


@pytest.fixture
def real_plan() -> D2ActionPlan:
    if not H12.is_file():
        pytest.skip("local frozen H12 artifact is unavailable")
    return export_stage49_h12_action_plan(
        H12.relative_to(ROOT),
        step_index=1,
    )


def test_real_h12_action_plan_counts_hashes_and_ledger(
    real_plan: D2ActionPlan,
    tmp_path: Path,
) -> None:
    assert real_plan.source_version == "theta1"
    assert real_plan.target_version == "theta2"
    assert real_plan.counts.to_dict() == {
        "compiled": 548,
        "scheduled_exact": 46,
        "natural_exact": 88,
        "records": 682,
    }
    output = tmp_path / "plan.json"
    real_plan.write(output)
    loaded = D2ActionPlan.load(output)
    assert loaded == real_plan
    assert loaded.content_sha256 == real_plan.content_sha256
    ledger = build_d2_phase_ledger(real_plan, embedding_dim=512)
    assert ledger.boundaries["retained_prefix"] == {
        "mixed_lookup_tokens": 50099,
        "all_exact_lookup_tokens": 637954,
        "lookup_reduction": pytest.approx(12.733878919738916),
    }
    assert ledger.boundaries["integrated_post_append"] == {
        "mixed_lookup_tokens": 347062,
        "all_exact_lookup_tokens": 934917,
        "lookup_reduction": pytest.approx(2.693804450413468),
    }
    assert ledger.boundaries["method_independent_append"] == {
        "delta_lookup_tokens": 213669,
        "latest_lookup_tokens": 682,
        "lookup_tokens": 214351,
    }
    compiled = next(
        value
        for value in ledger.mixed
        if value.phase == "compiled_retained"
    )
    assert compiled.compute_tokens == 587855
    assert compiled.lookup_tokens == 0
    assert compiled.physical_collective_bytes is None


def test_single_rank_adapter_preserves_actions_and_stage5_semantics(
    real_plan: D2ActionPlan,
) -> None:
    wave = D2WavePlan.single_rank(real_plan, "stage-a-test")
    requests = wave.to_stage5_requests()
    assert len(requests) == 682
    assert {value.record_id for value in requests} == set(range(682))
    assert sum(value.requested_action == "migrate" for value in requests) == 548
    assert sum(value.requested_action == "exact" for value in requests) == 134
    assert all(
        value.retained_tokens > 0
        for value in requests
        if value.requested_action == "migrate"
    )
    assert all(
        value.retained_tokens == 0
        for value in requests
        if value.requested_reason == "natural_exact"
    )
    ledger = build_d2_phase_ledger(real_plan, embedding_dim=512)
    report = D2WaveReport.from_single_rank_adapter(
        wave,
        ledger.to_dict(),
    )
    assert report.status == "stage_a_adapter_validated"
    assert not report.scientific_result
    assert len(report.coverage_record_ids) == 682


def test_action_plan_rejects_content_tampering(
    real_plan: D2ActionPlan,
) -> None:
    value = real_plan.to_dict()
    value["records"][0]["delta_tokens"] += 1
    with pytest.raises(ValueError):
        D2ActionPlan.from_dict(value)
    value = real_plan.to_dict()
    value["content_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        D2ActionPlan.from_dict(value)


def test_request_characterization_reports_only_static_ceilings(
    real_plan: D2ActionPlan,
) -> None:
    target_items = {
        record.record_id: tuple(
            index % 17
            for index in range(record.final_tokens)
        )
        for record in real_plan.records
    }
    result = characterize_d2_requests(
        real_plan,
        target_items,
        embedding_dim=512,
    )
    assert (
        result["mixed"]["full_wave"]["requested_ids"]
        == 347062
    )
    assert (
        result["all_exact"]["full_wave"]["requested_ids"]
        == 934917
    )
    assert result["scope"] == {
        "request_multiset_known_before_wave": True,
        "unique_counts_are_global_static_ceilings": True,
        "actual_remote_fraction_measured": False,
        "actual_collective_bytes_measured": False,
        "transport_dtype_selected": False,
    }


def test_action_plan_loader_rejects_noncanonical_hash(
    real_plan: D2ActionPlan,
    tmp_path: Path,
) -> None:
    value = real_plan.to_dict()
    value["policy"] = "changed"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        D2ActionPlan.load(path)


def test_owner_maps_and_capacity_are_deterministic(
    real_plan: D2ActionPlan,
) -> None:
    modulo = build_d2_record_owner_map(
        real_plan,
        4,
        "modulo",
    )
    old_lpt = build_d2_record_owner_map(
        real_plan,
        4,
        "old_kv_lpt",
    )
    assert modulo == build_d2_record_owner_map(
        real_plan,
        4,
        "modulo",
    )
    assert len(d2_record_owner_map_sha256(modulo)) == 64
    assert len(d2_record_owner_map_sha256(old_lpt)) == 64
    capacity = characterize_d2_capacity(
        real_plan,
        model_bytes=724328448,
        item_embedding_bytes=639272960,
        program_bytes=33587200,
        capacity_bytes=47699722240,
    )
    one = next(
        value
        for value in capacity["layouts"]
        if value["world_size"] == 1
        and value["owner_strategy"] == "modulo"
    )
    two = next(
        value
        for value in capacity["layouts"]
        if value["world_size"] == 2
        and value["owner_strategy"] == "modulo"
    )
    four_lpt = next(
        value
        for value in capacity["layouts"]
        if value["world_size"] == 4
        and value["owner_strategy"] == "old_kv_lpt"
    )
    assert capacity["cohort"]["old_kv_bytes"] == 28383969280
    assert capacity["cohort"]["complete_new_kv_bytes"] == 30635360256
    assert not one["all_full_model_total_capacity_admitted"]
    assert two["all_full_model_total_capacity_admitted"]
    assert four_lpt["maximum_strict_cow_kv_bytes"] == 14807465984
    assert capacity["scope"]["sharded_embedding_dense_is_estimated"]
