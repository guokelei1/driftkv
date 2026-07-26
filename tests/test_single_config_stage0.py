from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "freeze_cohortkv_single_config_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "freeze_cohortkv_single_config_v1",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
EXPECTED_RECORDS = MODULE.EXPECTED_RECORDS
EXPECTED_PREFIX_TOKENS = MODULE.EXPECTED_PREFIX_TOKENS
FAILURE_INJECTIONS = MODULE.FAILURE_INJECTIONS
PROTOCOL = MODULE.PROTOCOL
build_result_schema = MODULE.build_result_schema
fixed_count_assignment = MODULE.fixed_count_assignment
split_user_roles = MODULE.split_user_roles


def test_role_split_is_complete_disjoint_and_frozen() -> None:
    roles = split_user_roles(list(range(EXPECTED_RECORDS)))
    assert len(roles) == EXPECTED_RECORDS
    counts = {
        role: sum(value == role for value in roles.values())
        for role in set(roles.values())
    }
    assert counts == {
        "fit": 40,
        "program_selection": 60,
        "certificate": 60,
        "final_test": 522,
    }
    assert roles == split_user_roles(list(range(EXPECTED_RECORDS)))


def test_controlled_source_assignment_has_exact_counts() -> None:
    assignments, counts = fixed_count_assignment(EXPECTED_RECORDS)
    assert counts == {0: 136, 4: 205, 10: 341}
    assert len(assignments) == EXPECTED_RECORDS
    assert {version: assignments.count(version) for version in counts} == counts
    assert assignments == fixed_count_assignment(EXPECTED_RECORDS)[0]


def test_integrated_result_schema_freezes_primary_matrix() -> None:
    workload_hash = "a" * 64
    schema = build_result_schema(workload_hash)
    assert schema["properties"]["protocol"]["const"] == PROTOCOL
    assert schema["properties"]["workload_content_sha256"]["const"] == workload_hash
    runs = schema["properties"]["rq4_system"]["properties"]["runs"]
    assert runs["minItems"] == 18
    required_points = {
        (
            clause["contains"]["properties"]["method"]["const"],
            clause["contains"]["properties"]["destination"]["const"],
            clause["contains"]["properties"]["gpu_count"]["const"],
        )
        for clause in runs["allOf"]
    }
    assert required_points == {
        (method, destination, gpu_count)
        for method in MODULE.PRIMARY_METHODS
        for destination in MODULE.DESTINATIONS
        for gpu_count in MODULE.GPU_COUNTS
    }
    assert all(clause["maxContains"] == 1 for clause in runs["allOf"])
    run = runs["items"]["properties"]
    assert run["destination"]["enum"] == ["hbm", "dram"]
    assert run["gpu_count"]["enum"] == [1, 2, 4]
    assert run["prefix_tokens"]["const"] == EXPECTED_PREFIX_TOKENS
    assert (
        run["logical_output_bytes"]["const"]
        == MODULE.EXPECTED_LOGICAL_TARGET_BYTES_FP16
    )
    assert run["manifest"]["properties"]["record_count"]["const"] == EXPECTED_RECORDS
    source_requirements = {
        clause["if"]["properties"]["method"]["const"]: clause["then"][
            "properties"
        ]["source_representations"]["const"]
        for clause in runs["items"]["allOf"]
        if "method" in clause["if"]["properties"]
    }
    assert source_requirements["compiled"] == ["normalized_capsule_fp16"]
    assert source_requirements["exact"] == ["raw_history"]
    assert source_requirements["residual_p"] == [
        "raw_history",
        "residual_hidden_suffix_fp16",
    ]
    gpu_requirements = {
        clause["if"]["properties"]["gpu_count"]["const"]: clause["then"][
            "properties"
        ]["per_gpu"]["minItems"]
        for clause in runs["items"]["allOf"]
        if "gpu_count" in clause["if"]["properties"]
    }
    assert gpu_requirements == {1: 1, 2: 2, 4: 4}
    failures = schema["properties"]["rq4_failures"]["properties"]["injections"]
    assert failures["minItems"] == len(FAILURE_INJECTIONS)
    assert {
        clause["contains"]["properties"]["name"]["const"]
        for clause in failures["allOf"]
    } == set(FAILURE_INJECTIONS)
    assert failures["items"]["properties"]["detected"]["const"] is True
    semantic_condition = next(
        condition
        for condition in failures["items"]["allOf"]
        if condition["if"]["properties"]["name"].get("const")
        == "semantic_theta4_program_perturbation"
    )
    assert (
        semantic_condition["then"]["properties"]["theta4_final_action"][
            "const"
        ]
        == "recompute"
    )
    guard = schema["properties"]["rq4_failures"]["properties"][
        "guard_design"
    ]
    assert guard["properties"]["selection_role"]["const"] == "program_selection"
    assert guard["properties"]["theta4_perturbation_detected"]["const"] is True
    rq3 = schema["properties"]["rq3_frontier"]["properties"]
    assert (
        rq3["selection_points"]["minItems"]
        == MODULE.EXPECTED_FRONTIER_POINTS
        == 177
    )
    assert (
        rq3["selective_grid_audit"]["items"]["properties"][
            "expected_unique_intervals"
        ]["const"]
        == MODULE.EXPECTED_SELECTIVE_INTERVALS
        == 53
    )
    rq5 = schema["properties"]["rq5_economics"]["properties"]
    assert (
        rq5["int8_capsule"]["properties"]["logical_data_bytes"]["const"]
        == MODULE.EXPECTED_LOGICAL_CAPSULE_DATA_BYTES_INT8
    )
    assert (
        rq5["break_even"]["properties"]["conclusion"]["enum"]
        == ["finite_break_even", "no_time_break_even"]
    )


def test_generated_workload_distinguishes_internal_and_raw_user_ids() -> None:
    root = SCRIPT.parents[1]
    workload = json.loads(
        (
            root
            / "configs"
            / "cohortkv_single_config_v1"
            / "workload_manifest.json"
        ).read_text()
    )
    records = workload["records"]
    assert [record["record_id"] for record in records] == list(
        range(EXPECTED_RECORDS)
    )
    assert [record["user_id"] for record in records] == sorted(
        record["user_id"] for record in records
    )
    assert all("raw_user_id" in record for record in records)
    assert any(
        record["raw_user_id"] != record["user_id"] for record in records
    )
    assert workload["summary"]["prefix_tokens"] == EXPECTED_PREFIX_TOKENS
    content_hash = workload.pop("content_sha256")
    assert MODULE.sha256_bytes(MODULE.canonical_json_bytes(workload)) == content_hash


def test_residual_action_declares_full_hidden_suffix_state() -> None:
    root = SCRIPT.parents[1]
    blueprint = json.loads(
        (
            root
            / "configs"
            / "cohortkv_single_config_v1"
            / "blueprint.json"
        ).read_text()
    )
    residual = blueprint["action_contracts"]["residual_p"]
    assert residual["inputs"] == [
        "raw_history",
        "residual_hidden_suffix_fp16",
    ]
    representation = blueprint["source_contract"]["representations"][
        "residual_hidden_suffix_fp16"
    ]
    assert representation["not_part_of_default_normalized_capsule"] is True
    selective = blueprint["action_contracts"]["selective_contiguous"]
    assert (
        selective["implementation_guard"][
            "existing_migrate_contiguous_cache_is_compatible"
        ]
        is False
    )
