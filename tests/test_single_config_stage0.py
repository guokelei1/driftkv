from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest

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
validate_stage5_closure_semantics = MODULE.validate_stage5_closure_semantics


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
        if "source_representations" in clause["then"]["properties"]
    }
    assert source_requirements["compiled"] == ["normalized_capsule_fp16"]
    assert source_requirements["selective_contiguous"] == [
        "old_kv_fp16",
        "raw_history",
    ]
    assert source_requirements["exact"] == ["raw_history"]
    assert source_requirements["residual_p"] == [
        "raw_history",
        "residual_hidden_suffix_bf16",
    ]
    gpu_requirements = {
        clause["if"]["properties"]["gpu_count"]["const"]: clause["then"][
            "properties"
        ]["per_gpu"]["minItems"]
        for clause in runs["items"]["allOf"]
        if "gpu_count" in clause["if"]["properties"]
    }
    assert gpu_requirements == {1: 1, 2: 2, 4: 4}
    assert set(FAILURE_INJECTIONS) == {
        "semantic_theta0_theta1_program_perturbation",
        "mid_job",
        "pre_commit",
    }
    closure = schema["properties"]["stage5_closure"]["properties"]
    assert (
        closure["canary_artifact"]["properties"]["protocol"]["const"]
        == "cohortkv_single_config_stage5_formal_canary_v1"
    )
    failures = closure["abort_jobs"]
    assert failures["minItems"] == failures["maxItems"] == 2
    assert {
        clause["contains"]["properties"]["fault"]["const"]
        for clause in failures["allOf"]
    } == {"mid_job", "pre_commit"}
    assert failures["items"]["properties"]["outcome"]["const"] == "aborted"
    assert failures["items"]["properties"]["target_visible"]["const"] is False
    assert (
        failures["items"]["properties"]["old_readback"]["properties"][
            "passed"
        ]["const"]
        is True
    )
    readback = failures["items"]["properties"]["old_readback"]["properties"]
    assert readback["expected_records"]["const"] == EXPECTED_RECORDS
    assert readback["read_records"]["const"] == EXPECTED_RECORDS
    assert (
        closure["normal_job"]["allOf"][0]["properties"]["outcome"]["const"]
        == "committed"
    )
    normal = closure["normal_job"]["allOf"][0]["properties"]
    assert normal["target_manifest"]["required"] == [
        "protocol",
        "commit_hook",
        "lineage_sha256",
        "destination_manifest",
        "lineage",
    ]
    semantic = closure["semantic_fallback_job"]["allOf"][0]["properties"]
    assert semantic["semantic_perturbation_detected"]["const"] is True
    assert semantic["affected_cohort_final_action"]["const"] == "exact"
    preflight = closure["normal_job"]["properties"]["preflight"][
        "properties"
    ]
    assert preflight["selection_role"]["const"] == "program_selection"
    assert (
        preflight["guard_hook"]["const"]
        == "post_retained_prefix_pre_append"
    )
    assert preflight["decisions"]["minItems"] == EXPECTED_RECORDS
    assert preflight["decisions"]["maxItems"] == EXPECTED_RECORDS
    cohort = preflight["cohorts"]["items"]["properties"]
    assert set(cohort["checks"]["required"]) == {
        "artifact_identity",
        "program_identity",
        "program_shape",
        "old_kv_presence",
        "capacity",
        "semantic_canary",
    }
    cow = closure["copy_on_write_capacity"]
    assert cow["properties"]["all_devices_passed"]["const"] is True
    assert (
        cow["properties"]["old_extents_retained_until_commit"]["const"]
        is True
    )
    assert "rq4_failures" not in schema["required"]
    assert "rq5_economics" not in schema["required"]
    deployed = schema["properties"]["rq2_compiler"]["properties"][
        "cohorts"
    ]["items"]["properties"]["deployed_representation_certificate"]
    assert deployed["properties"]["source_dtype"]["const"] == "float16"
    assert deployed["properties"]["program_dtype"]["const"] == "float16"
    assert deployed["properties"]["output_dtype"]["const"] == "float16"
    assert deployed["properties"]["residual_hidden_suffix_dtype"]["const"] == (
        "bfloat16"
    )
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
    profiled = rq3["profiled_selective_actions"]
    assert profiled["minItems"] == profiled["maxItems"] == 3
    assert (
        profiled["items"]["properties"]["action"]["const"]
        == MODULE.STAGE1_PROFILED_SELECTIVE_ACTION
    )
    assert (
        profiled["items"]["properties"]["certificate_passed"]["const"]
        is False
    )
    accounting = schema["properties"]["source_state_accounting"][
        "properties"
    ]
    for name, contract in MODULE.STAGE5_ACCOUNTING_FROZEN_INPUTS.items():
        frozen = accounting["inputs"]["properties"][name]["properties"]
        assert frozen["protocol"]["const"] == contract["protocol"]
        assert frozen["status"]["const"] == contract["status"]
        assert frozen["sha256"]["const"] == contract["sha256"]
    assert (
        accounting["active_direct_oldkv"]["properties"][
            "additional_per_record_source_state_bytes"
        ]["const"]
        == 0
    )
    assert (
        accounting["rejected_fp16_normalized_capsule"]["properties"][
            "beats_paired_exact_points"
        ]["const"]
        == 0
    )
    assert (
        accounting["claim_boundary"]["properties"]["int8_claim"]["const"]
        is False
    )
    assert (
        accounting["claim_boundary"]["properties"][
            "physical_ssd_performance_claim"
        ]["const"]
        is False
    )


def _stage5_validator_fixture() -> dict:
    formal_canary = MODULE.load_stage5_formal_canary_contract()
    records_per_device = EXPECTED_RECORDS // 2
    tokens_per_device = records_per_device * 2
    payload_per_device = (
        2 * 16 * tokens_per_device * 512 * 2
        + records_per_device * 8
        + (records_per_device + 1) * 8
    )
    capacity = {
        "device": "cuda:0",
        "model_and_program_bytes": 10,
        "old_kv_bytes": 20,
        "complete_new_kv_bytes": payload_per_device,
        "transient_bytes": 5,
        "allocator_margin_bytes": 5,
        "required_bytes": payload_per_device + 40,
        "capacity_bytes": payload_per_device + 100,
        "passed": True,
    }
    capacities = [
        capacity,
        {**capacity, "device": "cuda:1"},
    ]
    normal_decisions = [
        {
            "record_id": record_id,
            "cohort_id": "theta0-cohort",
            "requested_action": "exact",
            "requested_reason": "scheduled_exact",
            "final_action": "exact",
            "fallback_reason": None,
            "source_version": "theta0",
            "target_version": "theta1",
            "last_exact_version_before": "theta0",
            "last_exact_version_after": "theta1",
            "migration_depth_before": 0,
            "migration_depth_after": 0,
            "state_kind_after": "exact",
            "retained_tokens": 1,
            "final_tokens": 2,
        }
        for record_id in range(EXPECTED_RECORDS)
    ]
    normal_decisions[0] = {
        "record_id": 0,
        "cohort_id": "theta0-cohort",
        "requested_action": "migrate",
        "requested_reason": "scheduler_migrate",
        "final_action": "migrate",
        "fallback_reason": None,
        "source_version": "theta0",
        "target_version": "theta1",
        "last_exact_version_before": "theta0",
        "last_exact_version_after": "theta0",
        "migration_depth_before": 0,
        "migration_depth_after": 1,
        "state_kind_after": "migrated",
        "retained_tokens": 1,
        "final_tokens": 2,
    }
    semantic_decisions = [dict(value) for value in normal_decisions]
    semantic_decisions[0] = {
        "record_id": 0,
        "cohort_id": "theta0-cohort",
        "requested_action": "migrate",
        "requested_reason": "scheduler_migrate",
        "final_action": "exact",
        "fallback_reason": "semantic_canary",
        "source_version": "theta0",
        "target_version": "theta1",
        "last_exact_version_before": "theta0",
        "last_exact_version_after": "theta1",
        "migration_depth_before": 0,
        "migration_depth_after": 0,
        "state_kind_after": "exact",
        "retained_tokens": 1,
        "final_tokens": 2,
    }

    threshold_sha256 = formal_canary["sha256"]
    normal_program_sha256 = formal_canary[
        "canonical_program_sha256"
    ]
    perturbed_program_sha256 = formal_canary[
        "perturbed_program_sha256"
    ]

    def preflight(decisions, passed, program_sha256):
        checks = {
            "artifact_identity": True,
            "program_identity": True,
            "program_shape": True,
            "old_kv_presence": True,
            "capacity": True,
            "semantic_canary": passed,
        }
        maximum = float(formal_canary["maximum_relative_l2"])
        observed = maximum / 2.0 if passed else maximum * 1.5
        return {
            "protocol": "cohortkv_stage5_fixed_preflight_v1",
            "selection_role": "program_selection",
            "labels_used": False,
            "guard_hook": "post_retained_prefix_pre_append",
            "elapsed_seconds": 0.0,
            "input_measurement_seconds": 0.0,
            "runtime_validation_seconds": 0.0,
            "decision_seconds": 0.0,
            "all_cohorts_passed": passed,
            "cohorts": [
                {
                    "cohort_id": "theta0-cohort",
                    "source_version": "theta0",
                    "target_version": "theta1",
                    "passed": passed,
                    "checks": checks,
                    "fallback_reason": (
                        None if passed else "semantic_canary"
                    ),
                    "migration_required": True,
                    "expected_artifact_sha256": formal_canary[
                        "edge_artifact_sha256"
                    ],
                    "observed_artifact_sha256": formal_canary[
                        "edge_artifact_sha256"
                    ],
                    "expected_program_sha256": program_sha256,
                    "observed_program_sha256": program_sha256,
                    "expected_program_shape": [1, 2, 3],
                    "observed_program_shape": [1, 2, 3],
                    "expected_threshold_artifact_sha256": (
                        threshold_sha256
                    ),
                    "expected_old_record_ids": [0],
                    "present_old_record_ids": [0],
                    "expected_old_records_source": (
                        "prior_committed_manifest"
                    ),
                    "present_old_records_source": (
                        "destination_readback"
                    ),
                    "canary": {
                        "cohort_id": "theta0-cohort",
                        "record_ids": [0],
                        "source_version": "theta0",
                        "target_version": "theta1",
                        "program_sha256": program_sha256,
                        "metric": "kv_relative_l2",
                        "observed_relative_l2": observed,
                        "maximum_relative_l2": maximum,
                        "candidate_sha256": "e" * 64,
                        "reference_sha256": "f" * 64,
                        "threshold_artifact_sha256": threshold_sha256,
                        "threshold_source": "program_selection",
                        "labels_used": False,
                        "passed": passed,
                    },
                    "device_capacity": capacities,
                    "measurement": {
                        "artifact_seconds": 0.0,
                        "old_kv_presence_seconds": 0.0,
                        "capacity_seconds": 0.0,
                        "semantic_canary_seconds": 0.0,
                        "total_seconds": 0.0,
                    },
                }
            ],
            "decisions": decisions,
        }

    def target(decisions, job_id):
        metadata = {
            "protocol": "cohortkv_single_config_stage5_minimal_closure_v1",
            "commit_hook": "post_append_full_cache",
            "lineage": decisions,
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        digest = MODULE.sha256_bytes(encoded.encode())
        extents = []
        for index, start in enumerate(
            range(0, EXPECTED_RECORDS, records_per_device)
        ):
            record_ids = list(
                range(
                    start,
                    min(start + records_per_device, EXPECTED_RECORDS),
                )
            )
            tokens = sum(
                int(decisions[record_id]["final_tokens"])
                for record_id in record_ids
            )
            payload = (
                2 * 16 * tokens * 512 * 2
                + len(record_ids) * 8
                + (len(record_ids) + 1) * 8
            )
            extents.append(
                {
                    "extent_id": f"extent-{index:08d}",
                    "record_ids": record_ids,
                    "migration_anchor_version": "theta1",
                    "served_kv_target": "theta1",
                    "num_layers": 16,
                    "token_count": tokens,
                    "kv_width": 512,
                    "dtype": "float16",
                    "payload_bytes": payload,
                    "location": (
                        "hbm://stage5-fixture/"
                        f"cuda:{index}/target/extent-{index:08d}"
                    ),
                    "device": f"cuda:{index}",
                    "checksum_sha256": None,
                }
            )
        return {
            "protocol": (
                "cohortkv_single_config_stage5_minimal_closure_v1"
            ),
            "commit_hook": "post_append_full_cache",
            "lineage": decisions,
            "lineage_sha256": digest,
            "destination_manifest": {
                "protocol": "streamkv_destination_manifest_v1",
                "job_id": job_id,
                "target_version": "theta1",
                "destination_id": "stage5-fixture",
                "destination_kind": "hbm",
                "publication_mode": "direct_device",
                "record_count": EXPECTED_RECORDS,
                "token_count": sum(
                    int(item["token_count"]) for item in extents
                ),
                "payload_bytes": sum(
                    int(item["payload_bytes"]) for item in extents
                ),
                "metadata": metadata,
                "metadata_sha256": digest,
                "extents": extents,
            },
        }

    def readback(version):
        return {
            "protocol": "cohortkv_stage5_manifest_readback_v1",
            "target_version": version,
            "expected_records": EXPECTED_RECORDS,
            "read_records": EXPECTED_RECORDS,
            "manifest_equal": True,
            "all_metadata_equal": True,
            "all_finite": True,
            "all_checksums_equal": True,
            "passed": True,
            "elapsed_seconds": 0.0,
        }

    def committed(decisions, passed, job_id, program_sha256):
        return {
            "protocol": (
                "cohortkv_single_config_stage5_minimal_closure_v1"
            ),
            "job_id": job_id,
            "target_version": "theta1",
            "outcome": "committed",
            "fault": None,
            "preflight": preflight(
                decisions,
                passed,
                program_sha256,
            ),
            "guard_invocations": 2,
            "staged_extents": 2,
            "target_manifest": target(decisions, job_id),
            "target_visible": True,
            "partial_target_visible": False,
            "staging_reclaimed": True,
            "old_readback": None,
            "target_readback": readback("theta1"),
            "elapsed_seconds": 0.0,
        }

    def aborted(fault):
        return {
            "protocol": (
                "cohortkv_single_config_stage5_minimal_closure_v1"
            ),
            "job_id": f"{fault}-job",
            "target_version": "theta1",
            "outcome": "aborted",
            "fault": fault,
            "preflight": preflight(
                normal_decisions,
                True,
                normal_program_sha256,
            ),
            "guard_invocations": 2,
            "staged_extents": 2,
            "target_manifest": None,
            "target_visible": False,
            "partial_target_visible": False,
            "staging_reclaimed": True,
            "old_readback": readback("theta0"),
            "target_readback": None,
            "elapsed_seconds": 0.0,
        }

    semantic = committed(
        semantic_decisions,
        False,
        "semantic-job",
        perturbed_program_sha256,
    )
    semantic["semantic_perturbation_detected"] = True
    semantic["affected_cohort_final_action"] = "exact"
    return {
        "protocol": "cohortkv_single_config_stage5_minimal_closure_v1",
        "canary_artifact": {
            name: formal_canary[name]
            for name in (
                "path",
                "sha256",
                "protocol",
                "source_version",
                "target_version",
                "selection_role",
                "labels_used",
                "metric",
                "maximum_relative_l2",
            )
        },
        "copy_on_write_gpu_count": 2,
        "copy_on_write_capacity": {
            "mode": "copy_on_write",
            "old_extents_retained_until_commit": True,
            "all_devices_passed": True,
            "devices": capacities,
            "observed_free_capacity": {
                "measurement_boundary": (
                    "target models and direct programs resident; before "
                    "per-case old-cache publication"
                ),
                "all_devices_passed": True,
                "devices": [
                    {
                        "device": value["device"],
                        "free_bytes": payload_per_device + 100,
                        "required_free_bytes": (
                            payload_per_device + 30
                        ),
                        "passed": True,
                    }
                    for value in capacities
                ],
            },
        },
        "normal_job": committed(
            normal_decisions,
            True,
            "normal-job",
            normal_program_sha256,
        ),
        "semantic_fallback_job": semantic,
        "abort_jobs": [
            aborted("mid_job"),
            aborted("pre_commit"),
        ],
    }


def test_stage5_semantic_validator_rejects_false_capacity_or_readback() -> None:
    value = _stage5_validator_fixture()
    validate_stage5_closure_semantics(value)

    value["copy_on_write_capacity"]["devices"][0]["required_bytes"] += 1
    with pytest.raises(ValueError, match="capacity arithmetic"):
        validate_stage5_closure_semantics(value)

    value = _stage5_validator_fixture()
    value["abort_jobs"][0]["old_readback"]["read_records"] -= 1
    with pytest.raises(ValueError, match="readback coverage"):
        validate_stage5_closure_semantics(value)

    value = _stage5_validator_fixture()
    value["canary_artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="canary artifact contract"):
        validate_stage5_closure_semantics(value)

    value = _stage5_validator_fixture()
    decision = value["normal_job"]["preflight"]["decisions"][0]
    decision.update(
        {
            "final_action": "exact",
            "fallback_reason": "semantic_canary",
            "last_exact_version_after": "theta1",
            "migration_depth_after": 0,
            "state_kind_after": "exact",
        }
    )
    with pytest.raises(ValueError, match="final action"):
        validate_stage5_closure_semantics(value)

    value = _stage5_validator_fixture()
    value["normal_job"]["target_manifest"]["destination_manifest"][
        "token_count"
    ] += 1
    with pytest.raises(ValueError, match="manifest totals"):
        validate_stage5_closure_semantics(value)

    value = _stage5_validator_fixture()
    value["copy_on_write_capacity"]["observed_free_capacity"][
        "devices"
    ][0]["free_bytes"] = 1
    with pytest.raises(ValueError, match="free-capacity arithmetic"):
        validate_stage5_closure_semantics(value)


def test_stage5_validator_fixture_matches_generated_json_schema() -> None:
    generated = build_result_schema()
    jsonschema.Draft202012Validator.check_schema(generated)
    schema = generated["properties"]["stage5_closure"]

    jsonschema.validate(
        instance=_stage5_validator_fixture(),
        schema=schema,
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
        "residual_hidden_suffix_bf16",
    ]
    representation = blueprint["source_contract"]["representations"][
        "residual_hidden_suffix_bf16"
    ]
    assert representation["not_part_of_default_normalized_capsule"] is True
    selective = blueprint["action_contracts"]["selective_contiguous"]
    assert (
        selective["implementation_guard"][
            "existing_migrate_contiguous_cache_is_compatible"
        ]
        is False
    )
    observation = selective["stage1_observation"]
    assert observation["profiled_system_action"] == {
        "m": 12,
        "start_layer": 0,
        "end_layer": 11,
    }
    assert observation["profiled_system_action_inputs"] == [
        "old_kv_fp16",
        "raw_history",
    ]
    assert observation["transition_hidden_bytes"] == 0
