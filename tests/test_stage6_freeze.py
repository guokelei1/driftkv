from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_cohortkv_stage6.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "freeze_cohortkv_stage6",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def frozen_values() -> dict[str, dict]:
    return {
        name: MODULE.load_json(path)
        for name, path in MODULE.FROZEN_CONFIGS.items()
    }


def stage49_values() -> dict[str, dict]:
    return {
        name: MODULE.load_json(path)
        for name, path in MODULE.STAGE49_CANDIDATES.items()
    }


def stage49_descriptors() -> dict[str, dict]:
    return {
        name: MODULE.descriptor(path)
        for name, path in MODULE.STAGE49_CANDIDATES.items()
    }


def test_stage6_schema_requires_lifecycle_and_freeze_closure() -> None:
    schema = MODULE.freeze.build_result_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert {"lifecycle", "stage6_closure"} <= set(schema["required"])
    corrected = schema["properties"]["lifecycle"]["properties"][
        "corrected_growing_history"
    ]["properties"]
    assert corrected["selected_candidate"]["const"] == (
        "staggered_renewal_h12"
    )
    assert corrected["cost_endpoint"]["const"] == "token_debt_total10"
    assert corrected["full_cohort_hbm_claim"]["const"] is False
    assert (
        corrected["end_to_end_state_movement_claim"]["const"] is False
    )


def test_stage6_lifecycle_freezes_h12_and_retains_cost_endpoint() -> None:
    lifecycle = MODULE.build_lifecycle(
        frozen_values(),
        MODULE.load_json(MODULE.STAGE49_RAW),
        stage49_values(),
        stage49_descriptors(),
    )
    corrected = lifecycle["corrected_growing_history"]
    candidates = {
        value["candidate_name"]: value
        for value in corrected["candidates"]
    }
    assert corrected["selected_candidate"] == "staggered_renewal_h12"
    assert corrected["cost_endpoint"] == "token_debt_total10"
    assert candidates["token_debt_total10"][
        "primary_sum_u_over_sum_e"
    ] < candidates["staggered_renewal_h12"][
        "primary_sum_u_over_sum_e"
    ]
    assert candidates["staggered_renewal_h12"][
        "scheduled_exact_records"
    ] > candidates["token_debt_total10"]["scheduled_exact_records"]
    assert corrected["target_append_excluded"] is True
    assert corrected["groupwise_host_staging"] is True
    assert corrected["full_cohort_hbm_claim"] is False
    assert corrected["end_to_end_state_movement_claim"] is False
    assert lifecycle["fixed_history"]["maximum_migration_depth"] == 4


def test_stage6_rq3_augmentation_matches_revised_schema() -> None:
    raw = MODULE.load_json(MODULE.STAGE1_RAW)
    summary = MODULE.load_json(MODULE.FROZEN_CONFIGS["stage1"])
    value = MODULE.build_rq3(raw, summary)
    schema = MODULE.freeze.build_result_schema()["properties"][
        "rq3_frontier"
    ]
    jsonschema.validate(instance=value, schema=schema)
    assert len(value["selection_points"]) == 177
    assert len(value["profiled_selective_actions"]) == 3
    assert all(
        item["action"] == {"m": 12, "start_layer": 0, "end_layer": 11}
        and item["certificate_passed"] is False
        for item in value["profiled_selective_actions"]
    )


def test_stage6_tbd_disposition_is_complete_and_fail_closed() -> None:
    value = MODULE.build_tbd_disposition()
    assert value["markers"] == 0
    assert value["all_markers_disposed"] is True
    assert value["disposition_counts"] == {}
    assert value["entries"] == []


def test_stage6_missing_formal_stage5_fails_closed(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-stage5.json"
    with pytest.raises(FileNotFoundError, match="Stage 6 input"):
        MODULE.build_outputs(missing)


def test_stage6_formal_build_is_deterministic_when_available() -> None:
    path = MODULE.repo_path(MODULE.STAGE5_RAW)
    if not path.is_file():
        pytest.skip("formal Stage 5 artifact is not available")
    first = MODULE.build_outputs()
    second = MODULE.build_outputs()
    assert first == second
    final = json.loads(first[MODULE.FINAL_OUTPUT])
    assert final["stage6_closure"]["selected_candidate"] == (
        "staggered_renewal_h12"
    )
    assert final["stage6_closure"]["old_gpu_matrix_rerun"] is False
    assert final["stage6_closure"]["checks"]["all_passed"] is True
