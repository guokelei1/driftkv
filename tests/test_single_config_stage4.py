import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "configs"
    / "cohortkv_single_config_v1"
    / "stage4_system_summary.json"
)
METHODS = (
    "compiled",
    "selective_contiguous",
    "exact",
    "residual_p",
    "no_transform",
)
DESTINATIONS = ("hbm", "dram")
GPU_COUNTS = (1, 2, 4)
pytestmark = pytest.mark.skipif(
    not SUMMARY.is_file(),
    reason="Stage 4 frozen summary is not materialized yet",
)


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_stage4_summary_closes_the_full_normal_path_matrix() -> None:
    summary = load_summary()

    assert summary["protocol"] == "cohortkv_single_config_stage4_frozen_v1"
    assert summary["status"] == "stage4_frozen"
    boundary = summary["measurement_boundary"]
    assert boundary["full_points"] == 30
    assert boundary["primary_points"] == 18
    assert boundary["control_points"] == 12
    assert boundary["recommendation_labels_used"] is False
    assert boundary["final_test_quality_evaluated"] is False
    runs = summary["runs"]
    assert len(runs) == 30
    assert {
        (run["method"], run["destination"], run["gpu_count"])
        for run in runs
    } == {
        (method, destination, gpu_count)
        for method in METHODS
        for destination in DESTINATIONS
        for gpu_count in GPU_COUNTS
    }


def test_stage4_tuning_is_complete_and_independent_per_endpoint() -> None:
    points = load_summary()["runtime_tuning"]["points"]

    assert len(points) == 30
    for point in points:
        expected_candidates = (
            54 if point["method"] in {"compiled", "exact"} else 27
        )
        assert point["candidate_count"] == expected_candidates
        assert len(point["finalists"]) == 3
        assert point["winner_id"] in {
            value["candidate_id"] for value in point["finalists"]
        }
        assert point["winner_runtime_config"] == next(
            value["runtime_config"]
            for value in point["finalists"]
            if value["candidate_id"] == point["winner_id"]
        )
        if point["destination"] == "dram":
            assert point["pinned_extent_probe"]["passed"] is True
        else:
            assert point["pinned_extent_probe"] is None


def test_stage4_runs_preserve_bytes_coverage_and_capacity() -> None:
    for run in load_summary()["runs"]:
        assert run["record_count"] == 682
        assert run["prefix_tokens"] == 1_087_785
        assert run["logical_output_bytes"] == 35_644_538_880
        assert run["physical_output_bytes"] >= run["logical_output_bytes"]
        assert run["logical_input_bytes"] == sum(
            value["logical_bytes"] for value in run["input_components"]
        )
        assert run["physical_input_bytes"] == sum(
            value["physical_bytes"] for value in run["input_components"]
        )
        assert len(run["per_gpu"]) == run["gpu_count"]
        assert sum(
            value["record_count"] for value in run["per_gpu"]
        ) == run["record_count"]
        assert sum(
            value["prefix_tokens"] for value in run["per_gpu"]
        ) == run["prefix_tokens"]
        assert sum(
            value["logical_input_bytes"] for value in run["per_gpu"]
        ) == run["logical_input_bytes"]
        assert sum(
            value["logical_output_bytes"] for value in run["per_gpu"]
        ) == run["logical_output_bytes"]
        assert run["capacity_preflight"]["passed"] is True
        assert len(run["capacity_preflight"]["per_job"]) == 5
        assert run["correctness"]["finite"] is True
        assert run["correctness"]["allclose"] is True
        assert run["correctness"]["record_order_valid"] is True
        assert run["correctness"]["lengths_offsets_valid"] is True
        assert run["correctness"]["valid_element_count"] == 17_822_269_440
        assert run["manifest"]["complete"] is True
        assert run["manifest"]["duplicate_free"] is True
        assert len(run["timing"]["samples_seconds"]) == 3
        assert math.isclose(
            run["tokens_per_second"],
            run["prefix_tokens"] / run["timing"]["median_seconds"],
            rel_tol=1e-12,
        )


def test_stage4_preserves_selective_failure_and_records_pareto_pivot() -> None:
    summary = load_summary()
    selective = [
        run for run in summary["runs"] if run["method"] == "selective_contiguous"
    ]

    assert len(selective) == 6
    assert all(run["certificate_passed"] is False for run in selective)
    assert all(run["publishable_sync_action"] is False for run in selective)
    assert all(
        run["action_configuration"]
        == {"m": 12, "start_layer": 0, "end_layer": 11}
        for run in selective
    )
    assert summary["downstream_rule"]["normal_path_closed"] is True
    assert summary["downstream_rule"]["end_to_end_pareto_gate_passed"] is False
    assert summary["derived"]["compiled_beats_exact_points"] == 0
    assert summary["derived"]["compiled_beats_selective_points"] == 6
    assert summary["downstream_rule"]["next_stage"] == (
        "stage4_5_source_state_footprint_optimization"
    )
    assert summary["downstream_rule"]["representative_iteration_points"] == [
        "compiled:hbm:1",
        "compiled:hbm:4",
    ]
    assert "paused" in summary["downstream_rule"]["stage5_status"]


def test_stage4_summary_is_frozen_into_parent_blueprint() -> None:
    blueprint = json.loads(
        (
            ROOT
            / "configs"
            / "cohortkv_single_config_v1"
            / "blueprint.json"
        ).read_text()
    )
    descriptor = blueprint["frozen_inputs"]["stage4_system_summary"]
    source = blueprint["frozen_inputs"]["stage4_source_manifest"]

    assert blueprint["status"] == "stage4_core_frozen"
    assert blueprint["scope"]["completed_stages"] == [0, 1, 2, 3, 4]
    assert descriptor["path"] == (
        "configs/cohortkv_single_config_v1/stage4_system_summary.json"
    )
    assert descriptor["bytes"] == SUMMARY.stat().st_size
    assert sha256_file(ROOT / descriptor["path"]) == descriptor["sha256"]
    assert sha256_file(ROOT / source["path"]) == source["sha256"]
