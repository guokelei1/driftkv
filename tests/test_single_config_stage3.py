import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "configs"
    / "cohortkv_single_config_v1"
    / "stage3_operator_summary.json"
)
SCRIPT = ROOT / "scripts" / "benchmark_cohortkv_stage3_operator.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_cohortkv_stage3_operator",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_stage3_summary_freezes_common_output_boundary() -> None:
    summary = load_summary()

    assert summary["protocol"] == "cohortkv_single_config_stage3_frozen_v1"
    assert summary["status"] == "stage3_frozen"
    assert summary["measurement_boundary"] == {
        "execution": (
            "GPU-resident FP16 capsule and program through complete "
            "preallocated unpadded FP16 K/V extent write"
        ),
        "source_role": "program_selection",
        "source_records": 60,
        "source_valid_tokens": 88085,
        "source_versions": ["theta0", "theta4", "theta10"],
        "operators": [
            "reference_fp32",
            "packed_fp16",
            "fused_fp16",
        ],
        "target_allocation_timed": False,
        "source_io_timed": False,
        "destination_transport_timed": False,
        "final_test_evaluated": False,
        "recommendation_labels_used": False,
        "gpu_name": "NVIDIA A40",
    }
    assert summary["contracts"]["output_extent"]["padding_published"] is False


def test_stage3_full_distribution_correctness_is_frozen() -> None:
    correctness = load_summary()["correctness"]

    assert correctness["layouts"] == 9
    assert correctness["records_per_layout"] == 60
    assert correctness["valid_tokens_per_layout"] == 88085
    assert correctness["valid_fp16_kv_elements_per_layout"] == 1443184640
    assert (
        correctness["total_valid_element_comparisons_per_path"]
        == 12988661760
    )
    assert correctness["all_source_padding_zero"] is True
    assert correctness["all_dense_output_padding_zero"] is True
    assert correctness["all_outputs_finite"] is True
    assert correctness["all_destination_pointers_preserved"] is True
    assert correctness["all_dense_extent_outputs_identical"] is True
    assert correctness["transport_mismatched_elements"] == 0
    assert max(correctness["maximum_absolute_difference"].values()) <= 0.02


def test_stage3_freezes_complete_candidate_grid_and_stable_fused_default() -> None:
    summary = load_summary()
    candidates = summary["candidate_screen"]
    selection = summary["selection"]

    assert len(candidates) == 18
    assert {
        (value["operator"], value["batch_size"], value["bucket_width"])
        for value in candidates
    } == {
        (operator, batch, bucket)
        for operator in ("packed_fp16", "fused_fp16")
        for batch in (1, 2, 4)
        for bucket in (16, 32, 64)
    }
    assert selection["candidate_id"] == "fused_fp16_b4_w32"
    assert selection["operator"] == "fused_fp16"
    assert selection["batch_size"] == 4
    assert selection["bucket_width"] == 32
    assert (
        selection["fused_stability_gate"][
            "all_fused_samples_below_all_packed_samples"
        ]
        is True
    )
    assert (
        selection["fused_stability_gate"]["packed_fallback_applied"] is False
    )
    assert (
        selection["fused_stability_gate"]["fused_speedup_over_packed"] > 1
    )


def test_stage3_stability_gate_checks_the_selected_fused_candidate() -> None:
    screen = [
        {
            "candidate_id": candidate_id,
            "operator": operator,
            "batch_size": 4,
            "bucket_width": bucket,
            "packing": {"padding_tokens": padding},
        }
        for candidate_id, operator, bucket, padding in (
            ("fused_selected", "fused_fp16", 16, 0),
            ("fused_second", "fused_fp16", 32, 1),
            ("fused_third", "fused_fp16", 64, 2),
            ("fused_screen_control", "fused_fp16", 16, 3),
            ("packed_control", "packed_fp16", 64, 4),
        )
    ]

    def timing(value: float) -> dict:
        return {
            "values_ms": [value, value, value],
            "median_ms": value,
            "maximum_temporary_peak_bytes": 0,
        }

    finalists = {
        "fused_selected": timing(5.0),
        "fused_second": timing(6.0),
        "fused_third": timing(7.0),
        "fused_screen_control": timing(2.0),
        "packed_control": timing(4.0),
    }
    selection, _ = MODULE.select_final_candidate(
        screen,
        finalists,
        ["fused_selected", "fused_second", "fused_third"],
    )

    assert selection["candidate_id"] == "packed_control"
    assert selection["fused_stability_gate"]["tested_fused_candidate"] == (
        "fused_selected"
    )
    assert (
        selection["fused_stability_gate"][
            "all_fused_samples_below_all_packed_samples"
        ]
        is False
    )
    assert (
        selection["fused_stability_gate"]["packed_fallback_applied"] is True
    )


def test_stage3_profile_records_epilogue_and_temporary_boundary() -> None:
    profile = load_summary()["representative_profile"]
    latency = profile["latency"]
    fused_name = next(
        name for name in latency if name.startswith("fused_triton_")
    )

    assert profile["records"] == 4
    assert profile["sequence_width"] == 2047
    assert latency[fused_name]["median_ms"] < latency["packed_float16"][
        "median_ms"
    ]
    assert latency[fused_name]["maximum_temporary_peak_bytes"] == 0
    assert latency["packed_float16"]["maximum_temporary_peak_bytes"] > 0
    assert "split_compact_extent_write" in profile[
        "epilogue_stage_medians_ms"
    ]["packed_float16_extent"]
    assert "affine_bias_length_split_direct_extent_write" in profile[
        "epilogue_stage_medians_ms"
    ][f"{fused_name}_extent"]


def test_stage3_summary_is_frozen_into_parent_blueprint() -> None:
    blueprint = json.loads(
        (
            ROOT
            / "configs"
            / "cohortkv_single_config_v1"
            / "blueprint.json"
        ).read_text()
    )
    descriptor = blueprint["frozen_inputs"]["stage3_operator_summary"]

    assert blueprint["status"] == "stage4_core_frozen"
    assert blueprint["scope"]["completed_stages"] == [0, 1, 2, 3, 4]
    assert descriptor["path"] == (
        "configs/cohortkv_single_config_v1/stage3_operator_summary.json"
    )
    assert descriptor["bytes"] == SUMMARY.stat().st_size
    assert sha256_file(ROOT / descriptor["path"]) == descriptor["sha256"]
