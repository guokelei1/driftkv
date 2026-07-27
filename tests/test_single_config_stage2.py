import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "configs"
    / "cohortkv_single_config_v1"
    / "stage2_compiler_summary.json"
)
SOURCE_VERSIONS = ("theta0", "theta4", "theta10")
SCRIPT = ROOT / "scripts" / "compile_cohortkv_stage2.py"
SPEC = importlib.util.spec_from_file_location(
    "compile_cohortkv_stage2",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_stage2_summary_freezes_deployed_certificate_boundary() -> None:
    summary = load_summary()

    assert summary["protocol"] == "cohortkv_single_config_stage2_frozen_v1"
    assert summary["status"] == "stage2_frozen"
    assert summary["measurement_boundary"] == {
        "execution": (
            "serialized certificate shards reloaded into deployed "
            "numeric representations"
        ),
        "primary_source_dtype": "float16",
        "runtime_program_dtype": "float16",
        "output_dtype": "float16",
        "residual_hidden_suffix_dtype": "bfloat16",
        "certificate_users_per_pair": 60,
        "final_test_users_evaluated": False,
        "recommendation_labels_used": False,
        "world_size": 3,
        "gpu_name": "NVIDIA A40",
    }
    assert [pair["source_version"] for pair in summary["pairs"]] == list(
        SOURCE_VERSIONS
    )


def test_stage2_sequence_materialization_is_label_free() -> None:
    sequence = MODULE.prepare_unlabeled_sequence(
        {
            "item_ids": [1, 2, 3, 4],
            "behaviors": [0, 1, 0, 1],
            "time_deltas": [0.0, 1.0, 2.0, 3.0],
            "labels": [8, 7, 6, 5],
        },
        3,
    )

    assert sequence == {
        "item_ids": [2, 3, 4],
        "behaviors": [1, 0, 1],
        "time_deltas": [1.0, 2.0, 3.0],
    }


def test_stage2_freezes_primary_actions_and_fallback_chains() -> None:
    summary = load_summary()
    expected_fallbacks = {
        "theta0": ["structural_p8", "recompute"],
        "theta4": ["recompute"],
        "theta10": ["structural_p8", "recompute"],
    }

    for pair in summary["pairs"]:
        certificate = pair["selected_certificate"]
        assert pair["selected_action"] == "compiled_full_affine"
        assert pair["fallback_actions"] == expected_fallbacks[
            pair["source_version"]
        ]
        assert pair["executable_fallback_actions"] == pair[
            "fallback_actions"
        ]
        assert certificate["certificate_passed"] is True
        assert certificate["cost_ratio_to_exact"] < 0.3
        assert certificate["worst_recovery_lower_bound"] >= 0.7
        assert certificate["worst_coverage_lower_bound"] >= 0.8


def test_stage2_threshold_sweep_and_bf16_correction_are_frozen() -> None:
    summary = load_summary()

    for pair in summary["pairs"]:
        assert [
            value["recovery_target"] for value in pair["threshold_sweep"]
        ] == [0.5, 0.6, 0.7, 0.8, 0.9]
        assert [
            value["selected_action"] for value in pair["threshold_sweep"]
        ] == [
            "compiled_full_affine",
            "compiled_full_affine",
            "compiled_full_affine",
            "compiled_full_affine",
            "recompute",
        ]
        serialized = pair["serialized_certificate_source"]
        assert serialized["residual_hidden_suffix_dtype"] == "bfloat16"
        assert serialized["residual_hidden_suffix_absmax"] > 65_504
        assert (
            serialized["residual_hidden_suffix_fp16_overflow_values"] > 0
        )
        assert serialized["temporary_shards_retained"] is False

    correction = summary["design_correction"]
    assert correction["rejected_representation"] == (
        "residual_hidden_suffix_fp16"
    )
    assert correction["replacement"] == "residual_hidden_suffix_bf16"
    assert correction["bytes_per_element_unchanged"] is True
    assert correction["primary_compiled_path_changed"] is False


def test_stage2_checked_plans_match_frozen_summary() -> None:
    summary = load_summary()

    for pair in summary["pairs"]:
        descriptor = pair["executable_plan"]
        path = ROOT / descriptor["path"]
        payload = json.loads(path.read_text())
        assert sha256_file(path) == descriptor["sha256"]
        assert payload["protocol"] == (
            "cohortkv_executable_migration_plan_v1"
        )
        assert payload["status"] == "executable"
        assert payload["labels_used"] is False
        assert payload["source_version"] == pair["source_version"]
        assert payload["target_version"] == "theta11"
        assert payload["selected_action"] == pair["selected_action"]
        assert payload["fallback_actions"] == pair["fallback_actions"]
        assert payload["source_representations"]["structural_p8"] == [
            "raw_history",
            "residual_hidden_suffix_p8_bf16",
        ]
        certificate = payload["deployed_representation_certificate"]
        assert certificate["passed"] is True
        assert certificate["source_dtype"] == "float16"
        assert certificate["program_dtype"] == "float16"
        assert certificate["output_dtype"] == "float16"
        assert certificate["residual_hidden_suffix_dtype"] == "bfloat16"


def test_stage2_summary_is_frozen_into_parent_blueprint() -> None:
    blueprint = json.loads(
        (
            ROOT
            / "configs"
            / "cohortkv_single_config_v1"
            / "blueprint.json"
        ).read_text()
    )
    descriptor = blueprint["frozen_inputs"]["stage2_compiler_summary"]

    assert 2 in blueprint["scope"]["completed_stages"]
    assert descriptor["path"] == (
        "configs/cohortkv_single_config_v1/stage2_compiler_summary.json"
    )
    assert sha256_file(ROOT / descriptor["path"]) == descriptor["sha256"]
