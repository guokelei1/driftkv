import importlib
import json
from pathlib import Path

import pytest

from hstu_kvcache.migration import (
    STAGE5_ACCOUNTING_PROTOCOL,
    build_stage5_source_state_accounting,
)
from hstu_kvcache.migration.stage5_accounting import (
    validate_stage5_source_state_accounting,
)

ACCOUNTING = importlib.import_module(
    "hstu_kvcache.migration.stage5_accounting"
)
ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "configs/cohortkv_single_config_v1/stage2_compiler_summary.json"
STAGE4 = ROOT / "configs/cohortkv_single_config_v1/stage4_system_summary.json"
STAGE4_5 = (
    ROOT / "configs/cohortkv_single_config_v1/stage4_5_source_plan_summary.json"
)


def test_stage5_accounting_is_derived_from_frozen_artifacts() -> None:
    value = build_stage5_source_state_accounting(
        STAGE2,
        STAGE4,
        STAGE4_5,
    )

    assert value["protocol"] == STAGE5_ACCOUNTING_PROTOCOL
    assert value["workload"]["records"] == 682
    direct = value["active_direct_oldkv"]
    assert direct["additional_per_record_source_state_bytes"] == 0
    assert direct["independent_capture_required"] is False
    assert direct["existing_old_kv_logical_bytes"] == 35_644_538_880
    assert direct["program_set"]["serialized_file_bytes"] == 100_777_103
    assert (
        direct["program_set"]["resident_tensor_bytes_per_worker"]
        == 100_761_600
    )
    assert direct["program_set"]["serialization_timing_available"] is False
    assert [
        point["gpu_count"] for point in direct["normal_path_points"]
    ] == [1, 2, 4]
    assert all(
        point["abort_safe"] is False
        for point in direct["normal_path_points"]
    )
    assert direct["copy_on_write_abort_safe_peak_measured"] is False


def test_stage5_accounting_preserves_setup_and_negative_boundaries() -> None:
    value = build_stage5_source_state_accounting(
        STAGE2,
        STAGE4,
        STAGE4_5,
    )

    setup = value["offline_setup"]
    assert setup["stage2_one_time_seconds"] == pytest.approx(
        308.90116163284976
    )
    assert setup["direct_program_composition_seconds"] == pytest.approx(
        1.5458995138760656
    )
    capsule = value["rejected_fp16_normalized_capsule"]
    assert capsule["logical_bytes"] == 17_822_269_440
    assert capsule["physical_bytes"] == 17_823_519_546
    assert capsule["matched_points"] == 6
    assert capsule["beats_paired_exact_points"] == 0
    assert capsule["source_read_fraction_min"] == pytest.approx(
        0.9135332101149201
    )
    assert capsule["source_read_fraction_max"] == pytest.approx(
        0.9691457049370941
    )
    assert len(value["dram_resident_backup"]["points"]) == 2
    boundary = value["claim_boundary"]
    assert boundary["physical_ssd_performance_claim"] is False
    assert boundary["capsule_capture_claim"] is False
    assert boundary["int8_claim"] is False
    assert boundary["time_break_even_claim"] is False


def test_stage5_accounting_semantics_are_exactly_rederived() -> None:
    value = build_stage5_source_state_accounting(
        STAGE2,
        STAGE4,
        STAGE4_5,
    )
    validate_stage5_source_state_accounting(
        value,
        STAGE2,
        STAGE4,
        STAGE4_5,
    )

    value["active_direct_oldkv"]["program_set"][
        "resident_tensor_bytes_per_worker"
    ] += 1
    with pytest.raises(ValueError, match="exact frozen-input derivation"):
        validate_stage5_source_state_accounting(
            value,
            STAGE2,
            STAGE4,
            STAGE4_5,
        )


def test_stage5_accounting_rejects_changed_parent_protocol(
    tmp_path: Path,
) -> None:
    value = json.loads(STAGE2.read_text())
    value["protocol"] = "changed"
    changed = tmp_path / "stage2.json"
    changed.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="stage2 protocol"):
        build_stage5_source_state_accounting(
            changed,
            STAGE4,
            STAGE4_5,
        )


def test_stage5_accounting_rejects_nonfrozen_status(tmp_path: Path) -> None:
    value = json.loads(STAGE4.read_text())
    value["status"] = "tampered"
    changed = tmp_path / "stage4.json"
    changed.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="stage4 status"):
        build_stage5_source_state_accounting(
            STAGE2,
            changed,
            STAGE4_5,
        )


def test_stage5_accounting_rejects_changed_frozen_bytes(
    tmp_path: Path,
) -> None:
    value = json.loads(STAGE2.read_text())
    value["untrusted_extra"] = True
    changed = tmp_path / "stage2.json"
    changed.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="stage2 frozen SHA-256"):
        build_stage5_source_state_accounting(
            changed,
            STAGE4,
            STAGE4_5,
        )


def test_stage5_accounting_rejects_broken_upstream_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = json.loads(STAGE4_5.read_text())
    value["upstream"]["stage4_summary"]["sha256"] = "0" * 64
    changed = tmp_path / "stage4_5.json"
    changed.write_text(json.dumps(value))
    monkeypatch.setitem(
        ACCOUNTING._INPUT_SHA256,
        "stage4_5",
        ACCOUNTING._sha256(changed),
    )

    with pytest.raises(ValueError, match="upstream chain"):
        build_stage5_source_state_accounting(
            STAGE2,
            STAGE4,
            changed,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value["source_plan"].update(
                {"placement": "cold SSD"}
            ),
            "direct old-K/V route",
        ),
        (
            lambda value: next(
                point
                for point in value["system"]["points"]
                if point["method"] == "compiled_old_kv"
            )["correctness"].update({"allclose": False}),
            "direct old-K/V evidence",
        ),
    ),
)
def test_stage5_accounting_rejects_route_or_correctness_relabeling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    value = json.loads(STAGE4_5.read_text())
    mutate(value)
    changed = tmp_path / "stage4_5.json"
    changed.write_text(json.dumps(value))
    monkeypatch.setitem(
        ACCOUNTING._INPUT_SHA256,
        "stage4_5",
        ACCOUNTING._sha256(changed),
    )

    with pytest.raises(ValueError, match=message):
        build_stage5_source_state_accounting(
            STAGE2,
            STAGE4,
            changed,
        )
