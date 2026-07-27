import json
from pathlib import Path

SUMMARY = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "cohortkv_single_config_v1"
    / "stage1_frontier_summary.json"
)


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text())


def test_stage1_summary_freezes_complete_pair_coverage() -> None:
    summary = load_summary()

    assert summary["protocol"] == "cohortkv_single_config_stage1_frozen_v1"
    assert summary["status"] == "stage1_frozen"
    assert summary["measurement_boundary"]["selection_points"] == 177
    assert summary["measurement_boundary"]["selective_intervals_per_pair"] == 53
    assert summary["measurement_boundary"]["final_test_evaluated"] is False
    assert [pair["source_version"] for pair in summary["pairs"]] == [
        "theta0",
        "theta4",
        "theta10",
    ]


def test_stage1_freezes_failed_certificate_and_diagnostic_system_action() -> None:
    summary = load_summary()

    for pair in summary["pairs"]:
        action = pair["profiled_selective_action"]
        assert action["configuration"] == {
            "m": 12,
            "start_layer": 0,
            "end_layer": 11,
        }
        assert action["source_representations"] == [
            "old_kv_fp16",
            "raw_history",
        ]
        assert action["publishable_sync_action"] is False
        assert pair["certificate"]["passed"] is False
        assert pair["certificate"]["published_action"] == "recompute"
        assert pair["compiled_strictly_dominates_all_53_selective_points"] is True

    downstream = summary["downstream_rule"]
    assert downstream["transition_hidden_bytes_for_frozen_system_baseline"] == 0
    assert downstream["system_baseline_source_representations"] == [
        "old_kv_fp16",
        "raw_history",
    ]
