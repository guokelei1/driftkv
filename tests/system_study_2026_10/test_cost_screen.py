"""Accounting regressions: denominators, one-time work, and causal source shapes."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from system_study_2026_10.cost_screen import (BASE, RELEASES, UnifiedRelayAdapter,
                                             read, source_shapes)


def screen():
    return read(ROOT / "results/system_study_2026_10/cost/initial/summary.json")["branches"]


def test_existing_accounting_reproduces_both_native_denominators():
    actual = screen()["current"]["protocols"]
    retained = read(ROOT / "results/implementation_2026_10/cost/summary.json")["protocols"]
    for protocol, row in actual.items():
        for name in ("complete_added_flops", "once_rebuild_flops", "calibration_flops",
                     "maintenance_flops", "extra_ranking_flops", "extra_writer_flops",
                     "source_reconstruction_sensitivity_flops"):
            assert row[name] == retained[protocol][name]
        assert row["complete_added_flops"] == row["preparation_flops"] + row["service_correction_flops"]
    assert actual["stride16_candidates16"]["once_rebuild_flops"] != actual["original_feedback"]["once_rebuild_flops"]


def test_extra_training_does_not_multiply_serving_or_one_time_origin():
    branches = screen()
    current, larger = branches["current"], branches["Q2_calibration128"]
    assert larger["calibration_flops"] > current["calibration_flops"]
    for branch in branches.values():
        assert branch["calibration_flops"] == branch["initial_origin"]["calibration_flops"] + sum(
            stage["total_flops"] for stage in branch["calibration_stages"].values())
    for protocol, row in larger["protocols"].items():
        old = current["protocols"][protocol]
        for name in ("once_rebuild_flops", "maintenance_flops", "extra_ranking_flops", "extra_writer_flops"):
            assert row[name] == old[name]
        assert row["complete_added_flops"] - old["complete_added_flops"] == larger["calibration_flops"] - current["calibration_flops"]


def test_wider_reader_keeps_code_and_shared_update_costs():
    branches = screen()
    old, wider = branches["current"], branches["Q1_read6plus6"]
    assert wider["memory"]["code_bytes_per_row"] == old["memory"]["code_bytes_per_row"] == 1152
    assert wider["memory"]["immutable_origin_bank_bytes_per_replica"] == old["memory"]["immutable_origin_bank_bytes_per_replica"]
    for protocol, row in wider["protocols"].items():
        base = old["protocols"][protocol]
        assert row["maintenance_flops"] == base["maintenance_flops"]
        for version, release in row["by_release"].items():
            for role in ("ranking", "writer"):
                actual = release[role + "_read_flops"]
                previous = base["by_release"][version][role + "_read_flops"]
                assert actual["gate_query_normalization_flops"] == previous["gate_query_normalization_flops"]
                assert actual["total"] == sum(v for k, v in actual.items() if k != "total")


def test_source_shape_prediction_ignores_at_or_after_cutover_events():
    config = read(BASE / "publication/v3.json")["adapter_config"]
    adapter = UnifiedRelayAdapter(**config)
    first, target = RELEASES[0][1], RELEASES[1][1]
    times = np.array([first - 20, first - 10, first + 1, first + 2], dtype=np.int64)
    events = {1: (times, np.ones(len(times), dtype=np.int64))}
    future = {1: (np.concatenate((times, [target, target + 1])), np.ones(len(times) + 2, dtype=np.int64))}
    assert source_shapes(adapter, events, [1], "v4", 100) == source_shapes(adapter, future, [1], "v4", 100)
