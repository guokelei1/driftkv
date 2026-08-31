import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def pipeline():
    from run_yambda500m_large_qualification import (
        BASE_CONTRACT, EXECUTION_CONTRACT, LargePipeline,
    )

    return LargePipeline(BASE_CONTRACT, EXECUTION_CONTRACT, threads=56)


def test_large_contract_and_execution_are_bound_to_passing_canary():
    value = pipeline()
    assert value.contract["decision_basis"]["frozen_primary"] == "10L_H320_heads10_context1024"
    assert value.execution is not None
    assert value.execution["execution_amendment"]["global_train_batch_size"] == 96
    assert value.execution["execution_amendment"]["full_eval_batch_size_per_rank"] == 64
    assert value.execution["execution_amendment"]["reuse_cohort_size_per_rank"] == 12


def test_large_matrix_keeps_fifth_d14_edge_and_marks_partial_e14():
    value = pipeline()
    assert value.horizon_label("D14", 5, 14) == "E14_partial"
    assert value.primary_horizon("D14", 5) == 7
    command = value.raw_full_command(
        "D14", 5, 14, Path("out"), Path("v4.pt"), Path("v5.pt"), 64,
    )
    assert command[command.index("--start-day") + 1] == "287"
    assert command[command.index("--end-day") + 1] == "301"
    assert "large_D14_E14_partial_edge5_full_only" in command


def test_large_formal_reuse_scope_is_only_five_d14_e14_cells():
    from run_yambda500m_large_qualification import (
        BASE_CONTRACT, D14_E14_SCOPE_AMENDMENT, EXECUTION_CONTRACT, LargePipeline,
    )

    value = LargePipeline(
        BASE_CONTRACT, EXECUTION_CONTRACT, threads=56,
        reuse_scope_path=D14_E14_SCOPE_AMENDMENT,
    )
    assert value.reuse_tasks() == [
        ("D14", 1, 14),
        ("D14", 2, 14),
        ("D14", 3, 14),
        ("D14", 4, 14),
        ("D14", 5, 14),
    ]


def test_large_current_scope_stops_after_full_only():
    value = pipeline()
    assert value.reuse_tasks() == []
    assert value.reuse_scope["reuse_scope"]["expected_cells"] == 0


def test_large_pro_is_d14_only_and_uses_frozen_c64_layout():
    value = pipeline()
    d7 = value.raw_reuse_command(
        "D7", 1, 7, Path("d7"), Path("v0.pt"), Path("v1.pt"), 12, 128,
    )
    d14 = value.raw_reuse_command(
        "D14", 1, 7, Path("d14"), Path("v0.pt"), Path("v1.pt"), 12, 128,
    )
    assert "--include-pro-lazy" not in d7
    assert d14[d14.index("--pro-repair-width") + 1] == "256"
    assert d14[d14.index("--pro-carriers") + 1] == "64"
    assert d14[d14.index("--pro-path") + 1] == "evokv_pro_lazy_reader_c64_rolling"


def test_large_training_queue_has_one_shared_and_fifteen_update_checkpoints():
    value = pipeline()
    paths = {value.checkpoint_dir("D7", 0)}
    for branch in ("D7", "D14"):
        updates = value.contract["scope"]["branches"][branch]["updates"]
        paths.update(value.checkpoint_dir(branch, version) for version in range(1, updates + 1))
    assert len(paths) == 16


def test_large_status_excludes_archived_and_canary_progress(tmp_path, monkeypatch, capsys):
    value = pipeline()
    monkeypatch.setattr(value, "output", tmp_path)
    monkeypatch.setattr(value, "state_path", tmp_path / "pipeline_state.json")
    monkeypatch.setattr(value, "resource_root", tmp_path / "resource_canary")
    active = tmp_path / "shared_v0/progress.json"
    active.parent.mkdir(parents=True)
    active.write_text('{"status":"formal_training_in_progress","completed_steps":100}\n')
    archived = tmp_path / "interruptions/old/shared_v0/progress.json"
    archived.parent.mkdir(parents=True)
    archived.write_text('{"status":"formal_training_in_progress","completed_steps":999}\n')
    implementation = tmp_path / "implementation_canary/test/progress.json"
    implementation.parent.mkdir(parents=True)
    implementation.write_text('{"status":"distributed_canary_complete","completed_steps":20}\n')
    value.status()
    import json
    output = json.loads(capsys.readouterr().out)
    assert output["current_training_progress"]["completed_steps"] == 100
    assert output["current_training_progress"]["path"].endswith("shared_v0/progress.json")
