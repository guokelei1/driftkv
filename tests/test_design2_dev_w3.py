import importlib.util
import sys
from pathlib import Path

import pytest

from hstu_kvcache.migration.design2_plan import (
    D2ActionPlan,
    build_d2_record_owner_map,
    d2_record_owner_map_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
ACTION_PLAN = (
    ROOT
    / "configs/cohortkv_d2"
    / "action_plan_theta1_theta2_staggered_renewal_h12.json"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MATERIALIZER = _load(
    "cohortkv_design2_dev_w3_materializer_test",
    ROOT / "scripts/materialize_cohortkv_design2_dev_w3_inputs.py",
)
LAUNCHER = _load(
    "cohortkv_design2_dev_w3_launcher_test",
    ROOT / "scripts/launch_cohortkv_design2_dev_w3.py",
)


def test_dev_w3_selection_is_owner_local_and_complete() -> None:
    if not ACTION_PLAN.is_file():
        pytest.skip("local D2 action plan is unavailable")
    plan = D2ActionPlan.load(ACTION_PLAN)
    selections, selected_ids = MATERIALIZER._select(plan)
    selection = selections["3"]
    owner_map = build_d2_record_owner_map(
        plan,
        3,
        "strict_cow_lpt",
    )
    assert selection["owner_map_sha256"] == (
        d2_record_owner_map_sha256(owner_map)
    )
    assert set(selection["ranks"]) == {"0", "1", "2"}
    assert len(selected_ids) == 9
    for rank, choices in selection["ranks"].items():
        assert set(choices) == {
            "compiled",
            "natural_exact",
            "scheduled_exact",
        }
        assert all(
            owner_map[record_id] == int(rank)
            for record_id in choices.values()
        )


def test_dev_w3_launcher_paths_and_devices_are_isolated() -> None:
    args = LAUNCHER.parse_args([])
    assert tuple(args.visible_devices) == ("0", "1", "3")
    assert LAUNCHER._parse_devices(args.visible_devices) == (
        "0",
        "1",
        "3",
    )
    output_dir = Path("/tmp/d2-w3")
    assert LAUNCHER._normal_artifact_path(output_dir).name == (
        "dev_w3_primitives.json"
    )
    assert LAUNCHER._hard_failure_artifact_path(output_dir).name == (
        "dev_w3_hard_failure.json"
    )
    with pytest.raises(ValueError):
        LAUNCHER._parse_devices(("0", "1"))
    with pytest.raises(ValueError):
        LAUNCHER._parse_devices(("0", "0", "3"))


def test_dev_w3_hard_failure_payload_cannot_enter_formal_gate() -> None:
    result = LAUNCHER.ProcessResult(
        returncode=1,
        elapsed_seconds=4.0,
        timed_out=False,
        stdout="",
        stderr="local_rank: 1 exitcode: 23",
        process_group_alive_after_exit=False,
        cleanup_signal=None,
        process_group_alive_after_cleanup=False,
    )
    payload = LAUNCHER._hard_failure_payload(
        ("0", "1", "3"),
        ["torchrun"],
        result,
        timeout_seconds=45.0,
        worker_timeout_seconds=120.0,
    )
    assert payload["status"] == "complete"
    assert payload["scientific_result"] is False
    assert payload["development_diagnostic"][
        "formal_stage_b_gate"
    ] is False
    assert payload["development_diagnostic"]["substitute_for_w4"] is False
    assert all(payload["checks"].values())
