import json
from pathlib import Path

import pytest
import torch

from hstu_kvcache.migration import (
    EXECUTABLE_PLAN_PROTOCOL,
    CompiledCacheAdapter,
    MigrationProgram,
    load_executable_plan,
    load_runtime_program,
    sha256_file,
    write_runtime_program,
)


def make_program() -> MigrationProgram:
    torch.manual_seed(71)
    return MigrationProgram(
        source_version="theta0",
        target_version="theta1",
        adapter=CompiledCacheAdapter(
            weights=torch.randn(2, 8, 16),
            biases=torch.randn(2, 16),
            source_rank=8,
            ridge=1e-3,
        ),
    )


def certificate_metric(
    metric: str,
    point_recovery: float,
    lower_bound: float,
    qualifying_users: int,
    coverage_lower_bound: float,
    passed: bool,
) -> dict:
    return {
        "metric": metric,
        "point_recovery": point_recovery,
        "bootstrap_lower_bound": lower_bound,
        "qualifying_users": qualifying_users,
        "valid_users": 60,
        "observed_coverage": qualifying_users / 60,
        "coverage_lower_bound": coverage_lower_bound,
        "passed": passed,
    }


def write_plan(tmp_path: Path, passed: bool = True) -> Path:
    program_path = tmp_path / "runtime.pt"
    descriptor = write_runtime_program(
        make_program(),
        program_path,
        {"source_program_sha256": "a" * 64},
    )
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    selected_certificate = {
        "action_name": "compiled_full_affine",
        "action_kind": "compiled",
        "cost_ratio": 0.1,
        "metrics": [
            certificate_metric(
                metric,
                0.9 if passed else 0.2,
                0.8 if passed else 0.1,
                55 if passed else 0,
                0.85 if passed else 0.0,
                passed,
            )
            for metric in ("cache", "score", "top100")
        ],
        "fidelity_passed": passed,
        "budget_passed": passed,
        "worst_recovery_lower_bound": 0.8 if passed else 0.1,
        "worst_coverage_lower_bound": 0.85 if passed else 0.0,
    }
    exact_certificate = {
        "action_name": "recompute",
        "action_kind": "exact",
        "cost_ratio": 1.0,
        "metrics": [
            certificate_metric(metric, 1.0, 1.0, 60, 0.95, True)
            for metric in ("cache", "score", "top100")
        ],
        "fidelity_passed": True,
        "budget_passed": False,
        "worst_recovery_lower_bound": 1.0,
        "worst_coverage_lower_bound": 0.95,
    }
    plan = {
        "protocol": EXECUTABLE_PLAN_PROTOCOL,
        "status": "executable",
        "labels_used": False,
        "source_version": "theta0",
        "target_version": "theta1",
        "model": {
            "num_layers": 2,
            "hidden_size": 8,
            "num_heads": 2,
            "head_dim": 4,
        },
        "actions": [
            {
                "name": "compiled_full_affine",
                "kind": "compiled",
                "required_state": "normalized_capsule_fp16",
                "program_path": "runtime.pt",
                "replay_depth": None,
            },
            {
                "name": "recompute",
                "kind": "exact",
                "required_state": "raw_history",
                "program_path": None,
                "replay_depth": None,
            },
        ],
        "selected_action": "compiled_full_affine",
        "selection_reason": "minimum_cost_certified_within_budget",
        "fallback_actions": ["recompute"],
        "contract": {
            "recovery_target": 0.7,
            "minimum_coverage": 0.8,
            "confidence_level": 0.9,
            "max_cost_ratio": 0.3,
            "bootstrap_samples": 1000,
            "minimum_probe_users": 50,
            "metrics": ["cache", "score", "top100"],
        },
        "certificates": [
            selected_certificate,
            exact_certificate,
        ],
        "source_representations": {
            "compiled_full_affine": ["normalized_capsule_fp16"],
            "recompute": ["raw_history"],
        },
        "deployed_representation_certificate": {
            "source_dtype": "float16",
            "program_dtype": "float16",
            "output_dtype": "float16",
            "passed": passed,
            "certificate_users": 60,
            "views": ["cache", "score", "top100"],
            "selected_certificate": selected_certificate,
        },
        "runtime_program": {
            **descriptor,
            "path": "runtime.pt",
        },
        "frozen_inputs": {
            "source_checkpoint": {
                "path": "checkpoint.bin",
                "sha256": sha256_file(checkpoint),
            },
            "role_manifest": {
                "path": "manifest.json",
                "sha256": sha256_file(manifest),
            },
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return plan_path


def test_runtime_program_round_trip_prepares_fp16(tmp_path: Path) -> None:
    path = tmp_path / "runtime.pt"
    descriptor = write_runtime_program(make_program(), path, {"fit_users": 40})

    loaded, actual = load_runtime_program(
        path,
        expected_sha256=descriptor["sha256"],
        expected_source_version="theta0",
        expected_target_version="theta1",
        expected_model={
            "num_layers": 2,
            "hidden_size": 8,
            "num_heads": 2,
            "head_dim": 4,
        },
    )

    assert loaded.adapter.weights.dtype == torch.float16
    assert loaded.adapter.biases.dtype == torch.float16
    assert loaded.adapter.weights.is_contiguous()
    assert actual["provenance"] == {"fit_users": 40}


def test_runtime_program_rejects_hash_and_version_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.pt"
    write_runtime_program(make_program(), path, {})

    with pytest.raises(ValueError, match="hash"):
        load_runtime_program(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="source version"):
        load_runtime_program(path, expected_source_version="theta-other")


def test_runtime_program_rejects_invalid_adapter_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.pt"
    write_runtime_program(make_program(), path, {})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["ridge"] = float("nan")
    torch.save(payload, path)

    with pytest.raises(ValueError, match="adapter metadata"):
        load_runtime_program(path)


def test_executable_plan_loads_program_and_fallback_chain(
    tmp_path: Path,
) -> None:
    plan = load_executable_plan(
        write_plan(tmp_path),
        repository_root=tmp_path,
    )

    assert plan.action_chain == ("compiled_full_affine", "recompute")
    assert plan.next_fallback("compiled_full_affine") == "recompute"
    assert plan.next_fallback("recompute") is None
    assert plan.required_representations("compiled_full_affine") == (
        "normalized_capsule_fp16",
    )
    assert plan.program.adapter.weights.dtype == torch.float16


def test_executable_plan_rejects_failed_deployed_certificate(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="certificate"):
        load_executable_plan(
            write_plan(tmp_path, passed=False),
            repository_root=tmp_path,
        )


def test_executable_plan_rejects_changed_frozen_input(
    tmp_path: Path,
) -> None:
    plan_path = write_plan(tmp_path)
    (tmp_path / "checkpoint.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="source_checkpoint hash"):
        load_executable_plan(plan_path, repository_root=tmp_path)


def test_executable_plan_rejects_program_path_mismatch(
    tmp_path: Path,
) -> None:
    plan_path = write_plan(tmp_path)
    payload = json.loads(plan_path.read_text())
    payload["actions"][0]["program_path"] = "different.pt"
    plan_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="program path"):
        load_executable_plan(plan_path, repository_root=tmp_path)


def test_executable_plan_rejects_duplicate_actions(
    tmp_path: Path,
) -> None:
    plan_path = write_plan(tmp_path)
    payload = json.loads(plan_path.read_text())
    payload["actions"].append(payload["actions"][0])
    plan_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unique"):
        load_executable_plan(plan_path, repository_root=tmp_path)


def test_executable_plan_rejects_its_own_hash_mismatch(
    tmp_path: Path,
) -> None:
    plan_path = write_plan(tmp_path)

    with pytest.raises(ValueError, match="plan hash"):
        load_executable_plan(
            plan_path,
            repository_root=tmp_path,
            expected_sha256="0" * 64,
        )


def test_executable_plan_rejects_divergent_selected_certificate(
    tmp_path: Path,
) -> None:
    plan_path = write_plan(tmp_path)
    payload = json.loads(plan_path.read_text())
    payload["deployed_representation_certificate"][
        "selected_certificate"
    ]["cost_ratio"] = 0.2
    plan_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="certificate"):
        load_executable_plan(plan_path, repository_root=tmp_path)
