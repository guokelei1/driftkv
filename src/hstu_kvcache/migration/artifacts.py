from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .low_rank import CompiledCacheAdapter
from .program import MigrationProgram
from .verification import FidelityContract, MigrationActionSpec

RUNTIME_PROGRAM_PROTOCOL = "cohortkv_runtime_program_v1"
EXECUTABLE_PLAN_PROTOCOL = "cohortkv_executable_migration_plan_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(path: str | Path, repository_root: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path(repository_root) / value


def _shape(value: object, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or item < 1 for item in value)
    ):
        raise ValueError(f"{name} must be a nonempty positive integer shape")
    return tuple(value)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_certificate_metric(
    value: dict,
    contract: FidelityContract,
) -> None:
    qualifying_users = value.get("qualifying_users")
    valid_users = value.get("valid_users")
    observed_coverage = value.get("observed_coverage")
    coverage_lower_bound = value.get("coverage_lower_bound")
    bootstrap_lower_bound = value.get("bootstrap_lower_bound")
    if (
        not _finite_number(value.get("point_recovery"))
        or not _finite_number(bootstrap_lower_bound)
        or not isinstance(qualifying_users, int)
        or isinstance(qualifying_users, bool)
        or not isinstance(valid_users, int)
        or isinstance(valid_users, bool)
        or valid_users < 1
        or not 0 <= qualifying_users <= valid_users
        or not _finite_number(observed_coverage)
        or not 0 <= observed_coverage <= 1
        or not math.isclose(
            observed_coverage,
            qualifying_users / valid_users,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not _finite_number(coverage_lower_bound)
        or not 0 <= coverage_lower_bound <= 1
        or not isinstance(value.get("passed"), bool)
    ):
        raise ValueError("executable plan metric certificate is invalid")
    expected_passed = (
        valid_users >= contract.minimum_probe_users
        and bootstrap_lower_bound >= contract.recovery_target
        and coverage_lower_bound >= contract.minimum_coverage
    )
    if value["passed"] is not expected_passed:
        raise ValueError("executable plan metric pass flag is inconsistent")


def write_runtime_program(
    program: MigrationProgram,
    path: str | Path,
    provenance: dict,
) -> dict:
    if not isinstance(provenance, dict):
        raise ValueError("runtime program provenance must be a dictionary")
    output_path = Path(path)
    prepared = program.to("cpu", dtype=torch.float16)
    weights = prepared.adapter.weights.contiguous()
    biases = prepared.adapter.biases.contiguous()
    if not bool(torch.isfinite(weights).all()) or not bool(
        torch.isfinite(biases).all()
    ):
        raise ValueError("runtime program contains nonfinite values")
    source_rank = prepared.adapter.source_rank
    ridge = prepared.adapter.ridge
    if (
        not isinstance(source_rank, int)
        or isinstance(source_rank, bool)
        or not 0 <= source_rank <= min(weights.shape[1:])
        or not isinstance(ridge, (int, float))
        or isinstance(ridge, bool)
        or not math.isfinite(ridge)
        or ridge < 0
    ):
        raise ValueError("runtime program adapter metadata is invalid")
    payload = {
        "protocol": RUNTIME_PROGRAM_PROTOCOL,
        "source_version": prepared.source_version,
        "target_version": prepared.target_version,
        "dtype": "float16",
        "weights_shape": list(weights.shape),
        "biases_shape": list(biases.shape),
        "source_rank": prepared.adapter.source_rank,
        "ridge": prepared.adapter.ridge,
        "weights": weights,
        "biases": biases,
        "provenance": provenance,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "protocol": RUNTIME_PROGRAM_PROTOCOL,
        "source_version": prepared.source_version,
        "target_version": prepared.target_version,
        "dtype": "float16",
        "weights_shape": list(weights.shape),
        "biases_shape": list(biases.shape),
    }


def load_runtime_program(
    path: str | Path,
    expected_sha256: str | None = None,
    expected_source_version: str | None = None,
    expected_target_version: str | None = None,
    expected_model: dict | None = None,
) -> tuple[MigrationProgram, dict]:
    program_path = Path(path)
    actual_sha256 = sha256_file(program_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("runtime program hash mismatch")
    payload = torch.load(
        program_path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("protocol") != RUNTIME_PROGRAM_PROTOCOL:
        raise ValueError("runtime program protocol mismatch")
    source_version = payload.get("source_version")
    target_version = payload.get("target_version")
    if (
        not isinstance(source_version, str)
        or not isinstance(target_version, str)
        or not source_version
        or not target_version
    ):
        raise ValueError("runtime program versions are invalid")
    if (
        expected_source_version is not None
        and source_version != expected_source_version
    ):
        raise ValueError("runtime program source version mismatch")
    if (
        expected_target_version is not None
        and target_version != expected_target_version
    ):
        raise ValueError("runtime program target version mismatch")
    if payload.get("dtype") != "float16":
        raise ValueError("runtime program must declare float16")
    weights = payload.get("weights")
    biases = payload.get("biases")
    if not isinstance(weights, torch.Tensor) or not isinstance(
        biases,
        torch.Tensor,
    ):
        raise ValueError("runtime program tensors are missing")
    if weights.dtype != torch.float16 or biases.dtype != torch.float16:
        raise ValueError("runtime program tensors must be float16")
    if tuple(weights.shape) != _shape(
        payload.get("weights_shape"),
        "weights_shape",
    ):
        raise ValueError("runtime program weight shape metadata mismatch")
    if tuple(biases.shape) != _shape(
        payload.get("biases_shape"),
        "biases_shape",
    ):
        raise ValueError("runtime program bias shape metadata mismatch")
    if not weights.is_contiguous() or not biases.is_contiguous():
        raise ValueError("runtime program tensors must be contiguous")
    if not bool(torch.isfinite(weights).all()) or not bool(
        torch.isfinite(biases).all()
    ):
        raise ValueError("runtime program contains nonfinite values")
    source_rank = payload.get("source_rank")
    ridge = payload.get("ridge")
    if (
        not isinstance(source_rank, int)
        or isinstance(source_rank, bool)
        or not 0 <= source_rank <= min(weights.shape[1:])
        or not isinstance(ridge, (int, float))
        or isinstance(ridge, bool)
        or not math.isfinite(ridge)
        or ridge < 0
        or not isinstance(payload.get("provenance"), dict)
    ):
        raise ValueError("runtime program adapter metadata is invalid")
    if expected_model is not None:
        expected_weights = (
            int(expected_model["num_layers"]),
            int(expected_model["hidden_size"]),
            2
            * int(expected_model["num_heads"])
            * int(expected_model["head_dim"]),
        )
        if tuple(weights.shape) != expected_weights:
            raise ValueError("runtime program differs from the model signature")
    program = MigrationProgram(
        source_version=source_version,
        target_version=target_version,
        adapter=CompiledCacheAdapter(
            weights=weights,
            biases=biases,
            source_rank=source_rank,
            ridge=float(ridge),
        ),
    )
    return program, {
        "path": str(program_path),
        "sha256": actual_sha256,
        "bytes": program_path.stat().st_size,
        "protocol": payload["protocol"],
        "source_version": source_version,
        "target_version": target_version,
        "dtype": payload["dtype"],
        "weights_shape": list(weights.shape),
        "biases_shape": list(biases.shape),
        "provenance": payload.get("provenance", {}),
    }


@dataclass(frozen=True)
class ExecutableMigrationPlan:
    source_version: str
    target_version: str
    selected_action: str
    fallback_actions: tuple[str, ...]
    actions: tuple[MigrationActionSpec, ...]
    source_representations: dict[str, tuple[str, ...]]
    program: MigrationProgram
    program_descriptor: dict
    payload: dict

    @property
    def action_chain(self) -> tuple[str, ...]:
        return self.selected_action, *self.fallback_actions

    def next_fallback(self, action: str) -> str | None:
        if action not in self.action_chain:
            raise KeyError(action)
        index = self.action_chain.index(action) + 1
        if index == len(self.action_chain):
            return None
        return self.action_chain[index]

    def required_representations(self, action: str) -> tuple[str, ...]:
        if action not in self.source_representations:
            raise KeyError(action)
        return self.source_representations[action]


def load_executable_plan(
    path: str | Path,
    repository_root: str | Path = ".",
    verify_input_hashes: bool = True,
    expected_sha256: str | None = None,
) -> ExecutableMigrationPlan:
    plan_path = Path(path)
    actual_plan_sha256 = sha256_file(plan_path)
    if (
        expected_sha256 is not None
        and actual_plan_sha256 != expected_sha256
    ):
        raise ValueError("executable plan hash mismatch")
    payload = json.loads(plan_path.read_text())
    if payload.get("protocol") != EXECUTABLE_PLAN_PROTOCOL:
        raise ValueError("executable plan protocol mismatch")
    if payload.get("status") != "executable":
        raise ValueError("executable plan is not complete")
    if payload.get("labels_used") is not False:
        raise ValueError("executable plan must be label-free")
    source_version = payload.get("source_version")
    target_version = payload.get("target_version")
    if not isinstance(source_version, str) or not isinstance(
        target_version,
        str,
    ):
        raise ValueError("executable plan versions are invalid")
    actions = tuple(
        MigrationActionSpec(
            name=value["name"],
            kind=value["kind"],
            required_state=value["required_state"],
            program_path=value.get("program_path"),
            replay_depth=value.get("replay_depth"),
        )
        for value in payload["actions"]
    )
    action_names = tuple(action.name for action in actions)
    if len(set(action_names)) != len(action_names):
        raise ValueError("executable plan action names must be unique")
    selected_action = payload.get("selected_action")
    fallback_actions = tuple(payload.get("fallback_actions", ()))
    chain = (selected_action, *fallback_actions)
    if (
        not isinstance(selected_action, str)
        or len(set(chain)) != len(chain)
        or any(action not in action_names for action in chain)
    ):
        raise ValueError("executable action chain is invalid")
    if selected_action != "recompute" and (
        not fallback_actions or fallback_actions[-1] != "recompute"
    ):
        raise ValueError("executable action chain must terminate in recompute")
    action_by_name = {action.name: action for action in actions}
    if action_by_name[chain[-1]].kind != "exact":
        raise ValueError("executable action chain must terminate in an exact action")
    if (
        "executable_fallback_actions" in payload
        and tuple(payload["executable_fallback_actions"]) != fallback_actions
    ):
        raise ValueError("executable fallback chain differs from the verified plan")
    representation_payload = payload.get("source_representations", {})
    if not isinstance(representation_payload, dict) or any(
        not isinstance(values, list)
        for values in representation_payload.values()
    ):
        raise ValueError("source representation requirements are invalid")
    source_representations = {
        name: tuple(values)
        for name, values in representation_payload.items()
    }
    if set(source_representations) != set(action_names):
        raise ValueError("source representations do not cover every action")
    if any(
        not values
        or len(set(values)) != len(values)
        or any(not isinstance(value, str) or not value for value in values)
        for values in source_representations.values()
    ):
        raise ValueError("source representation requirements are invalid")
    contract_payload = payload.get("contract", {})
    contract = FidelityContract(
        recovery_target=contract_payload["recovery_target"],
        minimum_coverage=contract_payload["minimum_coverage"],
        confidence_level=contract_payload["confidence_level"],
        max_cost_ratio=contract_payload["max_cost_ratio"],
        bootstrap_samples=contract_payload["bootstrap_samples"],
        minimum_probe_users=contract_payload["minimum_probe_users"],
        metrics=tuple(contract_payload.get("metrics", ())),
    )
    certificates = payload.get("certificates", [])
    if not isinstance(certificates, list) or any(
        not isinstance(value, dict)
        for value in certificates
    ):
        raise ValueError("executable plan certificates are invalid")
    certificate_by_action = {
        value.get("action_name"): value
        for value in certificates
    }
    if (
        len(certificates) != len(actions)
        or len(certificate_by_action) != len(actions)
        or set(certificate_by_action) != set(action_names)
    ):
        raise ValueError("executable plan certificates do not cover every action")
    for action in actions:
        value = certificate_by_action[action.name]
        metrics = value.get("metrics", [])
        if (
            value.get("action_kind") != action.kind
            or not _finite_number(value.get("cost_ratio"))
            or value["cost_ratio"] < 0
            or not isinstance(metrics, list)
            or any(not isinstance(metric, dict) for metric in metrics)
            or [metric.get("metric") for metric in metrics]
            != list(contract.metrics)
            or not isinstance(value.get("fidelity_passed"), bool)
            or not isinstance(value.get("budget_passed"), bool)
        ):
            raise ValueError("executable plan certificate is invalid")
        for metric in metrics:
            _validate_certificate_metric(metric, contract)
        worst_recovery = min(
            metric["bootstrap_lower_bound"] for metric in metrics
        )
        worst_coverage = min(
            metric["coverage_lower_bound"] for metric in metrics
        )
        if (
            value["fidelity_passed"]
            is not all(metric["passed"] for metric in metrics)
            or value["budget_passed"]
            is not (value["cost_ratio"] <= contract.max_cost_ratio)
            or not math.isclose(
                value.get("worst_recovery_lower_bound", math.nan),
                worst_recovery,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                value.get("worst_coverage_lower_bound", math.nan),
                worst_coverage,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("executable plan certificate is inconsistent")
    budgeted = [
        value
        for value in certificates
        if value["fidelity_passed"] and value["budget_passed"]
    ]
    passed = [
        value for value in certificates if value["fidelity_passed"]
    ]
    if budgeted:
        eligible = budgeted
        selection_reason = "minimum_cost_certified_within_budget"
    elif passed:
        eligible = passed
        selection_reason = "minimum_cost_certified_budget_overflow"
    else:
        eligible = [certificate_by_action["recompute"]]
        selection_reason = "forced_exact_no_candidate_certified"
    expected_selected = min(
        eligible,
        key=lambda value: (
            value["cost_ratio"],
            -value["worst_recovery_lower_bound"],
            value["action_name"],
        ),
    )
    expected_fallbacks = sorted(
        (
            value
            for value in passed
            if value["action_name"] != expected_selected["action_name"]
            and value["cost_ratio"] > expected_selected["cost_ratio"]
        ),
        key=lambda value: (
            value["cost_ratio"],
            -value["worst_recovery_lower_bound"],
            value["action_name"],
        ),
    )
    if (
        selected_action != expected_selected["action_name"]
        or payload.get("selection_reason") != selection_reason
        or fallback_actions
        != tuple(value["action_name"] for value in expected_fallbacks)
    ):
        raise ValueError("executable plan selection is inconsistent")
    certificate = payload.get("deployed_representation_certificate", {})
    selected_certificate = certificate.get("selected_certificate", {})
    if (
        certificate.get("source_dtype") != "float16"
        or certificate.get("program_dtype") != "float16"
        or certificate.get("output_dtype") != "float16"
        or certificate.get("passed") is not True
        or certificate.get("certificate_users", 0)
        < contract.minimum_probe_users
        or certificate.get("views") != list(contract.metrics)
        or selected_certificate.get("action_name") != selected_action
        or selected_certificate.get("fidelity_passed") is not True
        or selected_certificate.get("budget_passed") is not True
        or selected_certificate != certificate_by_action[selected_action]
        or any(
            certificate_by_action[action]["fidelity_passed"] is not True
            for action in chain
        )
    ):
        raise ValueError("deployed representation certificate is invalid")
    descriptor = payload["runtime_program"]
    program_path = _resolve_path(descriptor["path"], repository_root)
    program, actual_descriptor = load_runtime_program(
        program_path,
        expected_sha256=descriptor["sha256"],
        expected_source_version=source_version,
        expected_target_version=target_version,
        expected_model=payload["model"],
    )
    compiled_actions = tuple(
        action for action in actions if action.kind == "compiled"
    )
    if not compiled_actions or any(
        _resolve_path(action.program_path, repository_root).resolve()
        != program_path.resolve()
        for action in compiled_actions
    ):
        raise ValueError("compiled action program path mismatch")
    for name in (
        "sha256",
        "bytes",
        "protocol",
        "source_version",
        "target_version",
        "dtype",
        "weights_shape",
        "biases_shape",
    ):
        if descriptor.get(name) != actual_descriptor[name]:
            raise ValueError(f"runtime program descriptor {name} mismatch")
    if verify_input_hashes:
        frozen_inputs = payload.get("frozen_inputs", {})
        if not frozen_inputs:
            raise ValueError("executable plan has no frozen inputs")
        for name, value in frozen_inputs.items():
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("path"), str)
                or not value["path"]
                or not isinstance(value.get("sha256"), str)
                or len(value["sha256"]) != 64
            ):
                raise ValueError(f"{name} frozen-input descriptor is invalid")
            input_path = _resolve_path(value["path"], repository_root)
            if sha256_file(input_path) != value["sha256"]:
                raise ValueError(f"{name} hash mismatch")
    return ExecutableMigrationPlan(
        source_version=source_version,
        target_version=target_version,
        selected_action=selected_action,
        fallback_actions=fallback_actions,
        actions=actions,
        source_representations=source_representations,
        program=program,
        program_descriptor=actual_descriptor,
        payload=payload,
    )
