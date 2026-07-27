from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class FidelityContract:
    recovery_target: float
    minimum_coverage: float
    confidence_level: float
    max_cost_ratio: float
    bootstrap_samples: int
    minimum_probe_users: int
    metrics: tuple[str, ...] = ("cache", "score", "top100")

    def __post_init__(self) -> None:
        if not 0 < self.recovery_target <= 1:
            raise ValueError("recovery_target must be in (0, 1]")
        if not 0 < self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be in (0, 1]")
        if not 0.5 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0.5, 1)")
        if not 0 <= self.max_cost_ratio <= 1:
            raise ValueError("max_cost_ratio must be in [0, 1]")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if self.minimum_probe_users < 1:
            raise ValueError("minimum_probe_users must be positive")
        supported = {"cache", "score", "top100"}
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be unique and nonempty")
        if not set(self.metrics).issubset(supported):
            raise ValueError("unsupported contract metric")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MigrationActionSpec:
    name: str
    kind: str
    required_state: str
    program_path: str | None = None
    replay_depth: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.kind or not self.required_state:
            raise ValueError("action fields must be nonempty")
        if self.kind not in {
            "projection",
            "compiled",
            "selective_contiguous",
            "structural_replay",
            "exact",
        }:
            raise ValueError("unsupported migration action kind")
        if self.kind == "compiled" and not self.program_path:
            raise ValueError("compiled action requires a program path")
        if self.kind == "structural_replay":
            if self.replay_depth is None or self.replay_depth < 1:
                raise ValueError("structural replay requires a positive depth")
        elif self.replay_depth is not None:
            raise ValueError("replay depth is only valid for structural replay")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RecoveryCertificate:
    metric: str
    point_recovery: float
    bootstrap_lower_bound: float
    qualifying_users: int
    valid_users: int
    observed_coverage: float
    coverage_lower_bound: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ActionCertificate:
    action_name: str
    action_kind: str
    cost_ratio: float
    metrics: tuple[RecoveryCertificate, ...]
    fidelity_passed: bool
    budget_passed: bool

    @property
    def worst_recovery_lower_bound(self) -> float:
        return min(metric.bootstrap_lower_bound for metric in self.metrics)

    @property
    def worst_coverage_lower_bound(self) -> float:
        return min(metric.coverage_lower_bound for metric in self.metrics)

    def to_dict(self) -> dict:
        output = asdict(self)
        output["worst_recovery_lower_bound"] = self.worst_recovery_lower_bound
        output["worst_coverage_lower_bound"] = self.worst_coverage_lower_bound
        return output


@dataclass(frozen=True)
class VerifiedMigrationPlan:
    protocol: str
    source_version: str
    target_version: str
    contract: FidelityContract
    actions: tuple[MigrationActionSpec, ...]
    certificates: tuple[ActionCertificate, ...]
    selected_action: str
    selection_reason: str
    fallback_actions: tuple[str, ...]
    probe_users: int
    labels_used: bool = False

    def __post_init__(self) -> None:
        if not self.protocol or not self.source_version or not self.target_version:
            raise ValueError("plan protocol and versions must be nonempty")
        action_names = tuple(action.name for action in self.actions)
        certificate_names = tuple(
            certificate.action_name for certificate in self.certificates
        )
        if len(set(action_names)) != len(action_names):
            raise ValueError("plan action names must be unique")
        if set(action_names) != set(certificate_names):
            raise ValueError("every action must have one certificate")
        if self.selected_action not in action_names:
            raise ValueError("selected action is absent from the plan")
        if any(action not in action_names for action in self.fallback_actions):
            raise ValueError("fallback action is absent from the plan")
        if len(set(self.fallback_actions)) != len(self.fallback_actions):
            raise ValueError("fallback actions must be unique")
        if self.selected_action in self.fallback_actions:
            raise ValueError("selected action cannot be its own fallback")
        if self.probe_users < self.contract.minimum_probe_users:
            raise ValueError("plan has too few probe users")
        if self.labels_used:
            raise ValueError("verified migration planning must be label-free")

    def action(self, name: str) -> MigrationActionSpec:
        for action in self.actions:
            if action.name == name:
                return action
        raise KeyError(name)

    def certificate(self, name: str) -> ActionCertificate:
        for certificate in self.certificates:
            if certificate.action_name == name:
                return certificate
        raise KeyError(name)

    def next_fallback(self, name: str) -> str | None:
        chain = (self.selected_action, *self.fallback_actions)
        if name not in chain:
            raise KeyError(name)
        index = chain.index(name) + 1
        if index == len(chain):
            return None
        return chain[index]

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "contract": self.contract.to_dict(),
            "actions": [action.to_dict() for action in self.actions],
            "certificates": [
                certificate.to_dict()
                for certificate in self.certificates
            ],
            "selected_action": self.selected_action,
            "selection_reason": self.selection_reason,
            "fallback_actions": list(self.fallback_actions),
            "probe_users": self.probe_users,
            "labels_used": self.labels_used,
        }


def _error_value(config: dict, metric: str) -> float:
    if metric == "cache":
        return float(config["cache_error_rel"])
    if metric == "score":
        return max(0.0, 1.0 - float(config["score_cosine"]))
    if metric == "top100":
        return max(0.0, 1.0 - float(config["top100_overlap"]))
    raise ValueError(f"unsupported metric: {metric}")


def _ratio_of_means(
    reuse: np.ndarray,
    action: np.ndarray,
    exact: np.ndarray,
) -> float:
    denominator = float(np.mean(reuse) - np.mean(exact))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return float("nan")
    return float((np.mean(reuse) - np.mean(action)) / denominator)


def _wilson_lower_bound(
    successes: int,
    total: int,
    confidence_level: float,
) -> float:
    if total < 1:
        return float("nan")
    z = NormalDist().inv_cdf(confidence_level)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return max(0.0, (center - radius) / denominator)


def _certify_metric(
    records: list[dict],
    action_name: str,
    metric: str,
    contract: FidelityContract,
    rng: np.random.Generator,
) -> RecoveryCertificate:
    reuse_values = []
    action_values = []
    exact_values = []
    per_user = []
    for record in records:
        configs = record["configs"]
        reuse = _error_value(configs["reuse"], metric)
        action = _error_value(configs[action_name], metric)
        exact = _error_value(configs["recompute"], metric)
        denominator = reuse - exact
        if (
            math.isfinite(reuse)
            and math.isfinite(action)
            and math.isfinite(exact)
            and denominator > 1e-12
        ):
            reuse_values.append(reuse)
            action_values.append(action)
            exact_values.append(exact)
            per_user.append((reuse - action) / denominator)
    reuse_array = np.asarray(reuse_values, dtype=np.float64)
    action_array = np.asarray(action_values, dtype=np.float64)
    exact_array = np.asarray(exact_values, dtype=np.float64)
    recovery_array = np.asarray(per_user, dtype=np.float64)
    valid_users = len(recovery_array)
    if valid_users < contract.minimum_probe_users:
        return RecoveryCertificate(
            metric=metric,
            point_recovery=float("nan"),
            bootstrap_lower_bound=float("nan"),
            qualifying_users=0,
            valid_users=valid_users,
            observed_coverage=0.0,
            coverage_lower_bound=0.0,
            passed=False,
        )
    point = _ratio_of_means(
        reuse_array,
        action_array,
        exact_array,
    )
    bootstrap = np.empty(contract.bootstrap_samples, dtype=np.float64)
    for sample in range(contract.bootstrap_samples):
        indices = rng.integers(0, valid_users, size=valid_users)
        bootstrap[sample] = _ratio_of_means(
            reuse_array[indices],
            action_array[indices],
            exact_array[indices],
        )
    finite = bootstrap[np.isfinite(bootstrap)]
    lower = (
        float(np.quantile(finite, 1.0 - contract.confidence_level))
        if len(finite)
        else float("nan")
    )
    qualifying = int(
        np.count_nonzero(recovery_array >= contract.recovery_target)
    )
    coverage = qualifying / valid_users
    coverage_lower = _wilson_lower_bound(
        qualifying,
        valid_users,
        contract.confidence_level,
    )
    passed = (
        math.isfinite(lower)
        and lower >= contract.recovery_target
        and coverage_lower >= contract.minimum_coverage
    )
    return RecoveryCertificate(
        metric=metric,
        point_recovery=point,
        bootstrap_lower_bound=lower,
        qualifying_users=qualifying,
        valid_users=valid_users,
        observed_coverage=coverage,
        coverage_lower_bound=coverage_lower,
        passed=passed,
    )


def certify_action(
    records: list[dict],
    action: MigrationActionSpec,
    cost_ratio: float,
    contract: FidelityContract,
    seed: int,
) -> ActionCertificate:
    if len(records) < contract.minimum_probe_users:
        raise ValueError("insufficient probe records")
    if not math.isfinite(cost_ratio) or cost_ratio < 0:
        raise ValueError("cost ratio must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    metrics = tuple(
        _certify_metric(
            records,
            action.name,
            metric,
            contract,
            rng,
        )
        for metric in contract.metrics
    )
    return ActionCertificate(
        action_name=action.name,
        action_kind=action.kind,
        cost_ratio=cost_ratio,
        metrics=metrics,
        fidelity_passed=all(metric.passed for metric in metrics),
        budget_passed=cost_ratio <= contract.max_cost_ratio,
    )


def compile_verified_plan(
    protocol: str,
    source_version: str,
    target_version: str,
    actions: tuple[MigrationActionSpec, ...],
    records: list[dict],
    cost_ratios: dict[str, float],
    contract: FidelityContract,
    seed: int,
) -> VerifiedMigrationPlan:
    if len(set(action.name for action in actions)) != len(actions):
        raise ValueError("action names must be unique")
    if "recompute" not in {action.name for action in actions}:
        raise ValueError("verified action library requires recompute")
    missing = {
        action.name
        for action in actions
        if action.name not in cost_ratios
    }
    if missing:
        raise ValueError(f"missing action cost ratios: {sorted(missing)}")
    certificates = tuple(
        certify_action(
            records,
            action,
            float(cost_ratios[action.name]),
            contract,
            seed + index * 1009,
        )
        for index, action in enumerate(actions)
    )
    budgeted = [
        certificate
        for certificate in certificates
        if certificate.fidelity_passed and certificate.budget_passed
    ]
    passed = [
        certificate
        for certificate in certificates
        if certificate.fidelity_passed
    ]
    if budgeted:
        selected = min(
            budgeted,
            key=lambda value: (
                value.cost_ratio,
                -value.worst_recovery_lower_bound,
                value.action_name,
            ),
        )
        reason = "minimum_cost_certified_within_budget"
    elif passed:
        selected = min(
            passed,
            key=lambda value: (
                value.cost_ratio,
                -value.worst_recovery_lower_bound,
                value.action_name,
            ),
        )
        reason = "minimum_cost_certified_budget_overflow"
    else:
        selected = next(
            certificate
            for certificate in certificates
            if certificate.action_name == "recompute"
        )
        reason = "forced_exact_no_candidate_certified"
    fallback = [
        certificate
        for certificate in passed
        if certificate.action_name != selected.action_name
        and certificate.cost_ratio > selected.cost_ratio
    ]
    fallback.sort(
        key=lambda value: (
            value.cost_ratio,
            -value.worst_recovery_lower_bound,
            value.action_name,
        )
    )
    if (
        selected.action_name != "recompute"
        and "recompute" not in {
            certificate.action_name for certificate in fallback
        }
    ):
        fallback.append(
            next(
                certificate
                for certificate in certificates
                if certificate.action_name == "recompute"
            )
        )
    return VerifiedMigrationPlan(
        protocol=protocol,
        source_version=source_version,
        target_version=target_version,
        contract=contract,
        actions=actions,
        certificates=certificates,
        selected_action=selected.action_name,
        selection_reason=reason,
        fallback_actions=tuple(
            certificate.action_name for certificate in fallback
        ),
        probe_users=len(records),
    )
