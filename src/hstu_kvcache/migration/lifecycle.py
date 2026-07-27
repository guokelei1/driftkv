from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from typing import Literal

LifecycleAction = Literal["migrate", "exact"]
LifecycleStateKind = Literal["exact", "migrated"]


def _finite_nonnegative(value: float, name: str) -> float:
    prepared = float(value)
    if not math.isfinite(prepared) or prepared < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return prepared


@dataclass(frozen=True)
class CacheLifecycleState:
    record_id: int
    served_version: int
    last_exact_version: int
    migration_depth: int
    risk_score: float
    state_kind: LifecycleStateKind

    def __post_init__(self) -> None:
        if (
            isinstance(self.record_id, bool)
            or self.record_id < 0
            or isinstance(self.served_version, bool)
            or self.served_version < 0
            or isinstance(self.last_exact_version, bool)
            or not 0 <= self.last_exact_version <= self.served_version
            or isinstance(self.migration_depth, bool)
            or self.migration_depth < 0
            or self.state_kind not in {"exact", "migrated"}
        ):
            raise ValueError("cache lifecycle state is invalid")
        _finite_nonnegative(self.risk_score, "risk_score")
        if self.state_kind == "exact" and (
            self.last_exact_version != self.served_version
            or self.migration_depth != 0
            or self.risk_score != 0
        ):
            raise ValueError("exact cache lifecycle state is inconsistent")
        if self.state_kind == "migrated" and (
            self.last_exact_version >= self.served_version
            or self.migration_depth != self.served_version - self.last_exact_version
        ):
            raise ValueError("migrated cache lifecycle state is inconsistent")

    @classmethod
    def exact(cls, record_id: int, version: int) -> CacheLifecycleState:
        return cls(
            record_id=record_id,
            served_version=version,
            last_exact_version=version,
            migration_depth=0,
            risk_score=0.0,
            state_kind="exact",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "served_version": self.served_version,
            "last_exact_version": self.last_exact_version,
            "migration_depth": self.migration_depth,
            "risk_score": self.risk_score,
            "state_kind": self.state_kind,
        }


@dataclass(frozen=True)
class MonotoneRiskCalibration:
    correction_upper_bounds: tuple[float, ...]
    one_hop_risks: tuple[float, ...]
    propagation_gain: float
    quantile: float

    def __post_init__(self) -> None:
        if (
            not self.correction_upper_bounds
            or len(self.correction_upper_bounds) != len(self.one_hop_risks)
            or any(
                right < left
                for left, right in zip(
                    self.correction_upper_bounds,
                    self.correction_upper_bounds[1:],
                    strict=False,
                )
            )
            or any(
                right < left
                for left, right in zip(
                    self.one_hop_risks,
                    self.one_hop_risks[1:],
                    strict=False,
                )
            )
            or not 0.5 <= self.quantile < 1
        ):
            raise ValueError("risk calibration is invalid")
        for value in (
            *self.correction_upper_bounds,
            *self.one_hop_risks,
            self.propagation_gain,
        ):
            _finite_nonnegative(value, "risk calibration value")

    def predict_one_hop(self, correction_magnitude: float) -> float:
        prepared = _finite_nonnegative(
            correction_magnitude,
            "correction_magnitude",
        )
        index = bisect.bisect_left(self.correction_upper_bounds, prepared)
        return self.one_hop_risks[min(index, len(self.one_hop_risks) - 1)]

    def predict(
        self,
        previous_risk: float,
        correction_magnitude: float,
    ) -> float:
        return self.predict_one_hop(correction_magnitude) + (
            self.propagation_gain
            * _finite_nonnegative(previous_risk, "previous_risk")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "correction_upper_bounds": list(self.correction_upper_bounds),
            "one_hop_risks": list(self.one_hop_risks),
            "propagation_gain": self.propagation_gain,
            "quantile": self.quantile,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> MonotoneRiskCalibration:
        bounds = value.get("correction_upper_bounds")
        risks = value.get("one_hop_risks")
        if not isinstance(bounds, list) or not isinstance(risks, list):
            raise ValueError("risk calibration payload is invalid")
        return cls(
            correction_upper_bounds=tuple(float(item) for item in bounds),
            one_hop_risks=tuple(float(item) for item in risks),
            propagation_gain=float(value["propagation_gain"]),
            quantile=float(value["quantile"]),
        )


@dataclass(frozen=True)
class LinearSketchRiskCalibration:
    feature_name: str
    layer_quantile: float
    intercept: float
    feature_mean: float
    feature_scale: float
    feature_coefficient: float
    group_means: tuple[float, ...]
    group_scales: tuple[float, ...]
    group_coefficients: tuple[float, ...]
    num_edges: int
    maximum_depth: int
    ridge: float
    target: str

    def __post_init__(self) -> None:
        groups = self.num_edges * self.maximum_depth
        if (
            not self.feature_name
            or not self.target
            or not 0.5 <= self.layer_quantile <= 1.0
            or self.num_edges < 1
            or self.maximum_depth < 1
            or len(self.group_means) != groups
            or len(self.group_scales) != groups
            or len(self.group_coefficients) != groups
            or self.feature_scale <= 0
            or any(value <= 0 for value in self.group_scales)
        ):
            raise ValueError("linear sketch risk calibration is invalid")
        for value in (
            self.intercept,
            self.feature_mean,
            self.feature_scale,
            self.feature_coefficient,
            *self.group_means,
            *self.group_scales,
            *self.group_coefficients,
            self.ridge,
        ):
            if not math.isfinite(value):
                raise ValueError("linear sketch risk value is nonfinite")
        if self.ridge < 0:
            raise ValueError("linear sketch ridge must be nonnegative")

    def predict(
        self,
        source_version: int,
        migration_depth_after: int,
        feature_value: float,
    ) -> float:
        if (
            not 0 <= source_version < self.num_edges
            or not 1 <= migration_depth_after <= self.maximum_depth
        ):
            raise ValueError("linear sketch risk group is unavailable")
        prepared = _finite_nonnegative(feature_value, "feature_value")
        value = self.intercept + self.feature_coefficient * (
            (prepared - self.feature_mean) / self.feature_scale
        )
        selected = (
            source_version * self.maximum_depth
            + migration_depth_after
            - 1
        )
        for index, coefficient in enumerate(self.group_coefficients):
            indicator = 1.0 if index == selected else 0.0
            value += coefficient * (
                (indicator - self.group_means[index])
                / self.group_scales[index]
            )
        return math.exp(min(value, 20.0))

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "layer_quantile": self.layer_quantile,
            "intercept": self.intercept,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "feature_coefficient": self.feature_coefficient,
            "group_means": list(self.group_means),
            "group_scales": list(self.group_scales),
            "group_coefficients": list(self.group_coefficients),
            "num_edges": self.num_edges,
            "maximum_depth": self.maximum_depth,
            "ridge": self.ridge,
            "target": self.target,
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> LinearSketchRiskCalibration:
        return cls(
            feature_name=str(value["feature_name"]),
            layer_quantile=float(value["layer_quantile"]),
            intercept=float(value["intercept"]),
            feature_mean=float(value["feature_mean"]),
            feature_scale=float(value["feature_scale"]),
            feature_coefficient=float(value["feature_coefficient"]),
            group_means=tuple(float(item) for item in value["group_means"]),
            group_scales=tuple(
                float(item) for item in value["group_scales"]
            ),
            group_coefficients=tuple(
                float(item) for item in value["group_coefficients"]
            ),
            num_edges=int(value["num_edges"]),
            maximum_depth=int(value["maximum_depth"]),
            ridge=float(value["ridge"]),
            target=str(value["target"]),
        )


@dataclass(frozen=True)
class LifecycleDecision:
    record_id: int
    source_version: int
    target_version: int
    action: LifecycleAction
    reason: str
    predicted_risk: float
    candidate_evaluated: bool

    def __post_init__(self) -> None:
        if (
            self.target_version != self.source_version + 1
            or self.action not in {"migrate", "exact"}
            or not self.reason
        ):
            raise ValueError("lifecycle decision is invalid")
        _finite_nonnegative(self.predicted_risk, "predicted_risk")
        if self.reason == "max_migration_depth" and self.candidate_evaluated:
            raise ValueError("depth decision cannot evaluate a candidate")

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "action": self.action,
            "reason": self.reason,
            "predicted_risk": self.predicted_risk,
            "candidate_evaluated": self.candidate_evaluated,
        }


@dataclass(frozen=True)
class LifecyclePolicy:
    max_migration_depth: int
    risk_threshold: float
    calibration: MonotoneRiskCalibration

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_migration_depth, bool)
            or self.max_migration_depth < 1
        ):
            raise ValueError("max_migration_depth must be positive")
        _finite_nonnegative(self.risk_threshold, "risk_threshold")

    def requires_candidate(self, state: CacheLifecycleState) -> bool:
        return state.migration_depth < self.max_migration_depth

    def decide(
        self,
        state: CacheLifecycleState,
        target_version: int,
        correction_magnitude: float | None = None,
    ) -> LifecycleDecision:
        if target_version != state.served_version + 1:
            raise ValueError("lifecycle target must be the adjacent version")
        if not self.requires_candidate(state):
            return LifecycleDecision(
                record_id=state.record_id,
                source_version=state.served_version,
                target_version=target_version,
                action="exact",
                reason="max_migration_depth",
                predicted_risk=state.risk_score,
                candidate_evaluated=False,
            )
        if correction_magnitude is None:
            raise ValueError("candidate correction magnitude is required")
        predicted = self.calibration.predict(
            state.risk_score,
            correction_magnitude,
        )
        action: LifecycleAction = (
            "exact" if predicted >= self.risk_threshold else "migrate"
        )
        return LifecycleDecision(
            record_id=state.record_id,
            source_version=state.served_version,
            target_version=target_version,
            action=action,
            reason="risk_threshold" if action == "exact" else "risk_accepted",
            predicted_risk=predicted,
            candidate_evaluated=True,
        )

    def advance(
        self,
        state: CacheLifecycleState,
        decision: LifecycleDecision,
    ) -> CacheLifecycleState:
        if (
            decision.record_id != state.record_id
            or decision.source_version != state.served_version
            or decision.target_version != state.served_version + 1
        ):
            raise ValueError("lifecycle decision does not match state")
        if decision.action == "exact":
            return CacheLifecycleState.exact(
                state.record_id,
                decision.target_version,
            )
        return CacheLifecycleState(
            record_id=state.record_id,
            served_version=decision.target_version,
            last_exact_version=state.last_exact_version,
            migration_depth=state.migration_depth + 1,
            risk_score=decision.predicted_risk,
            state_kind="migrated",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_migration_depth": self.max_migration_depth,
            "risk_threshold": self.risk_threshold,
            "calibration": self.calibration.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> LifecyclePolicy:
        calibration = value.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError("lifecycle policy payload is invalid")
        return cls(
            max_migration_depth=int(value["max_migration_depth"]),
            risk_threshold=float(value["risk_threshold"]),
            calibration=MonotoneRiskCalibration.from_dict(calibration),
        )


@dataclass(frozen=True)
class SketchLifecyclePolicy:
    max_migration_depth: int
    risk_threshold: float
    calibration: LinearSketchRiskCalibration

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_migration_depth, bool)
            or self.max_migration_depth < 1
            or self.max_migration_depth > self.calibration.maximum_depth
        ):
            raise ValueError("sketch lifecycle depth is invalid")
        _finite_nonnegative(self.risk_threshold, "risk_threshold")

    def requires_candidate(self, state: CacheLifecycleState) -> bool:
        return state.migration_depth < self.max_migration_depth

    def decide(
        self,
        state: CacheLifecycleState,
        target_version: int,
        feature_value: float | None = None,
    ) -> LifecycleDecision:
        if target_version != state.served_version + 1:
            raise ValueError("lifecycle target must be the adjacent version")
        if not self.requires_candidate(state):
            return LifecycleDecision(
                record_id=state.record_id,
                source_version=state.served_version,
                target_version=target_version,
                action="exact",
                reason="max_migration_depth",
                predicted_risk=state.risk_score,
                candidate_evaluated=False,
            )
        if feature_value is None:
            raise ValueError("candidate sketch feature is required")
        predicted = self.calibration.predict(
            state.served_version,
            state.migration_depth + 1,
            feature_value,
        )
        action: LifecycleAction = (
            "exact" if predicted >= self.risk_threshold else "migrate"
        )
        return LifecycleDecision(
            record_id=state.record_id,
            source_version=state.served_version,
            target_version=target_version,
            action=action,
            reason="risk_threshold" if action == "exact" else "risk_accepted",
            predicted_risk=predicted,
            candidate_evaluated=True,
        )

    def advance(
        self,
        state: CacheLifecycleState,
        decision: LifecycleDecision,
    ) -> CacheLifecycleState:
        if (
            decision.record_id != state.record_id
            or decision.source_version != state.served_version
            or decision.target_version != state.served_version + 1
        ):
            raise ValueError("lifecycle decision does not match state")
        if decision.action == "exact":
            return CacheLifecycleState.exact(
                state.record_id,
                decision.target_version,
            )
        return CacheLifecycleState(
            record_id=state.record_id,
            served_version=decision.target_version,
            last_exact_version=state.last_exact_version,
            migration_depth=state.migration_depth + 1,
            risk_score=decision.predicted_risk,
            state_kind="migrated",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_migration_depth": self.max_migration_depth,
            "risk_threshold": self.risk_threshold,
            "calibration": self.calibration.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> SketchLifecyclePolicy:
        calibration = value.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError("sketch lifecycle policy payload is invalid")
        return cls(
            max_migration_depth=int(value["max_migration_depth"]),
            risk_threshold=float(value["risk_threshold"]),
            calibration=LinearSketchRiskCalibration.from_dict(calibration),
        )


@dataclass(frozen=True)
class BalancedLifecyclePolicy:
    max_migration_depth: int
    exact_fractions: tuple[float, ...]
    edge_severities: tuple[float, ...]
    scheduler_seed: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_migration_depth, bool)
            or self.max_migration_depth < 1
            or not self.exact_fractions
            or len(self.exact_fractions) != len(self.edge_severities)
            or isinstance(self.scheduler_seed, bool)
            or self.scheduler_seed < 0
        ):
            raise ValueError("balanced lifecycle policy is invalid")
        for value in self.exact_fractions:
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError("balanced exact fraction is invalid")
        for value in self.edge_severities:
            _finite_nonnegative(value, "edge_severity")

    def exact_fraction(self, source_version: int) -> float:
        if not 0 <= source_version < len(self.exact_fractions):
            raise ValueError("balanced lifecycle edge is unavailable")
        return self.exact_fractions[source_version]

    def _tie_break(self, record_id: int, source_version: int) -> int:
        payload = (
            f"{self.scheduler_seed}:{source_version}:{record_id}"
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8])

    def plan(
        self,
        states: tuple[CacheLifecycleState, ...],
        target_version: int,
    ) -> tuple[LifecycleDecision, ...]:
        if (
            not states
            or len({state.record_id for state in states}) != len(states)
            or any(
                target_version != state.served_version + 1
                for state in states
            )
        ):
            raise ValueError("balanced lifecycle planning states differ")
        source_version = target_version - 1
        fraction = self.exact_fraction(source_version)
        target_exact = math.floor(fraction * len(states) + 0.5)
        mandatory = {
            state.record_id
            for state in states
            if state.migration_depth >= self.max_migration_depth
        }
        optional = sorted(
            (
                state
                for state in states
                if state.record_id not in mandatory
            ),
            key=lambda state: (
                -state.migration_depth,
                self._tie_break(state.record_id, source_version),
                state.record_id,
            ),
        )
        selected = set(mandatory)
        selected.update(
            state.record_id
            for state in optional[
                : max(0, target_exact - len(mandatory))
            ]
        )
        severity = self.edge_severities[source_version]
        output = []
        for state in states:
            if state.record_id in selected:
                reason = (
                    "max_migration_depth"
                    if state.record_id in mandatory
                    else "balanced_exact_quota"
                )
                output.append(
                    LifecycleDecision(
                        record_id=state.record_id,
                        source_version=source_version,
                        target_version=target_version,
                        action="exact",
                        reason=reason,
                        predicted_risk=severity,
                        candidate_evaluated=False,
                    )
                )
            else:
                output.append(
                    LifecycleDecision(
                        record_id=state.record_id,
                        source_version=source_version,
                        target_version=target_version,
                        action="migrate",
                        reason="balanced_migrate",
                        predicted_risk=severity,
                        candidate_evaluated=False,
                    )
                )
        return tuple(output)

    def advance(
        self,
        state: CacheLifecycleState,
        decision: LifecycleDecision,
    ) -> CacheLifecycleState:
        if (
            decision.record_id != state.record_id
            or decision.source_version != state.served_version
            or decision.target_version != state.served_version + 1
        ):
            raise ValueError("lifecycle decision does not match state")
        if decision.action == "exact":
            return CacheLifecycleState.exact(
                state.record_id,
                decision.target_version,
            )
        return CacheLifecycleState(
            record_id=state.record_id,
            served_version=decision.target_version,
            last_exact_version=state.last_exact_version,
            migration_depth=state.migration_depth + 1,
            risk_score=decision.predicted_risk,
            state_kind="migrated",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_migration_depth": self.max_migration_depth,
            "exact_fractions": list(self.exact_fractions),
            "edge_severities": list(self.edge_severities),
            "scheduler_seed": self.scheduler_seed,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> BalancedLifecyclePolicy:
        return cls(
            max_migration_depth=int(value["max_migration_depth"]),
            exact_fractions=tuple(
                float(item) for item in value["exact_fractions"]
            ),
            edge_severities=tuple(
                float(item) for item in value["edge_severities"]
            ),
            scheduler_seed=int(value["scheduler_seed"]),
        )


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def fit_monotone_risk_calibration(
    correction_magnitudes: list[float],
    one_hop_errors: list[float],
    propagation_ratios: list[float],
    bins: int = 8,
    quantile: float = 0.9,
) -> MonotoneRiskCalibration:
    if (
        len(correction_magnitudes) != len(one_hop_errors)
        or not correction_magnitudes
        or isinstance(bins, bool)
        or bins < 1
        or not 0.5 <= quantile < 1
    ):
        raise ValueError("risk calibration samples are invalid")
    pairs = sorted(
        (
            _finite_nonnegative(correction, "correction_magnitude"),
            _finite_nonnegative(error, "one_hop_error"),
        )
        for correction, error in zip(
            correction_magnitudes,
            one_hop_errors,
            strict=True,
        )
    )
    group_size = math.ceil(len(pairs) / min(bins, len(pairs)))
    bounds = []
    risks = []
    for start in range(0, len(pairs), group_size):
        group = pairs[start : start + group_size]
        bounds.append(group[-1][0])
        current = _quantile([value[1] for value in group], quantile)
        risks.append(max(risks[-1] if risks else 0.0, current))
    prepared_ratios = [
        _finite_nonnegative(value, "propagation_ratio")
        for value in propagation_ratios
    ]
    gain = _quantile(prepared_ratios, quantile) if prepared_ratios else 1.0
    return MonotoneRiskCalibration(
        correction_upper_bounds=tuple(bounds),
        one_hop_risks=tuple(risks),
        propagation_gain=gain,
        quantile=quantile,
    )
