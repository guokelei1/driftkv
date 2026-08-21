"""Deterministic low-capacity fitting primitives for frozen Base scorers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class FeatureScaler:
    clip_low: np.ndarray
    clip_high: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        shapes = {np.asarray(value).shape for value in (self.clip_low, self.clip_high, self.mean, self.scale)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise ValueError("all scaler arrays must have one identical rank-one shape")
        if not all(
            np.isfinite(value).all()
            for value in (self.clip_low, self.clip_high, self.mean, self.scale)
        ):
            raise ValueError("scaler values must be finite")
        if np.any(self.clip_low > self.clip_high) or np.any(self.scale <= 0):
            raise ValueError("invalid clip bounds or scale")

    @property
    def dimension(self) -> int:
        return int(len(self.mean))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return (np.clip(values, self.clip_low, self.clip_high) - self.mean) / self.scale

    def as_dict(self) -> dict:
        return {
            "clip_low": self.clip_low.tolist(),
            "clip_high": self.clip_high.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }


def request_row_ids(lengths: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1 or np.any(lengths <= 0):
        raise ValueError("request lengths must be positive and rank one")
    return np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)


def equal_user_request_weights(uids: np.ndarray, request_mask: np.ndarray) -> np.ndarray:
    uids = np.asarray(uids, dtype=np.int64)
    request_mask = np.asarray(request_mask, dtype=bool)
    if uids.shape != request_mask.shape:
        raise ValueError("uids and request_mask must align")
    selected = uids[request_mask]
    if not len(selected):
        raise ValueError("at least one request must be selected")
    unique, inverse, counts = np.unique(selected, return_inverse=True, return_counts=True)
    del unique
    output = np.zeros(len(uids), dtype=np.float64)
    output[np.flatnonzero(request_mask)] = 1.0 / counts[inverse]
    return output


def fit_feature_scaler(
    features: np.ndarray,
    row_request: np.ndarray,
    request_mask: np.ndarray,
    *,
    quantiles: tuple[float, float] = (0.005, 0.995),
    std_floor: float = 1e-8,
) -> FeatureScaler:
    features = np.asarray(features)
    row_request = np.asarray(row_request, dtype=np.int64)
    request_mask = np.asarray(request_mask, dtype=bool)
    if features.ndim != 2 or len(row_request) != len(features):
        raise ValueError("feature rows and row_request must align")
    selected_rows = request_mask[row_request]
    if not selected_rows.any():
        raise ValueError("scaler selection is empty")
    lows, highs, means, scales = [], [], [], []
    for column in range(features.shape[1]):
        values = np.asarray(features[:, column][selected_rows], dtype=np.float64)
        low, high = np.quantile(values, quantiles)
        clipped = np.clip(values, low, high)
        scale = max(float(clipped.std()), std_floor)
        lows.append(float(low))
        highs.append(float(high))
        means.append(float(clipped.mean()))
        scales.append(scale)
    return FeatureScaler(
        np.asarray(lows), np.asarray(highs), np.asarray(means), np.asarray(scales)
    )


def linear_scores(features: np.ndarray, scaler: FeatureScaler, coefficients: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (features.shape[1],):
        raise ValueError("coefficient dimension does not match features")
    scores = np.zeros(len(features), dtype=np.float64)
    for column, coefficient in enumerate(coefficients):
        values = np.clip(features[:, column], scaler.clip_low[column], scaler.clip_high[column])
        scores += (values - scaler.mean[column]) / scaler.scale[column] * coefficient
    return scores


def objective_and_gradient(
    parameters: np.ndarray,
    *,
    features: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    row_request: np.ndarray,
    uids: np.ndarray,
    request_mask: np.ndarray,
    scaler: FeatureScaler,
    objective: str,
    l2: float,
    targets: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    parameters = np.asarray(parameters, dtype=np.float64)
    dimension = features.shape[1]
    if objective == "listwise":
        if parameters.shape != (dimension,) or targets is None:
            raise ValueError("listwise parameters/targets are invalid")
        coefficients = parameters
        intercept = 0.0
    elif objective == "binary":
        if parameters.shape != (dimension + 1,) or labels is None:
            raise ValueError("binary parameters/labels are invalid")
        coefficients, intercept = parameters[:dimension], float(parameters[-1])
    else:
        raise ValueError(f"unknown objective: {objective}")

    request_weights = equal_user_request_weights(uids, request_mask)
    normalizer = float(request_weights.sum())
    scores = linear_scores(features, scaler, coefficients) + intercept
    gradient_rows = np.zeros(len(features), dtype=np.float64)

    if objective == "listwise":
        maxima = np.maximum.reduceat(scores, starts)
        exponentials = np.exp(scores - maxima[row_request])
        sums = np.add.reduceat(exponentials, starts)
        probabilities = exponentials / sums[row_request]
        target_rows = starts + np.asarray(targets, dtype=np.int64)
        losses = maxima + np.log(sums) - scores[target_rows]
        data_loss = float(np.dot(request_weights, losses) / normalizer)
        gradient_rows = probabilities
        gradient_rows[target_rows] -= 1.0
        gradient_rows *= request_weights[row_request] / normalizer
    else:
        labels = np.asarray(labels, dtype=np.float64)
        selected = np.flatnonzero(request_mask)
        selected_scores = scores[selected]
        selected_labels = labels[selected]
        selected_weights = request_weights[selected] / normalizer
        data_loss = float(
            np.dot(
                selected_weights,
                np.logaddexp(0.0, selected_scores) - selected_labels * selected_scores,
            )
        )
        gradient_rows[selected] = (expit(selected_scores) - selected_labels) * selected_weights

    gradient = np.empty_like(parameters)
    for column in range(dimension):
        values = np.clip(features[:, column], scaler.clip_low[column], scaler.clip_high[column])
        standardized = (values - scaler.mean[column]) / scaler.scale[column]
        gradient[column] = np.dot(gradient_rows, standardized)
    penalty = 0.5 * float(l2) * float(np.dot(coefficients, coefficients))
    gradient[:dimension] += float(l2) * coefficients
    if objective == "binary":
        gradient[-1] = gradient_rows.sum()
    if not np.isfinite(data_loss + penalty) or not np.isfinite(gradient).all():
        raise FloatingPointError("base objective produced non-finite values")
    return data_loss + penalty, gradient


def fit_linear_base(
    *,
    features: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    row_request: np.ndarray,
    uids: np.ndarray,
    request_mask: np.ndarray,
    scaler: FeatureScaler,
    objective: str,
    l2: float,
    targets: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    initial: np.ndarray | None = None,
    max_iterations: int = 1000,
    ftol: float = 1e-12,
    gtol: float = 1e-8,
):
    dimension = features.shape[1] + int(objective == "binary")
    initial = np.zeros(dimension, dtype=np.float64) if initial is None else np.asarray(initial)

    def function(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return objective_and_gradient(
            parameters,
            features=features,
            starts=starts,
            lengths=lengths,
            row_request=row_request,
            uids=uids,
            request_mask=request_mask,
            scaler=scaler,
            objective=objective,
            l2=l2,
            targets=targets,
            labels=labels,
        )

    return minimize(
        function,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": ftol, "gtol": gtol, "maxls": 50},
    )
