from __future__ import annotations

import numpy as np
import torch

from hstu_kvcache.base_fitting import (
    equal_user_request_weights,
    fit_feature_scaler,
    fit_linear_base,
    objective_and_gradient,
    request_row_ids,
)
from hstu_kvcache.models import FrozenLinearBaseRanker


def finite_difference(function, parameters: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    output = np.empty_like(parameters)
    for index in range(len(parameters)):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        output[index] = (function(upper) - function(lower)) / (2 * epsilon)
    return output


def test_streaming_base_fit_matches_expanded() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [0.5], [1.5]], dtype=np.float64)
    lengths = np.asarray([3, 2])
    starts = np.asarray([0, 3])
    row_request = request_row_ids(lengths)
    uids = np.asarray([7, 8])
    targets = np.asarray([2, 0])
    mask = np.asarray([True, True])
    scaler = fit_feature_scaler(features, row_request, mask)
    parameters = np.asarray([0.2])

    def objective(value: np.ndarray) -> float:
        return objective_and_gradient(
            value,
            features=features,
            starts=starts,
            lengths=lengths,
            row_request=row_request,
            uids=uids,
            request_mask=mask,
            scaler=scaler,
            objective="listwise",
            l2=0.01,
            targets=targets,
        )[0]

    _, gradient = objective_and_gradient(
        parameters,
        features=features,
        starts=starts,
        lengths=lengths,
        row_request=row_request,
        uids=uids,
        request_mask=mask,
        scaler=scaler,
        objective="listwise",
        l2=0.01,
        targets=targets,
    )
    np.testing.assert_allclose(gradient, finite_difference(objective, parameters), atol=1e-8)


def test_base_scaler_uses_base_fit_only() -> None:
    features = np.asarray([[0.0], [1.0], [10_000.0]])
    lengths = np.ones(3, dtype=np.int64)
    row_request = request_row_ids(lengths)
    base_fit_mask = np.asarray([True, True, False])
    scaler = fit_feature_scaler(features, row_request, base_fit_mask, quantiles=(0.0, 1.0))
    assert scaler.clip_high.tolist() == [1.0]
    assert scaler.mean.tolist() == [0.5]


def test_base_parameters_are_deterministic_and_f_feedback_label_direction() -> None:
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
    lengths = np.ones(4, dtype=np.int64)
    starts = np.arange(4, dtype=np.int64)
    row_request = request_row_ids(lengths)
    uids = np.arange(4, dtype=np.int64)
    labels = np.asarray([0, 0, 1, 1])
    mask = np.ones(4, dtype=bool)
    scaler = fit_feature_scaler(features, row_request, mask, quantiles=(0.0, 1.0))
    kwargs = {
        "features": features,
        "starts": starts,
        "lengths": lengths,
        "row_request": row_request,
        "uids": uids,
        "request_mask": mask,
        "scaler": scaler,
        "objective": "binary",
        "l2": 0.01,
        "labels": labels,
    }
    first = fit_linear_base(**kwargs)
    second = fit_linear_base(**kwargs)
    assert first.success and second.success
    np.testing.assert_array_equal(first.x, second.x)
    assert first.x[0] > 0


def test_each_user_has_equal_total_fit_weight() -> None:
    weights = equal_user_request_weights(
        np.asarray([1, 1, 2]), np.asarray([True, True, True])
    )
    assert weights.tolist() == [0.5, 0.5, 1.0]


def test_frozen_base_forward_matches_fitter_clipping_and_has_no_parameters() -> None:
    artifact = {
        "coefficients": [2.0],
        "intercept": 0.25,
        "scaler": {
            "clip_low": [0.0],
            "clip_high": [2.0],
            "mean": [1.0],
            "scale": [0.5],
        },
    }
    scorer = FrozenLinearBaseRanker.from_frozen_artifact(artifact)
    features = torch.tensor([[-5.0], [1.0], [9.0]])
    assert torch.equal(scorer(features), torch.tensor([-3.75, 0.25, 4.25]))
    assert list(scorer.parameters()) == []
