"""Streamed token ridge equals the explicit centered average-token objective."""

from collections import defaultdict
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from design_one.calibrate_affine import RIDGE, fit_layer
from hstu_kvcache.design_one.affine import AffineKVViewLayer
from hstu_kvcache.models import HSTUKVCache


def test_streamed_affine_ridge_matches_token_reference_and_fit_only_normalization():
    generator = torch.Generator().manual_seed(172013)
    random = lambda *shape: torch.randn(*shape, dtype=torch.double, generator=generator)
    rows, features, targets = {}, [], []
    for index, length in enumerate((6, 6, 14, 4)):
        native = random(length, 8)
        context = torch.rand(length, 2, dtype=torch.double, generator=generator)
        context[:, 0] = (context[:, 0] > .5).double()
        residual = random(length, 8)+torch.arange(8, dtype=torch.double)+3
        if index == 3:
            native += 1000  # excluded validation data must not set coordinates
            residual *= 100
        full = native+residual
        uid = (10, 10, 11, 12)[index]
        key = f"{uid}:a{index}"
        rows[key] = dict(uid=uid,
            parent=HSTUKVCache(native[:, :4][None, None], native[:, 4:][None, None], length),
            teacher=HSTUKVCache(full[:, :4][None, None], full[:, 4:][None, None], length),
            context=context[None])
        if index < 3:
            features.append(torch.cat((native, context), -1))
            targets.append(residual)
    selected = list(rows)[:3]
    x, y = torch.cat(features), torch.cat(targets)
    center, scale = x.mean(0), x.std(0, correction=0).clamp_min(1e-4)
    bias, units = y.mean(0), y.std(0, correction=0).clamp_min(1e-6)
    normalized = (x-center)/scale
    gram = normalized.T@normalized/len(x)
    cross = normalized.T@((y-bias)/units)/len(x)
    reference = torch.linalg.solve(gram+RIDGE*torch.eye(x.shape[1], dtype=torch.double), cross)*units
    for batch_size in (1, 3):
        layer = AffineKVViewLayer(heads=2, head_dim=2).double()
        costs = defaultdict(int)
        record = fit_layer(layer, rows, selected, 0, batch_size=batch_size,
                           device=torch.device("cpu"), costs=costs)
        torch.testing.assert_close(layer.input_center, center, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(layer.input_scale, scale, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(layer.map_weight, reference, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(layer.map_bias, bias, atol=1e-12, rtol=1e-12)
        prediction = normalized@layer.map_weight+layer.map_bias
        torch.testing.assert_close(prediction.mean(0), y.mean(0), atol=1e-12, rtol=1e-12)
        assert record["valid_tokens"] == 26
        assert record["fit_solver_kv_mse"] == pytest.approx(float((prediction-y).square().mean()), rel=1e-10)
        assert costs["ridge_gram_cross_flops"] > 0 and costs["ridge_solve_flops_estimate"] > 0
