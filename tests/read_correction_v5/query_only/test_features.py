from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import pytest
import torch

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.read_correction import score_corrected
from hstu_kvcache.read_correction_v5.query_only import QueryFeatureCorrection, fit_feature_rule
from read_correction_v5.query_only.fit import fit_query_features


@pytest.mark.parametrize("mode", ["head_phi", "cross_phi"])
def test_closed_form_known_features_and_independent_ridge(mode):
    torch.manual_seed(74)
    rule = QueryFeatureCorrection(2, 3, mode).double()
    with torch.no_grad():
        rule.weight.normal_(std=.2)
        rule.bias.normal_(std=.1)
    query = torch.randn(24, 2, 4, 3, dtype=torch.float64)
    target = rule.rate(query).detach()
    fitted, stats = fit_feature_rule(query, target, feature_mode=mode, ridge=0)
    novel = torch.randn(7, 2, 5, 3, dtype=torch.float64)
    torch.testing.assert_close(fitted.rate(novel), rule.rate(novel), atol=2e-12, rtol=2e-12)
    counts = torch.arange(7)
    # NaN historical inputs cannot affect this query-only path.
    actual = fitted(novel, torch.tensor(float("nan")), torch.tensor(float("nan")), counts)
    torch.testing.assert_close(actual, rule.rate(novel) * counts[:, None, None, None], atol=1e-11, rtol=1e-11)
    assert actual[0].count_nonzero() == 0
    assert stats["solve_dtype"] == "float64"

    ridge = .17
    regularized, _ = fit_feature_rule(query, target, feature_mode=mode, ridge=ridge)
    x = rule.features(query)
    if mode == "head_phi":
        x = x.permute(1, 0, 2, 3).reshape(2, -1, 3)
        y = target.permute(1, 0, 2, 3).reshape(2, -1, 3)
    else:
        x = x.reshape(1, -1, 12)
        y = target.transpose(1, 2).reshape(1, -1, 6)
    x = (x - x.mean(1, keepdim=True)) / x.std(1, correction=0, keepdim=True)
    design = torch.cat((x, torch.ones_like(x[..., :1])), -1)
    penalty = torch.eye(design.shape[-1], dtype=torch.float64) * ridge
    penalty[-1, -1] = 0
    reference = torch.linalg.solve(design.transpose(-2, -1) @ design / design.shape[1] + penalty,
                                   design.transpose(-2, -1) @ y / design.shape[1])
    torch.testing.assert_close(regularized.weight.reshape_as(reference[:, :-1]), reference[:, :-1], atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(regularized.bias.reshape_as(reference[:, -1]), reference[:, -1], atol=2e-12, rtol=2e-12)


def test_layerwise_actual_query_matches_known_full_phi_residual_and_preserves_cache():
    torch.manual_seed(39)
    model = HSTU(HSTUConfig(num_items=60, num_behaviors=3, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=16, input_dropout=0., temporal_num_freqs=2)).eval().requires_grad_(False)
    rows = {}
    slope = torch.randn(2, 2, 4, 4) * .004
    bias = torch.randn(2, 2, 4) * .003
    for uid in range(16):
        count = 5 + uid % 3
        keys = torch.zeros(2, 1, count, 8)
        values = torch.randn_like(keys) * .02
        teacher_values = values.clone()
        for layer, block in enumerate(model.blocks):
            for head in range(2):
                span = slice(head * 4, (head + 1) * 4)
                # q @ key * attention.scale selects each q coordinate. The
                # native ELU+1 then makes an exact per-head phi target.
                keys[layer, 0, :4, span] = torch.eye(4) / block.attn.scale
                teacher_values[layer, 0, :4, span] += count * slope[layer, head]
                teacher_values[layer, 0, 4, span] += count * bias[layer, head]
        rows[uid] = {"parent": HSTUKVCache(keys, values, count),
            "teacher": HSTUKVCache(keys.clone(), teacher_values, count),
            "candidates": torch.tensor([1 + uid, 20 + uid, 38, 39]), "query_delta": float(uid + 1)}
    before = {uid: (row["parent"].k.clone(), row["parent"].v.clone()) for uid, row in rows.items()}
    modules, stats = fit_query_features(model, rows, list(range(12)), list(range(12, 16)),
        {"feature_mode": "head_phi", "attention_backend": "torch", "ridge": 0.}, "cpu", "medium", batch_size=4)
    for record in stats["layers"]:
        assert record["diagnostics"]["validation"]["fitted_read_mse"] < 1e-12
    for uid in range(12, 16):
        row = rows[uid]
        args = (row["candidates"][None], torch.tensor([row["query_delta"]]))
        count = torch.tensor([row["parent"].seq_len])
        actual, _ = score_corrected(model, row["parent"], *args, modules, count)
        wanted, _ = score_corrected(model, row["teacher"], *args, [None, None], count)
        torch.testing.assert_close(actual, wanted, atol=2e-6, rtol=2e-5)
    for uid, row in rows.items():
        assert torch.equal(row["parent"].k, before[uid][0])
        assert torch.equal(row["parent"].v, before[uid][1])
    assert stats["selection"].startswith("none")
    assert stats["cost"]["calibration_flops"] > 0
    assert stats["counts_by_uid"]["1"] == 6
