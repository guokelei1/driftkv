"""Input-statistics parity, affine export and real nested optimizer checks."""
from collections import Counter
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import query_centering as centered, probes
from experiments.ch3_motivation.code.probe_variants import FullItemProbeLadder, save_variant, load_variant
from tests.experiments.ch3_motivation.test_probes import fixture


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("mode", centered.MODES)
def test_original_row_moments_and_explicit_query_moments(mode):
    model, sample = fixture()
    original = FullItemProbeLadder.from_model(model, hidden_width=7, depth=1)
    actual = copy.deepcopy(original)
    original.fit_input_statistics(model, [sample])
    stats = centered.fit_statistics(actual, model, [sample], mode)
    assert torch.equal(original.row_mean, actual.row_mean)
    assert torch.equal(original.row_scale, actual.row_scale)
    _, trace = probes.native_score(model, sample["parent_cache"], sample["candidates"], sample["query_delta"], trace=True)
    mean = torch.stack([q.mean((0, 2)) for q in trace.queries])
    second = torch.stack([q.square().mean((0, 2)) for q in trace.queries])
    std = (second - mean.square()).clamp_min(1e-8).sqrt()
    torch.testing.assert_close(torch.tensor(stats["mean"]), mean)
    torch.testing.assert_close(torch.tensor(stats["centered_std"]), std)
    if mode == "rms":
        assert torch.equal(original.query_scale, actual.query_scale)
    else:
        torch.testing.assert_close(actual.query_scale, std)
    assert stats["users"] == 1 and stats["input_moment_flops"] > 0


def test_exported_bias_matches_centered_outputs_and_input_gradients():
    torch.manual_seed(19)
    cfg = probes.ProbeConfig(2, 2, 3, 4, hidden_width=5, depth=1)
    training = FullItemProbeLadder(cfg, "fgh").double()
    with torch.no_grad():
        training.query_scale.uniform_(.6, 1.8)
        training.row_scale.uniform_(.8, 1.5)
        for term in (training.g, training.h):
            for module in term:
                module.weights[-1].normal_(std=.2)
    exported = copy.deepcopy(training)
    mean = torch.randn_like(training.query_scale)
    centered.shift_biases(exported, mean, ("g", "h"), fold=True)
    counts = Counter()
    q = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    rows = torch.randn(1, 2, 6, cfg.row_dim, dtype=torch.float64, requires_grad=True)
    with centered.centered_reads(training, mean, counts):
        expected = [training.query_correction(1, q), training.interaction_correction(1, q, rows)]
        left_grad = torch.autograd.grad(sum(value.square().sum() for value in expected), (q, rows))
    actual = [exported.query_correction(1, q), exported.interaction_correction(1, q, rows)]
    right_grad = torch.autograd.grad(sum(value.square().sum() for value in actual), (q, rows))
    for left, right in zip(expected + list(left_grad), actual + list(right_grad), strict=True):
        torch.testing.assert_close(left, right, rtol=2e-12, atol=2e-12)
    assert counts["training_query_subtractions"] == 2 * q.numel()
    assert "query_correction" not in training.__dict__


def test_actual_nested_optimizer_folded_scores_serialization_and_costs(tmp_path, monkeypatch):
    model, sample = fixture()
    validation = copy.deepcopy(sample)
    validation["candidates"] = torch.tensor([[13, 16, 22]])
    train = probes.prepare_training_samples(model, [sample])
    val = probes.prepare_training_samples(model, [validation])
    probe = FullItemProbeLadder.from_model(model, hidden_width=7, depth=1)
    statistics = None
    real = centered.fit_probe_resident
    captured = []

    def capture_after_optimizer(model, current, train, val, **kwargs):
        report = real(model, current, train, val, **kwargs)
        with torch.no_grad():
            captured.append(current.score(model, val[0]["parent_cache"], val[0]["item_ids"],
                val[0]["candidates"], val[0]["query_delta"])[0].detach().clone())
        return report

    monkeypatch.setattr(centered, "fit_probe_resident", capture_after_optimizer)
    for stage in ("f", "fg", "fgh"):
        if probe.stage != stage:
            probe = probe.expanded(stage)
        report = centered.fit_query_conditioned(model, probe, train, val, mode="center_std",
            statistics=statistics, inherited_g=stage == "fgh", steps=4, patience=0,
            validation_every=1, learning_rate=1e-3, seed=17)
        statistics = report["query_conditioning_statistics"]
        score = probe.score(model, val[0]["parent_cache"], val[0]["item_ids"],
            val[0]["candidates"], val[0]["query_delta"])[0]
        torch.testing.assert_close(score, captured[-1], rtol=2e-5, atol=3e-6)
        assert report["total_calibration_flops_estimate"] == sum(report["arithmetic"].values())
        assert "fit_input_statistics" not in probe.__dict__
        assert "query_correction" not in probe.__dict__
        save_variant(tmp_path / f"{stage}.pt", probe, report)
        loaded, _ = load_variant(tmp_path / f"{stage}.pt")
        assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in probe.state_dict().items())
        if stage != "f":
            assert report["query_conditioning_execution"]["counts"]["training_query_subtractions"] > 0


def test_best_zero_preserves_inherited_fg_bits(monkeypatch):
    model, sample = fixture()
    fg = FullItemProbeLadder.from_model(model, stage="fg", hidden_width=7, depth=1)
    stats = centered.fit_statistics(fg, model, [sample], "center_std")
    with torch.no_grad():
        for module in fg.g:
            module.weights[-1].normal_(std=.1)
    before = {key: value.clone() for key, value in fg.state_dict().items()}
    fgh = fg.expanded("fgh")
    monkeypatch.setattr(centered, "fit_probe_resident", lambda *a, **k:
        dict(best_step=0, arithmetic={}, total_calibration_flops_estimate=0))
    centered.fit_query_conditioned(model, fgh, [sample], [sample], mode="center_std",
        statistics=stats, inherited_g=True)
    assert all(torch.equal(value, fgh.state_dict()[key]) for key, value in before.items())
    assert all(torch.count_nonzero(module.weights[-1]) == 0 for module in fgh.h)
