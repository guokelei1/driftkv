"""Dense-MLP function/gradient equality and executed first-affine counts."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.probes import ProbeConfig, HeadMLP
from experiments.ch3_motivation.code.probe_variants import FullItemProbeLadder
from experiments.ch3_motivation.code import probe_first_affine as projected
from experiments.ch3_motivation.code.factorized_probe_fitting import fit_probe_factorized
from experiments.ch3_motivation.code.resident_probe_fitting import fit_probe_resident
from experiments.ch3_motivation.code.probes import prepare_training_samples
from tests.experiments.ch3_motivation.test_probes import fixture


@pytest.mark.parametrize("depth", (1, 2))
def test_dense_outputs_and_all_input_parameter_gradients(depth):
    torch.manual_seed(91)
    dense = HeadMLP(2, 13, 6, 5, depth).double()
    with torch.no_grad():
        dense.weights[-1].normal_(std=.2)
    factorized = copy.deepcopy(dense)
    q = torch.randn(2, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    rows = torch.randn(2, 2, 6, 10, dtype=torch.float64, requires_grad=True)
    values = torch.cat((q[:, :, :, None].expand(-1, -1, -1, 6, -1),
                        rows[:, :, None].expand(-1, -1, 4, -1, -1)), -1)
    expected = dense(values)
    left = torch.autograd.grad(expected.square().sum(), (q, rows, *dense.parameters()))
    actual = projected.interaction(factorized, q, rows)
    right = torch.autograd.grad(actual.square().sum(), (q, rows, *factorized.parameters()))
    for a, b in zip((expected, *left), (actual, *right), strict=True):
        torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("n,q", ((1, 1), (6, 4), (1024, 16)))
def test_actual_affine_geometry_and_counter_sum(n, q):
    cfg = ProbeConfig(2, 2, 3, 4, hidden_width=5, depth=1, query_chunk=16, row_chunk=1024)
    result = projected.first_affine_flops(cfg, n, q, batch=2)
    lh, d, r, w = 4, 3, 10, 5
    expected_old = 2 * lh * q * n * (2 * (d + r) * w + w)
    # Explicit row GEMM+bias, query GEMM, then one broadcast addition.
    expected_new = 2 * lh * (2 * n * r * w + n * w + 2 * q * d * w + q * n * w)
    assert result == dict(original=expected_old, factorized=expected_new, delta=expected_new - expected_old)
    counts = dict(training_probe_rows=2 * n, training_native_queries=2 * q,
                  training_candidate_row_pairs=2 * q * n)
    assert projected.counted_first_affine(cfg, counts, "training") == result
    probe = FullItemProbeLadder(cfg, "fgh")
    original = probe.forward_flops(n, q, batch=2)
    with projected.fitting_ledger(probe, enabled=True):
        actual = probe.forward_flops(n, q, batch=2)
    assert actual["f_prepare_flops"] == original["f_prepare_flops"]
    assert actual["g_read_extra_flops"] == original["g_read_extra_flops"]
    for key in ("h_read_extra_flops", "read_extra_flops", "total_correction_forward_flops"):
        assert actual[key] == original[key] + result["delta"]
    assert "forward_flops" not in probe.__dict__


def test_actual_optimizer_and_explicit_cost_delta(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, sample = fixture()
        train = prepare_training_samples(model, [sample])
        probe = FullItemProbeLadder.from_model(model, stage="fgh", hidden_width=7, depth=1)
        with torch.no_grad():
            for modules in (probe.g, probe.h):
                for module in modules:
                    module.weights[-1].normal_(std=.01)
        reference = copy.deepcopy(probe)
        kwargs = dict(steps=3, patience=0, validation_every=1, learning_rate=3e-4, seed=17)
        left = fit_probe_resident(model, reference, train, train, **kwargs)
        right = fit_probe_factorized(model, probe, train, train, **kwargs)
        assert left["operation_counts"] == right["operation_counts"]
        assert left["best_step"] == right["best_step"]
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, probe.state_dict()[name], atol=1e-6, rtol=1e-4)
        opt = right["execution_optimizations"]
        delta = 3 * opt["training_first_affine"]["delta"] + opt["validation_first_affine"]["delta"]
        assert right["total_calibration_flops_estimate"] == left["total_calibration_flops_estimate"] + delta + opt["input_statistics_moment_flops"]
        assert right["total_calibration_flops_estimate"] == sum(right["arithmetic"].values())
        assert "interaction_correction" not in probe.__dict__
        assert "forward_flops" not in probe.__dict__
    finally:
        torch.set_num_threads(previous)
