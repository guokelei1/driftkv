"""Fixed-snapshot query alignment and candidate-variation diagnostics; CPU only."""
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.candidate_residual import candidate_residual
from hstu_kvcache.adaptation.reader import history_read, score
from tests.experiments.ch3_motivation.test_probes import fixture


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("variant", ["legacy", "hstu_reference"])
def test_real_operator_uses_student_query_for_both_cache_reads(variant):
    model, sample = fixture(variant)
    items = torch.arange(10, 26)[None]
    delta = torch.ones(1, 16)
    mapped, teacher = sample["parent_cache"], sample["teacher_cache"]
    result = candidate_residual(model, mapped, teacher, items, delta)
    values, student = score(model, mapped, items, delta, trace=True)
    _, independent_teacher = score(model, teacher, items, delta, trace=True)
    assert not torch.allclose(student.queries[1], independent_teacher.queries[1])
    for layer, q in enumerate(student.queries):
        actual = history_read(model.blocks[layer].attn, q, mapped.k[layer], mapped.v[layer])
        wanted = history_read(model.blocks[layer].attn, q, teacher.k[layer], teacher.v[layer])
        error = (wanted - actual).double()[0]
        torch.testing.assert_close(torch.tensor(result["raw_mse_by_layer_head"][layer], dtype=torch.float64),
                                   error.square().mean((1, 2)))
        torch.testing.assert_close(torch.tensor(result["candidate_raw_mse_by_layer_head"][layer], dtype=torch.float64),
                                   error.square().mean(2))
    torch.testing.assert_close(torch.tensor(result["mapped_scores"]), values[0])
    assert result["summary"]["length_normalized_read_mse"] == pytest.approx(
        result["summary"]["raw_read_mse"] / mapped.seq_len ** 2)
    assert 0. <= result["summary"]["centered_energy_ratio"] <= 1.
    json.dumps(result, allow_nan=False)


def test_identical_caches_have_zero_error_and_undefined_energy_ratio():
    model, sample = fixture()
    cache = sample["parent_cache"]
    before_k, before_v = cache.k.clone(), cache.v.clone()
    result = candidate_residual(model, cache, cache, torch.arange(10, 26)[None], torch.ones(1, 16))
    assert result["summary"] == dict(raw_read_mse=0., length_normalized_read_mse=0.,
                                     centered_energy_ratio=None)
    assert all(ratio is None for layer in result["centered_energy_ratio_by_layer_head"] for ratio in layer)
    assert torch.equal(cache.k, before_k) and torch.equal(cache.v, before_v)
    json.dumps(result, allow_nan=False)


def test_repeated_candidate_and_context_have_no_candidate_variation():
    model, sample = fixture()
    result = candidate_residual(model, sample["parent_cache"], sample["teacher_cache"],
                                torch.full((1, 16), 12), torch.ones(1, 16))
    assert result["summary"]["raw_read_mse"] > 0.
    assert result["summary"]["centered_energy_ratio"] == 0.


def test_time_changes_can_vary_residual_without_changing_item():
    model, sample = fixture()
    result = candidate_residual(model, sample["parent_cache"], sample["teacher_cache"],
                                torch.full((1, 16), 12), torch.arange(16)[None].float() * 3600.)
    assert result["summary"]["centered_energy_ratio"] > 0.
    assert "item-only" in result["scope"]
