"""CPU arithmetic/gradient oracles for the optional resident input adapter."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.probes import fit_probe, prepare_training_samples
from experiments.ch3_motivation.code.probe_variants import build_variant
from experiments.ch3_motivation.code.resident_probe_fitting import (
    PLACED_FIELDS, _place_samples, fit_probe_resident,
)
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def world(prepared=True):
    torch.manual_seed(129)
    config = HSTUConfig(num_items=50, num_behaviors=2, hidden_size=8, num_heads=2,
                       num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent = HSTU(config).eval().requires_grad_(False)
    model = copy.deepcopy(parent)
    model.item_emb.weight.add_(torch.randn_like(model.item_emb.weight)*.02)
    model.blocks[0].attn.k_proj.weight.add_(torch.randn_like(model.blocks[0].attn.k_proj.weight)*.03)
    samples = []
    for user in range(5):
        ids = torch.arange(1+user, 5+2*user)[None]
        behaviors, deltas = torch.ones_like(ids), torch.ones_like(ids, dtype=torch.float32)
        old, teacher = parent.compute_kv(ids, behaviors, deltas), model.compute_kv(ids, behaviors, deltas)
        samples.append(dict(uid=user, parent=old, parent_cache=old, teacher=teacher, teacher_cache=teacher,
            raw=(ids, behaviors, deltas), item_ids=ids,
            candidates=torch.tensor([[20+user, 30+user, 40+user]]),
            query_delta=torch.tensor([[1., 2.+user, 3.]]), unused_large=torch.ones(1000)))
    train, validation = samples[:3], samples[3:]
    if prepared:
        train, validation = prepare_training_samples(model, train), prepare_training_samples(model, validation)
    return model, train, validation


def snapshot(samples):
    result = []
    for sample in samples:
        tensors = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = (value, value.clone())
            elif isinstance(value, HSTUKVCache):
                tensors[key+".k"] = (value.k, value.k.clone())
                tensors[key+".v"] = (value.v, value.v.clone())
        result.append((dict(sample), tensors))
    return result


def assert_unchanged(samples, snapshots):
    for sample, (before, tensors) in zip(samples, snapshots, strict=True):
        assert set(sample) == set(before)
        assert all(sample[key] is value for key, value in before.items())
        assert sample["parent"] is sample["parent_cache"]
        assert sample["teacher"] is sample["teacher_cache"]
        for original, expected in tensors.values():
            assert torch.equal(original, expected)


def gradient_trace(probe):
    trace = []
    handles = [parameter.register_hook(lambda gradient, name=name:
        trace.append((name, gradient.detach().clone()))) for name, parameter in probe.named_parameters()]
    return trace, handles


@pytest.mark.parametrize("stage", ["f", "fg", "fgh"])
@pytest.mark.parametrize("prepared", [False, True])
def test_real_fitter_identical_losses_gradients_weights_and_counters(stage, prepared):
    model, train, validation = world(prepared)
    original = snapshot(train+validation)
    reference = build_variant("full_item", model, stage=stage, hidden_width=8, depth=1,
                              query_chunk=2, row_chunk=3)
    resident = copy.deepcopy(reference)
    reference_gradients, reference_handles = gradient_trace(reference)
    resident_gradients, resident_handles = gradient_trace(resident)
    kwargs = dict(steps=5, validation_every=1, patience=0, seed=17, learning_rate=1e-3)
    reference_rng = torch.random.get_rng_state().clone()
    expected = fit_probe(model, reference, iter(train), iter(validation), **kwargs)
    assert torch.equal(torch.random.get_rng_state(), reference_rng)
    actual = fit_probe_resident(model, resident, iter(train), iter(validation), **kwargs)
    assert torch.equal(torch.random.get_rng_state(), reference_rng)
    adapter = actual.pop("execution_adapter")
    assert actual == expected  # Every step's losses/gradient norms and the complete FLOPs ledger.
    assert len(reference_gradients) == len(resident_gradients) > 0
    for (name, expected_grad), (actual_name, actual_grad) in zip(reference_gradients, resident_gradients, strict=True):
        assert actual_name == name
        assert torch.equal(expected_grad, actual_grad)
    for name, value in reference.state_dict().items():
        assert torch.equal(value, resident.state_dict()[name])
    for left, right in zip(reference.parameters(), resident.parameters(), strict=True):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.equal(left.grad, right.grad)
    assert_unchanged(train+validation, original)
    assert adapter["binding"]["format"] == "resident_probe_fitting_v1"
    assert adapter["binding"]["placed_fields"] == list(PLACED_FIELDS)
    assert len(adapter["binding"]["helper_source"]["sha256"]) == 64
    metrics = adapter["metrics"]
    assert metrics["cuda_memory"] is None and metrics["model_device"] == "cpu"
    assert metrics["storage"]["new_resident_storage_bytes"] == 0
    assert metrics["storage"]["cross_device_logical_bytes"] == 0
    assert metrics["preplacement_seconds"] >= 0 and metrics["core_fit_seconds"] > 0
    for handle in reference_handles+resident_handles:
        handle.remove()


def test_used_storage_aliases_views_and_ignored_large_fields_are_counted_once(monkeypatch):
    model, train, validation = world()
    # Existing prepared scales share storage across users. Add an optional item
    # embedding and deliberately share one tensor between two used fields to
    # exercise the placement memo without requiring a GPU.
    shared = torch.ones(1, 3)
    train[0] = dict(train[0], candidates=shared, query_delta=shared.detach(),
                    item_embeddings=model.lookup_item_embeddings(train[0]["item_ids"]).detach())
    train[0]["unused_large"] = torch.ones(100000)
    before = snapshot(train+validation)
    placement_calls = []
    original_to = torch.Tensor.to

    def counted_to(tensor, *args, **kwargs):
        placement_calls.append(tensor)
        return original_to(tensor, *args, **kwargs)

    with monkeypatch.context() as placement_instrumentation:
        placement_instrumentation.setattr(torch.Tensor, "to", counted_to)
        placed_train, placed_validation, counts = _place_samples(train, validation, torch.device("cpu"))
    used = []
    for sample in train+validation:
        for field in PLACED_FIELDS:
            value = sample.get(field)
            used += [value] if isinstance(value, torch.Tensor) else [value.k, value.v] if isinstance(value, HSTUKVCache) else []
    storage = {(str(value.device), value.untyped_storage().data_ptr(), value.untyped_storage().nbytes()):
               value.untyped_storage().nbytes() for value in used}
    assert counts["used_tensor_references"] == len(used)
    assert counts["unique_input_tensor_objects"] == len({id(value) for value in used})
    assert counts["unique_preplacement_views"] == len(placement_calls)
    assert counts["unique_preplacement_views"] < counts["unique_input_tensor_objects"]
    assert counts["input_storage_bytes"] == sum(storage.values()) == counts["resident_storage_bytes"]
    assert counts["unique_resident_storages"] == len(storage)
    assert counts["new_resident_storages"] == counts["cross_device_tensor_copies"] == 0
    for original, placed in zip(train+validation, placed_train+placed_validation, strict=True):
        assert original is not placed
        assert placed["parent"] is original["parent"] and placed["teacher"] is original["teacher"]
        assert placed["raw"] is original["raw"] and placed["unused_large"] is original["unused_large"]
        assert placed["parent_cache"] is not original["parent_cache"]
        assert placed["parent_cache"].k is original["parent_cache"].k
        assert placed["parent_cache"].v is original["parent_cache"].v
        assert placed["read_moment_sum"] is original["read_moment_sum"]
    assert placed_train[0]["candidates"] is placed_train[0]["query_delta"]
    assert_unchanged(train+validation, before)


def test_two_stages_recompute_from_changed_samples_without_mutation_or_global_cache():
    model, train, validation = world(prepared=False)
    initial = build_variant("full_item", model, stage="f", hidden_width=8, depth=1)
    reference, resident = copy.deepcopy(initial), copy.deepcopy(initial)
    kwargs = dict(steps=2, validation_every=1, patience=0, seed=23)
    expected_first = fit_probe(model, reference, train, validation, **kwargs)
    actual_first = fit_probe_resident(model, resident, train, validation, **kwargs)
    actual_first.pop("execution_adapter")
    assert actual_first == expected_first
    # A later recipe supplies new candidate/time matrices from the same raw
    # caches. Neither resident inputs nor teacher moments may leak between calls.
    changed = [dict(sample, candidates=sample["candidates"].roll(1, 1),
                    query_delta=sample["query_delta"]+7.) for sample in train+validation]
    before = snapshot(changed)
    reference, resident = reference.expanded("fg"), resident.expanded("fg")
    resident.load_state_dict(reference.state_dict())
    expected_second = fit_probe(model, reference, changed[:3], changed[3:], **kwargs)
    actual_second = fit_probe_resident(model, resident, changed[:3], changed[3:], **kwargs)
    actual_second.pop("execution_adapter")
    assert actual_second == expected_second
    assert actual_second["fit_pool_loss_scales"] != actual_first["fit_pool_loss_scales"]
    assert all(torch.equal(value, resident.state_dict()[name]) for name, value in reference.state_dict().items())
    assert_unchanged(changed, before)
    assert all("teacher_scores" not in sample and "read_features" not in sample for sample in train+validation)
