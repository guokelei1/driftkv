"""Preparation must preserve the frozen function and composed release state."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]

from hstu_kvcache.methods.cachetide.structure import StructuredKVAdapter
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.methods.cachetide.prepare import (
    compile_item_tables, fixed_features_from_bank, native_feature_chunks, project_fixed_item_terms,
)
from hstu_kvcache.methods.cachetide.adapter import UnifiedRelayAdapter
from hstu_kvcache.methods.cachetide.features import build_bank, load_bank
from hstu_kvcache.methods.cachetide.publication import compile_full_catalogue


class Model:
    def __init__(self, embeddings):
        self.item_emb = SimpleNamespace(weight=embeddings)
        self.cfg = SimpleNamespace(num_items=len(embeddings)-1)
        self.lookups = 0

    def lookup_item_embeddings(self, ids):
        self.lookups += 1
        return self.item_emb.weight[ids]


def fixture(tmp_path, dtype):
    torch.manual_seed(20261015)
    source = StructuredKVAdapter(2, 6, 4, layout="shared12_private6").to(dtype).eval()
    with torch.no_grad():
        for layer in source.layers:
            layer.input_weight.normal_(std=.1)
            layer.input_bias.normal_(mean=.02, std=.08)
    embeddings = torch.randn(19, 24, dtype=dtype)
    stored = build_bank(source, embeddings, tmp_path/"bank.pt",
                        source_bindings={"test": "frozen origin"}, chunk_size=5)
    adapter = UnifiedRelayAdapter.from_structured(source, mode="A", read_shared=3,
        read_private=3, gate_normalization="rms", feature_mode="origin_encoder",
        feature_bank_path=stored["path"], feature_bank_sha256=stored["sha256"])
    bank = load_bank(stored["path"], stored["sha256"], device="cpu", dtype=dtype)
    cache = HSTUKVCache(torch.randn(2, 2, 9, 24, dtype=dtype),
                       torch.randn(2, 2, 9, 24, dtype=dtype), 9)
    items = torch.tensor([[0, 1, 1, 5, 18, 8, 3, 7, 9], [4, 4, 7, 8, 18, 0, 3, 5, 2]])
    return source, adapter, bank, embeddings, cache, items


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_fixed_projection_and_transient_features_preserve_origin(tmp_path, dtype):
    source, adapter, bank, embeddings, cache, items = fixture(tmp_path, dtype)
    prepared, work = project_fixed_item_terms(source, embeddings, chunk_size=4)
    tol = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(prepared.table, bank.table, atol=tol, rtol=tol)
    torch.testing.assert_close(prepared.bias, bank.bias, atol=0, rtol=0)
    pieces = list(native_feature_chunks(adapter, cache, items, prepared, chunk_size=4))
    assert [(a,b) for a,b,_ in pieces] == [(0,4),(4,8),(8,9)]
    actual = torch.cat([value for _,_,value in pieces], 2)
    expected = torch.stack([layer.fixed_features(cache.k[i], cache.v[i], items)[0]
                            for i, layer in enumerate(adapter.layers)])
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    assert work["item_projection_flops"] == bank.metadata["prepare_flops"]
    assert work["reference_item_projection_calls"] == 2*work["item_projection_calls"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_three_release_compiler_matches_reference_work_and_values(tmp_path, dtype):
    _, adapter, _, embeddings, _, _ = fixture(tmp_path, dtype)
    previous, reference_previous = None, None
    tol = 2e-6 if dtype == torch.float32 else 2e-12
    for version in (3,4,5):
        with torch.no_grad():
            for layer in adapter.layers:
                for rotation in layer.rotation:
                    rotation.normal_(std=.08)
                layer.item_update_weight.normal_(std=.07)
                layer.item_update_bias.normal_(std=.03)
        model = Model(embeddings+version*.01)
        got, work = compile_item_tables(adapter, model, previous, chunk_size=5)
        assert model.lookups == 4
        model.lookups = 0
        wanted, reference_work = compile_full_catalogue(adapter, model,
                                                       reference_previous, chunk_size=5)
        assert model.lookups == 8
        assert work["arithmetic"] == reference_work
        assert work["total_flops"] == sum(reference_work.values())
        for actual, expected in zip((got.P,got.Q,got.D),(wanted.P,wanted.Q,wanted.D),strict=True):
            torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        previous, reference_previous = got, wanted


def test_nonlinearity_cannot_be_moved_after_release_transform(tmp_path):
    _, adapter, bank, _, cache, items = fixture(tmp_path, torch.float64)
    layer = adapter.layers[0]
    eta = fixed_features_from_bank(layer, cache.k[0], cache.v[0], items, bank)
    # A concrete nonlinear witness: a release Q cannot be folded into phi or
    # native weights across SiLU, even though the preceding fixed affine can.
    assert (2*eta-torch.nn.functional.silu(2*torch.cat([
        layer.kv_features(cache.k[0],cache.v[0],branch)@weight
        for branch,weight in zip(layer.branches,bank.native_weights[0],strict=True)
    ],-1)+2*bank.table[0][items]+2*bank.bias[0])).abs().max() > .01


@pytest.mark.parametrize("problem, message", [
    ("mode", "feature mode"),
    ("bank_sha", "immutable feature bank"),
    ("missing_bank_sha", "SHA256 metadata"),
    ("xi_width", "summary width"),
    ("missing_layers", "layer feature metadata"),
    ("missing_layer", "fixed-feature metadata"),
])
def test_incompatible_or_missing_prior_features_rejected_before_lookup(tmp_path, problem, message):
    _, adapter, _, embeddings, _, _ = fixture(tmp_path, torch.float64)
    previous, _ = compile_item_tables(adapter, Model(embeddings))
    actor = copy.deepcopy(adapter)
    if problem == "mode":
        previous.layers[0].feature_mode = "native"
    elif problem == "bank_sha":
        previous.layers[0].feature_bank_sha256 = "different origin encoder"
    elif problem == "missing_bank_sha":
        previous.layers[0].feature_bank_sha256 = None
    elif problem == "xi_width":
        previous.layers[0].xi_width = 3
    elif problem == "missing_layers":
        previous.layers = None
    elif problem == "missing_layer":
        previous.layers = (None, previous.layers[1])
    model = Model(embeddings)
    with pytest.raises(ValueError, match=message):
        compile_item_tables(actor, model, previous)
    assert model.lookups == 0


def test_same_geometry_different_native_basis_rejected(tmp_path):
    source, _, _, embeddings, _, _ = fixture(tmp_path, torch.float64)
    adapter = UnifiedRelayAdapter.from_structured(source, mode="A", feature_mode="native")
    previous, _ = compile_item_tables(adapter, Model(embeddings))
    actor = copy.deepcopy(adapter)
    with torch.no_grad():
        actor.layers[0].native_basis_0.add_(.1)
    model = Model(embeddings)
    with pytest.raises(ValueError, match="fixed native basis"):
        compile_item_tables(actor, model, previous)
    assert model.lookups == 0
